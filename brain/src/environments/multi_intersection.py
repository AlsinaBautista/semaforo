# Semáforo Inteligente - Brain Module
"""PettingZoo parallel environment for the Necochea 3-intersection corridor.

This wraps ``sumo_rl``'s multi-agent ``SumoEnvironment`` so each traffic signal
(``c0``, ``c1``, ``c2``) is an independent agent sharing one SUMO simulation,
and exposes it through the PettingZoo *parallel* API that Ray RLlib consumes via
``ParallelPettingZooEnv``.

Two properties make it suitable for the MAPPO training described in
``scripts/train_mappo.py``:

* **Fixed, homogeneous spaces.** Every agent sees a fixed-length, ``[0, 1]``
  normalised observation and a ``Discrete`` action of identical size, so a
  *single shared policy* (parameter sharing) can drive all three signals.

* **Per-episode scenario injection.** ``reset(options={"scenario": spec})``
  rewrites the SUMO route file and seed before restarting the simulation, which
  is how curriculum learning and domain randomization feed drastic, ever-changing
  worlds (winter calm, summer exodus, sudestada storms) into the agent.

sumo_rl only reports an agent in its step dict on the simulation seconds when
that signal is *eligible to act*; signals can fall out of phase with each other.
To honour the parallel contract (and keep RLlib happy) this wrapper caches the
last observation/reward for any agent not acting on a given macro-step so every
agent is always present.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.environments import sumo_home as _sumo_home  # noqa: F401  sets SUMO_HOME
from src.environments.single_intersection import SingleIntersectionEnv
from src.rewards.spillback import diff_waiting_time_with_spillback
from src.training.scenarios import (
    ScenarioSpec,
    sample_mixed_task_scenario,
    sample_scenario,
    write_scenario_routes,
)

import sumo_rl  # noqa: E402  (import after SUMO_HOME bootstrap)
from sumo_rl.environment.observations import DefaultObservationFunction

# Single source of truth for the 4-bucket directional aggregation so the
# corridor and single-intersection envs' canonical layouts cannot drift.
_aggregate_directions = SingleIntersectionEnv._aggregate_directions


class Canonical17ObservationFunction(DefaultObservationFunction):
    """Canonical 17-dim observation shared with ``SingleIntersectionEnv``.

    Layout (everything clipped to ``[0, 1]``)::

        [queue_N, queue_E, queue_S, queue_W,            # 0-3
         density_N, density_E, density_S, density_W,    # 4-7
         phase_0, phase_1, phase_2, phase_3,            # 8-11  (one-hot)
         phase_elapsed_normalised,                      # 12
         down_N, down_E, down_S, down_W]                # 13-16 (downstream occ)

    Direction bucket order is **[N, E, S, W]**: ``_aggregate_directions``
    chunks the per-lane arrays in ``ts.lanes`` order, which follows each
    junction's incLanes order — uniformly [north_in, east, south_in, west]
    for c0/c1/c2 in corridor.net.xml. Any future inference-side constructor
    that fills real per-direction data MUST use this exact order or the
    policy receives a permutation.

    The downstream slots keep spillback visible to the policy (the receiving
    block's occupancy is part of the state, so a spillback-aware reward can
    shape behaviour while the state stays Markovian), and the fixed canonical
    layout matches what the Edge feeds the exported ONNX actor.
    """

    def __call__(self) -> np.ndarray:
        ts = self.ts
        obs = np.zeros(17, dtype=np.float32)
        obs[0:4] = _aggregate_directions(
            np.asarray(ts.get_lanes_queue(), dtype=np.float32), max_val=20.0
        )
        obs[4:8] = _aggregate_directions(
            np.asarray(ts.get_lanes_density(), dtype=np.float32), max_val=1.0
        )
        obs[8 + int(ts.green_phase) % 4] = 1.0
        max_phase_dur = float(ts.max_green + ts.yellow_time)
        obs[12] = np.clip(
            float(ts.time_since_last_phase_change) / max(max_phase_dur, 1.0), 0.0, 1.0
        )
        obs[13:17] = _aggregate_directions(
            np.asarray(ts.get_out_lanes_density(), dtype=np.float32), max_val=1.0
        )
        return np.clip(obs, 0.0, 1.0)

    def observation_space(self) -> spaces.Box:
        # Constant: sumo_rl instantiates the observation function before the
        # signal's lanes/phases exist, so this must not touch ``self.ts``.
        return spaces.Box(
            low=np.zeros(17, dtype=np.float32),
            high=np.ones(17, dtype=np.float32),
        )


# sumo_rl reward_fn accepts callables; map our extended names onto them.
_CUSTOM_REWARD_FNS = {
    "diff-waiting-time-spillback": diff_waiting_time_with_spillback,
}

# ---------------------------------------------------------------------------
# Fixed space dimensions (homogeneous across all corridor signals)
# ---------------------------------------------------------------------------
# The corridor signals are homogeneous (each alternates an NS-green and an
# EW-green phase), so a fixed Discrete action and the canonical 17-dim
# observation (Canonical17ObservationFunction above) let one shared policy
# control every agent. Actions are additionally clamped against each signal's
# true green-phase count at runtime, so a network with a different phase
# layout can never trigger an invalid transition.
_FIXED_OBS_SIZE = 17
_NUM_GREEN_PHASES = 2

_DEFAULT_NET = (
    Path(__file__).resolve().parents[2]
    / "sumo_networks" / "corridor" / "corridor.net.xml"
)
_DEFAULT_ROUTE = (
    Path(__file__).resolve().parents[2]
    / "sumo_networks" / "corridor" / "corridor.rou.xml"
)


class MultiIntersectionEnv(ParallelEnv):
    """PettingZoo parallel environment for a corridor of traffic signals.

    Args:
        net_file: Path to the SUMO ``.net.xml`` corridor network.
        route_file: Default demand file (used when no scenario is injected).
        num_seconds: Simulated seconds per episode.
        delta_time: Simulation seconds between agent decisions.
        min_green: Minimum green duration enforced by SUMO.
        max_green: Maximum green duration.
        yellow_time: Yellow interval inserted on phase change.
        reward_fn: sumo_rl reward key (e.g. ``"diff-waiting-time"``,
            ``"queue"``, ``"pressure"``).
        scenario_dir: Where per-episode generated route files are written.
        use_gui: Launch ``sumo-gui`` instead of headless ``sumo``.
        auto_randomize: If ``True``, every ``reset()`` without an explicit
            scenario samples and injects a fresh randomized scenario drawn from
            the current curriculum stage (domain randomization for training).
        curriculum_stage: Initial curriculum stage used by ``auto_randomize``.
        rng_seed: Seed for this env's scenario-sampling RNG (per worker).
        rehearsal_ratio: When ``> 0`` and ``auto_randomize`` is on, scenario
            sampling uses the asymmetric mixed-task mixture (rehearsal of past
            curriculum stages against catastrophic forgetting) instead of the
            plain stage pool.
        include_global_state: If ``True``, per-agent observations become a
            ``Dict`` of the local ``"obs"`` plus the global corridor ``"state"``
            (all agents' observations concatenated) for a centralized critic
            (CTDE).  Execution stays decentralized: the actor reads only
            ``"obs"``.
    """

    metadata = {"render_modes": ["human"], "name": "necochea_corridor_v1"}

    def __init__(
        self,
        net_file: str | Path = _DEFAULT_NET,
        route_file: str | Path = _DEFAULT_ROUTE,
        *,
        num_seconds: int = 3600,
        delta_time: int = 5,
        min_green: int = 5,
        max_green: int = 60,
        yellow_time: int = 3,
        reward_fn: str = "diff-waiting-time",
        scenario_dir: str | Path = "/tmp/semaforo_scenarios",
        use_gui: bool = False,
        auto_randomize: bool = False,
        curriculum_stage: int = 0,
        rng_seed: int | None = None,
        rehearsal_ratio: float = 0.0,
        include_global_state: bool = False,
    ) -> None:
        super().__init__()
        self._net_file = str(Path(net_file).resolve())
        self._default_route = str(Path(route_file).resolve())
        self._num_seconds = num_seconds
        self._delta_time = delta_time
        self._min_green = min_green
        self._max_green = max_green
        self._yellow_time = yellow_time
        self._reward_fn = reward_fn
        self._scenario_dir = Path(scenario_dir)
        self._use_gui = use_gui
        self._auto_randomize = auto_randomize
        self.curriculum_stage = int(curriculum_stage)
        self._rng = random.Random(rng_seed)
        self._rehearsal_ratio = float(rehearsal_ratio)
        self._include_global_state = include_global_state

        self._sumo: sumo_rl.SumoEnvironment | None = None
        self._max_green_total = max_green + yellow_time

        # Agent roster is known up-front from the corridor topology.
        self.possible_agents = ["c0", "c1", "c2"]
        self.agents = list(self.possible_agents)

        local_obs_space = spaces.Box(
            low=0.0, high=1.0, shape=(_FIXED_OBS_SIZE,), dtype=np.float32
        )
        if self._include_global_state:
            state_size = _FIXED_OBS_SIZE * len(self.possible_agents)
            self._obs_space: spaces.Space = spaces.Dict(
                {
                    "obs": local_obs_space,
                    "state": spaces.Box(
                        low=0.0, high=1.0, shape=(state_size,), dtype=np.float32
                    ),
                }
            )
        else:
            self._obs_space = local_obs_space
        self._act_space = spaces.Discrete(_NUM_GREEN_PHASES)

        # Per-agent observation cache so every agent is reported every step.
        self._obs_cache: dict[str, np.ndarray] = {
            a: np.zeros(_FIXED_OBS_SIZE, dtype=np.float32) for a in self.possible_agents
        }

    # ------------------------------------------------------------------ #
    # PettingZoo space accessors
    # ------------------------------------------------------------------ #
    def observation_space(self, agent: str) -> spaces.Space:  # noqa: D102
        return self._obs_space

    def action_space(self, agent: str) -> spaces.Discrete:  # noqa: D102
        return self._act_space

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        """Reset the corridor, optionally injecting a randomized scenario.

        Args:
            seed: RNG seed for SUMO departures.
            options: May contain ``"scenario"`` (a :class:`ScenarioSpec`) or
                ``"route_file"`` (an explicit path) to drive demand for this
                episode.  When neither is present the default route is used.

        Returns:
            ``(observations, infos)`` dicts keyed by agent id.
        """
        route_file = self._default_route
        scenario: ScenarioSpec | None = (options or {}).get("scenario")
        # Domain randomization: when no scenario is supplied and auto-randomize
        # is on (training), draw a fresh one from the current curriculum stage.
        if scenario is None and self._auto_randomize and not (options or {}).get("route_file"):
            if self._rehearsal_ratio > 0.0:
                scenario = sample_mixed_task_scenario(
                    self._rng,
                    stage=self.curriculum_stage,
                    rehearsal_ratio=self._rehearsal_ratio,
                )
            else:
                scenario = sample_scenario(self._rng, stage=self.curriculum_stage)
        if scenario is not None:
            route_file = str(write_scenario_routes(scenario, self._scenario_dir))
            if seed is None:
                seed = scenario.seed
        elif options and options.get("route_file"):
            route_file = str(options["route_file"])

        # Build a *fresh* SumoEnvironment every episode. Reusing one instance and
        # relaunching SUMO in-process reuses the same TraCI label and is flaky
        # across platforms ("Connection closed by SUMO"); a clean instance gets a
        # new connection label and a cleanly-torn-down subprocess, which makes
        # per-episode scenario/route switching reliable.
        if self._sumo is not None:
            try:
                self._sumo.close()
            except Exception:  # noqa: BLE001 – defensive cleanup
                pass
        self._sumo = sumo_rl.SumoEnvironment(
            net_file=self._net_file,
            route_file=route_file,
            use_gui=self._use_gui,
            single_agent=False,
            num_seconds=self._num_seconds,
            delta_time=self._delta_time,
            min_green=self._min_green,
            max_green=self._max_green,
            yellow_time=self._yellow_time,
            reward_fn=_CUSTOM_REWARD_FNS.get(self._reward_fn, self._reward_fn),
            observation_class=Canonical17ObservationFunction,
            sumo_seed=seed if seed is not None else "random",
        )

        raw_obs = self._sumo.reset(seed=seed)
        # sumo_rl multi-agent reset returns just the obs dict.
        if isinstance(raw_obs, tuple):
            raw_obs = raw_obs[0]

        self.agents = list(self.possible_agents)
        for a in self.possible_agents:
            self._obs_cache[a] = np.zeros(_FIXED_OBS_SIZE, dtype=np.float32)
        self._absorb(raw_obs)

        observations = self._build_observations(self.agents)
        infos = {a: {} for a in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        """Advance the corridor by one decision interval.

        Args:
            actions: Mapping ``agent -> green-phase index``.  Agents not eligible
                to act this interval are ignored by SUMO.

        Returns:
            Standard PettingZoo parallel 5-tuple of per-agent dicts.
        """
        assert self._sumo is not None, "reset() must be called before step()"

        safe_actions = self._clamp_actions(actions)
        raw_obs, raw_rewards, dones, info = self._sumo.step(safe_actions)
        self._absorb(raw_obs)

        done_all = bool(dones.get("__all__", False))
        observations = self._build_observations(self.possible_agents)
        rewards = {a: float(raw_rewards.get(a, 0.0)) for a in self.possible_agents}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: done_all for a in self.possible_agents}
        infos = {a: dict(info) if isinstance(info, dict) else {} for a in self.possible_agents}

        if done_all:
            self.agents = []
        return observations, rewards, terminations, truncations, infos

    def set_curriculum_stage(self, stage: int) -> None:
        """Set the curriculum stage future auto-randomized resets draw from."""
        self.curriculum_stage = max(0, int(stage))

    def render(self) -> None:  # noqa: D102
        # Rendering is delegated to sumo-gui via use_gui at construction time.
        return None

    def close(self) -> None:
        """Shut down the SUMO simulation and release the TraCI connection."""
        if self._sumo is not None:
            try:
                self._sumo.close()
            except Exception:  # noqa: BLE001 – defensive cleanup
                pass

    def _clamp_actions(self, actions: dict[str, int]) -> dict[str, int]:
        """Map each action into its signal's valid green-phase range.

        The fixed Discrete action space may be larger than a given signal's true
        green-phase count; clamping with modulo guarantees sumo_rl can always
        find a defined yellow transition and never raises on an invalid phase.
        """
        assert self._sumo is not None
        safe: dict[str, int] = {}
        for agent, act in actions.items():
            ts = self._sumo.traffic_signals.get(agent)
            n = getattr(ts, "num_green_phases", _NUM_GREEN_PHASES) or _NUM_GREEN_PHASES
            safe[agent] = int(act) % int(n)
        return safe

    # ------------------------------------------------------------------ #
    # Observation handling
    # ------------------------------------------------------------------ #
    def _global_state(self) -> np.ndarray:
        """Concatenate every agent's cached observation into the global S_t.

        Fixed ``possible_agents`` order (c0, c1, c2); values are already
        float32 and clipped to ``[0, 1]`` by :meth:`_absorb`.
        """
        return np.concatenate([self._obs_cache[a] for a in self.possible_agents])

    def _build_observations(
        self, agents: list[str]
    ) -> dict[str, np.ndarray | dict[str, np.ndarray]]:
        """Build per-agent observations, attaching the global state if enabled."""
        if not self._include_global_state:
            return {a: self._obs_cache[a].copy() for a in agents}
        state = self._global_state()
        return {
            a: {"obs": self._obs_cache[a].copy(), "state": state.copy()}
            for a in agents
        }

    def _absorb(self, raw_obs: dict[str, np.ndarray]) -> None:
        """Fold raw sumo_rl observations into the fixed-size, clipped cache.

        Only agents present in ``raw_obs`` (those that acted) are updated; the
        rest retain their previous observation so the parallel contract holds.
        """
        if not isinstance(raw_obs, dict):
            return
        for agent, vec in raw_obs.items():
            if agent not in self._obs_cache:
                continue
            arr = np.asarray(vec, dtype=np.float32).flatten()
            buf = np.zeros(_FIXED_OBS_SIZE, dtype=np.float32)
            n = min(len(arr), _FIXED_OBS_SIZE)
            buf[:n] = arr[:n]
            self._obs_cache[agent] = np.clip(buf, 0.0, 1.0)
