"""Single-intersection Gymnasium environment for traffic signal control.

This module wraps ``sumo_rl.SumoEnvironment`` to provide a clean, fixed-size
observation space, swappable reward functions, and episode-level statistics
logging.  The wrapper is designed to be a drop-in for any single SUMO
intersection (just swap ``net_file`` / ``route_file``).

Typical usage::

    env = SingleIntersectionEnv(
        net_file="path/to/single_intersection.net.xml",
        route_file="path/to/single_intersection.rou.xml",
        reward_fn="pressure",
    )
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.close()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np

from src.environments import sumo_home as _sumo_home  # noqa: F401  sets SUMO_HOME

try:
    import sumo_rl  # noqa: F401 – triggers registration of SUMO envs
except ImportError as exc:
    raise ImportError(
        "sumo-rl is required.  Install with: pip install sumo-rl>=1.4.5"
    ) from exc

from src.rewards.composite import CompositeReward
from src.rewards.pressure import compute_pressure_reward
from src.rewards.queue_length import compute_queue_reward
from src.rewards.spillback import served_downstream_occupancies

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_NUM_DIRECTIONS = 4  # N, S, E, W
_NUM_PHASES = 4  # typical 4-phase signal plan
_OBS_SIZE = (
    _NUM_DIRECTIONS  # queue lengths
    + _NUM_DIRECTIONS  # densities
    + _NUM_PHASES  # one-hot current phase
    + 1  # normalised phase elapsed time
    + _NUM_DIRECTIONS  # downstream (outgoing lane) occupancies
)

_DEFAULT_NET = Path(__file__).resolve().parents[2] / "networks" / "single_intersection" / "single_intersection.net.xml"
_DEFAULT_ROUTE = Path(__file__).resolve().parents[2] / "networks" / "single_intersection" / "single_intersection.rou.xml"


class SingleIntersectionEnv(gym.Env):
    """Gymnasium wrapper for a single SUMO intersection controlled by RL.

    The environment exposes a **fixed-size** observation vector and a
    ``Discrete(4)`` action space regardless of the underlying SUMO network
    complexity, making it straightforward to plug into Stable-Baselines3 or
    any other Gymnasium-compatible algorithm.

    Attributes:
        observation_space: ``Box(low=0, high=1, shape=(_OBS_SIZE,))``.
        action_space: ``Discrete(4)`` – selects the next signal phase.
        metadata: Render mode metadata for Gymnasium compatibility.
    """

    metadata: dict[str, Any] = {"render_modes": ["human"]}

    # --------------------------------------------------------------------- #
    # Construction helpers
    # --------------------------------------------------------------------- #

    def __init__(
        self,
        net_file: str | Path,
        route_file: str | Path,
        *,
        use_gui: bool = False,
        delta_time: int = 5,
        min_green: int = 5,
        max_green: int = 120,
        yellow_time: int = 3,
        reward_fn: str = "queue",
        num_seconds: int = 3600,
        seed: int | None = None,
    ) -> None:
        """Initialise the single-intersection environment.

        Args:
            net_file: Path to a SUMO ``.net.xml`` network file.
            route_file: Path to a SUMO ``.rou.xml`` route / demand file.
            use_gui: If ``True``, launch ``sumo-gui`` instead of ``sumo``.
            delta_time: Simulation seconds between RL decisions.
            min_green: Minimum green-phase duration in seconds.
            max_green: Maximum green-phase duration in seconds.
            yellow_time: Duration of the yellow transition phase.
            reward_fn: Reward function name – ``"pressure"``, ``"queue"``,
                or ``"composite"``.
            num_seconds: Total simulation duration in seconds.
            seed: Random seed for reproducibility.
        """
        super().__init__()

        self._net_file = str(Path(net_file).resolve())
        self._route_file = str(Path(route_file).resolve())
        self._use_gui = use_gui
        self._delta_time = delta_time
        self._reward_fn_name = reward_fn
        self._num_seconds = num_seconds
        self._seed = seed

        # Build underlying sumo_rl environment
        self._sumo_env = sumo_rl.SumoEnvironment(
            net_file=self._net_file,
            route_file=self._route_file,
            use_gui=use_gui,
            single_agent=True,
            delta_time=delta_time,
            min_green=min_green,
            max_green=max_green,
            yellow_time=yellow_time,
            num_seconds=num_seconds,
            sumo_seed=seed if seed is not None else "random",
        )

        # Spaces ---------------------------------------------------------- #
        self.action_space = gym.spaces.Discrete(_NUM_PHASES)
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(_OBS_SIZE,), dtype=np.float32
        )

        # Reward function -------------------------------------------------- #
        self._composite_reward = CompositeReward()
        self._last_action: int | None = None
        self._phase_elapsed: float | None = None
        self._served_downstream: list[float] | None = None

        # Episode statistics ----------------------------------------------- #
        self._episode_count: int = 0
        self._step_count: int = 0
        self._cumulative_reward: float = 0.0
        self._episode_start_time: float = 0.0

    # --------------------------------------------------------------------- #
    # Factory
    # --------------------------------------------------------------------- #

    @classmethod
    def make_default(
        cls,
        *,
        use_gui: bool = False,
        reward_fn: str = "queue",
        seed: int | None = None,
    ) -> "SingleIntersectionEnv":
        """Create an environment using the bundled single-intersection network.

        Args:
            use_gui: Launch SUMO with GUI.
            reward_fn: Reward function name.
            seed: Random seed.

        Returns:
            A fully-configured ``SingleIntersectionEnv`` instance.

        Raises:
            FileNotFoundError: If the default network files are missing.
        """
        if not _DEFAULT_NET.exists():
            raise FileNotFoundError(
                f"Default network file not found: {_DEFAULT_NET}.  "
                "Generate it with the network builder first."
            )
        if not _DEFAULT_ROUTE.exists():
            raise FileNotFoundError(
                f"Default route file not found: {_DEFAULT_ROUTE}.  "
                "Generate it with the route builder first."
            )
        return cls(
            net_file=_DEFAULT_NET,
            route_file=_DEFAULT_ROUTE,
            use_gui=use_gui,
            reward_fn=reward_fn,
            seed=seed,
        )

    # --------------------------------------------------------------------- #
    # Gymnasium API
    # --------------------------------------------------------------------- #

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and return the initial observation.

        Args:
            seed: Optional seed override (passed to the base env).
            options: Additional reset options (currently unused).

        Returns:
            A tuple ``(observation, info)`` where *observation* is a
            normalised numpy vector and *info* contains episode metadata.
        """
        if seed is not None:
            super().reset(seed=seed)

        raw_obs, info = self._sumo_env.reset()

        self._last_action = None
        self._phase_elapsed = None
        self._served_downstream = None
        self._step_count = 0
        self._cumulative_reward = 0.0
        self._episode_start_time = time.monotonic()

        obs = self._transform_observation(raw_obs)
        return obs, info

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        """Execute one decision step (``delta_time`` simulation seconds).

        Args:
            action: Integer in ``[0, 3]`` selecting the traffic-light phase.

        Returns:
            Standard Gymnasium 5-tuple ``(obs, reward, terminated, truncated, info)``.
        """
        ts_id = list(self._sumo_env.traffic_signals.keys())[0]
        ts = self._sumo_env.traffic_signals[ts_id]
        current_phase = ts.green_phase
        num_phases = ts.num_green_phases

        # 1. Handle out-of-bounds actions (since action_space is hardcoded to 4)
        if action >= num_phases:
            action = current_phase  # Treat invalid actions as "keep phase"

        # 2. Prevent sumo-rl KeyError by forcing sequential phase transitions
        if action != current_phase and action != (current_phase + 1) % num_phases:
            action = (current_phase + 1) % num_phases

        raw_obs, raw_reward, terminated, truncated, info = self._sumo_env.step(action)

        obs = self._transform_observation(raw_obs)
        reward = self._compute_reward(obs, action, raw_reward)

        self._step_count += 1
        self._cumulative_reward += float(reward)

        # Enrich info with episode-level statistics
        info["step"] = self._step_count
        info["cumulative_reward"] = self._cumulative_reward

        if terminated or truncated:
            self._episode_count += 1
            elapsed = time.monotonic() - self._episode_start_time
            info["episode"] = {
                "r": self._cumulative_reward,
                "l": self._step_count,
                "t": elapsed,
            }
            logger.info(
                "Episode %d finished – steps=%d  reward=%.2f  wall=%.1fs",
                self._episode_count,
                self._step_count,
                self._cumulative_reward,
                elapsed,
            )

        self._last_action = action
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment (delegated to sumo-gui when enabled)."""
        if self._use_gui:
            self._sumo_env.render()

    def close(self) -> None:
        """Shut down the SUMO simulation and release resources."""
        try:
            self._sumo_env.close()
        except Exception:  # noqa: BLE001 – defensive cleanup
            logger.warning("Error closing SUMO environment", exc_info=True)

    # --------------------------------------------------------------------- #
    # Observation transform
    # --------------------------------------------------------------------- #

    def _transform_observation(self, raw_obs: Any) -> np.ndarray:
        """Convert a sumo_rl observation into our fixed-size normalised vector.

        The output vector layout is::

            [queue_N, queue_E, queue_S, queue_W,            # 0-3
             density_N, density_E, density_S, density_W,    # 4-7
             phase_0, phase_1, phase_2, phase_3,            # 8-11  (one-hot)
             phase_elapsed_normalised,                      # 12
             down_N, down_E, down_S, down_W]                # 13-16 (downstream occ)

        Bucket order is [N, E, S, W] — the per-lane arrays follow the
        junction's incLanes order (see Canonical17ObservationFunction in
        multi_intersection.py for the full rationale).

        All values are clipped to ``[0, 1]``.

        Args:
            raw_obs: Raw observation from sumo_rl (numpy array or dict).

        Returns:
            Normalised ``np.float32`` vector of shape ``(_OBS_SIZE,)``.
        """
        obs = np.zeros(_OBS_SIZE, dtype=np.float32)

        if isinstance(raw_obs, dict):
            raw_array = np.concatenate(
                [np.atleast_1d(v) for v in raw_obs.values()], dtype=np.float32
            )
        elif isinstance(raw_obs, np.ndarray):
            raw_array = raw_obs.astype(np.float32)
        else:
            raw_array = np.array(raw_obs, dtype=np.float32)

        # sumo_rl single-agent observations typically contain:
        #   phase one-hot | elapsed | densities | queues
        # The exact layout varies; we adapt from whatever length we get.
        raw_len = len(raw_array)

        if raw_len >= _OBS_SIZE:
            # Enough data – take the first _OBS_SIZE elements
            obs[:] = raw_array[:_OBS_SIZE]
        else:
            # Fewer elements – fill what we can
            obs[:raw_len] = raw_array

        # Extract / infer queue and density segments robustly.
        # sumo_rl default observation order (single_agent, per TS):
        #   [phase_one_hot..., min_green_flag, density..., queue...]
        # We rearrange into our canonical layout.
        try:
            ts_id = list(self._sumo_env.traffic_signals.keys())[0]
            ts = self._sumo_env.traffic_signals[ts_id]

            # Queue lengths (halting vehicles per lane), normalised
            queues_raw = np.array(
                [ts.get_lanes_queue() if hasattr(ts, "get_lanes_queue") else []],
                dtype=np.float32,
            ).flatten()
            # Aggregate per direction (assume 4 groups)
            queues = self._aggregate_directions(queues_raw, max_val=20.0)
            obs[0:4] = queues

            # Densities per lane, normalised
            densities_raw = np.array(
                [ts.get_lanes_density() if hasattr(ts, "get_lanes_density") else []],
                dtype=np.float32,
            ).flatten()
            densities = self._aggregate_directions(densities_raw, max_val=1.0)
            obs[4:8] = densities

            # Current phase one-hot
            current_phase = ts.green_phase if hasattr(ts, "green_phase") else 0
            phase_oh = np.zeros(_NUM_PHASES, dtype=np.float32)
            phase_idx = int(current_phase) % _NUM_PHASES
            phase_oh[phase_idx] = 1.0
            obs[8:12] = phase_oh

            # Normalised elapsed time in current phase
            elapsed = (
                ts.time_since_last_phase_change
                if hasattr(ts, "time_since_last_phase_change")
                else 0.0
            )
            self._phase_elapsed = float(elapsed)
            max_phase_dur = float(self._sumo_env.max_green + self._sumo_env.yellow_time)
            obs[12] = np.clip(elapsed / max(max_phase_dur, 1.0), 0.0, 1.0)

            # Downstream (outgoing lane) occupancies — spillback visibility
            down_raw = np.array(
                [ts.get_out_lanes_density() if hasattr(ts, "get_out_lanes_density") else []],
                dtype=np.float32,
            ).flatten()
            obs[13:17] = self._aggregate_directions(down_raw, max_val=1.0)
            self._served_downstream = served_downstream_occupancies(ts)
        except Exception:  # noqa: BLE001
            self._served_downstream = None
            # Fallback: use the raw array normalised to [0, 1]
            if raw_len > 0:
                max_val = np.abs(raw_array).max()
                if max_val > 0:
                    normalised = np.clip(raw_array / max_val, 0.0, 1.0)
                    obs[: min(raw_len, _OBS_SIZE)] = normalised[: _OBS_SIZE]

        return np.clip(obs, 0.0, 1.0)

    @staticmethod
    def _aggregate_directions(
        per_lane: np.ndarray, *, max_val: float = 1.0
    ) -> np.ndarray:
        """Aggregate per-lane values into four directional buckets.

        If the number of lanes is not evenly divisible by 4 the remainder is
        added to the last bucket.

        Args:
            per_lane: 1-D array of per-lane metric values.
            max_val: Value used for normalisation (clipped to ``[0, 1]``).

        Returns:
            Array of shape ``(4,)`` with normalised directional aggregates.
        """
        result = np.zeros(_NUM_DIRECTIONS, dtype=np.float32)
        n = len(per_lane)
        if n == 0:
            return result
        chunk = max(n // _NUM_DIRECTIONS, 1)
        for i in range(_NUM_DIRECTIONS):
            start = i * chunk
            end = (i + 1) * chunk if i < _NUM_DIRECTIONS - 1 else n
            result[i] = per_lane[start:end].sum()
        if max_val > 0:
            result = np.clip(result / max_val, 0.0, 1.0)
        return result

    # --------------------------------------------------------------------- #
    # Reward computation
    # --------------------------------------------------------------------- #

    def _compute_reward(
        self, obs: np.ndarray, action: int, raw_reward: float
    ) -> float:
        """Compute the reward using the selected function.

        Args:
            obs: Normalised observation vector.
            action: Action taken this step.
            raw_reward: Original reward from sumo_rl.

        Returns:
            Scalar reward value.
        """
        if self._reward_fn_name == "pressure":
            return compute_pressure_reward(obs)
        if self._reward_fn_name == "queue":
            return compute_queue_reward(obs)
        if self._reward_fn_name == "composite":
            last = self._last_action if self._last_action is not None else action
            return self._composite_reward.compute(
                obs,
                last,
                action,
                time_since_phase_change=self._phase_elapsed,
                downstream_occupancy=self._served_downstream,
            )
        # Fallback – use the raw sumo_rl reward
        return float(raw_reward)

    # --------------------------------------------------------------------- #
    # Properties
    # --------------------------------------------------------------------- #

    @property
    def episode_count(self) -> int:
        """Number of completed episodes since construction."""
        return self._episode_count
