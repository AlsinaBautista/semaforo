"""Corridor environment for MARL traffic signal control.

Wraps a SUMO environment with multiple intersections and exposes it
as a Stable-Baselines3 ``VecEnv``. This allows for true parameter sharing:
the single PPO policy is queried for all intersections simultaneously,
and learns from the combined experiences of all intersections.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from src.environments import sumo_home as _sumo_home  # noqa: F401
import sumo_rl

from src.rewards.smart_delay import compute_smart_delay_reward
from src.rewards.spillback import served_downstream_occupancies

logger = logging.getLogger(__name__)

_NUM_DIRECTIONS = 4
_NUM_PHASES = 2
_OBS_SIZE_OWN = 16  # Own state size
_OBS_SIZE_NEIGHBOR = 4  # Neighbor state size [phase(2), elapsed(1), avenue_queue(1)]
_OBS_SIZE_TOTAL = _OBS_SIZE_OWN + _OBS_SIZE_NEIGHBOR * 2  # 16 + 4 + 4 = 24


class CorridorVecEnv(VecEnv):
    """MARL Vector Environment for a linear traffic corridor.

    Args:
        net_file: SUMO .net.xml file.
        route_file: SUMO .rou.xml file.
        delta_time: Decision interval in seconds.
        use_gui: Whether to use sumo-gui.
        min_green: Minimum green time in seconds.
        max_green: Maximum green time in seconds.
        yellow_time: Yellow time in seconds.
        num_seconds: Total simulation time in seconds.
        seed: Random seed.
    """

    def __init__(
        self,
        net_file: str | Path,
        route_file: str | Path,
        delta_time: int = 5,
        use_gui: bool = False,
        min_green: int = 5,
        max_green: int = 3600,
        yellow_time: int = 3,
        num_seconds: int = 3600,
        seed: int | None = None,
    ) -> None:
        self._net_file = str(Path(net_file).resolve())
        self._route_file = str(Path(route_file).resolve())
        
        # We instantiate a multi-agent SUMO environment.
        self._sumo_env = sumo_rl.SumoEnvironment(
            net_file=self._net_file,
            route_file=self._route_file,
            use_gui=use_gui,
            num_seconds=num_seconds,
            delta_time=delta_time,
            min_green=min_green,
            max_green=max_green,
            yellow_time=yellow_time,
            single_agent=False,
            sumo_seed=seed if seed is not None else "random",
        )

        # Assumes sorted ts_ids correspond to the corridor order (c0 -> c1 -> c2).
        self._ts_ids = sorted(list(self._sumo_env.traffic_signals.keys()))
        self._num_envs = len(self._ts_ids)

        action_space = gym.spaces.Discrete(_NUM_PHASES)
        observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(_OBS_SIZE_TOTAL,), dtype=np.float32
        )

        super().__init__(self._num_envs, observation_space, action_space)

        # Internal state to handle async step
        self._actions: dict[str, int] = {}
        self._phase_elapsed: dict[str, float] = {ts_id: 0.0 for ts_id in self._ts_ids}
        self._last_obs_raw: dict[str, np.ndarray] = {}

    def reset(self) -> np.ndarray:
        raw_obs_dict, _, info_dict = self._sumo_env.reset()
        self._last_obs_raw = raw_obs_dict
        self._phase_elapsed = {ts_id: 0.0 for ts_id in self._ts_ids}
        return self._build_observations(raw_obs_dict)

    def step_async(self, actions: np.ndarray) -> None:
        """Store actions to be executed in step_wait."""
        self._actions = {}
        for i, ts_id in enumerate(self._ts_ids):
            # Same boundary correction as single_intersection.py
            ts = self._sumo_env.traffic_signals[ts_id]
            current_phase = ts.green_phase
            num_phases = ts.num_green_phases

            act = int(actions[i])
            if act >= num_phases:
                act = current_phase
            if act != current_phase and act != (current_phase + 1) % num_phases:
                act = (current_phase + 1) % num_phases

            self._actions[ts_id] = act

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        """Execute the stored actions and return new state."""
        raw_obs, raw_reward, dones_dict, infos = self._sumo_env.step(self._actions)
        self._last_obs_raw = raw_obs

        obs_array = self._build_observations(raw_obs)
        
        rewards = np.zeros(self._num_envs, dtype=np.float32)
        dones = np.zeros(self._num_envs, dtype=bool)
        info_list: list[dict[str, Any]] = []

        global_done = False
        for i, ts_id in enumerate(self._ts_ids):
            ts = self._sumo_env.traffic_signals[ts_id]
            
            # Smart delay reward requires own state observation vector.
            # We slice the first 16 elements (own state).
            own_obs = obs_array[i, :_OBS_SIZE_OWN]
            rewards[i] = compute_smart_delay_reward(ts, own_obs)
            
            is_done = dones_dict.get(ts_id, False)
            dones[i] = is_done
            
            # We might use info for statistics
            info_list.append(infos.get(ts_id, {}))
            
            if dones_dict.get("__all__", False):
                global_done = True

        if global_done:
            obs_array = self.reset()

        return obs_array, rewards, dones, info_list

    def close(self) -> None:
        self._sumo_env.close()

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        # Return none or raise to fulfill API
        return [None] * self._num_envs

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list[Any]:
        return [None] * self._num_envs

    def env_is_wrapped(self, wrapper_class: type, indices=None) -> list[bool]:
        return [False] * self._num_envs

    def _build_observations(self, raw_obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        """Construct the 24-element observation vector for each agent."""
        obs_array = np.zeros((self._num_envs, _OBS_SIZE_TOTAL), dtype=np.float32)

        own_states = {}
        for ts_id in self._ts_ids:
            own_states[ts_id] = self._build_own_state(ts_id)

        for i, ts_id in enumerate(self._ts_ids):
            # Own state
            obs_array[i, :_OBS_SIZE_OWN] = own_states[ts_id]

            # Upstream state (i - 1)
            up_start = _OBS_SIZE_OWN
            up_end = up_start + _OBS_SIZE_NEIGHBOR
            if i > 0:
                up_id = self._ts_ids[i - 1]
                obs_array[i, up_start:up_end] = self._build_neighbor_state(
                    up_id, direction_to_me="E"  # c0 is west of c1, traffic flows East
                )
            
            # Downstream state (i + 1)
            down_start = up_end
            down_end = down_start + _OBS_SIZE_NEIGHBOR
            if i < self._num_envs - 1:
                down_id = self._ts_ids[i + 1]
                obs_array[i, down_start:down_end] = self._build_neighbor_state(
                    down_id, direction_to_me="W" # c2 is east of c1, traffic flows West
                )

        return obs_array

    def _build_own_state(self, ts_id: str) -> np.ndarray:
        """Build the 16-element own state vector."""
        ts = self._sumo_env.traffic_signals[ts_id]
        obs = np.zeros(_OBS_SIZE_OWN, dtype=np.float32)

        try:
            # 0-3: Queue lengths
            queues_raw = np.array(ts.get_lanes_queue()).flatten()
            queues = self._aggregate_directions(queues_raw, max_val=20.0)
            obs[0:4] = queues

            # 4-7: Densities
            densities_raw = np.array(ts.get_lanes_density()).flatten()
            densities = self._aggregate_directions(densities_raw, max_val=1.0)
            obs[4:8] = densities

            # 8-9: Phase one-hot
            current_phase = ts.green_phase
            phase_idx = int(current_phase) % _NUM_PHASES
            obs[8 + phase_idx] = 1.0

            # 10: Elapsed time
            elapsed = ts.time_since_last_phase_change
            self._phase_elapsed[ts_id] = float(elapsed)
            obs[10] = np.clip(elapsed / 120.0, 0.0, 1.0)

            # 11-14: Downstream
            down_raw = np.array(ts.get_out_lanes_density()).flatten()
            obs[11:15] = self._aggregate_directions(down_raw, max_val=1.0)

            # 15: Imbalance
            q_ns = obs[0] + obs[2]
            q_ew = obs[1] + obs[3]
            imbalance = abs(q_ns - q_ew) / max(q_ns + q_ew, 0.01)
            obs[15] = np.clip(imbalance, 0.0, 1.0)

        except Exception as e:
            logger.error(f"Error building own state for {ts_id}: {e}")

        return np.clip(obs, 0.0, 1.0)

    def _build_neighbor_state(self, neighbor_id: str, direction_to_me: str) -> np.ndarray:
        """Build the 4-element neighbor state vector.
        
        direction_to_me: 'E' means traffic flows from neighbor to me via Eastward lanes.
                         So we look at the neighbor's Eastward queue (index 1).
                         'W' means traffic flows from neighbor to me via Westward lanes.
                         So we look at the neighbor's Westward queue (index 3).
        """
        ts = self._sumo_env.traffic_signals[neighbor_id]
        obs = np.zeros(_OBS_SIZE_NEIGHBOR, dtype=np.float32)

        try:
            # 0-1: Phase one-hot
            current_phase = ts.green_phase
            phase_idx = int(current_phase) % _NUM_PHASES
            obs[phase_idx] = 1.0

            # 2: Elapsed time
            elapsed = ts.time_since_last_phase_change
            obs[2] = np.clip(elapsed / 120.0, 0.0, 1.0)

            # 3: Avenue queue (queue towards me)
            queues_raw = np.array(ts.get_lanes_queue()).flatten()
            queues = self._aggregate_directions(queues_raw, max_val=20.0)
            if direction_to_me == "E":
                # Looking at Eastward queue of neighbor
                obs[3] = queues[1]
            elif direction_to_me == "W":
                # Looking at Westward queue of neighbor
                obs[3] = queues[3]
            
        except Exception as e:
            logger.error(f"Error building neighbor state for {neighbor_id}: {e}")

        return np.clip(obs, 0.0, 1.0)

    @staticmethod
    def _aggregate_directions(per_lane: np.ndarray, *, max_val: float = 1.0) -> np.ndarray:
        """Aggregate per-lane values into [N, E, S, W] buckets."""
        num_lanes = len(per_lane)
        if num_lanes == 0:
            return np.zeros(4, dtype=np.float32)
        
        # We assume the lanes are ordered starting from North, going clockwise.
        # N, E, S, W
        lanes_per_dir = num_lanes // 4
        if lanes_per_dir == 0:
            return np.zeros(4, dtype=np.float32)

        agg = np.zeros(4, dtype=np.float32)
        for i in range(4):
            start = i * lanes_per_dir
            end = start + lanes_per_dir
            # Sum the values and normalize
            val = np.sum(per_lane[start:end])
            agg[i] = val / max_val
            
        return np.clip(agg, 0.0, 1.0)
