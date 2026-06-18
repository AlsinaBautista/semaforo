#!/usr/bin/env python3
# Semáforo Inteligente - Brain Module
"""CLI script to train a MARL PPO agent on a SUMO corridor.

Usage::

    python scripts/train_corridor.py \
        --net-file sumo_networks/corridor/corridor.net.xml \
        --route-file sumo_networks/corridor/corridor.rou.xml \
        --total-timesteps 500000 \
        --save-path models/ppo_corridor
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import gymnasium as gym

# Ensure the project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor, VecEnvWrapper
from src.agents.ppo_agent import PPOTrafficAgent
from src.environments.corridor_env import CorridorVecEnv


class CorridorGymWrapper(gym.Env):
    """Wraps CorridorVecEnv to act as a single Gym environment for SubprocVecEnv."""
    def __init__(self, **kwargs):
        self.venv = CorridorVecEnv(**kwargs)
        # Observation space: 3 agents * 24 features
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3, 24), dtype=np.float32)
        # Action space: 3 agents, each has 2 actions
        self.action_space = gym.spaces.MultiDiscrete([2, 2, 2])
        
    def reset(self, *, seed=None, options=None):
        # We ignore seed/options for the underlying sumo-rl env for now as it's already instantiated
        obs = self.venv.reset()
        return obs, {}
        
    def step(self, action):
        self.venv.step_async(action)
        obs, rewards, dones, infos = self.venv.step_wait()
        # SubprocVecEnv expects a single done boolean
        global_done = bool(np.all(dones))
        # Return 5 values for gymnasium: obs, reward, terminated, truncated, info
        return obs, rewards, global_done, False, {"agents_info": infos}
        
    def close(self):
        self.venv.close()


class FlattenCorridorVecEnv(VecEnvWrapper):
    """Flattens the stacked environments back into individual agents for PPO."""
    def __init__(self, venv):
        super().__init__(venv)
        self.agents_per_env = 3
        self.num_envs = venv.num_envs * self.agents_per_env
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(24,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)
        
    def reset(self):
        obs = self.venv.reset()  # Shape: (n_envs, 3, 24)
        return obs.reshape(self.num_envs, 24)
        
    def step_async(self, actions):
        # actions shape: (n_envs * 3,)
        actions = actions.reshape(self.venv.num_envs, self.agents_per_env)
        self.venv.step_async(actions)
        
    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        
        # Flatten observations and rewards
        flat_obs = obs.reshape(self.num_envs, 24)
        flat_rewards = rewards.reshape(self.num_envs)
        
        # dones is (n_envs,). We repeat it for each agent.
        flat_dones = np.repeat(dones, self.agents_per_env)
        
        flat_infos = []
        for info in infos:
            agents_info = info.get("agents_info", [{}, {}, {}])
            flat_infos.extend(agents_info)
            
        return flat_obs, flat_rewards, flat_dones, flat_infos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a MARL PPO agent on a SUMO corridor.",
    )
    parser.add_argument(
        "--net-file",
        type=str,
        default="brain/sumo_networks/corridor/corridor.net.xml",
        help="Path to the SUMO .net.xml network file.",
    )
    parser.add_argument(
        "--route-file",
        type=str,
        default="brain/sumo_networks/corridor/corridor.rou.xml",
        help="Path to the SUMO .rou.xml route file.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=500_000,
        help="Total training timesteps (default: 500000).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="brain/models/ppo_corridor_v1",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--delta-time",
        type=int,
        default=5,
        help="Decision interval in simulation seconds (default: 5).",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=8,
        help="Number of parallel SUMO instances (default: 8).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO with the GUI (for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    n_envs = args.n_envs
    if n_envs > 1 and args.gui:
        print("Warning: SUMO GUI is not supported with multiple environments. Disabling GUI.")
        args.gui = False

    print("=" * 60)
    print("  Semáforo Inteligente — MARL Corridor Training")
    print("=" * 60)
    print(f"  Net file       : {args.net_file}")
    print(f"  Route file     : {args.route_file}")
    print(f"  Timesteps      : {args.total_timesteps:,}")
    print(f"  Save path      : {args.save_path}")
    print(f"  Delta time     : {args.delta_time}s")
    print(f"  Parallel envs  : {n_envs} SUMO instances ({n_envs * 3} agents)")
    print("=" * 60)

    def make_env():
        return CorridorGymWrapper(
            net_file=args.net_file,
            route_file=args.route_file,
            delta_time=args.delta_time,
            use_gui=args.gui,
        )

    if n_envs > 1:
        # Run multiple SUMO instances in separate processes
        base_env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        base_env = DummyVecEnv([make_env])

    # Flatten the outputs so PPO sees n_envs * 3 parallel agents
    env = FlattenCorridorVecEnv(base_env)
    
    # Wrap in VecMonitor to track episodic returns
    env = VecMonitor(env)

    # Initialise and train the agent.
    agent = PPOTrafficAgent()
    model = agent.train(
        env=env,
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
    )

    print(f"\nModel saved to {args.save_path}")
    env.close()
    print("\nDone ✓")


if __name__ == "__main__":
    main()
