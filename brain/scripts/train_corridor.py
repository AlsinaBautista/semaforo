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
import json
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3.common.vec_env import VecMonitor
from src.agents.ppo_agent import PPOTrafficAgent
from src.environments.corridor_env import CorridorVecEnv


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
        "--gui",
        action="store_true",
        help="Launch SUMO with the GUI (for debugging).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Semáforo Inteligente — MARL Corridor Training")
    print("=" * 60)
    print(f"  Net file       : {args.net_file}")
    print(f"  Route file     : {args.route_file}")
    print(f"  Timesteps      : {args.total_timesteps:,}")
    print(f"  Save path      : {args.save_path}")
    print(f"  Delta time     : {args.delta_time}s")
    print("=" * 60)

    # Create the Vector Environment. 
    # CorridorVecEnv naturally acts as a VecEnv with 3 sub-environments (one per intersection).
    env = CorridorVecEnv(
        net_file=args.net_file,
        route_file=args.route_file,
        delta_time=args.delta_time,
        use_gui=args.gui,
    )
    
    # Wrap in VecMonitor to track episodic returns
    env = VecMonitor(env)

    # Initialise and train the agent.
    # Note: PPO expects obs shape (N, ...), which CorridorVecEnv provides directly.
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
