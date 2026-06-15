#!/usr/bin/env python3
# Semáforo Inteligente - Brain Module
"""CLI script to train a PPO agent on a single SUMO intersection.

Usage::

    python scripts/train_single.py \\
        --net-file sumo_networks/single_intersection/intersection.net.xml \\
        --route-file sumo_networks/single_intersection/intersection.rou.xml \\
        --total-timesteps 100000 \\
        --save-path models/ppo_single
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

from src.agents.ppo_agent import PPOTrafficAgent
from src.environments.single_intersection import SingleIntersectionEnv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on a single SUMO intersection.",
    )
    parser.add_argument(
        "--net-file",
        type=str,
        required=True,
        help="Path to the SUMO .net.xml network file.",
    )
    parser.add_argument(
        "--route-file",
        type=str,
        required=True,
        help="Path to the SUMO .rou.xml route file.",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total training timesteps (default: 100000).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/ppo_single",
        help="Directory to save the trained model (default: models/ppo_single).",
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
    parser.add_argument(
        "--n-envs",
        type=int,
        default=0,
        help="Number of parallel environments (default: 0 = os.cpu_count()).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: create environment, train agent, save model and metrics."""
    args = parse_args()

    n_envs = args.n_envs if args.n_envs > 0 else (os.cpu_count() or 1)
    
    if n_envs > 1 and args.gui:
        print("Warning: SUMO GUI is not supported with multiple environments. Disabling GUI.")
        args.gui = False

    print("=" * 60)
    print("  Semáforo Inteligente — PPO Single-Intersection Training")
    print("=" * 60)
    print(f"  Net file       : {args.net_file}")
    print(f"  Route file     : {args.route_file}")
    print(f"  Timesteps      : {args.total_timesteps:,}")
    print(f"  Save path      : {args.save_path}")
    print(f"  Delta time     : {args.delta_time}s")
    print(f"  Parallel envs  : {n_envs}")
    print("=" * 60)

    # Create training environment.
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    env_kwargs = {
        "net_file": args.net_file,
        "route_file": args.route_file,
        "delta_time": args.delta_time,
        "use_gui": args.gui,
    }

    vec_env_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    env = make_vec_env(
        SingleIntersectionEnv,
        n_envs=n_envs,
        env_kwargs=env_kwargs,
        vec_env_cls=vec_env_cls,
    )

    # Initialise and train the agent.
    agent = PPOTrafficAgent()
    model = agent.train(
        env=env,
        total_timesteps=args.total_timesteps,
        save_path=args.save_path,
    )

    # Evaluate the trained model briefly.
    print("\n--- Quick evaluation (3 episodes) ---")
    metrics = agent.evaluate(env, n_episodes=3)
    print(f"  Mean reward : {metrics['mean_reward']:.2f}")
    print(f"  Std reward  : {metrics['std_reward']:.2f}")

    # Persist metrics alongside the model.
    metrics_path = Path(args.save_path) / "training_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    env.close()
    print("\nDone ✓")


if __name__ == "__main__":
    main()
