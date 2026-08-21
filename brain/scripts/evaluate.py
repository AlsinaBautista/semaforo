#!/usr/bin/env python3
# Semáforo Inteligente - Brain Module
"""CLI script to evaluate a trained PPO model against the fixed-time baseline.

Usage::

    python scripts/evaluate.py \\
        --model-path models/ppo_single/ppo_traffic_final \\
        --net-file sumo_networks/single_intersection/intersection.net.xml \\
        --route-file sumo_networks/single_intersection/intersection.rou.xml \\
        --episodes 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.agents.fixed_time_baseline import FixedTimeBaseline
from src.agents.ppo_agent import PPOTrafficAgent
from src.environments.single_intersection import SingleIntersectionEnv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO model vs fixed-time baseline.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved SB3 model (without .zip extension).",
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
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5).",
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
        help="Run SUMO with GUI to visualize the simulation.",
    )
    return parser.parse_args()


def run_fixed_baseline(
    env: SingleIntersectionEnv,
    n_episodes: int,
    delta_time: int,
) -> list[float]:
    """Run the fixed-time baseline for *n_episodes* and collect rewards.

    Args:
        env: The evaluation environment.
        n_episodes: Number of episodes.
        delta_time: Decision interval (must match the env).

    Returns:
        List of total episode rewards.
    """
    baseline = FixedTimeBaseline(phase_durations=[30, 30], delta_time=delta_time)
    episode_rewards: list[float] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        baseline.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = baseline.decide(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)

    return episode_rewards


def main() -> None:
    """Entry point: evaluate PPO and baseline, print comparison table."""
    args = parse_args()

    print("=" * 60)
    print("  Semáforo Inteligente — Model Evaluation")
    print("=" * 60)

    env = SingleIntersectionEnv(
        net_file=args.net_file,
        route_file=args.route_file,
        delta_time=args.delta_time,
        use_gui=args.gui,
    )

    # --- PPO agent ---------------------------------------------------------
    print(f"\nEvaluating PPO model: {args.model_path}")
    agent = PPOTrafficAgent()
    agent.load(args.model_path)
    
    # Custom evaluation loop to slow down the GUI
    import time
    ppo_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action = agent.decide(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if args.gui:
                time.sleep(0.1)  # Artificial delay so the user can see it!
        ppo_rewards.append(total_reward)
    
    ppo_metrics = {
        "mean_reward": float(np.mean(ppo_rewards)),
        "std_reward": float(np.std(ppo_rewards)),
    }

    # --- Fixed-time baseline -----------------------------------------------
    print("Evaluating fixed-time baseline …")
    fixed_rewards = run_fixed_baseline(env, args.episodes, args.delta_time)

    env.close()

    # --- Comparison table --------------------------------------------------
    ppo_mean = ppo_metrics["mean_reward"]
    ppo_std = ppo_metrics["std_reward"]
    fixed_mean = float(np.mean(fixed_rewards))
    fixed_std = float(np.std(fixed_rewards))
    improvement = ((ppo_mean - fixed_mean) / abs(fixed_mean) * 100) if fixed_mean != 0 else 0.0

    print("\n" + "=" * 60)
    print(f"  {'Controller':<20} {'Mean Reward':>14} {'Std':>10}")
    print("-" * 60)
    print(f"  {'PPO Agent':<20} {ppo_mean:>14.2f} {ppo_std:>10.2f}")
    print(f"  {'Fixed-Time':<20} {fixed_mean:>14.2f} {fixed_std:>10.2f}")
    print("-" * 60)
    print(f"  {'Improvement':<20} {improvement:>13.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
