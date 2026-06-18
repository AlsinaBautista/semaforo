#!/usr/bin/env python3
# Semáforo Inteligente - Brain Module
"""CLI script to evaluate MARL PPO against fixed and green wave baselines.

Usage::

    python scripts/evaluate_corridor.py \\
        --model-path models/ppo_corridor_v1/ppo_traffic_final \\
        --episodes 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import traci

# Ensure the project root is on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.ppo_agent import PPOTrafficAgent
from src.environments.corridor_env import CorridorVecEnv
from src.coordination.green_wave import Intersection, compute_offsets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MARL PPO against baselines on a corridor.",
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
        "--model-path",
        type=str,
        default="",
        help="Path to the trained MARL PPO model (.zip). If empty, skips PPO eval.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to evaluate each strategy (default: 1).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch SUMO with the GUI (for debugging/visualisation).",
    )
    return parser.parse_args()


def evaluate_strategy(
    env: CorridorVecEnv,
    strategy_name: str,
    n_episodes: int = 1,
    agent: PPOTrafficAgent | None = None,
    offsets: list[float] | None = None,
) -> dict[str, float]:
    """Evaluate a specific control strategy.

    Args:
        env: The CorridorVecEnv.
        strategy_name: Display name.
        n_episodes: Number of episodes to run.
        agent: PPOTrafficAgent if strategy is PPO, else None.
        offsets: List of green wave offsets in seconds, else None.

    Returns:
        Dictionary of averaged metrics.
    """
    print(f"\nEvaluating strategy: {strategy_name}")
    all_rewards = []
    
    # Base fixed duration in seconds (assuming 60s phases)
    fixed_duration = 60
    # Steps per phase: 60s / 5s (delta_time) = 12 steps
    steps_per_phase = fixed_duration // env._sumo_env.delta_time

    for ep in range(n_episodes):
        obs = env.reset()
        ep_rewards = []
        dones = np.array([False] * env.num_envs)
        
        step_idx = 0
        while not np.any(dones):
            if agent is not None:
                # PPO Strategy
                actions = agent.decide(obs)
            else:
                # Fixed Time or Green Wave Strategy
                actions = np.zeros(env.num_envs, dtype=int)
                for i in range(env.num_envs):
                    # Base time elapsed for this intersection
                    t = step_idx * env._sumo_env.delta_time
                    if offsets is not None:
                        # Shift time by offset
                        t -= offsets[i]
                    
                    # Cycle time: 60s green + 60s green = 120s
                    cycle_time = t % 120
                    # Phase 0 is first 60s, Phase 1 is second 60s
                    # If t < 0, it behaves nicely in python modulo
                    if cycle_time < 60:
                        actions[i] = 0
                    else:
                        actions[i] = 1

            env.step_async(actions)
            obs, rewards, dones, infos = env.step_wait()
            ep_rewards.append(rewards.sum())
            step_idx += 1

        total_reward = np.sum(ep_rewards)
        all_rewards.append(total_reward)
        print(f"  Episode {ep + 1}/{n_episodes} - Total Reward: {total_reward:.2f}")

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
    }


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Semáforo Inteligente — MARL Corridor Evaluation")
    print("=" * 60)

    # Re-create environment (don't use VecMonitor to get raw rewards)
    env = CorridorVecEnv(
        net_file=args.net_file,
        route_file=args.route_file,
        use_gui=args.gui,
    )

    results = {}

    # 1. PPO Evaluation
    if args.model_path and Path(args.model_path).with_suffix(".zip").exists():
        agent = PPOTrafficAgent()
        agent.load(args.model_path)
        results["PPO MARL"] = evaluate_strategy(
            env, "PPO MARL", args.episodes, agent=agent
        )
    else:
        print(f"\nSkipping PPO evaluation (model not found: {args.model_path})")

    # 2. Fixed-Time Evaluation
    results["Fixed-Time"] = evaluate_strategy(
        env, "Fixed-Time (Uncoordinated)", args.episodes
    )

    # 3. Theoretical Green Wave Evaluation
    # c0 is at 0m, c1 is at 350m, c2 is at 700m
    corridor_intersections = [
        Intersection("c0", position_m=0.0),
        Intersection("c1", position_m=350.0),
        Intersection("c2", position_m=700.0),
    ]
    # Target speed for route ew_through is maxSpeed 13.89 m/s (50 km/h)
    offsets = compute_offsets(corridor_intersections, target_speed_kmh=50.0)
    results["Green Wave (Theoretical)"] = evaluate_strategy(
        env, "Green Wave (Theoretical)", args.episodes, offsets=offsets
    )

    env.close()

    # Print summary
    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"{'Strategy':<30} | {'Mean Reward':>12} | {'Std':>8}")
    print("-" * 60)
    
    # Sort results by mean reward (descending)
    sorted_results = sorted(results.items(), key=lambda x: x[1]["mean_reward"], reverse=True)
    for name, metrics in sorted_results:
        print(f"{name:<30} | {metrics['mean_reward']:>12.2f} | {metrics['std_reward']:>8.2f}")
    
    print("=" * 60)
    if "PPO MARL" in results and "Fixed-Time" in results:
        ppo_reward = results["PPO MARL"]["mean_reward"]
        fixed_reward = results["Fixed-Time"]["mean_reward"]
        # Since rewards are negative, a smaller absolute value is better
        # Percentage improvement formulation for negative rewards:
        if fixed_reward < 0:
            improvement = ((fixed_reward - ppo_reward) / fixed_reward) * 100
            print(f"  PPO Improvement over Fixed-Time: {improvement:.1f}%")

    if "PPO MARL" in results and "Green Wave (Theoretical)" in results:
        ppo_reward = results["PPO MARL"]["mean_reward"]
        gw_reward = results["Green Wave (Theoretical)"]["mean_reward"]
        if gw_reward < 0:
            improvement = ((gw_reward - ppo_reward) / gw_reward) * 100
            print(f"  PPO Improvement over Green Wave: {improvement:.1f}%")
            
    print("=" * 60)


if __name__ == "__main__":
    main()
