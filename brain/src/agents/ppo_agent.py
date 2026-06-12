# Semáforo Inteligente - Brain Module
"""PPO agent for single-intersection traffic-signal control.

Uses Stable-Baselines3's PPO implementation with an MLP policy.  The
default hyper-parameters are tuned for the ``SingleIntersectionEnv``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy


class PPOTrafficAgent:
    """Proximal Policy Optimization agent for traffic-signal control.

    Args:
        learning_rate: Adam optimiser learning rate.
        n_steps: Number of steps per rollout buffer collection.
        batch_size: Mini-batch size for policy updates.
        n_epochs: Number of gradient-descent epochs per update.
        gamma: Discount factor.
        gae_lambda: GAE lambda for advantage estimation.
        clip_range: PPO clipping parameter.
        verbose: SB3 verbosity level (0 = silent, 1 = info).
    """

    def __init__(
        self,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        verbose: int = 1,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.verbose = verbose

        self._model: PPO | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        env: gym.Env,
        total_timesteps: int = 100_000,
        save_path: str | Path = "models/ppo_traffic",
        eval_env: gym.Env | None = None,
        eval_freq: int = 5_000,
    ) -> PPO:
        """Train the PPO model on the given environment.

        Args:
            env: Gymnasium training environment.
            total_timesteps: Total number of environment steps.
            save_path: Directory where the trained model is saved.
            eval_env: Optional separate environment for periodic evaluation.
            eval_freq: Evaluation frequency in timesteps.

        Returns:
            The trained ``PPO`` model instance.
        """
        self._model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=self.learning_rate,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            verbose=self.verbose,
        )

        callbacks = []
        if eval_env is not None:
            callbacks.append(
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(save_path),
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False,
                )
            )

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks if callbacks else None,
        )

        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        model_file = save_dir / "ppo_traffic_final"
        self._model.save(str(model_file))

        return self._model

    def evaluate(
        self,
        env: gym.Env,
        n_episodes: int = 5,
    ) -> dict[str, Any]:
        """Evaluate the current model on *env* for *n_episodes*.

        Args:
            env: Gymnasium evaluation environment.
            n_episodes: Number of episodes to run.

        Returns:
            Dictionary with ``mean_reward``, ``std_reward``, and
            ``episode_rewards``.

        Raises:
            RuntimeError: If no model has been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")

        mean_reward, std_reward = evaluate_policy(
            self._model, env, n_eval_episodes=n_episodes, deterministic=True
        )

        # Collect per-episode rewards for downstream analysis.
        episode_rewards: list[float] = []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_reward = 0.0
            done = False
            while not done:
                action, _ = self._model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated
            episode_rewards.append(total_reward)

        return {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "episode_rewards": episode_rewards,
        }

    def load(self, model_path: str | Path) -> PPO:
        """Load a previously saved PPO model.

        Args:
            model_path: Path to the saved ``.zip`` model file (without
                extension is fine — SB3 appends it automatically).

        Returns:
            The loaded ``PPO`` model.
        """
        self._model = PPO.load(str(model_path))
        return self._model

    def decide(self, obs: np.ndarray) -> int:
        """Select an action given an observation (inference mode).

        Args:
            obs: Current environment observation vector.

        Returns:
            Discrete action index.

        Raises:
            RuntimeError: If no model has been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)
