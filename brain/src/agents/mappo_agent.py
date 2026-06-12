# Semáforo Inteligente - Brain Module
"""Multi-Agent PPO (MAPPO) agent via Ray RLlib.

This module is a **Phase 4** placeholder.  The full implementation will
use Ray RLlib's multi-agent API to train a shared PPO policy (or
independent policies) across multiple intersection agents.

TODO (Phase 4):
    * Configure RLlib ``PPOConfig`` for multi-agent training.
    * Map each traffic-signal agent to a policy.
    * Implement centralised-critic / decentralised-actor variant.
    * Add checkpoint save / restore.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MAPPOTrafficAgent:
    """Multi-Agent PPO controller for coordinated traffic signals.

    This agent is designed to work with ``MultiIntersectionEnv`` (Phase 4).

    Args:
        num_agents: Number of intersection agents.
        learning_rate: Optimiser learning rate.
        share_policy: If ``True``, all agents share the same policy
            network (parameter sharing).

    Note:
        All methods raise ``NotImplementedError`` until Phase 4.
    """

    def __init__(
        self,
        num_agents: int = 3,
        learning_rate: float = 3e-4,
        share_policy: bool = True,
    ) -> None:
        self.num_agents = num_agents
        self.learning_rate = learning_rate
        self.share_policy = share_policy

        # TODO (Phase 4): Initialise RLlib PPOConfig.
        self._algo = None

    def train(
        self,
        env_config: dict[str, Any],
        total_timesteps: int = 500_000,
        save_path: str = "models/mappo_traffic",
    ) -> None:
        """Train the MAPPO policy on the multi-intersection environment.

        Args:
            env_config: Configuration dict passed to the env constructor.
            total_timesteps: Approximate total environment steps.
            save_path: Directory for checkpoints.

        Raises:
            NotImplementedError: Pending Phase 4.
        """
        # TODO (Phase 4): Build RLlib Algorithm, call .train() loop.
        raise NotImplementedError("MAPPOTrafficAgent is a Phase 4 placeholder.")

    def evaluate(
        self,
        env_config: dict[str, Any],
        n_episodes: int = 5,
    ) -> dict[str, Any]:
        """Evaluate trained policies.

        Args:
            env_config: Configuration dict for the evaluation environment.
            n_episodes: Number of evaluation episodes.

        Returns:
            Metrics dictionary (per-agent and aggregated).

        Raises:
            NotImplementedError: Pending Phase 4.
        """
        # TODO (Phase 4): Run evaluation loop.
        raise NotImplementedError("MAPPOTrafficAgent is a Phase 4 placeholder.")

    def load(self, checkpoint_path: str) -> None:
        """Restore a trained algorithm from a checkpoint.

        Args:
            checkpoint_path: Path to the RLlib checkpoint directory.

        Raises:
            NotImplementedError: Pending Phase 4.
        """
        # TODO (Phase 4): Restore from checkpoint.
        raise NotImplementedError("MAPPOTrafficAgent is a Phase 4 placeholder.")

    def decide(self, obs: dict[str, np.ndarray]) -> dict[str, int]:
        """Compute actions for all agents given their observations.

        Args:
            obs: Mapping from agent id to observation array.

        Returns:
            Mapping from agent id to discrete action.

        Raises:
            NotImplementedError: Pending Phase 4.
        """
        # TODO (Phase 4): Use trained policy to compute actions.
        raise NotImplementedError("MAPPOTrafficAgent is a Phase 4 placeholder.")
