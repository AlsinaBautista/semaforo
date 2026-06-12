# Semáforo Inteligente - Brain Module
"""Centralized-critic PPO RLModule for CTDE MAPPO on the corridor.

Centralized Training with Decentralized Execution:

* The **actor** consumes only the agent's local observation
  (``obs["obs"]``, 32 dims) — sampling and inference never touch the global
  state, so execution stays fully decentralized.
* The **critic** consumes the global corridor state S_t (``obs["state"]``,
  all agents' observations concatenated) and is used exclusively inside the
  Learner: the GAE connector and the PPO loss call :meth:`compute_values`.

Written against Ray RLlib 2.55.x (new API stack).
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import TorchCategorical
from ray.rllib.core.rl_module.apis import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, out_dim),
    )


class CentralizedCriticPPOTorchRLModule(TorchRLModule, ValueFunctionAPI):
    """MAPPO module: decentralized actor, centralized critic over global S_t.

    Expects a ``gymnasium.spaces.Dict`` observation space with keys ``"obs"``
    (local observation) and ``"state"`` (global corridor state).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # RLlib auto-fills catalog_class=PPOCatalog, which cannot encode our
        # Dict observation space and would emit a noisy warning on failure.
        kwargs.pop("catalog_class", None)
        super().__init__(*args, **kwargs)

    @override(RLModule)
    def setup(self) -> None:
        obs_dim = self.observation_space["obs"].shape[0]
        state_dim = self.observation_space["state"].shape[0]
        num_actions = int(self.action_space.n)
        hidden = self.model_config.get("hidden_dim", 256)

        self.actor = _mlp(obs_dim, hidden, num_actions)
        self.critic = _mlp(state_dim, hidden, 1)
        # Required: TorchRLModule.get_*_action_dist_cls() return this attribute.
        self.action_dist_cls = TorchCategorical

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # Inference/exploration path: actor only — decentralized execution.
        return {Columns.ACTION_DIST_INPUTS: self.actor(batch[Columns.OBS]["obs"])}

    @override(RLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        # No Columns.EMBEDDINGS on purpose: the PPO loss then calls
        # compute_values(batch), which reads the global state branch.
        return {Columns.ACTION_DIST_INPUTS: self.actor(batch[Columns.OBS]["obs"])}

    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        # Called by the GeneralAdvantageEstimation learner connector and the
        # PPO loss; must return shape (B,).
        return self.critic(batch[Columns.OBS]["state"]).squeeze(-1)
