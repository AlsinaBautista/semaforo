#!/usr/bin/env python3
# Semáforo Inteligente - Brain Module
"""Verify that rewards and gradients stay within mathematically bounded limits.

Closes the verification loop for audit findings A-3/A-4:

1. Every reward function — including the reworked night-mode PBRS — returns
   values inside ``[-1.0, 1.0]`` for simulated state tensors, including
   adversarial extremes (queues at 10x saturation) that used to trigger the
   unbounded cliff.
2. The PBRS term is a true potential-based shaping: its discounted sum over a
   trajectory telescopes to ``gamma^T * phi(s_T) - phi(s_0)`` (optimal-policy
   invariance) and every per-step term is bounded.
3. The anti-flicker penalty fires only on a physical min-green violation.
4. The CTDE centralized-critic module produces finite values and, after the
   training-time gradient clipping (``grad_clip=0.5``, as in the PPOConfig),
   every gradient element lies inside ``[-1.0, 1.0]``.

Run from the repo root::

    PYTHONPATH=brain .venv/bin/python brain/scripts/verify_rewards.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

BRAIN_ROOT = Path(__file__).resolve().parent.parent
if str(BRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(BRAIN_ROOT))

import numpy as np

from src.rewards import (
    CompositeReward,
    compute_night_mode_reward,
    compute_night_potential,
    compute_pressure_reward,
    compute_queue_reward,
)

GAMMA = 0.99
MIN_GREEN = 5.0
BOUND = 1.0


def _assert_bounded(value: float, label: str) -> None:
    assert np.isfinite(value), f"{label}: non-finite reward {value}"
    assert -BOUND <= value <= BOUND, f"{label}: reward {value} outside [-1, 1]"


def check_reward_bounds(rng: random.Random) -> None:
    """All reward functions stay in [-1, 1] under random + adversarial states."""
    composite = CompositeReward(min_green=MIN_GREEN)

    # Adversarial dict states: empty, zeros, saturated, 10x saturation cliff.
    adversarial = [
        {"queue": [], "flow": []},
        {"queue": [0, 0, 0, 0], "flow": [0, 0, 0, 0]},
        {"queue": [20, 20, 20, 20], "flow": [0, 0, 0, 0]},
        {"queue": [200, 200, 200, 200], "flow": [0, 0, 0, 0]},
        {"queue": [10_000, 10_000], "flow": [0, 0]},
    ]
    for i, obs in enumerate(adversarial):
        _assert_bounded(compute_pressure_reward(obs), f"pressure/adv{i}")
        _assert_bounded(compute_queue_reward(obs), f"queue/adv{i}")
        _assert_bounded(
            composite.compute(obs, 0, 1, time_since_phase_change=1.0),
            f"composite/adv{i}",
        )
        _assert_bounded(
            compute_night_mode_reward(obs, 0, [0, 1], [2, 3]),
            f"night/adv{i}",
        )

    # Random states (dict and canonical numpy layouts).
    for i in range(500):
        queues = [rng.uniform(0, 200) for _ in range(4)]
        flows = [rng.uniform(0, 50) for _ in range(4)]
        obs_dict = {"queue": queues, "flow": flows}
        obs_vec = np.array([rng.random() for _ in range(13)], dtype=np.float32)
        elapsed = rng.choice([None, rng.uniform(0, 70)])

        _assert_bounded(compute_pressure_reward(obs_dict), f"pressure/rng{i}")
        _assert_bounded(compute_queue_reward(obs_vec), f"queue/rng{i}")
        _assert_bounded(
            composite.compute(
                obs_vec,
                rng.randint(0, 3),
                rng.randint(0, 3),
                time_since_phase_change=elapsed,
            ),
            f"composite/rng{i}",
        )
        prev = {"queue": [rng.uniform(0, 500) for _ in range(4)]}
        _assert_bounded(
            compute_night_mode_reward(
                obs_dict, 0, [0, 1], [2, 3], prev_observation=prev, gamma=GAMMA
            ),
            f"night/rng{i}",
        )
    print("  [1/4] reward bounds in [-1, 1] (incl. 10x-saturation cliff)  OK")


def check_pbrs_invariance(rng: random.Random) -> None:
    """Discounted shaping sum telescopes to gamma^T*phi(s_T) - phi(s_0)."""
    avenue, cross = [0, 1], [2, 3]
    for _ in range(50):
        T = rng.randint(2, 40)
        # Trajectory mixing night-mode states (cross empty) and normal states.
        traj = []
        for _ in range(T + 1):
            cross_q = 0.0 if rng.random() < 0.6 else rng.uniform(1, 30)
            traj.append(
                {"queue": [rng.uniform(0, 60), rng.uniform(0, 60), cross_q, cross_q]}
            )

        discounted_sum = 0.0
        for t in range(T):
            f_t = compute_night_mode_reward(
                traj[t + 1], 0, avenue, cross,
                prev_observation=traj[t], gamma=GAMMA,
            )
            _assert_bounded(f_t, f"pbrs/step{t}")
            discounted_sum += (GAMMA ** t) * f_t

        phi_0 = compute_night_potential(traj[0], avenue, cross)
        phi_T = compute_night_potential(traj[T], avenue, cross)
        expected = (GAMMA ** T) * phi_T - phi_0
        assert np.isclose(discounted_sum, expected, atol=1e-9), (
            f"PBRS telescoping violated: {discounted_sum} != {expected}"
        )
    print("  [2/4] PBRS telescoping invariance (policy-invariant shaping)  OK")


def check_anti_flicker() -> None:
    """Flicker penalty fires only on a physical min-green violation."""
    cr = CompositeReward(
        pressure_weight=0.0, queue_weight=0.0, flickering_weight=1.0,
        min_green=MIN_GREEN,
    )
    obs = {"queue": [0, 0], "flow": [0, 0]}  # pressure = queue = 0

    # Switch before min-green -> violation -> -1.
    assert cr.compute(obs, 0, 1, time_since_phase_change=2.0) == -1.0
    # Switch at/after min-green -> legal -> 0.
    assert cr.compute(obs, 0, 1, time_since_phase_change=MIN_GREEN) == 0.0
    assert cr.compute(obs, 0, 1, time_since_phase_change=30.0) == 0.0
    # Unknown timing -> violation unprovable -> 0.
    assert cr.compute(obs, 0, 1, time_since_phase_change=None) == 0.0
    # No switch -> never penalised, even inside min-green.
    assert cr.compute(obs, 1, 1, time_since_phase_change=1.0) == 0.0
    print("  [3/4] anti-flicker penalises only min-green violations        OK")


def check_ctde_gradients() -> None:
    """Centralized-critic module: finite values, gradients in [-1, 1]."""
    import logging

    import torch
    from gymnasium import spaces
    from ray.rllib.core.columns import Columns
    from ray.rllib.core.rl_module.rl_module import RLModuleSpec

    from src.training.centralized_critic_module import (
        CentralizedCriticPPOTorchRLModule,
    )

    # RLlib 2.55's RLModule.__init__ unconditionally builds the deprecated
    # RLModuleConfig for backwards compat and logs a DeprecationWarning for
    # every module (including RLlib's own defaults) — internal noise, not us.
    logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)
    logging.getLogger("ray.rllib").setLevel(logging.ERROR)

    obs_space = spaces.Dict(
        {
            "obs": spaces.Box(0.0, 1.0, shape=(32,), dtype=np.float32),
            "state": spaces.Box(0.0, 1.0, shape=(96,), dtype=np.float32),
        }
    )
    module = RLModuleSpec(
        module_class=CentralizedCriticPPOTorchRLModule,
        observation_space=obs_space,
        action_space=spaces.Discrete(2),
        model_config={"hidden_dim": 256},
    ).build()

    torch.manual_seed(0)
    batch_size = 64
    batch = {
        Columns.OBS: {
            "obs": torch.rand(batch_size, 32),
            "state": torch.rand(batch_size, 96),
        }
    }

    fwd_out = module._forward_train(batch)
    logits = fwd_out[Columns.ACTION_DIST_INPUTS]
    values = module.compute_values(batch)
    assert values.shape == (batch_size,), f"values shape {values.shape} != (B,)"
    assert torch.isfinite(logits).all(), "non-finite actor logits"
    assert torch.isfinite(values).all(), "non-finite critic values"

    # Surrogate PPO-style loss: policy term + value regression onto bounded
    # targets in [-1, 1] (the regime guaranteed by the bounded rewards above).
    actions = torch.randint(0, 2, (batch_size,))
    logp = torch.log_softmax(logits, dim=-1)
    policy_loss = -logp[torch.arange(batch_size), actions].mean()
    value_targets = torch.empty(batch_size).uniform_(-1.0, 1.0)
    value_loss = torch.nn.functional.mse_loss(values, value_targets)
    (policy_loss + value_loss).backward()

    params = [p for p in module.parameters() if p.grad is not None]
    assert params, "no gradients produced"
    # Same clipping the training config applies (PPOConfig grad_clip=0.5).
    torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
    for p in params:
        assert torch.isfinite(p.grad).all(), "non-finite gradient"
        assert p.grad.abs().max().item() <= BOUND, (
            f"gradient element {p.grad.abs().max().item()} outside [-1, 1]"
        )
    print("  [4/4] CTDE critic gradients finite and in [-1, 1] post-clip   OK")


def main() -> None:
    rng = random.Random(42)
    print("verify_rewards: bounded-reward / bounded-gradient checks (A-3, A-4)")
    check_reward_bounds(rng)
    check_pbrs_invariance(rng)
    check_anti_flicker()
    check_ctde_gradients()
    print("OK — all reward/gradient bounds verified")


if __name__ == "__main__":
    main()
