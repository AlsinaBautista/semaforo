"""Composite reward function combining pressure, queue, and flickering penalty.

This multi-objective reward allows the designer to trade off between
throughput optimality (pressure), queue minimisation, and signal stability
(flickering penalty) via configurable weights.
"""

from __future__ import annotations

from src.rewards.pressure import compute_pressure_reward
from src.rewards.queue_length import compute_queue_reward
from src.rewards.spillback import compute_spillback_penalty

import numpy as np


class CompositeReward:
    """Multi-objective reward with configurable weights.

    The composite reward is computed as::

        R = w_p · R_pressure + w_q · R_queue + w_f · R_flicker + w_s · R_spill

    ``R_flicker`` penalises phase switches as a function of how long the
    previous phase was held: ``-1`` below ``min_green`` (a switch the signal
    heads cannot legally perform), decaying linearly to ``0`` over the
    ``stability_window`` after ``min_green``.  The decay is what damps
    legal-but-erratic ping-ponging at the minimum allowed period, which a
    pure min-green gate cannot (the environment already blocks sub-min-green
    switches, so a gated-only penalty never fires in training).  Mature
    switches (held longer than ``min_green + stability_window``) are free.

    ``R_spill`` penalises holding a green toward a saturated downstream
    block (see :func:`compute_spillback_penalty`); it is 0 when no
    downstream information is supplied.

    All component rewards are individually in ``[-1, 0]`` so the composite
    is also bounded.

    Args:
        pressure_weight: Weight for the pressure reward component.
        queue_weight: Weight for the queue-length reward component.
        flickering_weight: Weight for the flickering penalty.
        min_green: Physical minimum green duration (seconds) below which a
            phase switch counts as flickering.
        stability_window: Seconds after ``min_green`` over which the
            flickering penalty decays linearly to zero.
        spillback_weight: Weight for the downstream-saturation penalty.
        rho_crit: Downstream occupancy above which spillback risk is
            penalised.
    """

    def __init__(
        self,
        pressure_weight: float = 0.5,
        queue_weight: float = 0.3,
        flickering_weight: float = 0.1,
        min_green: float = 5.0,
        stability_window: float = 10.0,
        spillback_weight: float = 0.1,
        rho_crit: float = 0.85,
    ) -> None:
        total = pressure_weight + queue_weight + flickering_weight + spillback_weight
        if not np.isclose(total, 1.0, atol=0.05):
            import warnings

            warnings.warn(
                f"Composite reward weights sum to {total:.2f}, "
                f"which deviates significantly from 1.0.",
                stacklevel=2,
            )

        self.pressure_weight = pressure_weight
        self.queue_weight = queue_weight
        self.flickering_weight = flickering_weight
        self.min_green = min_green
        self.stability_window = stability_window
        self.spillback_weight = spillback_weight
        self.rho_crit = rho_crit

    def compute(
        self,
        observation: dict | np.ndarray,
        last_action: int,
        current_action: int,
        *,
        time_since_phase_change: float | None = None,
        downstream_occupancy=None,
    ) -> float:
        """Compute the composite reward.

        Args:
            observation: Current observation (canonical numpy vector or dict).
            last_action: Action taken at the previous decision step.
            current_action: Action taken at the current decision step.
            time_since_phase_change: Seconds since the last phase change.
                On a switch, the flickering penalty is ``-1`` below
                ``min_green`` and decays linearly to ``0`` over the
                ``stability_window``; when ``None`` (unknown), no penalty
                can be proven and none is applied.
            downstream_occupancy: Occupancies in ``[0, 1]`` of the downstream
                lanes *served by the current green*.  ``None`` (unknown)
                yields no spillback penalty.

        Returns:
            Scalar reward value, typically in ``[-1, 0]``.
        """
        r_pressure = compute_pressure_reward(observation)
        r_queue = compute_queue_reward(observation)
        r_flicker = self._flicker_penalty(
            last_action, current_action, time_since_phase_change
        )
        r_spill = compute_spillback_penalty(
            downstream_occupancy, rho_crit=self.rho_crit
        )

        reward = (
            self.pressure_weight * r_pressure
            + self.queue_weight * r_queue
            + self.flickering_weight * r_flicker
            + self.spillback_weight * r_spill
        )
        return float(reward)

    def _flicker_penalty(
        self,
        last_action: int,
        current_action: int,
        time_since_phase_change: float | None,
    ) -> float:
        """Switch penalty in ``[-1, 0]`` decaying with phase-hold time."""
        if current_action == last_action or time_since_phase_change is None:
            return 0.0
        held_past_min = time_since_phase_change - self.min_green
        if held_past_min < 0.0:
            return -1.0
        if self.stability_window <= 0.0:
            return 0.0
        return -float(max(0.0, 1.0 - held_past_min / self.stability_window))

    def __repr__(self) -> str:
        return (
            f"CompositeReward(pressure={self.pressure_weight}, "
            f"queue={self.queue_weight}, flicker={self.flickering_weight}, "
            f"spillback={self.spillback_weight})"
        )
