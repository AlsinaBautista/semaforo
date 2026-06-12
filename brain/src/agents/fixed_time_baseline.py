# Semáforo Inteligente - Brain Module
"""Fixed-time baseline signal controller.

This agent cycles through phases with configurable durations and serves
as the classical benchmark against which RL agents are compared.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class FixedTimeBaseline:
    """Deterministic fixed-cycle traffic-signal controller.

    The controller simply advances through a pre-defined sequence of
    phases, each held for a fixed duration.  It ignores the observation
    entirely, making it a pure time-driven baseline.

    Args:
        phase_durations: Ordered list of green-phase durations (seconds).
            Defaults to ``[30, 30]`` (NS green 30 s, EW green 30 s).
        delta_time: Decision interval of the environment (seconds).
            Must match the environment's ``delta_time`` so that the
            controller counts elapsed time correctly.

    Example::

        baseline = FixedTimeBaseline(phase_durations=[30, 30], delta_time=5)
        action = baseline.decide(obs)  # returns 0 or 1
    """

    def __init__(
        self,
        phase_durations: list[int] | None = None,
        delta_time: int = 5,
    ) -> None:
        self.phase_durations = phase_durations or [30, 30]
        self.delta_time = delta_time

        self._current_phase: int = 0
        self._elapsed: int = 0  # seconds elapsed in current phase

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, obs: Any = None) -> int:
        """Return the current phase action and advance the internal clock.

        The observation *obs* is accepted for interface compatibility but
        is not used.

        Args:
            obs: Environment observation (ignored).

        Returns:
            Discrete action index for the current phase.
        """
        action = self._current_phase
        self._elapsed += self.delta_time

        if self._elapsed >= self.phase_durations[self._current_phase]:
            self._current_phase = (self._current_phase + 1) % len(
                self.phase_durations
            )
            self._elapsed = 0

        return action

    def reset(self) -> None:
        """Reset the controller to phase 0 with zero elapsed time."""
        self._current_phase = 0
        self._elapsed = 0

    @property
    def num_phases(self) -> int:
        """Total number of phases in the cycle."""
        return len(self.phase_durations)

    @property
    def cycle_length(self) -> int:
        """Total cycle length in seconds."""
        return sum(self.phase_durations)
