# Semáforo Inteligente - Brain Module
"""Unit tests for reward functions.

Tests cover:
* ``compute_pressure_reward`` — dict and flat-array inputs.
* ``compute_queue_reward`` — dict and flat-array inputs.
* ``CompositeReward`` — weighted combination and flickering penalty.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.rewards.composite import CompositeReward
from src.rewards.pressure import compute_pressure_reward
from src.rewards.queue_length import compute_queue_reward


# =====================================================================
# Pressure reward
# =====================================================================


class TestPressureReward:
    """Tests for :func:`compute_pressure_reward`."""

    def test_dict_input_positive_pressure(self) -> None:
        """Incoming > outgoing → negative reward."""
        obs = {"incoming": np.array([5, 3]), "outgoing": np.array([2, 1])}
        reward = compute_pressure_reward(obs)
        assert reward == pytest.approx(-5.0)

    def test_dict_input_negative_pressure(self) -> None:
        """Outgoing > incoming → positive reward."""
        obs = {"incoming": np.array([1, 1]), "outgoing": np.array([3, 4])}
        reward = compute_pressure_reward(obs)
        assert reward == pytest.approx(5.0)

    def test_dict_input_zero_pressure(self) -> None:
        """Equal counts → zero reward."""
        obs = {"incoming": np.array([3, 3]), "outgoing": np.array([3, 3])}
        reward = compute_pressure_reward(obs)
        assert reward == pytest.approx(0.0)

    def test_flat_array_input(self) -> None:
        """Flat array split in half: first half incoming, second outgoing."""
        obs = np.array([4, 3, 2, 1])  # incoming=[4,3], outgoing=[2,1]
        reward = compute_pressure_reward(obs)
        assert reward == pytest.approx(-4.0)


# =====================================================================
# Queue-length reward
# =====================================================================


class TestQueueReward:
    """Tests for :func:`compute_queue_reward`."""

    def test_dict_input(self) -> None:
        """Dict with queue_lengths key."""
        obs = {"queue_lengths": np.array([3, 0, 5, 2])}
        reward = compute_queue_reward(obs)
        assert reward == pytest.approx(-10.0)

    def test_flat_array(self) -> None:
        """Flat array interpreted as queue lengths directly."""
        obs = np.array([1.0, 2.0, 3.0])
        reward = compute_queue_reward(obs)
        assert reward == pytest.approx(-6.0)

    def test_all_zeros(self) -> None:
        """No queues → zero reward."""
        obs = np.array([0.0, 0.0, 0.0])
        reward = compute_queue_reward(obs)
        assert reward == pytest.approx(0.0)


# =====================================================================
# Composite reward
# =====================================================================


class TestCompositeReward:
    """Tests for :class:`CompositeReward`."""

    def test_no_flicker(self) -> None:
        """Same action → no flicker penalty."""
        cr = CompositeReward(weight_pressure=1.0, weight_queue=0.0, weight_flicker=-1.0)
        obs = np.array([5, 3, 2, 1])  # pressure input (flat)
        r = cr(obs, current_action=0, previous_action=0)
        # pressure = -(5+3 - 2-1) = -5; flicker = 0
        assert r == pytest.approx(-5.0)

    def test_with_flicker(self) -> None:
        """Phase change → flicker penalty applied."""
        cr = CompositeReward(weight_pressure=1.0, weight_queue=0.0, weight_flicker=-2.0)
        obs = np.array([5, 3, 2, 1])
        r = cr(obs, current_action=1, previous_action=0)
        # pressure = -5; flicker = 1 * (-2) = -2
        assert r == pytest.approx(-7.0)

    def test_reset_clears_history(self) -> None:
        """After reset, no previous action exists."""
        cr = CompositeReward(weight_pressure=0.0, weight_queue=0.0, weight_flicker=-10.0)
        cr(np.array([0, 0]), current_action=0)
        cr.reset()
        # First call after reset: no previous action → no flicker.
        r = cr(np.array([0, 0]), current_action=1)
        assert r == pytest.approx(0.0)
