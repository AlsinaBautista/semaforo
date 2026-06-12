# Semáforo Inteligente - Brain Module
"""Unit tests for the M/D/1 queueing model and Webster's formulas.

Tests validate the mathematical correctness of:
* ``MD1Queue`` — utilisation, average queue length, average delay.
* ``webster_optimal_cycle`` — optimal cycle calculation.
* ``webster_delay`` — combined uniform + overflow delay.
* ``allocate_green_times`` — proportional green-time allocation.
"""

from __future__ import annotations

import pytest

from src.queue_theory.md1_model import MD1Queue
from src.queue_theory.webster import (
    allocate_green_times,
    webster_delay,
    webster_optimal_cycle,
)


# =====================================================================
# M/D/1 Queue
# =====================================================================


class TestMD1Queue:
    """Tests for :class:`MD1Queue`."""

    def test_utilization(self) -> None:
        """ρ = λ / μ."""
        q = MD1Queue(arrival_rate=0.3, service_rate=0.5)
        assert q.utilization() == pytest.approx(0.6)

    def test_avg_queue_length(self) -> None:
        """Lq = ρ² / (2(1−ρ)) for M/D/1."""
        q = MD1Queue(arrival_rate=0.3, service_rate=0.5)
        rho = 0.6
        expected = rho**2 / (2 * (1 - rho))  # 0.36 / 0.8 = 0.45
        assert q.avg_queue_length() == pytest.approx(expected)

    def test_avg_delay(self) -> None:
        """Wq = ρ / (2μ(1−ρ))."""
        q = MD1Queue(arrival_rate=0.3, service_rate=0.5)
        rho = 0.6
        expected = rho / (2 * 0.5 * (1 - rho))  # 0.6 / 0.4 = 1.5
        assert q.avg_delay() == pytest.approx(expected)

    def test_avg_system_time(self) -> None:
        """W = Wq + 1/μ."""
        q = MD1Queue(arrival_rate=0.3, service_rate=0.5)
        expected = q.avg_delay() + 1.0 / 0.5
        assert q.avg_system_time() == pytest.approx(expected)

    def test_avg_vehicles_in_system(self) -> None:
        """L = Lq + ρ."""
        q = MD1Queue(arrival_rate=0.3, service_rate=0.5)
        expected = q.avg_queue_length() + q.utilization()
        assert q.avg_vehicles_in_system() == pytest.approx(expected)

    def test_unstable_raises(self) -> None:
        """λ >= μ should raise ValueError."""
        with pytest.raises(ValueError, match="unstable"):
            MD1Queue(arrival_rate=0.5, service_rate=0.5)

    def test_negative_rate_raises(self) -> None:
        """Non-positive rates should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            MD1Queue(arrival_rate=-0.1, service_rate=0.5)

    def test_repr(self) -> None:
        """String representation includes key parameters."""
        q = MD1Queue(arrival_rate=0.3, service_rate=0.5)
        r = repr(q)
        assert "0.3" in r
        assert "0.5" in r


# =====================================================================
# Webster's optimal cycle
# =====================================================================


class TestWebsterOptimalCycle:
    """Tests for :func:`webster_optimal_cycle`."""

    def test_known_values(self) -> None:
        """Verify with hand-calculated example: Y=0.6, L=12 → C=57.5."""
        c = webster_optimal_cycle(y_total=0.6, l_total=12)
        assert c == pytest.approx(57.5)

    def test_low_demand(self) -> None:
        """Low Y → longer optimal cycle is not needed → smaller C."""
        c_low = webster_optimal_cycle(y_total=0.2, l_total=10)
        c_high = webster_optimal_cycle(y_total=0.8, l_total=10)
        assert c_low < c_high

    def test_y_out_of_range_raises(self) -> None:
        """Y >= 1 is over-saturated: should raise."""
        with pytest.raises(ValueError):
            webster_optimal_cycle(y_total=1.0, l_total=10)

    def test_negative_lost_time_raises(self) -> None:
        """Non-positive lost time should raise."""
        with pytest.raises(ValueError):
            webster_optimal_cycle(y_total=0.5, l_total=0)


# =====================================================================
# Webster's delay
# =====================================================================


class TestWebsterDelay:
    """Tests for :func:`webster_delay`."""

    def test_positive_delay(self) -> None:
        """Delay should always be positive for valid inputs."""
        d = webster_delay(
            cycle_length=60,
            green_ratio=0.5,
            flow_ratio=0.8,
            arrival_rate=0.25,
        )
        assert d > 0

    def test_low_saturation_less_delay(self) -> None:
        """Lower flow_ratio → less delay."""
        d_low = webster_delay(60, 0.5, 0.3, 0.25)
        d_high = webster_delay(60, 0.5, 0.8, 0.25)
        assert d_low < d_high

    def test_invalid_green_ratio_raises(self) -> None:
        """Green ratio outside (0,1) should raise."""
        with pytest.raises(ValueError):
            webster_delay(60, 0.0, 0.5, 0.25)

    def test_invalid_flow_ratio_raises(self) -> None:
        """Flow ratio >= 1 should raise."""
        with pytest.raises(ValueError):
            webster_delay(60, 0.5, 1.0, 0.25)


# =====================================================================
# Green-time allocation
# =====================================================================


class TestAllocateGreenTimes:
    """Tests for :func:`allocate_green_times`."""

    def test_proportional_split(self) -> None:
        """Green times should sum to effective green = C - total_lost."""
        greens = allocate_green_times(
            cycle_length=90,
            critical_ratios=[0.3, 0.2, 0.1],
            lost_time_per_phase=4.0,
        )
        assert len(greens) == 3
        assert sum(greens) == pytest.approx(90 - 12)

    def test_equal_ratios(self) -> None:
        """Equal ratios → equal green times."""
        greens = allocate_green_times(60, [0.25, 0.25], 5.0)
        assert greens[0] == pytest.approx(greens[1])

    def test_cycle_too_short_raises(self) -> None:
        """Cycle shorter than total lost time should raise."""
        with pytest.raises(ValueError, match="too short"):
            allocate_green_times(10, [0.3, 0.3], 6.0)
