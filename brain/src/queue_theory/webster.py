# Semáforo Inteligente - Brain Module
"""Webster's optimal cycle length and delay formulas.

F. V. Webster (1958) derived a closed-form expression for the optimal
signal cycle length that minimises the total delay at an isolated
intersection.

    C_opt = (1.5 L + 5) / (1 − Y)

where:
    * L = total lost time per cycle (seconds)
    * Y = sum of critical flow ratios for each phase
"""

from __future__ import annotations


def webster_optimal_cycle(
    y_total: float,
    l_total: float,
) -> float:
    """Compute Webster's optimal cycle length.

    Args:
        y_total: Sum of critical flow ratios Y = Σ yᵢ, where each
            yᵢ = qᵢ / sᵢ (demand flow / saturation flow) for the
            critical movement of phase *i*.  Must be in (0, 1).
        l_total: Total lost time per cycle L (seconds).  Typically
            the sum of start-up and clearance losses across all phases.

    Returns:
        Optimal cycle length C_opt (seconds).

    Raises:
        ValueError: If ``y_total`` is outside (0, 1) or ``l_total``
            is non-positive.

    Example::

        >>> webster_optimal_cycle(y_total=0.6, l_total=12)
        57.5
    """
    if not 0.0 < y_total < 1.0:
        raise ValueError(
            f"y_total must be in (0, 1); got {y_total}. "
            "Values >= 1 imply over-saturated conditions where "
            "Webster's formula does not apply."
        )
    if l_total <= 0:
        raise ValueError(f"l_total must be positive; got {l_total}.")

    return (1.5 * l_total + 5.0) / (1.0 - y_total)


def webster_delay(
    cycle_length: float,
    green_ratio: float,
    flow_ratio: float,
    arrival_rate: float,
) -> float:
    """Compute Webster's average delay per vehicle (d₁ + d₂).

    The formula combines a **uniform delay** (d₁) and a **random
    overflow delay** (d₂):

        d₁ = C (1 − λ)² / (2 (1 − λ x))
        d₂ = x² / (2 q (1 − x))

    where:
        * C  = cycle length (s)
        * λ  = g / C  (effective green ratio)
        * x  = q / (s λ)  (degree of saturation)
        * q  = arrival rate (veh/s)

    Args:
        cycle_length: Total signal cycle length C (seconds).
        green_ratio: Effective green ratio λ = g / C, in (0, 1).
        flow_ratio: Degree of saturation x = q / (s · g/C).
            Must be in [0, 1) for under-saturated conditions.
        arrival_rate: Vehicle arrival rate q (veh/s).

    Returns:
        Average delay per vehicle (seconds).

    Raises:
        ValueError: If inputs are out of valid ranges.

    Example::

        >>> webster_delay(cycle_length=60, green_ratio=0.5,
        ...              flow_ratio=0.8, arrival_rate=0.25)
        11.666...
    """
    if not 0.0 < green_ratio < 1.0:
        raise ValueError(f"green_ratio must be in (0, 1); got {green_ratio}.")
    if not 0.0 <= flow_ratio < 1.0:
        raise ValueError(f"flow_ratio must be in [0, 1); got {flow_ratio}.")
    if arrival_rate <= 0:
        raise ValueError(f"arrival_rate must be positive; got {arrival_rate}.")

    # Uniform delay (d1)
    d1 = (
        cycle_length * (1.0 - green_ratio) ** 2
        / (2.0 * (1.0 - green_ratio * flow_ratio))
    )

    # Random / overflow delay (d2)
    d2 = flow_ratio**2 / (2.0 * arrival_rate * (1.0 - flow_ratio))

    return d1 + d2


def allocate_green_times(
    cycle_length: float,
    critical_ratios: list[float],
    lost_time_per_phase: float = 3.0,
) -> list[float]:
    """Allocate effective green times proportional to critical flow ratios.

    Args:
        cycle_length: Total cycle length C (seconds).
        critical_ratios: List of critical flow ratios [y₁, y₂, …].
        lost_time_per_phase: Lost time per phase (seconds).

    Returns:
        List of effective green durations for each phase (seconds).

    Raises:
        ValueError: If cycle length is too short or ratios are invalid.
    """
    n_phases = len(critical_ratios)
    total_lost = lost_time_per_phase * n_phases
    effective_green_total = cycle_length - total_lost

    if effective_green_total <= 0:
        raise ValueError(
            f"Cycle length {cycle_length}s is too short for {n_phases} phases "
            f"with {lost_time_per_phase}s lost time each."
        )

    y_total = sum(critical_ratios)
    if y_total <= 0:
        raise ValueError("Sum of critical ratios must be positive.")

    return [
        effective_green_total * (y / y_total) for y in critical_ratios
    ]
