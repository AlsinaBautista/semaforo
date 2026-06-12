# Semáforo Inteligente - Brain Module
"""Green-wave offset calculator for arterial corridors.

A *green wave* synchronises traffic signals along a corridor so that a
platoon of vehicles travelling at a target speed can pass through
consecutive intersections without stopping.  The required offset between
signal *i* and signal *i+1* is simply:

    offset_i = distance_i / speed

where ``distance_i`` is the spacing between intersections and ``speed``
is the target progression speed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Intersection:
    """Minimal intersection descriptor for green-wave planning.

    Attributes:
        id: Unique identifier.
        position_m: Position along the corridor (metres from origin).
        cycle_length: Signal cycle length (seconds).
    """

    id: str
    position_m: float
    cycle_length: float = 60.0


def compute_offsets(
    corridor: list[Intersection],
    target_speed_kmh: float = 50.0,
) -> list[float]:
    """Calculate green-wave offsets for a linear corridor.

    Offsets are computed relative to the **first** intersection (offset
    = 0).  Each subsequent offset is the travel time from the first
    intersection at the target speed.

    Args:
        corridor: Ordered list of :class:`Intersection` objects along
            the corridor (sorted by position from upstream to
            downstream).
        target_speed_kmh: Target platoon speed in km/h.

    Returns:
        List of offset times (seconds), one per intersection.  The
        first element is always ``0.0``.

    Raises:
        ValueError: If fewer than two intersections are provided or
            the target speed is non-positive.

    Example::

        >>> corridor = [
        ...     Intersection("A", 0),
        ...     Intersection("B", 500),
        ...     Intersection("C", 1200),
        ... ]
        >>> compute_offsets(corridor, target_speed_kmh=50)
        [0.0, 36.0, 86.4]
    """
    if len(corridor) < 2:
        raise ValueError("At least two intersections are required.")
    if target_speed_kmh <= 0:
        raise ValueError("target_speed_kmh must be positive.")

    speed_ms = target_speed_kmh / 3.6  # convert km/h → m/s
    origin = corridor[0].position_m

    offsets: list[float] = []
    for intersection in corridor:
        distance = intersection.position_m - origin
        offset = distance / speed_ms
        offsets.append(round(offset, 2))

    return offsets


def compute_bidirectional_offsets(
    corridor: list[Intersection],
    target_speed_kmh: float = 50.0,
) -> tuple[list[float], list[float]]:
    """Compute offsets for both directions of travel.

    Args:
        corridor: Ordered list of intersections (upstream → downstream).
        target_speed_kmh: Target speed in km/h.

    Returns:
        A tuple ``(forward_offsets, reverse_offsets)``.
    """
    forward = compute_offsets(corridor, target_speed_kmh)
    reversed_corridor = list(reversed(corridor))
    reverse = compute_offsets(reversed_corridor, target_speed_kmh)
    return forward, reverse
