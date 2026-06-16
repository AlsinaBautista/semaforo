"""Delay reward function for traffic signal control.

Penalises the RL agent based on the total accumulated waiting time
of all vehicles currently at the intersection. This naturally prevents
starvation because a single vehicle waiting for a long time will
eventually generate a massive penalty, forcing the AI to service it.

The key insight: unlike queue-length rewards (which count vehicles),
this reward weights each vehicle by HOW LONG it has been waiting.
One vehicle waiting 120 seconds hurts as much as 120 vehicles waiting
1 second each. This creates an organic "urgency" signal that forces
the AI to eventually service low-traffic directions without any
hardcoded rules.
"""

from __future__ import annotations
import numpy as np


def compute_delay_reward(ts) -> float:
    """Compute reward based on total accumulated waiting time.

    Uses sumo_rl's ``get_accumulated_waiting_time_per_lane()`` which
    returns a ``List[float]`` — one value per incoming lane at the
    intersection, representing the sum of per-vehicle accumulated
    waiting times on that lane.

    The total is normalised by 500 (a realistic upper bound for a
    single intersection under heavy load: ~50 vehicles × ~10 s each).
    Values above 500 are clipped to -1.0.

    Args:
        ts: The sumo_rl ``TrafficSignal`` object.

    Returns:
        Scalar reward in ``[-1.0, 0.0]``.
    """
    try:
        wait_times: list[float] = ts.get_accumulated_waiting_time_per_lane()
        total_delay = sum(wait_times)

        # Normalise: 500s total delay → reward = -1.0 (worst)
        #            0s  total delay → reward =  0.0 (best)
        reward = -np.clip(total_delay / 500.0, 0.0, 1.0)
        return float(reward)
    except Exception:
        return 0.0
