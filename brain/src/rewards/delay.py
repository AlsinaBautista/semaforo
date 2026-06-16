"""Delay reward function for traffic signal control.

Penalises the RL agent based on the total accumulated waiting time
of all vehicles currently at the intersection. This naturally prevents
starvation because a single vehicle waiting for a long time will
eventually generate a massive penalty, forcing the AI to service it.
"""

from __future__ import annotations
import numpy as np

def compute_delay_reward(ts) -> float:
    """Compute reward based on total accumulated waiting time.
    
    Args:
        ts: The sumo_rl TrafficSignal object.
        
    Returns:
        Negative scalar representing total delay (normalised).
    """
    try:
        # get_accumulated_waiting_time_per_lane returns a list/dict of waiting times
        waiting_times = ts.get_accumulated_waiting_time_per_lane()
        if isinstance(waiting_times, dict):
            total_delay = sum(waiting_times.values())
        else:
            total_delay = sum(waiting_times)
            
        # Normalise by a large constant (e.g., 1000 seconds total delay)
        # to keep the reward generally within [-1, 0]
        reward = -np.clip(total_delay / 1000.0, 0.0, 1.0)
        return float(reward)
    except Exception:
        return 0.0
