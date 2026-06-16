"""Smart delay reward for adaptive traffic signal control.

Combines two objectives:
1. **Delay penalty**: Penalises accumulated waiting time across all vehicles.
   This is the primary learning signal — the AI must minimise total delay.
2. **Wasted green penalty**: Penalises giving green to an empty direction
   while the other direction has queued vehicles. This directly teaches the
   AI to never switch to an empty street when there is traffic elsewhere.

The reward naturally handles the user's requirements:
- Keep green on the congested avenue as long as it has traffic.
- Only switch when vehicles appear on the other direction AND have waited.
- When both directions are empty, no penalty (AI can do whatever).

Phase-to-direction mapping for the single intersection:
- Green phase 0 → North-South (north_in + south_in)
- Green phase 1 → East-West  (east_in + west_in)

Observation layout used:
- obs[0:4] = queue lengths per direction [N, E, S, W], normalised [0, 1]
"""

from __future__ import annotations
import numpy as np


def compute_smart_delay_reward(
    ts,
    obs: np.ndarray,
) -> float:
    """Compute combined delay + wasted-green reward.

    Args:
        ts: The sumo_rl ``TrafficSignal`` object.
        obs: The normalised observation vector (17 elements).

    Returns:
        Scalar reward in approximately ``[-1.0, 0.0]``.
    """
    # ------------------------------------------------------------------
    # Component 1: Delay penalty  (weight: 0.6)
    # Penalises total accumulated waiting time across all lanes.
    # ------------------------------------------------------------------
    try:
        wait_times: list[float] = ts.get_accumulated_waiting_time_per_lane()
        total_delay = sum(wait_times)
        r_delay = -np.clip(total_delay / 500.0, 0.0, 1.0)
    except Exception:
        r_delay = 0.0

    # ------------------------------------------------------------------
    # Component 2: Wasted green penalty  (weight: 0.4)
    # Fires when the current green phase serves a direction with no queue
    # while the other direction has waiting vehicles.
    # ------------------------------------------------------------------
    # Queue per direction from observation vector
    q_north = float(obs[0])
    q_east  = float(obs[1])
    q_south = float(obs[2])
    q_west  = float(obs[3])

    # Aggregate by phase direction
    q_ns = q_north + q_south   # Queue on North-South axis
    q_ew = q_east + q_west     # Queue on East-West axis

    # Determine which direction currently has green
    # Phase 0 = N-S green, Phase 1 = E-W green
    green_phase = ts.green_phase

    if green_phase == 0:
        green_queue = q_ns
        red_queue = q_ew
    else:
        green_queue = q_ew
        red_queue = q_ns

    # Compute wasted green penalty
    # Case: green direction is empty BUT red direction has traffic → BAD
    # The penalty scales with how much traffic is stuck at red
    if green_queue < 0.02 and red_queue > 0.05:
        # Strong penalty: proportional to how much traffic is wasting time
        r_wasted = -np.clip(red_queue / 1.0, 0.0, 1.0)
    else:
        r_wasted = 0.0

    # ------------------------------------------------------------------
    # Combined reward
    # ------------------------------------------------------------------
    reward = 0.6 * r_delay + 0.4 * r_wasted
    return float(reward)
