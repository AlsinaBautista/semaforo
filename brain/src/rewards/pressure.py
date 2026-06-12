"""Pressure-based reward function for traffic signal control.

Implements the *Max Pressure* control concept introduced by Varaiya (2013).
The pressure at an intersection is defined as the difference between the
number of vehicles approaching (upstream queue) and departing (downstream
capacity/flow).  A negative-pressure reward encourages the RL agent to
activate the phase that relieves the highest pressure, which is a known
throughput-optimal policy for signalised networks.

Reference:
    Varaiya, P. (2013). *Max pressure control of a network of signalized
    intersections*. Transportation Research Part C, 36, 177-195.
"""

from __future__ import annotations

import numpy as np


def compute_pressure_reward(
    observation: dict | np.ndarray,
    *,
    incoming_key: str = "queue",
    outgoing_key: str = "flow",
    max_queue: float = 20.0,
) -> float:
    """Compute a pressure-based reward from the observation.

    When *observation* is a **numpy array** with our canonical layout the
    first four elements are treated as incoming queue lengths (normalised)
    and the next four as densities (proxy for downstream occupancy).

    When *observation* is a **dict** the function looks up
    ``incoming_key`` and ``outgoing_key`` entries.

    The reward is defined as::

        R = -(Σ incoming_i  -  Σ outgoing_i)

    and is then clipped to the ``[-1, 0]`` range so that the optimal
    (zero-pressure) state yields a reward of **0**.

    Args:
        observation: Either a 1-D numpy array (canonical env observation)
            or a dict with queue / flow arrays.
        incoming_key: Dict key for upstream (incoming) vehicle counts.
        outgoing_key: Dict key for downstream (outgoing) vehicle counts.
        max_queue: Maximum expected total queue length, used for
            normalisation when working with raw (un-normalised) values.

    Returns:
        A scalar reward in ``[-1.0, 0.0]``.
    """
    if isinstance(observation, dict):
        incoming = np.asarray(observation.get(incoming_key, [0.0]), dtype=np.float32)
        outgoing = np.asarray(observation.get(outgoing_key, [0.0]), dtype=np.float32)
    elif isinstance(observation, np.ndarray):
        # Canonical layout: [queue_N, queue_E, queue_S, queue_W, density_N, ...]
        # (bucket order [N, E, S, W] = incLanes order; see multi_intersection.py)
        incoming = observation[:4].copy()
        outgoing = observation[4:8].copy()
    else:
        return 0.0

    pressure = float(incoming.sum() - outgoing.sum())

    # Normalise so that the worst-case pressure maps to -1
    norm = max(max_queue, 1.0)
    reward = -np.clip(pressure / norm, 0.0, 1.0)

    return float(reward)
