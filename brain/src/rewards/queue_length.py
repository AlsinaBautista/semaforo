"""Queue-length reward function for traffic signal control.

A simple and widely-used reward formulation that penalises the RL agent
proportionally to the total number of vehicles queued at the intersection.
The reward is normalised to ``[-1, 0]`` so it can be combined easily with
other objectives.
"""

from __future__ import annotations

import numpy as np


def compute_queue_reward(
    observation: dict | np.ndarray,
    *,
    max_queue: float = 20.0,
) -> float:
    """Compute a queue-length reward from the observation.

    The reward is defined as::

        R = - Σ queue_i / max_queue

    where ``queue_i`` are the per-direction queue lengths and
    ``max_queue`` is the normalisation constant (total expected maximum).

    When *observation* is our canonical numpy vector the first four
    elements are used as queue lengths (they are already normalised to
    ``[0, 1]``).

    Args:
        observation: Either a 1-D numpy array (canonical env layout) or a
            dict with a ``"queue"`` key.
        max_queue: Maximum total queue length used for normalisation when
            the observation is a dict with raw counts.

    Returns:
        A scalar reward in ``[-1.0, 0.0]``.
    """
    if isinstance(observation, dict):
        queues = np.asarray(observation.get("queue", [0.0]), dtype=np.float32)
        total = float(queues.sum())
        norm = max(max_queue, 1.0)
        reward = -np.clip(total / norm, 0.0, 1.0)
    elif isinstance(observation, np.ndarray):
        # Canonical layout: first 4 values are normalised queues in [0, 1]
        queues = observation[:4]
        total = float(queues.sum())
        # Already normalised individually; sum of 4 normalised values ∈ [0, 4]
        reward = -np.clip(total / 4.0, 0.0, 1.0)
    else:
        reward = 0.0

    return float(reward)
