"""Observation builder for the PPO Brain from real camera feeds.

Takes vehicle counts from multiple cameras (one per direction) and
constructs the 16-element observation vector that the trained PPO
model expects. This bridges the gap between YOLO's raw detections
and the Brain's mathematical input.

Observation vector layout (must match SingleIntersectionEnv):
    [queue_N, queue_E, queue_S, queue_W,        # 0-3   (normalised [0,1])
     density_N, density_E, density_S, density_W, # 4-7   (normalised [0,1])
     phase_0_NS, phase_1_EW,                     # 8-9   (one-hot)
     phase_elapsed_normalised,                    # 10    (normalised [0,1])
     down_N, down_E, down_S, down_W,             # 11-14 (set to 0 — no downstream info)
     queue_imbalance]                             # 15    (|NS-EW| / total)
"""

from __future__ import annotations

import time

import numpy as np


# Maximum expected vehicles per camera for normalisation.
# 20 vehicles ≈ a full queue visible from a single traffic camera.
_MAX_VEHICLES_PER_CAMERA = 20


class ObservationBuilder:
    """Build the PPO observation vector from per-direction vehicle counts.

    Maintains internal state for:
    - Current traffic light phase (0 = N-S green, 1 = E-W green).
    - Time elapsed since the last phase change.

    Args:
        initial_phase: Starting phase (0 = N-S, 1 = E-W).
    """

    def __init__(self, initial_phase: int = 0) -> None:
        self._current_phase: int = initial_phase
        self._phase_start_time: float = time.time()

    @property
    def current_phase(self) -> int:
        """Currently active green phase (0 = N-S, 1 = E-W)."""
        return self._current_phase

    def set_phase(self, phase: int) -> None:
        """Update the current phase (called after Brain decides).

        Args:
            phase: New phase index (0 = N-S, 1 = E-W).
        """
        if phase != self._current_phase:
            self._current_phase = phase
            self._phase_start_time = time.time()

    def build(
        self,
        count_north: int,
        count_east: int,
        count_south: int,
        count_west: int,
    ) -> np.ndarray:
        """Construct the 16-element observation vector.

        Args:
            count_north: Vehicles detected by the north-facing camera.
            count_east: Vehicles detected by the east-facing camera.
            count_south: Vehicles detected by the south-facing camera.
            count_west: Vehicles detected by the west-facing camera.

        Returns:
            Normalised ``np.float32`` vector of shape ``(16,)``.
        """
        obs = np.zeros(16, dtype=np.float32)

        # --- Queues (positions 0-3) ---
        # Normalise vehicle counts to [0, 1]
        q_n = min(count_north / _MAX_VEHICLES_PER_CAMERA, 1.0)
        q_e = min(count_east / _MAX_VEHICLES_PER_CAMERA, 1.0)
        q_s = min(count_south / _MAX_VEHICLES_PER_CAMERA, 1.0)
        q_w = min(count_west / _MAX_VEHICLES_PER_CAMERA, 1.0)
        obs[0] = q_n
        obs[1] = q_e
        obs[2] = q_s
        obs[3] = q_w

        # --- Densities (positions 4-7) ---
        # In real cameras we approximate density ≈ queue
        # (we can't measure lane occupancy from a single frame)
        obs[4] = q_n
        obs[5] = q_e
        obs[6] = q_s
        obs[7] = q_w

        # --- Phase one-hot (positions 8-9) ---
        obs[8] = 1.0 if self._current_phase == 0 else 0.0
        obs[9] = 1.0 if self._current_phase == 1 else 0.0

        # --- Elapsed time normalised (position 10) ---
        elapsed = time.time() - self._phase_start_time
        obs[10] = np.clip(elapsed / 120.0, 0.0, 1.0)

        # --- Downstream occupancies (positions 11-14) ---
        # Not measurable from cameras; set to 0
        obs[11:15] = 0.0

        # --- Queue imbalance (position 15) ---
        q_ns = q_n + q_s
        q_ew = q_e + q_w
        total = q_ns + q_ew
        if total > 0.01:
            obs[15] = np.clip(abs(q_ns - q_ew) / total, 0.0, 1.0)
        else:
            obs[15] = 0.0

        return obs
