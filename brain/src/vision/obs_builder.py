"""Observation builder for the RL Brain from real camera feeds.

Takes vehicle counts from multiple cameras (one per direction) and
constructs the **canonical 17-element** observation vector that the trained
MAPPO actor expects. This bridges the gap between YOLO's raw detections and
the Brain's mathematical input.

Observation vector layout — MUST match ``Canonical17ObservationFunction`` in
``src/environments/multi_intersection.py`` (the layout the exported ONNX actor
was traced on); any divergence feeds the policy a permutation:

    [queue_N, queue_E, queue_S, queue_W,        # 0-3   (normalised [0,1], /20 veh)
     density_N, density_E, density_S, density_W, # 4-7   (normalised [0,1])
     phase_0, phase_1, phase_2, phase_3,         # 8-11  (one-hot, 4 slots)
     phase_elapsed_normalised,                    # 12    (elapsed / (max_green+yellow))
     down_N, down_E, down_S, down_W]             # 13-16 (downstream occ — 0 from cameras)

Notes vs. the training layout:
- Direction bucket order is **[N, E, S, W]** (incLanes order; see the canonical
  function's docstring). ``build`` fills N/E/S/W into slots 0-3 and 4-7 in that order.
- The phase one-hot has **4 reserved slots**. A standard 2-phase intersection only
  ever lights slot 8 (NS green) or 9 (EW green); slots 10-11 exist for signals with
  >2 green phases and stay 0 here.
- ``queue_imbalance`` is NOT part of the canonical contract and is intentionally absent.
- Downstream occupancies are not measurable from a single camera and stay 0; this is
  out-of-distribution for an isolated intersection vs. the corridor it was trained on,
  but is the faithful value (no downstream neighbour exists).
"""

from __future__ import annotations

import time

import numpy as np


# Maximum expected vehicles per camera for normalisation.
# 20 vehicles ≈ a full queue visible from a single traffic camera. Matches the
# canonical queue normalisation (``_aggregate_directions(..., max_val=20.0)``).
_MAX_VEHICLES_PER_CAMERA = 20

# Elapsed-time normalisation window. The canonical observation divides
# ``time_since_last_phase_change`` by ``max_green + yellow_time``; the corridor
# that trained the deployed actor used max_green=60, yellow_time=3 (see
# MultiIntersectionEnv.__init__). Keep this in sync if the policy is retrained
# with different signal timings.
_MAX_GREEN_S = 60
_YELLOW_TIME_S = 3
_PHASE_ELAPSED_WINDOW_S = float(_MAX_GREEN_S + _YELLOW_TIME_S)

# Canonical observation dimensionality.
_OBS_SIZE = 17


class ObservationBuilder:
    """Build the canonical 17-dim observation vector from per-direction counts.

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
        """Construct the canonical 17-element observation vector.

        Args:
            count_north: Vehicles detected by the north-facing camera.
            count_east: Vehicles detected by the east-facing camera.
            count_south: Vehicles detected by the south-facing camera.
            count_west: Vehicles detected by the west-facing camera.

        Returns:
            Normalised ``np.float32`` vector of shape ``(17,)``.
        """
        obs = np.zeros(_OBS_SIZE, dtype=np.float32)

        # --- Queues (positions 0-3, order [N, E, S, W]) ---
        # Normalise vehicle counts to [0, 1] against a full-queue camera view.
        q_n = min(count_north / _MAX_VEHICLES_PER_CAMERA, 1.0)
        q_e = min(count_east / _MAX_VEHICLES_PER_CAMERA, 1.0)
        q_s = min(count_south / _MAX_VEHICLES_PER_CAMERA, 1.0)
        q_w = min(count_west / _MAX_VEHICLES_PER_CAMERA, 1.0)
        obs[0] = q_n
        obs[1] = q_e
        obs[2] = q_s
        obs[3] = q_w

        # --- Densities (positions 4-7) ---
        # A single camera frame can't measure lane occupancy, so we approximate
        # density ≈ normalised queue (same as the legacy builder).
        obs[4] = q_n
        obs[5] = q_e
        obs[6] = q_s
        obs[7] = q_w

        # --- Phase one-hot (positions 8-11, 4 reserved slots) ---
        # Mirror the canonical ``obs[8 + green_phase % 4] = 1.0``. For a 2-phase
        # intersection only slots 8 (NS) and 9 (EW) are ever set.
        obs[8 + (int(self._current_phase) % 4)] = 1.0

        # --- Elapsed time normalised (position 12) ---
        elapsed = time.time() - self._phase_start_time
        obs[12] = np.clip(elapsed / _PHASE_ELAPSED_WINDOW_S, 0.0, 1.0)

        # --- Downstream occupancies (positions 13-16) ---
        # Not measurable from cameras and no downstream neighbour at an isolated
        # intersection; stays 0.
        obs[13:17] = 0.0

        return np.clip(obs, 0.0, 1.0)
