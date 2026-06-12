# Semáforo Inteligente - Brain Module
"""Traffic-simulation metrics helpers.

Functions in this module consume raw data from SUMO (e.g. vehicle trip
info, queue lengths over time) and produce summary statistics for
evaluation dashboards and reports.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def avg_waiting_time(
    waiting_times: list[float] | np.ndarray,
) -> float:
    """Compute the mean waiting time across vehicles.

    Args:
        waiting_times: Per-vehicle waiting times (seconds).

    Returns:
        Mean waiting time (seconds).  Returns ``0.0`` for empty input.
    """
    if len(waiting_times) == 0:
        return 0.0
    return float(np.mean(waiting_times))


def throughput(
    departed: int,
    arrived: int,
    sim_time: float,
) -> dict[str, float]:
    """Compute throughput metrics.

    Args:
        departed: Number of vehicles that entered the network.
        arrived: Number of vehicles that reached their destination.
        sim_time: Total simulation time (seconds).

    Returns:
        Dictionary with ``throughput_veh_per_hour``,
        ``completion_rate``, and ``sim_time_s``.
    """
    hours = sim_time / 3600.0 if sim_time > 0 else 1.0
    return {
        "throughput_veh_per_hour": arrived / hours,
        "completion_rate": arrived / departed if departed > 0 else 0.0,
        "sim_time_s": sim_time,
    }


def queue_stats(
    queue_lengths: list[list[float]] | np.ndarray,
) -> dict[str, Any]:
    """Compute summary statistics for per-step queue-length data.

    Args:
        queue_lengths: 2-D array of shape ``(timesteps, lanes)``
            containing the queue length for each lane at each
            simulation step.

    Returns:
        Dictionary with ``mean_total_queue``, ``max_total_queue``,
        ``mean_per_lane``, and ``std_per_lane``.
    """
    arr = np.asarray(queue_lengths, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    total_per_step = arr.sum(axis=1)

    return {
        "mean_total_queue": float(np.mean(total_per_step)),
        "max_total_queue": float(np.max(total_per_step)),
        "mean_per_lane": np.mean(arr, axis=0).tolist(),
        "std_per_lane": np.std(arr, axis=0).tolist(),
    }


def summarise_trip_info(trip_info_df: pd.DataFrame) -> dict[str, float]:
    """Summarise a SUMO ``tripinfo`` XML output loaded as a DataFrame.

    Expected columns: ``waitingTime``, ``duration``, ``timeLoss``.

    Args:
        trip_info_df: DataFrame parsed from SUMO's ``--tripinfo-output``.

    Returns:
        Dictionary with ``mean_waiting``, ``mean_duration``,
        ``mean_time_loss``, and ``total_vehicles``.
    """
    return {
        "mean_waiting": float(trip_info_df["waitingTime"].mean()),
        "mean_duration": float(trip_info_df["duration"].mean()),
        "mean_time_loss": float(trip_info_df["timeLoss"].mean()),
        "total_vehicles": len(trip_info_df),
    }
