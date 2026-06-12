# Semáforo Inteligente - Brain Module
"""Matplotlib plotting helpers for training analysis and evaluation.

All functions return the ``(fig, ax)`` tuple so callers can further
customise the plots before saving or displaying.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(
    rewards: Sequence[float],
    window: int = 50,
    title: str = "Training Reward Curve",
    xlabel: str = "Episode",
    ylabel: str = "Reward",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot per-episode rewards with a rolling-mean overlay.

    Args:
        rewards: Sequence of per-episode total rewards.
        window: Rolling-average window size.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        ``(fig, ax)`` Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, label="Raw")

    if len(rewards) >= window:
        rolling = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            episodes[window - 1 :],
            rolling,
            linewidth=2,
            label=f"Rolling mean ({window})",
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_queue_comparison(
    rl_queues: Sequence[float],
    fixed_queues: Sequence[float],
    title: str = "Queue Length: RL vs Fixed-Time",
    xlabel: str = "Simulation Step",
    ylabel: str = "Total Queue Length",
) -> tuple[plt.Figure, plt.Axes]:
    """Compare total queue lengths between an RL agent and a fixed-time baseline.

    Args:
        rl_queues: Per-step total queue lengths under the RL policy.
        fixed_queues: Per-step total queue lengths under fixed-time control.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        ``(fig, ax)`` Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(rl_queues, label="RL Agent", linewidth=1.5)
    ax.plot(fixed_queues, label="Fixed-Time", linewidth=1.5, linestyle="--")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_phase_diagram(
    phases: Sequence[int],
    delta_time: int = 5,
    title: str = "Signal Phase Diagram",
    xlabel: str = "Time (s)",
    ylabel: str = "Phase",
) -> tuple[plt.Figure, plt.Axes]:
    """Visualise the signal-phase sequence over time as a step plot.

    Args:
        phases: Sequence of discrete phase indices over time.
        delta_time: Seconds between consecutive phase observations.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        ``(fig, ax)`` Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(12, 3))

    times = np.arange(len(phases)) * delta_time
    ax.step(times, phases, where="post", linewidth=1.5)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yticks(sorted(set(phases)))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig, ax


def save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 150,
) -> None:
    """Save a Matplotlib figure to disk.

    Args:
        fig: Figure object to save.
        path: Output file path (e.g. ``"plots/training.png"``).
        dpi: Resolution in dots per inch.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
