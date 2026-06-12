"""Spillback penalty: discourage injecting traffic into a saturated street.

The corridor agents are otherwise blind to downstream capacity: holding a
green toward a block that is already full pushes vehicles into the queue of
the next intersection (spillback) and, at network level, locks the grid.
This module penalises exactly that action — *serving* a movement whose
receiving (downstream) lane is above a critical occupancy.

The math lives in a pure function so the business logic is unit-testable
without SUMO; ``diff_waiting_time_with_spillback`` is the thin sumo_rl
adapter used as a ``reward_fn`` callable during MAPPO training.
"""

from __future__ import annotations

import numpy as np


def compute_spillback_penalty(
    served_downstream_occupancies,
    *,
    rho_crit: float = 0.85,
) -> float:
    """Bounded penalty for feeding green into a saturated downstream lane.

    Only the *served* movements matter: a full street the signal is not
    currently discharging into incurs no penalty (red already protects it).
    The penalty ramps linearly from 0 at ``rho_crit`` to -1 at full
    occupancy::

        P = -clip((max(occ) - rho_crit) / (1 - rho_crit), 0, 1)

    Args:
        served_downstream_occupancies: Occupancies in ``[0, 1]`` of the
            downstream lanes receiving flow from the currently green
            movements.  Empty / ``None`` means nothing is being served
            (e.g. all-red) and yields 0.
        rho_crit: Occupancy above which the receiving lane is considered
            saturated (risk of spillback into the next intersection).

    Returns:
        A scalar penalty in ``[-1.0, 0.0]``.
    """
    if served_downstream_occupancies is None:
        return 0.0
    occ = np.asarray(list(served_downstream_occupancies), dtype=np.float32)
    if occ.size == 0:
        return 0.0
    worst = float(np.clip(occ, 0.0, 1.0).max())
    denom = max(1.0 - rho_crit, 1e-6)
    return -float(np.clip((worst - rho_crit) / denom, 0.0, 1.0))


def served_downstream_occupancies(ts) -> list[float]:
    """Occupancies of the downstream lanes fed by the currently open links.

    A link is *serving* when its signal index shows green (``G``/``g``) in
    the current state string.  Densities reuse sumo_rl's own per-lane
    formula (``get_out_lanes_density``) so reward and observation agree.

    Args:
        ts: A ``sumo_rl`` ``TrafficSignal`` instance (live TraCI connection).

    Returns:
        List of occupancies in ``[0, 1]``, one per served downstream lane.
    """
    state = ts.sumo.trafficlight.getRedYellowGreenState(ts.id)
    links = ts.sumo.trafficlight.getControlledLinks(ts.id)
    out_density = dict(zip(ts.out_lanes, ts.get_out_lanes_density()))

    served: list[float] = []
    for signal_char, link in zip(state, links):
        if signal_char not in ("G", "g") or not link:
            continue
        out_lane = link[0][1]
        if out_lane in out_density:
            served.append(float(out_density[out_lane]))
    return served


def diff_waiting_time_with_spillback(
    ts,
    *,
    spillback_weight: float = 1.0,
    rho_crit: float = 0.85,
) -> float:
    """sumo_rl reward callable: ``diff-waiting-time`` plus spillback penalty.

    Args:
        ts: A ``sumo_rl`` ``TrafficSignal`` instance.
        spillback_weight: Scale of the (bounded) spillback penalty relative
            to the diff-waiting-time base term.
        rho_crit: Critical downstream occupancy (see
            :func:`compute_spillback_penalty`).

    Returns:
        Base reward plus weighted spillback penalty.
    """
    base = ts._diff_waiting_time_reward()
    penalty = compute_spillback_penalty(
        served_downstream_occupancies(ts), rho_crit=rho_crit
    )
    return float(base + spillback_weight * penalty)
