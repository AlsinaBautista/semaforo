"""Business-logic tests: gridlock (spillback) and 1-frame visual noise (flicker).

These tests validate the *deterministic reward landscape* that shapes the
MAPPO policy, not a trained checkpoint (stochastic, out of unit-test scope):
if holding green into a saturated downstream block scores strictly worse
than cutting it, and switching on a one-step observation blip scores
strictly worse than holding, gradient ascent on this reward pushes the
model toward cutting the green in gridlock and away from erratic switching.

Scenario 1 — Gridlock: the receiving block ("la cuadra siguiente") is at
100 % capacity while the signal holds green toward it.

Scenario 2 — Visual noise: a single-frame YOLO ghost/dropout perturbs the
queue observation for one decision step right after a legal phase change.
"""

import numpy as np
import pytest

from src.rewards.composite import CompositeReward
from src.rewards.pressure import compute_pressure_reward
from src.rewards.spillback import (
    compute_spillback_penalty,
    diff_waiting_time_with_spillback,
    served_downstream_occupancies,
)

# Canonical single-intersection observation layout (17 dims).
_OBS_SIZE = 17
_PHASE_NS, _PHASE_EW = 0, 2


def make_obs(queues, densities, phase, elapsed=0.5, downstream=(0.0,) * 4):
    """Build a canonical 17-dim observation vector."""
    obs = np.zeros(_OBS_SIZE, dtype=np.float32)
    obs[0:4] = queues
    obs[4:8] = densities
    obs[8 + phase] = 1.0
    obs[12] = elapsed
    obs[13:17] = downstream
    return obs


# --------------------------------------------------------------------------- #
# Fake sumo_rl TrafficSignal (no SUMO needed)
# --------------------------------------------------------------------------- #
class _FakeTrafficlight:
    def __init__(self, state, links):
        self._state = state
        self._links = links

    def getRedYellowGreenState(self, _id):
        return self._state

    def getControlledLinks(self, _id):
        return self._links


class _FakeSumo:
    def __init__(self, state, links):
        self.trafficlight = _FakeTrafficlight(state, links)


class FakeTrafficSignal:
    """Minimal stand-in exposing the sumo_rl surface the adapter touches."""

    id = "t0"

    def __init__(self, *, state, links, out_lanes, out_densities, base_reward=0.0):
        self.sumo = _FakeSumo(state, links)
        self.out_lanes = out_lanes
        self._out_densities = out_densities
        self._base_reward = base_reward

    def get_out_lanes_density(self):
        return list(self._out_densities)

    def _diff_waiting_time_reward(self):
        return self._base_reward


def _arterial_signal(state, *, arterial_occ, cross_occ):
    """2-link signal: index 0 feeds the arterial out-lane, index 1 the cross."""
    return FakeTrafficSignal(
        state=state,
        links=[
            [("ew_in_0", "arterial_out", "via0")],
            [("ns_in_0", "cross_out", "via1")],
        ],
        out_lanes=["arterial_out", "cross_out"],
        out_densities=[arterial_occ, cross_occ],
    )


# --------------------------------------------------------------------------- #
# Spillback penalty math
# --------------------------------------------------------------------------- #
def test_spillback_penalty_bounds():
    assert compute_spillback_penalty(None) == 0.0
    assert compute_spillback_penalty([]) == 0.0
    assert compute_spillback_penalty([0.0]) == 0.0
    # At the critical threshold the penalty just starts: still ~0
    # (float32 cast of the occupancy gives sub-1e-6 residue).
    assert compute_spillback_penalty([0.85], rho_crit=0.85) == pytest.approx(0.0, abs=1e-6)
    # Full downstream -> maximum penalty.
    assert compute_spillback_penalty([1.0]) == pytest.approx(-1.0)


def test_spillback_penalty_monotonic_in_occupancy():
    occs = np.linspace(0.85, 1.0, 8)
    penalties = [compute_spillback_penalty([o]) for o in occs]
    assert all(b <= a for a, b in zip(penalties, penalties[1:]))
    assert all(-1.0 <= p <= 0.0 for p in penalties)


def test_spillback_only_penalises_served_lanes():
    # Saturated arterial behind a RED is already protected: no penalty.
    red_arterial = _arterial_signal("rG", arterial_occ=1.0, cross_occ=0.1)
    assert served_downstream_occupancies(red_arterial) == [0.1]
    assert diff_waiting_time_with_spillback(red_arterial) == pytest.approx(0.0)

    # Same saturation, but the green is feeding it: full penalty.
    green_arterial = _arterial_signal("Gr", arterial_occ=1.0, cross_occ=0.1)
    assert served_downstream_occupancies(green_arterial) == [1.0]
    assert diff_waiting_time_with_spillback(green_arterial) == pytest.approx(-1.0)


# --------------------------------------------------------------------------- #
# Scenario 1 — Gridlock: next block at 100 % capacity
# --------------------------------------------------------------------------- #
def test_gridlock_model_cuts_the_green():
    """Holding green into a 100 %-occupied block must lose to cutting it."""
    cr = CompositeReward()
    # EW green, moderate own queues, EW receiving block completely full.
    obs = make_obs(
        queues=[0.2, 0.2, 0.5, 0.5],
        densities=[0.4, 0.4, 0.6, 0.6],
        phase=_PHASE_EW,
        downstream=[0.1, 0.1, 1.0, 1.0],
    )
    mature = 30.0  # held long enough that no flicker penalty applies

    r_hold = cr.compute(
        obs, _PHASE_EW, _PHASE_EW,
        time_since_phase_change=mature,
        downstream_occupancy=[1.0, 1.0],  # green keeps feeding the full block
    )
    r_cut = cr.compute(
        obs, _PHASE_EW, _PHASE_NS,
        time_since_phase_change=mature,
        downstream_occupancy=[0.1, 0.1],  # cross movement discharges freely
    )

    assert r_cut > r_hold, (
        f"gridlock: cutting the green ({r_cut:.3f}) must strictly beat "
        f"holding it ({r_hold:.3f})"
    )
    # Greedy decision over the reward landscape cuts the green.
    decision = max([(r_hold, "hold"), (r_cut, "cut")])[1]
    assert decision == "cut"


def test_total_gridlock_no_longer_scores_as_optimum():
    """Everything saturated (in ≈ out) used to score R = 0 via pressure."""
    gridlock = make_obs(
        queues=[1.0] * 4,
        densities=[1.0] * 4,
        phase=_PHASE_EW,
        downstream=[1.0] * 4,
    )
    # The documented hole: pressure alone is blind to total gridlock.
    assert compute_pressure_reward(gridlock) == pytest.approx(0.0)

    cr = CompositeReward()
    r = cr.compute(
        gridlock, _PHASE_EW, _PHASE_EW,
        time_since_phase_change=30.0,
        downstream_occupancy=[1.0, 1.0],
    )
    # Composite with downstream visibility marks it as strictly bad.
    assert r <= -(cr.queue_weight + cr.spillback_weight) + 1e-6
    assert r < 0.0


# --------------------------------------------------------------------------- #
# Scenario 2 — Visual noise: 1-frame observation blip
# --------------------------------------------------------------------------- #
def test_one_frame_blip_does_not_flip_the_decision():
    """Right after a legal switch, a 1-step queue blip must not pay to switch."""
    cr = CompositeReward()
    elapsed = cr.min_green  # just reached legality — the erratic window
    for blip in np.linspace(0.0, 0.3, 7):
        obs = make_obs(
            queues=[0.2 + blip, 0.2 + blip, 0.2, 0.2],  # NS ghost-inflated
            densities=[0.3] * 4,
            phase=_PHASE_EW,
            downstream=[0.1] * 4,
        )
        r_hold = cr.compute(
            obs, _PHASE_EW, _PHASE_EW,
            time_since_phase_change=elapsed,
            downstream_occupancy=[0.1, 0.1],
        )
        r_switch = cr.compute(
            obs, _PHASE_EW, _PHASE_NS,
            time_since_phase_change=elapsed,
            downstream_occupancy=[0.1, 0.1],
        )
        assert r_hold > r_switch, (
            f"blip={blip:.2f}: switching ({r_switch:.3f}) must not beat "
            f"holding ({r_hold:.3f}) immediately after a phase change"
        )


def test_min_period_pingpong_is_no_longer_free():
    """Regression: switching exactly at min_green used to cost nothing."""
    cr = CompositeReward()
    quiet = make_obs(
        queues=[0.0] * 4, densities=[0.0] * 4, phase=_PHASE_EW,
        downstream=[0.0] * 4,
    )
    r_switch = cr.compute(
        quiet, _PHASE_EW, _PHASE_NS,
        time_since_phase_change=cr.min_green,
        downstream_occupancy=[0.0, 0.0],
    )
    r_hold = cr.compute(
        quiet, _PHASE_EW, _PHASE_EW,
        time_since_phase_change=cr.min_green,
        downstream_occupancy=[0.0, 0.0],
    )
    assert r_hold == pytest.approx(0.0)
    assert r_switch == pytest.approx(-cr.flickering_weight)  # full decay penalty


def test_oscillating_policy_accumulates_less_reward_than_holding():
    """A legal 10s ping-pong (the previously free ride) now loses to holding.

    Identical observation stream (with a 1-step blip in the middle); the
    only difference between the two policies is the switching pattern.
    """
    cr = CompositeReward()
    steps = 12
    delta_t = 5.0
    base_obs = make_obs(
        queues=[0.2] * 4, densities=[0.3] * 4, phase=_PHASE_EW,
        downstream=[0.1] * 4,
    )
    blip_obs = make_obs(
        queues=[0.5, 0.5, 0.2, 0.2], densities=[0.3] * 4, phase=_PHASE_EW,
        downstream=[0.1] * 4,
    )
    stream = [blip_obs if i == 6 else base_obs for i in range(steps)]
    down = [0.1, 0.1]

    def rollout(actions):
        total, last, elapsed = 0.0, actions[0], 30.0
        for obs, act in zip(stream, actions):
            total += cr.compute(
                obs, last, act,
                time_since_phase_change=elapsed,
                downstream_occupancy=down,
            )
            elapsed = delta_t if act != last else elapsed + delta_t
            last = act
        return total

    hold = rollout([_PHASE_EW] * steps)
    # Period-10s ping-pong: switch every 2nd decision (every 10 s).
    oscillate = rollout([_PHASE_EW, _PHASE_EW, _PHASE_NS, _PHASE_NS] * 3)

    assert hold > oscillate, (
        f"holding ({hold:.3f}) must strictly beat 10s ping-pong "
        f"({oscillate:.3f}) on an identical observation stream"
    )
