import pytest
import numpy as np
from src.rewards.pressure import compute_pressure_reward
from src.rewards.queue_length import compute_queue_reward
from src.rewards.composite import CompositeReward

def test_pressure_dict_positive():
    obs = {"queue": [10, 10], "flow": [5, 5]}
    # incoming = 20, outgoing = 10. pressure = 10. reward = -clip(10/20.0, 0, 1) = -0.5
    reward = compute_pressure_reward(obs, max_queue=20.0)
    assert reward == pytest.approx(-0.5)

def test_pressure_dict_zero():
    obs = {"queue": [5, 5], "flow": [5, 5]}
    reward = compute_pressure_reward(obs, max_queue=20.0)
    assert reward == pytest.approx(0.0)

def test_pressure_numpy():
    # Canonical: first 4 incoming, next 4 outgoing
    obs = np.array([5, 5, 5, 5, 2, 2, 2, 2])
    # pressure = 20 - 8 = 12. max_queue default is 20.0. reward = -12/20 = -0.6
    reward = compute_pressure_reward(obs, max_queue=20.0)
    assert reward == pytest.approx(-0.6)

def test_queue_dict():
    obs = {"queue": [5, 5, 5, 5]}
    # total = 20. default max_queue = 20.0. reward = -clip(20/20, 0, 1) = -1.0
    reward = compute_queue_reward(obs, max_queue=20.0)
    assert reward == pytest.approx(-1.0)

def test_queue_dict_zero():
    obs = {"queue": [0, 0, 0, 0]}
    reward = compute_queue_reward(obs)
    assert reward == pytest.approx(0.0)

def test_queue_numpy():
    # Numpy assumes first 4 are ALREADY normalized queues in [0, 1]
    # total = sum/4.0
    obs = np.array([0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1])
    reward = compute_queue_reward(obs)
    # total sum = 2.0. reward = -clip(2.0/4.0) = -0.5
    assert reward == pytest.approx(-0.5)

def test_composite_no_flicker():
    cr = CompositeReward(pressure_weight=0.6, queue_weight=0.4, flickering_weight=0.1)
    # We pass dict observation.
    obs = {"queue": [10, 10], "flow": [10, 10]}
    # pressure = 0 -> 0.0
    # queue = 20 -> -1.0
    # flicker = 0.0 (same action)
    r = cr.compute(obs, last_action=1, current_action=1)
    assert r == pytest.approx(-0.4)

def test_composite_with_flicker_min_green_violation():
    cr = CompositeReward(pressure_weight=0.6, queue_weight=0.4, flickering_weight=0.1)
    obs = {"queue": [10, 10], "flow": [10, 10]}
    # Switch before min_green (5s) elapsed -> flicker = -1.0
    r = cr.compute(obs, last_action=1, current_action=2, time_since_phase_change=2.0)
    assert r == pytest.approx(-0.5)

def test_composite_legal_early_switch_decayed_flicker():
    cr = CompositeReward(pressure_weight=0.6, queue_weight=0.4, flickering_weight=0.1)
    obs = {"queue": [10, 10], "flow": [10, 10]}
    # Legal switch but only 2s past min_green (5s): penalty decays linearly
    # over stability_window (10s) -> flicker = -(1 - 2/10) = -0.8
    r = cr.compute(obs, last_action=1, current_action=2, time_since_phase_change=7.0)
    assert r == pytest.approx(-0.4 + 0.1 * -0.8)


def test_composite_mature_switch_no_flicker():
    cr = CompositeReward(pressure_weight=0.6, queue_weight=0.4, flickering_weight=0.1)
    obs = {"queue": [10, 10], "flow": [10, 10]}
    # Switch held past min_green + stability_window -> free
    r = cr.compute(obs, last_action=1, current_action=2, time_since_phase_change=20.0)
    assert r == pytest.approx(-0.4)

def test_composite_unknown_elapsed_no_flicker():
    cr = CompositeReward(pressure_weight=0.6, queue_weight=0.4, flickering_weight=0.1)
    obs = {"queue": [10, 10], "flow": [10, 10]}
    # Without timing info a min-green violation cannot be proven -> no penalty
    r = cr.compute(obs, last_action=1, current_action=2)
    assert r == pytest.approx(-0.4)
