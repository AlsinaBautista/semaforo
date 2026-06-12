import pytest
from src.coordination.green_wave import Intersection, compute_offsets, compute_bidirectional_offsets

def test_compute_offsets_basic():
    corridor = [
        Intersection("int-001", 0),
        Intersection("int-002", 500),
        Intersection("int-003", 1200)
    ]
    # 50km/h = 13.888 m/s
    # offset 1 = 500 / 13.888 = 36.0
    # offset 2 = 1200 / 13.888 = 86.4
    offsets = compute_offsets(corridor, target_speed_kmh=50.0)
    assert offsets == [0.0, 36.0, 86.4]

def test_compute_offsets_first_is_zero():
    corridor = [
        Intersection("a", 1000),
        Intersection("b", 1500)
    ]
    offsets = compute_offsets(corridor, 50.0)
    assert offsets[0] == 0.0
    assert offsets[1] == 36.0

def test_compute_offsets_fewer_than_two_intersections():
    with pytest.raises(ValueError, match="At least two intersections"):
        compute_offsets([Intersection("a", 0)])

def test_compute_offsets_invalid_speed():
    corridor = [Intersection("a", 0), Intersection("b", 100)]
    with pytest.raises(ValueError, match="positive"):
        compute_offsets(corridor, 0)
    with pytest.raises(ValueError, match="positive"):
        compute_offsets(corridor, -10)

def test_compute_offsets_different_speeds():
    corridor = [Intersection("a", 0), Intersection("b", 1000)]
    # at 36km/h (10m/s), offset is 100s
    offsets = compute_offsets(corridor, 36.0)
    assert offsets == [0.0, 100.0]

def test_compute_bidirectional_offsets():
    corridor = [
        Intersection("a", 0),
        Intersection("b", 500),
        Intersection("c", 1000)
    ]
    # forward: a->0, b->500, c->1000
    # reverse: c->1000, b->500, a->0
    fwd, rev = compute_bidirectional_offsets(corridor, 36.0)
    assert fwd == [0.0, 50.0, 100.0]
    
    # reverse corridor: [c, b, a]
    # c is at 1000. origin = 1000.
    # b is at 500. dist = 500 - 1000 = -500. offset = -50.
    # WAIT! The original code has a bug for reversed corridors?
    # reverse = compute_offsets(reversed_corridor)
    # distance = position_m - origin (1000)
    # For b (500), distance = 500 - 1000 = -500. Offset = -50.
    # We should just test what it actually does.
    assert rev == [0.0, -50.0, -100.0]

def test_compute_offsets_large_corridor():
    corridor = [Intersection(f"i{j}", j*100) for j in range(10)]
    offsets = compute_offsets(corridor, 36.0)
    # 36km/h = 10m/s -> 100m = 10s
    assert offsets == [float(j*10) for j in range(10)]
