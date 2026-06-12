# Semáforo Inteligente - Brain Module
"""Curriculum + domain-randomization scenarios for the Necochea corridor.

This module turns abstract "what could the real world throw at us" descriptions
into concrete SUMO ``.rou.xml`` demand files for the 3-intersection corridor on
Avenida 59.  It is the source of the *drastic* and *randomized* conditions the
MAPPO agent trains against so that the learned policy generalises instead of
overfitting one nominal flow.

Two ideas are combined:

* **Curriculum learning** — scenarios are grouped into ordered difficulty stages
  (:data:`CURRICULUM`).  Training starts on easy, almost-symmetric flows and
  only unlocks the chaotic ones (asymmetric summer exodus, sudestada storms,
  incident spikes) once the agent is competent, which stabilises early learning.

* **Domain randomization** — every episode samples a fresh :class:`ScenarioSpec`
  *and* jitters its continuous knobs (demand scale, asymmetry, turn ratios,
  weather-driven vehicle dynamics, random seed).  The agent therefore never sees
  the exact same world twice and cannot memorise a single timing plan.

The corridor route/edge names mirror ``sumo_networks/corridor/corridor.rou.xml``:

    EW through : west_entry_to_c0 -> c0_to_c1 -> c1_to_c2 -> c2_to_east_entry
    WE through : east_entry_to_c2 -> c2_to_c1 -> c1_to_c0 -> c0_to_west_entry
    cross @ cN : cN_north_in -> cN_south_out   (and the reverse sn_cN)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence
from xml.sax.saxutils import escape

# ---------------------------------------------------------------------------
# Weather -> vehicle dynamics
# ---------------------------------------------------------------------------
# Weather mostly affects *how vehicles move*, not how many. A sudestada (the
# violent SE storm Necochea is known for) means standing water, poor visibility
# and cautious drivers: lower top speed, gentler acceleration, longer gaps and
# higher driver imperfection (sigma). Heat-haze summer afternoons make drivers
# marginally more aggressive.


@dataclass(frozen=True)
class WeatherProfile:
    """Multiplicative modifiers applied to the base ``car`` vType."""

    name: str
    speed_factor: float = 1.0      # scales maxSpeed
    accel_factor: float = 1.0      # scales accel
    decel_factor: float = 1.0      # scales decel (lower = longer stopping)
    min_gap_factor: float = 1.0    # scales minGap (car-following caution)
    sigma: float = 0.5             # driver imperfection in [0, 1]


WEATHER_CLEAR = WeatherProfile("clear")
WEATHER_HEAT_HAZE = WeatherProfile(
    "heat_haze", speed_factor=1.05, accel_factor=1.10, min_gap_factor=0.9, sigma=0.6
)
WEATHER_RAIN = WeatherProfile(
    "rain", speed_factor=0.80, accel_factor=0.85, decel_factor=0.80,
    min_gap_factor=1.4, sigma=0.6,
)
WEATHER_SUDESTADA = WeatherProfile(
    "sudestada", speed_factor=0.55, accel_factor=0.65, decel_factor=0.60,
    min_gap_factor=2.0, sigma=0.75,
)

# ---------------------------------------------------------------------------
# Scenario specification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioSpec:
    """A fully-resolved description of one episode's traffic world.

    Attributes:
        name: Human-readable scenario id (logged + used in filenames).
        duration_s: Simulated seconds of demand to emit.
        base_flow: Reference vehicles/hour for the *main* arterial direction
            before asymmetry is applied.
        ew_bias: Fraction of arterial demand on the East->West axis, in
            ``[0, 1]``.  ``0.5`` is symmetric; ``0.9`` is a heavy WE exodus.
        cross_ratio: Cross-street demand as a fraction of arterial demand.
        bus_ratio: Fraction of arterial vehicles that are buses.
        turn_ratio: Fraction of through vehicles that turn off at an
            intersection (loads the cross movements).
        weather: Weather profile driving vehicle dynamics.
        seed: SUMO/demand RNG seed for this episode.
    """

    name: str
    duration_s: int = 3600
    base_flow: float = 600.0
    ew_bias: float = 0.5
    cross_ratio: float = 0.25
    bus_ratio: float = 0.03
    turn_ratio: float = 0.10
    weather: WeatherProfile = WEATHER_CLEAR
    seed: int = 0
    # Per-scenario jitter half-widths used by domain randomization.
    jitter: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Canonical scenarios (the "drastic" worlds)
# ---------------------------------------------------------------------------

WINTER_QUIET = ScenarioSpec(
    name="winter_quiet",
    base_flow=120.0,
    ew_bias=0.5,
    cross_ratio=0.30,
    bus_ratio=0.05,
    turn_ratio=0.12,
    weather=WEATHER_CLEAR,
    jitter={"base_flow": 40.0, "ew_bias": 0.05},
)

WINTER_NIGHT_EMPTY = ScenarioSpec(
    name="winter_night_empty",
    base_flow=25.0,
    ew_bias=0.5,
    cross_ratio=0.40,
    bus_ratio=0.0,
    turn_ratio=0.10,
    weather=WEATHER_CLEAR,
    jitter={"base_flow": 15.0},
)

SHOULDER_BALANCED = ScenarioSpec(
    name="shoulder_balanced",
    base_flow=500.0,
    ew_bias=0.5,
    cross_ratio=0.28,
    bus_ratio=0.04,
    turn_ratio=0.12,
    weather=WEATHER_CLEAR,
    jitter={"base_flow": 120.0, "ew_bias": 0.08, "cross_ratio": 0.05},
)

RAIN_MODERATE = ScenarioSpec(
    name="rain_moderate",
    base_flow=450.0,
    ew_bias=0.55,
    cross_ratio=0.25,
    bus_ratio=0.04,
    turn_ratio=0.12,
    weather=WEATHER_RAIN,
    jitter={"base_flow": 120.0, "ew_bias": 0.1},
)

SUMMER_EXODUS_BEACH = ScenarioSpec(
    name="summer_exodus_beach",   # midday rush toward the coast (E->W toward beach)
    base_flow=1100.0,
    ew_bias=0.85,
    cross_ratio=0.18,
    bus_ratio=0.05,
    turn_ratio=0.08,
    weather=WEATHER_HEAT_HAZE,
    jitter={"base_flow": 250.0, "ew_bias": 0.07, "turn_ratio": 0.04},
)

SUMMER_RETURN_DOWNTOWN = ScenarioSpec(
    name="summer_return_downtown",  # evening return toward downtown (W->E)
    base_flow=1050.0,
    ew_bias=0.15,
    cross_ratio=0.20,
    bus_ratio=0.05,
    turn_ratio=0.10,
    weather=WEATHER_HEAT_HAZE,
    jitter={"base_flow": 250.0, "ew_bias": 0.07},
)

SUDESTADA_STORM = ScenarioSpec(
    name="sudestada_storm",  # sudden SE storm: moderate demand, degraded dynamics
    base_flow=550.0,
    ew_bias=0.6,
    cross_ratio=0.30,
    bus_ratio=0.04,
    turn_ratio=0.15,
    weather=WEATHER_SUDESTADA,
    jitter={"base_flow": 200.0, "ew_bias": 0.15, "cross_ratio": 0.08},
)

INCIDENT_ASYMMETRIC_SPIKE = ScenarioSpec(
    name="incident_asymmetric_spike",  # crash/closure dumps all demand one way
    base_flow=1300.0,
    ew_bias=0.95,
    cross_ratio=0.10,
    bus_ratio=0.03,
    turn_ratio=0.20,
    weather=WEATHER_CLEAR,
    jitter={"base_flow": 300.0, "ew_bias": 0.04, "turn_ratio": 0.06},
)

# ---------------------------------------------------------------------------
# Curriculum: ordered stages, easy -> chaotic
# ---------------------------------------------------------------------------

CURRICULUM: list[list[ScenarioSpec]] = [
    # Stage 0 — learn the basics on calm, near-symmetric flow.
    [WINTER_QUIET, WINTER_NIGHT_EMPTY, SHOULDER_BALANCED],
    # Stage 1 — introduce volume, mild asymmetry and rain dynamics.
    [SHOULDER_BALANCED, RAIN_MODERATE, SUMMER_RETURN_DOWNTOWN],
    # Stage 2 — full chaos: heavy asymmetric exodus, storms, incident spikes.
    [
        SUMMER_EXODUS_BEACH,
        SUMMER_RETURN_DOWNTOWN,
        SUDESTADA_STORM,
        INCIDENT_ASYMMETRIC_SPIKE,
    ],
]

ALL_SCENARIOS: list[ScenarioSpec] = [
    WINTER_QUIET, WINTER_NIGHT_EMPTY, SHOULDER_BALANCED, RAIN_MODERATE,
    SUMMER_EXODUS_BEACH, SUMMER_RETURN_DOWNTOWN, SUDESTADA_STORM,
    INCIDENT_ASYMMETRIC_SPIKE,
]


# ---------------------------------------------------------------------------
# Domain randomization
# ---------------------------------------------------------------------------


def randomize(spec: ScenarioSpec, rng: random.Random) -> ScenarioSpec:
    """Return a jittered copy of *spec* for one episode of domain randomization.

    Each continuous knob listed in ``spec.jitter`` is perturbed uniformly within
    +/- its half-width and clamped to a sane range.  A fresh integer ``seed`` is
    always drawn so SUMO's own stochastic departures differ too.

    Args:
        spec: The base scenario to perturb.
        rng: A seeded ``random.Random`` for reproducibility.

    Returns:
        A new :class:`ScenarioSpec` with randomized continuous parameters.
    """

    def jit(value: float, key: str, lo: float, hi: float) -> float:
        half = spec.jitter.get(key, 0.0)
        if half <= 0.0:
            return value
        return float(min(hi, max(lo, value + rng.uniform(-half, half))))

    return replace(
        spec,
        base_flow=jit(spec.base_flow, "base_flow", 10.0, 2500.0),
        ew_bias=jit(spec.ew_bias, "ew_bias", 0.02, 0.98),
        cross_ratio=jit(spec.cross_ratio, "cross_ratio", 0.02, 0.8),
        turn_ratio=jit(spec.turn_ratio, "turn_ratio", 0.0, 0.5),
        seed=rng.randint(0, 2**31 - 1),
    )


def sample_scenario(
    rng: random.Random,
    *,
    stage: int | None = None,
    pool: Sequence[ScenarioSpec] | None = None,
) -> ScenarioSpec:
    """Sample and randomize one scenario for an episode.

    Args:
        rng: Seeded RNG.
        stage: Curriculum stage index.  Scenarios are drawn from all stages up
            to and including ``stage`` (so mastered easy cases are not forgotten).
            Ignored if ``pool`` is given.
        pool: Explicit candidate scenarios; overrides ``stage``.

    Returns:
        A randomized :class:`ScenarioSpec`.
    """
    if pool is None:
        if stage is None:
            pool = ALL_SCENARIOS
        else:
            stage = max(0, min(stage, len(CURRICULUM) - 1))
            pool = [s for i in range(stage + 1) for s in CURRICULUM[i]]
    base = rng.choice(list(pool))
    return randomize(base, rng)


def sample_mixed_task_scenario(
    rng: random.Random,
    *,
    stage: int,
    rehearsal_ratio: float = 0.25,
    decay: float = 0.5,
) -> ScenarioSpec:
    """Sample a scenario from an asymmetric mixture of current and past stages.

    Mixed-task rehearsal against catastrophic forgetting: with probability
    ``1 - rehearsal_ratio`` the episode comes from the *current* curriculum
    stage; otherwise it replays a *past* stage ``k < stage`` chosen with
    geometric weights ``decay**(stage - 1 - k)`` (the most recent past stage
    is favoured, older stages keep non-zero mass).  Because PPO is on-policy,
    rehearsal happens at the task level — fresh rollouts on historical
    scenarios — rather than by replaying stale trajectories.

    Args:
        rng: Seeded RNG.
        stage: Current curriculum stage index.
        rehearsal_ratio: Probability of drawing a historical-stage episode.
        decay: Geometric decay applied to older past stages.

    Returns:
        A randomized :class:`ScenarioSpec`.
    """
    stage = max(0, min(stage, len(CURRICULUM) - 1))
    if stage == 0 or rng.random() >= rehearsal_ratio:
        return sample_scenario(rng, pool=CURRICULUM[stage])

    weights = [decay ** (stage - 1 - k) for k in range(stage)]
    past_stage = rng.choices(range(stage), weights=weights, k=1)[0]
    return sample_scenario(rng, pool=CURRICULUM[past_stage])


# ---------------------------------------------------------------------------
# SUMO route-file generation
# ---------------------------------------------------------------------------

_BASE_CAR_SPEED = 13.89  # m/s (~50 km/h) nominal arterial design speed
_BASE_BUS_SPEED = 11.11
_CROSS_NODES = ("c0", "c1", "c2")


def _vtypes_xml(weather: WeatherProfile) -> str:
    """Build weather-adjusted vType definitions."""
    car_speed = round(_BASE_CAR_SPEED * weather.speed_factor, 3)
    bus_speed = round(_BASE_BUS_SPEED * weather.speed_factor, 3)
    accel = round(2.6 * weather.accel_factor, 3)
    decel = round(4.5 * weather.decel_factor, 3)
    bus_accel = round(1.5 * weather.accel_factor, 3)
    bus_decel = round(3.5 * weather.decel_factor, 3)
    min_gap = round(2.5 * weather.min_gap_factor, 3)
    bus_gap = round(3.0 * weather.min_gap_factor, 3)
    sigma = round(weather.sigma, 3)
    return (
        f'  <vType id="car" accel="{accel}" decel="{decel}" sigma="{sigma}" '
        f'length="5.0" minGap="{min_gap}" maxSpeed="{car_speed}" guiShape="passenger"/>\n'
        f'  <vType id="bus" accel="{bus_accel}" decel="{bus_decel}" sigma="{sigma}" '
        f'length="12.0" minGap="{bus_gap}" maxSpeed="{bus_speed}" guiShape="bus" '
        f'color="0,128,255"/>\n'
    )


def _prob(veh_per_hour: float) -> float:
    """Convert vehicles/hour to a per-second insertion probability in [0, 1]."""
    return max(0.0, min(1.0, veh_per_hour / 3600.0))


def scenario_to_routes_xml(spec: ScenarioSpec) -> str:
    """Render a complete SUMO ``.rou.xml`` document for *spec*.

    Args:
        spec: A (typically already randomized) scenario specification.

    Returns:
        XML text ready to write to disk.
    """
    end = spec.duration_s
    ew_flow = spec.base_flow * spec.ew_bias
    we_flow = spec.base_flow * (1.0 - spec.ew_bias)
    cross_flow = spec.base_flow * spec.cross_ratio / len(_CROSS_NODES)
    bus_ew = ew_flow * spec.bus_ratio
    bus_we = we_flow * spec.bus_ratio
    turn_flow = spec.base_flow * spec.turn_ratio

    parts: list[str] = ['<?xml version="1.0" encoding="UTF-8"?>\n']
    parts.append(f"<!-- scenario: {escape(spec.name)} weather: "
                 f"{escape(spec.weather.name)} seed: {spec.seed} -->\n")
    parts.append("<routes>\n")
    parts.append(_vtypes_xml(spec.weather))

    # Route definitions (must match corridor.net.xml edge ids).
    parts.append(
        '  <route id="ew_through" edges="west_entry_to_c0 c0_to_c1 c1_to_c2 c2_to_east_entry"/>\n'
        '  <route id="we_through" edges="east_entry_to_c2 c2_to_c1 c1_to_c0 c0_to_west_entry"/>\n'
        '  <route id="w_to_c1n" edges="west_entry_to_c0 c0_to_c1 c1_north_out"/>\n'
        '  <route id="c0s_to_e" edges="c0_south_in c0_to_c1 c1_to_c2 c2_to_east_entry"/>\n'
    )
    for n in _CROSS_NODES:
        parts.append(f'  <route id="ns_{n}" edges="{n}_north_in {n}_south_out"/>\n')
        parts.append(f'  <route id="sn_{n}" edges="{n}_south_in {n}_north_out"/>\n')

    # SUMO rejects a <flow> whose repetition probability is <= 0, which kills the
    # whole simulation. Near-empty scenarios (e.g. winter nights) legitimately
    # produce zero demand on some movements, so we simply omit those flows.
    min_emit = 1e-5  # below this the 5-decimal probability would print as 0

    def flow(fid: str, vtype: str, route: str, prob: float) -> str:
        if prob < min_emit:
            return ""
        return (
            f'  <flow id="{fid}" type="{vtype}" route="{route}" begin="0" '
            f'end="{end}.0" probability="{prob:.5f}" departLane="best" departSpeed="max"/>\n'
        )

    # Arterial through demand (the asymmetric part).
    parts.append(flow("flow_ew", "car", "ew_through", _prob(ew_flow)))
    parts.append(flow("flow_we", "car", "we_through", _prob(we_flow)))
    parts.append(flow("flow_bus_ew", "bus", "ew_through", _prob(bus_ew)))
    parts.append(flow("flow_bus_we", "bus", "we_through", _prob(bus_we)))

    # Cross-street demand at each intersection.
    for n in _CROSS_NODES:
        parts.append(flow(f"flow_ns_{n}", "car", f"ns_{n}", _prob(cross_flow)))
        parts.append(flow(f"flow_sn_{n}", "car", f"sn_{n}", _prob(cross_flow)))

    # Turning movements that load the coordination problem.
    parts.append(flow("flow_w_c1n", "car", "w_to_c1n", _prob(turn_flow * 0.5)))
    parts.append(flow("flow_c0s_e", "car", "c0s_to_e", _prob(turn_flow * 0.5)))

    parts.append("</routes>\n")
    return "".join(parts)


def write_scenario_routes(spec: ScenarioSpec, out_dir: str | Path) -> Path:
    """Write *spec*'s route file to ``out_dir`` and return its path.

    The filename embeds the scenario name and seed so concurrent rollout
    workers never collide and episodes are reproducible.

    Args:
        spec: Scenario to render.
        out_dir: Directory to write into (created if missing).

    Returns:
        Path to the written ``.rou.xml`` file.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{spec.name}_{spec.seed}.rou.xml"
    path.write_text(scenario_to_routes_xml(spec), encoding="utf-8")
    return path
