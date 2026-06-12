"""
Green Wave Benchmark — Compare green wave offsets vs. fixed timing.

Runs the 3-intersection SUMO corridor twice:
  1. With computed green wave offsets (offset = distance / speed)
  2. With fixed timing (all offsets = 0, no coordination)

Collects per-intersection metrics:
  - Average waiting time (seconds)
  - Average queue length (vehicles)
  - Total throughput (vehicles that completed their route)

Outputs a comparison table so we can prove the green wave works.
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field

import traci

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

SUMO_NET_FILE = PROJECT_ROOT / "brain" / "sumo_networks" / "corridor" / "corridor.net.xml"
SUMO_ROU_FILE = PROJECT_ROOT / "brain" / "sumo_networks" / "corridor" / "corridor.rou.xml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("benchmark")

TL_IDS = ["c0", "c1", "c2"]
OFFSET_SEC = 25  # 350m / (50km/h / 3.6) ≈ 25s


@dataclass
class SimMetrics:
    """Aggregated metrics from a single simulation run."""
    name: str
    total_vehicles_arrived: int = 0
    total_waiting_time: float = 0.0
    total_queue_samples: int = 0
    total_queue_sum: float = 0.0
    simulation_steps: int = 0
    per_tl_waiting: dict = field(default_factory=dict)
    per_tl_queue: dict = field(default_factory=dict)

    @property
    def avg_waiting_time(self) -> float:
        return self.total_waiting_time / max(1, self.total_vehicles_arrived)

    @property
    def avg_queue_length(self) -> float:
        return self.total_queue_sum / max(1, self.total_queue_samples)


def get_sumo_binary() -> str:
    """Find the SUMO binary."""
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        try:
            import sumo
            sumo_home = os.path.dirname(sumo.__file__)
            os.environ["SUMO_HOME"] = sumo_home
            return os.path.join(sumo_home, "bin", "sumo")
        except ImportError:
            pass
    return "sumo"


import xml.etree.ElementTree as ET
import shutil

def apply_offsets_to_net_xml(base_net_file: Path, mode: str) -> Path:
    """Creates a temporary .net.xml with offsets applied."""
    tree = ET.parse(base_net_file)
    root = tree.getroot()
    
    for idx, tl_id in enumerate(TL_IDS):
        # Find the tlLogic node for this traffic light
        for logic in root.findall(f".//tlLogic[@id='{tl_id}']"):
            if mode == "green_wave":
                logic.set("offset", str(idx * OFFSET_SEC))
            else:
                logic.set("offset", "0")
                
    out_path = base_net_file.parent / f"corridor_{mode}.net.xml"
    tree.write(out_path)
    return out_path

def run_simulation(mode: str) -> SimMetrics:
    """Run a full SUMO simulation with the given traffic light mode.

    Args:
        mode: 'green_wave' for coordinated offsets, 'fixed' for all-zero offsets.

    Returns:
        SimMetrics with aggregated results.
    """
    metrics = SimMetrics(name=mode)
    
    net_file = apply_offsets_to_net_xml(SUMO_NET_FILE, mode)

    sumo_binary = get_sumo_binary()
    traci_cmd = [
        sumo_binary,
        "-n", str(net_file),
        "-r", str(SUMO_ROU_FILE),
        "--step-length", "1",
        "--no-step-log",
        "--quit-on-end",
        "--start",
        "--seed", "42",  # Reproducible
    ]

    traci.start(traci_cmd)

    # Initialize per-TL metrics
    for tl_id in TL_IDS:
        metrics.per_tl_waiting[tl_id] = []
        metrics.per_tl_queue[tl_id] = []

    step = 0
    arrived_total = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

            # Accumulate arrived vehicles
            arrived_total += traci.simulation.getArrivedNumber()

            for tl_id in TL_IDS:
                controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))

                # Waiting vehicles (speed < 0.1 m/s)
                waiting = sum(
                    traci.lane.getLastStepHaltingNumber(lane)
                    for lane in controlled_lanes
                )
                metrics.per_tl_waiting[tl_id].append(waiting)

                # Queue length
                queue = sum(
                    traci.lane.getLastStepVehicleNumber(lane)
                    for lane in controlled_lanes
                )
                metrics.per_tl_queue[tl_id].append(queue)
                metrics.total_queue_sum += queue
                metrics.total_queue_samples += 1

    except traci.exceptions.FatalTraCIError:
        pass
    finally:
        # Get final stats
        metrics.total_vehicles_arrived = arrived_total
        metrics.total_waiting_time = sum(
            sum(w) for w in metrics.per_tl_waiting.values()
        )
        metrics.simulation_steps = step
        try:
            traci.close()
        except Exception:
            pass

    return metrics


def print_comparison(gw: SimMetrics, fixed: SimMetrics):
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print("   BENCHMARK: Green Wave vs. Fixed Timing")
    print("=" * 70)

    print(f"\n{'Metric':<35} {'Green Wave':>15} {'Fixed':>15}")
    print("-" * 70)
    print(f"{'Simulation steps':<35} {gw.simulation_steps:>15} {fixed.simulation_steps:>15}")
    print(f"{'Vehicles arrived':<35} {gw.total_vehicles_arrived:>15} {fixed.total_vehicles_arrived:>15}")

    # Per-TL average queue
    for tl_id in TL_IDS:
        gw_avg = sum(gw.per_tl_queue[tl_id]) / max(1, len(gw.per_tl_queue[tl_id]))
        fx_avg = sum(fixed.per_tl_queue[tl_id]) / max(1, len(fixed.per_tl_queue[tl_id]))
        improvement = ((fx_avg - gw_avg) / max(fx_avg, 0.01)) * 100
        print(f"{'  Avg queue ' + tl_id:<35} {gw_avg:>14.1f}v {fx_avg:>14.1f}v  ({improvement:+.1f}%)")

    # Per-TL average halting
    for tl_id in TL_IDS:
        gw_avg = sum(gw.per_tl_waiting[tl_id]) / max(1, len(gw.per_tl_waiting[tl_id]))
        fx_avg = sum(fixed.per_tl_waiting[tl_id]) / max(1, len(fixed.per_tl_waiting[tl_id]))
        improvement = ((fx_avg - gw_avg) / max(fx_avg, 0.01)) * 100
        print(f"{'  Avg halting ' + tl_id:<35} {gw_avg:>14.1f}v {fx_avg:>14.1f}v  ({improvement:+.1f}%)")

    # Overall
    gw_total_queue = gw.avg_queue_length
    fx_total_queue = fixed.avg_queue_length
    overall_improvement = ((fx_total_queue - gw_total_queue) / max(fx_total_queue, 0.01)) * 100

    print("-" * 70)
    print(f"{'OVERALL avg queue':<35} {gw_total_queue:>14.2f}v {fx_total_queue:>14.2f}v  ({overall_improvement:+.1f}%)")
    print("=" * 70)

    if overall_improvement > 0:
        print(f"\n✅ Green Wave reduces average queue length by {overall_improvement:.1f}%")
    else:
        print(f"\n⚠️  Green Wave did NOT improve queue length ({overall_improvement:.1f}%)")

    # Save results to JSON
    results = {
        "green_wave": {
            "steps": gw.simulation_steps,
            "vehicles_arrived": gw.total_vehicles_arrived,
            "avg_queue": round(gw_total_queue, 3),
            "per_tl": {
                tl: {
                    "avg_queue": round(sum(gw.per_tl_queue[tl]) / max(1, len(gw.per_tl_queue[tl])), 3),
                    "avg_halting": round(sum(gw.per_tl_waiting[tl]) / max(1, len(gw.per_tl_waiting[tl])), 3),
                }
                for tl in TL_IDS
            },
        },
        "fixed": {
            "steps": fixed.simulation_steps,
            "vehicles_arrived": fixed.total_vehicles_arrived,
            "avg_queue": round(fx_total_queue, 3),
            "per_tl": {
                tl: {
                    "avg_queue": round(sum(fixed.per_tl_queue[tl]) / max(1, len(fixed.per_tl_queue[tl])), 3),
                    "avg_halting": round(sum(fixed.per_tl_waiting[tl]) / max(1, len(fixed.per_tl_waiting[tl])), 3),
                }
                for tl in TL_IDS
            },
        },
        "improvement_pct": round(overall_improvement, 2),
    }

    results_path = PROJECT_ROOT / "brain" / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n📊 Results saved to {results_path}")


def main():
    if not SUMO_NET_FILE.exists() or not SUMO_ROU_FILE.exists():
        logger.error("SUMO files not found. Run generate_corridor.py first!")
        sys.exit(1)

    logger.info("Running benchmark: Green Wave vs Fixed Timing...")
    logger.info("This will run 2 full SUMO simulations (3600s each).")

    logger.info("\n[1/2] Running GREEN WAVE simulation...")
    gw_metrics = run_simulation("green_wave")
    logger.info("  → Done: %d steps, %d vehicles arrived", gw_metrics.simulation_steps, gw_metrics.total_vehicles_arrived)

    logger.info("\n[2/2] Running FIXED TIMING simulation...")
    fixed_metrics = run_simulation("fixed")
    logger.info("  → Done: %d steps, %d vehicles arrived", fixed_metrics.simulation_steps, fixed_metrics.total_vehicles_arrived)

    print_comparison(gw_metrics, fixed_metrics)


if __name__ == "__main__":
    main()
