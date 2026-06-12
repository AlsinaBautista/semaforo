"""
Green Wave Simulation — Connects SUMO to the Flutter Dashboard via MQTT.

Runs a 3-intersection corridor in SUMO, applies green wave offsets,
extracts real-time telemetry from TraCI, and publishes protobuf messages
to Mosquitto so the Flutter Dashboard can visualize them.
"""

import os
import sys
import time
import logging
from pathlib import Path

import traci
import paho.mqtt.client as mqtt

# Add the project root to sys.path to access protobufs
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import telemetry_pb2

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SUMO_NET_FILE = PROJECT_ROOT / "brain" / "sumo_networks" / "corridor" / "corridor.net.xml"
SUMO_ROU_FILE = PROJECT_ROOT / "brain" / "sumo_networks" / "corridor" / "corridor.rou.xml"
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("green_wave")

# Mapping from SUMO traffic light IDs to Dashboard IDs
TL_MAP = {
    "c0": "int-001",
    "c1": "int-002",
    "c2": "int-003",
}

# Green wave parameters
CYCLE_LENGTH = 90        # seconds
OFFSET_SEC = 25          # 350m / (50km/h ÷ 3.6) ≈ 25s
PUBLISH_INTERVAL = 1     # publish telemetry every N simulation steps


def phase_name_from_state(state: str) -> str:
    """Derive a human-readable phase name from the SUMO signal state string.
    
    SUMO uses characters like G/g (green), y (yellow), r (red).
    We inspect the state string to determine the dominant phase.
    """
    if not state:
        return "UNKNOWN"
    
    greens = [i for i, c in enumerate(state) if c in ("G", "g")]
    yellows = [i for i, c in enumerate(state) if c == "y"]
    
    if yellows:
        return "YELLOW"
    if not greens:
        return "ALL_RED"
    
    # For a corridor, even-index signals tend to be EW, odd-index NS
    # But safer to just check which half of the state string has greens
    mid = len(state) // 2
    ew_greens = sum(1 for i in greens if i < mid)
    ns_greens = sum(1 for i in greens if i >= mid)
    
    if ew_greens > ns_greens:
        return "EW_GREEN"
    elif ns_greens > ew_greens:
        return "NS_GREEN"
    else:
        return "GREEN"


def on_connect(client, userdata, flags, rc, properties=None):
    logger.info("Connected to MQTT broker at %s:%d", MQTT_BROKER, MQTT_PORT)


def main():
    logger.info("=" * 60)
    logger.info("  Green Wave Simulation Starting")
    logger.info("=" * 60)

    # Ensure SUMO files exist
    if not SUMO_NET_FILE.exists():
        logger.error("SUMO net file not found: %s", SUMO_NET_FILE)
        logger.error("Run generate_corridor.py first!")
        sys.exit(1)
    if not SUMO_ROU_FILE.exists():
        logger.error("SUMO route file not found: %s", SUMO_ROU_FILE)
        sys.exit(1)

    # ---- MQTT setup -------------------------------------------------------
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id="sumo-greenwave",
    )
    client.on_connect = on_connect
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_start()

    # ---- SUMO setup -------------------------------------------------------
    # Always use headless sumo (not sumo-gui) for background execution
    sumo_binary = "sumo"
    
    # Check if SUMO_HOME is set; if eclipse-sumo pip package is installed,
    # the binary lives inside the venv.
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not sumo_home:
        # Try to find it from the eclipse-sumo package
        try:
            import sumo  # noqa: F401
            sumo_home = os.path.dirname(sumo.__file__)
            os.environ["SUMO_HOME"] = sumo_home
            sumo_binary = os.path.join(sumo_home, "bin", "sumo")
            logger.info("Using eclipse-sumo from pip: %s", sumo_binary)
        except ImportError:
            logger.warning("SUMO_HOME not set and eclipse-sumo not found. Trying system sumo...")

    traci_cmd = [
        sumo_binary,
        "-n", str(SUMO_NET_FILE),
        "-r", str(SUMO_ROU_FILE),
        "--step-length", "1",
        "--no-step-log",
        "--quit-on-end",
        "--start",
    ]

    logger.info("Starting SUMO: %s", " ".join(traci_cmd))
    traci.start(traci_cmd)
    logger.info("SUMO TraCI connected!")

    # ---- Configure green wave offsets -------------------------------------
    tl_ids = list(TL_MAP.keys())
    offsets = {}
    for idx, tl_id in enumerate(tl_ids):
        logics = traci.trafficlight.getAllProgramLogics(tl_id)
        if logics:
            logic = logics[0]
            # Set consistent phase durations: EW_GREEN=40, Yellow=5, NS_GREEN=40, Yellow=5
            if len(logic.phases) >= 4:
                logic.phases[0].duration = 40.0
                logic.phases[1].duration = 5.0
                logic.phases[2].duration = 40.0
                logic.phases[3].duration = 5.0

            # Apply green wave offset: c0=0s, c1=25s, c2=50s
            offset = idx * OFFSET_SEC
            logic.offset = offset
            offsets[tl_id] = offset
            traci.trafficlight.setProgramLogic(tl_id, logic)
            logger.info("  %s → offset=%ds, phases=%d", tl_id, offset, len(logic.phases))

    logger.info("Green wave offsets configured: %s", offsets)
    logger.info("Publishing telemetry to MQTT. Watch your Dashboard!")
    logger.info("-" * 60)

    # ---- Simulation loop --------------------------------------------------
    step = 0
    try:
        while traci.simulation.getMinExpectedNumber() > 0:
            traci.simulationStep()
            step += 1

            if step % PUBLISH_INTERVAL != 0:
                continue

            for tl_id, pub_id in TL_MAP.items():
                # Current signal state (e.g. "GGGrrrGGGrrr")
                state = traci.trafficlight.getRedYellowGreenState(tl_id)
                phase_idx = traci.trafficlight.getPhase(tl_id)
                phase = phase_name_from_state(state)

                # Count vehicles on all controlled lanes (deduplicated)
                controlled_lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))
                veh_count = sum(
                    traci.lane.getLastStepVehicleNumber(lane) for lane in controlled_lanes
                )

                # Average speed across controlled lanes
                speeds = [traci.lane.getLastStepMeanSpeed(lane) for lane in controlled_lanes]
                avg_speed = sum(speeds) / max(1, len(speeds))

                # Flow rate: vehicles * average_speed gives a throughput-like metric
                flow_rate = max(0.0, avg_speed * veh_count / 10.0)

                # Build protobuf
                t = telemetry_pb2.Telemetry()
                t.timestamp = int(time.time() * 1000)
                t.intersection_id = pub_id
                t.phase = phase
                t.phase_elapsed_s = phase_idx
                t.flow_rate = flow_rate
                t.vehicle_count = veh_count
                t.pedestrian_count = 0

                topic = f"semaforo/{pub_id}/telemetry"
                client.publish(topic, t.SerializeToString())

            # Log progress periodically
            if step % 60 == 0:
                sim_time = traci.simulation.getTime()
                remaining = traci.simulation.getMinExpectedNumber()
                logger.info(
                    "Step %d | SimTime=%.0fs | Vehicles remaining=%d",
                    step, sim_time, remaining,
                )

            # Pace the simulation (0.05s = 20x real-time)
            time.sleep(0.05)

    except traci.exceptions.FatalTraCIError:
        logger.info("SUMO simulation ended.")
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        try:
            traci.close()
        except Exception:
            pass
        client.loop_stop()
        client.disconnect()
        logger.info("Simulation finished after %d steps.", step)


if __name__ == "__main__":
    while True:
        try:
            main()
            logging.getLogger("green_wave").info("Simulation ended. Restarting in 3 seconds...")
            time.sleep(3)
        except KeyboardInterrupt:
            logging.getLogger("green_wave").info("Shutting down.")
            break
        except Exception as e:
            logging.getLogger("green_wave").error("Error: %s. Restarting in 5s...", e)
            time.sleep(5)
