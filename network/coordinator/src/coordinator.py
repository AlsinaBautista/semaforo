# =============================================================================
# Semáforo Inteligente — Green Wave Coordinator
# =============================================================================
"""
Main coordinator module for multi-intersection green wave synchronization.

The GreenWaveCoordinator:
  1. Connects to the MQTT broker.
  2. Loads the network topology (intersections and roads).
  3. Subscribes to telemetry from all intersections.
  4. Computes coordinated timing plans (green wave offsets).
  5. Publishes timing plans and sync pulses.
  6. Handles graceful shutdown via signals.

Usage:
    python -m src.coordinator
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
import yaml

from .graph_manager import NetworkGraphManager
from .sync_service import SyncService
from .emergency import EmergencyHandler

# Add brain to path to import auditor
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../brain")))
try:
    from src.environments.shadow_auditor import ShadowAuditor
except ImportError:
    ShadowAuditor = None

# Exported MAPPO actor (see brain/src/export/onnx_exporter.py). Like the topic
# contract below, a missing model aborts startup rather than silently falling
# back to a heuristic.
_DEFAULT_ONNX_MODEL = (
    Path(__file__).resolve().parents[3] / "brain" / "models" / "policy.onnx"
)
ONNX_MODEL_PATH = os.getenv("ONNX_MODEL_PATH", str(_DEFAULT_ONNX_MODEL))

# Brain observation contract (brain/src/environments/multi_intersection.py):
# [phase one-hot (2), min-green flag (1), 12 lane densities, 12 lane queues],
# zero-padded to 32 and clipped to [0, 1].
_OBS_SIZE = 17
_NUM_LANES = 12
_LANE_CAPACITY_VEHICLES = 20.0  # vehicles per lane considered saturation


class AIPolicyWrapper:
    """Política de IA (MAPPO) que toma decisiones basadas en RL.

    Carga el actor exportado a ONNX y sirve inferencia con onnxruntime.
    """

    _PHASE_BY_ACTION = ("GREEN_NS", "GREEN_EW")  # Discrete(2): 0 -> NS, 1 -> EW

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else Path(ONNX_MODEL_PATH)
        if not path.is_file():
            raise RuntimeError(
                f"ONNX actor model not found: {path.resolve()}. "
                "The coordinator refuses to start without the exported AI "
                "policy; run `PYTHONPATH=brain .venv/bin/python "
                "brain/scripts/export_model.py` (override with ONNX_MODEL_PATH)."
            )
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run the actor on a batch of observations; returns action logits."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != _OBS_SIZE:
            raise ValueError(
                f"Expected observation batch of shape (B, {_OBS_SIZE}), "
                f"got {obs.shape}."
            )
        return self._session.run(None, {self._input_name: obs})[0]

    def predict(self, telemetry_payload: dict[str, Any]) -> str:
        """Map Edge telemetry to a recommended phase string."""
        obs = self._telemetry_to_observation(telemetry_payload)
        logits = self.infer(obs[None, :])
        return self._PHASE_BY_ACTION[int(np.argmax(logits[0]))]

    def _telemetry_to_observation(
        self, telemetry: dict[str, Any]
    ) -> np.ndarray:
        """Best-effort projection of aggregate telemetry onto the canonical
        17-dim observation: ``[queue x4, density x4, phase one-hot x4,
        phase_elapsed, downstream x4]``.

        The Edge reports only the active phase and a total vehicle count (no
        per-lane data, no phase timer, no downstream occupancy), so the count
        is spread uniformly over the queue and density slots and the unknown
        slots stay zero.
        """
        vec = np.zeros(_OBS_SIZE, dtype=np.float32)

        vehicle_count = float(telemetry.get("vehicle_count", 0) or 0)
        per_direction = min(
            1.0, (vehicle_count / _NUM_LANES) / _LANE_CAPACITY_VEHICLES
        )
        vec[0:4] = per_direction                    # directional queues
        vec[4:8] = per_direction                    # directional densities

        phase = telemetry.get("phase")
        if phase == "GREEN_NS":
            vec[8] = 1.0
        elif phase == "GREEN_EW":
            vec[9] = 1.0
        # Amber/all-red/unknown phases carry no green one-hot.

        # No phase timer in telemetry: elapsed stays 0 ("just switched"); the
        # Edge SafetyLayer enforces the real min-green regardless.
        # Indices 13-16 (downstream occupancies) stay zero: not in telemetry.

        return np.clip(vec, 0.0, 1.0)

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "8883"))
MQTT_TLS_CA = os.getenv("MQTT_TLS_CA", "network/configs/certs/ca.crt")
MQTT_TLS_CERT = os.getenv("MQTT_TLS_CERT", "network/configs/certs/coordinator.crt")
MQTT_TLS_KEY = os.getenv("MQTT_TLS_KEY", "network/configs/certs/coordinator.key")
TOPOLOGY_FILE = os.getenv("TOPOLOGY_FILE", "network/configs/network_topology.json")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Topic patterns — loaded from the network contract (single source of truth).
# No hardcoded fallbacks: a missing contract file must abort startup rather
# than silently diverge from the Edge.
_DEFAULT_TOPICS_FILE = (
    Path(__file__).resolve().parents[3] / "network" / "schemas" / "topics.yaml"
)
TOPICS_FILE = os.getenv("TOPICS_FILE", str(_DEFAULT_TOPICS_FILE))


def _load_topics(path: str = TOPICS_FILE) -> dict[str, Any]:
    """Load the MQTT topic contract; fail loudly if it is missing/invalid."""
    contract_path = Path(path)
    if not contract_path.is_file():
        raise RuntimeError(
            f"MQTT topic contract file not found: {contract_path.resolve()}. "
            "The coordinator refuses to start with hardcoded topic fallbacks; "
            "see network/schemas/topics.yaml (override with TOPICS_FILE)."
        )
    with contract_path.open("r", encoding="utf-8") as fh:
        contract = yaml.safe_load(fh)
    channels = (contract or {}).get("channels")
    if not channels:
        raise RuntimeError(
            f"MQTT topic contract {contract_path.resolve()} has no 'channels' section."
        )
    for required in ("telemetry", "commands", "timing"):
        if required not in channels:
            raise RuntimeError(
                f"MQTT topic contract {contract_path.resolve()} is missing "
                f"required channel '{required}'."
            )
    return contract


TOPICS = _load_topics()
TOPIC_TELEMETRY = TOPICS["channels"]["telemetry"]["subscribe"]
TOPIC_COMMAND_FMT = TOPICS["channels"]["commands"]["pattern"]
TOPIC_TIMING_FMT = TOPICS["channels"]["timing"]["pattern"]

# Coordinator settings
PLAN_RECOMPUTE_INTERVAL_S = 60  # Recompute timing plans every N seconds
DEFAULT_CYCLE_LENGTH_S = 90
DEFAULT_DESIGN_SPEED_KMH = 50.0

logger = logging.getLogger("coordinator")


class SequenceCounter:
    """Thread-safe strictly increasing counter for outgoing payloads.

    A single counter shared across all topics keeps the per-node sequence
    strictly increasing (gaps are fine — the Edge only requires
    sequence_number > last_applied_seq).
    """

    def __init__(self, start: int = 1) -> None:
        self._next = start
        self._lock = threading.Lock()

    def next(self) -> int:
        with self._lock:
            value = self._next
            self._next += 1
            return value


class GreenWaveCoordinator:
    """
    Coordinates multiple traffic intersections to achieve green wave
    progression along corridors. Subscribes to intersection telemetry,
    computes optimal phase offsets, and publishes timing plans.
    """

    def __init__(self, topology_file: str = TOPOLOGY_FILE) -> None:
        self._running = False
        self._last_plan_time = 0.0

        # Seed from wall-clock milliseconds so a restarted coordinator never
        # reuses sequence numbers already applied by a long-lived Edge node.
        self._sequence = SequenceCounter(start=int(time.time() * 1000))

        # Telemetry cache: intersection_id -> latest telemetry dict
        self._telemetry: dict[str, dict[str, Any]] = {}

        # Load network topology
        self._graph_manager = NetworkGraphManager(topology_file)
        logger.info(
            "Loaded topology: %d intersections, %d roads",
            self._graph_manager.intersection_count,
            self._graph_manager.road_count,
        )

        # MQTT client setup
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="semaforo-coordinator",
            protocol=mqtt.MQTTv5,
        )
        
        if MQTT_TLS_CA and os.path.exists(MQTT_TLS_CA):
            import ssl
            logger.info("Enabling MQTT TLS with CA: %s", MQTT_TLS_CA)
            self._client.tls_set(
                ca_certs=MQTT_TLS_CA,
                certfile=MQTT_TLS_CERT if os.path.exists(MQTT_TLS_CERT) else None,
                keyfile=MQTT_TLS_KEY if os.path.exists(MQTT_TLS_KEY) else None,
                cert_reqs=ssl.CERT_REQUIRED
            )
            
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        self._client.on_disconnect = self._on_disconnect

        # Sub-services
        self._sync_service = SyncService(self._client)
        self._emergency_handler = EmergencyHandler(self._client, self._graph_manager)
        
        # Shared AI policy: a single InferenceSession serves every telemetry
        # message (and the shadow auditor).
        self._ai_policy = AIPolicyWrapper()

        # Auditoría Modo Sombra
        self._auditor = None
        if ShadowAuditor:
            self._auditor = ShadowAuditor(
                ai_policy=self._ai_policy,
                mqtt_broker=MQTT_HOST,
                mqtt_port=MQTT_PORT,
                log_dir="audit_logs"
            )
            logger.info("ShadowAuditor service initialized.")

    # -------------------------------------------------------------------------
    # MQTT Callbacks
    # -------------------------------------------------------------------------

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        rc: int,
        properties: Any = None,
    ) -> None:
        """Called when the coordinator connects to the MQTT broker."""
        logger.info("Connected to MQTT broker (rc=%s)", rc)

        # Subscribe to all intersection telemetry
        client.subscribe(TOPIC_TELEMETRY, qos=1)
        logger.info("Subscribed to %s", TOPIC_TELEMETRY)

        # Let sub-services set up their subscriptions
        self._emergency_handler.subscribe()

    def _on_message(
        self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
    ) -> None:
        """Process incoming MQTT messages."""
        try:
            # Canonical namespace: semaforo/<channel>/<node_id>
            topic_parts = msg.topic.split("/")
            if len(topic_parts) < 3:
                return

            channel = topic_parts[1]
            node_id = topic_parts[2]

            if channel == "telemetry":
                self._handle_telemetry(node_id, msg.payload)
            else:
                logger.debug("Unhandled channel: %s", channel)

        except Exception:
            logger.exception("Error processing message on topic %s", msg.topic)

    def _on_disconnect(
        self, client: mqtt.Client, userdata: Any, flags: Any, rc: int, properties: Any = None
    ) -> None:
        """Handle disconnection from the MQTT broker."""
        if rc != 0:
            logger.warning("Unexpected disconnection (rc=%s), will auto-reconnect", rc)
        else:
            logger.info("Disconnected from MQTT broker")

    # -------------------------------------------------------------------------
    # Telemetry Handling
    # -------------------------------------------------------------------------

    def _handle_telemetry(self, node_id: str, payload: bytes) -> None:
        """
        Process JSON telemetry from an Edge node and update internal state.
        """
        try:
            msg = json.loads(payload.decode("utf-8"))
            if not isinstance(msg, dict):
                raise ValueError(f"telemetry payload is not a JSON object: {type(msg).__name__}")

            intersection_id = msg.get("intersection_id", node_id)
            data = {
                "intersection_id": intersection_id,
                "phase": msg.get("phase", "UNKNOWN"),
                "vehicle_count": msg.get("total_vehicles", 0),
                "total_vehicles": msg.get("total_vehicles", 0),  # Alias for shadow auditor
                # The Edge does not emit flow_rate today (see topics.yaml);
                # tolerate its absence.
                "flow_rate": msg.get("flow_rate", 0.0),
            }

            self._telemetry[intersection_id] = data

            if self._auditor:
                self._auditor.on_telemetry_received(data)

            # --- TOMA DE DECISIONES Y PUBLICACIÓN DE COMANDO (MAPPO) ---
            recommended_phase = self._ai_policy.predict(data)

            # The telemetry topic carries the Edge's own node_id: reply directly.
            command_topic = TOPIC_COMMAND_FMT.format(node_id=node_id)
            self._publish_json(command_topic, self.build_phase_command(recommended_phase))

            logger.debug(
                "Telemetry from %s: phase=%s, vehicles=%d -> AI recommended phase: %s",
                intersection_id,
                data.get("phase", "unknown"),
                data.get("vehicle_count", 0),
                recommended_phase
            )
        except Exception as e:
            logger.warning(
                "Invalid telemetry from %s: %s", node_id, e
            )

    # -------------------------------------------------------------------------
    # Command Emission
    # -------------------------------------------------------------------------

    def build_phase_command(self, phase: str, priority: str = "HIGH") -> dict[str, Any]:
        """Build a CHANGE_PHASE command payload (network dispatch not included)."""
        return {
            "action": "CHANGE_PHASE",
            "phase": phase,
            "priority": priority,
            "timestamp": int(time.time() * 1000),
        }

    def _publish_json(
        self, topic: str, payload: dict[str, Any], qos: int = 1, retain: bool = False
    ) -> dict[str, Any]:
        """
        Single egress point for every JSON payload the coordinator emits.
        Injects a root-level strictly increasing sequence_number, then
        publishes. Returns the payload actually sent.
        """
        payload = dict(payload)
        payload["sequence_number"] = self._sequence.next()
        self._client.publish(topic, json.dumps(payload), qos=qos, retain=retain)
        return payload

    # -------------------------------------------------------------------------
    # Timing Plan Computation
    # -------------------------------------------------------------------------

    def _compute_and_publish_plans(self) -> None:
        """
        Compute coordinated timing plans for all corridors and publish them.
        Green wave offsets are calculated based on distance between
        intersections and the design speed.
        """
        corridors = self._graph_manager.identify_corridors()

        for corridor in corridors:
            plan = self._compute_corridor_plan(corridor)
            if plan:
                self._publish_timing_plan(plan)

        self._last_plan_time = time.time()
        logger.info("Published timing plans for %d corridors", len(corridors))

    def _compute_corridor_plan(
        self, corridor: list[str]
    ) -> dict[str, Any] | None:
        """
        Compute green wave timing for a single corridor.

        The offset for each intersection is calculated as:
            offset = distance_from_start / design_speed

        This ensures that a vehicle traveling at the design speed
        will arrive at each intersection just as the green phase begins.
        """
        if len(corridor) < 2:
            return None

        plan: dict[str, Any] = {
            "plan_id": f"plan-{int(time.time())}",
            "timestamp": int(time.time() * 1000),
            "zone_id": "zone-default",
            "cycle_length_s": DEFAULT_CYCLE_LENGTH_S,
            "corridors": [],
            "source": "coordinator",
        }

        corridor_data: dict[str, Any] = {
            "corridor_id": f"corridor-{corridor[0]}-{corridor[-1]}",
            "name": f"Corridor {corridor[0]} to {corridor[-1]}",
            "design_speed_kmh": DEFAULT_DESIGN_SPEED_KMH,
            "intersections": [],
        }

        cumulative_distance = 0.0
        design_speed_ms = DEFAULT_DESIGN_SPEED_KMH / 3.6  # Convert to m/s

        for i, int_id in enumerate(corridor):
            if i > 0:
                distance = self._graph_manager.get_edge_distance(
                    corridor[i - 1], int_id
                )
                cumulative_distance += distance

            offset_s = int(cumulative_distance / design_speed_ms) if design_speed_ms > 0 else 0

            corridor_data["intersections"].append(
                {
                    "intersection_id": int_id,
                    "offset_s": offset_s % DEFAULT_CYCLE_LENGTH_S,
                    "cycle_length_s": DEFAULT_CYCLE_LENGTH_S,
                }
            )

        plan["corridors"].append(corridor_data)
        return plan

    def _publish_timing_plan(self, plan: dict[str, Any]) -> None:
        """Publish a timing plan to the appropriate MQTT topic."""
        zone_id = plan.get("zone_id", "default")
        topic = TOPIC_TIMING_FMT.format(zone_id=zone_id)
        self._publish_json(topic, plan, qos=1, retain=True)
        logger.debug("Published timing plan %s to %s", plan["plan_id"], topic)

    # -------------------------------------------------------------------------
    # Main Loop
    # -------------------------------------------------------------------------

    def run(self) -> None:
        """Start the coordinator main loop."""
        self._running = True

        # Connect to MQTT broker
        logger.info("Connecting to MQTT broker at %s:%d", MQTT_HOST, MQTT_PORT)
        self._client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        self._client.loop_start()

        logger.info("Green Wave Coordinator started")

        try:
            while self._running:
                now = time.time()

                # Periodically recompute timing plans
                if now - self._last_plan_time >= PLAN_RECOMPUTE_INTERVAL_S:
                    if self._telemetry:
                        self._compute_and_publish_plans()

                # Publish sync pulse
                self._sync_service.tick()

                time.sleep(1.0)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        """Gracefully shut down the coordinator."""
        logger.info("Shutting down coordinator...")
        self._running = False
        
        if self._auditor:
            self._auditor.shutdown()
            
        self._client.loop_stop()
        self._client.disconnect()
        logger.info("Coordinator stopped")

    # -------------------------------------------------------------------------
    # Signal Handling
    # -------------------------------------------------------------------------

    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handle OS signals for graceful shutdown."""
        sig_name = signal.Signals(signum).name
        logger.info("Received signal %s, initiating shutdown...", sig_name)
        self._running = False


# =============================================================================
# Entry Point
# =============================================================================

def main() -> None:
    """Entry point for the coordinator service."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    coordinator = GreenWaveCoordinator()

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, coordinator.handle_signal)
    signal.signal(signal.SIGTERM, coordinator.handle_signal)

    coordinator.run()


if __name__ == "__main__":
    main()
