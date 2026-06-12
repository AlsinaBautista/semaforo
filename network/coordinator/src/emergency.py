# =============================================================================
# Semáforo Inteligente — Emergency Handler
# =============================================================================
"""
Handles emergency vehicle preemption for the traffic network.

When an emergency event is received, the handler:
  1. Computes the shortest path from the emergency origin to destination.
  2. Generates a sequence of override commands to clear the path.
  3. Publishes override commands (QoS 2) to each intersection along the route.
  4. Restores normal operation after the emergency vehicle passes.

Emergency Topic:
    semaforo/{intersection_id}/emergency  (QoS 2)

Command Topic:
    semaforo/commands/{node_id}           (QoS 2 for emergency overrides)
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import paho.mqtt.client as mqtt

from .graph_manager import NetworkGraphManager

logger = logging.getLogger("coordinator.emergency")

# MQTT Topics
TOPIC_EMERGENCY = "semaforo/+/emergency"
TOPIC_COMMAND_FMT = "semaforo/commands/{intersection_id}"

# Emergency preemption settings
PREEMPTION_DURATION_S = 30  # Duration of emergency override per intersection
CLEAR_PHASE = "ALL_RED"  # Phase to set on cross streets
GREEN_PHASE_PREFIX = "EMERGENCY_GREEN"  # Phase for emergency vehicle direction


class EmergencyHandler:
    """
    Manages emergency vehicle preemption across the traffic network.

    Subscribes to emergency events (QoS 2 for guaranteed delivery) and
    computes path-clearing sequences using the network topology.
    """

    def __init__(
        self,
        client: mqtt.Client,
        graph_manager: NetworkGraphManager,
    ) -> None:
        self._client = client
        self._graph_manager = graph_manager

        # Track active emergencies: emergency_id -> affected intersections
        self._active_emergencies: dict[str, list[str]] = {}

        logger.info("EmergencyHandler initialized")

    def subscribe(self) -> None:
        """Subscribe to emergency topics with QoS 2."""
        self._client.subscribe(TOPIC_EMERGENCY, qos=2)
        self._client.message_callback_add(TOPIC_EMERGENCY, self._on_emergency)
        logger.info("Subscribed to emergency topics (QoS 2)")

    # -------------------------------------------------------------------------
    # Emergency Processing
    # -------------------------------------------------------------------------

    def _on_emergency(
        self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage
    ) -> None:
        """
        Handle incoming emergency messages.

        Expected payload:
        {
            "emergency_id": "em-001",
            "vehicle_type": "ambulance",
            "origin_intersection": "int-001",
            "destination_intersection": "int-003",
            "direction": "north",
            "eta_s": 45
        }
        """
        try:
            data = json.loads(msg.payload.decode("utf-8"))
            emergency_id = data.get("emergency_id", f"em-{int(time.time())}")

            logger.warning(
                "🚨 Emergency received: %s (type=%s, from=%s to=%s)",
                emergency_id,
                data.get("vehicle_type", "unknown"),
                data.get("origin_intersection", "unknown"),
                data.get("destination_intersection", "unknown"),
            )

            self._process_emergency(emergency_id, data)

        except json.JSONDecodeError:
            logger.error("Invalid JSON in emergency message")
        except Exception:
            logger.exception("Error processing emergency message")

    def _process_emergency(
        self, emergency_id: str, data: dict[str, Any]
    ) -> None:
        """
        Process an emergency event: compute path and publish override commands.
        """
        origin = data.get("origin_intersection")
        destination = data.get("destination_intersection")
        direction = data.get("direction", "north")

        if not origin or not destination:
            logger.error(
                "Emergency %s missing origin or destination", emergency_id
            )
            return

        # Compute shortest path through the network
        path = self._graph_manager.shortest_path(origin, destination)

        if not path:
            logger.error(
                "No path found for emergency %s from %s to %s",
                emergency_id,
                origin,
                destination,
            )
            return

        logger.info(
            "Emergency %s: clearing path %s", emergency_id, " -> ".join(path)
        )

        # Track this emergency
        self._active_emergencies[emergency_id] = path

        # Publish override commands for each intersection along the path
        self._clear_path(emergency_id, path, direction)

    def _clear_path(
        self, emergency_id: str, path: list[str], direction: str
    ) -> None:
        """
        Publish override commands to clear a path for an emergency vehicle.

        Commands are published with staggered timing based on the estimated
        travel time between intersections.
        """
        cumulative_delay = 0.0

        for i, intersection_id in enumerate(path):
            # Compute estimated delay based on distance from previous
            if i > 0:
                distance = self._graph_manager.get_edge_distance(
                    path[i - 1], intersection_id
                )
                speed_limit = self._graph_manager.get_edge_speed_limit(
                    path[i - 1], intersection_id
                )
                speed_ms = speed_limit / 3.6
                travel_time = distance / speed_ms if speed_ms > 0 else 0
                cumulative_delay += travel_time

            command: dict[str, Any] = {
                "timestamp": int((time.time() + cumulative_delay) * 1000),
                "action": "override",
                "phase": f"{GREEN_PHASE_PREFIX}_{direction.upper()}",
                "duration_s": PREEMPTION_DURATION_S,
                "priority": "emergency",
                "plan_id": f"emergency-{emergency_id}",
            }

            topic = TOPIC_COMMAND_FMT.format(intersection_id=intersection_id)
            payload = json.dumps(command)

            # Publish with QoS 2 for guaranteed delivery
            self._client.publish(topic, payload, qos=2)

            logger.info(
                "Emergency override: %s -> %s (delay=%.1fs)",
                intersection_id,
                command["phase"],
                cumulative_delay,
            )

    # -------------------------------------------------------------------------
    # Recovery
    # -------------------------------------------------------------------------

    def clear_emergency(self, emergency_id: str) -> None:
        """
        Clear an active emergency and restore normal operation.

        Publishes 'restore' commands to all affected intersections.
        """
        path = self._active_emergencies.pop(emergency_id, None)

        if not path:
            logger.warning("Emergency %s not found in active list", emergency_id)
            return

        for intersection_id in path:
            command: dict[str, Any] = {
                "timestamp": int(time.time() * 1000),
                "action": "set_phase",
                "phase": "RESTORE_NORMAL",
                "duration_s": 0,
                "priority": "high",
                "plan_id": f"recovery-{emergency_id}",
            }

            topic = TOPIC_COMMAND_FMT.format(intersection_id=intersection_id)
            payload = json.dumps(command)
            self._client.publish(topic, payload, qos=1)

        logger.info(
            "Emergency %s cleared, %d intersections restored",
            emergency_id,
            len(path),
        )

    @property
    def active_emergencies(self) -> dict[str, list[str]]:
        """Get currently active emergencies and their affected paths."""
        return dict(self._active_emergencies)
