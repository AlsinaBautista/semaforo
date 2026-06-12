# =============================================================================
# Semáforo Inteligente — Tests for EmergencyHandler topic contract (B-3)
# =============================================================================
"""
Verifies that EmergencyHandler publishes override commands to the canonical
topic pattern  semaforo/commands/{node_id}  defined in network/schemas/topics.yaml,
and NOT to the old semaforo/{id}/command pattern.
"""
from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, call, patch

import pytest

from src.emergency import EmergencyHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_graph_manager(path: list[str]) -> MagicMock:
    """Stub GraphManager that returns a fixed path and trivial distances."""
    gm = MagicMock()
    gm.shortest_path.return_value = path
    gm.get_edge_distance.return_value = 100.0
    gm.get_edge_speed_limit.return_value = 50.0
    return gm


def _make_client() -> MagicMock:
    return MagicMock()


def _trigger_emergency(handler: EmergencyHandler, data: dict) -> None:
    """Simulate an incoming MQTT emergency message."""
    msg = MagicMock()
    msg.payload = json.dumps(data).encode()
    handler._on_emergency(MagicMock(), None, msg)


# ---------------------------------------------------------------------------
# B-3: topic contract tests
# ---------------------------------------------------------------------------

class TestEmergencyTopicContract:
    """All publish() calls must target semaforo/commands/<node_id>."""

    def test_override_topic_matches_contract(self):
        path = ["int-001", "int-002", "int-003"]
        client = _make_client()
        handler = EmergencyHandler(client, _make_graph_manager(path))

        _trigger_emergency(
            handler,
            {
                "emergency_id": "em-test-01",
                "vehicle_type": "ambulance",
                "origin_intersection": "int-001",
                "destination_intersection": "int-003",
                "direction": "north",
            },
        )

        published_topics = [c.args[0] for c in client.publish.call_args_list]
        for node_id, topic in zip(path, published_topics):
            expected = f"semaforo/commands/{node_id}"
            assert topic == expected, (
                f"B-3: expected '{expected}', got '{topic}' — "
                "old semaforo/{{id}}/command pattern still in use"
            )

    def test_no_legacy_pattern_published(self):
        path = ["int-A", "int-B"]
        client = _make_client()
        handler = EmergencyHandler(client, _make_graph_manager(path))

        _trigger_emergency(
            handler,
            {
                "emergency_id": "em-test-02",
                "origin_intersection": "int-A",
                "destination_intersection": "int-B",
                "direction": "south",
            },
        )

        for c in client.publish.call_args_list:
            topic = c.args[0]
            assert not topic.endswith("/command"), (
                f"B-3: legacy topic pattern detected: '{topic}'"
            )
            assert topic.startswith("semaforo/commands/"), (
                f"B-3: topic does not match contract: '{topic}'"
            )

    def test_recovery_topic_matches_contract(self):
        path = ["int-X"]
        client = _make_client()
        handler = EmergencyHandler(client, _make_graph_manager(path))

        _trigger_emergency(
            handler,
            {
                "emergency_id": "em-rec-01",
                "origin_intersection": "int-X",
                "destination_intersection": "int-X",
                "direction": "east",
            },
        )
        client.publish.reset_mock()

        handler.clear_emergency("em-rec-01")

        for c in client.publish.call_args_list:
            topic = c.args[0]
            assert topic == "semaforo/commands/int-X", (
                f"B-3 recovery: unexpected topic '{topic}'"
            )
