"""
Protobuf roundtrip tests — encode and decode all message types.

Ensures that our protobuf definitions correctly serialize and
deserialize, catching schema drift early.
"""

import sys
import time
from pathlib import Path

import pytest

# Add scripts/ to path for generated protobuf modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import telemetry_pb2


class TestTelemetryProto:
    """Test Telemetry protobuf roundtrip serialization."""

    def test_basic_roundtrip(self):
        """Encode a Telemetry message to bytes and decode it back."""
        original = telemetry_pb2.Telemetry()
        original.timestamp = 1717891200000
        original.intersection_id = "int-001"
        original.phase = "NS_GREEN"
        original.phase_elapsed_s = 15
        original.flow_rate = 42.5
        original.vehicle_count = 23
        original.pedestrian_count = 5

        serialized = original.SerializeToString()
        assert len(serialized) > 0

        decoded = telemetry_pb2.Telemetry()
        decoded.ParseFromString(serialized)

        assert decoded.timestamp == 1717891200000
        assert decoded.intersection_id == "int-001"
        assert decoded.phase == "NS_GREEN"
        assert decoded.phase_elapsed_s == 15
        assert abs(decoded.flow_rate - 42.5) < 0.01
        assert decoded.vehicle_count == 23
        assert decoded.pedestrian_count == 5

    def test_empty_message(self):
        """An empty Telemetry message should serialize and deserialize."""
        original = telemetry_pb2.Telemetry()
        serialized = original.SerializeToString()

        decoded = telemetry_pb2.Telemetry()
        decoded.ParseFromString(serialized)

        assert decoded.timestamp == 0
        assert decoded.intersection_id == ""
        assert decoded.phase == ""
        assert decoded.vehicle_count == 0

    def test_all_phases(self):
        """All expected phase names should roundtrip correctly."""
        phases = ["NS_GREEN", "NS_YELLOW", "EW_GREEN", "EW_YELLOW", "ALL_RED", "YELLOW", "GREEN", "UNKNOWN"]
        for phase in phases:
            t = telemetry_pb2.Telemetry()
            t.phase = phase
            decoded = telemetry_pb2.Telemetry()
            decoded.ParseFromString(t.SerializeToString())
            assert decoded.phase == phase, f"Phase '{phase}' did not roundtrip"

    def test_large_vehicle_count(self):
        """Edge case: very high vehicle counts."""
        t = telemetry_pb2.Telemetry()
        t.vehicle_count = 999999
        t.pedestrian_count = 50000
        decoded = telemetry_pb2.Telemetry()
        decoded.ParseFromString(t.SerializeToString())
        assert decoded.vehicle_count == 999999
        assert decoded.pedestrian_count == 50000

    def test_negative_flow_rate(self):
        """Flow rate should handle zero and near-zero values."""
        t = telemetry_pb2.Telemetry()
        t.flow_rate = 0.0
        decoded = telemetry_pb2.Telemetry()
        decoded.ParseFromString(t.SerializeToString())
        assert decoded.flow_rate == 0.0

    def test_unicode_intersection_id(self):
        """Intersection IDs with special characters should roundtrip."""
        t = telemetry_pb2.Telemetry()
        t.intersection_id = "int-ñ-001-café"
        decoded = telemetry_pb2.Telemetry()
        decoded.ParseFromString(t.SerializeToString())
        assert decoded.intersection_id == "int-ñ-001-café"

    def test_timestamp_current(self):
        """Timestamp with current epoch millis should roundtrip."""
        now_ms = int(time.time() * 1000)
        t = telemetry_pb2.Telemetry()
        t.timestamp = now_ms
        decoded = telemetry_pb2.Telemetry()
        decoded.ParseFromString(t.SerializeToString())
        assert decoded.timestamp == now_ms

    def test_multiple_messages_independent(self):
        """Two separately serialized messages should not interfere."""
        t1 = telemetry_pb2.Telemetry()
        t1.intersection_id = "int-001"
        t1.vehicle_count = 10

        t2 = telemetry_pb2.Telemetry()
        t2.intersection_id = "int-002"
        t2.vehicle_count = 99

        b1 = t1.SerializeToString()
        b2 = t2.SerializeToString()

        d1 = telemetry_pb2.Telemetry()
        d1.ParseFromString(b1)
        d2 = telemetry_pb2.Telemetry()
        d2.ParseFromString(b2)

        assert d1.intersection_id == "int-001"
        assert d1.vehicle_count == 10
        assert d2.intersection_id == "int-002"
        assert d2.vehicle_count == 99

    def test_binary_not_valid_json(self):
        """Protobuf binary should NOT be valid UTF-8 JSON (regression for coordinator bug)."""
        t = telemetry_pb2.Telemetry()
        t.timestamp = 1717891200000
        t.intersection_id = "int-001"
        t.phase = "NS_GREEN"
        t.vehicle_count = 42
        serialized = t.SerializeToString()

        # This should NOT be parseable as JSON
        import json
        with pytest.raises((json.JSONDecodeError, UnicodeDecodeError)):
            json.loads(serialized.decode("utf-8"))
