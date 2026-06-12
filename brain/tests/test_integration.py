"""
Integration Test: SUMO -> TraCI -> MQTT

Runs a small headless SUMO simulation (similar to run_green_wave.py)
and verifies that Telemetry protobuf messages are actually published to Mosquitto.
"""

import sys
import time
import subprocess
import threading
from pathlib import Path

import pytest
import paho.mqtt.client as mqtt

# Add scripts directory for protobuf
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import telemetry_pb2

MQTT_BROKER = "localhost"
MQTT_PORT = 1883

class TestGreenWaveIntegration:
    def test_sumo_mqtt_pipeline(self):
        """Run the simulation script and capture MQTT telemetry."""
        
        received_messages = []
        
        def on_message(client, userdata, msg):
            try:
                t = telemetry_pb2.Telemetry()
                t.ParseFromString(msg.payload)
                received_messages.append(t)
            except Exception as e:
                pass

        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
        client.on_message = on_message
        
        try:
            client.connect(MQTT_BROKER, MQTT_PORT, 5)
            client.subscribe("semaforo/+/telemetry")
            client.loop_start()
        except Exception as e:
            pytest.skip(f"MQTT broker not available: {e}")

        # Start the green wave script as a subprocess
        script_path = PROJECT_ROOT / "brain" / "src" / "training" / "run_green_wave.py"
        
        # Run it for a few seconds
        proc = subprocess.Popen(
            [sys.executable, str(script_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={"PYTHONPATH": str(PROJECT_ROOT)}
        )
        
        # Wait 10 seconds to allow SUMO to start and publish a few messages
        time.sleep(10)
        
        proc.terminate()
        proc.wait(timeout=5)
        
        client.loop_stop()
        client.disconnect()
        
        # We should have received multiple telemetry messages from c0, c1, c2
        assert len(received_messages) > 0, "No telemetry messages received from Mosquitto"
        
        # Verify the content of the messages
        intersections = set(m.intersection_id for m in received_messages)
        assert "int-001" in intersections
        assert "int-002" in intersections
        assert "int-003" in intersections
        
        # Verify fields are populated
        msg = received_messages[0]
        assert msg.timestamp > 0
        assert hasattr(msg, "phase")
        assert msg.phase != ""
        assert hasattr(msg, "flow_rate")
        assert hasattr(msg, "vehicle_count")
