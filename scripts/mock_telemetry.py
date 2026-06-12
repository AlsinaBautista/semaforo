import time
import random
import paho.mqtt.client as mqtt
import telemetry_pb2

# Configuration
BROKER = 'localhost'
PORT = 8883
TOPICS = ['semaforo/int-001/telemetry', 'semaforo/int-002/telemetry', 'semaforo/int-003/telemetry']

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"Connected to Mosquitto MQTT broker at {BROKER}:{PORT}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

import ssl
client.tls_set(
    ca_certs="../network/configs/certs/ca.crt",
    certfile="../network/configs/certs/coordinator.crt",
    keyfile="../network/configs/certs/coordinator.key",
    cert_reqs=ssl.CERT_REQUIRED
)

client.on_connect = on_connect

client.connect(BROKER, PORT, 60)
client.loop_start()

try:
    print("Publishing mock telemetry data... Press Ctrl+C to stop.")
    while True:
        for i, topic in enumerate(TOPICS):
            # Create a mock Telemetry protobuf
            t = telemetry_pb2.Telemetry()
            t.timestamp = int(time.time() * 1000)
            t.intersection_id = f"int-00{i+1}"
            t.phase = random.choice(["NS_GREEN", "EW_GREEN", "ALL_RED", "NS_YELLOW", "EW_YELLOW"])
            t.phase_elapsed_s = random.randint(1, 30)
            t.flow_rate = random.uniform(10.0, 60.0)
            t.vehicle_count = random.randint(5, 50)
            t.pedestrian_count = random.randint(0, 15)
            
            # Serialize to string/bytes
            payload = t.SerializeToString()
            client.publish(topic, payload)
            
            # 10% chance to simulate a camera timeout (emergency)
            is_offline = random.random() < 0.1
            status_topic = f"semaforo/status/int-00{i+1}"
            status_payload = f'{{"status":"{"offline" if is_offline else "online"}","reason":"camera_timeout","node_id":"int-00{i+1}"}}'
            client.publish(status_topic, status_payload, qos=1, retain=True)

            # Publish audit metrics (Tiempo Ahorrado)
            import json
            audit_topic = f"semaforo/audit/int-00{i+1}"
            # Random wait time saved between 0 and 5 seconds per cycle
            audit_payload = json.dumps({
                "intersection_id": f"int-00{i+1}",
                "wait_time_saved_s": random.uniform(0.0, 5.0)
            })
            client.publish(audit_topic, audit_payload)

            print(f"Published to {topic} -> Phase: {t.phase}, Flow: {t.flow_rate:.1f}, Vehicles: {t.vehicle_count}")
        
        time.sleep(2)
except KeyboardInterrupt:
    print("\nStopping publisher.")
finally:
    client.loop_stop()
    client.disconnect()
