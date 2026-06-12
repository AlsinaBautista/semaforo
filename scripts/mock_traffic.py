import time
import random
import paho.mqtt.client as mqtt
import telemetry_pb2

BROKER = "127.0.0.1"
PORT = 1883

def main():
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.connect(BROKER, PORT)
    client.loop_start()

    intersections = ["int-001", "int-002", "int-003"]
    phases = ["GREEN_NS", "YELLOW_NS", "ALL_RED", "GREEN_EW", "YELLOW_EW"]
    
    current_phase_idx = {i: 0 for i in intersections}
    phase_timer = {i: 0 for i in intersections}
    
    vehicles = {i: random.randint(5, 15) for i in intersections}

    print("🚥 Iniciando Simulador de Tráfico (Mock)... presiona Ctrl+C para detener.")

    try:
        while True:
            for node in intersections:
                # Actualizar fase (ciclo simple)
                phase_timer[node] += 1
                if phase_timer[node] > 10:  # Cambiar cada 10 segundos
                    current_phase_idx[node] = (current_phase_idx[node] + 1) % len(phases)
                    phase_timer[node] = 0
                
                # Simular llegada y salida de vehículos
                if phases[current_phase_idx[node]].startswith("GREEN"):
                    vehicles[node] = max(0, vehicles[node] - random.randint(1, 3))
                else:
                    vehicles[node] += random.randint(0, 2)
                    
                t = telemetry_pb2.Telemetry()
                t.intersection_id = node
                t.phase = phases[current_phase_idx[node]]
                t.vehicle_count = vehicles[node]
                t.pedestrian_count = random.randint(0, 5)
                t.flow_rate = random.uniform(1.0, 5.0) if vehicles[node] > 0 else 0.0

                topic = f"semaforo/{node}/telemetry"
                client.publish(topic, t.SerializeToString())
                print(f"[{node}] Publicado: {t.phase} | Vehículos: {t.vehicle_count}")

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo simulador...")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    main()
