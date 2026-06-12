import sys
import os
import time
import json
import paho.mqtt.client as mqtt

# Añadir el path a los protos generados para serializar/deserializar
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
import telemetry_pb2
import command_pb2

# Configuración del entorno de pruebas
BROKER = "127.0.0.1"
PORT = 1883 # Puerto sin TLS para inyección directa desde entorno local
NODE_ID = "int-001"
EDGE_ID = "edge-001"
TELEMETRY_TOPIC = f"semaforo/{NODE_ID}/telemetry"
COMMAND_TOPIC = f"semaforo/commands/{EDGE_ID}"
STATUS_TOPIC = f"semaforo/status/{EDGE_ID}"

received_command = None
emergency_triggered = False

def on_message(client, userdata, msg):
    """
    Callback para recibir las respuestas del Brain o del nodo Edge.
    """
    global received_command, emergency_triggered
    if msg.topic == COMMAND_TOPIC:
        try:
            cmd = json.loads(msg.payload.decode())
            received_command = cmd
        except Exception as e:
            pass
    elif msg.topic == STATUS_TOPIC:
        try:
            status_data = json.loads(msg.payload.decode())
            if status_data.get("status") == "offline" or status_data.get("reason") in ["emergency", "safety_violation", "camera_timeout"]:
                emergency_triggered = True
        except Exception as e:
            pass

def run_tests():
    global received_command, emergency_triggered
    
    print("==========================================================")
    print("🚦 QA Test Suite: MAPPO Decisions & SafetyLayer")
    print("==========================================================")

    # 1. Conexión al Broker MQTT
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    
    try:
        client.connect(BROKER, PORT, 60)
    except ConnectionRefusedError:
        print("[ERROR] No se pudo conectar a Mosquitto en localhost:1883. ¿Está corriendo el Docker?")
        sys.exit(1)
        
    client.subscribe(COMMAND_TOPIC)
    client.subscribe(STATUS_TOPIC)
    client.loop_start()
    time.sleep(1) # Dar tiempo a que la suscripción asiente

    # ==========================================================
    # CASO A: Test de Decisión del RL (Avenida vacía, Transversal llena)
    # ==========================================================
    print("\n[TEST A] Inyectando telemetría sintética (Avenida vacía, Transversal llena)...")
    
    t = telemetry_pb2.Telemetry()
    t.timestamp = int(time.time() * 1000)
    t.intersection_id = NODE_ID
    t.phase = "NS_GREEN" # Actual: Avenida con verde
    t.phase_elapsed_s = 20 # Tiempo de verde ya gastado
    
    # Avenida (Norte-Sur) vacía
    t.queues["north"].count = 0
    t.queues["south"].count = 0
    
    # Transversal (Este-Oeste) muy congestionada
    t.queues["east"].count = 25
    t.queues["west"].count = 20
    
    t.vehicle_count = 45
    
    # Publicamos la telemetría mock para que el Brain la capture
    client.publish(TELEMETRY_TOPIC, t.SerializeToString())
    print("   -> Publicado en", TELEMETRY_TOPIC)
    print("   -> Esperando decisión del Brain (MAPPO)...")
    
    timeout = 5.0
    start_time = time.time()
    while received_command is None and (time.time() - start_time) < timeout:
        time.sleep(0.1)
        
    if received_command:
        print(f"   [PASS] Comando interceptado. El cerebro dictamina:")
        print(f"          Action: {received_command.get('action')} | Target Phase: {received_command.get('phase')}")
        if received_command.get("phase") == "GREEN_EW":
            print("          Evaluación: CORRECTO (Privilegió a la vía transversal)")
        else:
            print("          Evaluación: ADVERTENCIA (La política MAPPO no eligió la vía óptima)")
    else:
        print("   [FAIL] Timeout. No hubo respuesta del Brain. (Verifica si el coordinador de Python está ejecutándose)")

    # ==========================================================
    # CASO B: Test de Atentado / Seguridad en C++ (SafetyLayer)
    # ==========================================================
    print("\n[TEST B] Evaluando C++ SafetyLayer: Inyectando Comando Malicioso (ALL_GREEN)...")
    received_command = None
    emergency_triggered = False

    cmd = {
        "timestamp": int(time.time() * 1000),
        "action": "CHANGE_PHASE",
        "phase": "ALL_GREEN", # FASE PROHIBIDA (Conflicto de choques)
        "priority": "HIGH"
    }
    
    client.publish(COMMAND_TOPIC, json.dumps(cmd))
    print("   -> Publicado paquete malicioso en", COMMAND_TOPIC)
    print("   -> Esperando reacción del Edge node (SafetyLayer)...")
    
    timeout = 3.0
    start_time = time.time()
    while not emergency_triggered and (time.time() - start_time) < timeout:
        time.sleep(0.1)
        
    if emergency_triggered:
        print("   [PASS] La SafetyLayer en C++ interceptó el paquete prohibido y forzó Degradación de Gracia (Emergency Mode).")
    else:
        print("   [FAIL] Timeout. El nodo Edge no reportó estado de emergencia. (Verifica si el backend de C++ está corriendo y validando).")

    # Limpieza
    client.loop_stop()
    client.disconnect()
    print("\n==========================================================")
    print("🏁 QA Suite Finalizada.")
    print("==========================================================")

if __name__ == "__main__":
    run_tests()
