import sys
import os
import time
import threading
import paho.mqtt.client as mqtt
import cv2
import numpy as np

try:
    import mss
except ImportError:
    print("[ERROR] Falta instalar mss: pip install mss")
    sys.exit(1)

# TraCI para comunicarnos con SUMO
try:
    import traci
except ImportError:
    print("[ERROR] SUMO TraCI no está instalado. pip install traci")
    sys.exit(1)

# Importar protobufs para deserializar telemetría de YOLO
sys.path.append(os.path.join(os.path.dirname(__file__), '../scripts'))
import telemetry_pb2

# Configuración
BROKER = "127.0.0.1"
PORT = 1883
NODE_ID = "int-001"
TELEMETRY_TOPIC = f"semaforo/{NODE_ID}/telemetry"

yolo_vehicle_count = 0
yolo_pedestrian_count = 0

def on_message(client, userdata, msg):
    """
    Actualiza la métrica global con lo que detecta YOLO desde el Edge en C++.
    """
    global yolo_vehicle_count, yolo_pedestrian_count
    if msg.topic == TELEMETRY_TOPIC:
        try:
            t = telemetry_pb2.Telemetry()
            t.ParseFromString(msg.payload)
            yolo_vehicle_count = t.vehicle_count
            yolo_pedestrian_count = t.pedestrian_count
        except Exception:
            pass

def mqtt_listener():
    """
    Corre en un hilo separado para mantener la suscripción MQTT activa.
    """
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    
    try:
        client.connect(BROKER, PORT, 60)
        client.subscribe(TELEMETRY_TOPIC)
        client.loop_forever()
    except Exception as e:
        print(f"[MQTT] Error al conectar: {e}")

def capture_and_stream():
    """
    Captura una porción de la pantalla donde se asume que corre SUMO GUI.
    Y la retransmite mediante un pipeline de GStreamer por UDP.
    El binario C++ de inferencia deberá estar escuchando en udpsrc port=5000.
    """
    print("[SIL-Video] Iniciando captura de pantalla para GStreamer UDP Sink...")
    sct = mss.MSS()
    
    # Definir la región de la pantalla (SUMO GUI). 
    # Ajustar estas coordenadas a tu entorno de escritorio.
    monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
    
    # Pipeline GStreamer (x264enc por UDP)
    gst_pipeline = (
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc tune=zerolatency bitrate=2000 speed-preset=ultrafast ! "
        "rtph264pay config-interval=1 pt=96 ! "
        "udpsink host=127.0.0.1 port=5000 sync=false"
    )
    
    out = cv2.VideoWriter(gst_pipeline, cv2.CAP_GSTREAMER, 0, 15.0, (800, 600), True)
    
    if not out.isOpened():
        print("[SIL-Video] WARN: OpenCV falló al abrir GStreamer. ¿Estás en Mac/Linux sin soporte GStreamer en cv2?")
        return

    while True:
        try:
            # Capturar región de la pantalla
            img = np.array(sct.grab(monitor))
            # Quitar canal alfa para enviarlo correctamente al pipeline
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            out.write(frame)
            time.sleep(1/15.0) # FPS Limiter
        except Exception:
            break

def run_sil_test():
    print("==========================================================")
    print("👁️ QA Test Suite: Software-in-the-Loop (SIL) - YOLO Vision")
    print("==========================================================")
    
    # 1. Hilo MQTT para escuchar a YOLO
    threading.Thread(target=mqtt_listener, daemon=True).start()
    
    # 2. Hilo de Captura de Pantalla para emular la cámara Edge
    threading.Thread(target=capture_and_stream, daemon=True).start()
    
    print("[SIL] Conectando a SUMO TraCI...")
    try:
        # SUMO debe haber sido lanzado con: sumo-gui --remote-port 8813
        traci.init(8813)
        traci_connected = True
        print("[SIL] Conectado a SUMO correctamente.")
    except traci.exceptions.FatalTraCIError:
        print("[SIL] [WARN] TraCI no responde en el puerto 8813.")
        print("[SIL] Para ver esta prueba funcionar, asegúrate de correr SUMO primero:")
        print("      sumo-gui -n network.net.xml --remote-port 8813")
        print("\n[SIL] Ejecutando simulación de datos ficticios para demostrar la estructura del test...")
        traci_connected = False

    errors = []
    
    print("\n--- INICIANDO LECTURAS ---")
    
    # Ejecutamos 30 pasos de simulación para recolectar un delta de error
    for step in range(30):
        if traci_connected:
            traci.simulationStep()
            # Obtenemos el "Ground Truth" (conteo real en la simulación)
            sumo_count = traci.vehicle.getIDCount()
        else:
            # Datos ficticios de prueba
            import random
            sumo_count = random.randint(15, 20)

        # Pequeña pausa para permitir que el frame llegue a C++, se procese por TensorRT/YOLO y vuelva por MQTT
        time.sleep(0.5) 
        
        # Calculamos el margen de error absoluto (Ground Truth vs Inferencia)
        error = abs(yolo_vehicle_count - sumo_count)
        errors.append(error)
        
        print(f"[Step {step:02d}] 🚗 SUMO (Real): {sumo_count:02d} | 👁️ YOLO (Visión): {yolo_vehicle_count:02d} | ⚠️ Delta Error: {error}")
        
    if traci_connected:
        traci.close()
        
    # Análisis Final (Mean Absolute Error o Mean Squared Error)
    mse = sum([e**2 for e in errors]) / len(errors)
    
    print("\n==========================================================")
    print("📊 RESULTADOS DE CERTIFICACIÓN SIL")
    print("==========================================================")
    print(f"Error Cuadrático Medio (MSE) de Inferencia: {mse:.2f}")
    
    if mse < 3.0:
        print("=> [PASS] 🟢 El modelo de visión es estable y preciso frente al simulador físico.")
    else:
        print("=> [FAIL] 🔴 Alta divergencia visual. Posibles causas:")
        print("           - Oclusión masiva de vehículos en la vista de cámara.")
        print("           - Latencia excesiva en el UDP stream o hardware saturado.")
        print("           - Umbrales de confianza (Confidence Thresholds) muy bajos/altos en detector.cpp.")

if __name__ == "__main__":
    run_sil_test()
