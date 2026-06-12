import time
import csv
import json
import os
import paho.mqtt.client as mqtt

class ShadowAuditor:
    def __init__(self, ai_policy, mqtt_broker="localhost", mqtt_port=1883, log_dir="audit_logs"):
        """
        Inicializa el Auditor en Modo Sombra.
        
        Args:
            ai_policy: El objeto del agente de RL cargado que tiene un método predict().
            mqtt_broker: Dirección del broker MQTT.
            mqtt_port: Puerto del broker MQTT.
            log_dir: Directorio donde se guardarán los CSV.
        """
        self.ai_policy = ai_policy
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.log_dir = log_dir
        
        os.makedirs(self.log_dir, exist_ok=True)
        self.csv_path = os.path.join(self.log_dir, f"shadow_audit_{int(time.time())}.csv")
        
        # Inicializar archivo CSV con cabeceras
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "intersection_id", "actual_phase", 
                "ai_phase", "total_queue", "simulated_saved_seconds"
            ])
            
        # Configurar cliente MQTT para publicar métricas agregadas
        self.mqtt_client = mqtt.Client(client_id=f"shadow_auditor_{int(time.time())}")
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            print(f"[ShadowAuditor] Conectado a MQTT {self.mqtt_broker}:{self.mqtt_port}")
        except Exception as e:
            print(f"[ShadowAuditor] Advertencia: No se pudo conectar a MQTT: {e}")
            self.mqtt_client = None

        self.aggregated_saved_time = 0.0

    def on_telemetry_received(self, telemetry_payload):
        """
        Procesa un payload de telemetría proveniente del Edge.
        """
        intersection_id = telemetry_payload.get('intersection_id', 'unknown')
        actual_phase = telemetry_payload.get('phase')
        total_queue = telemetry_payload.get('total_vehicles', 0)
        
        # Consultar la política del agente RL
        try:
            # Asumimos que ai_policy.predict() toma la observación y devuelve un string o int de fase
            ai_recommended_phase = self.ai_policy.predict(telemetry_payload)
        except Exception as e:
            print(f"[ShadowAuditor] Error en predicción de IA: {e}")
            ai_recommended_phase = "ERROR"

        saved_wait_time = 0.0
        # Lógica heurística: Si el semáforo está en rojo para la avenida (ej. GREEN_EW) y la IA
        # pide verde (ej. GREEN_NS) y hay cola en la avenida, se asume que ahorramos
        # N segundos por vehículo esperando.
        if str(actual_phase) != str(ai_recommended_phase):
            # Asumimos un ahorro promedio de 2 segundos por vehículo por cada ciclo de diferencia
            saved_wait_time = total_queue * 2.0
            
        self.aggregated_saved_time += saved_wait_time
        
        # 1. Escribir datos detallados en CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                time.time(), intersection_id, actual_phase, 
                ai_recommended_phase, total_queue, saved_wait_time
            ])
            
        # 2. Publicar métricas agregadas en MQTT para el Dashboard de Flutter
        if self.mqtt_client:
            audit_topic = f"semaforo/audit/{intersection_id}"
            audit_payload = {
                "timestamp": time.time(),
                "intersection_id": intersection_id,
                "actual_phase": actual_phase,
                "ai_phase": ai_recommended_phase,
                "cycle_saved_seconds": saved_wait_time,
                "total_saved_seconds": self.aggregated_saved_time
            }
            self.mqtt_client.publish(audit_topic, json.dumps(audit_payload))
            
    def shutdown(self):
        """Limpia los recursos."""
        if self.mqtt_client:
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
        print(f"[ShadowAuditor] Finalizado. Reporte CSV guardado en {self.csv_path}")
