# 🚦 Semáforo Inteligente

> Sistema de control de tráfico inteligente con aprendizaje por refuerzo, visión por computadora y coordinación en red para intersecciones urbanas.

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![C++](https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white)
![Flutter](https://img.shields.io/badge/Flutter-3.x-02569B?logo=flutter&logoColor=white)
![MQTT](https://img.shields.io/badge/MQTT-Mosquitto-660066?logo=eclipse-mosquitto&logoColor=white)
![SUMO](https://img.shields.io/badge/SUMO-Simulation-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📖 Descripción

Semáforo Inteligente es una plataforma integral para la gestión y optimización del tráfico urbano. Combina simulación microscópica de tráfico (SUMO), aprendizaje por refuerzo profundo, detección de vehículos y peatones en tiempo real con visión por computadora, y una interfaz de monitoreo en Flutter — todo conectado mediante una red MQTT para coordinación distribuida entre intersecciones.

---

## 🏗️ Arquitectura

El sistema se compone de cuatro módulos principales que se comunican a través de un broker MQTT central:

```
┌─────────────┐    MQTT     ┌─────────────┐    MQTT     ┌──────────────┐
│  Edge (C++) │◄──────────►│  Network    │◄──────────►│  Brain (Py)  │
│  Detección  │            │  Mosquitto  │            │  RL + SUMO   │
│  Jetson     │            │  Coordinator│            │  Entrenamiento│
└─────────────┘            └──────┬──────┘            └──────────────┘
                                  │
                                  │ WebSocket
                                  ▼
                           ┌──────────────┐
                           │  Dashboard   │
                           │  Flutter Web │
                           └──────────────┘
```

### Flujo de datos

1. **Edge** captura video, detecta vehículos/peatones y publica telemetría.
2. **Brain** entrena agentes RL en SUMO y publica comandos de fase óptimos.
3. **Network** coordina múltiples intersecciones (ola verde, emergencias, sincronización).
4. **Dashboard** visualiza el estado en tiempo real vía WebSocket.

---

## 📦 Módulos

### 🔌 Edge — Detección en Borde (C++)

Módulo de inferencia en tiempo real desplegado en NVIDIA Jetson. Ejecuta detección de objetos (YOLOv8) y estimación de densidad vehicular. Publica datos de telemetría al broker MQTT.

- **Ubicación:** `edge/`
- **Lenguaje:** C++17 con TensorRT
- **Hardware:** NVIDIA Jetson Orin Nano

### 🧠 Brain — Aprendizaje por Refuerzo (Python)

Motor de entrenamiento y evaluación de agentes RL para optimización de fases semafóricas. Utiliza SUMO como entorno de simulación microscópica de tráfico.

- **Ubicación:** `brain/`
- **Lenguaje:** Python 3.10+
- **Frameworks:** Gymnasium, Stable-Baselines3, TraCI (SUMO)

### 🌐 Network — Comunicación MQTT

Infraestructura de mensajería y coordinación entre intersecciones. Incluye el broker Mosquitto, el coordinador de ola verde y el servicio de sincronización.

- **Ubicación:** `network/`
- **Broker:** Eclipse Mosquitto 2
- **Coordinador:** Python (paho-mqtt, NetworkX)

### 📊 Dashboard — Interfaz de Monitoreo (Flutter)

Panel de control web/móvil para visualización en tiempo real del estado del tráfico, métricas de rendimiento y control manual de intersecciones.

- **Ubicación:** `dashboard/`
- **Framework:** Flutter 3.x (Web + Mobile)

---

## 🛠️ Tech Stack

| Capa         | Tecnología                                    |
|--------------|-----------------------------------------------|
| Simulación   | SUMO (Simulation of Urban Mobility)           |
| RL           | Stable-Baselines3, Gymnasium                  |
| Detección    | YOLOv8, TensorRT, OpenCV                      |
| Mensajería   | Eclipse Mosquitto (MQTT 5.0)                  |
| Coordinación | Python, NetworkX, paho-mqtt                   |
| Dashboard    | Flutter, Dart, mqtt_client                    |
| Serialización| Protocol Buffers                              |
| Despliegue   | Docker, Docker Compose                        |
| Hardware     | NVIDIA Jetson Orin Nano                       |

---

## ✅ Prerrequisitos

- **SUMO** ≥ 1.18.0 — [Instalación](https://sumo.dlr.de/docs/Installing/index.html)
- **Python** ≥ 3.10
- **C++17** compatible compiler (GCC 9+ / Clang 10+)
- **Flutter** ≥ 3.0 — [Instalación](https://docs.flutter.dev/get-started/install)
- **Eclipse Mosquitto** ≥ 2.0 — [Instalación](https://mosquitto.org/download/)
- **Docker** y **Docker Compose** (para despliegue containerizado)
- **NVIDIA Jetson** (solo para módulo Edge en producción)

---

## 🚀 Quick Start

### 1. Configuración del entorno de desarrollo

```bash
# Clonar el repositorio
git clone https://github.com/becashub/semaforo.git
cd semaforo

# Ejecutar el script de configuración
chmod +x scripts/setup_dev.sh
./scripts/setup_dev.sh
```

### 2. Brain — Entrenamiento RL

```bash
cd brain
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Entrenar un agente
python -m src.training.train --episodes 1000
```

### 3. Network — Broker MQTT y Coordinador

```bash
# Con Docker Compose (recomendado)
docker compose up mosquitto coordinator

# O manualmente
mosquitto -c network/mosquitto/mosquitto.conf &
cd network/coordinator
pip install -e .
python -m src.coordinator
```

### 4. Edge — Detección (Jetson)

```bash
cd edge
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./semaforo_edge --config ../configs/edge_config.yaml
```

### 5. Dashboard — Interfaz Flutter

```bash
cd dashboard
flutter pub get
flutter run -d chrome   # Web
flutter run              # Mobile
```

### 6. Simulación completa

```bash
chmod +x scripts/run_simulation.sh
./scripts/run_simulation.sh
```

---

## 📁 Estructura del Proyecto

```
semaforo/
├── brain/                  # Módulo de RL y simulación
│   ├── src/
│   │   ├── environment/    # Entorno Gymnasium + SUMO
│   │   ├── agent/          # Agentes RL
│   │   ├── reward/         # Funciones de recompensa
│   │   └── training/       # Scripts de entrenamiento
│   ├── configs/            # Configuraciones SUMO
│   └── tests/
├── edge/                   # Módulo de detección en borde
│   ├── src/                # Código C++ (inferencia, MQTT)
│   ├── include/            # Headers
│   └── models/             # Modelos ONNX/TensorRT
├── network/                # Módulo de comunicación
│   ├── mosquitto/          # Configuración del broker
│   ├── coordinator/        # Coordinador de ola verde
│   └── configs/            # Topología de red
├── dashboard/              # Interfaz Flutter
│   └── lib/
├── proto/                  # Esquemas Protocol Buffers
├── docs/                   # Documentación
├── scripts/                # Scripts de utilidad
├── docker-compose.yml
└── README.md
```

---

## 📄 Licencia

Este proyecto está licenciado bajo la **MIT License**. Consulta el archivo [LICENSE](LICENSE) para más detalles.

---

## 👥 Contribuidores

Desarrollado por **Bautista Tomas Alsina**.

---

*Semáforo Inteligente — Optimizando el flujo urbano con inteligencia artificial.*
