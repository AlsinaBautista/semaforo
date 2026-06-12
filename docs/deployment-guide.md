# 🚦 Semáforo Inteligente — Deployment Guide

> Guide for deploying the Semáforo Inteligente system in development and production environments.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Jetson Setup](#jetson-setup)
3. [Docker Deployment](#docker-deployment)
4. [Flutter Build](#flutter-build)
5. [Monitoring](#monitoring)

---

## Prerequisites

### Software Requirements

- **Docker** ≥ 24.0 and **Docker Compose** ≥ 2.20
- **Python** ≥ 3.10 (for brain and coordinator modules)
- **NVIDIA JetPack** ≥ 6.0 (for Jetson edge deployment)
- **Flutter** ≥ 3.0 (for dashboard)
- **SUMO** ≥ 1.18.0 (for simulation/training only)

### Hardware Requirements

| Component     | Development        | Production                 |
|---------------|--------------------|----------------------------|
| Brain         | Any x86_64 machine | GPU server (CUDA)          |
| Edge          | x86_64 (simulated) | NVIDIA Jetson Orin Nano    |
| Network       | Docker on laptop   | Cloud VM or on-premise     |
| Dashboard     | Browser (Chrome)   | Web server + CDN           |

---

## Jetson Setup

> *TODO: Complete this section with detailed Jetson deployment instructions.*

### Planned Content

- JetPack SDK installation
- TensorRT engine generation from ONNX models
- Camera configuration and CSI setup
- Edge service installation as systemd unit
- Network configuration for MQTT connectivity
- Over-the-air update mechanism

---

## Docker Deployment

### Development

```bash
# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Production

> *TODO: Complete this section with production Docker deployment.*

### Planned Content

- Docker Compose production override file
- TLS certificate configuration for Mosquitto
- Environment variable management
- Volume backup and restore procedures
- Resource limits and health checks
- Container registry setup

---

## Flutter Build

### Web Build

```bash
cd dashboard
flutter build web --release
```

### Planned Content

- Web deployment to CDN/static hosting
- Mobile build (Android APK / iOS IPA)
- Environment-specific MQTT endpoint configuration
- CI/CD pipeline for automated builds

---

## Monitoring

> *TODO: Complete this section with monitoring and observability setup.*

### Planned Content

- Mosquitto `$SYS` topic monitoring
- Prometheus metrics exporter for MQTT
- Grafana dashboard templates
- Alert configuration (PagerDuty / Slack)
- Log aggregation (Loki / ELK)
- Traffic KPI dashboards
- System health checks and watchdogs

---

*This guide is a work in progress. Sections marked with TODO will be completed as the project matures.*
