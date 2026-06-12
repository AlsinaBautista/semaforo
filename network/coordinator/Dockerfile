# =============================================================================
# Semáforo Inteligente — Coordinator Dockerfile
# =============================================================================
# Builds the Green Wave Coordinator service.
# =============================================================================

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Environment defaults
ENV MQTT_HOST=mosquitto
ENV MQTT_PORT=1883
ENV TOPOLOGY_FILE=/app/configs/network_topology.json
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import socket; s=socket.create_connection(('${MQTT_HOST}', int('${MQTT_PORT}')), timeout=5); s.close()" || exit 1

# Run the coordinator
CMD ["python", "-m", "src.coordinator"]
