# =============================================================================
# Semáforo Inteligente — Mosquitto Dockerfile
# =============================================================================
# Custom Mosquitto image with project-specific configuration.
# =============================================================================

FROM eclipse-mosquitto:2

# Copy configuration files
COPY mosquitto.conf /mosquitto/config/mosquitto.conf
COPY acl.conf /mosquitto/config/acl.conf

# Create required directories
RUN mkdir -p /mosquitto/data /mosquitto/log

# Expose MQTT and WebSocket ports
EXPOSE 1883 9001

# Default command (uses config from /mosquitto/config/mosquitto.conf)
CMD ["mosquitto", "-c", "/mosquitto/config/mosquitto.conf"]
