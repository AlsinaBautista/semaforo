# =============================================================================
# Semáforo Inteligente — Sync Service
# =============================================================================
"""
Publishes periodic synchronization pulses to keep all intersections
time-aligned for coordinated signal control.

The sync pulse is published on:
    semaforo/{zone}/sync/pulse

Each pulse contains a timestamp and cycle reference so that intersection
controllers can align their phase timing to the coordinator's clock.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import paho.mqtt.client as mqtt

logger = logging.getLogger("coordinator.sync")

# Sync pulse interval in seconds
SYNC_INTERVAL_S = 10.0

# Topic pattern for sync pulses
TOPIC_SYNC_FMT = "semaforo/{zone_id}/sync/pulse"

# Default zone
DEFAULT_ZONE = "zone-default"


class SyncService:
    """
    Publishes periodic synchronization pulses to keep intersection
    controllers time-aligned.

    The sync pulse contains:
      - timestamp: Current Unix time in milliseconds.
      - cycle_ref: Reference cycle number since coordinator start.
      - zone_id: The zone this pulse applies to.
    """

    def __init__(
        self,
        client: mqtt.Client,
        zone_id: str = DEFAULT_ZONE,
        interval_s: float = SYNC_INTERVAL_S,
    ) -> None:
        self._client = client
        self._zone_id = zone_id
        self._interval_s = interval_s
        self._last_pulse_time = 0.0
        self._cycle_counter = 0
        self._start_time = time.time()

        logger.info(
            "SyncService initialized (zone=%s, interval=%.1fs)",
            zone_id,
            interval_s,
        )

    def tick(self) -> None:
        """
        Check if it's time to publish a sync pulse and do so if needed.
        Should be called periodically from the coordinator's main loop.
        """
        now = time.time()
        if now - self._last_pulse_time >= self._interval_s:
            self._publish_pulse()
            self._last_pulse_time = now

    def _publish_pulse(self) -> None:
        """Publish a sync pulse to the MQTT broker."""
        self._cycle_counter += 1

        pulse: dict[str, Any] = {
            "timestamp": int(time.time() * 1000),
            "cycle_ref": self._cycle_counter,
            "zone_id": self._zone_id,
            "uptime_s": int(time.time() - self._start_time),
        }

        topic = TOPIC_SYNC_FMT.format(zone_id=self._zone_id)
        payload = json.dumps(pulse)

        self._client.publish(topic, payload, qos=0)
        logger.debug(
            "Sync pulse #%d published to %s", self._cycle_counter, topic
        )

    def force_pulse(self) -> None:
        """Force an immediate sync pulse (e.g., after plan update)."""
        self._publish_pulse()
        self._last_pulse_time = time.time()
        logger.info("Forced sync pulse published")
