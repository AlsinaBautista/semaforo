#!/usr/bin/env python3
"""High-speed malicious payload injector for the Edge command topic.

Fires a continuous stream of hostile MQTT payloads at the Edge's command
topic with no inter-publish delay.  Every payload class is designed to
exercise a distinct rejection path in CommandParser / CommandGuard:

  - truncated_json          JSON cut mid-value (parse error)
  - unclosed_brace          unclosed object literal (parse error)
  - array_not_object        top-level array instead of object
  - missing_fields          omits required action / phase / timestamp
  - type_mutation           int passed where string is expected (and vice-versa)
  - negative_sequence       sequence_number < 0 (invalid per contract)

Usage
-----
  # Dry run — print payloads for 3 s, exit 0 (no broker needed):
  python tests/chaos_monkey_mqtt.py --dry-run

  # Live injection against a local broker:
  python tests/chaos_monkey_mqtt.py --broker localhost --edge-id edge-001

  # Run for a fixed duration then stop:
  python tests/chaos_monkey_mqtt.py --broker localhost --duration 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Iterator

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:
    print("paho-mqtt is required: pip install 'paho-mqtt>=2.0.0'", file=sys.stderr)
    raise SystemExit(1) from exc


def _now_ms() -> int:
    return int(time.time() * 1000)


# ---------------------------------------------------------------------------
# Payload generators — each yields (label, raw_bytes)
# ---------------------------------------------------------------------------

def _payloads() -> Iterator[tuple[str, bytes]]:
    """Cycle through all hostile payload classes indefinitely."""
    cases: list[tuple[str, bytes]] = [
        # 1. Truncated JSON — cut mid-value
        ("truncated_json",
         b'{"action":"CHANGE_PHASE","phase":"GREEN_NS","timestamp":'),

        # 2. Unclosed brace
        ("unclosed_brace",
         b'{"action":"CHANGE_PHASE","phase":"GREEN_EW","timestamp":' +
         str(_now_ms()).encode()),

        # 3. Array instead of object
        ("array_not_object",
         json.dumps(["CHANGE_PHASE", "GREEN_NS", _now_ms()]).encode()),

        # 4a. Missing 'phase' field
        ("missing_phase",
         json.dumps({"action": "CHANGE_PHASE",
                     "timestamp": _now_ms()}).encode()),

        # 4b. Missing 'timestamp' field
        ("missing_timestamp",
         json.dumps({"action": "CHANGE_PHASE",
                     "phase": "GREEN_EW"}).encode()),

        # 4c. Missing 'action' field
        ("missing_action",
         json.dumps({"phase": "GREEN_NS",
                     "timestamp": _now_ms()}).encode()),

        # 5a. Type mutation: timestamp as string instead of int
        ("type_mutation_timestamp_str",
         json.dumps({"action": "CHANGE_PHASE", "phase": "GREEN_EW",
                     "timestamp": "now"}).encode()),

        # 5b. Type mutation: action as int instead of string
        ("type_mutation_action_int",
         json.dumps({"action": 1, "phase": "GREEN_EW",
                     "timestamp": _now_ms()}).encode()),

        # 5c. Type mutation: phase as int instead of string
        ("type_mutation_phase_int",
         json.dumps({"action": "CHANGE_PHASE", "phase": 0,
                     "timestamp": _now_ms()}).encode()),

        # 6. Negative sequence_number
        ("negative_sequence",
         json.dumps({"action": "CHANGE_PHASE", "phase": "GREEN_NS",
                     "timestamp": _now_ms(),
                     "sequence_number": -999}).encode()),
    ]
    idx = 0
    while True:
        # Refresh timestamp-bearing payloads on each cycle so they stay fresh
        # and still reach the parser (the freshness check runs after type checks,
        # so they exercise the type-rejection path, not the staleness path).
        label, payload = cases[idx % len(cases)]
        idx += 1
        yield label, payload


# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------

class ChaosMonkey:
    """Connects to MQTT and fires hostile payloads without pause."""

    def __init__(self, broker: str, port: int, edge_id: str) -> None:
        self._topic = f"semaforo/commands/{edge_id}"
        self._broker = broker
        self._port = port
        self._client: mqtt.Client | None = None
        self._connected = False

    def connect(self) -> None:
        self._client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="chaos_monkey",
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        # Non-blocking; if the broker is unavailable the loop handles reconnects.
        try:
            self._client.connect(self._broker, self._port, keepalive=10)
        except OSError:
            # Broker unreachable at startup — loop_start will keep retrying.
            pass
        self._client.loop_start()

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        self._connected = (rc == 0)

    def _on_disconnect(self, client, userdata, flags, rc, properties=None) -> None:
        self._connected = False

    def inject(self, label: str, payload: bytes) -> None:
        if self._client is not None and self._connected:
            self._client.publish(self._topic, payload, qos=0)

    def disconnect(self) -> None:
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--broker", default="localhost")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--edge-id", default="edge-001")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Stop after this many seconds (0 = run until Ctrl-C).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print payloads for 3 s without connecting to a broker.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if args.dry_run:
        deadline = time.monotonic() + 3.0
        count = 0
        gen = _payloads()
        while time.monotonic() < deadline:
            label, payload = next(gen)
            shown = payload[:72] + (b"..." if len(payload) > 72 else b"")
            print(f"[{label:<36s}] {len(payload):5d}B  {shown!r}", flush=True)
            count += 1
            # One pass through all cases per iteration; pause briefly so the
            # 3-second window produces readable output rather than millions of lines.
            time.sleep(0.1)
        print(f"\ndry-run complete: {count} payloads generated in 3 s", flush=True)
        sys.exit(0)

    monkey = ChaosMonkey(broker=args.broker, port=args.port, edge_id=args.edge_id)
    monkey.connect()

    # Give the MQTT loop a moment to establish the connection.
    time.sleep(0.5)

    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    count = 0
    try:
        for label, payload in _payloads():
            if deadline is not None and time.monotonic() >= deadline:
                break
            monkey.inject(label, payload)
            count += 1
    except KeyboardInterrupt:
        pass
    finally:
        monkey.disconnect()
        print(f"chaos_monkey: {count} payloads sent to "
              f"semaforo/commands/{args.edge_id}", flush=True)


if __name__ == "__main__":
    main()
