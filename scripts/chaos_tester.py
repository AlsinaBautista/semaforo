#!/usr/bin/env python3
# Semáforo Inteligente - Chaos Engineering Suite
"""Inject real-time failures into the MQTT control plane to prove the Edge node
fails *safe*, never *catastrophic*.

This is an authorized resilience tool for the Semáforo Inteligente backend. It
attacks the link between the Brain/Coordinator and the C++ Edge nodes the same
way the real world will — corrupt payloads, delayed/replayed commands, broker
flapping and floods — and verifies that the Edge's hardened command parser and
``SafetyLayer`` respond correctly:

    * malformed / corrupt JSON          -> parser rejects, no crash
    * stale ("datos viejos") commands   -> freshness gate -> fixed-cycle fallback
    * spoofed future timestamps         -> rejected
    * oversized payloads                -> rejected before parsing (DoS guard)
    * command silence / broker flapping -> watchdog -> fixed-cycle fallback
    * publish flood + latency MITM       -> delayed delivery looks stale -> fallback

The expected, *correct* outcome of every attack is the same: the intersection
keeps running a deterministic safe cycle and the node publishes a ``degraded``
status — it must never segfault, freeze, or apply an unsafe phase.

USE ONLY against infrastructure you own or are authorized to test.

Examples
--------
Run the whole suite against a local broker / one edge::

    python scripts/chaos_tester.py --broker localhost --edge-id edge-001 --scenario all

Hammer just the malformed-JSON vector continuously::

    python scripts/chaos_tester.py --edge-id edge-001 --scenario malformed --repeat 50
"""

from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys
import threading
import time
from dataclasses import dataclass, field

try:
    import paho.mqtt.client as mqtt
except ImportError as exc:  # pragma: no cover
    print("paho-mqtt is required: pip install 'paho-mqtt>=2.0.0'", file=sys.stderr)
    raise SystemExit(1) from exc


def _now_ms() -> int:
    return int(time.time() * 1000)


def _cmd_topic(edge_id: str) -> str:
    return f"semaforo/commands/{edge_id}"


VALID_PHASES = [
    "GREEN_NS", "YELLOW_NS", "ALL_RED", "GREEN_EW",
    "YELLOW_EW", "GREEN_NS_LEFT", "GREEN_EW_LEFT",
]


# --------------------------------------------------------------------------- #
# Attack payload generators
# --------------------------------------------------------------------------- #
def malformed_payloads() -> list[tuple[str, bytes]]:
    """A battery of corrupt / hostile payloads the Edge must reject safely."""
    rand_blob = "".join(random.choices(string.printable, k=64))
    return [
        ("empty", b""),
        ("whitespace", b"   \t\n"),
        ("truncated_object", b'{"action":"CHANGE_PHASE","phase":'),
        ("dangling_value", b'{"action":}'),
        ("garbage_braces", b"}{"),
        ("binary_noise", bytes([0xFF, 0xFE, 0x00, 0x01, 0x02, 0x7F]) + b" noise"),
        ("not_an_object_array", b"[1, 2, 3]"),
        ("not_an_object_scalar", b"42"),
        ("json_null", b"null"),
        ("action_wrong_type", b'{"action": 123, "phase": "GREEN_EW", "timestamp": 0}'),
        ("missing_phase", json.dumps(
            {"action": "CHANGE_PHASE", "timestamp": _now_ms()}).encode()),
        ("missing_timestamp", json.dumps(
            {"action": "CHANGE_PHASE", "phase": "GREEN_EW"}).encode()),
        ("timestamp_wrong_type", json.dumps(
            {"action": "CHANGE_PHASE", "phase": "GREEN_EW",
             "timestamp": "soon"}).encode()),
        ("unknown_action", json.dumps(
            {"action": "SELF_DESTRUCT", "phase": "GREEN_EW",
             "timestamp": _now_ms()}).encode()),
        ("injection_phase", json.dumps(
            {"action": "CHANGE_PHASE", "phase": "GREEN_EW; DROP TABLE signals;--",
             "timestamp": _now_ms()}).encode()),
        ("unknown_phase", json.dumps(
            {"action": "CHANGE_PHASE", "phase": "RAINBOW",
             "timestamp": _now_ms()}).encode()),
        ("random_blob", rand_blob.encode()),
        ("deeply_nested", ("{" + '"a":' * 2000 + "1" + "}" * 2000).encode()),
    ]


def oversized_payload(size_kb: int = 64) -> bytes:
    """A valid-looking but enormous payload to exercise the size guard."""
    padding = "A" * (size_kb * 1024)
    return json.dumps({
        "action": "CHANGE_PHASE", "phase": "GREEN_EW",
        "timestamp": _now_ms(), "junk": padding,
    }).encode()


def stale_command(age_ms: int = 10_000) -> bytes:
    """A perfectly well-formed command, but with an old timestamp (replay/delay)."""
    return json.dumps({
        "action": "CHANGE_PHASE",
        "phase": random.choice(["GREEN_NS", "GREEN_EW"]),
        "priority": "HIGH",
        "timestamp": _now_ms() - age_ms,
    }).encode()


def future_command(skew_ms: int = 30_000) -> bytes:
    """A well-formed command timestamped in the future (clock skew / spoof)."""
    return json.dumps({
        "action": "CHANGE_PHASE", "phase": "GREEN_EW",
        "priority": "HIGH", "timestamp": _now_ms() + skew_ms,
    }).encode()


def valid_command() -> bytes:
    """A fresh, valid command used to confirm recovery after an attack."""
    return json.dumps({
        "action": "CHANGE_PHASE", "phase": random.choice(VALID_PHASES),
        "priority": "NORMAL", "timestamp": _now_ms(),
    }).encode()


# --------------------------------------------------------------------------- #
# Chaos tester
# --------------------------------------------------------------------------- #
@dataclass
class ChaosTester:
    """Drives chaos scenarios against an Edge node's MQTT command topic."""

    broker: str = "localhost"
    port: int = 1883
    edge_id: str = "edge-001"
    username: str | None = None
    password: str | None = None
    verbose: bool = True
    _client: mqtt.Client = field(init=False, default=None)
    _status: list[tuple[float, str]] = field(default_factory=list, init=False)

    # -- connection -------------------------------------------------------- #
    def connect(self) -> None:
        """Open the attacker MQTT connection and watch the node's status topic."""
        self._client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id=f"chaos_tester_{random.randint(0, 1_000_000)}",
        )
        if self.username:
            self._client.username_pw_set(self.username, self.password or "")
        self._client.on_message = self._on_status
        self._client.connect(self.broker, self.port, keepalive=15)
        self._client.subscribe(f"semaforo/status/{self.edge_id}", qos=1)
        self._client.loop_start()
        self._log(f"connected to {self.broker}:{self.port}, watching status of "
                  f"{self.edge_id}")

    def disconnect(self) -> None:
        """Close the attacker connection cleanly."""
        if self._client is not None:
            self._client.loop_stop()
            self._client.disconnect()

    def _on_status(self, _client, _userdata, msg) -> None:
        """Record node status transitions so we can see the safe-fallback reaction."""
        try:
            payload = msg.payload.decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            payload = repr(msg.payload)
        self._status.append((time.time(), payload))
        self._log(f"  <- node status: {payload}")

    def _publish(self, payload: bytes, qos: int = 1) -> None:
        self._client.publish(_cmd_topic(self.edge_id), payload, qos=qos)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    # -- individual attacks ------------------------------------------------ #
    def attack_malformed(self, repeat: int = 1, delay: float = 0.2) -> None:
        """Flood the command topic with corrupt / hostile JSON payloads."""
        self._log(f"\n[malformed] sending corrupt payloads x{repeat}")
        for _ in range(repeat):
            for name, payload in malformed_payloads():
                self._log(f"  -> {name} ({len(payload)}B)")
                self._publish(payload)
                time.sleep(delay)

    def attack_oversized(self, size_kb: int = 64) -> None:
        """Send an enormous payload to exercise the pre-parse size guard."""
        self._log(f"\n[oversized] sending {size_kb}KB payload")
        self._publish(oversized_payload(size_kb))
        time.sleep(0.5)

    def attack_stale(self, count: int = 10, age_ms: int = 10_000,
                     delay: float = 0.3) -> None:
        """Send a stream of valid-but-old commands ('datos viejos')."""
        self._log(f"\n[stale] sending {count} commands aged {age_ms}ms "
                  "(expect AI block -> fixed cycle)")
        for _ in range(count):
            self._publish(stale_command(age_ms))
            time.sleep(delay)

    def attack_future(self, count: int = 5, delay: float = 0.3) -> None:
        """Send commands timestamped in the future (clock skew / spoof)."""
        self._log(f"\n[future] sending {count} future-timestamped commands")
        for _ in range(count):
            self._publish(future_command())
            time.sleep(delay)

    def attack_latency_mitm(self, duration_s: float = 20.0,
                            min_delay: float = 2.0, max_delay: float = 6.0) -> None:
        """Man-in-the-middle latency: intercept real Coordinator commands and
        re-publish them after an artificial delay, so by the time they reach the
        Edge they are stale.  Demonstrates network-latency-induced fallback.

        Note: this double-publishes on the command topic; run it in a test
        environment where that is acceptable.
        """
        self._log(f"\n[latency-mitm] delaying intercepted commands "
                  f"{min_delay:.1f}-{max_delay:.1f}s for {duration_s:.0f}s")
        shadow = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                             client_id=f"chaos_mitm_{random.randint(0, 1_000_000)}")
        if self.username:
            shadow.username_pw_set(self.username, self.password or "")

        def _delayed_republish(payload: bytes) -> None:
            time.sleep(random.uniform(min_delay, max_delay))
            self._publish(payload)
            self._log(f"  -> replayed delayed command ({len(payload)}B)")

        def _on_cmd(_c, _u, msg) -> None:
            threading.Thread(
                target=_delayed_republish, args=(msg.payload,), daemon=True
            ).start()

        shadow.on_message = _on_cmd
        shadow.connect(self.broker, self.port, keepalive=15)
        shadow.subscribe(_cmd_topic(self.edge_id), qos=1)
        shadow.loop_start()
        # If nothing else is publishing, seed our own traffic to delay.
        deadline = time.time() + duration_s
        while time.time() < deadline:
            self._publish(valid_command())  # will be intercepted + delayed
            time.sleep(2.0)
        shadow.loop_stop()
        shadow.disconnect()

    def attack_flood(self, count: int = 2000, qos: int = 0) -> None:
        """Saturate the command topic to induce queue latency / backpressure."""
        self._log(f"\n[flood] publishing {count} messages as fast as possible")
        for _ in range(count):
            self._publish(valid_command(), qos=qos)

    def attack_disconnect_flapping(self, cycles: int = 8, up: float = 0.4,
                                   down: float = 0.4) -> None:
        """Rapidly connect/disconnect to emulate an unstable broker link."""
        self._log(f"\n[flapping] {cycles} connect/disconnect cycles")
        for i in range(cycles):
            try:
                self._client.reconnect()
                self._log(f"  cycle {i}: up")
                time.sleep(up)
                self._client.disconnect()
                self._log(f"  cycle {i}: down")
                time.sleep(down)
            except Exception as exc:  # noqa: BLE001
                self._log(f"  cycle {i}: error {exc}")
        # Restore a clean connection for subsequent observation.
        self._client.reconnect()

    def confirm_recovery(self, count: int = 6, delay: float = 0.5) -> None:
        """Send a streak of fresh valid commands and watch for 'online' status.

        After the guard's recovery streak the Edge should release the AI from the
        fixed-cycle fallback.
        """
        self._log(f"\n[recovery] sending {count} fresh valid commands")
        for _ in range(count):
            self._publish(valid_command())
            time.sleep(delay)
        time.sleep(2.0)

    # -- orchestration ----------------------------------------------------- #
    def run_full_suite(self) -> None:
        """Run every attack in sequence, then confirm recovery."""
        self.attack_malformed()
        self.attack_oversized()
        self.attack_future()
        self.attack_stale()
        self.attack_flood()
        self.attack_disconnect_flapping()
        self.confirm_recovery()

    def summary(self) -> None:
        """Print the node status transitions observed during the run."""
        print("\n" + "=" * 56)
        print("  CHAOS RUN SUMMARY — observed node status transitions")
        print("=" * 56)
        if not self._status:
            print("  (no status messages received — is the edge node running and\n"
                  "   publishing to semaforo/status/{}? )".format(self.edge_id))
        for ts, payload in self._status:
            print(f"  {time.strftime('%H:%M:%S', time.localtime(ts))}  {payload}")
        print("=" * 56)
        print("  EXPECTED: 'degraded'/'ai_blocked'/'command_silence' during attacks,\n"
              "            then 'online'/'ai_restored' after recovery — and the\n"
              "            process must still be alive (no crash).")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--broker", default=os.getenv("MQTT_BROKER", "localhost"))
    p.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    p.add_argument("--edge-id", default="edge-001")
    p.add_argument("--username", default=os.getenv("MQTT_USERNAME"))
    p.add_argument("--password", default=os.getenv("MQTT_PASSWORD"))
    p.add_argument(
        "--scenario",
        choices=["all", "malformed", "oversized", "stale", "future",
                 "latency", "flood", "flapping", "recovery"],
        default="all",
    )
    p.add_argument("--repeat", type=int, default=1,
                   help="Repeat count for repeatable scenarios (e.g. malformed).")
    p.add_argument("--age-ms", type=int, default=10_000,
                   help="Age for stale commands in milliseconds.")
    p.add_argument("--duration", type=float, default=20.0,
                   help="Duration for the latency MITM scenario (seconds).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print payloads without connecting to a broker.")
    return p.parse_args()


def _dry_run() -> None:
    """Print the attack payloads without touching the network."""
    print("DRY RUN — malformed payload battery:")
    for name, payload in malformed_payloads():
        shown = payload[:80] + (b"..." if len(payload) > 80 else b"")
        print(f"  {name:24s} {len(payload):6d}B  {shown!r}")
    print(f"\n  oversized: {len(oversized_payload())}B")
    print(f"  stale    : {stale_command().decode()}")
    print(f"  future   : {future_command().decode()}")
    print(f"  valid    : {valid_command().decode()}")


def main() -> None:
    """Entry point: connect, run the selected scenario, print the summary."""
    args = parse_args()

    if args.dry_run:
        _dry_run()
        return

    tester = ChaosTester(
        broker=args.broker, port=args.port, edge_id=args.edge_id,
        username=args.username, password=args.password,
    )
    tester.connect()
    time.sleep(1.0)  # let the status subscription settle

    try:
        scenario = args.scenario
        if scenario == "all":
            tester.run_full_suite()
        elif scenario == "malformed":
            tester.attack_malformed(repeat=args.repeat)
        elif scenario == "oversized":
            tester.attack_oversized()
        elif scenario == "stale":
            tester.attack_stale(count=max(args.repeat, 10), age_ms=args.age_ms)
        elif scenario == "future":
            tester.attack_future(count=max(args.repeat, 5))
        elif scenario == "latency":
            tester.attack_latency_mitm(duration_s=args.duration)
        elif scenario == "flood":
            tester.attack_flood()
        elif scenario == "flapping":
            tester.attack_disconnect_flapping()
        elif scenario == "recovery":
            tester.confirm_recovery()

        time.sleep(2.0)  # allow trailing status messages to arrive
    finally:
        tester.summary()
        tester.disconnect()


if __name__ == "__main__":
    main()
