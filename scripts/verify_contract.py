#!/usr/bin/env python3
# =============================================================================
# Semáforo Inteligente — Network Contract Verification
# =============================================================================
"""
Standalone check that the Coordinator honors the MQTT contract declared in
network/schemas/topics.yaml:

  1. The contract file loads and contains the canonical channels.
  2. GreenWaveCoordinator instantiates offline (no broker).
  3. An emitted command payload is valid JSON, satisfies the schema's
     required fields / allowed values, and carries a root-level integer
     sequence_number >= 1 that is strictly increasing across publishes.

No network traffic: the MQTT client is replaced with an in-memory stub
before anything is emitted.

Usage:
    .venv/bin/python scripts/verify_contract.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TOPICS_FILE = REPO_ROOT / "network" / "schemas" / "topics.yaml"
TOPOLOGY_FILE = REPO_ROOT / "network" / "configs" / "network_topology.json"

sys.path.insert(0, str(REPO_ROOT / "network" / "coordinator"))


class CapturingClient:
    """Stub MQTT client: records publishes instead of dispatching them."""

    def __init__(self) -> None:
        self.published: list[tuple[str, str, int, bool]] = []

    def publish(self, topic, payload, qos=0, retain=False, **kwargs):
        self.published.append((topic, payload, qos, retain))


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"CONTRACT VIOLATION: {message}")
        sys.exit(1)


def main() -> None:
    # --- 1. Contract file -----------------------------------------------------
    check(TOPICS_FILE.is_file(), f"missing contract file: {TOPICS_FILE}")
    with TOPICS_FILE.open("r", encoding="utf-8") as fh:
        contract = yaml.safe_load(fh)

    channels = contract.get("channels", {})
    for name in ("telemetry", "commands", "status"):
        check(name in channels, f"channel '{name}' missing from topics.yaml")

    check(
        channels["telemetry"]["subscribe"] == "semaforo/telemetry/+",
        f"telemetry subscription is {channels['telemetry'].get('subscribe')!r}, "
        "expected 'semaforo/telemetry/+'",
    )
    command_pattern = channels["commands"]["pattern"]
    check(
        command_pattern == "semaforo/commands/{node_id}",
        f"commands pattern is {command_pattern!r}, expected 'semaforo/commands/{{node_id}}'",
    )
    command_schema = channels["commands"]["payload"]
    print(f"[1/3] topics.yaml OK ({TOPICS_FILE.relative_to(REPO_ROOT)})")

    # --- 2. Coordinator instantiation (offline) --------------------------------
    import src.coordinator as coordinator_mod

    # The ShadowAuditor has filesystem/MQTT side effects; keep it out of this
    # offline check regardless of whether its import resolved.
    coordinator_mod.ShadowAuditor = None

    check(
        coordinator_mod.TOPIC_TELEMETRY == channels["telemetry"]["subscribe"],
        "coordinator TOPIC_TELEMETRY does not match topics.yaml",
    )
    check(
        coordinator_mod.TOPIC_COMMAND_FMT == command_pattern,
        "coordinator TOPIC_COMMAND_FMT does not match topics.yaml",
    )

    coord = coordinator_mod.GreenWaveCoordinator(topology_file=str(TOPOLOGY_FILE))
    stub = CapturingClient()
    coord._client = stub
    print("[2/3] GreenWaveCoordinator instantiated offline, publish stubbed")

    # --- 3. Emit two test commands and validate the wire payloads --------------
    node_id = "edge-001"
    topic = command_pattern.format(node_id=node_id)
    for _ in range(2):
        coord._publish_json(topic, coord.build_phase_command("GREEN_NS"))

    check(len(stub.published) == 2, f"expected 2 captured publishes, got {len(stub.published)}")

    now_ms = int(time.time() * 1000)
    previous_seq = 0
    for sent_topic, raw_payload, qos, retain in stub.published:
        check(sent_topic == topic, f"published to {sent_topic!r}, expected {topic!r}")
        check(qos == 1, f"command QoS is {qos}, contract requires 1")
        check(retain is False, "commands must not be retained")

        try:
            payload = json.loads(raw_payload)
        except (TypeError, ValueError) as exc:
            check(False, f"payload is not valid JSON: {exc}")

        for field in command_schema["required"]:
            check(field in payload, f"required field '{field}' missing: {raw_payload}")
        check(
            payload["action"] in command_schema["allowed_actions"],
            f"action {payload['action']!r} not in allowed_actions",
        )
        check(
            payload["phase"] in command_schema["allowed_phases"],
            f"phase {payload['phase']!r} not in allowed_phases",
        )
        check(
            isinstance(payload["timestamp"], int)
            and abs(now_ms - payload["timestamp"]) < 5_000,
            f"timestamp {payload.get('timestamp')!r} is not a fresh ms-epoch integer",
        )

        seq = payload.get("sequence_number")
        check(
            isinstance(seq, int) and not isinstance(seq, bool) and seq >= 1,
            f"sequence_number must be an integer >= 1, got {seq!r}",
        )
        check(
            seq > previous_seq,
            f"sequence_number not strictly increasing: {seq} after {previous_seq}",
        )
        previous_seq = seq

    print(f"[3/3] command payloads OK (sequence_numbers: "
          f"{[json.loads(p)['sequence_number'] for _, p, _, _ in stub.published]})")
    print("CONTRACT OK")


if __name__ == "__main__":
    main()
