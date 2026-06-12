# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Mission-critical intelligent traffic-signal backend. A C++ **Edge** node runs YOLO inference + a deterministic safety interlock at each intersection; a Python **Brain** trains a MAPPO agent in SUMO and publishes phase decisions; a Python **Coordinator** orchestrates green waves; everything talks over **MQTT (Mosquitto)**. There is also a Flutter `dashboard/`, but it is out of scope for backend work.

The guiding constraint throughout: a software fault must **never** cause an unsafe signal state or a frozen intersection. Favor fail-safe behavior (all-red / fixed cycle) over cleverness.

## Repository reality vs. README

The README's directory tree and some commands are aspirational and do **not** match the code. Trust the code and the commands below. Notably the real layout is `brain/src/environments/`, `brain/src/agents/`, `brain/src/rewards/` (plural), and the Edge binary takes a **positional** config arg, not `--config`.

## Brain (Python RL) — commands

The dependency-heavy interpreter is the **repo-root** venv (`.venv`, Python 3.14: has `sumo_rl`, `ray[rllib]`, `torch`, `stable-baselines3`). Brain code imports as `src.*` and expects `brain/` on `sys.path`.

```bash
# From repo root — run anything in the brain package:
PYTHONPATH=brain .venv/bin/python brain/scripts/train_mappo.py --smoke      # 1 short RLlib iter, sanity check
PYTHONPATH=brain .venv/bin/python brain/scripts/train_mappo.py --iterations 300 --num-env-runners 6
PYTHONPATH=brain .venv/bin/python brain/scripts/train_single.py --net-file ... --route-file ...

# Tests (pytest config lives in brain/pyproject.toml; testpaths=tests):
cd brain && PYTHONPATH=$PWD ../.venv/bin/python -m pytest tests/ -q
# Single test:
cd brain && PYTHONPATH=$PWD ../.venv/bin/python -m pytest tests/test_green_wave.py::test_name -q
```

Gotchas:
- **`import sumo_rl` fails with "Please declare SUMO_HOME"** unless you go through the package. Importing anything under `src.environments` runs `src/environments/sumo_home.py`, which auto-points `SUMO_HOME` at the bundled `sumo` wheel. Always import env modules (or call `ensure_sumo_home()`) before touching `sumo_rl`.
- Imports are `from src.foo import ...`, **not** `from brain.src.foo`. A few legacy files may still use `brain.src` — fix them to `src` if you hit an ImportError.
- `tests/test_rewards.py` has **known pre-existing failures**; `tests/test_rewards_fixed.py` is the corrected version. Don't treat `test_rewards.py` failures as regressions you caused — check git/diff scope first.
- SUMO subprocess relaunch on macOS is flaky; `MultiIntersectionEnv` deliberately **recreates** the underlying `SumoEnvironment` each `reset()` rather than reusing it.

### Deploying a retrained policy (post-training checklist)

Training auto-exports to `models/mappo_corridor/policy.onnx`, but the Coordinator reads `brain/models/policy.onnx` — the gap is deliberate (a smoke run must never clobber the deployed model). To deploy:
1. `PYTHONPATH=brain .venv/bin/python brain/scripts/export_model.py --checkpoint models/mappo_corridor --output brain/models/policy.onnx`
2. Validate: `.venv/bin/python scripts/verify_brain_inference.py --model brain/models/policy.onnx`
3. Start the coordinator WITHOUT `ONNX_MODEL_PATH=brain/models/policy_stub.onnx` (retire the pre-retrain stub).

The 17-dim observation's direction bucket order is **[N, E, S, W]** (incLanes order, uniform across c0/c1/c2) — any inference-side constructor filling real per-direction data must match it exactly or the policy receives a permutation.

## Edge (C++) — commands

C++17, `-Wall -Wextra -Wpedantic -Werror` (warnings are build failures). Dependencies via Conan (`edge/conanfile.txt`) or system packages; `paho.mqtt.cpp` and `nlohmann/json` are pulled by CMake `FetchContent`.

```bash
cd edge && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release && make -j$(nproc)
./semaforo_edge ../configs/node_config.yaml          # positional config arg

# Tests (GoogleTest, two targets):
ctest --output-on-failure          # from edge/build
./test_safety_layer                # property-based interlock proofs
./test_command_parser              # malformed/stale-payload rejection proofs
```

When iterating on a single self-contained class, a fast loop is to compile that `.cpp` against a stub header + the real third-party headers (`/opt/homebrew/include`) rather than the full CMake graph — the SafetyLayer and CommandParser are written to be testable in isolation.

## Edge architecture — the parts that span files

The Edge node is the safety-critical core. Three cooperating mechanisms, understandable only by reading `main.cpp` together with the headers:

- **SafetyLayer** (`safety_layer.{hpp,cpp}`) is a **deterministic, self-timed, authoritative FSM** and the single source of truth for what the signal heads show. `validate(desired_phase)` takes the AI's *request* and returns the phase to actually drive, internally enforcing: min-green, a **max-green watchdog** (forces a switch if a green is held too long — anti-freeze), amber + all-red **clearance** on every green→green change (a green can never go directly to a conflicting green), and a fixed **anti-starvation emergency ring** when in fallback. The clock is injectable (`validate(phase, now)`) for deterministic tests. Invariants are proven in `tests/test_safety_layer.cpp`.

- **Hardened command ingestion** (`command_parser.hpp`): `CommandParser` validates untrusted MQTT JSON **without ever throwing** (rejects malformed/oversized/stale/future-timestamped/unknown-phase payloads). `CommandGuard` is a hysteresis state machine that drops the node into the SafetyLayer's fixed-cycle fallback when the Brain becomes untrustworthy (consecutive bad payloads or total command silence) and restores AI control only after a streak of fresh valid commands.

- **Actor / Single-Writer concurrency** (`command_queue.hpp` + `main.cpp`): the Paho MQTT callback thread is a **producer only** — it parses and `push`es a `ParsedCommand` onto the SPSC `CommandQueue`. The **main loop is the single writer**: it drains the queue, runs the guard, picks the desired phase (camera-down → emergency; fresh Brain command → remote phase; else local ONNX policy), and is the *only* thread that calls `SafetyLayer::validate`, `set_emergency_mode`, and the signal actuator. Do not reintroduce control-state writes on the callback thread.

When changing camera handling: `VideoPipeline` (`pipeline.cpp`) is **non-movable** (a reconnect thread captures `this`) and isolates `cv::VideoCapture` behind a `shared_ptr` double-buffer + stop-token — the background reconnect opens a *new* capture and atomically swaps it in; it never `release()`/`open()`s the object the main thread is reading. Preserve this pattern.

## MQTT wire contract (and a known break)

Topic schema (read `edge/src/mqtt_client.cpp` + `network/coordinator/src/coordinator.py`):
- Edge → telemetry: `semaforo/telemetry/<node_id>`  (currently **JSON**)
- Edge ← commands: `semaforo/commands/<node_id>`  (JSON: `action`, `phase`, `priority`, `timestamp` ms)
- Edge → status/LWT: `semaforo/status/<node_id>`

**Known architectural defect (do not assume the loop is closed):** the Coordinator subscribes to `semaforo/+/telemetry` and parses **protobuf**, while the Edge publishes to `semaforo/telemetry/<id>` as **JSON** — topic *and* serialization mismatch, so Brain telemetry is currently dropped. The `proto/*.proto` schemas are the intended format. Treat unifying this (one topic namespace, one encoding, contract-tested) as the real fix, not a patch.

## Chaos / resilience testing

`scripts/chaos_tester.py` attacks a node's command topic (malformed JSON, stale/future timestamps, oversized payloads, latency MITM, broker flapping, flood) and watches `semaforo/status/<id>`. The correct outcome of every attack is the same: the intersection keeps a safe cycle and publishes a `degraded` status — it must never crash or apply an unsafe phase.

```bash
.venv/bin/python scripts/chaos_tester.py --dry-run                                  # print payloads, no broker
.venv/bin/python scripts/chaos_tester.py --broker localhost --edge-id edge-001 --scenario all
```

## Conventions

- The SafetyLayer treats **any** change between two different greens as a conflict (mediated by amber + all-red) — intentionally conservative; don't "optimize" it into direct green-to-green without a conflict matrix and updated property tests.
- ThreadSanitizer is the intended race gate for the Edge but is broken on Apple-Silicon/macOS here (segfaults on a trivial program). Run TSan in CI on a Linux runner; locally, validate concurrency by design review + non-instrumented stress.
- New SafetyLayer / parser behavior must come with property-based tests (randomized command storms asserting invariants), matching the existing `test_safety_layer.cpp` / `test_command_parser.cpp` style.
