#!/usr/bin/env python3
# =============================================================================
# Semáforo Inteligente — Brain Inference Verification
# =============================================================================
"""
Standalone end-to-end check of the Brain→Coordinator inference path:

  1. The MAPPO actor exports to ONNX from the trained RLlib checkpoint
     (asymmetric trace: actor only, centralized critic pruned).
  2. The Coordinator's AIPolicyWrapper loads the .onnx into an
     onnxruntime.InferenceSession.
  3. A dummy [1, 32] observation produces a valid (1, 2) logits matrix.
  4. predict() maps real telemetry to a valid, deterministic phase string.

The export runs in a subprocess: the brain package and the coordinator
package are both imported as `src`, so they cannot share one interpreter.

Usage:
    .venv/bin/python scripts/verify_brain_inference.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPO_ROOT / "models" / "mappo_corridor"
ONNX_PATH = REPO_ROOT / "brain" / "models" / "policy.onnx"
EXPORT_CLI = REPO_ROOT / "brain" / "scripts" / "export_model.py"

sys.path.insert(0, str(REPO_ROOT / "network" / "coordinator"))


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"INFERENCE VIOLATION: {message}")
        sys.exit(1)


def main() -> None:
    # --- 1. Export the actor to ONNX (subprocess: brain owns `src` there) ------
    check(CHECKPOINT.is_dir(), f"missing MAPPO checkpoint: {CHECKPOINT}")
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORT_CLI),
            "--checkpoint", str(CHECKPOINT),
            "--output", str(ONNX_PATH),
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
    check(result.returncode == 0, f"export_model.py failed (rc={result.returncode})")
    check(ONNX_PATH.is_file(), f"export reported success but {ONNX_PATH} is missing")
    print(f"[1/4] actor exported to {ONNX_PATH.relative_to(REPO_ROOT)}")

    # --- 2. Instantiate the Coordinator's wrapper ------------------------------
    from src.coordinator import AIPolicyWrapper

    wrapper = AIPolicyWrapper(model_path=ONNX_PATH)
    print("[2/4] AIPolicyWrapper loaded the ONNX model into onnxruntime")

    # --- 3. Dummy [1, 32] observation -> valid action matrix -------------------
    dummy = np.random.rand(1, 32).astype(np.float32)
    out = wrapper.infer(dummy)
    check(isinstance(out, np.ndarray), f"infer() returned {type(out).__name__}, not ndarray")
    check(out.shape == (1, 2), f"logits shape is {out.shape}, expected (1, 2)")
    check(out.dtype == np.float32, f"logits dtype is {out.dtype}, expected float32")
    check(bool(np.isfinite(out).all()), f"logits contain non-finite values: {out}")
    print(f"[3/4] dummy [1, 32] inference OK (logits: {out[0].tolist()})")

    # --- 4. Telemetry round-trip through predict() ------------------------------
    telemetry = {
        "intersection_id": "verify",
        "phase": "GREEN_NS",
        "vehicle_count": 12,
        "total_vehicles": 12,
        "flow_rate": 0.0,
    }
    phase = wrapper.predict(telemetry)
    check(
        phase in {"GREEN_NS", "GREEN_EW"},
        f"predict() returned {phase!r}, expected GREEN_NS or GREEN_EW",
    )
    check(
        wrapper.predict(telemetry) == phase,
        "predict() is not deterministic for identical telemetry",
    )
    print(f"[4/4] predict() round-trip OK (telemetry -> {phase})")
    print("BRAIN INFERENCE OK")


if __name__ == "__main__":
    main()
