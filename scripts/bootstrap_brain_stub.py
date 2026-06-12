#!/usr/bin/env python
"""Regenerate the pre-retrain ONNX policy stub (deterministic).

All *.onnx files are gitignored, so a fresh clone has no model on disk and
both the Coordinator and `scripts/verify_brain_inference.py` refuse to start.
This script rebuilds `brain/models/policy_stub.onnx` byte-for-byte: a uniform
placeholder actor over the canonical 17-dim observation —

    obs (B, 17) float32 → MatMul(W = zeros 17×4) → Add(b = [0.25]×4)
                        → action_probs (B, 4)  (constant uniform output)

It is NOT a trained policy. It exists so the stack boots before the MAPPO
retrain; the real model is exported with `brain/scripts/export_model.py`.

Usage:
    .venv/bin/python scripts/bootstrap_brain_stub.py   # or: make brain-stub
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[1]
STUB_PATH = REPO_ROOT / "brain" / "models" / "policy_stub.onnx"

_OBS_SIZE = 17   # canonical observation (brain/src/environments/multi_intersection.py)
_NUM_ACTIONS = 4
_OPSET = 17
_IR_VERSION = 8  # pinned so regeneration is stable across onnx versions


def check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        sys.exit(1)


def build_stub() -> onnx.ModelProto:
    graph = helper.make_graph(
        nodes=[
            helper.make_node("MatMul", ["obs", "W"], ["zeroed"]),
            helper.make_node("Add", ["zeroed", "b"], ["action_probs"]),
        ],
        name="policy_stub",
        inputs=[
            helper.make_tensor_value_info(
                "obs", TensorProto.FLOAT, ["B", _OBS_SIZE]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "action_probs", TensorProto.FLOAT, ["B", _NUM_ACTIONS]
            )
        ],
        initializer=[
            numpy_helper.from_array(
                np.zeros((_OBS_SIZE, _NUM_ACTIONS), dtype=np.float32), "W"
            ),
            numpy_helper.from_array(
                np.full((_NUM_ACTIONS,), 0.25, dtype=np.float32), "b"
            ),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="semaforo-policy-stub",
        opset_imports=[helper.make_opsetid("", _OPSET)],
    )
    model.ir_version = _IR_VERSION
    return model


def main() -> None:
    model = build_stub()
    onnx.checker.check_model(model)
    print(f"[1/3] stub graph built and passed onnx.checker (opset {_OPSET})")

    STUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(STUB_PATH))
    print(f"[2/3] written to {STUB_PATH.relative_to(REPO_ROOT)}")

    # Smoke inference: any observation must map to the constant uniform output.
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(STUB_PATH), providers=["CPUExecutionProvider"]
    )
    obs = np.random.rand(1, _OBS_SIZE).astype(np.float32)
    out = session.run(None, {"obs": obs})[0]
    check(out.shape == (1, _NUM_ACTIONS), f"output shape {out.shape}")
    check(bool(np.allclose(out, 0.25)), f"output not uniform 0.25: {out}")
    print(f"[3/3] onnxruntime smoke inference OK (output {out[0].tolist()})")
    print("\nOK: policy_stub.onnx regenerated. This is a uniform placeholder, "
          "NOT a trained policy — retrain + export before trusting inference.")


if __name__ == "__main__":
    main()
