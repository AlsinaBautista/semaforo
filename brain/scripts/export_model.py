#!/usr/bin/env python3
# Semáforo Inteligente - Brain Module
"""CLI script to export the trained MAPPO actor to ONNX format.

Usage::

    python scripts/export_model.py \\
        --checkpoint ../models/mappo_corridor \\
        --output models/policy.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.export.onnx_exporter import export_to_onnx

# The MAPPO checkpoint is written by train_mappo.py relative to the cwd it is
# launched from (repo root), hence one level above brain/.
DEFAULT_CHECKPOINT = PROJECT_ROOT.parent / "models" / "mappo_corridor"
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "policy.onnx"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Export the trained MAPPO actor to ONNX.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="RLlib Algorithm checkpoint root or RLModule checkpoint dir "
        f"(default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help=f"Destination path for the ONNX file (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--module-id",
        type=str,
        default="shared_signal_policy",
        help="Module id inside the checkpoint (default: shared_signal_policy).",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip ONNX verification after export.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: export the model and print confirmation."""
    args = parse_args()

    print("=" * 60)
    print("  Semáforo Inteligente — ONNX Model Export")
    print("=" * 60)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Output path  : {args.output}")
    print(f"  Module id    : {args.module_id}")
    print(f"  Opset        : {args.opset}")
    print(f"  Verify       : {not args.no_verify}")
    print("=" * 60)

    output = export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        module_id=args.module_id,
        opset_version=args.opset,
        verify=not args.no_verify,
    )

    print(f"\n✓ ONNX model exported to: {output}")


if __name__ == "__main__":
    main()
