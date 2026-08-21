#!/usr/bin/env python3
"""Shadow-mode runner: YOLO + MAPPO actor (ONNX) on real traffic cameras.

This is the **first-pilot shadow harness**: it reads real camera feeds (one per
direction, files or RTSP), detects vehicles with YOLOv8, builds the canonical
17-dim observation, and runs the *deployed* MAPPO actor (``policy.onnx``) via
onnxruntime to produce a recommended phase **without ever driving any lights**.
Every decision is appended to a structured JSONL log so the policy can later be
evaluated against the intersection's real fixed-time controller.

It deliberately bypasses the Edge/coordinator/MQTT stack: in shadow mode there
is no actuation, so the smallest honest surface is camera -> YOLO -> policy -> log.

The model is run as a *counterfactual rollout*: the observation's phase one-hot
and elapsed timer track the policy's *own* recommended phase (i.e. "what would
the AI do if it were in control"). To compare against the real signal, capture
the real phase out-of-band and merge it into the log via ``ground_truth_phase``
(left null here — there is no signal tap in shadow mode).

Usage (live RTSP, headless, NS/EW single cameras per axis):

    PYTHONPATH=brain .venv/bin/python brain/scripts/yolo_inference.py \\
        --videos rtsp://cam-ns/stream rtsp://cam-ew/stream \\
        --directions NS EW \\
        --policy models/mappo_corridor/policy.onnx \\
        --log pilot_shadow.jsonl

Usage (recorded videos with annotated output for review):

    PYTHONPATH=brain .venv/bin/python brain/scripts/yolo_inference.py \\
        --videos cam_n.mp4 cam_e.mp4 cam_s.mp4 cam_w.mp4 \\
        --directions N E S W \\
        --policy models/mappo_corridor/policy.onnx \\
        --log pilot_shadow.jsonl --output output_demo.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vision.vehicle_detector import VehicleDetector
from src.vision.obs_builder import ObservationBuilder


# Colours for visualisation (BGR)
_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_YELLOW = (0, 220, 220)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)
_PANEL_BG = (40, 40, 40)
_PHASE_COLOURS = {0: _GREEN, 1: _RED}  # Phase 0=NS green, 1=EW green

# Discrete(2) action -> phase string, matching the Coordinator's AIPolicyWrapper.
_PHASE_BY_ACTION = ("GREEN_NS", "GREEN_EW")
_OBS_SIZE = 17


class OnnxPolicy:
    """Deployed MAPPO actor loaded from ONNX, served with onnxruntime.

    Mirrors the Coordinator's ``AIPolicyWrapper`` contract: a (B, 17) float32
    observation in, action logits out; the recommended action is ``argmax``.
    """

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX policy not found: {path}")
        self._session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        out_shape = self._session.get_outputs()[0].shape
        self.num_actions = int(out_shape[-1]) if isinstance(out_shape[-1], int) else 2
        if self.num_actions > len(_PHASE_BY_ACTION):
            print(
                f"WARNING: policy emits {self.num_actions} logits but only "
                f"{len(_PHASE_BY_ACTION)} phases are defined; actions beyond "
                f"{len(_PHASE_BY_ACTION) - 1} will be clamped. This usually means "
                f"the stub model is loaded instead of the trained actor."
            )

    def decide(self, obs: np.ndarray) -> tuple[int, list[float]]:
        """Return (action_index, logits) for a single 17-dim observation."""
        batch = np.asarray(obs, dtype=np.float32).reshape(1, _OBS_SIZE)
        logits = self._session.run(None, {self._input_name: batch})[0][0]
        action = min(int(np.argmax(logits)), len(_PHASE_BY_ACTION) - 1)
        return action, [float(x) for x in logits]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shadow-mode YOLO + MAPPO (ONNX) traffic inference harness.",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Camera sources (files, device indices, or rtsp:// URLs), 2 or 4.",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        required=True,
        help="Direction each camera faces: N, E, S, W, NS, or EW. "
             "Use NS/EW when a single camera covers both directions of an axis.",
    )
    parser.add_argument(
        "--policy",
        type=str,
        required=True,
        help="Path to the deployed MAPPO actor exported to ONNX (17-dim input).",
    )
    parser.add_argument(
        "--yolo-weights",
        type=str,
        default=str(PROJECT_ROOT.parent / "yolov8n.pt"),
        help="Path to YOLOv8 weights (default: ../../yolov8n.pt).",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="JSONL path to append one structured record per decision "
             "(recommended for the pilot — this is the evidence).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional annotated MP4 for review (omit for headless live runs).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Max frames to process (0 = all).",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=5,
        help="Frames between Brain decisions (default: 5).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display output in real-time window.",
    )
    return parser.parse_args()


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    colour: tuple[int, int, int] = _GREEN,
) -> None:
    """Draw bounding boxes and labels on a frame (in-place)."""
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = f"{det['class_name']} {det['confidence']:.0%}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)


def draw_count_badge(
    frame: np.ndarray,
    count: int,
    direction: str,
    is_green: bool,
) -> None:
    """Draw vehicle count and traffic light status badge on a frame."""
    h, w = frame.shape[:2]
    colour = _GREEN if is_green else _RED
    status = "GREEN" if is_green else "RED"

    # Badge background
    cv2.rectangle(frame, (5, 5), (220, 75), _PANEL_BG, -1)
    cv2.rectangle(frame, (5, 5), (220, 75), colour, 2)

    # Direction and count
    cv2.putText(frame, f"Camera {direction}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _WHITE, 2)
    cv2.putText(frame, f"Vehicles: {count}", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _YELLOW, 2)
    cv2.putText(frame, status, (12, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)


def build_decision_panel(
    width: int,
    counts: dict[str, int],
    phase: int,
    frame_idx: int,
    fps: float,
) -> np.ndarray:
    """Create a decision panel showing the Brain's current state."""
    panel_h = 120
    panel = np.full((panel_h, width, 3), 40, dtype=np.uint8)

    # Title
    cv2.putText(panel, "SEMAFORO INTELIGENTE — Shadow Recommendation",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _WHITE, 2)

    # Counts
    y = 50
    x = 10
    for direction, count in counts.items():
        text = f"{direction}: {count}"
        cv2.putText(panel, text, (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _YELLOW, 1)
        x += 120

    # Phase decision
    phase_text = "N-S VERDE | E-W ROJO" if phase == 0 else "E-W VERDE | N-S ROJO"
    phase_colour = _GREEN
    cv2.putText(panel, f"Recommend: {phase_text}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, phase_colour, 2)

    # Traffic light indicators
    circle_y = 100
    # N-S indicator
    cv2.putText(panel, "N-S:", (10, circle_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)
    ns_colour = _GREEN if phase == 0 else _RED
    cv2.circle(panel, (65, circle_y - 5), 8, ns_colour, -1)

    # E-W indicator
    cv2.putText(panel, "E-W:", (100, circle_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1)
    ew_colour = _GREEN if phase == 1 else _RED
    cv2.circle(panel, (155, circle_y - 5), 8, ew_colour, -1)

    # Frame info
    time_text = f"Frame: {frame_idx}  |  FPS: {fps:.1f}"
    cv2.putText(panel, time_text, (width - 280, circle_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    return panel


def main() -> None:
    args = parse_args()

    # Validate inputs
    if len(args.videos) != len(args.directions):
        print("ERROR: Number of videos must match number of directions.")
        sys.exit(1)

    print("=" * 60)
    print("  Semáforo Inteligente — Shadow-Mode Inference Harness")
    print("=" * 60)

    # --- Load models ---
    print(f"\nLoading YOLO weights: {args.yolo_weights}")
    detector = VehicleDetector(weights=args.yolo_weights, confidence=0.3)

    print(f"Loading MAPPO actor (ONNX): {args.policy}")
    policy = OnnxPolicy(args.policy)

    obs_builder = ObservationBuilder(initial_phase=0)

    # --- Structured decision log (the pilot evidence) ---
    log_fh = None
    if args.log:
        log_fh = open(args.log, "a", encoding="utf-8", buffering=1)  # line-buffered
        print(f"  Logging decisions to: {args.log}")

    # --- Open video captures ---
    caps: list[cv2.VideoCapture] = []
    for video_path in args.videos:
        # int() lets "0", "1" select local capture devices; otherwise file/RTSP.
        source: int | str = int(video_path) if video_path.isdigit() else video_path
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            sys.exit(1)
        caps.append(cap)
        print(f"  Opened: {video_path} "
              f"({int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}, "
              f"{cap.get(cv2.CAP_PROP_FPS):.0f} fps)")

    # Use dimensions from first video for layout
    frame_w = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    frame_h = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    source_fps = caps[0].get(cv2.CAP_PROP_FPS) or 10.0

    # --- Optional annotated-video output ---
    out = None
    tile_w = tile_h = grid_cols = grid_rows = 0
    if args.output or args.show:
        n_cams = len(caps)
        grid_cols, grid_rows = (2, 1) if n_cams <= 2 else (2, 2)
        tile_w = 640
        tile_h = int(tile_w * frame_h / frame_w)
        canvas_w = tile_w * grid_cols
        canvas_h = tile_h * grid_rows + 120  # extra for decision panel
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(args.output, fourcc, source_fps, (canvas_w, canvas_h))
            print(f"  Output video: {args.output} ({canvas_w}x{canvas_h})")

    # --- Map directions to indices ---
    dir_to_cameras: dict[str, list[int]] = {"N": [], "E": [], "S": [], "W": []}
    for i, direction in enumerate(args.directions):
        direction = direction.upper()
        if direction == "NS":
            dir_to_cameras["N"].append(i)
            dir_to_cameras["S"].append(i)
        elif direction == "EW":
            dir_to_cameras["E"].append(i)
            dir_to_cameras["W"].append(i)
        elif direction in dir_to_cameras:
            dir_to_cameras[direction].append(i)
        else:
            print(f"WARNING: Unknown direction '{direction}', skipping.")

    # --- Main processing loop ---
    print("\nProcessing (Ctrl-C to stop)...")
    frame_idx = 0
    current_phase = 0
    process_start = time.time()

    while True:
        frames: list[np.ndarray | None] = []
        all_done = True

        for cap in caps:
            ret, frame = cap.read()
            if ret:
                all_done = False
                frames.append(frame)
            else:
                frames.append(None)

        if all_done:
            break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        # --- Detect vehicles per camera ---
        cam_detections: list[list[dict]] = []
        cam_counts: list[int] = []
        for i, frame in enumerate(frames):
            if frame is not None:
                dets = detector.detect(frame)
                cam_detections.append(dets)
                cam_counts.append(len(dets))
            else:
                cam_detections.append([])
                cam_counts.append(0)

        # --- Aggregate counts per direction ---
        dir_counts = {"N": 0, "E": 0, "S": 0, "W": 0}
        for direction, cam_indices in dir_to_cameras.items():
            for ci in cam_indices:
                if ci < len(cam_counts):
                    dir_counts[direction] = max(dir_counts[direction], cam_counts[ci])

        # --- Brain decision (every N frames) ---
        if frame_idx % args.decision_interval == 0:
            obs = obs_builder.build(
                count_north=dir_counts["N"],
                count_east=dir_counts["E"],
                count_south=dir_counts["S"],
                count_west=dir_counts["W"],
            )
            t0 = time.perf_counter()
            action, logits = policy.decide(obs)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            if log_fh is not None:
                now = time.time()
                record = {
                    "ts": now,
                    "iso": datetime.now(timezone.utc).isoformat(),
                    "frame": frame_idx,
                    "counts": dir_counts,
                    "obs": [round(float(x), 5) for x in obs],
                    "logits": [round(x, 5) for x in logits],
                    "action": action,
                    "recommended_phase": _PHASE_BY_ACTION[action],
                    "phase_elapsed_s": round(now - obs_builder._phase_start_time, 3),
                    "ground_truth_phase": None,  # fill from a signal tap/annotation
                    "infer_ms": round(infer_ms, 3),
                }
                log_fh.write(json.dumps(record) + "\n")

            # Counterfactual rollout: the policy drives its own phase state.
            if action != current_phase:
                current_phase = action
                obs_builder.set_phase(current_phase)

        # --- Visualisation (only when producing video/window) ---
        if out is not None or args.show:
            canvas = np.full(
                (tile_h * grid_rows + 120, tile_w * grid_cols, 3), 30, dtype=np.uint8
            )
            for i, (frame, dets) in enumerate(zip(frames, cam_detections)):
                if frame is None:
                    continue
                is_green = (
                    (current_phase == 0 and args.directions[i].upper() in ("N", "S", "NS"))
                    or (current_phase == 1 and args.directions[i].upper() in ("E", "W", "EW"))
                )
                box_colour = _GREEN if is_green else _RED
                draw_detections(frame, dets, colour=box_colour)
                draw_count_badge(frame, cam_counts[i], args.directions[i].upper(), is_green)
                row = i // grid_cols
                col = i % grid_cols
                tile = cv2.resize(frame, (tile_w, tile_h))
                canvas[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = tile

            elapsed = time.time() - process_start
            fps = frame_idx / max(elapsed, 0.01)
            panel = build_decision_panel(
                tile_w * grid_cols, dir_counts, current_phase, frame_idx, fps
            )
            canvas[grid_rows * tile_h:, :] = panel

            if out is not None:
                out.write(canvas)
            if args.show:
                cv2.imshow("Semaforo Inteligente — Shadow", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        frame_idx += 1
        if frame_idx % 100 == 0:
            elapsed = time.time() - process_start
            fps = frame_idx / max(elapsed, 0.01)
            print(f"  Processed {frame_idx} frames "
                  f"({fps:.1f} fps, "
                  f"N={dir_counts['N']} E={dir_counts['E']} "
                  f"S={dir_counts['S']} W={dir_counts['W']}, "
                  f"Recommend={'N-S' if current_phase == 0 else 'E-W'})")

    # --- Cleanup ---
    for cap in caps:
        cap.release()
    if out is not None:
        out.release()
    if log_fh is not None:
        log_fh.close()
    if args.show:
        cv2.destroyAllWindows()

    total_time = time.time() - process_start
    print(f"\n✅ Done! Processed {frame_idx} frames in {total_time:.1f}s")
    if args.log:
        print(f"   Decision log: {args.log}")
    if args.output:
        print(f"   Output video: {args.output}")


if __name__ == "__main__":
    main()
