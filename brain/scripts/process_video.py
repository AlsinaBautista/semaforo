#!/usr/bin/env python3
"""Process video from AI City Challenge dataset using YOLOv8 tracking.

Extracts vehicle counts and tracks using `ultralytics` YOLOv8.
Applies an optional ROI (Region of Interest) mask to ignore parked cars
or vehicles outside the road. Outputs tracked bounding boxes to a JSON
file and an annotated .mp4 video for manual verification.

Usage:
    python scripts/process_video.py \
        --video ../datasets/ai_city_challenge/train/S01/c001/vdo.avi \
        --roi ../datasets/ai_city_challenge/train/S01/c001/roi.jpg \
        --out-json S01_c001_tracks.json \
        --out-video S01_c001_output.mp4
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Video Processor for AIC22")
    parser.add_argument("--video", type=str, required=True, help="Input video file (.avi/.mp4)")
    parser.add_argument("--roi", type=str, help="ROI image mask (.jpg/.png)")
    parser.add_argument("--weights", type=str, default="../../yolov8n.pt", help="YOLOv8 weights")
    parser.add_argument("--out-json", type=str, default="yolo_tracks.json", help="Output JSON file")
    parser.add_argument("--out-video", type=str, default="yolo_output.mp4", help="Output video file")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--show", action="store_true", help="Show video during processing")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load YOLO
    print(f"Loading YOLO model: {args.weights}")
    model = YOLO(args.weights)

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Could not open video {args.video}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video loaded: {width}x{height} @ {fps} FPS ({total_frames} frames)")

    # Load ROI if provided
    roi_mask = None
    if args.roi and Path(args.roi).exists():
        print(f"Loading ROI mask from: {args.roi}")
        # Load grayscale
        roi_img = cv2.imread(args.roi, cv2.IMREAD_GRAYSCALE)
        if roi_img is not None:
            # Ensure ROI matches video size
            if roi_img.shape != (height, width):
                roi_img = cv2.resize(roi_img, (width, height))
            # Binarize mask: > 127 is 255 (white, keep), else 0 (black, ignore)
            _, roi_mask = cv2.threshold(roi_img, 127, 255, cv2.THRESH_BINARY)
        else:
            print(f"WARNING: Could not load ROI mask {args.roi}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(args.out_video, fourcc, fps, (width, height))
    print(f"Writing annotated video to: {args.out_video}")

    # Tracking data struct
    # Format: { frame_index: [ { id: 1, class: 'car', bbox: [x1, y1, x2, y2] }, ... ] }
    tracking_data = {
        "metadata": {
            "video": args.video,
            "width": width,
            "height": height,
            "fps": fps,
        },
        "frames": {}
    }

    frame_idx = 0
    unique_ids = set()

    print("Processing...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if args.max_frames > 0 and frame_idx >= args.max_frames:
            break

        # Apply ROI mask to hide non-road areas from YOLO
        # We multiply the frame by the mask (broadcasting to 3 channels)
        masked_frame = frame
        if roi_mask is not None:
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)

        # Run YOLO tracking (persist=True keeps IDs across frames)
        # tracker="botsort.yaml" is standard, but default is fine.
        results = model.track(masked_frame, persist=True, conf=args.conf, verbose=False)
        
        frame_detections = []
        
        if results and results[0].boxes:
            boxes = results[0].boxes
            
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                
                # Some detections might not get an ID immediately
                track_id = int(box.id[0]) if box.id is not None else -1
                if track_id != -1:
                    unique_ids.add(track_id)
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                frame_detections.append({
                    "id": track_id,
                    "class": VEHICLE_CLASSES[cls_id],
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
                })
                
                # Draw on the original unmasked frame
                color = (0, 255, 0) if track_id != -1 else (0, 0, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"#{track_id} {VEHICLE_CLASSES[cls_id]}"
                cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        tracking_data["frames"][frame_idx] = frame_detections
        
        # Info overlay
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Unique Vehicles: {len(unique_ids)}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out_vid.write(frame)

        if args.show:
            # Resize for viewing
            view_w = 1280
            view_h = int(height * (view_w / width))
            view_frame = cv2.resize(frame, (view_w, view_h))
            cv2.imshow("YOLO Tracking", view_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Processed {frame_idx} frames... Unique vehicles so far: {len(unique_ids)}")

    # Cleanup
    cap.release()
    out_vid.release()
    if args.show:
        cv2.destroyAllWindows()

    # Save JSON
    print(f"Saving tracking data to: {args.out_json}")
    with open(args.out_json, "w") as f:
        json.dump(tracking_data, f, indent=2)

    print(f"✅ Finished! Processed {frame_idx} frames.")
    print(f"Total unique vehicles detected: {len(unique_ids)}")

if __name__ == "__main__":
    main()
