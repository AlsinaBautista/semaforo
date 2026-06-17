"""Vehicle detector using YOLOv8 for traffic camera feeds.

Wraps the Ultralytics YOLOv8 model to detect and count vehicles
in frames from traffic cameras mounted on semaphore poles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ultralytics import YOLO


# COCO class IDs for vehicles
_VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """Detect vehicles in a single camera frame using YOLOv8.

    Args:
        weights: Path to the YOLOv8 ``.pt`` weights file.
        confidence: Minimum detection confidence threshold.
        device: Inference device (``"cpu"``, ``"mps"``, ``"cuda"``).
    """

    def __init__(
        self,
        weights: str | Path = "yolov8n.pt",
        confidence: float = 0.3,
        device: str = "cpu",
    ) -> None:
        self._model = YOLO(str(weights))
        self._confidence = confidence
        self._device = device

    def detect(self, frame: np.ndarray) -> list[dict[str, Any]]:
        """Run vehicle detection on a single BGR frame.

        Args:
            frame: BGR image as a numpy array (H, W, 3).

        Returns:
            List of detections, each a dict with keys:
            ``class_id``, ``class_name``, ``confidence``,
            ``bbox`` (x1, y1, x2, y2), ``center`` (cx, cy).
        """
        results = self._model.predict(
            frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )

        detections: list[dict[str, Any]] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in _VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                detections.append({
                    "class_id": cls_id,
                    "class_name": _VEHICLE_CLASSES[cls_id],
                    "confidence": float(box.conf[0]),
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                })

        return detections

    def count_vehicles(self, frame: np.ndarray) -> int:
        """Count total vehicles detected in a frame.

        Args:
            frame: BGR image as a numpy array.

        Returns:
            Number of vehicles detected.
        """
        return len(self.detect(frame))
