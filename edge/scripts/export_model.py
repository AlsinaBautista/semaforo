"""
Export YOLOv8n to ONNX format for the Edge module.
"""

import sys
import logging
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).resolve().parent
EDGE_DIR = SCRIPT_DIR.parent
MODELS_DIR = EDGE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

try:
    from ultralytics import YOLO
except ImportError:
    logging.error("ultralytics is not installed. Please install it using: pip install ultralytics")
    sys.exit(1)

def main():
    logging.info("Downloading YOLOv8n model...")
    # Load a pretrained YOLOv8n model
    model = YOLO('yolov8n.pt')

    out_path = MODELS_DIR / "yolov8n_vehicles.onnx"
    
    logging.info(f"Exporting model to ONNX format at: {out_path}")
    
    # Export the model
    # We specify dynamic=False to fix the input shape to 640x640, 
    # which makes it easier for C++ ONNX Runtime to allocate memory.
    success = model.export(
        format='onnx',
        imgsz=640,
        dynamic=False,
        opset=12,
        simplify=True
    )
    
    if success:
        # ultralytics saves it in the current working directory as yolov8n.onnx
        # We need to move it to our target directory
        exported_file = Path("yolov8n.onnx")
        if exported_file.exists():
            exported_file.rename(out_path)
            logging.info(f"✅ Model successfully exported to {out_path}")
        else:
            logging.error("Model exported but output file not found in current directory.")
    else:
        logging.error("Model export failed.")

if __name__ == "__main__":
    main()
