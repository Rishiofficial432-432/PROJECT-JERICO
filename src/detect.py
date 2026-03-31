import cv2
import numpy as np
import sys

try:
    from ultralytics import YOLO
    # Load YOLOv8 nano model (fastest) and keep it in memory
    model = YOLO("yolov8n.pt")
except ImportError:
    model = None
    print("Warning: 'ultralytics' not found. Run 'pip install ultralytics' first.", file=sys.stderr)

def run_inference(frame):
    """
    Real-time inference using YOLOv8, detecting 'person' and COCO 'weapon' equivalents.
    Outputs: [class_id, confidence, x_min, y_min, x_max, y_max]
    """
    detections = []
    
    if model is None:
        return detections
        
    # Run YOLO with low verbosity
    results = model(frame, verbose=False)
    
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                cls_idx = int(box.cls[0].item())
                conf = box.conf[0].item()
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Map COCO classes to dashboard schema -> 0: Weapon/Threat, 1: Person
                # COCO 'person' is class 0
                if cls_idx == 0:
                    detections.append([1, conf, x1, y1, x2, y2])
                # COCO dangerous objects: knife (43), baseball bat (34), scissors (76)
                elif cls_idx in [43, 34, 76]:
                    detections.append([0, conf, x1, y1, x2, y2])
                    
    return detections

if __name__ == "__main__":
    print("YOLOv8 Detection active.")
