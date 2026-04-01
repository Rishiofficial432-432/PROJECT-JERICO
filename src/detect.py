import cv2
import numpy as np
import sys
import logging
import os

logger = logging.getLogger(__name__)

# Schema constants — no more magic numbers
CLASS_WEAPON = 0
CLASS_PERSON = 1

try:
    from ultralytics import YOLO
    person_model = YOLO("yolov8n.pt")            # detects people
    weapon_weights = "models/gun_bestweight.pt"
    if not os.path.exists(weapon_weights):
        weapon_weights = "models/best_model.pt"
    weapon_model = YOLO(weapon_weights)           # custom gun weights
    logger.info("YOLOv8 person model loaded successfully")
    logger.info("Custom weapon model loaded successfully")
except ImportError:
    person_model = None
    weapon_model = None
    logger.warning("ultralytics not found. Object detection disabled. Install via: pip install ultralytics")
except Exception as e:
    person_model = None
    weapon_model = None
    logger.error(f"Failed to load YOLO models: {e}")

def run_inference(frame, person_conf=0.5, weapon_conf=0.4):
    detections = []
    if person_model is None or weapon_model is None:
        return detections

    # People
    for r in person_model(frame, verbose=False):
        if r.boxes is not None:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and box.conf[0] >= person_conf:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    detections.append([CLASS_PERSON, float(box.conf[0]), x1,y1,x2,y2])

    # Guns — your custom model
    for r in weapon_model(frame, verbose=False):
        if r.boxes is not None:
            for box in r.boxes:
                if float(box.conf[0]) >= weapon_conf:
                    x1,y1,x2,y2 = box.xyxy[0].tolist()
                    detections.append([CLASS_WEAPON, float(box.conf[0]), x1,y1,x2,y2])

    return detections

if __name__ == "__main__":
    print(f"Person model: {person_model}")
    print(f"Weapon model: {weapon_model}")
    print("Dual detection active ✅")
