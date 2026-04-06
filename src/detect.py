import cv2
import numpy as np
import sys
import logging
import os
import asyncio

logger = logging.getLogger(__name__)

# Schema constants — no more magic numbers
CLASS_WEAPON       = 0
CLASS_PERSON       = 1
CLASS_FIRE         = 2
CLASS_ROAD_ANOMALY = 3

try:
    from ultralytics import YOLO
    
    # 1. People
    person_model = YOLO("yolov8n.pt")
    
    # 2. Weapons
    weapon_weights = "models/gun_bestweight.pt"
    if not os.path.exists(weapon_weights):
        weapon_weights = "models/best_model.pt"
    weapon_model = YOLO(weapon_weights)
    
    # 3. Fire
    fire_weights = "models/fire_detection.pt"
    fire_model = YOLO(fire_weights) if os.path.exists(fire_weights) else None
    
    # 4. Road Anomaly (New)
    road_weights = "models/road_anomaly.pt"
    road_model = YOLO(road_weights) if os.path.exists(road_weights) else None

    logger.info("YOLOv8 models loaded successfully ✅")
    if road_model: logger.info("Road Anomaly model loaded successfully ✅")
    
    # Warmup models to prevent fuse race conditions during parallel inference
    try:
        logger.info("Warming up models to fuse layers...")
        dummy = np.zeros((320, 320, 3), dtype=np.uint8)
        if person_model: person_model(dummy, verbose=False)
        if weapon_model: weapon_model(dummy, verbose=False)
        if fire_model: fire_model(dummy, verbose=False)
        if road_model: road_model(dummy, verbose=False)
        logger.info("Warmup complete ✅")
    except Exception as e:
        logger.warning(f"Warmup failed (safe to ignore): {e}")
    
except ImportError:
    person_model = weapon_model = fire_model = road_model = None
    logger.warning("ultralytics not found. Object detection disabled.")
except Exception as e:
    person_model = weapon_model = fire_model = road_model = None
    logger.error(f"Failed to load YOLO models: {e}")

# ─────────────────────────────────────────────
#  Private inference helpers (blocking)
# ─────────────────────────────────────────────

def _run_person(frame, conf):
    if not person_model: return []
    res = person_model(frame, verbose=False)
    return [[CLASS_PERSON, float(box.conf[0]), *box.xyxy[0].tolist()] 
            for r in res for box in r.boxes if int(box.cls[0]) == 0 and box.conf[0] >= conf] if res else []

def _run_weapon(frame, conf):
    if not weapon_model: return []
    res = weapon_model(frame, verbose=False)
    return [[CLASS_WEAPON, float(box.conf[0]), *box.xyxy[0].tolist()] 
            for r in res for box in r.boxes if box.conf[0] >= conf] if res else []

def _run_fire(frame, conf):
    if not fire_model: return []
    res = fire_model(frame, verbose=False)
    return [[CLASS_FIRE, float(box.conf[0]), *box.xyxy[0].tolist()] 
            for r in res for box in r.boxes if box.conf[0] >= conf] if res else []

def _run_road(frame, conf):
    if not road_model: return []
    res = road_model(frame, verbose=False)
    detections = []
    h, w = frame.shape[:2]
    
    for r in res:
        if r.probs is not None:  # Road anomaly is an image classification model
            top1_idx = int(r.probs.top1)
            top1_conf = float(r.probs.top1conf)
            name = r.names[top1_idx]
            
            # Consider Accident, Fight, Snatching as road anomaly events
            if name in ['Accident', 'Fight', 'Snatching'] and top1_conf >= conf:
                # Classifier doesn't give a bounding box, so use entire frame
                detections.append([CLASS_ROAD_ANOMALY, top1_conf, 0, 0, w, h])
                
    return detections

# ─────────────────────────────────────────────
#  Public Interface
# ─────────────────────────────────────────────

async def run_inference_async(frame, person_conf=0.5, weapon_conf=0.4, fire_conf=0.35, road_conf=0.4, executor=None):
    """Run all 4 YOLO models in parallel via ThreadPoolExecutor."""
    if not executor:
        # Fallback to sync if no executor (not recommended for API)
        return run_inference(frame, person_conf, weapon_conf, fire_conf, road_conf)
        
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, _run_person, frame, person_conf),
        loop.run_in_executor(executor, _run_weapon, frame, weapon_conf),
        loop.run_in_executor(executor, _run_fire,   frame, fire_conf),
        loop.run_in_executor(executor, _run_road,   frame, road_conf),
    ]
    results = await asyncio.gather(*tasks)
    # Flatten list of lists
    return [det for sublist in results for det in sublist]

def run_inference(frame, person_conf=0.5, weapon_conf=0.4, fire_conf=0.35, road_conf=0.4):
    """Sync version (legacy/dashboard) - runs models one-by-one."""
    p = _run_person(frame, person_conf)
    w = _run_weapon(frame, weapon_conf)
    f = _run_fire(frame, fire_conf)
    r = _run_road(frame, road_conf)
    return p + w + f + r

if __name__ == "__main__":
    print(f"Person model: {person_model}")
    print(f"Weapon model: {weapon_model}")
    print(f"Fire model:   {fire_model}")
    print(f"Road model:   {road_model}")
    print("Quad detection active ✅")
