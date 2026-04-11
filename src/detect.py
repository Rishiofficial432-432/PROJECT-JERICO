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
CLASS_VEHICLE      = 4

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

    # 5. Vehicle Detection (New)
    vehicle_weights = "models/vehical_detection.pt"
    vehicle_model = YOLO(vehicle_weights) if os.path.exists(vehicle_weights) else None

    logger.info("YOLOv8 models (Quint) loaded successfully ✅")
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
    try:
        res = road_model(frame, verbose=False)
        detections = []
        h, w = frame.shape[:2]

        for r in res:
            if r.boxes is not None and len(r.boxes) > 0:
                # Object detection mode - detect any anomaly above threshold
                for box in r.boxes:
                    cls = int(box.cls[0])
                    conf_val = float(box.conf[0])
                    name = r.names[cls]
                    if conf_val >= conf:
                        detections.append([CLASS_ROAD_ANOMALY, conf_val, *box.xyxy[0].tolist(), name])
            elif r.probs is not None:  # Classification mode
                top1_idx = int(r.probs.top1)
                top1_conf = float(r.probs.top1conf)
                name = r.names[top1_idx]

                # Consider all events as potential road anomaly events
                # Include: Accident, Fight, Snatching, Robbery, Burglary, etc.
                road_event_keywords = ['accident', 'fight', 'snatching', 'robbery', 'burglary', 'assault', 'theft', 'vandalism', 'suspicious']
                if any(keyword in name.lower() for keyword in road_event_keywords) and top1_conf >= conf:
                    # Classifier doesn't give a bounding box, so use entire frame
                    detections.append([CLASS_ROAD_ANOMALY, top1_conf, 0, 0, w, h])
                # Also detect anything above higher threshold as general anomaly
                elif top1_conf >= conf * 2:  # Higher bar but still catch anything
                    detections.append([CLASS_ROAD_ANOMALY, top1_conf, 0, 0, w, h])

        return detections
    except Exception as e:
        logger.warning(f"Road anomaly detection failed: {e}")
        return []

def _run_vehicle(frame, conf):
    if not vehicle_model: return []
    res = vehicle_model(frame, verbose=False)
    # Mapping Bus, Car, Motorcycle, Pickup, Truck all to CLASS_VEHICLE
    return [[CLASS_VEHICLE, float(box.conf[0]), *box.xyxy[0].tolist()] 
            for r in res for box in r.boxes if box.conf[0] >= conf] if res else []

# ─────────────────────────────────────────────
#  Public Interface
# ─────────────────────────────────────────────

async def run_inference_async(frame, person_conf=0.5, weapon_conf=0.4, fire_conf=0.35, road_conf=0.4, vehical_conf=0.4, executor=None):
    """Run all 5 YOLO models in parallel via ThreadPoolExecutor."""
    if not executor:
        # Fallback to sync if no executor (not recommended for API)
        return run_inference(frame, person_conf, weapon_conf, fire_conf, road_conf)
        
    loop = asyncio.get_event_loop()
    tasks = [
        loop.run_in_executor(executor, _run_person,  frame, person_conf),
        loop.run_in_executor(executor, _run_weapon,  frame, weapon_conf),
        loop.run_in_executor(executor, _run_fire,    frame, fire_conf),
        loop.run_in_executor(executor, _run_road,    frame, road_conf),
        loop.run_in_executor(executor, _run_vehicle, frame, vehical_conf),
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
    v = _run_vehicle(frame, 0.4)
    return p + w + f + r + v

if __name__ == "__main__":
    print(f"Person model: {person_model}")
    print(f"Weapon model: {weapon_model}")
    print(f"Fire model:   {fire_model}")
    print(f"Road model:   {road_model}")
    print("Quad detection active ✅")
