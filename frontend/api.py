"""
Jerico API Server — FastAPI backend for the frontend dashboard.
Serves live model metrics, handles footage uploads, and streams inference results.
Queue-buffered async processing for maximum performance.
"""

import sys
import os
import asyncio
import time
import json
import logging
import tempfile
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Optional

# Keep src imports working
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

from detect import (
    run_inference_async, 
    CLASS_WEAPON, CLASS_PERSON, CLASS_FIRE, CLASS_ROAD_ANOMALY,
    person_model, weapon_model, fire_model, road_model
)
from threat_logic import (
    WEAPON_CONF_THRESHOLD, FIRE_CONF_THRESHOLD, ROAD_ANOMALY_CONF_THRESHOLD,
    evaluate_threat, refine_detections
)
from worker_pool import get_executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Optional Model check
# ─────────────────────────────────────────────

try:
    from detect_anomaly import load_anomaly_model, lookup_features, predict_anomaly
    _anomaly_model, _anomaly_device = load_anomaly_model()
    _anomaly_ready = True
except Exception as e:
    logger.error(f"Anomaly model failed: {e}")
    _anomaly_model = _anomaly_device = None
    _anomaly_ready = False

try:
    from scene_understanding import SceneAnalyzer
    _scene_analyzer = SceneAnalyzer()
    _scene_ready = True
except Exception as e:
    logger.warning(f"Scene analyzer failed: {e}")
    _scene_analyzer = None
    _scene_ready = False

# ─────────────────────────────────────────────
#  In-memory state store
# ─────────────────────────────────────────────

_state = {
    "incidents_today": 0,
    "last_scan_ts": None,
    "anomaly_score": 0.0,
    "weapon_conf": 0.0,
    "fire_conf": 0.0,
    "road_conf": 0.0,
    "violence_score": 0.0,
    "scene_label": "No footage analysed",
    "scene_conf": 0.0,
    "threat_active": False,
    "threat_type": [],
    "active_camera": "—",
    "last_frame_b64": "",
    "last_filename": "—",
    "dispatch_count": 0,
}

_events: deque = deque(maxlen=100)
_events_lock = threading.Lock()

def _log_event(camera: str, event_type: str, confidence: Optional[float], status: str):
    with _events_lock:
        _events.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "camera": camera,
            "event": event_type,
            "confidence": f"{confidence:.2f}" if confidence is not None else "—",
            "status": status,
        })

# ─────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="Jerico API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def get_status():
    yolo_ready = all([person_model, weapon_model])
    return {
        "yolo":    {"ready": yolo_ready, "error": None},
        "anomaly": {"ready": _anomaly_ready, "error": None},
        "scene":   {"ready": _scene_ready, "error": None},
        "fire":    {"ready": bool(fire_model), "error": None},
        "road":    {"ready": bool(road_model), "error": None},
        "models_active": sum([yolo_ready, _anomaly_ready, _scene_ready, bool(fire_model), bool(road_model)]),
    }

@app.get("/api/metrics")
def get_metrics():
    last_scan = "Never"
    if _state["last_scan_ts"]:
        elapsed = int(time.time() - _state["last_scan_ts"])
        last_scan = f"{elapsed}s ago" if elapsed < 60 else f"{elapsed // 60}m ago"

    return {**_state, "last_scan": last_scan}

@app.get("/api/events")
def get_events():
    with _events_lock:
        return list(_events)

@app.get("/api/events/stream")
async def sse_stream():
    async def _gen():
        while True:
            payload = json.dumps({"metrics": get_metrics(), "events": get_events()})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(1)
    return StreamingResponse(_gen(), media_type="text/event-stream")

# ─────────────────────────────────────────────
#  Queue-Buffered Video Processing
# ─────────────────────────────────────────────

async def _frame_producer(cap, interval, queue, num_workers):
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % interval == 0:
            await queue.put((frame_idx, frame.copy()))
        frame_idx += 1
    for _ in range(num_workers):
        await queue.put(None) # Signal completion to all workers

async def _inference_worker(queue, results_list, th, executor):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        idx, frame = item
        # Parallel model inference
        dets = await run_inference_async(
            frame, 
            person_conf=th["person"], 
            weapon_conf=0.1, # Allow refinement logic to handle the threshold
            fire_conf=th["fire"], 
            road_conf=th["road"], 
            executor=executor
        )
        
        # Apply Accuracy Hardening (Spatial Validation)
        refined_dets = refine_detections(dets)
        r = _analyse_detections(refined_dets, th, frame)
        results_list.append((idx, frame, r))
        
        # Real-time Flagging: Update global state immediately if threat found or every 5th analysed frame
        is_threat = r["has_threat"] or r["has_fire"]
        if is_threat or idx % 5 == 0:
            frame_b64 = _encode_frame_b64(frame, r["boxes"])
            _state.update({
                "last_scan_ts": time.time(),
                "weapon_conf": max(_state["weapon_conf"], r.get("weapon_conf", 0)),
                "fire_conf": max(_state["fire_conf"], r.get("fire_conf", 0)),
                "road_conf": max(_state["road_conf"], r.get("road_conf", 0)),
                "threat_active": _state["threat_active"] or is_threat,
                "threat_type": list(set(_state["threat_type"]) | set(r["threat_types"])),
                "last_frame_b64": frame_b64,
            })
            if is_threat:
                _log_event(_state["active_camera"], " + ".join(r["threat_types"]), 
                          max(r["weapon_conf"], r["fire_conf"], r["road_conf"]), "active")
        
        queue.task_done()

@app.post("/api/upload")
async def upload_footage(
    file: UploadFile = File(...),
    camera_name: str = "CCTV-01",
    person_conf: float = 0.60,
    weapon_conf: float = 0.85,
    fire_conf:   float = 0.70,
    road_conf:   float = 0.60,
    anomaly_alert: float = 70.0,
    # Handle both naming conventions from frontend
    person_conf_threshold: float = None,
    weapon_conf_threshold: float = None,
):
    # Use threshold variants if provided
    if person_conf_threshold is not None:
        person_conf = person_conf_threshold
    if weapon_conf_threshold is not None:
        weapon_conf = weapon_conf_threshold

    ext = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    _state.update({
        "active_camera": camera_name,
        "threat_active": False,
        "threat_type": [],
        "weapon_conf": 0.0,
        "fire_conf": 0.0,
        "road_conf": 0.0,
        "last_filename": file.filename
    })

    th = {"person": person_conf, "weapon": weapon_conf, "fire": fire_conf, "road": road_conf}
    executor = get_executor()

    try:
        if ext in IMAGE_EXTS:
            frame = cv2.imread(tmp_path)
            dets = await run_inference_async(
                frame, 
                person_conf=th["person"], 
                weapon_conf=th["weapon"], 
                fire_conf=th["fire"], 
                road_conf=th["road"], 
                executor=executor
            )
            res = _analyse_detections(dets, th, frame)
            
            # Scene
            scene_l, scene_c = ("—", 0.0)
            if _scene_analyzer:
                scene_l, scene_c = await asyncio.to_thread(_scene_analyzer.analyze_frame, frame)

            frame_b64 = _encode_frame_b64(frame, res["boxes"])
            _update_state_post_process(res, scene_l, scene_c, frame_b64, file.filename)
            total_dets = len(res.get("boxes", []))
            return {
                **res,
                "media_type": "image",
                "frame_b64": frame_b64,
                "scene_label": scene_l,
                "camera": camera_name,
                "total_detections": total_dets,
            }
        
        else:
            cap = cv2.VideoCapture(tmp_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            # Continuous analysis: Sample every 0.3s or at least 1 frame
            interval = max(1, int(fps * 0.3)) 
            
            queue = asyncio.Queue(maxsize=20)
            results = []
            num_workers = 3
            
            # Start producer and workers
            producer = asyncio.create_task(_frame_producer(cap, interval, queue, num_workers))
            workers = [asyncio.create_task(_inference_worker(queue, results, th, executor)) for _ in range(num_workers)]
            
            await asyncio.gather(producer, *workers)
            cap.release()

            if not results: return {"error": "No frames processed"}
            results.sort(key=lambda x: x[0]) # Chronological order

            # ── Accuracy Hardening: Temporal Persistence Filter ─────────────
            def check_persistence(type_key, min_frames=2, instant_threshold=0.98):
                type_results = [r[2] for r in results]
                confirmed_conf = 0.0
                active_types = set()
                thresh_map = {"weapon_conf": WEAPON_CONF_THRESHOLD, "fire_conf": FIRE_CONF_THRESHOLD, "road_conf": ROAD_ANOMALY_CONF_THRESHOLD}
                current_thresh = thresh_map.get(type_key, 0.5)

                for i in range(len(type_results)):
                    r_frame = type_results[i]
                    conf = r_frame.get(type_key, 0)
                    if conf >= instant_threshold:
                        confirmed_conf = max(confirmed_conf, conf)
                        active_types.add(type_key.replace("_conf", "").capitalize())
                    elif conf >= current_thresh:
                        # Window: 2 frames before, 2 frames after
                        window = type_results[max(0, i-2):i+3]
                        count = sum(1 for wr in window if wr.get(type_key, 0) >= current_thresh * 0.85)
                        if count >= min_frames:
                            confirmed_conf = max(confirmed_conf, conf)
                            active_types.add(type_key.replace("_conf", "").capitalize())
                return confirmed_conf, active_types

            max_w, types_w = check_persistence("weapon_conf")
            max_f, types_f = check_persistence("fire_conf")
            max_r, types_r = check_persistence("road_conf")
            threat_types = types_w | types_f | types_r
            
            # Anomaly correlation
            anom_score = 0.0
            if _anomaly_ready:
                f_path = lookup_features(file.filename)
                if f_path:
                    scores = predict_anomaly(f_path, _anomaly_model, _anomaly_device)
                    anom_score = float(np.max(scores)) * 100

            # Final Logic: Correlate with Anomaly Score
            has_threat = max_w > 0 or max_r > 0
            has_fire = max_f > 0
            if anom_score < 30.0 and max(max_w, max_f, max_r) < 0.95:
                # If scene is "too normal", ignore marginal detections
                has_threat = has_fire = False
                threat_types = []

            # Visual fallback
            worst = sorted(results, key=lambda x: (x[2]["weapon_conf"], x[2]["fire_conf"], x[2]["road_conf"]), reverse=True)[0]
            frame_b64 = _encode_frame_b64(worst[1], worst[2]["boxes"])

            _update_state_post_process(
                {"weapon_conf": max_w, "fire_conf": max_f, "road_conf": max_r, "has_threat": has_threat, "has_fire": has_fire, "threat_types": list(threat_types)},
                "—", 0.0, frame_b64, file.filename, anom_score
            )

            duration = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / (cap.get(cv2.CAP_PROP_FPS) or 30.0))
            return {
                "media_type": "video",
                "frame_b64": frame_b64,
                "weapon_conf": max_w,
                "fire_conf": max_f,
                "road_conf": max_r,
                "threat_types": list(threat_types),
                "anomaly_score": anom_score,
                "camera": camera_name,
                "has_threat": has_threat,
                "has_fire": has_fire,
                "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration_seconds": round(duration, 1),
                "total_detections": sum(len(r[2].get("boxes", [])) for r in results),
            }

    finally:
        os.unlink(tmp_path)

def _update_state_post_process(res, scene_l, scene_c, b64, fname, anom=0.0):
    _state.update({
        "last_scan_ts": time.time(),
        "weapon_conf": res.get("weapon_conf", 0),
        "fire_conf": res.get("fire_conf", 0),
        "road_conf": res.get("road_conf", 0),
        "scene_label": scene_l,
        "scene_conf": scene_c,
        "anomaly_score": anom,
        "threat_active": res["has_threat"] or res["has_fire"],
        "threat_type": res["threat_types"],
        "last_frame_b64": b64,
        "last_filename": fname,
    })
    if _state["threat_active"]:
        _state["incidents_today"] += 1
        _log_event(_state["active_camera"], " + ".join(res["threat_types"]), max(res["weapon_conf"], res["fire_conf"], res["road_conf"]), "active")

# ─────────────────────────────────────────────
#  Detection & Encoding Helpers
# ─────────────────────────────────────────────

def _analyse_detections(detections, th, frame):
    h, w = frame.shape[:2]
    res = {"has_threat": False, "has_fire": False, "weapon_conf": 0.0, "fire_conf": 0.0, "road_conf": 0.0, "threat_types": [], "boxes": []}
    
    for det in detections:
        cls, conf, x1, y1, x2, y2 = det
        label = {CLASS_WEAPON: "weapon", CLASS_PERSON: "person", CLASS_FIRE: "fire", CLASS_ROAD_ANOMALY: "road_anomaly"}.get(int(cls), "unknown")
        
        res["boxes"].append({"class": label, "conf": round(conf, 2), "bbox_px": [int(x1), int(y1), int(x2), int(y2)]})
        
        if int(cls) == CLASS_WEAPON:
            res["has_threat"] = True
            res["weapon_conf"] = max(res["weapon_conf"], conf)
            if "Weapon" not in res["threat_types"]: res["threat_types"].append("Weapon")
        elif int(cls) == CLASS_FIRE:
            res["has_fire"] = True
            res["fire_conf"] = max(res["fire_conf"], conf)
            if "Fire" not in res["threat_types"]: res["threat_types"].append("Fire")
        elif int(cls) == CLASS_ROAD_ANOMALY:
            res["has_threat"] = True
            res["road_conf"] = max(res["road_conf"], conf)
            if "Road Anomaly" not in res["threat_types"]: res["threat_types"].append("Road Anomaly")
            
    return res

_COLORS = {"weapon": (0,0,220), "fire": (0,120,240), "person": (220,80,0), "road_anomaly": (147,51,234)}

def _encode_frame_b64(frame, boxes):
    out = frame.copy()
    h, w = out.shape[:2]
    if w > 1280:
        scale = 1280 / w
        out = cv2.resize(out, (1280, int(h * scale)))
    
    for det in boxes:
        color = _COLORS.get(det["class"], (180,180,180))
        x1, y1, x2, y2 = det["bbox_px"]
        # Basic scale adjust if resized
        if w > 1280:
            s = 1280/w
            x1, y1, x2, y2 = int(x1*s), int(y1*s), int(x2*s), int(y2*s)
        
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"{det['class']} {det['conf']:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    _, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# (Removed duplicate definition)

@app.get("/api/last_frame")
def get_last_frame():
    return {"frame_b64": _state["last_frame_b64"], "filename": _state["last_filename"], "camera": _state["active_camera"], "threat_active": _state["threat_active"], "threat_type": _state["threat_type"], "has_frame": bool(_state["last_frame_b64"])}

@app.post("/api/dispatch")
def trigger_dispatch(camera: str = "—", threat_type: str = "Unknown"):
    _state["dispatch_count"] += 1
    ts = datetime.now().strftime("%H:%M:%S")
    _log_event(camera, f"🚨 Emergency dispatch #{_state['dispatch_count']}", None, "dispatched")
    return {"dispatched": True, "dispatch_no": _state["dispatch_count"], "timestamp": ts}

@app.on_event("startup")
def on_startup():
    logger.info("=== Jerico API v2.0 Started ===")
    logger.info(f"  Models: Person & Weapon={'✅' if person_model and weapon_model else '❌'}, Fire={'✅' if fire_model else '❌'}, Road={'✅' if road_model else '❌'}")

@app.on_event("shutdown")
def on_shutdown():
    logger.info("=== Jerico API v2.0 Shutting Down ===")
    get_executor().shutdown(wait=False, cancel_futures=True)
    logger.info("Worker pool shutdown complete.")
