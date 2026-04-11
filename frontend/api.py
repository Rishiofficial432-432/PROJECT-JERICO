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
from pydantic import BaseModel

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

from detect import (
    run_inference_async, 
    CLASS_WEAPON, CLASS_PERSON, CLASS_FIRE, CLASS_ROAD_ANOMALY, CLASS_VEHICLE,
    person_model, weapon_model, fire_model, road_model, vehicle_model
)
from threat_logic import (
    WEAPON_CONF_THRESHOLD, FIRE_CONF_THRESHOLD, ROAD_ANOMALY_CONF_THRESHOLD,
    evaluate_threat, refine_detections
)
from worker_pool import get_executor
from video_utils import apply_temporal_anchor, format_mmss_ms

try:
    import importlib
    dotenv_mod = importlib.import_module("dotenv")
    dotenv_mod.load_dotenv()
except Exception:
    # dotenv is optional at runtime; shell exports still work.
    pass

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
    "scene_description": "Waiting for footage...",
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
_live_lock = asyncio.Lock()

def _log_event(camera: str, event_type: str, confidence: Optional[float], status: str):
    with _events_lock:
        _events.appendleft({
            "time": datetime.now().strftime("%H:%M:%S"),
            "camera": camera,
            "event": event_type,
            "confidence": f"{confidence:.2f}" if confidence is not None else "—",
            "status": status,
        })


class LiveFramePayload(BaseModel):
    frame_b64: str
    camera_name: str = "Webcam-01"
    person_conf: float = 0.45
    weapon_conf: float = 0.55
    fire_conf: float = 0.82
    road_conf: float = 0.40
    vehicle_conf: float = 0.35


def _decode_data_url_to_frame(data_url: str):
    if not data_url:
        return None
    try:
        if "," in data_url:
            data_url = data_url.split(",", 1)[1]
        raw = base64.b64decode(data_url)
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Failed decoding live frame: {e}")
        return None

# ─────────────────────────────────────────────
#  FastAPI app
# ─────────────────────────────────────────────

app = FastAPI(title="Jerico API", version="2.0.0")

def read_index():
    return RedirectResponse(url="/index.html")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def get_status():
    yolo_ready = all([person_model, weapon_model])
    model_details = {
        "person_detector": {
            "label": "YOLOv8 Person Detector",
            "family": "YOLOv8",
            "weights": "yolov8n.pt",
            "ready": bool(person_model),
        },
        "weapon_detector": {
            "label": "Weapon Detector",
            "family": "YOLOv8",
            "weights": "models/gun_bestweight.pt (fallback: models/best_model.pt)",
            "ready": bool(weapon_model),
        },
        "fire_detector": {
            "label": "Fire Detector",
            "family": "YOLOv8",
            "weights": "models/fire_detection.pt",
            "ready": bool(fire_model),
        },
        "road_detector": {
            "label": "Road Anomaly Detector",
            "family": "YOLOv8",
            "weights": "models/road_anomaly.pt",
            "ready": bool(road_model),
        },
        "vehicle_detector": {
            "label": "Vehicle Detector",
            "family": "YOLOv8",
            "weights": "models/vehical_detection.pt",
            "ready": bool(vehicle_model),
        },
        "anomaly_model": {
            "label": "Temporal Social Anomaly",
            "family": "MIL",
            "weights": "models/best_anomaly_model.pth",
            "ready": bool(_anomaly_ready),
        },
        "scene_reasoner": {
            "label": "Scene Reasoner",
            "family": "Gemini 3.1 Pro (fallback local)",
            "weights": getattr(_scene_analyzer, "_model_name", "gemini-3.1-pro-preview"),
            "ready": bool(_scene_ready),
        },
    }
    return {
        "yolo":    {"ready": yolo_ready, "error": None},
        "anomaly": {"ready": _anomaly_ready, "error": None},
        "scene":   {"ready": _scene_ready, "error": None},
        "fire":    {"ready": bool(fire_model), "error": None},
        "road":    {"ready": bool(road_model), "error": None},
        "vehicle": {"ready": bool(vehicle_model), "error": None},
        "models_active": sum(1 for m in model_details.values() if m["ready"]),
        "models_total": len(model_details),
        "architecture_mode": "Reasoning Firewall (YOLO + Temporal Anchor + Gemini)",
        "threat_logic": {
            "weapon_conf_threshold": WEAPON_CONF_THRESHOLD,
            "fire_conf_threshold": FIRE_CONF_THRESHOLD,
            "road_conf_threshold": ROAD_ANOMALY_CONF_THRESHOLD,
            "armed_person_pixel_distance": 120,
            "tracker": "CentroidTracker",
        },
        "models": model_details,
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


@app.get("/api/last_frame")
def get_last_frame():
    return {
        "has_frame": bool(_state.get("last_frame_b64")),
        "frame_b64": _state.get("last_frame_b64", ""),
        "camera": _state.get("active_camera", "—"),
        "filename": _state.get("last_filename", "live"),
        "threat_active": _state.get("threat_active", False),
    }


@app.post("/api/live/frame")
async def ingest_live_frame(payload: LiveFramePayload):
    if _live_lock.locked():
        return {"accepted": False, "busy": True}

    async with _live_lock:
        frame = _decode_data_url_to_frame(payload.frame_b64)
        if frame is None:
            return JSONResponse(status_code=400, content={"error": "Invalid live frame payload"})

        th = {
            "person": payload.person_conf,
            "weapon": payload.weapon_conf,
            "fire": payload.fire_conf,
            "road": payload.road_conf,
            "vehicle": payload.vehicle_conf,
        }

        dets = await run_inference_async(
            frame,
            person_conf=th["person"],
            weapon_conf=th["weapon"],
            fire_conf=th["fire"],
            road_conf=th["road"],
            vehical_conf=th["vehicle"],
            executor=get_executor(),
        )
        refined = refine_detections(dets)
        res = _analyse_detections(refined, th, frame)

        scene_l, scene_c = ("—", 0.0)
        if _scene_analyzer:
            scene_l, scene_c = await asyncio.to_thread(_scene_analyzer.analyze_frame, frame)

        frame_b64 = _encode_frame_b64(frame, res["boxes"])
        _state.update({
            "active_camera": payload.camera_name,
            "last_filename": "live_webcam",
        })
        _update_state_post_process(res, scene_l, scene_c, frame_b64, "live_webcam", 0.0)

        if res["has_threat"] or res["has_fire"]:
            _log_event(payload.camera_name, " + ".join(res["threat_types"] or ["threat"]),
                       max(res.get("weapon_conf", 0), res.get("fire_conf", 0), res.get("road_conf", 0)), "active")

        return {
            "accepted": True,
            "busy": False,
            "threat_active": _state.get("threat_active", False),
            "threat_types": _state.get("threat_type", []),
        }

# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
#  Detection & Encoding Helpers
# ─────────────────────────────────────────────

def _encode_frame_b64(frame, boxes=None):
    """Encodes a frame to base64 for the frontend, optionally drawing boxes."""
    try:
        if boxes is not None:
            temp_frame = frame.copy()
            for b in boxes:
                if len(b) >= 4:
                    x1, y1, x2, y2 = map(int, b[:4])
                    cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', temp_frame)
        else:
            _, buffer = cv2.imencode('.jpg', frame)
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()
    except Exception as e:
        logger.error(f"Encoding error: {e}")
        return ""

def _analyse_detections(detections, th, frame):
    """
    Bridge between raw detections and Situation Summary.
    Applies thresholds and identifies active threat flags.
    """
    # 1. Get situation object from threat_logic (which uses refinement)
    threats = evaluate_threat(detections)
    
    # 2. Extract metrics for dashboard
    w_conf = max([t["confidence"] for t in threats if "weapon" in t["type"].lower()] or [0.0])
    f_conf = max([t["confidence"] for t in threats if t["type"] == "fire"] or [0.0])
    r_conf = max([t["confidence"] for t in threats if t["type"] in ["road anomaly detected", "road_anomaly"]] or [0.0])
    
    # Return unified dict for state updates
    return {
        "has_threat": len(threats) > 0,
        "has_fire": any(t["type"] == "fire" for t in threats),
        "threat_types": list(set(t["type"] for t in threats)),
        "weapon_conf": w_conf,
        "fire_conf": f_conf,
        "road_conf": r_conf,
        "boxes": [d[2:6] for d in detections], # x1,y1,x2,y2
        "vehicle_count": sum(1 for d in detections if d[0] == 4) # CLASS_VEHICLE is 4
    }

#  Queue-Buffered Video Processing
# ─────────────────────────────────────────────

async def _frame_producer(cap, interval, fps_val, queue, num_workers):
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        if frame_idx % interval == 0:
            anchored = apply_temporal_anchor(frame.copy(), frame_idx / max(fps_val, 1.0))
            await queue.put((frame_idx, anchored))
        frame_idx += 1
    for _ in range(num_workers):
        await queue.put(None) # Signal completion to all workers

async def _inference_worker(queue, results_list, th, executor, yield_queue=None, total_frames=1):
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            if yield_queue:
                await yield_queue.put(None)
            break
        idx, frame = item
        try:
            # Parallel model inference
            dets = await run_inference_async(
                frame,
                person_conf=th.get("person", 0.5),
                weapon_conf=0.1, # Allow refinement logic to handle the threshold
                fire_conf=th.get("fire", 0.5),
                road_conf=th.get("road", 0.5),
                vehical_conf=th.get("vehicle", 0.5),
                executor=executor
            )

            # Apply Accuracy Hardening (Spatial Validation)
            refined_dets = refine_detections(dets)
            r = _analyse_detections(refined_dets, th, frame)
            results_list.append((idx, frame, r))

            if yield_queue:
                b64 = _encode_frame_b64(frame, r["boxes"])
                pct = min(99, int((idx / total_frames) * 100)) if total_frames > 0 else 0
                await yield_queue.put({"progress": pct, "frame_b64": b64})

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
        except Exception as e:
            logger.error(f"Worker frame processing failed at idx={idx}: {e}")
            if yield_queue:
                pct = min(99, int((idx / total_frames) * 100)) if total_frames > 0 else 0
                await yield_queue.put({"progress": pct})
        finally:
            queue.task_done()
@app.post("/api/upload")
async def upload_footage(
    file: UploadFile = File(...),
    camera_name: str = "CCTV-01",
    person_conf: float = 0.45,
    weapon_conf: float = 0.55,
    fire_conf:   float = 0.82,
    road_conf:   float = 0.40,
    anomaly_alert: float = 65.0,
    vehicle_conf: float = 0.35,
    person_conf_threshold: float = None,
    weapon_conf_threshold: float = None,
):
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

    th = {"person": person_conf, "weapon": weapon_conf, "fire": fire_conf, "road": road_conf, "vehicle": vehicle_conf}
    executor = get_executor()

    if ext in IMAGE_EXTS:
        try:
            frame = cv2.imread(tmp_path)
            if frame is None:
                return JSONResponse(status_code=400, content={"error": "Could not read image file."})
            frame = apply_temporal_anchor(frame, 0.0)
            
            dets = await run_inference_async(
                frame, 
                person_conf=th["person"], 
                weapon_conf=th["weapon"], 
                fire_conf=th["fire"], 
                road_conf=th["road"], 
                vehical_conf=th["vehicle"],
                executor=executor
            )
            res = _analyse_detections(dets, th, frame)
            
            scene_l, scene_c = ("—", 0.0)
            if _scene_analyzer:
                scene_l, scene_c = await asyncio.to_thread(_scene_analyzer.analyze_frame, frame)

            frame_b64 = _encode_frame_b64(frame, res["boxes"])
            _update_state_post_process(res, scene_l, scene_c, frame_b64, file.filename)
            return {
                **res,
                "media_type": "image",
                "frame_b64": frame_b64,
                "scene_label": scene_l,
                "scene_description": scene_l,
                "camera": camera_name,
                "total_detections": len(res.get("boxes", [])),
            }
        finally:
            try:
                os.unlink(tmp_path)
            except: pass
    else:
        # Video Processing setup
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            try:
                os.unlink(tmp_path)
            except: pass
            return JSONResponse(status_code=400, content={"error": "Could not open video file."})
            
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps_val * 0.3)) 
        
        async def _vid_generator():
            try:
                queue = asyncio.Queue(maxsize=20)
                yield_queue = asyncio.Queue(maxsize=32)
                results = []
                num_workers = 3
                
                producer = asyncio.create_task(_frame_producer(cap, interval, fps_val, queue, num_workers))
                workers = [asyncio.create_task(_inference_worker(queue, results, th, executor, yield_queue, total)) for _ in range(num_workers)]
                
                active_workers = num_workers
                while active_workers > 0:
                    try:
                        item = await asyncio.wait_for(yield_queue.get(), timeout=20)
                    except asyncio.TimeoutError:
                        # Avoid indefinite hang if worker exits unexpectedly.
                        if all(w.done() for w in workers):
                            break
                        continue
                    if item is None:
                        active_workers -= 1
                    else:
                        yield json.dumps(item) + "\n"

                # Surface worker exceptions if any (for logs) without killing response.
                await producer
                await asyncio.gather(*workers, return_exceptions=True)
                
                if cap and cap.isOpened():
                    cap.release()

                if not results:
                    yield json.dumps({"error": "No frames processed"}) + "\n"
                    return

                results.sort(key=lambda x: x[0])

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
                            for ty in r_frame.get("threat_types", []):
                                if type_key.replace("_conf", "").lower() in ty.lower():
                                    active_types.add(ty)
                        elif conf >= current_thresh:
                            count = sum(1 for wr in type_results[max(0, i-2):i+3] if wr.get(type_key, 0) >= current_thresh * 0.85)
                            if count >= min_frames:
                                confirmed_conf = max(confirmed_conf, conf)
                                for ty in r_frame.get("threat_types", []):
                                    if type_key.replace("_conf", "").lower() in ty.lower():
                                        active_types.add(ty)
                    return confirmed_conf, active_types

                max_w, types_w = check_persistence("weapon_conf", min_frames=1)
                max_f, types_f = check_persistence("fire_conf", min_frames=2)
                max_r, types_r = check_persistence("road_conf", min_frames=2)
                threat_types = types_w | types_f | types_r
                
                narrative = []
                for idx, frame, r in results:
                    ts = round(idx / fps_val, 1)
                    if r["threat_types"]:
                        narrative.append(f"{ts}s: {','.join(r['threat_types'])}")
                
                final_summary = " | ".join(narrative[:5]) + ("..." if len(narrative) > 5 else "")
                if not final_summary: final_summary = "Normal activity observed."

                # Sorting by (weapon, fire, road, frame_index) ensuring that the latest high-confidence frame is picked
                worst = sorted(results, key=lambda x: (x[2].get("weapon_conf",0), x[2].get("fire_conf",0), x[2].get("road_conf",0), x[0]), reverse=True)
                frame_b64 = _encode_frame_b64(worst[0][1], worst[0][2]["boxes"]) if worst else ""

                max_v = max([r[2].get("vehicle_count", 0) for r in results]) if results else 0

                has_threat = len(threat_types) > 0
                has_fire = max_f > 0

                reasoning_firewall = None
                if _scene_analyzer and (has_threat or has_fire) and worst:
                    worst_idx = worst[0][0]
                    center_ts = worst_idx / max(fps_val, 1.0)
                    window = [
                        fr for idx, fr, _ in results
                        if abs((idx / max(fps_val, 1.0)) - center_ts) <= 1.5
                    ][:12]
                    try:
                        reasoning_firewall = await asyncio.to_thread(
                            _scene_analyzer.validate_threat_reasoning,
                            window,
                            format_mmss_ms(center_ts),
                        )
                        if reasoning_firewall.get("is_accident") and not reasoning_firewall.get("is_fire"):
                            threat_types.discard("fire")
                            threat_types.add("road_accident")
                            has_fire = False
                    except Exception as e:
                        logger.warning(f"Reasoning firewall failed: {e}")

                summary_text = final_summary
                if reasoning_firewall and reasoning_firewall.get("reasoning"):
                    summary_text = reasoning_firewall.get("reasoning")

                _update_state_post_process(
                    {"weapon_conf": max_w, "fire_conf": max_f, "road_conf": max_r, "has_threat": has_threat, "has_fire": has_fire, "threat_types": list(threat_types), "vehicle_count": max_v},
                    summary_text, 1.0, frame_b64, file.filename, 0.0
                )

                duration = (total / fps_val) if fps_val > 0 else 0
                
                final_res = {
                    "media_type": "video",
                    "frame_b64": frame_b64,
                    "weapon_conf": max_w,
                    "fire_conf": max_f,
                    "road_conf": max_r,
                    "threat_types": list(threat_types),
                    "summary": final_summary,
                    "camera": camera_name,
                    "has_threat": has_threat,
                    "has_fire": has_fire,
                    "reasoning_firewall": reasoning_firewall or {},
                    "corrected_timestamp": (reasoning_firewall or {}).get("corrected_timestamp", ""),
                    "total_frames": total,
                    "duration_seconds": round(duration, 1),
                    "total_detections": sum(len(r[2].get("boxes", [])) for r in results),
                    "scene_label": final_summary,
                    "scene_description": final_summary,
                }
                yield json.dumps({"result": final_res}) + "\n"
            finally:
                if cap and cap.isOpened(): cap.release()
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        return StreamingResponse(_vid_generator(), media_type="application/x-ndjson")

def _update_state_post_process(res, scene_l, scene_c, b64, fname, anom=0.0):
    _state.update({
        "last_scan_ts": time.time(),
        "weapon_conf": res.get("weapon_conf", 0),
        "fire_conf": res.get("fire_conf", 0),
        "road_conf": res.get("road_conf", 0),
        "vehicle_count": res.get("vehicle_count", 0),
        "scene_label": scene_l,
        "scene_description": scene_l,
        "scene_conf": scene_c,
        "anomaly_score": anom,
        "threat_active": res["has_threat"] or res["has_fire"],
        "threat_type": res.get("threat_types", []),
        "last_frame_b64": b64,
        "last_filename": fname
    })


# Static files mounting (Root)

# Final Static File Configuration
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

@app.get("/")
async def root_redirect():
    return RedirectResponse(url="/index.html")

app.mount("/", StaticFiles(directory="/Users/admin/Documents/vs code/PICA/PROJECT-JERICO-main/frontend", html=True), name="frontend")
