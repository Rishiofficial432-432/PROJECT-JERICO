"""
Jerico API Server — FastAPI backend for the frontend dashboard.
Serves live model metrics, handles footage uploads, and streams inference results.

Run with:
    uvicorn frontend.api:app --host 0.0.0.0 --port 8000 --reload
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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
#  Model loading (mirrors dashboard.py)
# ─────────────────────────────────────────────

_models_loaded = False
_model_errors: dict[str, str] = {}

try:
    from detect import run_inference, CLASS_WEAPON, CLASS_PERSON, CLASS_FIRE
    from threat_logic import WEAPON_CONF_THRESHOLD, FIRE_CONF_THRESHOLD
    _models_loaded = True
except Exception as e:
    logger.error(f"detect.py import failed: {e}")
    _model_errors["yolo"] = str(e)
    CLASS_WEAPON, CLASS_PERSON, CLASS_FIRE = 0, 1, 2
    WEAPON_CONF_THRESHOLD = 0.85
    FIRE_CONF_THRESHOLD   = 0.55

try:
    from detect_anomaly import load_anomaly_model, lookup_features, predict_anomaly
    _anomaly_model, _anomaly_device = load_anomaly_model()
    _anomaly_ready = True
except Exception as e:
    logger.error(f"Anomaly model failed: {e}")
    _anomaly_model = _anomaly_device = None
    _anomaly_ready = False
    _model_errors["anomaly"] = str(e)

try:
    from scene_understanding import SceneAnalyzer
    _scene_analyzer = SceneAnalyzer()
    _scene_ready = True
except Exception as e:
    logger.warning(f"Scene analyzer failed: {e}")
    _scene_analyzer = None
    _scene_ready = False
    _model_errors["scene"] = str(e)

# ─────────────────────────────────────────────
#  In-memory state store
# ─────────────────────────────────────────────

_state = {
    "incidents_today": 0,
    "last_scan_ts": None,         # datetime of last frame processed
    "anomaly_score": 0.0,         # latest anomaly score (0-100)
    "weapon_conf": 0.0,
    "fire_conf": 0.0,
    "violence_score": 0.0,
    "scene_label": "No footage analysed",
    "scene_conf": 0.0,
    "threat_active": False,
    "threat_type": [],
    "active_camera": "—",
}

# Circular event log (max 100 entries)
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

app = FastAPI(title="Jerico API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  /api/status  — model health
# ─────────────────────────────────────────────

@app.get("/api/status")
def get_status():
    weight_path = Path("models/best_anomaly_model.pth")
    last_update = "Never trained"
    if weight_path.exists():
        mtime = weight_path.stat().st_mtime
        last_update = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")

    return {
        "yolo":    {"ready": _models_loaded,  "error": _model_errors.get("yolo")},
        "anomaly": {"ready": _anomaly_ready,  "error": _model_errors.get("anomaly")},
        "scene":   {"ready": _scene_ready,    "error": _model_errors.get("scene")},
        "fire":    {"ready": _models_loaded,  "error": None},   # shares yolo pipeline
        "weights_updated": last_update,
        "models_active": sum([_models_loaded, _anomaly_ready, _scene_ready, _models_loaded]),
    }

# ─────────────────────────────────────────────
#  /api/metrics  — live dashboard numbers
# ─────────────────────────────────────────────

@app.get("/api/metrics")
def get_metrics():
    last_scan = "Never"
    if _state["last_scan_ts"]:
        elapsed = int(time.time() - _state["last_scan_ts"])
        if elapsed < 60:
            last_scan = f"{elapsed}s ago"
        else:
            last_scan = f"{elapsed // 60}m ago"

    return {
        "incidents_today": _state["incidents_today"],
        "last_scan": last_scan,
        "anomaly_score": round(_state["anomaly_score"], 1),
        "weapon_conf": round(_state["weapon_conf"], 2),
        "fire_conf": round(_state["fire_conf"], 2),
        "violence_score": round(_state["violence_score"], 1),
        "scene_label": _state["scene_label"],
        "scene_conf": round(_state["scene_conf"], 2),
        "threat_active": _state["threat_active"],
        "threat_type": _state["threat_type"],
        "active_camera": _state["active_camera"],
    }

# ─────────────────────────────────────────────
#  /api/events  — event log
# ─────────────────────────────────────────────

@app.get("/api/events")
def get_events():
    with _events_lock:
        return list(_events)

# ─────────────────────────────────────────────
#  /api/events/stream  — Server-Sent Events
# ─────────────────────────────────────────────

async def _sse_generator():
    """Async SSE — yields metrics+events every second without blocking the event loop."""
    while True:
        metrics = get_metrics()
        with _events_lock:
            events = list(_events)[:10]
        payload = json.dumps({"metrics": metrics, "events": events})
        yield f"data: {payload}\n\n"
        await asyncio.sleep(1)          # non-blocking — event loop stays free

@app.get("/api/events/stream")
async def sse_stream():
    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# ─────────────────────────────────────────────
#  /api/upload  — process image or video
# ─────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

@app.post("/api/upload")
async def upload_footage(
    file: UploadFile = File(...),
    camera_name: str = "CCTV-01",
    person_conf_threshold: float = 0.50,
    weapon_conf_threshold: float = 0.65,
    fire_conf_threshold:   float = 0.60,
    anomaly_alert_pct:     float = 70.0,
):
    if not _models_loaded:
        raise HTTPException(503, "YOLO models not loaded — check server logs.")

    ext = Path(file.filename).suffix.lower()
    if ext not in IMAGE_EXTS | VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    _state["active_camera"] = camera_name

    thresholds = {
        "person": person_conf_threshold,
        "weapon": weapon_conf_threshold,
        "fire":   fire_conf_threshold,
        "anomaly_alert_pct": anomaly_alert_pct,
    }

    try:
        if ext in IMAGE_EXTS:
            # Run CPU-bound inference in a thread pool — keeps event loop free
            result = await asyncio.to_thread(_process_image, tmp_path, camera_name, thresholds)
        else:
            result = await asyncio.to_thread(_process_video, tmp_path, camera_name, thresholds, file.filename)
        return result
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

# ─────────────────────────────────────────────
#  Image inference
# ─────────────────────────────────────────────

def _process_image(path: str, camera: str, th: dict) -> dict:
    frame = cv2.imread(path)
    if frame is None:
        raise HTTPException(400, "Could not decode image.")

    detections = run_inference(
        frame,
        person_conf=th["person"],
        weapon_conf=th["weapon"],
        fire_conf=th["fire"],
    )
    _state["last_scan_ts"] = time.time()

    results = _analyse_detections(detections, th, frame)

    # Encode annotated frame as base64 JPEG
    frame_b64 = _encode_frame_b64(frame, results["boxes"])

    # Scene
    scene_label, scene_conf = "—", 0.0
    if _scene_analyzer:
        try:
            scene_label, scene_conf = _scene_analyzer.analyze_frame(frame)
        except Exception as e:
            logger.warning(f"Scene analyze failed: {e}")

    _state.update({
        "weapon_conf": results["weapon_conf"],
        "fire_conf": results["fire_conf"],
        "scene_label": scene_label,
        "scene_conf": scene_conf,
        "anomaly_score": 0.0,
        "violence_score": 0.0,
        "threat_active": results["has_threat"] or results["has_fire"],
        "threat_type": results["threat_types"],
    })
    if results["has_threat"] or results["has_fire"]:
        _state["incidents_today"] += 1
        event_label = " + ".join(results["threat_types"]) if results["threat_types"] else "Threat"
        _log_event(camera, event_label, max(results["weapon_conf"], results["fire_conf"]), "active")
    else:
        _log_event(camera, "Image scan — no threats", None, "passed")

    return {
        "media_type": "image",
        "camera": camera,
        "frame_b64": frame_b64,
        "detections": results["boxes"],
        "has_threat": results["has_threat"],
        "has_fire": results["has_fire"],
        "weapon_conf": results["weapon_conf"],
        "fire_conf": results["fire_conf"],
        "threat_types": results["threat_types"],
        "scene_label": scene_label,
        "scene_conf": round(scene_conf, 2),
        "total_detections": len(results["boxes"]),
    }

# ─────────────────────────────────────────────
#  Video inference  (sample frames)
# ─────────────────────────────────────────────

def _process_video(path: str, camera: str, th: dict, filename: str) -> dict:
    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    duration = total_frames / fps
    anomaly_alert = th.get("anomaly_alert_pct", 70.0)

    # Try anomaly model first
    anomaly_scores = None
    if _anomaly_ready and _anomaly_model is not None:
        feat_path = lookup_features(filename)
        if feat_path:
            try:
                anomaly_scores = predict_anomaly(feat_path, _anomaly_model, _anomaly_device)
                logger.info(f"Anomaly scores from dataset features: {feat_path}")
            except Exception as e:
                logger.warning(f"Anomaly prediction failed: {e}")

    # Sample every N frames (max 60 samples for performance)
    sample_interval = max(1, total_frames // 60)
    frame_idx = 0
    all_boxes = []
    max_weapon_conf = 0.0
    max_fire_conf = 0.0
    max_anomaly = 0.0
    max_violence = 0.0
    threat_types = set()
    worst_frame = None
    worst_frame_boxes = []
    worst_frame_idx = 0

    scene_label, scene_conf = "No scene detected", 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % sample_interval == 0:
            _state["last_scan_ts"] = time.time()
            detections = run_inference(
                frame,
                person_conf=th["person"],
                weapon_conf=th["weapon"],
                fire_conf=th["fire"],
            )
            r = _analyse_detections(detections, th, frame)

            # Anomaly score for this segment vs user alert threshold
            if anomaly_scores is not None and total_frames > 0:
                seg_idx = min(int((frame_idx / total_frames) * 32), 31)
                seg_score = float(anomaly_scores[seg_idx]) * 100
                if seg_score > max_anomaly:
                    max_anomaly = seg_score
                if seg_score > max_violence:
                    max_violence = seg_score

            # Track worst weapon frame
            if r["weapon_conf"] > max_weapon_conf:
                max_weapon_conf = r["weapon_conf"]
                worst_frame = frame.copy()
                worst_frame_boxes = r["boxes"]
                worst_frame_idx = frame_idx
            # Track worst fire frame (only if no weapon frame yet)
            if r["fire_conf"] > max_fire_conf:
                max_fire_conf = r["fire_conf"]
                if worst_frame is None:
                    worst_frame = frame.copy()
                    worst_frame_boxes = r["boxes"]
                    worst_frame_idx = frame_idx
            # If still nothing, keep the first sampled frame
            if worst_frame is None and frame_idx == 0:
                worst_frame = frame.copy()
                worst_frame_boxes = r["boxes"]
                worst_frame_idx = frame_idx

            threat_types.update(r["threat_types"])
            all_boxes.extend(r["boxes"])

            # Scene every 30 samples
            if _scene_analyzer and frame_idx % (sample_interval * 30) == 0:
                try:
                    scene_label, scene_conf = _scene_analyzer.analyze_frame(frame)
                except Exception:
                    pass

        frame_idx += 1

    cap.release()

    has_threat = max_weapon_conf > 0
    has_fire = max_fire_conf > 0
    threat_list = list(threat_types)

    # Encode the worst frame with boxes drawn on it
    frame_b64 = ""
    if worst_frame is not None:
        frame_b64 = _encode_frame_b64(worst_frame, worst_frame_boxes)

    _state.update({
        "weapon_conf": max_weapon_conf,
        "fire_conf": max_fire_conf,
        "anomaly_score": round(max_anomaly, 1),
        "violence_score": round(max_violence, 1),
        "scene_label": scene_label,
        "scene_conf": scene_conf,
        "threat_active": has_threat or has_fire,
        "threat_type": threat_list,
    })

    if has_threat or has_fire:
        _state["incidents_today"] += 1
        _log_event(camera, " + ".join(threat_list) if threat_list else "Threat", max(max_weapon_conf, max_fire_conf), "active")
    else:
        _log_event(camera, "Video scan complete — no threats", None, "passed")

    return {
        "media_type": "video",
        "camera": camera,
        "frame_b64": frame_b64,           # ← actual annotated worst frame
        "total_frames": total_frames,
        "duration_seconds": round(duration, 1),
        "worst_frame": worst_frame_idx,
        "has_threat": has_threat,
        "has_fire": has_fire,
        "weapon_conf": round(max_weapon_conf, 2),
        "fire_conf": round(max_fire_conf, 2),
        "anomaly_score": round(max_anomaly, 1),
        "violence_score": round(max_violence, 1),
        "scene_label": scene_label,
        "scene_conf": round(scene_conf, 2),
        "threat_types": threat_list,
        "total_detections": len(all_boxes),
    }

# ─────────────────────────────────────────────
#  Shared detection analysis helper
# ─────────────────────────────────────────────

def _analyse_detections(detections, th: dict, frame) -> dict:
    """Filter detections using per-class thresholds from `th` dict."""
    h, w = frame.shape[:2]
    has_threat = False
    has_fire = False
    weapon_conf = 0.0
    fire_conf = 0.0
    threat_types = []
    boxes = []

    person_thresh = th.get("person", 0.50)
    weapon_thresh = th.get("weapon", 0.65)
    fire_thresh   = th.get("fire",   0.60)

    logger.info(
        f"Thresholds — person: {person_thresh:.2f}, "
        f"weapon: {weapon_thresh:.2f}, fire: {fire_thresh:.2f}"
    )

    for det in detections:
        cls_id, conf, x1, y1, x2, y2 = det
        if int(cls_id) == CLASS_PERSON and conf < person_thresh:
            continue
        if int(cls_id) == CLASS_WEAPON and conf < weapon_thresh:
            continue
        if int(cls_id) == CLASS_FIRE   and conf < fire_thresh:
            continue

        label = {CLASS_WEAPON: "weapon", CLASS_PERSON: "person", CLASS_FIRE: "fire"}.get(int(cls_id), "unknown")

        boxes.append({
            "class": label,
            "conf": round(float(conf), 2),
            # Normalised [0-1] coords for the frontend overlay
            "bbox": [
                round(x1 / w, 4), round(y1 / h, 4),
                round(x2 / w, 4), round(y2 / h, 4),
            ],
            # Absolute pixel coords so we can draw on the frame server-side
            "bbox_px": [int(x1), int(y1), int(x2), int(y2)],
        })

        if int(cls_id) == CLASS_WEAPON:
            has_threat = True
            weapon_conf = max(weapon_conf, conf)
            if "Weapon" not in threat_types:
                threat_types.append("Weapon")
        elif int(cls_id) == CLASS_FIRE:
            has_fire = True
            fire_conf = max(fire_conf, conf)
            if "Fire" not in threat_types:
                threat_types.append("Fire")

    return {
        "has_threat": has_threat,
        "has_fire": has_fire,
        "weapon_conf": round(float(weapon_conf), 2),
        "fire_conf": round(float(fire_conf), 2),
        "threat_types": threat_types,
        "boxes": boxes,
    }


# ─────────────────────────────────────────────
#  Draw boxes on frame → base64 JPEG
# ─────────────────────────────────────────────

# Colours per class (BGR)
_COLORS = {
    "weapon": (0,   0,   220),   # red
    "fire":   (0,   120, 240),   # orange
    "person": (220, 80,  0),     # blue
}

def _encode_frame_b64(frame: np.ndarray, boxes: list, quality: int = 80) -> str:
    """
    Draw bounding boxes on a copy of `frame`, encode as JPEG, return base64 string.
    `boxes` is the list from _analyse_detections (with bbox_px keys).
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Resize if very large to keep response lean (max 1280px wide)
    if w > 1280:
        scale = 1280 / w
        out = cv2.resize(out, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = out.shape[:2]

    for det in boxes:
        cls   = det["class"]
        conf  = det["conf"]
        color = _COLORS.get(cls, (180, 180, 180))

        # Use pixel coords if available, else fall back to normalised
        if "bbox_px" in det:
            x1, y1, x2, y2 = det["bbox_px"]
            # Scale if we resized
            orig_h = frame.shape[0]
            orig_w = frame.shape[1]
            if orig_w > 1280:
                sf = 1280 / orig_w
                x1 = int(x1 * sf); y1 = int(y1 * sf)
                x2 = int(x2 * sf); y2 = int(y2 * sf)
        else:
            n = det["bbox"]
            x1, y1, x2, y2 = int(n[0]*w), int(n[1]*h), int(n[2]*w), int(n[3]*h)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls} {conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        # Label background
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("utf-8")

# ─────────────────────────────────────────────
#  Startup check log
# ─────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    logger.info("=== Jerico API Server started ===")
    logger.info(f"  YOLO:    {'✅' if _models_loaded else '❌'}")
    logger.info(f"  Anomaly: {'✅' if _anomaly_ready else '❌'}")
    logger.info(f"  Scene:   {'✅' if _scene_ready else '❌'}")
    _log_event("—", "API server started", None, "info")
