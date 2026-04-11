# Project Jerico Research Notes

Last updated: 2026-04-10

## 1. What Has Been Built So Far

Project Jerico is currently a multi-model CCTV threat analytics system with:
- Streamlit operations dashboard
- FastAPI backend for upload, metrics, events, and streaming
- Multi-model visual detection (person, weapon, fire, road anomaly, vehicle)
- Rule-based threat fusion and prioritization
- Lightweight tracking (ID + speed)
- Optional scene reasoning
- Multi-channel alerting (push, email, WhatsApp)
- Modernized multi-page HTML frontend UI

## 2. Models In Use

### 2.1 Object and Event Detection (YOLO family)
Implemented in src/detect.py.

Loaded models/checkpoints:
- Person detection: yolov8n.pt
- Weapon detection: models/gun_bestweight.pt (fallback: models/best_model.pt)
- Fire detection: models/fire_detection.pt
- Road anomaly detection: models/road_anomaly.pt
- Vehicle detection: models/vehical_detection.pt

Class schema:
- 0: weapon
- 1: person
- 2: fire
- 3: road anomaly
- 4: vehicle

Inference modes:
- Synchronous: run_inference(...)
- Asynchronous parallel: run_inference_async(...) using thread pool workers

### 2.2 Anomaly Model (Temporal)
Implemented in src/detect_anomaly.py.

Model:
- MILAnomalyModel (3-layer MLP)
- Input feature dimension: 4096
- Segment output: 32 anomaly scores per video

Architecture:
- FC(4096 -> 512) + ReLU + Dropout(0.6)
- FC(512 -> 32) + Dropout(0.6)
- FC(32 -> 1) + Sigmoid

Weights:
- models/best_anomaly_model.pth

Feature handling:
- Loads .txt features
- Pads/truncates to (32 x 4096)
- L2 normalization per segment

### 2.3 Scene Understanding Model
Implemented in src/scene_understanding.py.

Model:
- microsoft/Florence-2-large (Transformers)

Behavior:
- Uses detailed caption generation (<MORE_DETAILED_CAPTION>)
- Computes threat score from keyword hits in generated text
- Returns: (description_text, threat_score)

Device logic:
- CUDA if available, else CPU, prefers MPS on Apple Silicon when available

## 3. Algorithms and Decision Logic

### 3.1 Rule-based Threat Refinement
Implemented in src/threat_logic.py.

Core logic:
- Armed-person check:
  - Weapon treated as held if center is inside person box OR box distance < 120 px
- Getaway vehicle check:
  - Vehicle near armed person if box distance < 300 px
- Vehicle collision check:
  - Collision if distance < 100 px and IoU > 0.75
- Fire false-positive penalty:
  - If fire box overlaps vehicle (IoU > 0.50), confidence scaled by 0.35

Threat output categories:
- armed_person / unattended_weapon
- getaway_vehicle
- road anomaly detected
- fire
- road_anomaly or named road event

Priority mapping in threat objects:
- MEDIUM, HIGH, CRITICAL

### 3.2 Hybrid Threat Stack (Production Fusion)
Implemented in src/hybrid_stack.py.

Pipeline behavior:
- Filter detections by minimum object confidence
- Track filtered objects via CentroidTracker
- Evaluate fused threats via evaluate_threat(...)
- Combine with anomaly trigger
- Produce severity and gating signal

Output fields:
- filtered_detections
- tracking (track_id, speed)
- fused_threats
- anomaly_triggered
- severity
- should_run_heavy_reasoning

### 3.3 Lightweight Tracking
Implemented in src/tracker.py.

Tracker type:
- Centroid-based nearest-neighbor tracker

Key parameters:
- max_missed_frames = 12
- match_distance = 80 px

Outputs:
- Stable track_id per detection
- Speed estimate from center displacement

## 4. Thresholds and Operational Defaults

Defined in src/threat_logic.py and dashboard.py:

Threat logic thresholds:
- Weapon threshold: 0.55
- Fire threshold: 0.82
- Road anomaly threshold: 0.40
- Vehicle threshold: 0.35

Dashboard defaults:
- Confidence threshold: 0.60 (smart mode)
- Anomaly threshold: 0.70 (smart mode)
- Scene threshold: 0.25
- Fire threshold: 0.35 (display/inference gating)

## 5. Runtime and Concurrency

Thread pool:
- src/worker_pool.py defines shared ThreadPoolExecutor(max_workers=16)
- Used by async multi-model inference calls

FastAPI queue pipeline:
- Producer-consumer frame buffering
- Multi-worker async inference
- NDJSON streaming for progressive frontend updates

Persistence-style check in API video path:
- Uses neighborhood consistency and instant-threshold checks to reduce noisy one-frame triggers

## 6. Backend/API Features

Implemented in frontend/api.py.

Endpoints:
- GET /api/status
- GET /api/metrics
- GET /api/events
- GET /api/events/stream (SSE)
- POST /api/upload (image/video processing)

State tracked in memory:
- incident counters, latest confidences, active threat types, camera, last frame, scene summary

Media support:
- Images: .jpg .jpeg .png .bmp .webp
- Videos: .mp4 .avi .mov .mkv .webm

## 7. Dashboard and Alerting

Implemented in dashboard.py and src/alert.py.

Dashboard capabilities:
- Image and video upload processing
- Bounding box visualization with class-specific labels/colors
- Hybrid fusion in video path
- On-alert gated scene reasoning in video pipeline
- Threat analytics panels and status indicators

Alerting capabilities:
- Siren waveform generation
- Dispatch message generation with geolocation context
- ntfy push alerts
- SMTP email alerts
- WhatsApp alerts (CallMeBot)

## 8. Frontend UI Work Completed

Frontend pages updated and aligned:
- frontend/index.html
- frontend/upload.html
- frontend/alert.html
- frontend/critical_alert.html

UI upgrades completed:
- Unified modern visual language
- Improved typography and hierarchy
- Responsive behavior for tablet/mobile
- Navigation cleanup (no placeholder # links)
- Critical alert page redesigned in same design system

## 9. Tech Stack (Current)

Core packages from requirements.txt:
- torch>=2.4
- ultralytics
- transformers
- opencv-python
- streamlit
- numpy<2
- scipy
- pillow
- timm
- einops
- onnx
- tqdm

Backend framework package in code:
- fastapi (used by frontend/api.py)

## 10. Current Architecture Summary

End-to-end flow:
1. Ingest image/video
2. Run parallel YOLO inference
3. Refine detections with spatial/semantic rules
4. Track objects and estimate motion
5. Score temporal anomaly (if features available)
6. Optionally run scene reasoning for context
7. Fuse all signals into severity + threat list
8. Render annotated outputs and trigger alerts/dispatch

## 11. Practical Notes

- The system uses a hybrid AI + heuristic strategy, not model-only decisions.
- Scene reasoning currently uses Florence-2 captions with keyword scoring.
- Video path is the most feature-rich path (fusion + tracking + gated scene reasoning).
- Current design emphasizes operator usability and real-time decision support.
