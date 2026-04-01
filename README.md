<h1 align="center">Project Jerico</h1>

<p align="center">Human-in-the-loop CCTV threat analytics using YOLO detection, MIL anomaly scoring, and scene understanding.</p>

## Visual Overview

### Dashboard Preview

![Project Jerico Dashboard Preview](docs/images/dashboard-preview.svg)

### Detection Pipeline

![Project Jerico Pipeline](docs/images/pipeline.svg)

## Overview

Project Jerico is a Streamlit dashboard for analyzing uploaded video or image evidence and surfacing high-risk events.

The system combines three signals:

1. Object detection (person and weapon detection via YOLO)
2. Video anomaly scoring (MIL model over UCF-Crime style 32-segment features)
3. Scene-level semantic context (CLIP text-image matching)

When risk is detected, the dashboard renders a critical alert, synthesizes siren audio, and generates a structured dispatch message with geolocation details.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/dashboard.py
```

## Core Features

1. Upload and process both images and videos
2. Dual detection flow:
	 - Real-time frame object detection
	 - Segment-level anomaly probability from pre-extracted features
3. CLIP-based scene interpretation for contextual threat cues
4. Behavioral smoothing and motion heuristics to reduce alert flicker
5. Geo-tagged dispatch message generation with maps deep link
6. Streamlit controls for confidence thresholds and display options

## Tech Stack

- Python 3.x
- Streamlit
- OpenCV
- PyTorch
- Ultralytics YOLO
- Transformers (CLIP)
- NumPy

## Current Project Structure

```text
PROJECT-JERICO-main/
├── CHANGES.md
├── DATASET_SETUP.md
├── README.md
├── SETUP_GUIDE.md
├── config.py
├── launcher_utility.py
├── requirements.txt
├── run.bat
├── run.sh
├── yolov8n.pt
├── models/
│   ├── best_anomaly_model.pth
│   └── gun_bestweight.pt
└── src/
		├── alert.py
		├── dashboard.py
		├── detect.py
		├── detect_anomaly.py
		├── ingest.py
		├── scene_understanding.py
		├── threat_logic.py
		└── train_ucf_crime.py
```

## Module Guide

- src/dashboard.py
	- Main Streamlit app
	- Handles upload, frame loop, UI, alert state, and siren playback

- src/detect.py
	- Loads YOLO models and returns person/weapon detections

- src/detect_anomaly.py
	- Loads MIL model weights and predicts segment anomaly scores from .txt features

- src/scene_understanding.py
	- CLIP-based natural language scene categorization

- src/alert.py
	- Dispatch message builder and siren waveform generator

- src/train_ucf_crime.py
	- MIL training loop with checkpoint resume support

- src/ingest.py
	- Basic stream reader utility (camera/RTSP scaffold)

## Installation

1. Create and activate a virtual environment

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

If needed, install any missing package explicitly:

```bash
pip install torch opencv-python streamlit ultralytics transformers tqdm
```

## Running the Dashboard

Preferred command:

```bash
streamlit run src/dashboard.py
```

Then open the URL shown by Streamlit (usually http://localhost:8501).

Notes:

- run.bat already uses Streamlit correctly.
- run.sh now launches the app with streamlit run src/dashboard.py.

## Model and Data Requirements

### 1) Object Detection Weights

- YOLO base model file: yolov8n.pt
- Weapon model file: models/gun_bestweight.pt

The detection loader includes a fallback to models/best_model.pt for compatibility.

### 2) Anomaly Detection Weights

- Expected path: models/best_anomaly_model.pth

If missing, run training first.

### 3) Dataset Layout for Training/Inference Lookup

The training/inference code expects a DATASET directory containing list files and feature text files.

Minimum expected files:

```text
DATASET/
├── Anomaly_Train.txt
├── Normal_Train.txt
└── Features/
		└── ... category folders with .txt feature files ...
```

Feature format assumptions:

- 32 temporal segments per video
- 4096-dimensional vector per segment

## Training (MIL Model)

Run:

```bash
python src/train_ucf_crime.py
```

Training behavior:

1. Auto-detects CUDA or CPU
2. Builds dataset index from DATASET/**.txt
3. Uses ranking loss + sparsity + temporal smoothness
4. Writes checkpoint to models/checkpoint.pth
5. Updates best weights at models/best_anomaly_model.pth

## Dashboard Behavior Summary

1. User uploads image or video
2. For each frame, object detection runs
3. If feature file matches uploaded video name, anomaly score is generated from MIL model
4. SceneAnalyzer samples frames periodically and updates contextual threat memory
5. Threat state is derived from weapon detection, violence score, and motion heuristics
6. On threat trigger:
	 - Critical UI alert shown
	 - Dispatch message generated
	 - Siren audio generated and played with st.audio

## Known Limitations

1. The file python-3.12.9-amd64.exe in project root is a Windows installer and is not used by runtime
2. alert.py dispatch is currently message-generation logic only (no real external API call)
3. ingest.py is a scaffold and not fully wired into the dashboard pipeline

## Troubleshooting

### Streamlit does not open

Use:

```bash
streamlit run src/dashboard.py
```

### No detections appear

Check model paths in src/detect.py and verify files exist where expected.

### CLIP scene understanding unavailable

Install transformers and retry:

```bash
pip install transformers
```

### Video upload shows anomaly model load error

Make sure models/best_anomaly_model.pth exists, or train with:

```bash
python src/train_ucf_crime.py
```

### Feature lookup misses uploaded videos

The filename stem of the uploaded video must match a .txt feature file stem in DATASET recursively.

## Suggested Cleanup

1. Remove python-3.12.9-amd64.exe from repository
2. Optionally move yolov8n.pt into models and update references for a cleaner layout
3. Add explicit pinned versions in requirements.txt

## License

No explicit license file is present yet. Add a LICENSE file before public distribution.
