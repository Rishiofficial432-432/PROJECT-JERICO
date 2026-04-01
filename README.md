# Project Jerico: Intelligent CCTV Security System 🛡️

## 📖 The Idea (What & Why)
Project Jerico is a **Human-in-the-Loop CCTV Security & Anomaly Detection System**. It continuously ingests realtime CCTV feeds, analyzes the frames for anomalies (weapons, loitering, burglary, fighting, etc.), and surfaces potential threats to a localized dashboard. 

The core philosophy of this project is **Validate First, Alert Later.** Instead of auto-dialing emergency services based on AI output, the system routes high-confidence anomalies to a Streamlit-based web dashboard. A human security operative verifies the clip and clicks "Confirm" to escalate the alert, or "Deny" to log the event as a false positive and use it for future model retraining.

## 🛠️ How We Built It
The system revolves around machine learning pipelines designed for both frame-level and video-level context:
1. **Real-time Object Detection:** Built to scan individual frames for specific objects like weapons or abandoned items using bounding-box inference. 
2. **Video Anomaly Detection (UCF-Crime MIL Model):** A deep learning pipeline that analyzes temporal video features using Multiple Instance Learning (MIL) to identify complex anomalous behavior (like theft or fights) across sequences of frames.

### Libraries & Tech Stack
- **PyTorch & NumPy**: The backbone for defining the 3-layer MLP network, custom dataloaders, computing the MIL ranking loss, and managing tensor operations on the GPU.
- **OpenCV**: Used in `ingest.py` to capture, decode, and resize RTSP/HTTP video streams directly from CCTV hardware.
- **Streamlit**: Powers `dashboard.py` to seamlessly create a lightweight, responsive web UI for the human verification loop.
- **ONNX**: Intended for converting internal weights into a universal format to speed up inference on low-end hardware.

## 📂 Repository Structure
```text
PROJECT JERICO/
│
├── DATASET/                  # UCF-Crime dataset (Features, Anomaly/Normal text splits)
├── data/                     # Your localized CCTV footage splits (train/val/test)
├── models/                   # Saved model weights (best_anomaly_model.pth, checkpoint.pth)
├── logs/                     # Databases and logs tracking false positive denials
│
├── src/
│   ├── train_ucf_crime.py    # PyTorch MIL ranking script to train the anomaly model on your GPU
│   ├── ingest.py             # Captures CCTV stream using OpenCV 
│   ├── detect.py             # Runs real-time inference on ingested frames
│   ├── detect_anomaly.py     # Aggregates temporal features for video-level anomaly checking
│   ├── scene_understanding.py# High-level environmental context parsing
│   ├── threat_logic.py       # Threshold engine (e.g. confidence > 85% -> trigger alert)
│   ├── dashboard.py          # The Streamlit web UI for human verification
│   └── alert.py              # Handles webhook/email escalation upon human confirmation
│
├── requirements.txt          # Python dependencies (NumPy, PyTorch, Streamlit, OpenCV)
├── run.bat / run.sh          # One-click startup scripts for the dashboard
└── README.md                 # Project documentation
```

## 🧠 How Training is Going
The core anomaly detection engine is currently training on the vast **UCF-Crime** dataset using **Multiple Instance Learning (MIL)**.
- **Feature Extraction**: Instead of training on raw video pixels from scratch, the network learns from pre-extracted 4096-dimensional C3D spatial-temporal features. Each video is divided into 32 equal segments.
- **The Objective Component**: The network uses a custom Ranking Loss formula. It ensures that the highest anomaly score within an "anomalous" bag (video) is strictly greater than the highest score in a "normal" bag. It is also regularized by sparsity (anomalies are rare) and temporal smoothness (anomalies don't jump erratically between frames).
- **Compute Setup**: Training is executed directly on the localized NVIDIA GPU (`cuda:0`).
- **Resiliency**: The training script (`src/train_ucf_crime.py`) features robust auto-resume mechanics. It constantly updates `models/checkpoint.pth` (saving the epoch, optimizer state, and current loss parameters), ensuring that if the terminal closes, training simply picks up exactly where it left off on the next run.
