# 🚀 PROJECT JERICO: Complete Setup & Usage Guide

This guide walks you through installation, dataset setup, training, and running the anomaly detection dashboard.

---

## 📋 Prerequisites

- **Python 3.9+**
- **NVIDIA GPU** (CUDA 11.8+) — optional but highly recommended for training
- **50+ GB disk space** for the UCF-Crime dataset
- **pip** package manager

---

## 🛠️ Step 1: Installation

### 1.1 Clone/Navigate to Project
```bash
cd /workspaces/PROJECT-JERICO
```

### 1.2 Install Dependencies
```bash
pip install -r requirements.txt
```

This installs:
- **torch**: Deep learning framework with GPU support
- **transformers**: HuggingFace models (CLIP for scene understanding)
- **ultralytics**: YOLOv8 object detection
- **opencv-python**: Video processing
- **streamlit**: Web dashboard
- **numpy, tqdm, pillow**: Utilities

### 1.3 Verify Installation
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
```

---

## 📥 Step 2: Download & Setup Dataset

The codebase is **pre-optimized for UCF-Crime** dataset with C3D features.

### Option A: Quick Download from Kaggle (Recommended)

1. **Create Kaggle account** (if you don't have one): https://www.kaggle.com/
2. **Download dataset**: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
3. **Extract files** into `DATASET/` folder:
   ```
   DATASET/
   ├── Anomaly_Train.txt        (copy from download)
   ├── Normal_Train.txt         (copy from download)
   └── Features/                (entire folder from download)
       ├── Abuse/
       ├── Arrest/
       ├── Arson/
       ├── Assault/
       ├── Fighting/
       ├── Robbery/
       ├── Shooting/
       ├── Stealing/
       ├── Vandalism/
       └── Normal_Videos/
   ```

### Option B: Official UCF CRCV Lab

1. Visit: https://www.crcv.ucf.edu/projects/real-world/
2. Request C3D features from the dataset section
3. Extract same structure as above

### Step 2.3: Verify Dataset Structure
```bash
python src/train_ucf_crime.py
```

If dataset is correct, you'll see:
```
=========================================
      USING DEVICE: CUDA             
=========================================
Indexing feature files for Anomaly dataset...
Found 950 feature files for this split.
Indexing feature files for Normal dataset...
Found 950 feature files for this split.
```

If not, the script will show exactly what's missing.

---

## 🎓 Step 3: Train the Anomaly Model

### 3.1 Start Training
```bash
python src/train_ucf_crime.py
```

**What happens:**
- ✅ Validates dataset exists
- ✅ Loads C3D features (4096-D vectors)
- ✅ Creates MIL model (3-layer MLP)
- ✅ Checks for resume point (`models/checkpoint.pth`)
- ✅ Begins training loop with progress bars
- ✅ Auto-saves checkpoint every epoch
- ✅ Saves best model when loss improves

**Sample output:**
```
Epoch 1/1000: 100%|████████████| 100/100 [00:45<00:00, 0.45s/batch]
===> Epoch 1 Average Loss: 0.8234
-> Saved new best model to models/best_anomaly_model.pth with loss 0.8234

Epoch 2/1000: 100%|████████████| 100/100 [00:44<00:00, 0.44s/batch]
...
```

### 3.2 Monitor Training

In a **separate terminal**, you can watch progress:
```bash
# Check checkpoint info
ls -lh models/checkpoint.pth models/best_anomaly_model.pth

# (Optional) Monitor GPU usage
nvidia-smi -l 1  # Refresh every 1 second
```

### 3.3 Resume From Checkpoint

If training is interrupted, simply run again:
```bash
python src/train_ucf_crime.py
```

It will **automatically resume** from the last saved epoch.

### 3.4 Training Configuration

Edit `config.py` to customize:
```python
TRAIN_BATCH_SIZE = 32          # Larger = faster (needs more RAM)
TRAIN_EPOCHS = 1000            # Total epochs to train
LEARNING_RATE = 0.001          # Adagrad learning rate
WEIGHT_DECAY = 0.001           # L2 regularization
DROPOUT_RATE = 0.6             # Dropout in MLP
```

---

## 🎯 Step 4: Launch the Dashboard

Once the model is training (don't need to wait for it to finish):

### 4.1 Start the Dashboard
```bash
python launcher_utility.py
```

Or directly with Streamlit:
```bash
streamlit run dashboard.py
```

**Output:**
```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

### 4.2 Open Dashboard
Visit: `http://localhost:8501` in your browser

**Dashboard Features:**
- 📹 Upload CCTV footage (MP4/AVI/MOV)
- 🎯 Real-time anomaly scores for each segment
- 🔔 Automatic threat alerts when confidence > threshold
- 🧠 CLIP-based scene understanding
- 📍 Geo-location mapping
- 🔄 Hot-reload model weights during training

### 4.3 Using the Dashboard

1. Click **⚙️ Detection Settings** sidebar to adjust thresholds:
   - `Minimum Confidence Score`: YOLO detection threshold (0.0-1.0)
   - `Anomaly Violence Trigger`: Anomaly alert level (0.0-1.0)  
   - `Scene Confidence (CLIP)`: CLIP understanding threshold (0.0-1.0)

2. Enter camera location:
   - Camera Alias: e.g., "Front Entrance"
   - Latitude / Longitude: GPS coordinates
   - **Alerts will include exact location**

3. Upload video:
   - Click "Upload Video (MP4/AVI)" button
   - Select file from computer

4. Analysis runs automatically:
   - **YOLO** detects persons/weapons
   - **Anomaly Model** scores each segment
   - **CLIP** describes scene
   - Displays live results with bounding boxes

---

## 🔄 Parallel Workflow

You can train **and** use the dashboard **simultaneously**:

### Terminal 1: Training (background)
```bash
python src/train_ucf_crime.py > training.log 2>&1 &
```

### Terminal 2: Dashboard (foreground)
```bash
streamlit run dashboard.py
```

As training improves the model, the dashboard will hot-reload updated weights when you click "🔄 Reload Model Weights" in the sidebar.

---

## 📊 Understanding the Architecture

### Model Structure
```
C3D Features (32 segments × 4096 dims)
    ↓
Linear (4096 → 512)
    ↓
ReLU + Dropout (0.6)
    ↓
Linear (512 → 32)
    ↓
Dropout (0.6)
    ↓
Linear (32 → 1)
    ↓
Sigmoid
    ↓
Anomaly Score per Segment (32 scores)
```

### Loss Function
```
Total Loss = Ranking Loss + Sparsity Penalty + Smoothness Penalty

Ranking Loss:   max(anomaly_scores) > max(normal_scores) + margin
Sparsity:       Penalizes high anomaly scores (anomalies are rare)
Smoothness:     Encourages similar scores between adjacent segments
```

### Training Optimizer
- **Algorithm**: Adagrad
- **Learning Rate**: 0.001
- **Weight Decay**: 0.001 (L2 regularization)

---

## 🛠️ Configuration Reference

All settings are in `config.py`:

```python
# Dataset paths
DATASET_DIR = "DATASET"
ANOMALY_LIST_FILE = "Anomaly_Train.txt"
NORMAL_LIST_FILE = "Normal_Train.txt"

# Model paths
MODEL_CHECKPOINT_PATH = "models/checkpoint.pth"
BEST_MODEL_PATH = "models/best_anomaly_model.pth"

# Training
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 1000
LEARNING_RATE = 0.001

# Detection thresholds
CONFIDENCE_THRESHOLD = 0.60        # YOLO detection
ANOMALY_THRESHOLD = 0.75           # Anomaly alert
SCENE_CONFIDENCE_THRESHOLD = 0.85  # CLIP scene

# Model architecture
FEATURE_DIM = 4096        # C3D features
SEGMENT_COUNT = 32        # Segments per video
HIDDEN_DIM_1 = 512
HIDDEN_DIM_2 = 32

# Loss weights
SPARSITY_PENALTY = 0.00008
SMOOTHNESS_PENALTY = 0.00008
```

---

## 🐛 Troubleshooting

### Installation Issues

**Error:** `ModuleNotFoundError: No module named 'transformers'`
```bash
pip install transformers
```

**Error:** `CUDA out of memory during training`
- Reduce `TRAIN_BATCH_SIZE` in `config.py` (try 16 or 8)
- Or use CPU: Change `device = torch.device("cpu")` in training script

---

### Dataset Issues

**Error:** `DATASET/ not found`
```bash
# Create dataset directory
mkdir DATASET
# Then follow Step 2 to download features
```

**Error:** `Found 0 feature files`
- Check feature files exist: `ls DATASET/Features/Abuse/*.txt`
- Verify `Anomaly_Train.txt` and `Normal_Train.txt` are in `DATASET/`
- Check naming: Training script looks for filenames matching list entries exactly

**Error:** `Could not find feature text files matching the Anomaly_Train.txt`
- The `.txt` list files have entries like: `Abuse/Abuse001_x264.txt`
- But actual feature files should be at: `DATASET/Features/Abuse/Abuse001_x264.txt`
- Ensure folder structure matches exactly

---

### Dashboard Issues

**Error:** `Cannot process video: Anomaly model failed to load`
- Wait for training to complete first: `models/best_anomaly_model.pth` must exist
- Or check training log for errors

**Error:** `ModuleNotFoundError: No module named 'ultralytics'`
```bash
pip install ultralytics
```

**Error:** `Dashboard shows blank page`
- Check terminal for error messages
- Try: `streamlit run dashboard.py --logger.level=debug`

---

### Training Issues

**Error:** `GPU not being used (very slow training)`
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled torch:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Training is slow (< 1 epoch/min)**
- CPU training is normal but slow
- GPU training should be ~10-20 epochs/min
- Reduce `TRAIN_BATCH_SIZE` to check if memory is limiting

---

## 📈 Performance Tips

### Faster Training
1. Use GPU: `torch.cuda.is_available()` should be `True`
2. Increase batch size: `TRAIN_BATCH_SIZE = 64` (if you have 16+ GB VRAM)
3. Use proper CUDA drivers: `nvidia-smi` shows GPU stats

### Faster Inference
1. YOLOv8 nano is already used (`yolov8n.pt`)
2. CLIP is the bottleneck (~10% of dashboard runtime)
3. To skip CLIP: Comment out CLIP checks in `dashboard.py`

### Better Detection
1. Train longer: More epochs = better performance
2. Use balanced dataset: Equal anomalies and normal videos
3. Fine-tune thresholds based on your camera scenes

---

## 📚 What's Training?

The **MILAnomalyModel** learns to:
- Identify **which segments** contain anomalies (not just classify whole video)
- Enforce **ranking**: Anomalous bags > Normal bags
- Maintain **temporal smoothness**: Adjacent segments have similar scores
- Remain **sparse**: Only mark actual anomalies, not entire video

This is the **Multiple Instance Learning** paradigm — perfect for video anomaly detection where you have video-level labels but want segment-level scores.

---

## ✅ Verification Checklist

Before assuming setup is complete:

- [ ] `pip install -r requirements.txt` succeeded
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` shows GPU availability
- [ ] `DATASET/` folder exists with `Anomaly_Train.txt`, `Normal_Train.txt`
- [ ] `DATASET/Features/` has subfolders like `Abuse/`, `Fighting/`, `Normal_Videos/`
- [ ] Feature files exist: `ls DATASET/Features/Abuse/*.txt | head`
- [ ] `python src/train_ucf_crime.py` shows "Found X feature files"
- [ ] `models/best_anomaly_model.pth` created after ~1 epoch
- [ ] `streamlit run dashboard.py` opens at `http://localhost:8501`
- [ ] Dashboard sidebar shows model status and last update time

---

## 🎥 End-to-End Example

```bash
# 1. Setup (one-time)
cd /workspaces/PROJECT-JERICO
pip install -r requirements.txt

# 2. Download dataset (manually)
# Go to https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
# Extract into DATASET/ folder

# 3. Validate dataset
python src/train_ucf_crime.py
# Should say "Found 950 anomaly files, Found 950 normal files"
# (Ctrl+C to stop)

# 4. Train (in terminal 1, runs forever)
python src/train_ucf_crime.py

# 5. Dashboard (in terminal 2, different window)
streamlit run dashboard.py

# 6. Open browser and upload videos
# http://localhost:8501
```

---

## 🚀 Next Steps

1. **Monitor training** — Check loss decreasing every epoch
2. **Test with sample videos** — Upload to dashboard to see predictions
3. **Tune thresholds** — Adjust anomaly/confidence triggers for your environment
4. **Deploy** — Export model for edge deployment on security hardware
5. **Feedback loop** — Use false positives to retrain and improve

---

**Questions? Check specific error messages in the console — they guide you to solutions!** 🎯
