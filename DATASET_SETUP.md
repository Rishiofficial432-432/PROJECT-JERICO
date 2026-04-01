# 📥 UCF-Crime Dataset Setup Guide

## Expected Dataset Structure

For `train_ucf_crime.py` to work, you need to organize the UCF-Crime C3D features as follows:

```
PROJECT-JERICO/
├── DATASET/
│   ├── Anomaly_Train.txt          # List of anomaly video names (one per line)
│   ├── Normal_Train.txt           # List of normal video names (one per line)
│   └── Features/
│       ├── Abuse/
│       │   ├── Abuse001_x264.txt  # 32 lines × 4096 floats (C3D features)
│       │   ├── Abuse002_x264.txt
│       │   └── ...
│       ├── Fighting/
│       ├── Robbery/
│       ├── Shooting/
│       ├── Normal_Videos/         # Or any normal activity folder
│       └── ... (other categories)
```

## Download Options

### ✅ **Option 1: Kaggle (Recommended)**
Easiest and fastest — C3D features ready to use, no approval needed.

1. Go to: https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset
2. Download the dataset (with Kaggle account)
3. Extract: Copy `Features/` folder into `DATASET/`
4. Copy `Anomaly_Train.txt` and `Normal_Train.txt` into `DATASET/`

### **Option 2: Official UCF CRCV Lab**
Direct from the research group — requires access request.

1. Visit: https://www.crcv.ucf.edu/projects/real-world/
2. Request C3D features from the dataset section
3. Extract as above

### **Option 3: HuggingFace (Frame Images)**
If you prefer frame-level images instead of C3D features:

1. https://huggingface.co/datasets/hibana2077/UCF-Crime-Dataset
2. Note: You'll need to adapt the training script to use images instead of `.txt` files

## File Format Details

Each `.txt` feature file contains:
- **32 lines** (one per video segment)
- **4096 floats** per line (C3D feature vector)
- Space or comma-separated values

Example (first 2 lines of Abuse001_x264.txt):
```
0.123 0.456 0.789 ... (4096 values)
0.234 0.567 0.890 ... (4096 values)
...
```

## Training Data Organization

**Anomaly_Train.txt** should list anomalies (one video per line):
```
Abuse/Abuse001_x264.txt
Abuse/Abuse002_x264.txt
Assault/Assault001_x264.txt
Fighting/Fighting001_x264.txt
...
```

**Normal_Train.txt** should list normal activities:
```
Normal_Videos/Normal_001_x264.txt
Normal_Videos/Normal_002_x264.txt
...
```

## Validation

Before training, verify:
1. ✅ `DATASET/` folder exists
2. ✅ `Anomaly_Train.txt` and `Normal_Train.txt` exist
3. ✅ Feature files exist in `Features/` subdirectories
4. ✅ At least one feature file (`.txt`) can be found

## Running Training

Once setup is complete:

```bash
python src/train_ucf_crime.py
```

The script will:
1. Validate dataset structure
2. Load feature files indexed by the `.txt` lists
3. Create checkpoints in `models/checkpoint.pth`
4. Save best model to `models/best_anomaly_model.pth`

## Dashboard Integration

Once trained, the dashboard will automatically:
1. Load the best model checkpoint
2. Look up features for uploaded videos in `DATASET/Features/`
3. Run inference and display anomaly scores

## Troubleshooting

| Error | Solution |
|-------|----------|
| `DATASET/ not found` | Create `DATASET/` folder and download features |
| `Anomaly_Train.txt not found` | Ensure list files are in `DATASET/` root |
| `Found 0 feature files` | Check that `Features/` subfolder naming matches `.txt` list entries |
| `Feature file shape mismatch` | Verify `.txt` files have 32 lines × 4096 columns |

---

**Need help?** Check the training script validation messages — they provide guidance on fixing issues.
