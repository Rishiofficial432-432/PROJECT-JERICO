# Project Configuration File
# Centralized settings for the anomaly detection pipeline

# ========== Dataset Configuration ==========
DATASET_DIR = "DATASET"
ANOMALY_LIST_FILE = "Anomaly_Train.txt"
NORMAL_LIST_FILE = "Normal_Train.txt"

# ========== Model Configuration ==========
MODEL_CHECKPOINT_PATH = "models/checkpoint.pth"
BEST_MODEL_PATH = "models/best_anomaly_model.pth"
YOLO_WEIGHTS_PATH = "yolov8n.pt"

# ========== Training Configuration ==========
TRAIN_BATCH_SIZE = 32
TRAIN_EPOCHS = 1000
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
DROPOUT_RATE = 0.6

# ========== Model Architecture ==========
FEATURE_DIM = 4096           # C3D feature dimension
SEGMENT_COUNT = 32           # Segments per video
HIDDEN_DIM_1 = 512           # First hidden layer
HIDDEN_DIM_2 = 32            # Second hidden layer

# ========== Loss Function Parameters ==========
MIL_RANKING_MARGIN = 1.0
SPARSITY_PENALTY = 0.00008
SMOOTHNESS_PENALTY = 0.00008

# ========== Detection Configuration ==========
CONFIDENCE_THRESHOLD = 0.60
ANOMALY_THRESHOLD = 0.75
SCENE_CONFIDENCE_THRESHOLD = 0.85
WEAPON_CONF_THRESHOLD = 0.85

# ========== YOLO Detection Classes ==========
COCO_PERSON_CLASS = 0
COCO_WEAPON_CLASSES = [43, 34, 76]  # knife, baseball bat, scissors

# ========== Scene Understanding (CLIP) ==========
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
SCENE_HISTORY_SIZE = 10  # Behavioral smoothing window

# ========== Ingest Configuration ==========
VIDEO_STREAM_SOURCE = 0  # 0 for webcam, or RTSP URL
INGEST_FPS = 15
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# ========== Geo-location Defaults ==========
DEFAULT_CAMERA_LAT = 40.7128
DEFAULT_CAMERA_LON = -74.0060
DEFAULT_CAMERA_NAME = "Cam 04 - High Street"

# ========== Emergency Response ==========
THREAT_KEYWORDS = ["suspiciously", "hiding", "fight", "robbery", "weapon", "casing", "panic", "lurking"]
LOITERING_SECONDS_THRESHOLD = 90

# ========== Logging ==========
LOG_LEVEL = "INFO"
