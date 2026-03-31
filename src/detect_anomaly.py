import torch
import torch.nn as nn
import numpy as np
import os
import glob

class MILAnomalyModel(nn.Module):
    """3-layer MLP identical to the training script"""
    def __init__(self, input_dim=4096):
        super(MILAnomalyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(512, 32)
        self.dropout2 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def load_anomaly_model(weight_path="models/best_anomaly_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MILAnomalyModel().to(device)
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
    model.eval()
    return model, device

def lookup_features(video_filename, dataset_dir="DATASET"):
    """Search DATASET folder for the corresponding .txt feature file."""
    base_name = os.path.splitext(os.path.basename(video_filename))[0]
    search_pattern = os.path.join(dataset_dir, "**", f"{base_name}.txt")
    matches = glob.glob(search_pattern, recursive=True)
    if len(matches) > 0:
        return matches[0]
    return None

def predict_anomaly(feature_txt_path, model, device):
    """Loads 4096-D features from the matching txt file and predicts the 32 segment anomaly scores."""
    try:
        features = np.loadtxt(feature_txt_path)
    except Exception:
        features = np.zeros((32, 4096))
        
    if len(features.shape) == 1:
        features = np.expand_dims(features, axis=0)
        
    if features.shape[0] < 32:
        pad = np.zeros((32 - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    elif features.shape[0] > 32:
        features = features[:32, :]
        
    if features.shape[1] < 4096:
        pad_w = np.zeros((32, 4096 - features.shape[1]))
        features = np.hstack([features, pad_w])
    else:
        features = features[:, :4096]
        
    # Standard L2 Normalization
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1 
    features = features / norms
    
    features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        preds = model(features_tensor) # (1, 32, 1)
        
    return preds[0].cpu().numpy().flatten() # (32,)
