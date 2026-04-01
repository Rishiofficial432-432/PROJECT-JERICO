import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. Dataset Loading ---
class UCFCrimeFeatureDataset(Dataset):
    def __init__(self, data_dir, list_file, is_anomaly):
        self.data_dir = data_dir
        self.is_anomaly = is_anomaly
        self.feature_paths = []
        
        print(f"Indexing feature files for {'Anomaly' if is_anomaly else 'Normal'} dataset...")
        
        # Pre-build index for fast lookup to avoid searching per item
        self.video_to_path = {}
        all_txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
        for filepath in all_txt_files:
            bname = os.path.basename(filepath)
            vname = bname[:-4] # Remove '.txt'
            self.video_to_path[vname] = filepath
            
        with open(list_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                video_path_str = line.strip()
                if not video_path_str:
                    continue
                video_name = video_path_str.split('.')[0] # e.g. Abuse/Abuse001_x264
                video_basename = video_name.split('/')[-1] # e.g. Abuse001_x264
                
                if video_basename in self.video_to_path:
                    self.feature_paths.append(self.video_to_path[video_basename])
                else:
                    # In case the exact casing or naming is slightly off, we skip for robust running
                    pass
                    
        print(f"Found {len(self.feature_paths)} feature files for this split.")
                
    def __len__(self):
        return len(self.feature_paths)
        
    def __getitem__(self, idx):
        feat_file = self.feature_paths[idx]
        
        try:
            # Each text file usually has 32 lines, each line 4096 dimensions (C3D)
            features = np.loadtxt(feat_file)
        except Exception:
            features = np.zeros((32, 4096))
            
        # Ensure exact shape of 32x4096
        if len(features.shape) == 1:
            # Sometimes a file might only have 1 line
            features = np.expand_dims(features, axis=0)
            
        if features.shape[0] < 32:
            pad = np.zeros((32 - features.shape[0], features.shape[1]))
            features = np.vstack([features, pad])
        elif features.shape[0] > 32:
            features = features[:32, :]
            
        if features.shape[1] != 4096:
            # Fallback if dimensions are completely mismatched (e.g. accidentally loading Inception features)
            # Default to zero pad/truncate to map 4096 for our model.json
            if features.shape[1] < 4096:
                pad_w = np.zeros((32, 4096 - features.shape[1]))
                features = np.hstack([features, pad_w])
            else:
                features = features[:, :4096]
            
        # Standard L2 Normalization
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1 # Prevent div-by-zero
        features = features / norms
        
        return torch.tensor(features, dtype=torch.float32), torch.tensor(1.0 if self.is_anomaly else 0.0, dtype=torch.float32)

def generate_batch(batch):
    features = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    return torch.stack(features), torch.stack(labels)


# --- 2. Model Architecture ---
class MILAnomalyModel(nn.Module):
    """
    Replication of the 3-layer MLP architecture defined in model.json.
    Multiple Instance Learning network for identifying video-level anomalies.
    """
    def __init__(self, input_dim=4096):
        super(MILAnomalyModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.6)
        
        self.fc2 = nn.Linear(512, 32)
        # Linear activation -> no intermediate non-linearity
        self.dropout2 = nn.Dropout(0.6)
        
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
        self._init_weights()

    def _init_weights(self):
        # Keras 'glorot_normal' matches PyTorch's xavier_normal_
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        # x is (batch_size, 32, 4096)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        # output is (batch_size, 32, 1)
        return x


# --- 3. Custom MIL Ranking Loss ---
def mil_ranking_loss(y_pred, y_true):
    """
    y_pred: (B, 32, 1) - prediction scores for each segment
    y_true: (B,) - 1 for anomaly bag, 0 for normal bag
    """
    loss = torch.tensor(0.0, device=y_pred.device, requires_grad=True)
    
    pos_mask = (y_true == 1.0)
    neg_mask = (y_true == 0.0)
    
    pos_preds = y_pred[pos_mask] # (N_pos, 32, 1)
    neg_preds = y_pred[neg_mask] # (N_neg, 32, 1)
    
    if len(pos_preds) == 0 or len(neg_preds) == 0:
        return torch.sum(y_pred * 0) # Dummy gradients to avoid errors
    
    # We take the max score from each bag
    pos_max = torch.max(pos_preds, dim=1)[0] # (N_pos, 1)
    neg_max = torch.max(neg_preds, dim=1)[0] # (N_neg, 1)
    
    # Pairwise Ranking Margin Loss
    # We enforce that positive max score > negative max score + margin
    margin = 0.5
    
    # Vectorized computation of hinge loss across all pos-neg pairs
    pos_max_expanded = pos_max.expand(len(pos_preds), len(neg_preds))
    neg_max_expanded = neg_max.view(1, -1).expand(len(pos_preds), len(neg_preds))
    
    ranking_err = torch.relu(margin - pos_max_expanded + neg_max_expanded)
    loss = loss + torch.sum(ranking_err)
            
    # Sparsity penalty: limits anomalous signals strictly to exact scenes
    sparsity_loss = torch.sum(pos_preds) * 0.000001
    
    # Temporal smoothness: neighboring scenes should have comparable scores
    smoothness_loss = torch.sum((pos_preds[:, 1:, :] - pos_preds[:, :-1, :]) ** 2) * 0.000001
    
    total_loss = (loss / (len(pos_preds) * len(neg_preds))) + sparsity_loss + smoothness_loss
    return total_loss

# --- 4. Training Loop ---
def train_model():
    dataset_dir = "DATASET"
    anomaly_list = os.path.join(dataset_dir, "Anomaly_Train.txt")
    normal_list = os.path.join(dataset_dir, "Normal_Train.txt")
    
    # Validate dataset structure
    if not os.path.exists(dataset_dir):
        print(f"\n❌ ERROR: Dataset directory '{dataset_dir}' not found!")
        print(f"📥 Place UCF-Crime C3D features in: {os.path.abspath(dataset_dir)}/")
        print(f"Expected structure:")
        print(f"   DATASET/")
        print(f"   ├── Anomaly_Train.txt")
        print(f"   ├── Normal_Train.txt")
        print(f"   └── Features/")
        print(f"       ├── Abuse/")
        print(f"       ├── Fighting/")
        print(f"       └── ... (other categories)")
        return
    
    if not os.path.exists(anomaly_list):
        print(f"❌ ERROR: {anomaly_list} not found!")
        print(f"Please ensure Anomaly_Train.txt exists in the DATASET folder.")
        return
    
    if not os.path.exists(normal_list):
        print(f"❌ ERROR: {normal_list} not found!")
        print(f"Please ensure Normal_Train.txt exists in the DATASET folder.")
        return
    
    # Automatically switch to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=========================================")
    print(f"      USING DEVICE: {device.type.upper()}             ")
    print(f"=========================================")
    
    print("Loading datasets...")
    pos_dataset = UCFCrimeFeatureDataset(dataset_dir, anomaly_list, is_anomaly=True)
    neg_dataset = UCFCrimeFeatureDataset(dataset_dir, normal_list, is_anomaly=False)
    
    if len(pos_dataset) == 0 and len(neg_dataset) == 0:
        print("\n❌ ERROR: Could not find any feature files!")
        print(f"Expected C3D feature .txt files in: {os.path.abspath(dataset_dir)}/Features/")
        print(f"✅ SOLUTION:")
        print(f"1. Download UCF-Crime C3D features from Kaggle:")
        print(f"   https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset")
        print(f"2. Extract Features/ folder into: {os.path.abspath(dataset_dir)}/Features/")
        print(f"3. Ensure Anomaly_Train.txt and Normal_Train.txt list video names correctly")
        return
    
    if len(pos_dataset) == 0:
        print(f"\n⚠️  WARNING: No anomaly features found. Loaded {len(neg_dataset)} normal videos.")
    elif len(neg_dataset) == 0:
        print(f"\n⚠️  WARNING: No normal features found. Loaded {len(pos_dataset)} anomaly videos.")

    full_dataset = torch.utils.data.ConcatDataset([pos_dataset, neg_dataset])
    dataloader = DataLoader(full_dataset, batch_size=32, shuffle=True, collate_fn=generate_batch, drop_last=False)
    
    model = MILAnomalyModel().to(device)
    
    # Auto-Resume from Stateful Checkpoint
    start_epoch = 1
    best_loss = float("inf")
    checkpoint_path = "models/checkpoint.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading stateful checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Base config
        optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.001) 
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_loss = checkpoint.get('best_loss', float("inf"))
    else:
        # Base configuration utilizes SGD with Adagrad logic or direct Adagrad
        optimizer = optim.Adagrad(model.parameters(), lr=0.001, weight_decay=0.001) 
        if os.path.exists("models/best_anomaly_model.pth"):
            print("Loading pre-existing weights to resume training...")
            model.load_state_dict(torch.load("models/best_anomaly_model.pth", map_location=device, weights_only=True))

    epochs = 1000
    
    print(f"\nStarting Training from Epoch {start_epoch}...")
    for epoch in range(start_epoch - 1, epochs):
        model.train()
        epoch_loss = 0.0
        
        from tqdm import tqdm
        batch_iter = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", leave=True, dynamic_ncols=True, colour='green')
        for batch_idx, (features, labels) in enumerate(batch_iter):
            features = features.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            preds = model(features)
            loss = mil_ranking_loss(preds, labels)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 5 == 0:
                batch_iter.set_postfix(loss=f"{loss.item():.4f}")
                
        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"===> Epoch {epoch+1} Average Loss: {avg_loss:.4f}")
        
        # Stateful Checkpoint Generator
        os.makedirs("models", exist_ok=True)
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_loss': best_loss
        }, "models/checkpoint.pth")
        
        # Checkpoint saving
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "models/best_anomaly_model.pth")
            print(f"-> Saved new best model to models/best_anomaly_model.pth with loss {best_loss:.4f}")

if __name__ == "__main__":
    import gc
    torch.cuda.empty_cache()
    gc.collect()
    train_model()
