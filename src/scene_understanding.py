import torch
import cv2
from PIL import Image
import sys

try:
    from transformers import CLIPProcessor, CLIPModel
except ImportError:
    CLIPProcessor, CLIPModel = None, None

class SceneAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.categories = [
            "a car crash or traffic collision",
            "a violent street fight or physical assault",
            "an armed robbery, theft, or burglary",
            "normal peaceful street environment",
            "people walking and interacting normally"
        ]
        
        if CLIPModel is not None:
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        else:
            self.model = None
            
    def analyze_frame(self, frame):
        if self.model is None:
            return "model_not_installed", 0.0
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        
        inputs = self.processor(text=self.categories, images=image, return_tensors="pt", padding=True).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
            
        max_idx = probs.argmax()
        return self.categories[max_idx], probs[max_idx]
