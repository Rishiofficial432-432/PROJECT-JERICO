"""
Scene understanding using OpenAI CLIP (openai/CLIP package).
Uses openai-clip directly — does NOT require transformers, avoids PyTorch >= 2.4 restriction.

Install: pip install git+https://github.com/openai/CLIP.git
"""

import logging
import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Try openai/clip first (works with torch 2.2.x) ──────────────────────────
try:
    import torch
    import clip as _clip                         # openai/CLIP package
    _CLIP_AVAILABLE = True
except Exception as e:
    logger.warning(f"openai-clip not available: {e}")
    _CLIP_AVAILABLE = False
    _clip = None

# ── Scene label categories ───────────────────────────────────────────────────
_CATEGORIES = [
    "a person looking around suspiciously and nervously",
    "a person hiding their face with a mask, hoodie, or hand",
    "a violent street fight, physical assault, or shoving",
    "an armed robbery, theft, or burglary in progress",
    "a person holding a knife, gun, or dangerous weapon",
    "a person casing a building or checking door handles",
    "a person running away in a state of panic or guilt",
    "a person loitering, crouching, or lurking in shadows",
    "normal peaceful street or room environment with calm people",
    "people walking, talking, and interacting normally",
]


class SceneAnalyzer:
    """
    Zero-shot scene classifier using CLIP.
    Returns (label_string, confidence_float) for every frame.
    """

    def __init__(self):
        self._ready = False
        self._warned_once = False

        if not _CLIP_AVAILABLE:
            logger.warning("SceneAnalyzer: openai-clip not installed. Scene labels will be unavailable.")
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = _clip.load("ViT-B/32", device=self.device)
            self.model.eval()

            # Pre-encode text tokens (done once at init — fast at inference time)
            tokens = _clip.tokenize(_CATEGORIES).to(self.device)
            with torch.no_grad():
                self._text_feats = self.model.encode_text(tokens)
                self._text_feats /= self._text_feats.norm(dim=-1, keepdim=True)

            self.categories = _CATEGORIES
            self._ready = True
            logger.info("SceneAnalyzer: CLIP ViT-B/32 loaded successfully ✅")

        except Exception as e:
            logger.error(f"SceneAnalyzer: failed to load CLIP — {e}")

    # ─────────────────────────────────────────────────────────────────────────

    def analyze_frame(self, frame: np.ndarray) -> tuple[str, float]:
        """
        Classify a BGR OpenCV frame.
        Returns (scene_label, confidence) or ('—', 0.0) if unavailable.
        """
        if not self._ready:
            if not self._warned_once:
                logger.warning("SceneAnalyzer not ready — skipping scene label.")
                self._warned_once = True
            return "—", 0.0

        try:
            # Convert BGR → PIL RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Preprocess + encode image
            img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                img_feat = self.model.encode_image(img_tensor)
                img_feat /= img_feat.norm(dim=-1, keepdim=True)

            # Cosine similarity → softmax probabilities
            logits = (img_feat @ self._text_feats.T) * self.model.logit_scale.exp()
            probs = logits.softmax(dim=-1).detach().cpu().numpy()[0]

            best_idx = int(probs.argmax())
            return self.categories[best_idx], float(probs[best_idx])

        except Exception as e:
            logger.error(f"SceneAnalyzer.analyze_frame error: {e}")
            return "analysis_error", 0.0
