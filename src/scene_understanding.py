"""
Causal scene understanding and reasoning firewall for Jerico.

Primary path:
- Gemini 3.1 Pro API for structured causal reasoning and timestamp correction.

Fallback path:
- Lightweight local caption heuristic for graceful degradation when API/key is missing.
"""

import io
import json
import logging
import os
import importlib
from typing import Any, Dict, List, Tuple

import cv2
from PIL import Image

logger = logging.getLogger(__name__)

try:
    dotenv_mod = importlib.import_module("dotenv")
    dotenv_mod.load_dotenv()
except Exception:
    pass


class SceneAnalyzer:
    def __init__(self):
        self._ready = False
        self._reasoner = None
        self._model_name = os.getenv("GEMINI_MODEL", "gemini-3.1-pro-preview")

        self.threat_keywords = [
            "weapon", "gun", "knife", "fight", "attack", "robbery", "assault",
            "threat", "armed", "violent", "suspicious", "hiding", "running", "panic", "fire"
        ]

        self._init_gemini()

    def _init_gemini(self) -> None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("SceneAnalyzer: GEMINI_API_KEY/GOOGLE_API_KEY not set. Running in fallback mode.")
            return

        try:
            genai = importlib.import_module("google.generativeai")

            genai.configure(api_key=api_key)
            self._reasoner = genai.GenerativeModel(self._model_name)
            self._ready = True
            logger.info(f"SceneAnalyzer: Gemini reasoning firewall ready ✅ ({self._model_name})")
        except Exception as e:
            logger.error(f"SceneAnalyzer: failed to initialize Gemini model: {e}")
            self._reasoner = None
            self._ready = False

    @staticmethod
    def _frame_to_pil(frame) -> Image.Image:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    @staticmethod
    def _extract_json(raw_text: str) -> Dict[str, Any]:
        if not raw_text:
            return {}
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1).strip()

        try:
            return json.loads(text)
        except Exception:
            # Attempt relaxed extraction if the model wraps JSON in prose.
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(text[start:end + 1])
                except Exception:
                    return {}
            return {}

    def _generate_reasoning(self, parts: List[Any], prompt: str) -> str:
        if not self._reasoner:
            return ""

        # Try thinking-level config first; fallback if unsupported by SDK version.
        try:
            resp = self._reasoner.generate_content(
                parts + [prompt],
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                    "thinking_level": "high",
                },
            )
            return getattr(resp, "text", "") or ""
        except Exception:
            try:
                resp = self._reasoner.generate_content(
                    parts + [prompt],
                    generation_config={
                        "response_mime_type": "application/json",
                        "temperature": 0.2,
                    },
                )
                return getattr(resp, "text", "") or ""
            except Exception as e:
                logger.error(f"SceneAnalyzer: Gemini generation failed: {e}")
                return ""

    def validate_threat_reasoning(self, video_segment: List[Any], detected_timestamp: str) -> Dict[str, Any]:
        """
        Run causal reasoning for a short, temporally anchored segment.

        Returns a dict with keys:
        - is_fire
        - is_accident
        - corrected_timestamp
        - reasoning
        - hazards
        """
        if not self._ready:
            return {
                "is_fire": False,
                "is_accident": False,
                "corrected_timestamp": detected_timestamp,
                "reasoning": "Gemini not configured; fallback mode active.",
                "hazards": [],
            }

        prompt = f"""
ANALYSIS TASK: Evaluate the incident at {detected_timestamp} in the provided video frames.

THINKING PROTOCOL:
1. Check RED color vs FIRE physics: solid geometry and rigid motion vs flickering volumetric expansion.
2. TEMPORAL ANCHOR: Use burned corner timestamps to report the exact second of impact.
3. SPATIAL GROUNDING: Report hazard boxes in [ymin, xmin, ymax, xmax].

RESPONSE SCHEMA (JSON only):
{{
  "is_fire": boolean,
  "is_accident": boolean,
  "corrected_timestamp": "MM:SS.ms",
  "reasoning": "string",
  "hazards": [{{"box_2d": [ymin, xmin, ymax, xmax], "label": "string"}}]
}}
""".strip()

        parts = []
        for item in video_segment:
            if isinstance(item, Image.Image):
                parts.append(item)
            else:
                try:
                    parts.append(self._frame_to_pil(item))
                except Exception:
                    continue

        raw = self._generate_reasoning(parts, prompt)
        data = self._extract_json(raw)
        if not data:
            return {
                "is_fire": False,
                "is_accident": False,
                "corrected_timestamp": detected_timestamp,
                "reasoning": "Reasoning parse failed; keeping detected timestamp.",
                "hazards": [],
            }

        data.setdefault("is_fire", False)
        data.setdefault("is_accident", False)
        data.setdefault("corrected_timestamp", detected_timestamp)
        data.setdefault("reasoning", "Reasoning response received.")
        data.setdefault("hazards", [])
        return data

    def analyze_frame(self, frame) -> Tuple[str, float]:
        """
        Lightweight per-frame scene summary.
        Keeps API compatibility with existing upload flow.
        """
        try:
            # If Gemini is ready, use a compact causal frame summary.
            if self._ready and self._reasoner:
                image = self._frame_to_pil(frame)
                prompt = (
                    "Return compact JSON: "
                    "{\"scene\": string, \"threat_score\": number[0..1], \"notes\": string}. "
                    "Distinguish red object color from actual flame physics."
                )
                raw = self._generate_reasoning([image], prompt)
                data = self._extract_json(raw)
                if data:
                    scene = str(data.get("scene", "scene_analysis"))
                    score = float(data.get("threat_score", 0.0))
                    return scene, max(0.0, min(score, 1.0))

            # Fallback local heuristic.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            # Very coarse hot-color ratio proxy.
            lower1 = (0, 120, 120)
            upper1 = (20, 255, 255)
            lower2 = (160, 120, 120)
            upper2 = (179, 255, 255)
            m1 = cv2.inRange(hsv, lower1, upper1)
            m2 = cv2.inRange(hsv, lower2, upper2)
            hot_ratio = float((m1.sum() + m2.sum()) / max(1, hsv.shape[0] * hsv.shape[1] * 255))
            desc = "scene_fallback_hotcolor" if hot_ratio > 0.08 else "scene_fallback_normal"
            return desc, min(1.0, hot_ratio * 2.0)
        except Exception as e:
            logger.error(f"SceneAnalyzer error: {e}")
            return "analysis_error", 0.0
