from typing import Dict, List

try:
    from threat_logic import evaluate_threat
    from tracker import CentroidTracker
except ImportError:
    from src.threat_logic import evaluate_threat
    from src.tracker import CentroidTracker


PRIORITY_SCORE = {
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
    "CRITICAL": 4,
}


class HybridThreatStack:
    """
    Concrete production stack:
    - Always-on: YOLO detections + tracker + rule-based threat fusion
    - On-alert: optional heavyweight reasoning (scene model) is enabled by caller
    """

    def __init__(self, min_object_conf: float = 0.60):
        self.min_object_conf = min_object_conf
        self.tracker = CentroidTracker()

    def _prioritize(self, threats: List[Dict]) -> str:
        if not threats:
            return "LOW"
        top = max(threats, key=lambda t: PRIORITY_SCORE.get(t.get("priority", "LOW"), 1))
        return top.get("priority", "LOW")

    def process(self, detections: List[List[float]], anomaly_score: float, anomaly_threshold: float) -> Dict:
        filtered = [d for d in detections if d[1] >= self.min_object_conf]

        track_boxes = [[d[2], d[3], d[4], d[5]] for d in filtered]
        tracking = self.tracker.update(track_boxes)

        fused_threats = evaluate_threat(filtered)
        anomaly_triggered = anomaly_score >= anomaly_threshold

        severity = self._prioritize(fused_threats)
        if anomaly_triggered and severity in ("LOW", "MEDIUM"):
            severity = "HIGH"

        should_run_heavy_reasoning = bool(fused_threats) or anomaly_triggered

        return {
            "filtered_detections": filtered,
            "tracking": tracking,
            "fused_threats": fused_threats,
            "anomaly_triggered": anomaly_triggered,
            "severity": severity,
            "should_run_heavy_reasoning": should_run_heavy_reasoning,
        }
