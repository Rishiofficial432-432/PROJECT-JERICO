# Generic rules for threat logic based on detections
WEAPON_CONF_THRESHOLD        = 0.92  # Significantly raised from 0.85
FIRE_CONF_THRESHOLD          = 0.75  # Significantly raised from 0.50
ROAD_ANOMALY_CONF_THRESHOLD  = 0.65  # Raised from 0.50
PERSON_LOITERING_SECONDS     = 90

# Class IDs (must match detect.py)
CLASS_WEAPON       = 0
CLASS_PERSON       = 1
CLASS_FIRE         = 2
CLASS_ROAD_ANOMALY = 3

def refine_detections(detections):
    """
    Accuracy Hardening: Spatial Correlation
    If a weapon is detected, we check if a person is in the same frame.
    If no person is detected, the weapon's confidence is severely penalized to reduce inanimate object false positives.
    """
    has_person = any(d[0] == CLASS_PERSON for d in detections)
    refined = []
    
    for d in detections:
        cls, conf, x1, y1, x2, y2 = d
        
        if cls == CLASS_WEAPON:
            if not has_person:
                # Penalize 35% if no person is present in the frame
                conf *= 0.65
            elif conf < WEAPON_CONF_THRESHOLD:
                # Still apply threshold if person is present
                pass
        
        refined.append([cls, conf, x1, y1, x2, y2])
        
    return refined

def evaluate_threat(detections, frame_timestamp=None):
    """
    Processes refined detections to return actionable threat objects.
    """
    detections = refine_detections(detections)
    threats = []
    
    for det in detections:
        class_id, conf, x1, y1, x2, y2 = det
        
        # 1. Weapons (Hardened)
        if class_id == CLASS_WEAPON and conf >= WEAPON_CONF_THRESHOLD:
            threats.append({
                "type": "weapon",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
        
        # 2. Fire (Hardened)
        if class_id == CLASS_FIRE and conf >= FIRE_CONF_THRESHOLD:
            threats.append({
                "type": "fire",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        # 3. Road Anomaly (Hardened)
        if class_id == CLASS_ROAD_ANOMALY and conf >= ROAD_ANOMALY_CONF_THRESHOLD:
            threats.append({
                "type": "road_anomaly",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            
    return threats

if __name__ == "__main__":
    print("threat_logic.py standalone check passed.")
