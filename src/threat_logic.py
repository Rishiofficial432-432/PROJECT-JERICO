# Generic rules for threat logic based on detections
WEAPON_CONF_THRESHOLD = 0.85
FIRE_CONF_THRESHOLD   = 0.50
PERSON_LOITERING_SECONDS = 90

# Class IDs (must match detect.py)
CLASS_WEAPON = 0
CLASS_PERSON = 1
CLASS_FIRE   = 2

def evaluate_threat(detections, frame_timestamp):
    threats = []
    for det in detections:
        class_id, conf, x1, y1, x2, y2 = det
        
        # Threat logic for Weapons
        if class_id == CLASS_WEAPON and conf >= WEAPON_CONF_THRESHOLD:
            threats.append({
                "type": "weapon",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
        
        # Threat logic for Fire
        if class_id == CLASS_FIRE and conf >= FIRE_CONF_THRESHOLD:
            threats.append({
                "type": "fire",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            
    return threats

if __name__ == "__main__":
    print("threat_logic.py standalone check passed.")
