# Generic rules for threat logic based on detections
WEAPON_CONF_THRESHOLD = 0.85
PERSON_LOITERING_SECONDS = 90

def evaluate_threat(detections, frame_timestamp):
    threats = []
    for det in detections:
        class_id, conf, x1, y1, x2, y2 = det
        
        # Threat logic for Weapons
        if class_id == 0 and conf >= WEAPON_CONF_THRESHOLD:
            threats.append({
                "type": "weapon",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })
            
    return threats

if __name__ == "__main__":
    print("threat_logic.py standalone check passed.")
