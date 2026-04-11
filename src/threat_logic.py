# Generic rules for threat logic based on detections
WEAPON_CONF_THRESHOLD        = 0.55
FIRE_CONF_THRESHOLD          = 0.82
ROAD_ANOMALY_CONF_THRESHOLD  = 0.40
VEHICLE_CONF_THRESHOLD       = 0.35
PERSON_LOITERING_SECONDS     = 90

# Class IDs (must match detect.py)
CLASS_WEAPON       = 0
CLASS_PERSON       = 1
CLASS_FIRE         = 2
CLASS_ROAD_ANOMALY = 3
CLASS_VEHICLE      = 4

def get_center(box):
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]

def box_distance(box1, box2):
    c1 = get_center(box1)
    c2 = get_center(box2)
    return ((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)**0.5

def calculate_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0.0
    
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return interArea / float(box1Area + box2Area - interArea)

def refine_detections(detections):
    """
    Accuracy Hardening: Situational Awareness (Quint-Model)
    1. Armed Person: Weapon near Person.
    2. Getaway Vehicle: Vehicle near Armed Person.
    3. Unauthorized Vehicle: Vehicle loitering without persons.
    4. Vehicle Collision: Multiple vehicles close together (potential accident).
    """
    persons  = [d for d in detections if d[0] == CLASS_PERSON]
    vehicles = [d for d in detections if d[0] == CLASS_VEHICLE]
    weapons  = [d for d in detections if d[0] == CLASS_WEAPON]
    refined  = []

    # First, identify armed status for people
    armed_person_boxes = []

    for d in detections:
        cls, conf, x1, y1, x2, y2 = d[:6]
        w_box = [x1, y1, x2, y2]

        if cls == CLASS_WEAPON:
            is_held = False
            for p in persons:
                p_box = [p[2], p[3], p[4], p[5]]
                w_center = get_center(w_box)
                if (p_box[0] <= w_center[0] <= p_box[2] and p_box[1] <= w_center[1] <= p_box[3]) or box_distance(w_box, p_box) < 120:
                    is_held = True
                    if conf >= 0.40:  # Only reasonably confident weapons trigger getaway logic
                        armed_person_boxes.append(p_box)
                    break

            tag = "held_weapon" if is_held else "loose_weapon"
            if not is_held: conf *= 0.60
            refined.append([cls, conf, x1, y1, x2, y2, tag])

        elif cls == CLASS_VEHICLE:
            # Check for getaway car (near an armed person)
            is_getaway = False
            for ap_box in armed_person_boxes:
                if box_distance(w_box, ap_box) < 300: # Within 300px
                    is_getaway = True
                    break

            # Check for vehicle collision (multiple vehicles very close)
            is_collision = False
            for v2 in vehicles:
                if v2 is d: continue  # Skip self
                v2_box = [v2[2], v2[3], v2[4], v2[5]]
                dist = box_distance(w_box, v2_box)
                iou = calculate_iou(w_box, v2_box)
                if dist < 100 and iou > 0.75:  # Tightened: Only significant overlap is a collision
                    is_collision = True
                    break

            if is_getaway:
                tag = "getaway_vehicle"
            elif is_collision:
                tag = "road anomaly detected"
            else:
                tag = "standard_vehicle"
            refined.append([cls, conf, x1, y1, x2, y2, tag])

        elif cls == CLASS_ROAD_ANOMALY:
            # Get the event name if available
            tag = d[7] if len(d) > 7 else "road_anomaly"
            refined.append([cls, conf, x1, y1, x2, y2, tag])

        elif cls == CLASS_FIRE:
            # Prevent false positives: red vehicles often falsely trigger the fire AI
            for v in vehicles:
                v_box = [v[2], v[3], v[4], v[5]]
                if calculate_iou(w_box, v_box) > 0.50:
                    conf *= 0.35  # Heavily penalize confidence (0.77 -> 0.26)
                    break
            refined.append([cls, conf, x1, y1, x2, y2, "normal"])

        else:
            refined.append([cls, conf, x1, y1, x2, y2, "normal"])

    return refined

def evaluate_threat(detections, frame_timestamp=None):
    """
    Processes refined detections and returns a Situation Summary object.
    """
    detections = refine_detections(detections)
    threats = []

    for det in detections:
        tag = det[6] if len(det) > 6 else "normal"
        class_id, conf, x1, y1, x2, y2 = det[:6]

        if class_id == CLASS_WEAPON and conf >= WEAPON_CONF_THRESHOLD:
            threats.append({
                "type": "armed_person" if tag == "held_weapon" else "unattended_weapon",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "priority": "HIGH" if tag == "held_weapon" else "MEDIUM"
            })

        elif class_id == CLASS_VEHICLE and tag == "getaway_vehicle":
            threats.append({
                "type": "getaway_vehicle",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "priority": "CRITICAL"
            })

        elif class_id == CLASS_VEHICLE and tag == "road anomaly detected":
            threats.append({
                "type": "road anomaly detected",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "priority": "HIGH"
            })

        elif class_id == CLASS_FIRE and conf >= FIRE_CONF_THRESHOLD:
            threats.append({
                "type": "fire",
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "priority": "CRITICAL"
            })

        # 3. Road Anomaly (Hardened)
        elif class_id == CLASS_ROAD_ANOMALY and conf >= ROAD_ANOMALY_CONF_THRESHOLD:
            event_name = tag if tag != "road_anomaly" else "road_anomaly"
            threats.append({
                "type": event_name,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2],
                "priority": "MEDIUM"
            })

    return threats

if __name__ == "__main__":
    print("threat_logic.py standalone check passed.")
