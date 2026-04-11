import sys

with open("frontend/api.py", "r") as f:
    lines = f.readlines()

new_content = []
inserted = False

for line in lines:
    if line.strip().startswith("#  Queue-Buffered Video Processing") and not inserted:
        # Insert helpers before the video processing block
        new_content.append("# ─────────────────────────────────────────────\n")
        new_content.append("#  Detection & Encoding Helpers\n")
        new_content.append("# ─────────────────────────────────────────────\n\n")
        
        new_content.append("def _encode_frame_b64(frame, boxes=None):\n")
        new_content.append("    \"\"\"Encodes a frame to base64 for the frontend, optionally drawing boxes.\"\"\"\n")
        new_content.append("    try:\n")
        new_content.append("        if boxes is not None:\n")
        new_content.append("            temp_frame = frame.copy()\n")
        new_content.append("            for b in boxes:\n")
        new_content.append("                if len(b) >= 4:\n")
        new_content.append("                    x1, y1, x2, y2 = map(int, b[:4])\n")
        new_content.append("                    cv2.rectangle(temp_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)\n")
        new_content.append("            _, buffer = cv2.imencode('.jpg', temp_frame)\n")
        new_content.append("        else:\n")
        new_content.append("            _, buffer = cv2.imencode('.jpg', frame)\n")
        new_content.append("        return \"data:image/jpeg;base64,\" + base64.b64encode(buffer).decode()\n")
        new_content.append("    except Exception as e:\n")
        new_content.append("        logger.error(f\"Encoding error: {e}\")\n")
        new_content.append("        return \"\"\n\n")

        new_content.append("def _analyse_detections(detections, th, frame):\n")
        new_content.append("    \"\"\"\n")
        new_content.append("    Bridge between raw detections and Situation Summary.\n")
        new_content.append("    Applies thresholds and identifies active threat flags.\n")
        new_content.append("    \"\"\"\n")
        new_content.append("    # 1. Get situation object from threat_logic (which uses refinement)\n")
        new_content.append("    threats = evaluate_threat(detections)\n")
        new_content.append("    \n")
        new_content.append("    # 2. Extract metrics for dashboard\n")
        new_content.append("    w_conf = max([t[\"confidence\"] for t in threats if \"weapon\" in t[\"type\"].lower()] or [0.0])\n")
        new_content.append("    f_conf = max([t[\"confidence\"] for t in threats if t[\"type\"] == \"fire\"] or [0.0])\n")
        new_content.append("    r_conf = max([t[\"confidence\"] for t in threats if t[\"type\"] in [\"vehicle_collision\", \"road_anomaly\"]] or [0.0])\n")
        new_content.append("    \n")
        new_content.append("    # Return unified dict for state updates\n")
        new_content.append("    return {\n")
        new_content.append("        \"has_threat\": len(threats) > 0,\n")
        new_content.append("        \"has_fire\": any(t[\"type\"] == \"fire\" for t in threats),\n")
        new_content.append("        \"threat_types\": list(set(t[\"type\"] for t in threats)),\n")
        new_content.append("        \"weapon_conf\": w_conf,\n")
        new_content.append("        \"fire_conf\": f_conf,\n")
        new_content.append("        \"road_conf\": r_conf,\n")
        new_content.append("        \"boxes\": [d[2:6] for d in detections], # x1,y1,x2,y2\n")
        new_content.append("        \"vehicle_count\": sum(1 for d in detections if d[0] == 4) # CLASS_VEHICLE is 4\n")
        new_content.append("    }\n\n")
        inserted = True
    new_content.append(line)

with open("frontend/api.py", "w") as f:
    f.writelines(new_content)
