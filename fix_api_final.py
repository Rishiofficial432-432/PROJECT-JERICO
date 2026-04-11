import sys

with open("frontend/api.py", "r") as f:
    lines = f.readlines()

new_content = []
skip_until = -1

for i, line in enumerate(lines):
    if i < skip_until:
        continue
    
    # Fix _inference_worker
    if line.strip().startswith("async def _inference_worker("):
        new_content.append("async def _inference_worker(queue, results_list, th, executor, yield_queue=None, total_frames=1):\n")
        new_content.append("    while True:\n")
        new_content.append("        item = await queue.get()\n")
        new_content.append("        if item is None:\n")
        new_content.append("            queue.task_done()\n")
        new_content.append("            if yield_queue:\n")
        new_content.append("                await yield_queue.put(None)\n")
        new_content.append("            break\n")
        new_content.append("        idx, frame = item\n")
        new_content.append("        # Parallel model inference\n")
        new_content.append("        dets = await run_inference_async(\n")
        new_content.append("            frame, \n")
        new_content.append("            person_conf=th.get(\"person\", 0.5),\n")
        new_content.append("            weapon_conf=0.1, # Allow refinement logic to handle the threshold\n")
        new_content.append("            fire_conf=th.get(\"fire\", 0.5), \n")
        new_content.append("            road_conf=th.get(\"road\", 0.5), \n")
        new_content.append("            vehical_conf=th.get(\"vehicle\", 0.5), \n")
        new_content.append("            executor=executor\n")
        new_content.append("        )\n")
        new_content.append("        \n")
        new_content.append("        # Apply Accuracy Hardening (Spatial Validation)\n")
        new_content.append("        refined_dets = refine_detections(dets)\n")
        new_content.append("        r = _analyse_detections(refined_dets, th, frame)\n")
        new_content.append("        results_list.append((idx, frame, r))\n")
        new_content.append("\n")
        new_content.append("        if yield_queue:\n")
        new_content.append("            b64 = _encode_frame_b64(frame, r[\"boxes\"])\n")
        new_content.append("            pct = min(99, int((idx / total_frames) * 100)) if total_frames > 0 else 0\n")
        new_content.append("            await yield_queue.put({\"progress\": pct, \"frame_b64\": b64})\n")
        new_content.append("\n")
        new_content.append("        # Real-time Flagging: Update global state immediately if threat found or every 5th analysed frame\n")
        new_content.append("        is_threat = r[\"has_threat\"] or r[\"has_fire\"]\n")
        new_content.append("        if is_threat or idx % 5 == 0:\n")
        new_content.append("            frame_b64 = _encode_frame_b64(frame, r[\"boxes\"])\n")
        new_content.append("            _state.update({\n")
        new_content.append("                \"last_scan_ts\": time.time(),\n")
        new_content.append("                \"weapon_conf\": max(_state[\"weapon_conf\"], r.get(\"weapon_conf\", 0)),\n")
        new_content.append("                \"fire_conf\": max(_state[\"fire_conf\"], r.get(\"fire_conf\", 0)),\n")
        new_content.append("                \"road_conf\": max(_state[\"road_conf\"], r.get(\"road_conf\", 0)),\n")
        new_content.append("                \"threat_active\": _state[\"threat_active\"] or is_threat,\n")
        new_content.append("                \"threat_type\": list(set(_state[\"threat_type\"]) | set(r[\"threat_types\"])),\n")
        new_content.append("                \"last_frame_b64\": frame_b64,\n")
        new_content.append("            })\n")
        new_content.append("            if is_threat:\n")
        new_content.append("                _log_event(_state[\"active_camera\"], \" + \".join(r[\"threat_types\"]), \n")
        new_content.append("                          max(r[\"weapon_conf\"], r[\"fire_conf\"], r[\"road_conf\"]), \"active\")\n")
        new_content.append("        \n")
        new_content.append("        queue.task_done()\n")
        
        # Skip the old worker lines (roughly until upload_footage starts)
        j = i + 1
        while j < len(lines) and not lines[j].strip().startswith("@app.post(\"/api/upload\")"):
            j += 1
        skip_until = j
        continue
    
    new_content.append(line)

with open("frontend/api.py", "w") as f:
    f.writelines(new_content)
