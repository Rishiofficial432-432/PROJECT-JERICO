import sys

with open("frontend/api.py", "r") as f:
    lines = f.readlines()

new_lines = lines[:214] # Up to the end of _inference_worker queue.task_done()

replacement = """
@app.post("/api/upload")
async def upload_footage(
    file: UploadFile = File(...),
    camera_name: str = "CCTV-01",
    person_conf: float = 0.60,
    weapon_conf: float = 0.85,
    fire_conf:   float = 0.70,
    road_conf:   float = 0.60,
    anomaly_alert: float = 70.0,
    vehicle_conf: float = 0.50,
    person_conf_threshold: float = None,
    weapon_conf_threshold: float = None,
):
    if person_conf_threshold is not None:
        person_conf = person_conf_threshold
    if weapon_conf_threshold is not None:
        weapon_conf = weapon_conf_threshold

    ext = Path(file.filename).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    _state.update({
        "active_camera": camera_name,
        "threat_active": False,
        "threat_type": [],
        "weapon_conf": 0.0,
        "fire_conf": 0.0,
        "road_conf": 0.0,
        "last_filename": file.filename
    })

    th = {"person": person_conf, "weapon": weapon_conf, "fire": fire_conf, "road": road_conf, "vehicle": vehicle_conf}
    executor = get_executor()

    if ext in IMAGE_EXTS:
        try:
            frame = cv2.imread(tmp_path)
            if frame is None:
                return JSONResponse(status_code=400, content={"error": "Could not read image file."})
            
            dets = await run_inference_async(
                frame, 
                person_conf=th["person"], 
                weapon_conf=th["weapon"], 
                fire_conf=th["fire"], 
                road_conf=th["road"], 
                vehical_conf=th["vehicle"],
                executor=executor
            )
            res = _analyse_detections(dets, th, frame)
            
            scene_l, scene_c = ("—", 0.0)
            if _scene_analyzer:
                scene_l, scene_c = await asyncio.to_thread(_scene_analyzer.analyze_frame, frame)

            frame_b64 = _encode_frame_b64(frame, res["boxes"])
            _update_state_post_process(res, scene_l, scene_c, frame_b64, file.filename)
            return {
                **res,
                "media_type": "image",
                "frame_b64": frame_b64,
                "scene_label": scene_l,
                "camera": camera_name,
                "total_detections": len(res.get("boxes", [])),
            }
        finally:
            try:
                os.unlink(tmp_path)
            except: pass
    else:
        # Video Processing setup
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            try:
                os.unlink(tmp_path)
            except: pass
            return JSONResponse(status_code=400, content={"error": "Could not open video file."})
            
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(fps_val * 0.3)) 
        
        async def _vid_generator():
            try:
                queue = asyncio.Queue(maxsize=20)
                yield_queue = asyncio.Queue(maxsize=32)
                results = []
                num_workers = 3
                
                producer = asyncio.create_task(_frame_producer(cap, interval, queue, num_workers))
                workers = [asyncio.create_task(_inference_worker(queue, results, th, executor, yield_queue, total)) for _ in range(num_workers)]
                
                active_workers = num_workers
                while active_workers > 0:
                    item = await yield_queue.get()
                    if item is None:
                        active_workers -= 1
                    else:
                        yield json.dumps(item) + "\\n"
                
                if cap and cap.isOpened():
                    cap.release()

                if not results:
                    yield json.dumps({"error": "No frames processed"}) + "\\n"
                    return

                results.sort(key=lambda x: x[0])

                def check_persistence(type_key, min_frames=2, instant_threshold=0.98):
                    type_results = [r[2] for r in results]
                    confirmed_conf = 0.0
                    active_types = set()
                    thresh_map = {"weapon_conf": WEAPON_CONF_THRESHOLD, "fire_conf": FIRE_CONF_THRESHOLD, "road_conf": ROAD_ANOMALY_CONF_THRESHOLD}
                    current_thresh = thresh_map.get(type_key, 0.5)

                    for i in range(len(type_results)):
                        r_frame = type_results[i]
                        conf = r_frame.get(type_key, 0)
                        if conf >= instant_threshold:
                            confirmed_conf = max(confirmed_conf, conf)
                            for ty in r_frame.get("threat_types", []):
                                if type_key.replace("_conf", "").lower() in ty.lower():
                                    active_types.add(ty)
                        elif conf >= current_thresh:
                            count = sum(1 for wr in type_results[max(0, i-2):i+3] if wr.get(type_key, 0) >= current_thresh * 0.85)
                            if count >= min_frames:
                                confirmed_conf = max(confirmed_conf, conf)
                                for ty in r_frame.get("threat_types", []):
                                    if type_key.replace("_conf", "").lower() in ty.lower():
                                        active_types.add(ty)
                    return confirmed_conf, active_types

                max_w, types_w = check_persistence("weapon_conf", min_frames=1)
                max_f, types_f = check_persistence("fire_conf", min_frames=2)
                max_r, types_r = check_persistence("road_conf", min_frames=2)
                threat_types = types_w | types_f | types_r
                
                narrative = []
                for idx, frame, r in results:
                    ts = round(idx / fps_val, 1)
                    if r["threat_types"]:
                        narrative.append(f"{ts}s: {','.join(r['threat_types'])}")
                
                final_summary = " | ".join(narrative[:5]) + ("..." if len(narrative) > 5 else "")
                if not final_summary: final_summary = "Normal activity observed."

                worst = sorted(results, key=lambda x: (x[2].get("weapon_conf",0), x[2].get("fire_conf",0), x[2].get("road_conf",0)), reverse=True)
                frame_b64 = _encode_frame_b64(worst[0][1], worst[0][2]["boxes"]) if worst else ""

                max_v = max([r[2].get("vehicle_count", 0) for r in results]) if results else 0

                has_threat = len(threat_types) > 0
                has_fire = max_f > 0
                _update_state_post_process(
                    {"weapon_conf": max_w, "fire_conf": max_f, "road_conf": max_r, "has_threat": has_threat, "has_fire": has_fire, "threat_types": list(threat_types), "vehicle_count": max_v},
                    final_summary, 1.0, frame_b64, file.filename, 0.0
                )

                duration = (total / fps_val) if fps_val > 0 else 0
                
                final_res = {
                    "media_type": "video",
                    "frame_b64": frame_b64,
                    "weapon_conf": max_w,
                    "fire_conf": max_f,
                    "road_conf": max_r,
                    "threat_types": list(threat_types),
                    "summary": final_summary,
                    "camera": camera_name,
                    "has_threat": has_threat,
                    "has_fire": has_fire,
                    "total_frames": total,
                    "duration_seconds": round(duration, 1),
                    "total_detections": sum(len(r[2].get("boxes", [])) for r in results),
                    "scene_label": final_summary,
                }
                yield json.dumps({"result": final_res}) + "\\n"
            finally:
                if cap and cap.isOpened(): cap.release()
                try:
                    os.unlink(tmp_path)
                except:
                    pass
        
        return StreamingResponse(_vid_generator(), media_type="application/x-ndjson")

def _update_state_post_process(res, scene_l, scene_c, b64, fname, anom=0.0):
    _state.update({
        "last_scan_ts": time.time(),
        "weapon_conf": res.get("weapon_conf", 0),
        "fire_conf": res.get("fire_conf", 0),
        "road_conf": res.get("road_conf", 0),
        "vehicle_count": res.get("vehicle_count", 0),
        "scene_label": scene_l,
        "scene_conf": scene_c,
        "anomaly_score": anom,
        "threat_active": res["has_threat"] or res["has_fire"],
        "threat_type": res.get("threat_types", []),
        "last_frame_b64": b64,
        "last_filename": fname
    })

"""

# We also need to fix _inference_worker in new_lines.
worker_fixed = []
for line in new_lines:
    if line.startswith("async def _inference_worker("):
        worker_fixed.append("async def _inference_worker(queue, results_list, th, executor, yield_queue=None, total_frames=1):\n")
    elif "queue.task_done()" in line:
        worker_fixed.append(line)
        if "    queue.task_done()" in line and "if item is None:" in worker_fixed[-3]:
            worker_fixed.append("            if yield_queue:\n                await yield_queue.put(None)\n")
    elif "r = _analyse_detections(refined_dets, th, frame)" in line:
        worker_fixed.append(line)
        worker_fixed.append("""
        if yield_queue:
            b64 = _encode_frame_b64(frame, r["boxes"])
            pct = min(99, int((idx / total_frames) * 100)) if total_frames > 0 else 0
            await yield_queue.put({"progress": pct, "frame_b64": b64})
""")
    else:
        worker_fixed.append(line)

with open("frontend/api.py", "w") as f:
    f.writelines(worker_fixed)
    f.write(replacement)
