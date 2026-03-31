import streamlit as st
import cv2
import tempfile
import time
import os
from detect import run_inference
from detect_anomaly import load_anomaly_model, lookup_features, predict_anomaly
from scene_understanding import SceneAnalyzer
from alert import dispatch_authorities

@st.cache_resource
def get_anomaly_brain():
    return load_anomaly_model()

@st.cache_resource
def get_scene_analyzer():
    return SceneAnalyzer()

anomaly_model, device = get_anomaly_brain()
scene_analyzer = get_scene_analyzer()

st.set_page_config(page_title="CCTV Human Verification Dashboard", layout="wide")

st.title("🛡️ Human Verification Dashboard")

# --- Model Status Info ---
weight_path = "models/best_anomaly_model.pth"
last_update = "Never"
if os.path.exists(weight_path):
    mtime = os.path.getmtime(weight_path)
    last_update = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))

st.markdown(f"**Current Model State:** Trained up to `{last_update}` (Epoch ~{619 if '619' in last_update else 'Updating...'})")
st.markdown("Upload CCTV footage to run the anomaly detection models and review alerts.")

# --- Settings Sidebar ---
st.sidebar.header("⚙️ Detection Settings")

smart_mode = st.sidebar.toggle("🤖 Enable Smart Auto-Config", value=False, help="Intelligently locks settings to the optimal common configuration for all models.")

if not smart_mode:
    conf_threshold = st.sidebar.slider("Minimum Confidence Score", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    anomaly_threshold = st.sidebar.slider("Anomaly Violence Trigger", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    scene_threshold = st.sidebar.slider("Scene Confidence (CLIP)", min_value=0.0, max_value=1.0, value=0.85, step=0.05)
else:
    conf_threshold = 0.60
    anomaly_threshold = 0.70
    scene_threshold = 0.80
    st.sidebar.success("**Smart Config Active**\n\nThe intelligent recognition system is utilizing standard parameters for unified detection.")

show_boxes = st.sidebar.toggle("Show Bounding Boxes", value=True)

st.sidebar.markdown("---")
st.sidebar.header("📍 Camera Geo-Location Targeting")
cam_name = st.sidebar.text_input("Camera Alias", value="Cam 04 - High Street")
cam_lat = st.sidebar.number_input("Latitude", value=40.7128, format="%.6f")
cam_lon = st.sidebar.number_input("Longitude", value=-74.0060, format="%.6f")

active_geo_location = {"lat": cam_lat, "lon": cam_lon, "name": cam_name}
st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Model Synchronization")
if st.sidebar.button("🔄 Reload Model Weights", help="Forces the dashboard to load the newest .pth file from the training process."):
    st.cache_resource.clear()
    st.rerun()
st.sidebar.caption(f"Last Weights Found: {last_update}")
st.sidebar.markdown("---")
# ------------------------

uploaded_file = st.file_uploader("Upload Video (MP4/AVI)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    st.success("Video uploaded successfully!")
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    col1, col2 = st.columns([2, 1])
    
    # 1. Look for pre-extracted features
    feature_path = lookup_features(uploaded_file.name)
    segment_scores = None
    
    if feature_path:
        st.toast(f"Matched UCF Dataset Features: {os.path.basename(feature_path)}", icon="✅")
        segment_scores = predict_anomaly(feature_path, anomaly_model, device)
    else:
        st.toast(f"Starting Universal Zero-Shot Classifier for {uploaded_file.name}...", icon="🌐")
        
    with col1:
        stframe = st.empty()
        
    with col2:
        st.subheader("Live Threat Analytics")
        metric_bx = st.empty()
        status_text = st.empty()
        alert_box = st.empty()
        dispatch_box = st.empty()
        
    if "current_dispatch" not in st.session_state:
        st.session_state.current_dispatch = None

    cap = cv2.VideoCapture(tfile.name)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    prev_threat_state = None
    prev_status_text = None
    prev_metric_val = None
    prev_dispatch = None
    
    # Simple centroid motion tracker
    prev_centroids = []
    
    # Behavioral Smoothing Buffers
    scene_history = []  # Stores recent [category, prob] pairs
    MAX_HISTORY = 10    # About 5 seconds of scene context

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        has_threat = False
        threat_conf = 0.0
        
        # --- 1. ACTION ANOMALY SCORE ---
        anomaly_score = 0.0
        if segment_scores is not None and total_frames > 0:
            seg_idx = int((frame_idx / total_frames) * 32)
            if seg_idx >= 32: seg_idx = 31
            anomaly_score = float(segment_scores[seg_idx])
            
        violence_detected = anomaly_score >= anomaly_threshold
        
        # --- 2. UNIVERSAL ZERO-SHOT SCENE UNDERSTANDING ---
        if frame_idx > 0 and frame_idx % 15 == 0:  # Sample twice a second, skip frame 0
            scene_type, scene_prob = scene_analyzer.analyze_frame(frame)
            
            # Update history and keep it within window
            scene_history.append((scene_type, scene_prob))
            if len(scene_history) > MAX_HISTORY:
                scene_history.pop(0)

            # Consensus logic: Is there a consistent threat across history?
            threat_keywords = ["suspiciously", "hiding", "fight", "robbery", "weapon", "casing", "panic", "lurking"]
            recent_threats = [p for t, p in scene_history if any(kw in t for kw in threat_keywords) and p > scene_threshold]
            
            # Universal threat: High-conf consensus across the rolling window
            is_universal_threat = len(recent_threats) >= 2 # At least 2 suspicious checks in a row
            
            if is_universal_threat:
                # Trigger exact emergency service responses
                st.session_state.current_dispatch = dispatch_authorities(scene_type, scene_prob, active_geo_location)
                st.session_state.univ_override = True # Keep state for in-between frames
                st.session_state.safe_counter = 0 # Reset clear timer
            else:
                # Increment safe counter for every negative CLIP check
                st.session_state.safe_counter = getattr(st.session_state, 'safe_counter', 0) + 1
                
        # Clear CLIP overlay if no threat seen for 3 checks (~1.5s total)
        if getattr(st.session_state, 'safe_counter', 0) >= 3:
            st.session_state.current_dispatch = None
            st.session_state.univ_override = False
            scene_history = [] # Reset history when safe
                
        if getattr(st.session_state, 'univ_override', False):
            violence_detected = True

        # --- 3. OBJECT DETECTION & MOTION TRACKING ---
        detections = run_inference(frame)
        current_centroids = []
        speeds = []
        
        # First pass: calculate speeds for all detections
        for det in detections:
            cls_id, conf, x1, y1, x2, y2 = det
            if conf < conf_threshold:
                speeds.append(0.0)
                current_centroids.append((0,0))
                continue
                
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            current_centroids.append((cx, cy))
            
            min_dist = float('inf')
            for (px, py) in prev_centroids:
                dist = ((cx - px)**2 + (cy - py)**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    
            # If distance is too big, it's likely a new person entering frame, not a teleporting fast person
            speed = min_dist if min_dist < 150 else 0.0
            speeds.append(speed)
            
        prev_centroids = [c for c in current_centroids if c != (0,0)]
        
        # Calculate scene average speed to find outliers (aggressors)
        valid_speeds = [s for s in speeds if s > 0]
        avg_speed = sum(valid_speeds) / max(1, len(valid_speeds))
        
        # Second pass: Draw bounding boxes with behavioral color logic
        for i, det in enumerate(detections):
            cls_id, conf, x1, y1, x2, y2 = det
            
            if conf < conf_threshold:
                continue
                
            is_weapon = (cls_id == 0)
            if is_weapon:
                has_threat = True
                threat_conf = max(threat_conf, conf)
            
            # Behavioral logic: Suspicious if it's a weapon, OR if violence is happening AND they are moving somewhat fast, OR if they are independently sprinting/moving erratically.
            is_fast_actor = speeds[i] > max(avg_speed * 1.5, 8.0)
            is_sprinting = speeds[i] > max(avg_speed * 2.0, 15.0) # Dangerous / erratic independent movement
            
            is_suspicious = is_weapon or (violence_detected and is_fast_actor) or is_sprinting
            
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            label = f"{'WEAPON' if is_weapon else 'Person'} ({conf:.2f}) [Spd: {int(speeds[i])}]"
            
            if show_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Convert BGR to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", width="stretch")
        
        # --- 4. UI ALERTS (Anti-Flickering Logic) ---
        metric_color = "normal" if not violence_detected else "inverse"
        current_metric_val = f"{anomaly_score * 100:.1f} %"
        if segment_scores is not None and current_metric_val != prev_metric_val:
            metric_bx.metric("Dataset Model Violence Prob", current_metric_val, delta="CRITICAL" if violence_detected else "SAFE", delta_color=metric_color)
            prev_metric_val = current_metric_val

        thrt_str = []
        if has_threat: thrt_str.append('🗡️ WEAPON (YOLO)')
        if violence_detected: thrt_str.append('🧨 VIOLENT ACTION')
        
        new_status_text = f"**Identified Threats:** {', '.join(thrt_str) if thrt_str else 'None'}\n\n**Max Object Conf:** {threat_conf:.2f}"
        if new_status_text != prev_status_text:
            status_text.write(new_status_text)
            prev_status_text = new_status_text
        
        current_threat_state = has_threat or violence_detected
        current_dispatch = getattr(st.session_state, 'current_dispatch', None)
        
        if current_threat_state != prev_threat_state or (current_threat_state and current_dispatch != prev_dispatch):
            if current_threat_state:
                alert_box.error(f"🚨 CRITICAL ALERT: {' + '.join(thrt_str)} DETECTED 🚨")
                if current_dispatch:
                    dispatch_box.warning(current_dispatch)
            else:
                alert_box.success("✅ Secure: Normal Activity")
                dispatch_box.empty()
            prev_threat_state = current_threat_state
            prev_dispatch = current_dispatch
            
        time.sleep(0.01) # Control playback speed
        frame_idx += 1
        
    cap.release()
    st.info("Video processing complete.")
else:
    # Placeholder UI when waiting
    col1, col2 = st.columns([2, 1])
    with col1:
        st.image("https://via.placeholder.com/640x480.png?text=Waiting+for+video+upload...", width="stretch")
    with col2:
        st.subheader("Event Details")
        st.write("**Threat Type:** None")
        st.write("**Confidence Score:** N/A")

