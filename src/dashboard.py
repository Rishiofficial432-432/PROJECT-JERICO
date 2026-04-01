import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import time
import os
import logging
import numpy as np
from detect import run_inference, CLASS_WEAPON, CLASS_PERSON
from detect_anomaly import load_anomaly_model, lookup_features, predict_anomaly
from scene_understanding import SceneAnalyzer
from alert import dispatch_authorities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def play_siren_js():
    components.html(
        """
        <script>
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const osc = ctx.createOscillator();
        const gain = ctx.createGain();
        osc.connect(gain);
        gain.connect(ctx.destination);
        osc.type = 'sawtooth';
        gain.gain.setValueAtTime(0.4, ctx.currentTime);
        for (let i = 0; i < 6; i++) {
            osc.frequency.setValueAtTime(i % 2 === 0 ? 800 : 1200, ctx.currentTime + i * 0.5);
        }
        osc.start(ctx.currentTime);
        osc.stop(ctx.currentTime + 3);
        </script>
        """,
        height=0,
    )

@st.cache_resource
def get_anomaly_brain():
    try:
        return load_anomaly_model()
    except Exception as e:
        logger.error(f"Failed to load anomaly model: {e}")
        st.error(f"⚠️ Anomaly Model Load Error: {e}")
        return None, None

@st.cache_resource
def get_scene_analyzer():
    try:
        return SceneAnalyzer()
    except Exception as e:
        logger.error(f"Failed to initialize scene analyzer: {e}")
        return None

try:
    anomaly_model, device = get_anomaly_brain()
except Exception as e:
    anomaly_model, device = None, None
    logger.error(f"Critical initialization error: {e}")

try:
    scene_analyzer = get_scene_analyzer()
except Exception as e:
    scene_analyzer = None
    logger.error(f"Scene analyzer initialization warning: {e}")

st.set_page_config(page_title="CCTV Human Verification Dashboard", layout="wide")

st.title("🛡️ Human Verification Dashboard")

# --- Model Status Info ---
weight_path = "models/best_anomaly_model.pth"
last_update = "Never Trained"
if os.path.exists(weight_path):
    mtime = os.path.getmtime(weight_path)
    last_update = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))

status_emoji = "✅" if anomaly_model is not None else "⚠️"
st.markdown(f"**{status_emoji} Current Model State:** Trained up to `{last_update}`")
if anomaly_model is None:
    st.warning("⚠️ Anomaly detection model not loaded. Ensure DATASET/ folder exists and training has completed.")
st.markdown("Upload CCTV footage to run the anomaly detection models and review alerts.")

# Initialize session state for alerts
if "threat_triggered" not in st.session_state:
    st.session_state.threat_triggered = False
if "siren_played" not in st.session_state:
    st.session_state.siren_played = False

# --- Settings Sidebar ---
st.sidebar.header("⚙️ Detection Settings")

smart_mode = st.sidebar.toggle("🤖 Enable Smart Auto-Config", value=False, help="Intelligently locks settings to the optimal common configuration for all models.")

if not smart_mode:
    conf_threshold = st.sidebar.slider("Minimum Confidence Score", min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    anomaly_threshold = st.sidebar.slider("Anomaly Violence Trigger", min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    scene_threshold = st.sidebar.slider("Scene Confidence (CLIP)", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
else:
    conf_threshold = 0.60
    anomaly_threshold = 0.70
    scene_threshold = 0.25
    st.sidebar.success("**Smart Config Active**\n\nThe intelligent recognition system is utilizing standard parameters for unified detection.")

show_boxes = st.sidebar.toggle("Show Bounding Boxes", value=True)

st.sidebar.markdown("---")
st.sidebar.header("� Camera Geo-Location Targeting")
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

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

uploaded_file = st.file_uploader(
    "Upload Media (Video or Image)",
    type=["mp4", "avi", "mov", "mkv", "webm", "jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    is_image_file = file_ext in IMAGE_EXTENSIONS
    is_video_file = file_ext in VIDEO_EXTENSIONS

    if not is_image_file and not is_video_file:
        st.error(f"❌ Unsupported file format: {file_ext or 'unknown'}. Please upload an image or video file.")
        st.stop()

    if is_video_file and (anomaly_model is None or device is None):
        st.error("❌ Cannot process video: Anomaly model failed to load. Please check logs and restart.")
        st.stop()

    if is_image_file:
        st.success("Image uploaded successfully!")
        image_bytes = uploaded_file.read()
        image_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("❌ Failed to decode image. Please upload a valid image file.")
            st.stop()

        detections = run_inference(frame)
        has_threat = False
        threat_conf = 0.0

        for det in detections:
            cls_id, conf, x1, y1, x2, y2 = det
            if conf < conf_threshold:
                continue

            is_weapon = (cls_id == CLASS_WEAPON)
            if is_weapon:
                has_threat = True
                threat_conf = max(threat_conf, conf)

            color = (0, 0, 255) if is_weapon else (0, 255, 0)
            label = f"{'🔫 GUN' if is_weapon else 'Person'} ({conf:.2f})"

            if show_boxes:
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, label, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        col1, col2 = st.columns([2, 1])
        with col1:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", width="stretch")

        with col2:
            st.subheader("Image Threat Analytics")
            st.write(f"**Detections Found:** {len(detections)}")
            st.write(f"**Weapon Confidence:** {threat_conf:.2f}" if has_threat else "**Weapon Confidence:** 0.00")

            if scene_analyzer is not None:
                try:
                    scene_type, scene_prob = scene_analyzer.analyze_frame(frame)
                    st.write(f"**Scene Insight:** {scene_type}")
                    st.write(f"**Scene Confidence:** {scene_prob:.2f}")
                except Exception as e:
                    logger.warning(f"Scene analyzer failed on image: {e}")

            if has_threat:
                st.error("🚨🚨🚨 CRITICAL ALERT: WEAPON DETECTED IN IMAGE 🚨🚨🚨")
                threat_details = {
                    "weapon_detected": "Yes",
                    "weapon_confidence": f"{threat_conf:.2f}",
                    "objects_detected": len(detections),
                    "location": f"{active_geo_location['name']}",
                }
                
                # Dispatch to authorities
                dispatch_msg = dispatch_authorities(
                    "Armed Suspect - Gun Detected",
                    threat_conf,
                    active_geo_location,
                    threat_details
                )
                st.warning(dispatch_msg)
                
                # Play siren
                try:
                    play_siren_js()
                except Exception as e:
                    logger.error(f"Failed to generate siren: {e}")
            else:
                st.success("✅ Secure: No immediate weapon threat detected")

        st.info("Image processing complete.")
        st.stop()
    
    st.success("Video uploaded successfully!")
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=file_ext if is_video_file else ".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    col1, col2 = st.columns([2, 1])
    
    # 1. Look for pre-extracted features
    feature_path = lookup_features(uploaded_file.name)
    segment_scores = None
    
    if feature_path:
        st.toast(f"Matched UCF Dataset Features: {os.path.basename(feature_path)}", icon="✅")
        try:
            segment_scores = predict_anomaly(feature_path, anomaly_model, device)
        except Exception as e:
            logger.error(f"Anomaly prediction failed: {e}")
            st.error(f"Failed to predict anomaly: {e}")
            segment_scores = None
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
                
            is_weapon = (cls_id == CLASS_WEAPON)
            if is_weapon:
                has_threat = True
                threat_conf = max(threat_conf, conf)
            
            # Behavioral logic: Suspicious if it's a weapon, OR if violence is happening AND they are moving somewhat fast, OR if they are independently sprinting/moving erratically.
            is_fast_actor = speeds[i] > max(avg_speed * 1.5, 8.0)
            is_sprinting = speeds[i] > max(avg_speed * 2.0, 15.0) # Dangerous / erratic independent movement
            
            is_suspicious = is_weapon or (violence_detected and is_fast_actor) or is_sprinting
            
            color = (0, 0, 255) if is_suspicious else (0, 255, 0)
            label = f"{'🔫 GUN' if is_weapon else '🧍 Person'} ({conf:.2f}) [Spd: {int(speeds[i])}]"
            
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
                threat_msg = ' + '.join(thrt_str) if thrt_str else "UNKNOWN THREAT"
                alert_box.error(f"🚨🚨🚨 CRITICAL THREAT DETECTED 🚨🚨🚨\n\n**Threats:** {threat_msg}\n\n**Action:** Emergency dispatch activated!")
                
                if current_dispatch:
                    dispatch_box.warning(current_dispatch)
                
                if not st.session_state.siren_played:
                    st.session_state.siren_played = True
                    st.session_state.threat_triggered = True
                    threat_details = {
                        "weapons_detected": has_threat,
                        "violence_score": f"{anomaly_score*100:.1f}%",
                        "object_confidence": f"{threat_conf:.2f}",
                        "frame_number": frame_idx,
                        "total_frames": total_frames
                    }
                    
                    try:
                        play_siren_js()
                    except Exception as e:
                        logger.error(f"Failed to generate siren: {e}")
            else:
                alert_box.success("✅ Secure: Normal Activity")
                dispatch_box.empty()
                st.session_state.siren_played = False
                st.session_state.threat_triggered = False
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
        st.image("https://via.placeholder.com/640x480.png?text=Waiting+for+media+upload...", width="stretch")
    with col2:
        st.subheader("Event Details")
        st.write("**Threat Type:** None")
        st.write("**Confidence Score:** N/A")

