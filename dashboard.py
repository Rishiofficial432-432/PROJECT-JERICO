import streamlit as st
import streamlit.components.v1 as components
import cv2
import tempfile
import time
import os
import sys
import logging
import secrets
import numpy as np

# Keep imports working after moving dashboard.py to project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from detect import run_inference, CLASS_WEAPON, CLASS_PERSON, CLASS_FIRE
from detect_anomaly import load_anomaly_model, lookup_features, predict_anomaly
from scene_understanding import SceneAnalyzer
from alert import (
    dispatch_authorities,
    generate_siren_audio,
    send_ntfy_alert,
    send_email_alert,
    send_whatsapp_alert,
)
import scipy.io.wavfile as wavfile
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_jet_black_theme():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

            :root {
                --jet-bg: #040404;
                --panel: #0b0b0c;
                --panel-2: #111112;
                --line: #242426;
                --text: #f2f2f3;
                --muted: #9a9aa0;
            }

            .stApp {
                background: var(--jet-bg);
                color: var(--text);
                font-family: 'Space Grotesk', sans-serif;
            }

            [data-testid="stHeader"] {
                background: rgba(5, 5, 5, 0.7);
                border-bottom: 1px solid var(--line);
            }

            [data-testid="stSidebar"] {
                background: #090909;
                border-right: 1px solid var(--line);
            }

            [data-testid="stSidebar"] * {
                color: var(--text);
                font-family: 'Space Grotesk', sans-serif;
            }

            .hero-wrap {
                background: #0a0a0b;
                border: 1px solid var(--line);
                border-radius: 18px;
                padding: 1rem 1.2rem;ds
                margin-bottom: 1rem;
                box-shadow: none;
            }

            .hero-title {
                font-size: 1.35rem;
                font-weight: 700;
                letter-spacing: 0.2px;
                margin: 0;
            }

            .hero-sub {
                margin-top: 0.25rem;
                color: var(--muted);
                font-size: 0.95rem;
            }

            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(3, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 0.65rem 0 0.2rem;
            }

            .kpi-card {
                background: rgba(20, 20, 22, 0.9);
                border: 1px solid var(--line);
                border-radius: 14px;
                padding: 0.65rem 0.75rem;
            }

            .kpi-label {
                font-size: 0.72rem;
                color: var(--muted);
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }

            .kpi-value {
                margin-top: 0.22rem;
                font-size: 0.95rem;
                color: var(--text);
                font-weight: 600;
            }

            .stButton > button,
            .stDownloadButton > button {
                background: #121213;
                color: var(--text);
                border: 1px solid #2a2b31;
                border-radius: 12px;
                font-weight: 600;
            }

            .stButton > button:hover,
            .stDownloadButton > button:hover {
                border-color: #3a3a3f;
                background: #171719;
                color: #ffffff;
            }

            [data-testid="stMetric"] {
                background: rgba(15, 15, 16, 0.95);
                border: 1px solid var(--line);
                border-radius: 12px;
                padding: 0.45rem 0.6rem;
            }

            [data-testid="stFileUploader"] {
                background: rgba(15, 15, 16, 0.95);
                border: 1px dashed #3a3b42;
                border-radius: 12px;
                padding: 0.5rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_browser_location():
    """Inject JavaScript to get browser geolocation."""
    geolocation_js = """
    <script>
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                window.location_data = {
                    lat: position.coords.latitude,
                    lon: position.coords.longitude,
                    accuracy: position.coords.accuracy
                };
                console.log('Location:', window.location_data);
            },
            (error) => {
                console.log('Geolocation error:', error);
                // Fallback: fetch IP-based location
                fetch('https://ipapi.co/json/')
                    .then(r => r.json())
                    .then(data => {
                        window.location_data = {
                            lat: data.latitude || 0,
                            lon: data.longitude || 0,
                            accuracy: 5000,
                            source: 'IP'
                        };
                    });
            }
        );
    }
    </script>
    """
    components.html(geolocation_js, height=0)


def play_siren():
    """Generate wailing siren and play it with autoplay JavaScript workaround."""
    try:
        # Generate siren audio
        siren_audio = generate_siren_audio(duration=3)
        
        # Create unique file to avoid caching
        siren_path = f"/tmp/siren_{int(time.time() * 1000)}.wav"
        wavfile.write(siren_path, 44100, (siren_audio * 32767).astype(np.int16))
        
        # Play with JavaScript auto-start (bypasses browser autoplay restrictions)
        with open(siren_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        # Create HTML5 audio with explicit play trigger
        audio_b64 = __import__('base64').b64encode(audio_bytes).decode()
        
        components.html(
            f"""
            <audio id="sirenaudio" autoplay style="display:none;">
                <source src="data:audio/wav;base64,{audio_b64}" type="audio/wav">
            </audio>
            <script>
                const audio = document.getElementById('sirenaudio');
                audio.volume = 1.0;
                audio.play().catch(e => console.log('Autoplay blocked:', e));
                // Force play after a brief delay
                setTimeout(() => {{ audio.play().catch(e => console.log('Retry failed', e)); }}, 100);
            </script>
            """,
            height=0
        )
        
        logger.info(f"Siren triggered: {siren_path}")
    except Exception as e:
        logger.error(f"Failed to play siren: {e}")

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
apply_jet_black_theme()

# --- Model Status Info ---
weight_path = "models/best_anomaly_model.pth"
last_update = "Never Trained"
if os.path.exists(weight_path):
    mtime = os.path.getmtime(weight_path)
    last_update = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mtime))

status_emoji = "✅" if anomaly_model is not None else "⚠️"
model_state = "Ready" if anomaly_model is not None else "Unavailable"
st.markdown(
    f"""
    <section class="hero-wrap">
        <h1 class="hero-title">Project Jerico Surveillance Command</h1>
        <div class="hero-sub">Professional real-time threat triage with anomaly + object fusion pipelines.</div>
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Model Status</div>
                <div class="kpi-value">{status_emoji} {model_state}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Weights Updated</div>
                <div class="kpi-value">{last_update}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Operating Theme</div>
                    <div class="kpi-value">Jet Black</div>
            </div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)
if anomaly_model is None:
    st.warning("⚠️ Anomaly detection model not loaded. Ensure DATASET/ folder exists and training has completed.")
st.markdown("Upload CCTV footage to run the anomaly detection models and review alerts.")

# Initialize session state for alerts
if "threat_triggered" not in st.session_state:
    st.session_state.threat_triggered = False
if "siren_played" not in st.session_state:
    st.session_state.siren_played = False
if "device_lat" not in st.session_state:
    st.session_state.device_lat = 0
if "device_lon" not in st.session_state:
    st.session_state.device_lon = 0

# Capture browser geolocation
get_browser_location()

# --- Settings Sidebar ---
st.sidebar.header("Detection Settings")

smart_mode = st.sidebar.toggle("🤖 Enable Smart Auto-Config", value=False, help="Intelligently locks settings to the optimal common configuration for all models.")

if not smart_mode:
    conf_threshold      = st.sidebar.slider("Minimum Confidence Score",   min_value=0.0, max_value=1.0, value=0.60, step=0.05)
    anomaly_threshold   = st.sidebar.slider("Anomaly Violence Trigger",    min_value=0.0, max_value=1.0, value=0.75, step=0.05)
    scene_threshold     = st.sidebar.slider("Scene Confidence (CLIP)",     min_value=0.0, max_value=1.0, value=0.25, step=0.05)
    fire_conf_threshold = st.sidebar.slider("Fire Detection Confidence",   min_value=0.0, max_value=1.0, value=0.35, step=0.05)
else:
    conf_threshold      = 0.60   # General object / weapon detection
    anomaly_threshold   = 0.70   # Violence anomaly model
    scene_threshold     = 0.25   # CLIP scene understanding
    fire_conf_threshold = 0.35   # Fire detection model
    st.sidebar.success(
        "**Smart Config Active ✅**\n\n"
        "All models are running on unified optimal thresholds:\n"
        "- 🎯 Object Confidence: **0.60**\n"
        "- 🧨 Anomaly Violence: **0.70**\n"
        "- 🔍 Scene (CLIP): **0.25**\n"
        "- 🔥 Fire Detection: **0.35**"
    )

show_boxes = st.sidebar.toggle("Show Bounding Boxes", value=True)

st.sidebar.markdown("---")
st.sidebar.header("Location")
st.sidebar.info("GPS is auto-detected from your device. Click 'Allow Location' when prompted by browser.")
cam_name = st.sidebar.text_input("Camera Name", value="CCTV-01", help="Friendly name for this camera")

# Get GPS from IP as fallback
try:
    geo_api = requests.get('https://ipapi.co/json/', timeout=2).json()
    default_lat = geo_api.get('latitude', 0)
    default_lon = geo_api.get('longitude', 0)
except:
    default_lat, default_lon = 0, 0

active_geo_location = {"lat": default_lat, "lon": default_lon, "name": cam_name}
st.sidebar.markdown("---")
st.sidebar.header("Alerts")
alert_enabled = st.sidebar.toggle("Enable Notifications", value=True, help="Free push notifications via ntfy.sh")
if "ntfy_topic" not in st.session_state:
    st.session_state.ntfy_topic = f"jerico-alerts-{secrets.token_hex(6)}"

ntfy_server = st.sidebar.text_input("ntfy server", value="https://ntfy.sh")
ntfy_topic = st.sidebar.text_input("Topic name", value=st.session_state.ntfy_topic)
ntfy_token = st.sidebar.text_input("ntfy token", value=os.getenv("NTFY_TOKEN", ""), type="password")
email_to = st.sidebar.text_input("Alert email", value=os.getenv("ALERT_TO_EMAIL", ""))
whatsapp_phone = st.sidebar.text_input("WhatsApp phone (+countrycode)", value=os.getenv("WHATSAPP_PHONE", ""))
whatsapp_apikey = st.sidebar.text_input("WhatsApp API key (CallMeBot)", value=os.getenv("WHATSAPP_APIKEY", ""), type="password")
subscribe_url = f"{ntfy_server.rstrip('/')}/{ntfy_topic}"
st.sidebar.markdown(f"Subscribe URL: {subscribe_url}")
if st.sidebar.button("Send test notification"):
    test_result = send_ntfy_alert(
        ntfy_topic,
        "Test Alert",
        1.0,
        active_geo_location,
        {"source": "dashboard test"},
        ntfy_server,
        ntfy_token,
    )
    st.sidebar.success(test_result)
    if email_to:
        st.sidebar.info(send_email_alert(email_to, "Test Alert", 1.0, active_geo_location))
    if whatsapp_phone and whatsapp_apikey:
        st.sidebar.info(send_whatsapp_alert(whatsapp_phone, whatsapp_apikey, "Test Alert", 1.0, active_geo_location))

st.sidebar.info("Open the Subscribe URL, click 'Subscribe to topic', then allow browser notifications.")
st.sidebar.markdown("---")
st.sidebar.subheader("Model Synchronization")
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
        has_threat  = False
        has_fire    = False
        threat_conf = 0.0
        fire_conf_val = 0.0

        for det in detections:
            cls_id, conf, x1, y1, x2, y2 = det
            if conf < conf_threshold:
                continue

            is_weapon = (cls_id == CLASS_WEAPON)
            is_fire   = (cls_id == CLASS_FIRE)

            if is_weapon:
                has_threat = True
                threat_conf = max(threat_conf, conf)
            if is_fire and conf >= fire_conf_threshold:
                has_fire = True
                fire_conf_val = max(fire_conf_val, conf)

            if is_fire:
                color = (0, 100, 255)   # orange in BGR
                label = f"🔥 FIRE ({conf:.2f})"
            elif is_weapon:
                color = (0, 0, 255)
                label = f"🔫 GUN ({conf:.2f})"
            else:
                color = (0, 255, 0)
                label = f"Person ({conf:.2f})"

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
            st.write(f"**🔥 Fire Confidence:** {fire_conf_val:.2f}" if has_fire else "**🔥 Fire Confidence:** 0.00")

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
                    "location": f"{cam_name}",
                }
                active_geo_location["name"] = cam_name
                dispatch_msg = dispatch_authorities(
                    "Armed Suspect - Gun Detected",
                    threat_conf,
                    active_geo_location,
                    threat_details
                )
                st.warning(dispatch_msg)
                try:
                    play_siren()
                except Exception as e:
                    logger.error(f"Failed to generate siren: {e}")
                if alert_enabled:
                    try:
                        scene_image_path = f"/tmp/threat_scene_{int(time.time())}.jpg"
                        cv2.imwrite(scene_image_path, frame)
                        alert_result = send_ntfy_alert(ntfy_topic, "Gun Detected in Image", threat_conf, active_geo_location, threat_details, ntfy_server, ntfy_token)
                        st.success(alert_result)
                        email_result = send_email_alert(email_to, "Gun Detected in Image", threat_conf, active_geo_location, threat_details, scene_image_path)
                        st.info(email_result)
                        whatsapp_result = send_whatsapp_alert(whatsapp_phone, whatsapp_apikey, "Gun Detected in Image", threat_conf, active_geo_location)
                        st.info(whatsapp_result)
                    except Exception as e:
                        logger.error(f"Failed to send notification: {e}")
                        st.warning(f"⚠️ Notification failed: {e}")

            if has_fire:
                st.error("🔥🔥🔥 CRITICAL FIRE ALERT: FIRE DETECTED IN IMAGE 🔥🔥🔥")
                fire_details = {
                    "fire_detected": "Yes",
                    "fire_confidence": f"{fire_conf_val:.2f}",
                    "objects_detected": len(detections),
                    "location": f"{cam_name}",
                }
                active_geo_location["name"] = cam_name
                fire_dispatch_msg = dispatch_authorities(
                    "Fire Detected",
                    fire_conf_val,
                    active_geo_location,
                    fire_details
                )
                st.warning(fire_dispatch_msg)
                try:
                    play_siren()
                except Exception as e:
                    logger.error(f"Failed to generate siren: {e}")
                if alert_enabled:
                    try:
                        scene_image_path = f"/tmp/fire_scene_{int(time.time())}.jpg"
                        cv2.imwrite(scene_image_path, frame)
                        alert_result = send_ntfy_alert(ntfy_topic, "🔥 Fire Detected in Image", fire_conf_val, active_geo_location, fire_details, ntfy_server, ntfy_token)
                        st.success(alert_result)
                        email_result = send_email_alert(email_to, "🔥 Fire Detected in Image", fire_conf_val, active_geo_location, fire_details, scene_image_path)
                        st.info(email_result)
                        whatsapp_result = send_whatsapp_alert(whatsapp_phone, whatsapp_apikey, "🔥 Fire Detected in Image", fire_conf_val, active_geo_location)
                        st.info(whatsapp_result)
                    except Exception as e:
                        logger.error(f"Failed to send notification: {e}")
                        st.warning(f"⚠️ Fire notification failed: {e}")

            if not has_threat and not has_fire:
                st.success("✅ Secure: No immediate threat detected")

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
        if scene_analyzer is not None and frame_idx > 0 and frame_idx % 15 == 0:  # Sample twice a second, skip frame 0
            try:
                scene_type, scene_prob = scene_analyzer.analyze_frame(frame)

                # Update history and keep it within window
                scene_history.append((scene_type, scene_prob))
                if len(scene_history) > MAX_HISTORY:
                    scene_history.pop(0)

                # Consensus logic: Is there a consistent threat across history?
                threat_keywords = ["suspiciously", "hiding", "fight", "robbery", "weapon", "casing", "panic", "lurking"]
                recent_threats = [p for t, p in scene_history if any(kw in t for kw in threat_keywords) and p > scene_threshold]

                # Universal threat: High-conf consensus across the rolling window
                is_universal_threat = len(recent_threats) >= 2  # At least 2 suspicious checks in a row

                if is_universal_threat:
                    # Trigger exact emergency service responses
                    st.session_state.current_dispatch = dispatch_authorities(scene_type, scene_prob, active_geo_location)
                    st.session_state.univ_override = True  # Keep state for in-between frames
                    st.session_state.safe_counter = 0  # Reset clear timer
                else:
                    # Increment safe counter for every negative CLIP check
                    st.session_state.safe_counter = getattr(st.session_state, 'safe_counter', 0) + 1
            except Exception as e:
                logger.warning(f"Scene analyzer failed on video frame: {e}")
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
        has_fire   = False
        fire_conf_val = 0.0
        
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
            is_fire   = (cls_id == CLASS_FIRE)

            if is_weapon:
                has_threat = True
                threat_conf = max(threat_conf, conf)
            if is_fire and conf >= fire_conf_threshold:
                has_fire = True
                fire_conf_val = max(fire_conf_val, conf)
            
            # Behavioral logic
            is_fast_actor = speeds[i] > max(avg_speed * 1.5, 8.0)
            is_sprinting  = speeds[i] > max(avg_speed * 2.0, 15.0)
            is_suspicious = is_weapon or is_fire or (violence_detected and is_fast_actor) or is_sprinting
            
            if is_fire:
                color = (0, 100, 255)   # orange in BGR
                label = f"🔥 FIRE ({conf:.2f})"
            elif is_weapon:
                color = (0, 0, 255)
                label = f"🔫 GUN ({conf:.2f}) [Spd: {int(speeds[i])}]"
            else:
                color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                label = f"🧍 Person ({conf:.2f}) [Spd: {int(speeds[i])}]"
            
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
        if has_fire:   thrt_str.append('🔥 FIRE (YOLO)')
        if violence_detected: thrt_str.append('🧨 VIOLENT ACTION')
        
        new_status_text = f"**Identified Threats:** {', '.join(thrt_str) if thrt_str else 'None'}\n\n**Max Object Conf:** {threat_conf:.2f}"
        if new_status_text != prev_status_text:
            status_text.write(new_status_text)
            prev_status_text = new_status_text
        
        current_threat_state = has_threat or has_fire or violence_detected
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
                        "fire_detected": has_fire,
                        "violence_score": f"{anomaly_score*100:.1f}%",
                        "object_confidence": f"{threat_conf:.2f}",
                        "fire_confidence": f"{fire_conf_val:.2f}",
                        "frame_number": frame_idx,
                        "total_frames": total_frames
                    }
                    
                    try:
                        play_siren()
                    except Exception as e:
                        logger.error(f"Failed to generate siren: {e}")
                    
                    # Send push notification alert
                    if alert_enabled:
                        try:
                            # Save threat scene frame
                            scene_image_path = f"/tmp/threat_scene_{int(time.time())}.jpg"
                            cv2.imwrite(scene_image_path, frame)
                            
                            # Update active location with current camera name
                            active_geo_location["name"] = cam_name
                            
                            # Send free push notification
                            alert_result = send_ntfy_alert(
                                ntfy_topic,
                                threat_msg,
                                threat_conf if has_threat else anomaly_score,
                                active_geo_location,
                                threat_details,
                                ntfy_server,
                                ntfy_token,
                            )
                            logger.info(alert_result)

                            email_result = send_email_alert(
                                email_to,
                                threat_msg,
                                threat_conf if has_threat else anomaly_score,
                                active_geo_location,
                                threat_details,
                                scene_image_path,
                            )
                            logger.info(email_result)

                            whatsapp_result = send_whatsapp_alert(
                                whatsapp_phone,
                                whatsapp_apikey,
                                threat_msg,
                                threat_conf if has_threat else anomaly_score,
                                active_geo_location,
                            )
                            logger.info(whatsapp_result)
                        except Exception as e:
                            logger.error(f"Failed to send notification: {e}")
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

