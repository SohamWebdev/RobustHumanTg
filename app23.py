import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
import time
import streamlit.components.v1 as components

# --- Animated and Modern CSS ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Roboto:wght@400;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Roboto', 'Orbitron', sans-serif;
        background: linear-gradient(120deg, #232526 0%, #414345 100%);
        color: #f8f8f8;
    }
    .main {
        background: rgba(30, 30, 40, 0.95) !important;
        border-radius: 18px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 2rem;
    }
    .stButton > button {
        font-family: 'Orbitron', sans-serif;
        font-weight: bold;
        background: linear-gradient(90deg, #00c6ff 0%, #0072ff 100%);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 12px 28px;
        font-size: 1.1rem;
        margin: 0.5rem 0.5rem 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        transition: all 0.3s cubic-bezier(.25,.8,.25,1);
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%);
        color: #fff;
        transform: scale(1.07);
        box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .status-badge {
        display: inline-block;
        padding: 0.5em 1.2em;
        border-radius: 30px;
        font-weight: bold;
        font-size: 1.1em;
        margin-left: 1em;
        animation: pulse 1.2s infinite;
    }
    .status-normal {
        background: linear-gradient(90deg, #43e97b 0%, #38f9d7 100%);
        color: #222;
        box-shadow: 0 0 8px #43e97b80;
    }
    .status-abnormal {
        background: linear-gradient(90deg, #ff512f 0%, #dd2476 100%);
        color: #fff;
        box-shadow: 0 0 12px #ff512f80;
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(255,81,47,0.7); }
        70% { box-shadow: 0 0 0 12px rgba(255,81,47,0); }
        100% { box-shadow: 0 0 0 0 rgba(255,81,47,0); }
    }
    .fadein {
        animation: fadeIn 1s;
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .card {
        background: rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.2em 1.5em;
        margin-bottom: 1.5em;
        box-shadow: 0 2px 8px rgba(0,0,0,0.10);
    }
    </style>
""", unsafe_allow_html=True)

# --- Animated Header ---
st.markdown("""
    <div style='text-align:center; margin-bottom: 0.5em;'>
        <span style='
            font-family: Orbitron, sans-serif;
            font-size: 3rem;
            color: #00c6ff;
            letter-spacing: 2px;
            text-shadow: 0 2px 8px #0072ff80;
            animation: glow 2s infinite alternate;
        '>
            🚨 Abnormal Activity Detection Dashboard 🚨
        </span>
    </div>
    <style>
    @keyframes glow {
        from { text-shadow: 0 2px 8px #0072ff80; }
        to { text-shadow: 0 4px 24px #00c6ff; }
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div style='text-align:center; font-size:1.2em; margin-bottom:1.5em;'>Upload a video and click <b>Start Detection</b> to analyze abnormal human activities using YOLO.</div>", unsafe_allow_html=True)

# --- Layout with columns ---
col1, col2 = st.columns([2, 1])

with col1:
    uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
    start_button = st.button("▶️ Start Detection")
    show_heatmap = st.button("🔥 Show Heatmap")
    show_log = st.button("📄 Show Alert Log")
    show_screenshots = st.button("🖼️ Show Abnormal Screenshots")

frame_placeholder = st.empty()
status_placeholder = st.empty()  # Add this for the status badge
alert_placeholder = st.empty()

# --- Video path logic ---
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name
else:
    video_path = "sample.mp4"

LOG_FILE = "abnormal_activity_log.txt"
SOUND_FILE = "https://www.soundjay.com/phone/sounds/telephone-ring-03b.mp3"

def log_alert(activity):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"{timestamp} - ALERT: Abnormal {activity} detected\n")

def play_alert_sound():
    try:
        components.html(f"""
        <script>
            var audio = new Audio("{SOUND_FILE}");
            audio.play();
        </script>
    """, height=0)
    except Exception as e:
        st.warning("Sound error: " + str(e))

def analyze_crowd_behavior(frame):
    fgMask = backSub.apply(frame)
    kernel = np.ones((5, 5), np.uint8)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)
    fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)
    motion_pixels = cv2.countNonZero(fgMask)
    total_pixels = frame.shape[0] * frame.shape[1]
    crowd_density = motion_pixels / total_pixels
    return crowd_density, fgMask

def is_activity_abnormal(activity, crowd_density):
    always_abnormal = {"fighting", "robbery", "armed", "lying_down"}
    activity_thresholds = {
        "walking": 0.25, "standing": 0.2, "running": 0.15, "fighting": 0.1,
        "jumping": 0.2, "robbery": 0.05, "sitting": 0.3, "armed": 0.05, "lying_down": 0.2
    }
    return activity in always_abnormal or crowd_density > activity_thresholds.get(activity, 0.3)

if 'heatmap' not in st.session_state:
    st.session_state['heatmap'] = None

# --- Load model ---
model = YOLO("best.pt")
class_names = model.names

# --- Detection logic ---
if start_button:
    cap = cv2.VideoCapture(video_path)
    count = 0
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
    heatmap = None
    last_alert_time = 0
    ALERT_COOLDOWN = 2  # seconds
    no_motion_alert_shown = False  # Add this line

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % 3 != 0:
            continue

        frame = cv2.resize(frame, (1020, 500))
        crowd_density, fg_mask = analyze_crowd_behavior(frame)

        # --- Heatmap accumulation ---
        if heatmap is None:
            heatmap = np.zeros_like(fg_mask, dtype=np.float32)
        heatmap += (fg_mask > 0).astype(np.float32)

        results = model.predict(frame)
        abnormal_activities = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls)
                class_label = class_names[cls_id]
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                is_abnormal = is_activity_abnormal(class_label, crowd_density)
                color = (0, 0, 255) if is_abnormal else (0, 255, 0)
                label = f"{class_label} ({'ABNORMAL' if is_abnormal else 'Normal'})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                if is_abnormal:
                    abnormal_activities.append(class_label)
                    # Save abnormal screenshot
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_filename = os.path.join("abnormal_screenshots", f"abnormal_{class_label}_{timestamp}.jpg")
                    cv2.imwrite(screenshot_filename, frame)

                    # Save heatmap for this abnormal event
                    if heatmap is not None and np.max(heatmap) > 0:
                        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                        heatmap_uint8 = heatmap_norm.astype(np.uint8)
                        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
                        heatmap_filename = os.path.join("heatmaps", f"heatmap_{class_label}_{timestamp}.jpg")
                        cv2.imwrite(heatmap_filename, heatmap_colored)
                        # Optionally reset heatmap for next event
                        heatmap = np.zeros_like(heatmap)
                    else:
                        if not no_motion_alert_shown:
                            st.info("No motion detected for heatmap.")
                            no_motion_alert_shown = True

        overall_status = "ABNORMAL" if abnormal_activities else "NORMAL"
        status_class = "status-abnormal" if overall_status == "ABNORMAL" else "status-normal"
        status_html = f"<span class='status-badge {status_class}'>{overall_status}</span>"

        cv2.putText(frame, f"Density: {crowd_density:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Status: {overall_status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255) if overall_status=="ABNORMAL" else (0,255,0), 2)

        # --- Streamlit image display with fade-in ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Show status badge in its own placeholder
        status_placeholder.markdown(
            f"<div class='card fadein'>{status_html}</div>", unsafe_allow_html=True
        )

        # Show video frame in its own placeholder
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Only play sound and show alert if a new abnormal activity is detected (with cooldown)
        current_time = time.time()
        if abnormal_activities and (current_time - last_alert_time > ALERT_COOLDOWN):
            alert_placeholder.error(f"🚨 ALERT: Abnormal activity detected: {', '.join(abnormal_activities)}")
            for activity in abnormal_activities:
                log_alert(activity)
            play_alert_sound()
            last_alert_time = current_time

        # Save heatmap to session state
        if heatmap is not None:
            st.session_state['heatmap'] = heatmap
            # Save heatmap as image file in folder
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_norm.astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_filename = os.path.join("heatmaps", "crowd_heatmap.jpg")
            cv2.imwrite(heatmap_filename, heatmap_colored)  # This saves the heatmap in your project directory

        time.sleep(0.03)  # ~30 FPS

    cap.release()
    st.balloons()
    st.success("✅ Detection completed!")

    # --- Save final heatmap after detection completes ---
    if heatmap is not None and np.max(heatmap) > 0:
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        # Save to folder
        os.makedirs("heatmaps", exist_ok=True)
        heatmap_filename = os.path.join("heatmaps", "crowd_heatmap.jpg")
        cv2.imwrite(heatmap_filename, heatmap_colored)
        # Store in session state for Streamlit display
        st.session_state['heatmap'] = heatmap

# --- Show heatmap if button pressed ---
if show_heatmap:
    heatmap = st.session_state.get('heatmap', None)
    if heatmap is not None and np.max(heatmap) > 0:
        heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_uint8 = heatmap_norm.astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        st.markdown("<div class='card'><b>Heatmap of Crowd Movement</b></div>", unsafe_allow_html=True)
        st.image(heatmap_colored, caption="Crowd Heatmap", channels="BGR", use_column_width=True)
    else:
        st.info("Run detection first to generate a heatmap.")

# --- Show logs button ---
if show_log:
    st.markdown("<div class='card'><b>Alert Log</b></div>", unsafe_allow_html=True)
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            st.text_area("Alert Log", f.read(), height=200)
    else:
        st.info("No alerts logged yet.")

# --- Show abnormal screenshots ---
if show_screenshots:
    import glob
    import PIL.Image

    screenshot_files = sorted(glob.glob("abnormal_screenshots/*.jpg"))

    if screenshot_files:
        st.markdown("<div class='card'><b>Abnormal Activity Screenshots</b></div>", unsafe_allow_html=True)
        for img_path in screenshot_files:
            with PIL.Image.open(img_path) as img:
                st.image(img, caption=os.path.basename(img_path), use_column_width=True)
        st.info("All detected screenshots are shown above  will be deleted automatically.")
    else:
        st.info("No abnormal screenshots found yet.")

import os

# Create folders if they don't exist
os.makedirs("abnormal_screenshots", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)