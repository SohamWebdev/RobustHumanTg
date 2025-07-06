import streamlit as st
import cv2
import tempfile
import os
import time
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Page config
st.set_page_config(page_title="Abnormal Activity Detection Dashboard", layout="wide")

# Directories
os.makedirs("abnormal_screenshots", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# CSS for styling
st.markdown("""
    <style>
    .title { font-size:32px; font-weight:bold; color:#ffffff; padding:10px; background:#6c63ff; border-radius:10px; text-align:center; }
    .section { font-size:22px; font-weight:bold; margin-top:20px; }
    .card { background-color:#f0f2f6; padding:15px; border-radius:15px; box-shadow:0px 4px 6px rgba(0,0,0,0.1); margin-bottom:20px; }
    .alert { background:#ff4d4d; color:white; padding:10px; border-radius:10px; text-align:center; font-size:18px; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">📹 Abnormal Activity Detection Dashboard</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

# Sidebar options
show_heatmap = st.sidebar.checkbox("Show Crowd Heatmap", value=True)
show_screenshots = st.sidebar.checkbox("Show Abnormal Screenshots", value=True)

# Load YOLO model
model = YOLO("yolov8n.pt")

# Define abnormal behavior
def is_abnormal(detected_classes):
    count = Counter(detected_classes)
    return count.get('person', 0) > 10 or any(cls in count for cls in ['car', 'truck', 'bus'])

# Save screenshot of abnormal frame
def save_screenshot(frame, label="abnormal"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"abnormal_screenshots/{label}_{timestamp}.jpg"
    cv2.imwrite(filename, frame)

# Generate heatmap image
def save_heatmap(locations):
    if not locations:
        return
    xs, ys = zip(*locations)
    heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(64, 48))
    fig, ax = plt.subplots()
    sns.heatmap(heatmap.T, cmap='inferno', cbar=False, ax=ax)
    ax.invert_yaxis()
    plt.axis("off")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    heatmap_path = f"heatmaps/heatmap_{timestamp}.png"
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Video processing section
if uploaded_file is not None:
    # Delete previous screenshots
    for file in glob.glob("abnormal_screenshots/*.jpg"):
        os.remove(file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()
    st.sidebar.success("Video uploaded successfully!")

    frame_count = 0
    detected_locations = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, verbose=False)[0]

        classes = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]
            classes.append(cls_name)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            detected_locations.append((cx, cy))

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, cls_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        abnormal = is_abnormal(classes)
        if abnormal:
            save_screenshot(frame)
            with open("logs/alerts.log", "a") as log:
                log.write(f"{datetime.now()}: Abnormal activity detected\n")
            cv2.putText(frame, "ABNORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        frame = cv2.resize(frame, (720, 480))
        stframe.image(frame, channels="BGR", use_column_width=True)

    cap.release()

    if show_heatmap:
        save_heatmap(detected_locations)
        st.markdown("<div class='card'><b>Crowd Heatmap</b></div>", unsafe_allow_html=True)
        latest_heatmap = sorted(glob.glob("heatmaps/*.png"))[-1]
        st.image(latest_heatmap, use_column_width=True)

    if show_screenshots:
        screenshot_files = sorted(glob.glob("abnormal_screenshots/*.jpg"))
        if screenshot_files:
            st.markdown("<div class='card'><b>Abnormal Activity Screenshots</b></div>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, file in enumerate(screenshot_files):
                img = Image.open(file)
                with cols[i % 3]:
                    st.image(img, caption=os.path.basename(file), use_column_width=True)
        else:
            st.info("No abnormal screenshots available.")

