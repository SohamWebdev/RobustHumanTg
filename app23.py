import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from datetime import datetime
from ultralytics import YOLO
import time
import matplotlib.pyplot as plt

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # You can replace this with your custom model if needed

# Streamlit page setup
st.set_page_config(page_title="Abnormal Activity Detection Dashboard", layout="wide")
st.title("📹 Abnormal Activity Detection Dashboard")

# File uploader for video
video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
abnormal_classes = st.multiselect("Select abnormal classes", options=["person", "car", "truck"], default=["person"])

# Alert log and heatmap initialization
alert_log = []
heatmap = None

# Ensure directories exist
os.makedirs("abnormal_screenshots", exist_ok=True)
os.makedirs("heatmaps", exist_ok=True)

# Process video if uploaded
if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    heatmap = np.zeros((height, width), dtype=np.float32)

    # Streamlit placeholders
    frame_placeholder = st.empty()
    alert_placeholder = st.empty()
    heatmap_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        boxes = results.boxes
        classes = results.names if results.names else model.names

        abnormal_detected = False

        for box in boxes:
            class_id = int(box.cls[0])
            class_label = classes[class_id]
            conf = float(box.conf[0])

            if class_label in abnormal_classes:
                abnormal_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Update heatmap
                heatmap[y1:y2, x1:x2] += 1

                # Log alert
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                alert_log.append(f"[{timestamp}] Abnormal activity: {class_label} detected")

                # Save abnormal screenshot
                screenshot_filename = os.path.join("abnormal_screenshots", f"abnormal_{class_label}_{timestamp.replace(':', '-')}.jpg")
                if frame is not None and frame.size > 0:
                    cv2.imwrite(screenshot_filename, frame)

        # Display the processed frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", caption="Processed Frame")

        # Display alert log
        alert_placeholder.write("### 🚨 Alert Log")
        for alert in alert_log[-5:][::-1]:  # Show last 5 alerts
            alert_placeholder.write(alert)

        # Display heatmap
        if heatmap is not None and np.max(heatmap) > 0:
            heatmap_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_uint8 = heatmap_norm.astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            heatmap_bgr = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

            # Save heatmap
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            heatmap_filename = os.path.join("heatmaps", f"heatmap_{timestamp}.jpg")
            if heatmap_colored is not None and heatmap_colored.size > 0:
                cv2.imwrite(heatmap_filename, heatmap_colored)

            heatmap_placeholder.image(heatmap_bgr, channels="RGB", caption="Abnormal Activity Heatmap")

        time.sleep(0.03)

    cap.release()
    st.success("✅ Video processing completed.")
