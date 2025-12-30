import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

st.title("Real-Time Object Detection (Upload Image)")

st.write("Upload an image, and the app will detect objects using YOLOv8.")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    frame = np.array(image)

    # Load YOLOv8 small model
    model = YOLO("yolov8n.pt")
    results = model(frame)[0]

    # Draw bounding boxes
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = r.conf[0]
        cls = int(r.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    st.image(frame, caption="Detected Objects", channels="BGR")
