import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ----------------- SIDEBAR -----------------
st.sidebar.title("Settings")
st.sidebar.markdown("Upload an image and see objects detected in real-time using YOLOv8.")

# ----------------- MAIN APP -----------------
st.title("🖼️ Real-Time Object Detection")
st.markdown(
    """
    Upload any image and the app will detect objects using **YOLOv8**.
    Bounding boxes and labels will be displayed on the image.
    """
)

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    frame = np.array(image)

    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")  # small, fast model
    results = model(frame)[0]

    # Draw bounding boxes
    for r in results.boxes:
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = r.conf[0]
        cls = int(r.cls[0])
        label = f"{model.names[cls]} {conf:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    st.image(frame, caption="Detected Objects", channels="BGR")

# Footer
st.markdown("---")
st.markdown("Created by **sherinovia19** | Powered by YOLOv8 & Streamlit")
