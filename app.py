import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import pandas as pd
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS / BACKGROUND -----------------
st.markdown(
    """
    <style>
    body {
        background: linear-gradient(to right, #e0f7fa, #ffffff);
        font-family: 'Arial', sans-serif;
    }
    .stButton>button {
        background-color: #4A90E2;
        color: white;
        border-radius:10px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------- HEADER -----------------
st.markdown(
    """
    <div style='padding:20px; border-radius:15px; text-align:center; background-color:#4A90E2; color:white;'>
        <h1>🖼️ Real-Time Object Detection</h1>
        <p>Upload an image and detect objects using YOLOv8</p>
    </div>
    """, unsafe_allow_html=True
)

st.write("")

# ----------------- SIDEBAR -----------------
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    frame = np.array(image)

    # Load model
    model = YOLO("yolov8n.pt")
    results = model(frame)[0]

    # Filter by confidence
    filtered_boxes = [r for r in results.boxes if r.conf[0] >= confidence_threshold]

    # Get unique classes
    classes_detected = [model.names[int(r.cls[0])] for r in filtered_boxes]
    unique_classes = list(set(classes_detected))
    selected_classes = st.sidebar.multiselect("Select Classes to Display", unique_classes, default=unique_classes)

    # Bounding box colors
    class_colors = {cls: tuple(np.random.randint(0,255,3).tolist()) for cls in unique_classes}

    # Draw bounding boxes
    for r in filtered_boxes:
        cls_name = model.names[int(r.cls[0])]
        if cls_name not in selected_classes:
            continue
        x1, y1, x2, y2 = map(int, r.xyxy[0])
        conf = r.conf[0]
        label = f"{cls_name} {conf:.2f}"
        color = class_colors[cls_name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Display image
    st.image(frame, caption="Detected Objects", channels="BGR")

    # ----------------- METRICS -----------------
    cols = st.columns(3)
    cols[0].metric("Total Objects", len(filtered_boxes))
    cols[1].metric("Unique Classes", len(unique_classes))
    avg_conf = np.mean([r.conf[0] for r in filtered_boxes]) if filtered_boxes else 0
    cols[2].metric("Average Confidence", f"{avg_conf:.2f}")

    # ----------------- BAR CHART -----------------
    if filtered_boxes:
        class_counts = pd.Series([model.names[int(r.cls[0])] for r in filtered_boxes]).value_counts()
        st.bar_chart(class_counts)

    # ----------------- DOWNLOAD BUTTON -----------------
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        cv2.imwrite(tmp_file.name, frame)
        st.download_button(
            label="Download Processed Image",
            data=open(tmp_file.name, "rb").read(),
            file_name="detected_image.png",
            mime="image/png"
        )

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Created by <b>sherinovia19</b> | Powered by YOLOv8 & Streamlit</p>",
    unsafe_allow_html=True
)

