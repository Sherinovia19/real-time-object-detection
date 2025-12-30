import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import os

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Real-Time Object Detection",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- SIDEBAR -----------------
st.sidebar.title("Settings")
dark_mode = st.sidebar.checkbox("Dark Mode", value=False)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
multi_image_mode = st.sidebar.checkbox("Upload Multiple Images", value=False)

# Detect Streamlit Cloud
on_cloud = "STREAMLIT_SERVER_PORT" in os.environ

# Webcam toggle only locally
if not on_cloud:
    webcam_mode = st.sidebar.checkbox("Use Webcam / Live Mode", value=False)
else:
    webcam_mode = False
    st.sidebar.info("Webcam mode disabled on Streamlit Cloud")

# ----------------- DARK/LIGHT THEME -----------------
bg_color = "#121212" if dark_mode else "#f5f5f5"
text_color = "#ffffff" if dark_mode else "#000000"

st.markdown(
    f"""
    <style>
    body {{
        background-color: {bg_color};
        color: {text_color};
        font-family: 'Arial', sans-serif;
    }}
    .stButton>button {{
        background-color: #4A90E2;
        color: white;
        border-radius:10px;
    }}
    </style>
    """, unsafe_allow_html=True
)

# ----------------- HEADER -----------------
st.markdown(
    f"""
    <div style='padding:20px; border-radius:15px; text-align:center; background-color:#4A90E2; color:white;'>
        <h1>Real-Time Object Detection</h1>
        <p>Upload images or use webcam (local only) to detect objects using YOLOv8</p>
    </div>
    """, unsafe_allow_html=True
)
st.write("")

# ----------------- LOAD MODEL -----------------
# Direct load to avoid cache issues on Cloud
model = YOLO("yolov8n.pt")

# ----------------- IMAGE / WEBCAM INPUT -----------------
uploaded_files = []

if webcam_mode:
    frame = st.camera_input("Capture from Webcam")
    if frame:
        uploaded_files = [frame]
elif multi_image_mode:
    uploaded_files = st.file_uploader("Upload Images", type=["jpg","jpeg","png"], accept_multiple_files=True)
else:
    file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
    if file:
        uploaded_files = [file]

# ----------------- PROCESS EACH IMAGE -----------------
for uploaded_file in uploaded_files:
    image = Image.open(uploaded_file)
    if image.mode != "RGB":
        image = image.convert("RGB")
    frame = np.array(image)

    # Run YOLOv8
    with st.spinner("Processing..."):
        results = model(frame)[0]

    # Filter by confidence
    filtered_boxes = [r for r in results.boxes if r.conf[0] >= confidence_threshold]

    # Unique classes
    classes_detected = [model.names[int(r.cls[0])] for r in filtered_boxes]
    unique_classes = list(set(classes_detected))
    selected_classes = st.sidebar.multiselect(f"Select Classes ({uploaded_file.name})", unique_classes, default=unique_classes)

    # Bounding box colors
    class_colors = {cls: tuple(np.random.randint(0,255,3).tolist()) for cls in unique_classes}

    # Draw boxes
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

    st.image(frame, caption=f"Detected Objects - {uploaded_file.name}", channels="BGR")

    # Metrics
    cols = st.columns(3)
    cols[0].metric("Total Objects", len(filtered_boxes))
    cols[1].metric("Unique Classes", len(unique_classes))
    avg_conf = np.mean([r.conf[0] for r in filtered_boxes]) if filtered_boxes else 0
    cols[2].metric("Average Confidence", f"{avg_conf:.2f}")

    # Bar chart
    if filtered_boxes:
        class_counts = pd.Series([model.names[int(r.cls[0])] for r in filtered_boxes]).value_counts()
        st.bar_chart(class_counts)

    # Pie chart
    if filtered_boxes:
        fig, ax = plt.subplots()
        class_counts.plot.pie(autopct='%1.1f%%', ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    # Download image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        cv2.imwrite(tmp_file.name, frame)
        st.download_button(
            label="Download Processed Image",
            data=open(tmp_file.name, "rb").read(),
            file_name=f"detected_{uploaded_file.name}",
            mime="image/png"
        )

    # Download CSV
    if filtered_boxes:
        data = [{"class": model.names[int(r.cls[0])], "confidence": float(r.conf[0]),
                 "x1": int(r.xyxy[0][0]), "y1": int(r.xyxy[0][1]),
                 "x2": int(r.xyxy[0][2]), "y2": int(r.xyxy[0][3])} for r in filtered_boxes]
        df = pd.DataFrame(data)
        st.download_button(
            label="Download Detection CSV",
            data=df.to_csv(index=False),
            file_name=f"detections_{uploaded_file.name}.csv",
            mime="text/csv"
        )

    # Download JSON
    if filtered_boxes:
        st.download_button(
            label="Download Detection JSON",
            data=df.to_json(orient="records"),
            file_name=f"detections_{uploaded_file.name}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown(
    f"<p style='text-align:center; color:{text_color};'>Created by <b>sherinovia19</b> | Powered by YOLOv8 & Streamlit</p>",
    unsafe_allow_html=True
)
