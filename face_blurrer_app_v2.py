import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import io
import tempfile
import os
import time

def process_img(img, face_detection):
    h_img, w_img, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    out = face_detection.process(img_rgb)
    if out.detections is not None:
        for detection in out.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(w_img * bbox.xmin)
            y1 = int(h_img * bbox.ymin)
            h = int(h_img * bbox.height)
            w = int(w_img * bbox.width)
            img[y1:y1+h, x1:x1+w] = cv2.blur(img[y1:y1+h, x1:x1+w], (50,50))
            img = cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 0, 255), 3)
    return img

st.set_page_config(page_title="BlurVision AI", page_icon="😎", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
            color: white;
            text-align: center;
            padding: 50px 0;
        }
        h1 {
            color: #f4f4f4;
            font-size: 3rem;
            font-weight: bold;
        }
        .stButton > button {
            background-color: #ff7f50;
            color: white;
            font-size: 20px;
            border-radius: 12px;
            padding: 12px 24px;
            transition: 0.3s;
        }
        .stButton > button:hover {
            background-color: #e65c23;
            transform: scale(1.1);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "page" not in st.session_state:
    st.session_state.page = "home"

def go_to_app():
    st.session_state.page = "app"

def go_home():
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.markdown("""
        <div class="main">
            <h1>Welcome to BlurVision AI</h1>
            <p style="font-size: 1.5rem; max-width: 800px; margin: auto;">
                The ultimate tool for real-time face blurring on images, videos, and webcams.
                Protect privacy effortlessly with AI-powered face detection.
            </p>
            <br>
        </div>
    """, unsafe_allow_html=True)
    st.button("🚀 Get Started", on_click=go_to_app)

elif st.session_state.page == "app":
    st.button("⬅ Back to Home", on_click=go_home)
    st.title("📸 Face Blur Application")
    mode = st.selectbox("Select Mode", ["image", "video", "webcam"], index=0)

    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        if mode == "image":
            file = st.file_uploader("Upload an Image (JPG only)", type=["jpg"])
            if file:
                image = Image.open(file)
                img_array = np.array(image)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                processed_img = process_img(img_array, face_detection)
                st.image(processed_img, caption="Processed Image", channels="BGR")
                
                _, img_encoded = cv2.imencode(".jpg", processed_img)
                img_bytes = img_encoded.tobytes()
                st.download_button(label="📥 Download Processed Image", data=img_bytes, file_name="processed_image.jpg", mime="image/jpeg")
        
        elif mode == "video":
            file = st.file_uploader("Upload a Video (MP4 only)", type=["mp4"])
            if file:
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(file.read())
                cap = cv2.VideoCapture(tfile.name)
                
                fourcc = cv2.VideoWriter_fourcc(*'MP4V')
                output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                out = cv2.VideoWriter(output_path, fourcc, 25.0, (int(cap.get(3)), int(cap.get(4))))
                
                progress_bar = st.progress(0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                processed_frames = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    processed_frame = process_img(frame, face_detection)
                    out.write(processed_frame)
                    processed_frames += 1
                    progress_bar.progress(processed_frames / frame_count)
                
                cap.release()
                out.release()
                
                with open(output_path, "rb") as f:
                    st.download_button(label="📥 Download Processed Video", data=f, file_name="processed_video.mp4", mime="video/mp4")

        elif mode == "webcam":
            st.write("🎥 Live Webcam Processing")
            cap = cv2.VideoCapture(0)
            frame_placeholder = st.empty()
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'MP4V')
            out = cv2.VideoWriter(output_path, fourcc, 25.0, (640, 480))
            
            if st.button("▶ Start 10s Recording", key="start_recording"):
                start_time = time.time()
                progress_bar = st.progress(0)
                while cap.isOpened():
                    ret, frame = cap.read()
                    elapsed_time = int(time.time() - start_time)
                    remaining_time = max(0, 10 - elapsed_time)
                    progress_bar.progress((10 - remaining_time) / 10)
                    
                    if not ret or elapsed_time >= 10:
                        break
                    
                    processed_frame = process_img(frame, face_detection)
                    frame_placeholder.image(processed_frame, channels="BGR")
                    out.write(processed_frame)
                
                cap.release()
                out.release()
                st.success("✅ Recording complete!")
                
                with open(output_path, "rb") as f:
                    st.download_button(label="📥 Download Recorded Video", data=f, file_name="recorded_video.mp4", mime="video/mp4")
