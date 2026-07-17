"""Streamlit interface for YOLO detection and SORT tracking."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from object_detection.app import draw_premium_bbox, get_color
from object_detection.tracker import Sort


st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading AI model…")
def load_model(model_name: str) -> YOLO:
    """Load a model once and reuse it between Streamlit reruns."""
    return YOLO(model_name)


def extract_detections(results: Iterable, allowed_classes: Optional[List[int]]) -> np.ndarray:
    """Convert Ultralytics results into SORT's [xyxy, confidence, class] format."""
    detections: list[list[float]] = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            if allowed_classes is None or class_id in allowed_classes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, float(box.conf[0].item()), class_id])
    return np.asarray(detections, dtype=np.float32) if detections else np.empty((0, 6), dtype=np.float32)


def class_name(names: Union[Dict[int, str], List[str]], class_id: int) -> str:
    """Support both mapping and list representations used by YOLO models."""
    if isinstance(names, dict):
        return str(names.get(class_id, f"Class {class_id}"))
    return str(names[class_id]) if 0 <= class_id < len(names) else f"Class {class_id}"


def annotate_frame(frame: np.ndarray, tracks: np.ndarray, names: Union[Dict[int, str], List[str]]) -> np.ndarray:
    """Draw track IDs and object names on a copy of a BGR frame."""
    annotated = frame.copy()
    for x1, y1, x2, y2, track_id, class_id in tracks:
        track_id, class_id = int(track_id), int(class_id)
        draw_premium_bbox(
            annotated,
            (x1, y1, x2, y2),
            f"ID {track_id} | {class_name(names, class_id)}",
            get_color(track_id),
        )
    return annotated


def detect_image(model: YOLO, image: np.ndarray, confidence: float, classes: Optional[List[int]]) -> tuple[np.ndarray, int]:
    """Recognize objects in one image and return an annotated RGB image."""
    results = model(image, conf=confidence, verbose=False)
    detections = extract_detections(results, classes)
    tracker = Sort(min_hits=1)
    tracks = tracker.update(detections)
    return cv2.cvtColor(annotate_frame(image, tracks, model.names), cv2.COLOR_BGR2RGB), len(tracks)


def render_video(
    model: YOLO,
    source: Path,
    confidence: float,
    classes: Optional[List[int]],
) -> tuple[bytes, int, float]:
    """Track uploaded video frames and return a downloadable MP4."""
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The uploaded video has invalid dimensions.")

    output_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    output_path = Path(output_file.name)
    output_file.close()
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        output_path.unlink(missing_ok=True)
        raise RuntimeError("Could not create the processed video.")

    tracker = Sort(min_hits=2)
    progress = st.progress(0, text="Starting video analysis…")
    preview = st.empty()
    metrics = st.empty()
    frame_number = 0
    started = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            results = model(frame, conf=confidence, verbose=False)
            tracks = tracker.update(extract_detections(results, classes))
            annotated = annotate_frame(frame, tracks, model.names)
            writer.write(annotated)

            frame_number += 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            if frame_number == 1 or frame_number % 5 == 0:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                metrics.metric("Processing speed", f"{frame_number / elapsed:.1f} FPS")
                if total_frames > 0:
                    progress.progress(min(frame_number / total_frames, 1.0), text=f"Analyzing frame {frame_number:,} of {total_frames:,}")
    finally:
        capture.release()
        writer.release()

    progress.empty()
    with output_path.open("rb") as video_file:
        video_bytes = video_file.read()
    output_path.unlink(missing_ok=True)
    return video_bytes, frame_number, time.perf_counter() - started


def main() -> None:
    st.markdown(
        """<style>
        .block-container { max-width: 1200px; padding-top: 2.5rem; }
        .hero { padding: 1.75rem 2rem; border-radius: 18px; background: linear-gradient(120deg, #071a33, #0f766e); color: white; }
        .hero h1 { margin: 0; font-size: 2.2rem; } .hero p { margin-bottom: 0; opacity: .9; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hero'><h1>🎯 VisionTrack AI</h1><p>Recognize, label, and track objects from video or your webcam.</p></div>", unsafe_allow_html=True)
    st.write("")

    with st.sidebar:
        st.header("Detection settings")
        model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], help="Larger models are more accurate but slower.")
        confidence = st.slider("Minimum confidence", 0.1, 0.9, 0.35, 0.05)
        class_filter = st.text_input("Class IDs (optional)", placeholder="Example: 0, 2 for person and car")
        st.caption("COCO examples: 0 = person, 2 = car, 16 = dog.")

    try:
        allowed_classes = [int(value.strip()) for value in class_filter.split(",") if value.strip()] or None
    except ValueError:
        st.sidebar.error("Class IDs must be comma-separated integers.")
        st.stop()

    mode = st.radio("Choose an input", ["Upload video", "Webcam photo"], horizontal=True)
    try:
        model = load_model(model_name)
    except Exception as error:
        st.error(f"Could not load {model_name}: {error}")
        st.stop()

    if mode == "Webcam photo":
        photo = st.camera_input("Capture an image")
        if photo is not None:
            image = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Recognizing objects…"):
                annotated, count = detect_image(model, image, confidence, allowed_classes)
            st.image(annotated, caption=f"Detected {count} tracked object(s)", use_container_width=True)
        return

    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is None:
        st.info("Upload a video to start detection and tracking.")
        return

    st.video(uploaded_video)
    if st.button("Analyze video", type="primary", use_container_width=True):
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        input_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        input_path = Path(input_file.name)
        try:
            input_file.write(uploaded_video.getvalue())
            input_file.close()
            video_bytes, frame_count, elapsed = render_video(model, input_path, confidence, allowed_classes)
        except Exception as error:
            st.error(f"Video processing failed: {error}")
            return
        finally:
            input_file.close()
            input_path.unlink(missing_ok=True)

        st.success(f"Processed {frame_count:,} frame(s) in {elapsed:.1f} seconds.")
        st.video(video_bytes)
        st.download_button("Download tracked video", video_bytes, "visiontrack-output.mp4", "video/mp4", use_container_width=True)


if __name__ == "__main__":
    main()
