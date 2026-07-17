"""Streamlit interface for YOLO detection and SORT tracking."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from object_detection.app import draw_premium_bbox, get_color, ccw, intersect, get_side
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


def summarize_tracks(tracks: np.ndarray, names: Union[Dict[int, str], List[str]]) -> Counter:
    """Count visible tracks by their readable object label."""
    return Counter(class_name(names, int(track[5])) for track in tracks)


def detect_image(
    model: YOLO, image: np.ndarray, confidence: float, classes: Optional[List[int]]
) -> Tuple[np.ndarray, np.ndarray]:
    """Recognize objects in one image and return an annotated RGB image."""
    results = model(image, conf=confidence, verbose=False)
    detections = extract_detections(results, classes)
    tracker = Sort(min_hits=1)
    tracks = tracker.update(detections)
    return cv2.cvtColor(annotate_frame(image, tracks, model.names), cv2.COLOR_BGR2RGB), tracks


def render_video(
    model: YOLO,
    source: Path,
    confidence: float,
    classes: Optional[List[int]],
    line_coords: tuple[float, float, float, float],
) -> tuple[bytes, int, float, Counter, Counter]:
    """Track uploaded video frames and return a downloadable MP4, along with line crossing statistics."""
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

    # Calculate line coordinate bounds based on actual frame size
    L1 = (int(line_coords[0] * width), int(line_coords[1] * height))
    L2 = (int(line_coords[2] * width), int(line_coords[3] * height))

    tracker = Sort(min_hits=2)
    
    # Tracking variables
    track_centroids = {}
    track_sides = {}
    counted_ids = set()
    in_counts = Counter()
    out_counts = Counter()

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
            
            # Draw tracking bounding boxes
            annotated = annotate_frame(frame, tracks, model.names)
            
            # Check line crossings
            for track in tracks:
                x1, y1, x2, y2, track_id, class_id = track
                track_id, class_id = int(track_id), int(class_id)
                
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                P = (cx, cy)
                
                side = 1 if get_side(P, L1, L2) >= 0 else -1
                
                if track_id in track_centroids:
                    prev_P = track_centroids[track_id]
                    prev_side = track_sides[track_id]
                    
                    if side != prev_side:
                        if intersect(prev_P, P, L1, L2):
                            if track_id not in counted_ids:
                                counted_ids.add(track_id)
                                name = class_name(model.names, class_id)
                                if prev_side == 1 and side == -1:
                                    in_counts[name] += 1
                                else:
                                    out_counts[name] += 1
                
                track_centroids[track_id] = P
                track_sides[track_id] = side
            
            # Draw line on the frame
            cv2.line(annotated, L1, L2, (0, 165, 255), 3)
            cv2.circle(annotated, L1, 6, (0, 0, 255), -1)
            cv2.circle(annotated, L2, 6, (0, 0, 255), -1)
            cv2.putText(annotated, "COUNTING LINE", (L1[0] + 10, L1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, lineType=cv2.LINE_AA)
            
            writer.write(annotated)

            frame_number += 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            if frame_number == 1 or frame_number % 5 == 0:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                with metrics.container():
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Processing speed", f"{frame_number / elapsed:.1f} FPS")
                    col2.metric("Total IN", sum(in_counts.values()))
                    col3.metric("Total OUT", sum(out_counts.values()))
                if total_frames > 0:
                    progress.progress(min(frame_number / total_frames, 1.0), text=f"Analyzing frame {frame_number:,} of {total_frames:,}")
    finally:
        capture.release()
        writer.release()

    progress.empty()
    with output_path.open("rb") as video_file:
        video_bytes = video_file.read()
    output_path.unlink(missing_ok=True)
    return video_bytes, frame_number, time.perf_counter() - started, in_counts, out_counts


def main() -> None:
    st.markdown(
        """<style>
        .stApp { background: radial-gradient(circle at top right, #183f65 0, #071526 28rem, #05101f 100%); }
        .block-container { max-width: 1240px; padding-top: 2.25rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: #081827; border-right: 1px solid rgba(139, 206, 255, .12); }
        .hero { padding: 2.2rem; border-radius: 24px; background: linear-gradient(120deg, rgba(28, 105, 144, .9), rgba(35, 197, 171, .78)); color: white; box-shadow: 0 18px 48px rgba(0, 0, 0, .23); }
        .hero h1 { margin: 0; font-size: 2.45rem; letter-spacing: -.04em; } .hero p { margin: .4rem 0 0; opacity: .92; font-size: 1.04rem; }
        .eyebrow { color: #93f6e9; font-weight: 700; text-transform: uppercase; letter-spacing: .12em; font-size: .74rem; }
        .feature-card { padding: 1.1rem 1.2rem; min-height: 110px; border: 1px solid rgba(147,246,233,.22); border-radius: 16px; background: rgba(14, 36, 55, .78); }
        .feature-card strong { display:block; color: #d9fffa; margin-bottom: .25rem; font-size: 1.02rem; }
        [data-testid="stMetric"] { background: rgba(14, 36, 55, .72); border: 1px solid rgba(139, 206, 255, .16); border-radius: 14px; padding: .8rem; }
        div[data-testid="stFileUploader"] { padding: 1rem; border: 1px dashed rgba(147,246,233,.45); border-radius: 16px; background: rgba(14, 36, 55, .55); }
        .stButton > button { border-radius: 10px; font-weight: 700; min-height: 2.7rem; }
        </style>""",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='hero'><div class='eyebrow'>Computer vision workspace</div><h1>🎯 VisionTrack AI</h1><p>Recognize objects, display their names, and follow them across every frame.</p></div>", unsafe_allow_html=True)
    st.write("")

    intro_columns = st.columns(3)
    feature_text = [("📹 Upload a video", "Get an annotated MP4 ready to download."), ("📸 Use your webcam", "Capture a photo and recognize objects instantly."), ("🏷️ Track every object", "Persistent IDs and readable object labels.")]
    for column, (title, description) in zip(intro_columns, feature_text):
        column.markdown(f"<div class='feature-card'><strong>{title}</strong><span>{description}</span></div>", unsafe_allow_html=True)
    st.write("")

    with st.sidebar:
        st.markdown("## Control room")
        st.caption("Tune the AI before starting an analysis.")
        st.divider()
        st.markdown("#### Model settings")
        model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], help="Larger models are more accurate but slower.")
        confidence = st.slider("Minimum confidence", 0.1, 0.9, 0.35, 0.05)
        class_filter = st.text_input("Class IDs (optional)", placeholder="Example: 0, 2 for person and car")
        st.caption("COCO examples: 0 = person, 2 = car, 16 = dog.")
        
        st.divider()
        st.markdown("#### Counting Line Settings")
        line_orientation = st.radio("Orientation", ["Horizontal", "Vertical", "Custom"], index=0)
        if line_orientation == "Horizontal":
            line_y = st.slider("Line Height (Y ratio)", 0.0, 1.0, 0.5, 0.05)
            line_coords = (0.0, line_y, 1.0, line_y)
        elif line_orientation == "Vertical":
            line_x = st.slider("Line Width (X ratio)", 0.0, 1.0, 0.5, 0.05)
            line_coords = (line_x, 0.0, line_x, 1.0)
        else:
            col1, col2 = st.columns(2)
            x1 = col1.slider("Start X", 0.0, 1.0, 0.1, 0.05)
            y1 = col2.slider("Start Y", 0.0, 1.0, 0.5, 0.05)
            x2 = col1.slider("End X", 0.0, 1.0, 0.9, 0.05)
            y2 = col2.slider("End Y", 0.0, 1.0, 0.5, 0.05)
            line_coords = (x1, y1, x2, y2)

    try:
        allowed_classes = [int(value.strip()) for value in class_filter.split(",") if value.strip()] or None
    except ValueError:
        st.sidebar.error("Class IDs must be comma-separated integers.")
        st.stop()

    mode = st.radio("Input source", ["Upload video", "Webcam photo", "Live webcam stream"], horizontal=True, label_visibility="collapsed")
    try:
        model = load_model(model_name)
    except Exception as error:
        st.error(f"Could not load {model_name}: {error}")
        st.stop()

    if mode == "Webcam photo":
        st.subheader("Webcam recognition")
        st.caption("Take a photo. Every visible object will receive a name and an ID.")
        photo = st.camera_input("Capture a frame")
        if photo is not None:
            image = cv2.imdecode(np.frombuffer(photo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Recognizing objects…"):
                annotated, tracks = detect_image(model, image, confidence, allowed_classes)
            image_column, results_column = st.columns([1.65, 1])
            image_column.image(annotated, caption="Recognition result", use_container_width=True)
            with results_column:
                st.markdown("### What I found")
                st.metric("Objects detected", len(tracks))
                summary = summarize_tracks(tracks, model.names)
                if summary:
                    st.dataframe(
                        [{"Object": label.title(), "Count": count} for label, count in summary.most_common()],
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info("No objects matched the selected confidence or class filter.")
        return

    if mode == "Live webcam stream":
        st.subheader("Live Webcam Tracking & Line Crossing Counter")
        st.caption("Runs your local webcam stream, tracks objects, and counts line crossings in real-time.")
        
        run_feed = st.toggle("Start Webcam Feed", value=False)
        
        if run_feed:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not access webcam. Make sure it is connected and not in use by another app.")
            else:
                preview = st.empty()
                metrics = st.empty()
                
                # Initialize SORT tracker and tracking states
                tracker = Sort(min_hits=2)
                track_centroids = {}
                track_sides = {}
                counted_ids = set()
                in_counts = Counter()
                out_counts = Counter()
                
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                
                # Calculate line coords in pixels
                L1 = (int(line_coords[0] * width), int(line_coords[1] * height))
                L2 = (int(line_coords[2] * width), int(line_coords[3] * height))
                
                prev_time = time.perf_counter()
                
                try:
                    while run_feed:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from webcam.")
                            break
                            
                        results = model(frame, conf=confidence, verbose=False)
                        tracks = tracker.update(extract_detections(results, allowed_classes))
                        
                        annotated = annotate_frame(frame, tracks, model.names)
                        
                        # Check line crossings
                        for track in tracks:
                            x1, y1, x2, y2, track_id, class_id = track
                            track_id, class_id = int(track_id), int(class_id)
                            
                            cx = int((x1 + x2) / 2.0)
                            cy = int((y1 + y2) / 2.0)
                            P = (cx, cy)
                            
                            side = 1 if get_side(P, L1, L2) >= 0 else -1
                            
                            if track_id in track_centroids:
                                prev_P = track_centroids[track_id]
                                prev_side = track_sides[track_id]
                                
                                if side != prev_side:
                                    if intersect(prev_P, P, L1, L2):
                                        if track_id not in counted_ids:
                                            counted_ids.add(track_id)
                                            name = class_name(model.names, class_id)
                                            if prev_side == 1 and side == -1:
                                                in_counts[name] += 1
                                            else:
                                                out_counts[name] += 1
                            
                            track_centroids[track_id] = P
                            track_sides[track_id] = side
                        
                        # Draw line on the frame
                        cv2.line(annotated, L1, L2, (0, 165, 255), 3)
                        cv2.circle(annotated, L1, 6, (0, 0, 255), -1)
                        cv2.circle(annotated, L2, 6, (0, 0, 255), -1)
                        cv2.putText(annotated, "COUNTING LINE", (L1[0] + 10, L1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, lineType=cv2.LINE_AA)
                        
                        # Render preview
                        preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        
                        # Update metrics
                        curr_time = time.perf_counter()
                        fps = 1.0 / max(curr_time - prev_time, 1e-6)
                        prev_time = curr_time
                        
                        with metrics.container():
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Frame Rate", f"{fps:.1f} FPS")
                            col2.metric("Total IN", sum(in_counts.values()))
                            col3.metric("Total OUT", sum(out_counts.values()))
                            
                            # Break down counts
                            if sum(in_counts.values()) + sum(out_counts.values()) > 0:
                                st.markdown("#### Object-wise Counts")
                                breakdown_list = []
                                all_keys = set(in_counts.keys()).union(out_counts.keys())
                                for k in all_keys:
                                    breakdown_list.append({
                                        "Object": k.title(),
                                        "IN": in_counts.get(k, 0),
                                        "OUT": out_counts.get(k, 0)
                                    })
                                st.dataframe(breakdown_list, hide_index=True, use_container_width=True)
                                
                        time.sleep(0.01)
                finally:
                    cap.release()
        return

    st.subheader("Video tracking")
    st.caption("Upload a clip to create an annotated video with object names, persistent tracking IDs, and line crossing counting.")
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_video is None:
        st.info("Upload a video to start detection and tracking.")
        return

    left, right = st.columns([1.5, 1])
    left.video(uploaded_video)
    with right:
        st.markdown("### Ready to analyze")
        st.write(f"**File:** `{uploaded_video.name}`")
        st.write(f"**Confidence:** `{confidence:.0%}`")
        st.write("Your output includes names, IDs, counting line, and the live detection overlay.")
    if st.button("Analyze video", type="primary", use_container_width=True):
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        input_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        input_path = Path(input_file.name)
        try:
            input_file.write(uploaded_video.getvalue())
            input_file.close()
            video_bytes, frame_count, elapsed, in_counts, out_counts = render_video(model, input_path, confidence, allowed_classes, line_coords)
        except Exception as error:
            st.error(f"Video processing failed: {error}")
            return
        finally:
            input_file.close()
            input_path.unlink(missing_ok=True)

        st.success("Analysis complete — your tracked video is ready.")
        summary_columns = st.columns(4)
        summary_columns[0].metric("Frames processed", f"{frame_count:,}")
        summary_columns[1].metric("Processing time", f"{elapsed:.1f} s")
        summary_columns[2].metric("Total IN", sum(in_counts.values()))
        summary_columns[3].metric("Total OUT", sum(out_counts.values()))
        
        # Display object-wise counts
        if sum(in_counts.values()) + sum(out_counts.values()) > 0:
            st.markdown("#### Object-wise Counts")
            breakdown_list = []
            all_keys = set(in_counts.keys()).union(out_counts.keys())
            for k in all_keys:
                breakdown_list.append({
                    "Object": k.title(),
                    "IN": in_counts.get(k, 0),
                    "OUT": out_counts.get(k, 0)
                })
            st.dataframe(breakdown_list, hide_index=True, use_container_width=True)
            
        st.video(video_bytes)
        st.download_button("⬇ Download tracked video", video_bytes, "visiontrack-output.mp4", "video/mp4", use_container_width=True)


if __name__ == "__main__":
    main()
