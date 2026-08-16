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
from object_detection.intelligence import VideoIntelligenceEngine, Zone
from object_detection.intelligence.reports import build_report, report_json, timeline_csv
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


def draw_intelligence_overlay(frame: np.ndarray, insights, zones: List[Zone]) -> np.ndarray:
    """Add restrained zones, paths, direction arrows, and high-risk indicators."""
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    zone_colors = {"restricted": (55, 80, 235), "entry": (70, 190, 90), "exit": (220, 130, 40), "normal": (205, 150, 40)}
    for zone in zones:
        start, end = (int(zone.x1 * width), int(zone.y1 * height)), (int(zone.x2 * width), int(zone.y2 * height))
        color = zone_colors.get(zone.kind, zone_colors["normal"])
        cv2.rectangle(annotated, start, end, color, 2)
        cv2.putText(annotated, f"{zone.name} ({zone.kind})", (start[0] + 4, max(16, start[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for insight in insights:
        points = [(int(x), int(y)) for x, y in insight.motion.history]
        if len(points) > 1:
            cv2.polylines(annotated, [np.array(points, dtype=np.int32)], False, get_color(insight.track_id), 2)
        if insight.motion.previous_centroid is not None:
            previous = tuple(map(int, insight.motion.previous_centroid))
            current = tuple(map(int, insight.motion.centroid))
            cv2.arrowedLine(annotated, previous, current, get_color(insight.track_id), 2, tipLength=0.3)
        if max(insight.anomaly.score, insight.risk_score) >= 0.70:
            point = tuple(map(int, insight.motion.centroid))
            cv2.circle(annotated, point, 14, (20, 20, 240), 2)
    return annotated


def detect_image(
    model: YOLO,
    image: np.ndarray,
    confidence: float,
    classes: Optional[List[int]],
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Recognize objects in one image and return an annotated RGB image."""
    results = model(image, conf=confidence, verbose=False)
    detections = extract_detections(results, classes)
    tracker = Sort(min_hits=1, iou_threshold=iou_threshold)
    tracks = tracker.update(detections)
    return cv2.cvtColor(annotate_frame(image, tracks, model.names), cv2.COLOR_BGR2RGB), tracks


def render_video(
    model: YOLO,
    source: Path,
    confidence: float,
    classes: Optional[List[int]],
    line_coords: tuple[float, float, float, float],
    iou_threshold: float,
    max_age: int,
    min_hits: int,
    show_counting_line: bool,
    zones: List[Zone],
    trajectory_history: int,
) -> Tuple[bytes, int, float, Counter, Counter, Dict[str, float], Dict, bytes]:
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

    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    intelligence = VideoIntelligenceEngine(zones, trajectory_history)
    
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
    inference_total = 0.0
    peak_tracks = 0
    started = time.perf_counter()

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            inference_started = time.perf_counter()
            results = model(frame, conf=confidence, verbose=False)
            inference_total += time.perf_counter() - inference_started
            tracks = tracker.update(extract_detections(results, classes))
            peak_tracks = max(peak_tracks, len(tracks))
            insights, crowd = intelligence.update(tracks, time.time(), frame.shape[:2])
            
            # Draw tracking bounding boxes
            annotated = annotate_frame(frame, tracks, model.names)
            
            # Check line crossings
            for track in tracks if show_counting_line else []:
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
            if show_counting_line:
                cv2.line(annotated, L1, L2, (0, 165, 255), 3)
                cv2.circle(annotated, L1, 6, (0, 0, 255), -1)
                cv2.circle(annotated, L2, 6, (0, 0, 255), -1)
                cv2.putText(annotated, "COUNTING LINE", (L1[0] + 10, L1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, lineType=cv2.LINE_AA)
            annotated = draw_intelligence_overlay(annotated, insights, zones)
            
            writer.write(annotated)

            frame_number += 1
            elapsed = max(time.perf_counter() - started, 1e-6)
            if frame_number == 1 or frame_number % 5 == 0:
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                with metrics.container():
                    col1, col2, col3, col4, col5 = st.columns(5)
                    col1.metric("Processing speed", f"{frame_number / elapsed:.1f} FPS")
                    col2.metric("Inference latency", f"{inference_total / frame_number * 1000:.0f} ms")
                    col3.metric("Active tracks", len(tracks))
                    col4.metric("Flow events", sum(in_counts.values()) + sum(out_counts.values()))
                    col5.metric("Crowd density", crowd.density_per_100k_px)
                if total_frames > 0:
                    progress.progress(min(frame_number / total_frames, 1.0), text=f"Analyzing frame {frame_number:,} of {total_frames:,}")
    finally:
        capture.release()
        writer.release()

    progress.empty()
    with output_path.open("rb") as video_file:
        video_bytes = video_file.read()
    output_path.unlink(missing_ok=True)
    elapsed = time.perf_counter() - started
    performance = {
        "avg_inference_ms": inference_total / max(frame_number, 1) * 1000,
        "peak_tracks": peak_tracks,
        "processing_fps": frame_number / max(elapsed, 1e-6),
    }
    report = build_report(intelligence, frame_number / fps)
    report["objects_by_class"] = {
        class_name(model.names, int(class_id)): count
        for class_id, count in report.pop("objects_by_class_id", {}).items()
    }
    report["line_crossing"] = {"in": dict(in_counts), "out": dict(out_counts)}
    return video_bytes, frame_number, elapsed, in_counts, out_counts, performance, report, timeline_csv(intelligence)


def main() -> None:
    st.markdown(
        """<style>
        .stApp { background: linear-gradient(135deg, #07111f 0%, #0b1e33 55%, #102f42 100%); }
        .block-container { max-width: 1240px; padding-top: 2.2rem; padding-bottom: 4rem; }
        [data-testid="stSidebar"] { background: #081827; border-right: 1px solid rgba(98, 210, 195, .16); }
        .hero { padding: 2.1rem 2.3rem; border-radius: 22px; background: linear-gradient(120deg, #0d5c7a, #0d9488); color: white; box-shadow: 0 20px 45px rgba(0,0,0,.22); }
        .hero h1 { margin: .15rem 0; font-size: 2.45rem; letter-spacing: -.045em; } .hero p { margin: .45rem 0 0; opacity: .92; font-size: 1.05rem; }
        .eyebrow { font-size: .75rem; font-weight: 800; letter-spacing: .13em; color: #bffcf2; text-transform: uppercase; }
        .feature { border: 1px solid rgba(126, 239, 221, .18); border-radius: 14px; background: rgba(10, 31, 48, .72); padding: 1rem 1.1rem; min-height: 95px; }
        .feature strong { display: block; color: #dffffa; margin-bottom: .22rem; }
        [data-testid="stMetric"] { background: rgba(10, 31, 48, .7); border: 1px solid rgba(126, 239, 221, .17); border-radius: 14px; padding: .8rem; }
        .stButton > button { border-radius: 10px; min-height: 2.7rem; font-weight: 700; }
        </style>
        <div class="hero"><div class="eyebrow">Computer vision workspace</div><h1>VisionTrack AI</h1><p>Detect, identify, track, count, and export objects from images, video, and a local camera.</p></div>""",
        unsafe_allow_html=True,
    )
    feature_columns = st.columns(3)
    for column, content in zip(
        feature_columns,
        [("01  INPUT", "Image, video, or local webcam"), ("02  INSIGHT", "Names, IDs, confidence, and flow"), ("03  OUTPUT", "Live preview and annotated MP4 export")],
    ):
        column.markdown(f"<div class='feature'><strong>{content[0]}</strong>{content[1]}</div>", unsafe_allow_html=True)
    st.write("")

    with st.sidebar:
        st.markdown("## Control room")
        st.caption("Adjust the model and tracking behavior before analysis.")
        
        model_name = st.selectbox(
            "YOLO Model Size", 
            ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"], 
            help="Larger models provide higher accuracy but require more compute."
        )
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        iou_threshold = st.slider("Tracking IoU Threshold", 0.1, 0.9, 0.30, 0.05)
        class_filter = st.text_input("Filter Class IDs (comma-separated)", placeholder="e.g. 0, 2 for person and car")
        st.caption("Common COCO IDs: 0 = Person, 2 = Car, 5 = Bus, 7 = Truck, 16 = Dog.")
        with st.expander("Tracking behavior"):
            max_age = st.slider("Keep unmatched track (frames)", 1, 60, 15)
            min_hits = st.slider("Confirm track after matches", 1, 10, 2)
            trajectory_history = st.slider("Trajectory history (frames)", 10, 240, 60, 10)

        with st.expander("Intelligence zones", expanded=True):
            st.caption("Define up to three rectangular zones using normalized frame coordinates.")
            zone_count = st.select_slider("Number of zones", options=[0, 1, 2, 3], value=1)
            zone_specs = []
            for index in range(zone_count):
                st.markdown(f"**Zone {index + 1}**")
                name = st.text_input("Name", value=f"Zone {chr(65 + index)}", key=f"zone_name_{index}")
                kind = st.selectbox("Type", ["normal", "restricted", "entry", "exit"], index=1 if index == 0 else 0, key=f"zone_kind_{index}")
                first, second = st.columns(2)
                x1 = first.slider("Left", 0.0, 1.0, 0.35, 0.05, key=f"zone_x1_{index}")
                y1 = second.slider("Top", 0.0, 1.0, 0.25, 0.05, key=f"zone_y1_{index}")
                x2 = first.slider("Right", 0.0, 1.0, 0.65, 0.05, key=f"zone_x2_{index}")
                y2 = second.slider("Bottom", 0.0, 1.0, 0.75, 0.05, key=f"zone_y2_{index}")
                zone_specs.append((name, kind, x1, y1, x2, y2))
        
        st.divider()
        show_counting_line = st.toggle("Enable flow counting", value=True)
        st.markdown("#### Counting line")
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
    zones = [Zone(name.strip() or f"Zone {index + 1}", kind, x1, y1, x2, y2) for index, (name, kind, x1, y1, x2, y2) in enumerate(zone_specs) if x1 != x2 and y1 != y2]

    mode = st.radio(
        "Input source",
        ["Upload image", "Upload video", "Webcam photo", "Live webcam stream"],
        horizontal=True,
        label_visibility="collapsed",
    )
    try:
        model = load_model(model_name)
    except Exception as error:
        st.error(f"Could not load {model_name}: {error}")
        st.stop()

    if mode in ("Upload image", "Webcam photo"):
        st.subheader("Image recognition")
        st.caption("Use an image or camera photo to detect object names, boxes, and IDs.")
        input_file = (
            st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
            if mode == "Upload image"
            else st.camera_input("Capture a frame")
        )
        if input_file is not None:
            image = cv2.imdecode(np.frombuffer(input_file.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            with st.spinner("Recognizing objects…"):
                annotated, tracks = detect_image(model, image, confidence, allowed_classes, iou_threshold)
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
        st.subheader("Local webcam session")
        st.caption("Runs the camera attached to the computer hosting Streamlit. Sessions end automatically so the page remains responsive.")
        frame_limit = st.number_input("Frames to process", min_value=30, max_value=3600, value=300, step=30)
        
        run_feed = st.toggle("Start Webcam Feed", value=False)
        
        if run_feed:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Could not access webcam. Make sure it is connected and not in use by another app.")
            else:
                preview = st.empty()
                metrics = st.empty()
                
                # Initialize SORT tracker and tracking states
                tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
                intelligence = VideoIntelligenceEngine(zones, trajectory_history)
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
                processed_frames = 0
                
                try:
                    while run_feed and processed_frames < frame_limit:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from webcam.")
                            break
                            
                        inference_started = time.perf_counter()
                        results = model(frame, conf=confidence, verbose=False)
                        inference_ms = (time.perf_counter() - inference_started) * 1000
                        tracks = tracker.update(extract_detections(results, allowed_classes))
                        insights, crowd = intelligence.update(tracks, time.time(), frame.shape[:2])
                        
                        annotated = annotate_frame(frame, tracks, model.names)
                        annotated = draw_intelligence_overlay(annotated, insights, zones)
                        
                        # Check line crossings
                        for track in tracks if show_counting_line else []:
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
                        if show_counting_line:
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
                            col1, col2, col3, col4, col5 = st.columns(5)
                            col1.metric("Frame Rate", f"{fps:.1f} FPS")
                            col2.metric("Inference latency", f"{inference_ms:.0f} ms")
                            col3.metric("Active tracks", len(tracks))
                            col4.metric("Flow events", sum(in_counts.values()) + sum(out_counts.values()))
                            col5.metric("Crowd density", crowd.density_per_100k_px)
                            
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
                                
                        processed_frames += 1
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
            video_bytes, frame_count, elapsed, in_counts, out_counts, performance, intelligence_report, timeline_bytes = render_video(
                model,
                input_path,
                confidence,
                allowed_classes,
                line_coords,
                iou_threshold,
                max_age,
                min_hits,
                show_counting_line,
                zones,
                trajectory_history,
            )
        except Exception as error:
            st.error(f"Video processing failed: {error}")
            return
        finally:
            input_file.close()
            input_path.unlink(missing_ok=True)

        st.success("Analysis complete — your tracked video is ready.")
        summary_columns = st.columns(5)
        summary_columns[0].metric("Frames processed", f"{frame_count:,}")
        summary_columns[1].metric("Processing time", f"{elapsed:.1f} s")
        summary_columns[2].metric("Average inference", f"{performance['avg_inference_ms']:.0f} ms")
        summary_columns[3].metric("Peak active tracks", int(performance["peak_tracks"]))
        summary_columns[4].metric("Flow events", sum(in_counts.values()) + sum(out_counts.values()))
        
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

        st.markdown("### Video intelligence report")
        report_columns = st.columns(3)
        report_columns[0].metric("Unique tracked objects", intelligence_report["unique_objects"])
        report_columns[1].metric("Average speed", f"{intelligence_report['average_speed_px_s']:.1f} px/s")
        report_columns[2].metric("Timeline events", intelligence_report["event_count"])
        highest_risk = intelligence_report.get("highest_risk_event")
        if highest_risk:
            st.warning(f"Trajectory-based highest risk: Object #{highest_risk['track_id']} — score {highest_risk['risk_score']:.2f}. {highest_risk['reason']}")
        timeline_rows = intelligence_report.get("timeline", [])
        if timeline_rows:
            st.markdown("#### Event timeline")
            event_types = sorted({row["event_type"] for row in timeline_rows})
            selected_event_type = st.selectbox("Filter timeline by event type", ["All"] + event_types)
            selected_track_id = st.selectbox("Filter timeline by object", ["All"] + sorted({str(row["track_id"]) for row in timeline_rows if row["track_id"] is not None}))
            visible_events = [
                row for row in timeline_rows
                if (selected_event_type == "All" or row["event_type"] == selected_event_type)
                and (selected_track_id == "All" or str(row["track_id"]) == selected_track_id)
            ]
            st.dataframe(visible_events, hide_index=True, use_container_width=True)
        downloads = st.columns(3)
        downloads[0].download_button("Download annotated video", video_bytes, "visiontrack-output.mp4", "video/mp4", use_container_width=True)
        downloads[1].download_button("Download JSON report", report_json(intelligence_report), "visiontrack-report.json", "application/json", use_container_width=True)
        downloads[2].download_button("Download event timeline CSV", timeline_bytes, "visiontrack-timeline.csv", "text/csv", use_container_width=True)
            
        st.video(video_bytes)


if __name__ == "__main__":
    main()
