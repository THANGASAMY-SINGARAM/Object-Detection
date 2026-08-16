"""Streamlit interface for YOLO detection and SORT tracking."""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from object_detection.app import draw_premium_bbox, get_color, ccw, intersect, get_side
from object_detection.adaptive import AdaptiveInferenceController
from object_detection.detection import DetectionConfig, PRESETS, detection_statistics, run_detection
from object_detection.intelligence import VideoIntelligenceEngine, Zone
from object_detection.intelligence.reports import build_report, report_json, timeline_csv
from object_detection.tracker import Sort


st.set_page_config(
    page_title="VisionTrack AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
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
    detection_config: DetectionConfig,
    iou_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Recognize objects in one image and return an annotated RGB image."""
    detections = run_detection(model, image, detection_config)
    tracker = Sort(min_hits=1, iou_threshold=iou_threshold)
    tracks = tracker.update(detections)
    return cv2.cvtColor(annotate_frame(image, tracks, model.names), cv2.COLOR_BGR2RGB), tracks, detection_statistics(detections, model.names)


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
    detection_config: DetectionConfig,
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
    adaptive = AdaptiveInferenceController(mode="AUTO", target_fps=30)
    last_detections = None
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
            lost_tracks = sum(track.time_since_update > 0 for track in tracker.trackers)
            decision = adaptive.decide(last_detections, len(tracker.trackers), lost_tracks, frame_number / max(time.perf_counter() - started, 1e-6) if frame_number else None)
            if decision.should_detect:
                detections = run_detection(model, frame, replace(detection_config, imgsz=decision.resolution))
                last_detections = detections
                inference_total += time.perf_counter() - inference_started
            else:
                detections = np.empty((0, 6), dtype=np.float32)
            tracks = tracker.update(detections)
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
        .stApp { background-color:#08111d; background-image:linear-gradient(rgba(54,132,170,.045) 1px, transparent 1px), linear-gradient(90deg, rgba(54,132,170,.045) 1px, transparent 1px); background-size:42px 42px; color:#d7e3ef; }
        .block-container { max-width: 1440px; padding: 1.2rem 2rem 3rem; }
        [data-testid="stSidebar"] { background: #0c1826; border-right: 1px solid #1d3043; }
        [data-testid="stSidebar"] .block-container { padding: 1.25rem .9rem; }
        .vt-header { display:flex; align-items:center; justify-content:space-between; padding: .8rem 0 1.1rem; border-bottom: 1px solid #1d3043; margin-bottom: 1rem; }
        .vt-brand { font-size:1.05rem; letter-spacing:.08em; font-weight:800; color:#e8f6ff; } .vt-sub { font-size:.68rem; letter-spacing:.1em; color:#7591aa; margin-top:.18rem; }
        .vt-live { color:#6fe3c1; font-size:.78rem; font-weight:700; letter-spacing:.06em; } .vt-live::before { content:'●'; margin-right:.45rem; }
        .vt-section { color:#8ca3b8; font-size:.72rem; font-weight:800; letter-spacing:.11em; margin:1.5rem 0 .7rem; }
        [data-testid="stMetric"] { background:#0e1d2c; border:1px solid #203548; border-radius:12px; padding:.7rem .85rem; box-shadow:none; }
        [data-testid="stMetricLabel"] { color:#7892a9; font-size:.68rem; letter-spacing:.06em; } [data-testid="stMetricValue"] { color:#ecf7ff; font-size:1.25rem; }
        .stButton > button { border-radius:9px; min-height:2.45rem; font-weight:650; border-color:#2a4a62; }
        [data-testid="stFileUploader"] { border:1px dashed #31536e; border-radius:14px; padding:1rem; background:#0d1b29; }
        [data-testid="stTabs"] [role="tablist"] { gap:.45rem; border-bottom:1px solid #203548; } [data-testid="stTabs"] button { color:#829ab0; border-radius:8px 8px 0 0; }
        [data-testid="stTabs"] button[aria-selected="true"] { color:#72ddff; background:#10283a; }
        @media (max-width: 900px) { .block-container { padding:1rem; } .vt-header { align-items:flex-start; } }
        </style>
        <div class="vt-header"><div><div class="vt-brand">VISIONTRACK AI</div><div class="vt-sub">COMPUTER VISION WORKSPACE</div></div><div class="vt-live">SYSTEM ONLINE</div></div>""",
        unsafe_allow_html=True,
    )
    navigation = st.radio("Navigation", ["Live Analysis", "Tracking", "Analytics", "Events", "Reports"], horizontal=True, label_visibility="collapsed")

    with st.sidebar:
        st.markdown("## Control room")
        st.caption("Adjust the model and tracking behavior before analysis.")
        
        detection_mode = st.selectbox("Detection mode", ["FAST", "BALANCED", "ACCURACY", "CUSTOM"], index=1)
        preset_model, preset_size = PRESETS.get(detection_mode, ("yolov8s.pt", 960))
        model_name = st.selectbox("YOLO model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"], index=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"].index(preset_model) if detection_mode != "CUSTOM" else 1, disabled=detection_mode != "CUSTOM")
        if detection_mode != "CUSTOM":
            model_name = preset_model
        imgsz = st.select_slider("Inference resolution", options=[640, 768, 960, 1280], value=preset_size)
        confidence = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
        nms_iou = st.slider("Detection NMS IoU", 0.3, 0.9, 0.60, 0.05)
        max_det = st.number_input("Maximum detections", min_value=10, max_value=1000, value=300, step=10)
        iou_threshold = st.slider("Tracking IoU Threshold", 0.1, 0.9, 0.30, 0.05)
        class_filter = st.text_input("Filter Class IDs (comma-separated)", placeholder="e.g. 0, 2 for person and car")
        st.caption("Common COCO IDs: 0 = Person, 2 = Car, 5 = Bus, 7 = Truck, 16 = Dog.")
        with st.expander("Tracking behavior"):
            max_age = st.slider("Keep unmatched track (frames)", 1, 60, 15)
            min_hits = st.slider("Confirm track after matches", 1, 10, 2)
            trajectory_history = st.slider("Trajectory history (frames)", 10, 240, 60, 10)
        with st.expander("Image accuracy options"):
            enhance_contrast = st.toggle("Contrast enhancement", value=False)
            sharpen = st.toggle("Light sharpening", value=False)
            tiled_inference = st.toggle("Tiled inference for large images", value=False, help="Can improve small-object recall, but reduces speed and may create duplicate candidates before NMS.")

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
    detection_config = DetectionConfig(confidence, nms_iou, int(imgsz), int(max_det), allowed_classes, enhance_contrast, sharpen, tiled_inference)

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
                annotated, tracks, stats = detect_image(model, image, detection_config, iou_threshold)
            image_column, results_column = st.columns([1.65, 1])
            image_column.image(annotated, caption="Recognition result", use_container_width=True)
            with results_column:
                st.markdown("### What I found")
                st.metric("Objects detected", len(tracks))
                st.metric("Average confidence", f"{stats['average_confidence']:.2f}" if stats["average_confidence"] is not None else "N/A")
                if stats["total"] >= 25:
                    st.warning("Dense scene detected. Accuracy may be reduced because of object overlap and small objects.")
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
                        detections = run_detection(model, frame, detection_config)
                        inference_ms = (time.perf_counter() - inference_started) * 1000
                        tracks = tracker.update(detections)
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
                detection_config,
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
