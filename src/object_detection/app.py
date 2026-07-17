import sys
import time
import argparse
import urllib.request
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from ultralytics import YOLO

# Import custom SORT tracker
from .tracker import Sort

def get_color(track_id):
    """
    Generates a unique, deterministic, and bright color for a given track ID.
    """
    np.random.seed(int(track_id))
    # Generate random HSV values to guarantee high saturation and brightness
    h = np.random.randint(0, 180)
    s = np.random.randint(180, 255)
    v = np.random.randint(180, 255)
    
    # Convert HSV to BGR for OpenCV
    hsv_pixel = np.array([[[h, s, v]]], dtype=np.uint8)
    bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
    return tuple(int(c) for c in bgr_pixel[0, 0])

def draw_premium_bbox(img, bbox, label, color, thickness=2, corner_len=15):
    """
    Draws a clean, professional, human-crafted bounding box with a clear text label tag.
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Draw clean solid rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Setup label text details
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_thickness = 1
    t_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
    
    # Prevent the label from rendering off-screen at the top
    label_y = y1
    if label_y - t_size[1] - 6 < 0:
        label_y = y1 + t_size[1] + 6
        
    # Draw solid background block for text matching the box color
    cv2.rectangle(
        img, 
        (x1, label_y - t_size[1] - 6), 
        (x1 + t_size[0] + 6, label_y), 
        color, 
        -1
    )
    
    # Draw white text on top of the solid background block
    cv2.putText(
        img, 
        label, 
        (x1 + 3, label_y - 4), 
        font, 
        font_scale, 
        (255, 255, 255), 
        text_thickness, 
        lineType=cv2.LINE_AA
    )

def download_sample_video(destination: Path) -> Optional[Path]:
    """
    Downloads a short sample video of pedestrians for demonstration if needed.
    """
    url = "https://raw.githubusercontent.com/intel-iot-devkit/sample-videos/master/people-detection.mp4"
    if not destination.exists():
        print(f"Downloading sample video from {url} to {destination}...")
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, destination)
            print("Download successful!")
        except Exception as e:
            print(f"Warning: Could not download sample video: {e}")
            return None
    return destination

def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_side(P, L1, L2):
    return (P[0] - L1[0]) * (L2[1] - L1[1]) - (P[1] - L1[1]) * (L2[0] - L1[0])

def main():
    parser = argparse.ArgumentParser(description="Real-time Object Detection and Tracking with YOLOv8 & SORT")
    parser.add_argument(
        "--source", 
        type=str, 
        default="0", 
        help="Video source: '0' or device index for webcam, or a path to a video file."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="yolov8n.pt", 
        help="YOLO model configuration or weight file (default: yolov8n.pt)."
    )
    parser.add_argument(
        "--conf", 
        type=float, 
        default=0.35, 
        help="Confidence threshold for YOLO detections."
    )
    parser.add_argument(
        "--iou", 
        type=float, 
        default=0.3, 
        help="IoU threshold for SORT track association matching."
    )
    parser.add_argument(
        "--max-age", 
        type=int, 
        default=15, 
        help="Maximum frames a track can go unmatched before deletion."
    )
    parser.add_argument(
        "--min-hits", 
        type=int, 
        default=3, 
        help="Minimum matches to establish a confirmed track."
    )
    parser.add_argument(
        "--classes", 
        type=int, 
        nargs="+", 
        default=None, 
        help="Filter tracking to specific class IDs (e.g. --classes 0 2 for person and car)."
    )
    parser.add_argument(
        "--save", 
        type=str, 
        default=None, 
        help="Path to save output video (optional)."
    )
    parser.add_argument("--no-display", action="store_true", help="Run without an OpenCV window.")
    parser.add_argument("--download-sample", action="store_true", help="Download and run the sample video.")
    parser.add_argument(
        "--line", 
        type=float, 
        nargs=4, 
        default=[0.1, 0.5, 0.9, 0.5], 
        help="Line coordinates as fraction of width/height: start_x start_y end_x end_y (default: 0.1 0.5 0.9 0.5)"
    )
    args = parser.parse_args()

    # Load YOLO Model
    print(f"Loading YOLO model: {args.model}...")
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        sys.exit(1)
        
    class_names = model.names

    # Resolve Video Source
    # Check if index is numeric (meaning webcam)
    if args.source.isdigit():
        source_val = int(args.source)
        is_webcam = True
    else:
        source_val = args.source
        is_webcam = False
        if not Path(source_val).is_file():
            parser.error(f"Video file not found: {source_val}")

    if args.download_sample:
        sample_path = download_sample_video(Path("assets") / "people_sample.mp4")
        if sample_path is None:
            sys.exit(1)
        source_val = str(sample_path)
        is_webcam = False

    print(f"Initializing video input: {source_val}")
    cap = cv2.VideoCapture(source_val)
    
    if not cap.isOpened():
        if is_webcam:
            print(f"Error: Could not access webcam at index {source_val}.")
            sys.exit(1)
        else:
            print(f"Error: Could not open video file {source_val}.")
            sys.exit(1)

    # Output Video Writer setup if saving
    video_writer = None
    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        if not video_writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {output_path}")
        print(f"Saving tracking output video to: {output_path}")

    # Initialize SORT Tracker
    tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou)

    # Initialize tracking structures for line crossing
    L1 = None
    L2 = None
    track_centroids = {}
    track_sides = {}
    counted_ids = set()
    in_counts = {}
    out_counts = {}

    print("\n-----------------------------------------------------")
    print("Press 'q' key in the video screen to exit the program.")
    print("-----------------------------------------------------\n")

    prev_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream or failed to read frame.")
                break
                
            start_inference_time = time.time()
            
            # Run YOLO Object Detection on frame
            results = model(frame, conf=args.conf, verbose=False)
            
            # Extract detections in format [x1, y1, x2, y2, score, class_id]
            dets_list = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    
                    # Apply optional class filtering
                    if args.classes is None or cls in args.classes:
                        dets_list.append([x1, y1, x2, y2, conf, cls])
                        
            # Convert to numpy array
            if len(dets_list) > 0:
                dets = np.array(dets_list)
            else:
                # empty detections must be of shape (0, 6)
                dets = np.empty((0, 6))

            # Update SORT Tracker
            tracked_objects = tracker.update(dets)
            
            inference_duration = time.time() - start_inference_time
            
            # Initialize line coordinate bounds based on actual frame size if not done
            if L1 is None and L2 is None:
                h, w = frame.shape[:2]
                L1 = (int(args.line[0] * w), int(args.line[1] * h))
                L2 = (int(args.line[2] * w), int(args.line[3] * h))

            # Draw counting line
            if L1 is not None and L2 is not None:
                cv2.line(frame, L1, L2, (0, 165, 255), 3) # Orange
                cv2.circle(frame, L1, 6, (0, 0, 255), -1)
                cv2.circle(frame, L2, 6, (0, 0, 255), -1)
                cv2.putText(frame, "COUNTING LINE", (L1[0] + 10, L1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, lineType=cv2.LINE_AA)

            # Draw tracking output on the frame
            for obj in tracked_objects:
                x1, y1, x2, y2, track_id, class_id = obj
                track_id = int(track_id)
                class_id = int(class_id)
                
                # Compute centroid
                cx = int((x1 + x2) / 2.0)
                cy = int((y1 + y2) / 2.0)
                P = (cx, cy)
                
                # Determine side of line
                side = 1 if get_side(P, L1, L2) >= 0 else -1
                
                # Check line crossing
                if track_id in track_centroids:
                    prev_P = track_centroids[track_id]
                    prev_side = track_sides[track_id]
                    
                    if side != prev_side:
                        if intersect(prev_P, P, L1, L2):
                            if track_id not in counted_ids:
                                counted_ids.add(track_id)
                                name = class_names.get(class_id, f"Class {class_id}")
                                if prev_side == 1 and side == -1:
                                    in_counts[name] = in_counts.get(name, 0) + 1
                                else:
                                    out_counts[name] = out_counts.get(name, 0) + 1
                                    
                track_centroids[track_id] = P
                track_sides[track_id] = side
                
                # Fetch class name
                name = class_names.get(class_id, f"Class {class_id}")
                
                # Render label
                label = f"ID {track_id} | {name}"
                color = get_color(track_id)
                
                # Draw premium bbox
                draw_premium_bbox(frame, (x1, y1, x2, y2), label, color)

            # Calculate and display frame stats (FPS, Latency)
            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time
            
            # Frame overlay diagnostic panel (semi-transparent glassmorphic banner)
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (320, 115), (30, 30, 30), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            
            # Text lines inside diagnostic banner
            cv2.putText(
                frame, 
                f"FPS: {fps:.1f}", 
                (15, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                1, 
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame, 
                f"Active Tracks: {len(tracker.trackers)}", 
                (15, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 191, 255), 
                1, 
                lineType=cv2.LINE_AA
            )
            # Display IN/OUT counts
            total_in = sum(in_counts.values())
            total_out = sum(out_counts.values())
            cv2.putText(
                frame, 
                f"IN: {total_in}", 
                (15, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.55, 
                (50, 255, 50), 
                2, 
                lineType=cv2.LINE_AA
            )
            cv2.putText(
                frame, 
                f"OUT: {total_out}", 
                (15, 95), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.55, 
                (50, 50, 255), 
                2, 
                lineType=cv2.LINE_AA
            )

            # Save frame to output video writer if configured
            if video_writer is not None:
                video_writer.write(frame)

            if not args.no_display:
                cv2.imshow("Real-Time Object Detection and Tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit signal received from user.")
                    break
                
    except KeyboardInterrupt:
        print("Interrupt received. Stopping...")
        
    finally:
        # Release resources
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print("Video capture and windows released. Cleaned up successfully.")

if __name__ == "__main__":
    main()
