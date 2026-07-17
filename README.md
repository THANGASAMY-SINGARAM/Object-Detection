# YOLO + SORT Object Tracking

Real-time object detection and class-aware multi-object tracking using [Ultralytics YOLO](https://docs.ultralytics.com/) and a self-contained implementation of [SORT](https://arxiv.org/abs/1602.00763). It works with a webcam or video file, assigns persistent IDs to detected objects, and can render an annotated output video.

## Web app

The easiest way to use VisionTrack is the Streamlit web interface. It provides a modern UI for uploading a video, viewing tracked results, downloading the annotated MP4, or capturing a webcam photo for object recognition.

```powershell
streamlit run streamlit_app.py
```

Open the local URL displayed in the terminal (usually `http://localhost:8501`). The first run downloads the selected YOLO weights when they are not already present.

## Features

- YOLO inference with selectable model weights (defaults to `yolov8n.pt`)
- Class-aware SORT tracking to avoid matching objects across different categories
- Webcam, local-video, and downloadable sample-video inputs
- Live diagnostics for FPS, inference latency, and active tracks
- Headless mode and MP4 export for batch/server use

## Project layout

```text
.
├── main.py                       # Simple source-checkout entry point
├── streamlit_app.py               # Web interface for videos and webcam photos
├── src/object_detection/
│   ├── app.py                    # CLI, video pipeline, and drawing
│   └── tracker.py                # Kalman filter and SORT implementation
├── tests/test_tracker.py          # Tracker regression tests
├── requirements.txt
└── pyproject.toml                # Installable package configuration
```

## Quick start

Requires Python 3.9 or newer. Create an isolated environment, activate it, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run your default webcam:

```powershell
python main.py --source 0
```

Run a video and save the annotated output:

```powershell
python main.py --source .\input.mp4 --save .\outputs\tracked.mp4
```

For a server or batch job, suppress the display window:

```powershell
python main.py --source .\input.mp4 --save .\outputs\tracked.mp4 --no-display
```

Download and process the demonstration video:

```powershell
python main.py --download-sample
```

The first model run downloads the selected YOLO weight file if it is not already available locally.

## Options

| Option | Description |
| --- | --- |
| `--source` | Webcam index (for example, `0`) or video path. |
| `--model` | YOLO weight file, such as `yolov8n.pt` or `yolov8s.pt`. |
| `--conf` | Detection confidence threshold; default `0.35`. |
| `--iou` | SORT association IoU threshold; default `0.3`. |
| `--max-age` | Frames to retain an unmatched track; default `15`. |
| `--min-hits` | Matches required to confirm a track; default `3`. |
| `--classes` | COCO class IDs to keep, for example `--classes 0 2`. |
| `--save` | Destination for an annotated MP4. Parent folders are created automatically. |
| `--no-display` | Do not create an OpenCV window. |
| `--download-sample` | Download and use the sample pedestrian video. |

## Development

Run the regression tests without downloading model weights:

```powershell
python -m unittest discover -s tests -v
```

To install a command-line entry point during development:

```powershell
python -m pip install -e .
object-track --source 0
```

## Notes

- The tracker uses COCO class IDs exposed by the selected YOLO model. `0` is person and `2` is car for the standard COCO models.
- Downloaded weights, sample videos, local virtual environments, and generated outputs are excluded from Git via `.gitignore`.
- This repository is distributed under the [MIT License](LICENSE).
