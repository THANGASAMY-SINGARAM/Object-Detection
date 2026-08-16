"""Configurable YOLO inference with optional image enhancement and tiled detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

from .tracker import iou


PRESETS = {
    "FAST": ("yolov8n.pt", 640),
    "BALANCED": ("yolov8s.pt", 960),
    "ACCURACY": ("yolov8m.pt", 1280),
}


@dataclass(frozen=True)
class DetectionConfig:
    confidence: float = 0.35
    nms_iou: float = 0.60
    imgsz: int = 640
    max_det: int = 300
    classes: Optional[List[int]] = None
    enhance_contrast: bool = False
    sharpen: bool = False
    tiled_inference: bool = False
    tile_overlap: float = 0.20


def preprocess_image(frame: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """Apply only user-selected, conservative image enhancement."""
    import cv2

    result = frame.copy()
    if config.enhance_contrast:
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        l_channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_channel)
        result = cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
    if config.sharpen:
        result = cv2.filter2D(result, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32))
    return result


def run_detection(model, frame: np.ndarray, config: DetectionConfig) -> np.ndarray:
    """Return validated YOLO detections as [x1, y1, x2, y2, confidence, class_id]."""
    source = preprocess_image(frame, config)
    if config.tiled_inference and min(source.shape[:2]) >= config.imgsz:
        detections = _run_tiled(model, source, config)
    else:
        detections = _from_results(model(source, conf=config.confidence, iou=config.nms_iou, imgsz=config.imgsz, max_det=config.max_det, classes=config.classes, verbose=False))
    return _class_aware_nms(detections, config.nms_iou, config.max_det)


def detection_statistics(detections: np.ndarray, names) -> Dict[str, object]:
    confidences = detections[:, 4] if len(detections) else np.array([], dtype=float)
    counts: Dict[str, int] = {}
    for class_id in detections[:, 5].astype(int) if len(detections) else []:
        name = names.get(class_id, f"Class {class_id}") if isinstance(names, dict) else names[class_id]
        counts[str(name)] = counts.get(str(name), 0) + 1
    return {
        "total": int(len(detections)),
        "counts": counts,
        "average_confidence": float(confidences.mean()) if len(confidences) else None,
        "minimum_confidence": float(confidences.min()) if len(confidences) else None,
        "maximum_confidence": float(confidences.max()) if len(confidences) else None,
    }


def _from_results(results: Iterable) -> np.ndarray:
    values: List[List[float]] = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            values.append([x1, y1, x2, y2, float(box.conf[0].item()), int(box.cls[0].item())])
    return np.asarray(values, dtype=np.float32) if values else np.empty((0, 6), dtype=np.float32)


def _run_tiled(model, image: np.ndarray, config: DetectionConfig) -> np.ndarray:
    height, width = image.shape[:2]
    tile_size = config.imgsz
    stride = max(1, int(tile_size * (1.0 - config.tile_overlap)))
    values: List[np.ndarray] = []
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            tile = image[top:min(top + tile_size, height), left:min(left + tile_size, width)]
            if tile.size == 0:
                continue
            detections = _from_results(model(tile, conf=config.confidence, iou=config.nms_iou, imgsz=config.imgsz, max_det=config.max_det, classes=config.classes, verbose=False))
            if len(detections):
                detections[:, [0, 2]] += left
                detections[:, [1, 3]] += top
                values.append(detections)
    return np.vstack(values) if values else np.empty((0, 6), dtype=np.float32)


def _class_aware_nms(detections: np.ndarray, threshold: float, max_det: int) -> np.ndarray:
    if len(detections) == 0:
        return detections
    selected: List[np.ndarray] = []
    for class_id in np.unique(detections[:, 5].astype(int)):
        candidates = detections[detections[:, 5].astype(int) == class_id]
        candidates = candidates[np.argsort(candidates[:, 4])[::-1]]
        kept: List[np.ndarray] = []
        while len(candidates):
            current = candidates[0]
            kept.append(current)
            candidates = np.asarray([candidate for candidate in candidates[1:] if iou(current[:4], candidate[:4]) < threshold], dtype=np.float32)
            if candidates.size == 0:
                break
        selected.extend(kept)
    result = np.asarray(selected, dtype=np.float32)
    return result[np.argsort(result[:, 4])[::-1]][:max_det]
