"""Measured, hysteresis-based scheduling for YOLO refreshes between tracker updates."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

import numpy as np

from .tracker import iou


@dataclass
class AdaptiveDecision:
    complexity: float
    interval: int
    resolution: int
    reason: str
    should_detect: bool


class AdaptiveInferenceController:
    """Uses real detection/tracker signals; it never fabricates performance data."""

    def __init__(self, mode: str = "AUTO", target_fps: int = 30, cooldown_frames: int = 20) -> None:
        self.mode = mode
        self.target_fps = target_fps
        self.cooldown_frames = cooldown_frames
        self.frame_index = 0
        self.interval = 1 if mode == "QUALITY" else 3 if mode == "BALANCED" else 5
        self.resolution = 960 if mode == "QUALITY" else 768 if mode == "BALANCED" else 640
        self._last_change = 0
        self._last_complexity = 0.0

    def decide(self, detections: Optional[np.ndarray], active_tracks: int, lost_tracks: int, current_fps: Optional[float]) -> AdaptiveDecision:
        self.frame_index += 1
        if self.mode != "AUTO":
            interval, resolution = {"QUALITY": (1, 960), "BALANCED": (3, 768), "PERFORMANCE": (5, 640)}.get(self.mode, (1, 640))
            self.interval, self.resolution = interval, resolution
            return AdaptiveDecision(self._last_complexity, interval, resolution, self._reason(), self._should_detect(lost_tracks))
        complexity = self._complexity(detections, active_tracks, lost_tracks)
        if self.frame_index - self._last_change >= self.cooldown_frames or lost_tracks > 0:
            if complexity > 80:
                target_interval, target_resolution, reason = 1, 960, "very dense or unstable scene"
            elif complexity > 60:
                target_interval, target_resolution, reason = 2, 960, "dense scene detected"
            elif complexity > 30:
                target_interval, target_resolution, reason = 3, 768, "moderate scene complexity"
            else:
                target_interval, target_resolution, reason = 5, 640, "stable low-complexity scene"
            if current_fps is not None and current_fps < self.target_fps * 0.75 and complexity < 80:
                target_interval = min(target_interval + 1, 5)
                reason = "maintaining target FPS on a stable scene"
            if (target_interval, target_resolution) != (self.interval, self.resolution):
                self.interval, self.resolution, self._last_change = target_interval, target_resolution, self.frame_index
            self._last_complexity = complexity
        return AdaptiveDecision(complexity, self.interval, self.resolution, self._reason(), self._should_detect(lost_tracks))

    def _should_detect(self, lost_tracks: int) -> bool:
        return lost_tracks > 0 or self.frame_index == 1 or (self.frame_index - 1) % self.interval == 0

    def _reason(self) -> str:
        return "tracking became unreliable" if self.interval == 1 and self._last_complexity < 30 else "adaptive scene-complexity policy"

    @staticmethod
    def _complexity(detections: Optional[np.ndarray], active_tracks: int, lost_tracks: int) -> float:
        if detections is None or len(detections) == 0:
            return min(100.0, lost_tracks * 20.0)
        boxes = detections[:, :4]
        confidences = detections[:, 4]
        overlap = 0.0
        pairs = 0
        for index in range(len(boxes)):
            for other in range(index + 1, len(boxes)):
                overlap += iou(boxes[index], boxes[other])
                pairs += 1
        density = min(len(detections) * 4.0, 45.0)
        overlap_score = min((overlap / max(pairs, 1)) * 100.0, 30.0)
        confidence_score = (1.0 - float(confidences.mean())) * 15.0
        instability = min((lost_tracks * 10 + max(len(detections) - active_tracks, 0) * 2), 20.0)
        return round(min(100.0, density + overlap_score + confidence_score + instability), 1)
