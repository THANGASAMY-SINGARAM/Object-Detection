"""Simple crowd analytics for tracks classified as people."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import atan2, cos, degrees, sin
from typing import Deque, Iterable

from .trajectory import Motion


@dataclass(frozen=True)
class CrowdStats:
    people_count: int
    density_per_100k_px: float
    dominant_direction: str
    status: str
    sudden_growth: bool


class CrowdAnalyzer:
    def __init__(self, growth_window: int = 30, growth_threshold: int = 4) -> None:
        self.counts: Deque[int] = deque(maxlen=growth_window)
        self.growth_threshold = growth_threshold

    def update(self, person_motions: Iterable[Motion], frame_width: int, frame_height: int) -> CrowdStats:
        motions = list(person_motions)
        people_count = len(motions)
        previous = self.counts[0] if self.counts else people_count
        self.counts.append(people_count)
        growth = people_count - previous >= self.growth_threshold
        angles = [motion.direction_degrees for motion in motions if motion.direction_degrees is not None]
        if angles:
            angle = (degrees(atan2(sum(sin(value * 3.14159265 / 180) for value in angles), sum(cos(value * 3.14159265 / 180) for value in angles))) + 360) % 360
            labels = ["right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right"]
            direction = labels[int((angle + 22.5) // 45) % 8]
        else:
            direction = "not enough movement data"
        density = people_count / max(frame_width * frame_height, 1) * 100000
        status = "High traffic" if people_count >= 10 else "Growing" if growth else "Normal"
        return CrowdStats(people_count, round(density, 2), direction, status, growth)
