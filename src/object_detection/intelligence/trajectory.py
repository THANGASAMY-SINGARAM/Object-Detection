"""Centroid history, motion direction, and speed calculations."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Deque, Dict, Optional, Tuple


Point = Tuple[float, float]


@dataclass(frozen=True)
class Motion:
    """The latest explainable movement estimate for a tracked object."""

    track_id: int
    centroid: Point
    previous_centroid: Optional[Point]
    speed_px_s: float
    direction_degrees: Optional[float]
    direction_change_degrees: float
    history: Tuple[Point, ...]


class TrajectoryAnalyzer:
    """Maintains bounded centroid histories and derives frame-to-frame motion."""

    def __init__(self, max_history: int = 60) -> None:
        if max_history < 2:
            raise ValueError("max_history must be at least 2")
        self.max_history = max_history
        self._history: Dict[int, Deque[Tuple[float, Point]]] = defaultdict(
            lambda: deque(maxlen=self.max_history)
        )
        self._last_direction: Dict[int, float] = {}

    def update(self, track_id: int, centroid: Point, timestamp: float) -> Motion:
        history = self._history[track_id]
        previous = history[-1] if history else None
        speed, direction, change = 0.0, None, 0.0
        if previous is not None:
            previous_time, previous_centroid = previous
            elapsed = max(timestamp - previous_time, 1e-6)
            dx = centroid[0] - previous_centroid[0]
            dy = centroid[1] - previous_centroid[1]
            distance = hypot(dx, dy)
            speed = distance / elapsed
            if distance > 1e-6:
                direction = (degrees(atan2(dy, dx)) + 360.0) % 360.0
                if track_id in self._last_direction:
                    raw_change = abs(direction - self._last_direction[track_id])
                    change = min(raw_change, 360.0 - raw_change)
                self._last_direction[track_id] = direction
        history.append((timestamp, centroid))
        return Motion(
            track_id=track_id,
            centroid=centroid,
            previous_centroid=previous[1] if previous else None,
            speed_px_s=speed,
            direction_degrees=direction,
            direction_change_degrees=change,
            history=tuple(point for _, point in history),
        )

    def history_for(self, track_id: int) -> Tuple[Point, ...]:
        return tuple(point for _, point in self._history.get(track_id, ()))

    def forget(self, active_ids: set[int]) -> None:
        """Release histories that no longer belong to active tracks."""
        for track_id in list(self._history):
            if track_id not in active_ids:
                self._history.pop(track_id, None)
                self._last_direction.pop(track_id, None)
