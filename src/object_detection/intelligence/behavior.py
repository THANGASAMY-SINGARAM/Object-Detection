"""Configurable, explainable rule-based behavior classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set

from .trajectory import Motion


@dataclass(frozen=True)
class Behavior:
    name: str
    score: float
    reason: str


class BehaviorAnalyzer:
    """Labels motion only when a defined rule threshold is crossed."""

    def __init__(
        self,
        stopped_speed: float = 8.0,
        loiter_seconds: float = 15.0,
        direction_change_degrees: float = 120.0,
        acceleration_ratio: float = 2.5,
    ) -> None:
        self.stopped_speed = stopped_speed
        self.loiter_seconds = loiter_seconds
        self.direction_change_degrees = direction_change_degrees
        self.acceleration_ratio = acceleration_ratio
        self._first_seen: Dict[int, float] = {}
        self._previous_speed: Dict[int, float] = {}
        self._emitted: Dict[int, Set[str]] = {}

    def analyze(self, motion: Motion, timestamp: float, active_zone_kinds: Iterable[str]) -> List[Behavior]:
        track_id = motion.track_id
        first_seen = self._first_seen.setdefault(track_id, timestamp)
        duration = timestamp - first_seen
        previous_speed = self._previous_speed.get(track_id, motion.speed_px_s)
        self._previous_speed[track_id] = motion.speed_px_s
        emitted = self._emitted.setdefault(track_id, set())
        behaviors: List[Behavior] = []

        def add_once(name: str, score: float, reason: str) -> None:
            if name not in emitted:
                emitted.add(name)
                behaviors.append(Behavior(name, score, reason))

        if motion.previous_centroid is not None and motion.speed_px_s <= self.stopped_speed:
            add_once("Stopped object", 0.55, f"speed stayed below {self.stopped_speed:.1f} px/s")
        if duration >= self.loiter_seconds and motion.speed_px_s <= self.stopped_speed * 1.5:
            add_once("Loitering", min(duration / max(self.loiter_seconds * 2, 1), 1.0), f"dwell time reached {duration:.1f}s")
        if motion.direction_change_degrees >= self.direction_change_degrees:
            add_once("Sudden direction change", min(motion.direction_change_degrees / 180.0, 1.0), f"direction changed {motion.direction_change_degrees:.0f}°")
        if previous_speed > self.stopped_speed and motion.speed_px_s / previous_speed >= self.acceleration_ratio:
            add_once("Sudden acceleration", min(motion.speed_px_s / (previous_speed * self.acceleration_ratio), 1.0), "speed increased sharply")
        if motion.speed_px_s > self.stopped_speed and previous_speed / motion.speed_px_s >= self.acceleration_ratio:
            add_once("Sudden deceleration", min(previous_speed / (motion.speed_px_s * self.acceleration_ratio), 1.0), "speed decreased sharply")
        if not behaviors and motion.previous_centroid is not None:
            behaviors.append(Behavior("Normal movement", 0.0, "no configured behavior threshold was crossed"))
        return behaviors
