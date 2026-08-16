"""Explainable anomaly score built from deterministic behavior signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .behavior import Behavior
from .trajectory import Motion


@dataclass(frozen=True)
class AnomalyAssessment:
    score: float
    status: str
    reason: str


class AnomalyDetector:
    def __init__(self, normal_speed: float = 120.0, restricted_dwell_seconds: float = 8.0) -> None:
        self.normal_speed = normal_speed
        self.restricted_dwell_seconds = restricted_dwell_seconds

    def assess(self, motion: Motion, behaviors: Iterable[Behavior], in_restricted_zone: bool, dwell_seconds: float, crowd_growth: bool) -> AnomalyAssessment:
        signals: list[tuple[float, str]] = []
        if motion.speed_px_s > self.normal_speed:
            signals.append((min(motion.speed_px_s / (self.normal_speed * 2), 1.0) * 0.35, "speed exceeds the configured normal range"))
        if motion.direction_change_degrees >= 120:
            signals.append((0.20, "movement direction changed abruptly"))
        if in_restricted_zone and dwell_seconds >= self.restricted_dwell_seconds:
            signals.append((0.45, "object remained in a restricted zone beyond the configured dwell time"))
        if crowd_growth:
            signals.append((0.15, "person count increased sharply"))
        for behavior in behaviors:
            if behavior.name in {"Loitering", "Stopped object", "Sudden acceleration", "Sudden deceleration"}:
                signals.append((0.15 * behavior.score if behavior.score else 0.08, behavior.reason))
        score = min(sum(value for value, _ in signals), 1.0)
        status = "HIGH" if score >= 0.70 else "MEDIUM" if score >= 0.35 else "LOW"
        reason = "; ".join(reason for _, reason in signals[:2]) or "no anomaly rule threshold was crossed"
        return AnomalyAssessment(round(score, 2), status, reason)
