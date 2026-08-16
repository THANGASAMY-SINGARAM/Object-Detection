"""Orchestrates rule-based video intelligence for each tracked frame."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import atan2, degrees, hypot
from typing import Dict, Iterable, List, Tuple

from .anomaly import AnomalyAssessment, AnomalyDetector
from .behavior import Behavior, BehaviorAnalyzer
from .crowd import CrowdAnalyzer, CrowdStats
from .events import Event, EventLogger
from .trajectory import Motion, TrajectoryAnalyzer
from .zones import Zone, ZoneManager


@dataclass(frozen=True)
class TrackInsight:
    track_id: int
    class_id: int
    motion: Motion
    behaviors: Tuple[Behavior, ...]
    anomaly: AnomalyAssessment
    risk_score: float
    risk_message: str


class VideoIntelligenceEngine:
    """Stateful, explainable analytics designed to receive SORT track arrays."""

    def __init__(self, zones: Iterable[Zone] = (), trajectory_history: int = 60) -> None:
        self.trajectory = TrajectoryAnalyzer(trajectory_history)
        self.zones = ZoneManager(zones)
        self.behavior = BehaviorAnalyzer()
        self.anomaly = AnomalyDetector()
        self.crowd = CrowdAnalyzer()
        self.events = EventLogger()
        self._zone_entered_at: Dict[Tuple[int, str], float] = {}
        self.class_counts: Counter = Counter()
        self.object_classes: Dict[int, int] = {}
        self.unique_ids: set[int] = set()
        self.max_speed = 0.0
        self.speed_total = 0.0
        self.speed_samples = 0
        self.zone_activity: Counter = Counter()
        self.highest_risk: TrackInsight | None = None
        self._previous_crowd_growth = False
        self.last_crowd_stats: CrowdStats | None = None

    def update(self, tracks, timestamp: float, frame_size: Tuple[int, int]) -> Tuple[List[TrackInsight], CrowdStats]:
        height, width = frame_size
        insights: List[TrackInsight] = []
        person_motions: List[Motion] = []
        active_ids: set[int] = set()
        for x1, y1, x2, y2, track_id_raw, class_id_raw in tracks:
            track_id, class_id = int(track_id_raw), int(class_id_raw)
            active_ids.add(track_id)
            self.unique_ids.add(track_id)
            self.class_counts[class_id] += 1
            self.object_classes[track_id] = class_id
            centroid = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            motion = self.trajectory.update(track_id, centroid, timestamp)
            if motion.speed_px_s > 0:
                self.max_speed = max(self.max_speed, motion.speed_px_s)
                self.speed_total += motion.speed_px_s
                self.speed_samples += 1
            if class_id == 0:
                person_motions.append(motion)

            normalized = (centroid[0] / max(width, 1), centroid[1] / max(height, 1))
            transitions = self.zones.update(track_id, normalized)
            for transition in transitions:
                self.zone_activity[transition.zone.name] += 1
                key = (track_id, transition.zone.name)
                if transition.action == "entered":
                    self._zone_entered_at[key] = timestamp
                else:
                    self._zone_entered_at.pop(key, None)
                self.events.add(Event(timestamp, f"zone_{transition.action}", f"Object #{track_id} {transition.action} {transition.zone.name}", track_id, "warning" if transition.zone.kind == "restricted" else "info"))

            active_zone_names = self.zones.active_zones(track_id)
            active_zones = [zone for zone in self.zones.zones if zone.name in active_zone_names]
            restricted = any(zone.kind == "restricted" for zone in active_zones)
            dwell = max((timestamp - self._zone_entered_at.get((track_id, zone.name), timestamp) for zone in active_zones), default=0.0)
            behaviors = tuple(self.behavior.analyze(motion, timestamp, [zone.kind for zone in active_zones]))
            for behavior in behaviors:
                if behavior.name != "Normal movement":
                    self.events.add(Event(timestamp, "behavior", f"Object #{track_id}: {behavior.name} ({behavior.reason})", track_id, "warning", {"score": behavior.score}))
            # Crowd growth is evaluated once after the full frame is collected.
            anomaly = self.anomaly.assess(motion, behaviors, restricted, dwell, self._previous_crowd_growth)
            risk_score, risk_message = self._predict_risk(motion, normalized, active_zones)
            insight = TrackInsight(track_id, class_id, motion, behaviors, anomaly, risk_score, risk_message)
            insights.append(insight)
            if anomaly.score >= 0.35 or risk_score >= 0.45:
                severity = "critical" if max(anomaly.score, risk_score) >= 0.70 else "warning"
                self.events.add(Event(timestamp, "intelligence_alert", f"Object #{track_id}: {anomaly.reason if anomaly.score >= risk_score else risk_message}", track_id, severity))
            if self.highest_risk is None or max(insight.risk_score, insight.anomaly.score) > max(self.highest_risk.risk_score, self.highest_risk.anomaly.score):
                self.highest_risk = insight
        self.trajectory.forget(active_ids)
        crowd_stats = self.crowd.update(person_motions, width, height)
        self.last_crowd_stats = crowd_stats
        self._previous_crowd_growth = crowd_stats.sudden_growth
        return insights, crowd_stats

    def _predict_risk(self, motion: Motion, normalized: Tuple[float, float], active_zones: List[Zone]) -> Tuple[float, str]:
        """Trajectory-based proximity warning; this is not a learned prediction."""
        restricted = [zone for zone in self.zones.zones if zone.kind == "restricted" and zone not in active_zones]
        if motion.direction_degrees is None or motion.speed_px_s <= 1 or not restricted:
            return 0.0, "No trajectory-based risk prediction is available yet."
        nearest = min(restricted, key=lambda zone: hypot(zone.center[0] - normalized[0], zone.center[1] - normalized[1]))
        distance = hypot(nearest.center[0] - normalized[0], nearest.center[1] - normalized[1])
        bearing = (degrees(atan2(nearest.center[1] - normalized[1], nearest.center[0] - normalized[0])) + 360.0) % 360.0
        difference = abs(bearing - motion.direction_degrees)
        heading_error = min(difference, 360.0 - difference)
        if heading_error > 60.0:
            return 0.0, "Current trajectory is moving away from restricted zones."
        heading_factor = 1.0 - heading_error / 60.0
        score = max(0.0, min(1.0, 1.0 - distance / 0.75)) * min(motion.speed_px_s / 120.0, 1.0) * heading_factor
        if score < 0.45:
            return round(score, 2), "Current trajectory does not indicate a material restricted-zone risk."
        return round(score, 2), f"Trajectory-based prediction: object is moving toward restricted zone {nearest.name}."

    def summary(self) -> Dict[str, object]:
        return {
            "unique_objects": len(self.unique_ids),
            "objects_by_class_id": dict(Counter(self.object_classes.values())),
            "average_speed_px_s": round(self.speed_total / max(self.speed_samples, 1), 2),
            "maximum_speed_px_s": round(self.max_speed, 2),
            "most_active_zone": self.zone_activity.most_common(1)[0][0] if self.zone_activity else None,
            "event_count": len(self.events.events),
            "anomaly_count": sum(event.event_type == "intelligence_alert" for event in self.events.events),
            "crowd": {
                "people_count": self.last_crowd_stats.people_count,
                "density_per_100k_px": self.last_crowd_stats.density_per_100k_px,
                "dominant_direction": self.last_crowd_stats.dominant_direction,
                "status": self.last_crowd_stats.status,
            } if self.last_crowd_stats else {},
        }
