"""Unit tests for deterministic video-intelligence rules."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_detection.intelligence.anomaly import AnomalyDetector
from object_detection.intelligence.behavior import BehaviorAnalyzer
from object_detection.intelligence.engine import VideoIntelligenceEngine
from object_detection.intelligence.events import Event, EventLogger
from object_detection.intelligence.trajectory import TrajectoryAnalyzer
from object_detection.intelligence.zones import Zone, ZoneManager


class IntelligenceTests(unittest.TestCase):
    def test_trajectory_calculates_speed_and_direction(self) -> None:
        analyzer = TrajectoryAnalyzer(max_history=4)
        analyzer.update(1, (0.0, 0.0), 0.0)
        motion = analyzer.update(1, (30.0, 0.0), 2.0)
        self.assertEqual(motion.speed_px_s, 15.0)
        self.assertEqual(motion.direction_degrees, 0.0)
        self.assertEqual(len(motion.history), 2)

    def test_zone_entry_and_exit(self) -> None:
        manager = ZoneManager([Zone("Restricted", "restricted", 0.2, 0.2, 0.8, 0.8)])
        self.assertEqual(manager.update(2, (0.1, 0.1)), [])
        self.assertEqual(manager.update(2, (0.5, 0.5))[0].action, "entered")
        self.assertEqual(manager.update(2, (0.9, 0.9))[0].action, "left")

    def test_behavior_rules_do_not_label_normal_motion_as_anomaly(self) -> None:
        trajectory = TrajectoryAnalyzer()
        analyzer = BehaviorAnalyzer(loiter_seconds=5.0)
        trajectory.update(3, (0, 0), 0.0)
        motion = trajectory.update(3, (20, 0), 1.0)
        behaviors = analyzer.analyze(motion, 1.0, [])
        self.assertEqual(behaviors[0].name, "Normal movement")

    def test_restricted_dwell_receives_explainable_anomaly_score(self) -> None:
        trajectory = TrajectoryAnalyzer()
        trajectory.update(4, (0, 0), 0.0)
        motion = trajectory.update(4, (0, 0), 10.0)
        assessment = AnomalyDetector(restricted_dwell_seconds=5.0).assess(motion, [], True, 10.0, False)
        self.assertGreaterEqual(assessment.score, 0.45)
        self.assertIn("restricted zone", assessment.reason)

    def test_loitering_rule_uses_dwell_time(self) -> None:
        trajectory = TrajectoryAnalyzer()
        analyzer = BehaviorAnalyzer(loiter_seconds=5.0)
        first = trajectory.update(8, (5, 5), 0.0)
        analyzer.analyze(first, 0.0, [])
        motion = trajectory.update(8, (5, 5), 6.0)
        labels = {behavior.name for behavior in analyzer.analyze(motion, 6.0, [])}
        self.assertIn("Loitering", labels)

    def test_risk_prediction_requires_motion_toward_restricted_zone(self) -> None:
        engine = VideoIntelligenceEngine([Zone("Restricted", "restricted", 0.7, 0.2, 0.9, 0.8)])
        engine.update([[0, 30, 20, 50, 9, 0]], 0.0, (100, 100))
        insights, _ = engine.update([[40, 30, 60, 50, 9, 0]], 1.0, (100, 100))
        self.assertGreater(insights[0].risk_score, 0.0)

    def test_engine_records_zone_timeline_event(self) -> None:
        engine = VideoIntelligenceEngine([Zone("Zone A", "entry", 0.1, 0.1, 0.9, 0.9)])
        engine.update([[20, 20, 40, 40, 1, 0]], 1.0, (100, 100))
        self.assertEqual(engine.events.events[0].event_type, "zone_entered")

    def test_event_logger_filters_object_and_type(self) -> None:
        logger = EventLogger()
        logger.add(Event(1.0, "zone_entered", "entered", 5))
        logger.add(Event(2.0, "zone_left", "left", 6))
        self.assertEqual(len(logger.filtered(track_id=5)), 1)
        self.assertEqual(len(logger.filtered(event_type="zone_left")), 1)


if __name__ == "__main__":
    unittest.main()
