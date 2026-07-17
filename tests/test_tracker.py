"""Focused regression tests for the tracker primitives."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_detection.tracker import Sort, iou


class TrackerTests(unittest.TestCase):
    def test_iou_for_overlapping_boxes(self) -> None:
        self.assertAlmostEqual(iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_tracker_keeps_id_for_same_class(self) -> None:
        tracker = Sort(min_hits=1, iou_threshold=0.1)
        first = tracker.update(np.array([[0, 0, 10, 10, 0.9, 0]]))
        second = tracker.update(np.array([[1, 0, 11, 10, 0.9, 0]]))

        self.assertEqual(first.shape, (1, 6))
        self.assertEqual(first[0, 4], second[0, 4])

    def test_tracker_does_not_match_different_classes(self) -> None:
        tracker = Sort(min_hits=1, iou_threshold=0.1)
        tracker.update(np.array([[0, 0, 10, 10, 0.9, 0]]))
        second = tracker.update(np.array([[0, 0, 10, 10, 0.9, 2]]))

        self.assertEqual(second[0, 5], 2)


if __name__ == "__main__":
    unittest.main()
