"""Tests for deterministic detection configuration helpers."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_detection.detection import DetectionConfig, _class_aware_nms


class DetectionTests(unittest.TestCase):
    def test_configuration_keeps_requested_accuracy_settings(self) -> None:
        config = DetectionConfig(imgsz=1280, max_det=500, tiled_inference=True)
        self.assertEqual(config.imgsz, 1280)
        self.assertTrue(config.tiled_inference)

    def test_nms_removes_same_class_duplicate_but_keeps_other_class(self) -> None:
        detections = np.array([
            [0, 0, 20, 20, 0.9, 2],
            [1, 1, 21, 21, 0.8, 2],
            [1, 1, 21, 21, 0.8, 0],
        ], dtype=np.float32)
        result = _class_aware_nms(detections, 0.5, 10)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
