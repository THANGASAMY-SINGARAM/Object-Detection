import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_detection.adaptive import AdaptiveInferenceController


class AdaptiveTests(unittest.TestCase):
    def test_dense_scene_requests_more_frequent_detection(self):
        controller = AdaptiveInferenceController(cooldown_frames=0)
        dense = np.array([[0, 0, 20, 20, 0.5, 0]] * 25, dtype=np.float32)
        decision = controller.decide(dense, 25, 0, 30.0)
        self.assertEqual(decision.interval, 1)

    def test_lost_track_forces_detection(self):
        controller = AdaptiveInferenceController()
        decision = controller.decide(None, 1, 1, 30.0)
        self.assertTrue(decision.should_detect)


if __name__ == "__main__":
    unittest.main()
