"""Offline regression checks for video scale-test geometry and denominators."""
import copy
import unittest
from unittest.mock import patch

from PIL import Image

from experiments.video_distance_benchmark import (
    CountedDetector, metric_counts, native_summary, transform_frame,
)
from vision.testbench import compare


class Detector:
    def __init__(self):
        self.info = {"model": "fixture", "resolvedRevision": "immutable-fixture", "threads": 8}
        self.calls = 0

    def predict(self, image):
        self.calls += 1
        return []


class VideoDistanceTests(unittest.TestCase):
    def test_subpixel_projection_retains_native_canvas_and_realized_fraction(self):
        image = Image.new("RGB", (1280, 720), "white")
        truth = [{"id": "human", "bbox": [400, 200, 659.85, 500]}]
        before = copy.deepcopy(truth)
        transformed, boxes, geometry = transform_frame(image, truth, .69, 259.85)
        self.assertEqual(transformed.size, image.size)
        self.assertEqual(geometry["contentSize"], [3, 2])
        width = boxes[0]["bbox"][2] - boxes[0]["bbox"][0]
        self.assertAlmostEqual(width, 259.85 * 3 / 1280)
        self.assertLess(width, .69)
        self.assertLess(width, 1)
        self.assertEqual(truth, before)
        self.assertEqual(image.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(transformed.getpixel((0, 0)), (127, 127, 127))

    def test_native_is_unchanged_and_upscale_is_rejected(self):
        image = Image.new("RGB", (128, 72), (9, 70, 23))
        truth = [{"id": "human", "bbox": [10, 10, 30, 40]}]
        result, boxes, geometry = transform_frame(image, truth, None, 20)
        self.assertEqual(result.tobytes(), image.tobytes())
        self.assertEqual(boxes, truth)
        self.assertEqual(geometry["scaleX"], 1)
        with self.assertRaises(ValueError):
            transform_frame(image, truth, 32, 20)

    def test_tiny_whole_scene_prediction_is_not_a_localization_hit(self):
        rows = [{"id": "p", "sourceGroupId": "creator", "kind": "positive",
                 "truth": [{"bbox": [50, 50, 50.69, 50.69]}],
                 "predictions": {"raw": [{"bbox": [0, 0, 100, 100], "score": .9}]}},
                {"id": "n", "sourceGroupId": "creator", "kind": "negative", "truth": [],
                 "predictions": {"raw": [{"bbox": [20, 20, 30, 30], "score": .9}]}}]
        result = metric_counts(rows, "raw")
        self.assertEqual(result, {"frames": 1, "matched": 0, "targets": 1, "falsePositives": 1,
                                  "negativeFrames": 1, "negativeFalsePositives": 1,
                                  "negativeFramesWithDetections": 1})

    def test_partial_positive_evidence_never_reports_precision_or_false_positives(self):
        rows = [{"frameId": "p", "sourceId": "video", "sourceGroupId": "creator",
                 "truth": [{"id": "human", "bbox": [0, 0, 40, 40]}]}]
        prediction = {"boxes": [{"id": "hit", "bbox": [0, 0, 40, 40], "score": .9},
                                {"id": "unlabeled", "bbox": [50, 50, 60, 60], "score": .9}]}
        with patch("experiments.video_distance_benchmark.cached_result", return_value=(prediction, "a"*64)):
            summary, evidence = native_summary(rows, "unused", positive_only=True)
        self.assertEqual((summary["matched"], summary["targets"]), (1, 1))
        self.assertIsNone(summary["falsePositives"])
        self.assertIsNone(summary["precision"])
        self.assertIsNone(evidence[0]["falsePositives"])

    def test_exact_native_cache_is_reused_once_and_tiles_keep_six_logical_forwards(self):
        detector, events = Detector(), []
        cached = {"model": {**detector.info, "threads": 4}, "boxes": []}
        wrapped = CountedDetector(detector, cached, events.append)
        result = compare(Image.new("RGB", (64, 64)), "a"*64,
                         {"methods": ["original", "color", "tiles"], "tileGrid": 2,
                          "scale": 1, "boxThreshold": .25}, wrapped)
        self.assertEqual(events, [True, False, False, False, False, False])
        self.assertEqual(detector.calls, 5)
        self.assertEqual(result["modelPasses"], 6)

    def test_cache_different_model_revision_is_refused_before_prediction(self):
        detector, events = Detector(), []
        cached = {"model": {**detector.info, "resolvedRevision": "other"}, "boxes": []}
        wrapped = CountedDetector(detector, cached, events.append)
        with self.assertRaisesRegex(ValueError, "configuration differs"):
            wrapped.predict(Image.new("RGB", (64, 64)))
        self.assertEqual(events, [])
        self.assertEqual(detector.calls, 0)


if __name__ == "__main__":
    unittest.main()
