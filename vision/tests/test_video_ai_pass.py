from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adapters.tests.test_merge_video_collections import fixture, save
from vision.video_ai_pass import run


class Detector:
    info = {"model": "offline fixture", "weights": "fixed", "threads": 4}

    def __init__(self, failure=False):
        self.calls = 0
        self.failure = failure

    def predict(self, image):
        self.calls += 1
        if self.failure:
            raise RuntimeError("fixture failure")
        return [{"bboxPixelsXYXY": [2, 2, 10, 10], "confidence": .8}]


class VideoAIPassTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="video-ai-pass-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.main, self.research = self.root / "main", self.root / "research"
        self.main_data = fixture(self.main, "main-video", "brown")
        self.research_data = fixture(self.research, "research-video", "green")
        self.research_data["sources"][0].update(licenseAllowsTraining=False, evaluationOnly=True,
                                                licenseUrl="http://creativecommons.org/licenses/by-nc-sa/2.5/")
        self.research_data["frames"][0]["evaluationOnly"] = True
        save(self.research, self.research_data)
        self.inputs = [("main", self.main), ("research", self.research)]
        self.output = self.root / "ai-output"

    def test_each_frame_gets_original_coordinates_provenance_and_unverified_proposals(self):
        detector = Detector()
        with patch("builtins.print"):
            report = run(self.inputs, self.output, detector)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(report["completedFrames"], 2)
        self.assertEqual(report["sourceVideos"], 2)
        for record in report["frames"]:
            self.assertFalse(record["humanVerified"])
            self.assertFalse(record["trainingAllowed"])
            self.assertEqual(record["boxes"][0]["bbox"], [2, 2, 10, 10])
            self.assertFalse(record["boxes"][0]["humanVerified"])
            self.assertIsNone(record["humanMatches"])
        self.assertTrue(report["frames"][1]["evaluationOnly"])
        self.assertFalse(report["frames"][1]["sourceLicenseAllowsTraining"])
        self.assertFalse(report["frames"][0]["evaluationOnly"])

    def test_resume_uses_immutable_results_without_repeating_inference(self):
        detector = Detector()
        with patch("builtins.print"):
            first = run(self.inputs, self.output, detector)
            second = run(self.inputs, self.output, detector, resume=True)
        self.assertEqual(detector.calls, 2)
        self.assertEqual(first["frames"], second["frames"])

    def test_resume_rejects_changed_manifest_and_changed_model(self):
        with patch("builtins.print"):
            run(self.inputs, self.output, Detector())
        changed = Detector()
        changed.info = {**changed.info, "weights": "changed"}
        with self.assertRaisesRegex(ValueError, "source/model identity"):
            run(self.inputs, self.output, changed, resume=True)
        self.main_data["createdAt"] = "different manifest"
        save(self.main, self.main_data)
        with self.assertRaisesRegex(ValueError, "Resume inputs changed"):
            run(self.inputs, self.output, Detector(), resume=True)

    def test_inference_failure_is_reported_and_never_becomes_negative_ground_truth(self):
        with patch("builtins.print"):
            report = run(self.inputs, self.output, Detector(failure=True))
        self.assertEqual(report["status"], "completed_with_errors")
        self.assertEqual(report["completedFrames"], 0)
        self.assertEqual(report["failedFrames"], 2)
        self.assertTrue(all(record["status"] == "failed" for record in report["frames"]))
        self.assertTrue(all(record["trainingAllowed"] is False for record in report["frames"]))

    def test_input_media_and_manifests_are_not_rewritten(self):
        originals = {path: path.read_bytes() for folder in (self.main, self.research) for path in folder.rglob("*") if path.is_file()}
        with patch("builtins.print"):
            run(self.inputs, self.output, Detector())
        self.assertEqual(originals, {path: path.read_bytes() for path in originals})


if __name__ == "__main__":
    unittest.main()
