import io
import json
from pathlib import Path
import random
import tempfile
import unittest
import uuid
import shutil
from unittest.mock import patch

from PIL import Image, PngImagePlugin

from adapters.export_reviewed_detection_dataset import canonical, digest, freeze, readiness, write_json
from vision.training_pipeline import read_dataset, score_frames


class ReviewedDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[2] / "experiments" / "tmp-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.root = temporary_root / ("reviewed-" + uuid.uuid4().hex)
        self.root.mkdir()
        (self.root / "staging").mkdir()
        self.rows = []

    def tearDown(self):
        assert self.root.parent == Path(__file__).resolve().parents[2] / "experiments" / "tmp-tests"
        shutil.rmtree(self.root)

    def row(self, name, seed=1, group=None, evaluation=False, boxes=None, raw=None):
        if raw is None:
            rng = random.Random(seed)
            image = Image.frombytes("RGB", (64, 64), rng.randbytes(64*64*3))
            stream = io.BytesIO(); image.save(stream, "PNG"); raw = stream.getvalue()
        path = self.root / "staging" / (name + ".png")
        path.write_bytes(raw)
        boxes = boxes if boxes is not None else [{"bboxPixelsXYXY": [1, 2, 10, 12], "groupId": "unknown"}]
        row = {"id": name, "stagedPath": path.name, "originalSha256": digest(raw), "width": 64, "height": 64,
               "complete": True, "humanVerified": True, "reviewer": "Test fixture", "status": "visible" if boxes else "absent",
               "trainingAllowed": not evaluation, "evaluationOnly": evaluation, "boxes": boxes, "groupKeys": [group or name]}
        self.rows.append(row)
        return row

    def export(self, **extras):
        snapshot = {"schema": "mycoscout.reviewed-detection-snapshot.v1", "createdAt": "2026-10-28T00:00:00Z", "records": self.rows, **extras}
        write_json(self.root / "staging" / "snapshot.json", snapshot)
        result = freeze(self.root / "staging" / "snapshot.json", self.root / "dataset")
        return result, read_dataset(self.root / "dataset", require_ready=False)

    def test_same_video_never_crosses_partitions_and_eval_promotes_whole_group(self):
        self.row("a", group="video:one")
        self.row("b", seed=2, group="video:one", evaluation=True)
        _, data = self.export()
        self.assertEqual({r["split"] for r in data["records"]}, {"test"})
        self.assertEqual(len({r["leakageGroup"] for r in data["records"]}), 1)

    def test_decoded_duplicate_evaluation_source_cannot_train_even_if_unreviewed(self):
        original = self.row("a")
        source = Image.open(self.root / "staging" / "a.png")
        metadata = PngImagePlugin.PngInfo(); metadata.add_text("different", "metadata")
        stream = io.BytesIO(); source.save(stream, "PNG", pnginfo=metadata)
        other = stream.getvalue(); (self.root / "staging" / "eval.png").write_bytes(other)
        self.assertNotEqual(digest(other), original["originalSha256"])
        _, data = self.export(evaluationSources=[{"stagedPath": "eval.png", "originalSha256": digest(other)}])
        self.assertEqual(data["records"][0]["split"], "test")

    def test_conflicting_duplicate_boxes_are_excluded(self):
        self.row("a"); self.row("b", boxes=[{"bboxPixelsXYXY": [30, 30, 40, 40]}])
        result, data = self.export()
        self.assertEqual(result["images"], 0)
        self.assertEqual(len(data["excludedConflictingImages"]), 1)

    def test_identical_duplicates_collapse(self):
        self.row("a"); self.row("b")
        result, _ = self.export()
        self.assertEqual(result["images"], 1)
        self.assertEqual(result["duplicatesCollapsed"], 1)

    def test_prior_validation_group_never_becomes_training(self):
        self.row("a", group="session:site")
        _, data = self.export(frozenPartitions={"session:site": "validation"})
        self.assertEqual(data["records"][0]["split"], "validation")

    def test_readiness_requires_actual_independent_data(self):
        self.row("a")
        result, _ = self.export()
        self.assertFalse(result["readiness"]["ready"])
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            read_dataset(self.root / "dataset")

    def test_training_permission_missing_rejected(self):
        self.row("a")["trainingAllowed"] = False
        with self.assertRaisesRegex(ValueError, "permission"):
            self.export()

    def test_tampered_image_rejected_before_training(self):
        self.row("a"); _, data = self.export()
        (self.root / "dataset" / data["records"][0]["imagePath"]).write_bytes(b"altered")
        with self.assertRaisesRegex(ValueError, "image changed"):
            read_dataset(self.root / "dataset", require_ready=False)

    def test_path_escape_rejected(self):
        self.row("a")["stagedPath"] = "../outside.png"
        (self.root / "outside.png").write_bytes((self.root / "staging" / "a.png").read_bytes())
        with self.assertRaisesRegex(ValueError, "escaped"):
            self.export()

    def test_empty_negatives_are_labels_and_do_not_invent_positives(self):
        self.row("a", boxes=[]); _, data = self.export()
        self.assertEqual(sum(data["readiness"]["positiveImages"].values()), 0)


class DetectionMetricsTests(unittest.TestCase):
    def test_one_detection_cannot_match_two_mushrooms(self):
        score = score_frames([{"truth": [[1, 1, 4, 4], [1, 1, 4, 4]], "predictions": [{"bbox": [1, 1, 4, 4], "score": .9}]}])
        self.assertEqual(score["truePositives"], 1)
        self.assertEqual(score["falseNegatives"], 1)
        self.assertEqual(score["pixelWidthBins"]["under4"], {"matched": 1, "total": 2, "recall": .5})

    def test_empty_denominators_are_unknown(self):
        score = score_frames([{"truth": [], "predictions": []}])
        self.assertIsNone(score["recall"])
        self.assertIsNone(score["precision"])
        self.assertIsNone(score["pixelWidthBins"]["under4"]["recall"])

    def test_background_false_positives_count(self):
        score = score_frames([{"truth": [], "predictions": [{"bbox": [1, 1, 4, 4], "score": .9}]}])
        self.assertEqual(score["falsePositives"], 1)
        self.assertEqual(score["falsePositivesPerFrame"], 1)


class ActivatedDetectorThresholdTests(unittest.TestCase):
    """An activated candidate must use the operating point evaluated at training."""
    def setUp(self):
        temporary_root = Path(__file__).resolve().parents[2] / "experiments" / "tmp-tests"
        temporary_root.mkdir(parents=True, exist_ok=True)
        self.root = temporary_root / ("active-threshold-" + uuid.uuid4().hex)
        self.root.mkdir()
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"synthetic-checkpoint-placeholder-never-loaded")
        self.pointer = self.root / "active-model.json"
        write_json(self.pointer, {"checkpoint": str(checkpoint), "checkpointSha256": digest(checkpoint.read_bytes()), "jobId": "synthetic-test"})

    def tearDown(self):
        assert self.root.parent == Path(__file__).resolve().parents[2] / "experiments" / "tmp-tests"
        shutil.rmtree(self.root)

    def model(self):
        import torch
        class Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

            def forward(self, images):
                return [{"boxes": torch.tensor([[1., 1., 5., 5.], [6., 6., 10., 10.], [11., 11., 20., 20.], [21., 21., 28., 28.]]),
                         "scores": torch.tensor([.2, .5, .8, .95]), "labels": torch.tensor([1, 1, 1, 0])}]
        return Stub()

    def test_activated_detector_honors_saved_threshold_and_filters_background(self):
        from adapters.live_capture import ReviewedDetector
        with patch("vision.training_pipeline.load_detector", return_value=(self.model(), {"scoreThreshold": .72})):
            detector = ReviewedDetector(self.pointer)
            detections = detector.predict(Image.new("RGB", (32, 32)))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bboxPixelsXYXY"], [11., 11., 20., 20.])

    def test_activated_detector_preserves_a_lower_evaluated_threshold(self):
        from adapters.live_capture import ReviewedDetector
        with patch("vision.training_pipeline.load_detector", return_value=(self.model(), {"scoreThreshold": .1})):
            detector = ReviewedDetector(self.pointer)
            detections = detector.predict(Image.new("RGB", (32, 32)))
        self.assertEqual(len(detections), 3)


if __name__ == "__main__":
    unittest.main()
