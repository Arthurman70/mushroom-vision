from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

from vision.curriculum import make_curriculum
from vision.dataset import SCHEMA, read_json, sha256, split_groups, validate, write_json
from vision.metrics import evaluate, iou, load_predictions
from vision.verifier import features, samples, score_image, train


def record(index, **kwargs):
    row = {"id": f"img-{index}", "path": f"img-{index}.png", "sha256": f"{index:064x}",
           "originalSha256": f"{index:064x}", "groupId": f"post-{index}", "sourceUrl": f"fixture:img-{index}",
           "context": "closeup", "synthetic": False, "width": 100, "height": 100,
           "annotationStatus": "complete", "reviewer": "automated test fixture only",
           "annotations": [{"label": "mushroom", "bboxPixelsXYXY": [10, 10, 30, 30]}]}
    row.update(kwargs)
    return row


def prediction(*boxes):
    return {"status": "ok", "candidates": [{"confidence": score, "bboxPixelsXYXY": box} for score, box in boxes]}


class DatasetTests(unittest.TestCase):
    def test_original_and_post_connected_components_never_leak(self):
        data = {"schema": SCHEMA, "images": [record(i) for i in range(12)]}
        data["images"][1]["groupId"] = "post-0"
        data["images"][2]["originalSha256"] = data["images"][1]["originalSha256"]
        data["images"][2]["synthetic"] = True
        first = split_groups(deepcopy(data))
        second = split_groups(deepcopy(data))
        self.assertEqual(first, second)
        self.assertEqual(len(set(r["split"] for r in first["images"][:3])), 1)
        self.assertEqual(set(r["split"] for r in first["images"]), {"train", "validation", "test"})

    def test_too_few_groups_rejected(self):
        with self.assertRaisesRegex(ValueError, "three independent"):
            split_groups({"images": [record(1), record(2)]})

    def test_hash_size_and_leakage_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (100, 100)).save(root / "img-1.png")
            row = record(1)
            row["sha256"] = row["originalSha256"] = sha256(root / row["path"])
            path = root / "manifest.json"
            write_json(path, {"schema": SCHEMA, "images": [row]})
            self.assertTrue(validate(path)["valid"])
            duplicate = deepcopy(row)
            duplicate.update(id="duplicate", split="test")
            row["split"] = "train"
            write_json(path, {"schema": SCHEMA, "images": [row, duplicate]})
            self.assertTrue(any("crosses splits" in e for e in validate(path)["errors"]))
            row["sha256"] = "a" * 64
            write_json(path, {"schema": SCHEMA, "images": [row]})
            self.assertIn("img-1: file hash mismatch", validate(path)["errors"])

    def test_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            write_json(path, {"schema": SCHEMA, "images": [record(1, path="../outside.png")]})
            self.assertTrue(any("escapes" in e for e in validate(path)["errors"]))


class MetricsTests(unittest.TestCase):
    def test_duplicate_detections_are_false_positives_and_negatives_count(self):
        data = {"images": [record(1), record(2, annotations=[])]}
        preds = {"img-1": prediction((0.9, [10, 10, 30, 30]), (0.8, [10, 10, 30, 30])),
                 "img-2": prediction((0.7, [50, 50, 70, 70]))}
        result = evaluate(data, preds)
        self.assertEqual(result["real"]["truePositive"], 1)
        self.assertEqual(result["real"]["falsePositive"], 2)
        self.assertEqual(result["real"]["negativeImages"], 1)
        self.assertAlmostEqual(result["real"]["precision"], 1 / 3)

    def test_missing_and_failed_inference_not_scored_as_negative(self):
        data = {"images": [record(1), record(2), record(3)]}
        result = evaluate(data, {"img-1": prediction(), "img-2": {"status": "failed"}})
        self.assertEqual(result["real"]["falseNegative"], 1)
        self.assertEqual(result["skipped"], {"unreviewed": 0, "partial": 0, "missingInference": 1, "failedInference": 1})

    def test_nms_is_explicit_and_retains_raw_count(self):
        data = {"images": [record(1)]}
        predictions = {"img-1": prediction((0.9, [10, 10, 30, 30]), (0.8, [10, 10, 30, 30]))}
        result = evaluate(data, predictions, nms_iou=0.5)
        self.assertEqual(result["real"]["falsePositive"], 0)
        self.assertEqual(result["images"][0]["rawCandidateCount"], 2)
        self.assertEqual(result["images"][0]["candidateCount"], 1)
        self.assertEqual(result["postprocessing"]["classAgnosticNmsIou"], 0.5)

    def test_unreviewed_no_bogus_metrics(self):
        result = evaluate({"images": [record(1, annotationStatus="unreviewed")]}, {"img-1": prediction()})
        self.assertEqual(result["status"], "labels-or-inference-required")
        self.assertIsNone(result["real"]["precision"])
        self.assertIsNone(result["images"][0]["metrics"])

    def test_partial_annotations_never_create_detection_metrics(self):
        result = evaluate({"images": [record(1, annotationStatus="partial")]},
                          {"img-1": prediction((0.9, [10, 10, 30, 30]))})
        self.assertEqual(result["skipped"]["partial"], 1)
        self.assertEqual(result["evaluatedImages"], 0)
        self.assertIsNone(result["images"][0]["metrics"])

    def test_scale_and_real_context_separate(self):
        result = evaluate({"images": [record(1, context="far"), record(2, synthetic=True)]},
                          {"img-1": prediction((0.9, [10, 10, 30, 30])), "img-2": prediction()})
        self.assertEqual(result["real"]["recall"], 1)
        self.assertEqual(result["synthetic"]["recall"], 0)
        self.assertEqual(set(result["byContext"]), {"real:far", "synthetic:closeup"})
        self.assertEqual(result["byObjectPixelSize"]["16to31px"]["groundTruth"], 2)

    def test_iou_and_invalid_predictions(self):
        self.assertEqual(iou([0, 0, 10, 10], [20, 20, 30, 30]), 0)
        with self.assertRaises(ValueError):
            evaluate({"images": [record(1)]}, {"img-1": prediction((float("nan"), [10, 10, 30, 30]))})
        with self.assertRaises(ValueError):
            evaluate({"images": [record(1)]}, {"img-1": prediction((0.9, [0, 0, 101, 30]))})

    def test_missing_candidates_never_masquerades_as_empty_inference(self):
        with self.assertRaisesRegex(ValueError, "explicit candidates"):
            evaluate({"images": [record(1)]}, {"img-1": {"status": "ok"}})

    def test_cannot_evaluate_below_detector_export_floor(self):
        with self.assertRaisesRegex(ValueError, "below the detector export"):
            evaluate({"images": [record(1)]}, {"img-1": prediction()}, {"boxThreshold": 0.4}, confidence=0.25)

    def test_mixed_detector_configs_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [record(1), record(2)]
            for index, row in enumerate(rows):
                write_json(root / f"pred-{index}.json", {"schema": "mycoscout.unlocated-candidates.v1",
                           "image": {"sha256": row["sha256"]}, "candidates": [], "detector": {"model": f"different-{index}"}})
            with self.assertRaisesRegex(ValueError, "different detector configurations"):
                load_predictions(root, rows)

    def test_groundingdino_output_matches_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prediction.json"
            row = record(1)
            write_json(path, {"schema": "mycoscout.unlocated-candidates.v1", "image": {"sha256": row["sha256"]},
                              "candidates": [], "detector": {"model": "fixture"}})
            predictions, detector = load_predictions(path, [row])
            self.assertIn("img-1", predictions)
            self.assertEqual(detector["model"], "fixture")


class CurriculumTests(unittest.TestCase):
    def test_transformed_boxes_dimensions_and_original_group_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (100, 100), "red").save(root / "img-1.png")
            row = record(1, split="test")
            row["sha256"] = row["originalSha256"] = sha256(root / row["path"])
            data = {"schema": SCHEMA, "images": [row]}
            result = make_curriculum(data, root / "manifest.json", root / "stress", [0.25])
            variant = result["images"][0]
            self.assertEqual(variant["annotations"][0]["bboxPixelsXYXY"], [39.5, 39.5, 44.5, 44.5])
            self.assertEqual(variant["width"], 100)
            self.assertEqual(variant["transform"]["contentWidth"], 25)
            self.assertEqual(variant["split"], "test")
            self.assertEqual(variant["originalSha256"], row["sha256"])
            self.assertTrue(variant["synthetic"])
            self.assertIsNone(variant["distanceM"])
            self.assertTrue(validate(root / "stress" / "manifest.json")["valid"])

    def test_downsample_mode_separately_labels_resolution_degradation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (100, 100), "green").save(root / "img-1.png")
            row = record(1)
            row["sha256"] = row["originalSha256"] = sha256(root / row["path"])
            result = make_curriculum({"schema": SCHEMA, "images": [row]}, root / "manifest.json", root / "degrade", [0.5], mode="downsample")
            self.assertEqual(result["images"][0]["width"], 50)
            self.assertEqual(result["images"][0]["transform"]["mode"], "downsample")


class VerifierTests(unittest.TestCase):
    def test_insufficient_training_data_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least 6"):
            train({"images": []}, "manifest.json")

    def test_partial_reviewed_crops_are_available_to_verifier(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.new("RGB", (100, 100), "blue").save(root / "img-1.png")
            row = record(1, annotationStatus="partial", split="train")
            collected = samples({"images": [row]}, root / "manifest.json")
            self.assertEqual(len(collected["train"]), 1)

    def test_synthetic_fixture_full_training_and_checkpoint_scoring(self):
        # Generated colors/shapes test optimizer mechanics only. No field performance claim.
        with tempfile.TemporaryDirectory() as tmp:
            root, rows = Path(tmp), []
            for index in range(10):
                image = Image.new("RGB", (100, 100), (10 + index, 80, 20))
                draw = ImageDraw.Draw(image)
                draw.ellipse((10, 10, 40, 40), fill=(230, 30 + index, 20))
                image.save(root / f"img-{index}.png")
                split = "train" if index < 6 else "validation" if index < 8 else "test"
                row = record(index, split=split,
                             annotations=[{"label": "mushroom", "bboxPixelsXYXY": [10, 10, 41, 41]},
                                          {"label": "background", "bboxPixelsXYXY": [50, 50, 90, 90]}])
                row["sha256"] = row["originalSha256"] = sha256(root / row["path"])
                rows.append(row)
            checkpoint, report = train({"schema": SCHEMA, "images": rows}, root / "manifest.json", epochs=100)
            self.assertEqual(report["status"], "trained")
            self.assertEqual(report["sampleCounts"]["test"]["groups"], 2)
            scores = score_image(image, [[10, 10, 41, 41], [50, 50, 90, 90]], checkpoint)
            self.assertGreater(scores[0]["verifierScore"], scores[1]["verifierScore"])
            self.assertTrue(np.all(np.isfinite(features(image))))
            json.dumps(checkpoint, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
