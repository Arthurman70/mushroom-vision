"""Checks for split leakage, honest ranking metrics, and validation thresholds."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision.transfer import (SCHEMA, EXPECTED_WEIGHT_SHA, average_precision, auc_roc, choose_threshold,
                             fit_head, head_scores, rank_photos, validate_checkpoint, validate_manifest, write_json)


def fixture():
    rows = []
    for split, count in (("train", 20), ("validation", 10), ("test", 10)):
        for label in ("black_trumpet", "other_fungi"):
            for _ in range(count):
                index = len(rows)
                rows.append({"id": f"photo-{index}", "path": f"images/{index}.jpg",
                             "weakLabel": label, "split": split, "synthetic": False,
                             "groupId": f"observer-{index}", "observerId": index,
                             "observationId": index, "photoId": index,
                             "sha256": f"{index:064x}", "originalSha256": f"{index:064x}",
                             "photoLicense": {"code": "cc0"}, "sourceUrl": f"fixture:{index}"})
    return {"schema": SCHEMA, "images": rows}


class TransferTests(unittest.TestCase):
    def test_perfect_and_reversed_ranking(self):
        labels = np.array([0, 0, 1, 1])
        self.assertEqual(average_precision(labels, np.array([.1, .2, .8, .9])), 1)
        self.assertEqual(auc_roc(labels, np.array([.1, .2, .8, .9])), 1)
        self.assertEqual(auc_roc(labels, np.array([.9, .8, .2, .1])), 0)

    def test_tied_scores_are_not_order_dependent(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.ones(4) * .5
        self.assertEqual(auc_roc(labels, scores), .5)
        self.assertEqual(average_precision(labels, scores), .5)
        self.assertEqual(average_precision(labels[::-1], scores), .5)

    def test_threshold_target_with_insufficient_positive_evidence_is_explicit(self):
        labels = np.array([1, 1, 0, 0])
        _, report = choose_threshold(labels, np.array([.9, .8, .2, .1]))
        self.assertEqual(report["status"], "precision-target-not-met-fallback-to-validation-f1")
        self.assertFalse(report["guaranteesFuturePrecision"])

    def test_threshold_uses_validation_target_and_keeps_low_ranked_positives(self):
        labels = np.array([1] * 12 + [0] * 12)
        threshold, report = choose_threshold(labels, np.r_[np.linspace(.5, .9, 12), np.linspace(.1, .4, 12)])
        self.assertEqual(report["status"], "empirical-validation-target-met")
        self.assertEqual(report["validationAtThreshold"]["recall"], 1)
        self.assertGreaterEqual(report["validationAtThreshold"]["precision"], .8)
        self.assertTrue(.01 <= threshold <= .99)

    def test_independent_split_validation_rejects_observer_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "manifest.json"
            data = fixture()
            write_json(path, data)
            _, counts = validate_manifest(path, verify_files=False)
            self.assertEqual(counts["test"]["photos"], 20)
            data["images"][-1]["observerId"] = data["images"][0]["observerId"]
            write_json(path, data)
            with self.assertRaisesRegex(ValueError, "leakage"):
                validate_manifest(path, verify_files=False)

    def test_train_statistics_do_not_depend_on_future_inference_rows(self):
        vectors = np.array([[-3, 0], [-2, 1], [2, 1], [3, 0]], dtype=float)
        head = fit_head(vectors, np.array([0, 0, 1, 1]), .01)
        self.assertEqual(head["mean"], [0, .5])
        scores = head_scores(np.array([[-4, 0], [4, 0]], dtype=float), head)
        self.assertLess(scores[0], scores[1])
        self.assertEqual(head["mean"], [0, .5])

    def test_ranking_cannot_be_presented_as_species_or_flight_clearance(self):
        checkpoint = {"schema": "mycoscout.black-trumpet-photo-ranker.v2", "featureFamily": "handcrafted",
                      "officialWeightSha256": EXPECTED_WEIGHT_SHA, "probabilityCalibrated": False, "flightGateEnabled": False,
                      "head": {"weights": [0] * 130, "mean": [0] * 130, "std": [1] * 130, "bias": 0, "threshold": .5}}
        result = rank_photos([Image.new("RGB", (30, 30), "green")], checkpoint)[0]
        self.assertFalse(result["speciesConfirmed"])
        self.assertFalse(result["scoreCalibrated"])
        self.assertFalse(result["flightGateEnabled"])
        checkpoint["flightGateEnabled"] = True
        with self.assertRaisesRegex(ValueError, "disconnected"):
            validate_checkpoint(checkpoint)


if __name__ == "__main__":
    unittest.main()
