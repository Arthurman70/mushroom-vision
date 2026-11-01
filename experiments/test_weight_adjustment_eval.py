"""Fixed-regression integrity tests; no model construction or inference."""
import copy
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from PIL import Image

from experiments.evaluate_weight_adjustment import (
    CHECKPOINT_SCHEMA, CONDITIONS, NEGATIVE_ID, POSITIVE_IDS, baseline_records,
    checked_predictions, digest, evaluation_plan, gate_for, ground_baseline_records,
    native_metrics, native_records, rows_for, scan_native_ground, source_support,
    validate_benchmark, validate_metadata, write_new,
)
from experiments.weight_recipe import config_for
from vision.training_pipeline import MODEL_NAME


def benchmark_fixture():
    records = []
    for condition in CONDITIONS:
        frames = POSITIVE_IDS + ([NEGATIVE_ID] if condition in {"native", "6p9px"} else [])
        for frame in frames:
            positive = frame in POSITIVE_IDS
            records.append({"id": frame+"--"+condition, "frameId": frame, "condition": condition,
                "kind": "positive" if positive else "negative", "humanVerifiedTruth": True,
                "sourceGroupId": "same-creator", "truth": [{"id": "box", "bbox": [2, 3, 82, 103]}] if positive else [],
                "actualDistanceM": None, "synthetic": condition != "native", "derivedImageSha256": "saved-image",
                "imageSha256": "original-pixels",
                "predictions": {"raw": []}})
    return {"schema": "mycoscout.video-distance-report.v1", "status": "completed", "records": records,
        "sample": {"positiveFrames": 3, "negativeFrames": 1, "targets": 3, "sourceVideos": 1,
                   "sourceGroups": 1, "labelAuthority": "human"}}


def scored_fixture():
    source = benchmark_fixture()
    return baseline_records(source, source["records"])


class WeightEvaluationTests(unittest.TestCase):
    def temporary_root(self):
        root = Path(__file__).resolve().parent/("weight-eval-test-"+uuid.uuid4().hex)
        if not root.resolve().is_relative_to(Path(__file__).resolve().parent):
            raise ValueError("Test cleanup path is outside experiments workspace")
        root.mkdir(); self.addCleanup(lambda: shutil.rmtree(root))
        return root

    def test_exact_frozen_conditions_and_human_authority(self):
        source = benchmark_fixture()
        self.assertEqual(26, len(validate_benchmark(source)))
        for edit in (lambda x: x["records"].pop(),
                     lambda x: x["records"][0].update(humanVerifiedTruth=False),
                     lambda x: x["records"][0].update(actualDistanceM=10),
                     lambda x: x["records"][0].update(synthetic=True),
                     lambda x: x["records"][0].update(kind="negative")):
            bad = copy.deepcopy(source); edit(bad)
            with self.assertRaises(ValueError):
                validate_benchmark(bad)

    def test_repeated_ids_and_missing_conditions_rejected(self):
        source = benchmark_fixture()
        source["records"][-1] = copy.deepcopy(source["records"][0])
        with self.assertRaisesRegex(ValueError, "26 frozen"):
            validate_benchmark(source)

    def test_synthetic_scales_cannot_inflate_source_support(self):
        rows = scored_fixture()
        for row in rows:
            if row["condition"] != "native" and row["truth"]:
                row["truth"] = [[1, 1, 2, 2]]
        support = source_support(rows)
        self.assertEqual((4, 3, 1, 1, 0, 0), tuple(support[k] for k in
            ("nativeOriginals", "targets", "targetGroups", "negativeFrames", "smallTargets", "smallTargetGroups")))
        self.assertEqual(support, source_support(rows+[r for r in rows if r["condition"] != "native"]*10))

    def test_duplicate_native_originals_cannot_inflate_support(self):
        rows = scored_fixture()
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "Repeated native"):
            source_support(rows)

    def test_fixed_46_forward_plan_adds_only_four_native_ground_views(self):
        frozen = benchmark_fixture()["records"]
        plan = evaluation_plan(frozen)
        self.assertEqual((26, 20, 46), tuple(plan[k] for k in
            ("plannedRawForwards", "plannedGroundForwards", "plannedForwards")))
        self.assertEqual(4, len(plan["groundRecordOrder"]))
        self.assertTrue(all(identity.endswith("--native") for identity in plan["groundRecordOrder"]))
        self.assertEqual([0., .35, 1., 1.], plan["groundSettings"]["tileRegion"])
        self.assertFalse(plan["groundSettings"]["colorSensitive"])
        self.assertEqual(5, plan["groundSettings"]["passesPerNativeFrame"])

    def test_ground_selection_refuses_extra_native_and_synthetic_disguises(self):
        frozen = benchmark_fixture()["records"]
        self.assertEqual(4, len(native_records(frozen)))
        with self.assertRaises(ValueError):
            native_records(frozen+[copy.deepcopy(frozen[0])])
        frozen[0]["synthetic"] = True
        with self.assertRaises(ValueError):
            native_records(frozen)

    def test_live_ground_scan_projects_deduplicates_and_counts_actual_forwards(self):
        from vision.testbench import region_tile_boxes
        bounds = [(0, 0, 100, 100)]+region_tile_boxes(100, 100, 2, [0, .35, 1, 1])
        target = [45, 50, 50, 55]
        class MockDetector:
            box_threshold = .3
            forward_count = 0
            def predict(self, image):
                box = bounds[self.forward_count]
                self.forward_count += 1
                if box[0] <= target[0] < target[2] <= box[2] and box[1] <= target[1] < target[3] <= box[3]:
                    return [{"bboxPixelsXYXY": [target[0]-box[0], target[1]-box[1], target[2]-box[0], target[3]-box[1]],
                             "confidence": .9 if self.forward_count == 2 else .8}]
                return []
        detector = MockDetector()
        predictions, scan = scan_native_ground(Image.new("RGB", (100, 100)), detector)
        self.assertEqual(5, detector.forward_count)
        self.assertEqual((5, 5, 3, 1), tuple(scan[k] for k in
            ("actualForwards", "modelPasses", "preNmsCandidateCount", "candidateCount")))
        self.assertEqual(1, len(predictions))
        self.assertEqual(target, predictions[0]["bbox"])
        self.assertEqual(.9, predictions[0]["score"])

    def test_claimed_scan_passes_must_match_actual_detector_calls(self):
        class MockDetector:
            forward_count = 0
        detector = MockDetector()
        def fake_scan(*args, **kwargs):
            detector.forward_count += 4
            return [], {"modelPasses": 5, "passTimings": [{"kind": "context"}]+[{"kind": "tile"}]*4,
                "fullFrameContext": True, "tileRegion": [0, .35, 1, 1], "settings": {"boxThreshold": .3}, "nmsIou": .5}
        with patch("adapters.live_capture.scan_frame", fake_scan):
            with self.assertRaisesRegex(ValueError, "five-pass"):
                scan_native_ground(Image.new("RGB", (100, 100)), detector)

    def test_saved_ground_comparison_excludes_ai_training_references(self):
        root = self.temporary_root()
        frozen = benchmark_fixture()["records"]
        references = []
        for index, source in enumerate(native_records(frozen)):
            path = root/f"{index}.json"
            write_new(path, {"settings": {"methods": ["tiles"], "boxThreshold": .25, "tileRegion": [0, .35, 1, 1]},
                "image": {"sha256": source["imageSha256"]}, "methods": [{"id": "tiles", "candidates": []}]})
            references.append({"frameId": source["frameId"], "referenceAuthority": "human", "referenceComplete": True,
                "sha256": source["imageSha256"], "sourceGroupId": source["sourceGroupId"], "truth": source["truth"],
                "referenceStatus": "visible" if source["truth"] else "absent",
                "methods": {"ground_tiles": {"predictionPath": path.name, "predictionSha256": digest(path), "modelPasses": 5, "seconds": 2}}})
        references.append({"referenceAuthority": "ai", "frameId": "training-negative-not-used"})
        records, paths = ground_baseline_records({"records": references}, frozen, root)
        self.assertEqual((4, 4), (len(records), len(paths)))
        self.assertEqual(set(POSITIVE_IDS+[NEGATIVE_ID]), {r["frameId"] for r in records})
        self.assertEqual(3, native_metrics(records, .25)["falseNegatives"])
        references[0]["methods"]["ground_tiles"]["modelPasses"] = 4
        with self.assertRaises(ValueError):
            ground_baseline_records({"records": references}, frozen, root)

    def test_better_three_frame_result_is_still_insufficient(self):
        rows = scored_fixture()
        base = native_metrics(rows, .3)
        for row in rows:
            row["predictions"] = [{"bbox": b, "score": .8} for b in row["truth"]]
        candidate = native_metrics(rows, .3)
        gate = gate_for(candidate, base, source_support(rows))
        self.assertEqual("insufficient_evidence", gate["decision"])
        self.assertFalse(gate["activationAllowed"])
        self.assertEqual(1, gate["recallGain"])
        self.assertEqual(5, len(gate["supportReasons"]))

    def test_policy_performance_and_support_are_separate(self):
        support = {"targets": 20, "targetGroups": 3, "smallTargets": 10, "smallTargetGroups": 2, "negativeFrames": 5}
        base = {"recall": .5, "smallTargets": {"recall": .4}, "falsePositives": 3, "negativeFalsePositives": 2}
        candidate = {"recall": .6, "smallTargets": {"recall": .5}, "falsePositives": 2, "negativeFalsePositives": 1}
        gate = gate_for(candidate, base, support)
        self.assertEqual("candidate_better", gate["decision"])
        self.assertFalse(gate["activationAllowed"])
        candidate["negativeFalsePositives"] = 3
        self.assertEqual("candidate_not_better", gate_for(candidate, base, support)["decision"])

    def test_scores_misses_unmatched_and_negative_alerts_stay_separate(self):
        rows = scored_fixture()
        rows[0]["predictions"] = [{"bbox": rows[0]["truth"][0], "score": .8},
                                  {"bbox": rows[0]["truth"][0], "score": .7},
                                  {"bbox": [200, 200, 220, 220], "score": .29}]
        rows[3]["predictions"] = [{"bbox": [200, 200, 220, 220], "score": .9}]
        native = rows_for(rows, .3)[0]
        self.assertEqual((1, 2, 1, 1), tuple(native[k] for k in
            ("matched", "missed", "unmatchedPositiveAlerts", "negativeFalseAlerts")))

    def test_baseline_requires_exact_pixels_truth_and_source_group(self):
        source = benchmark_fixture()
        self.assertEqual(26, len(baseline_records(source, source["records"])))
        for fields in ({"derivedImageSha256": "changed"}, {"sourceGroupId": "other"},
                       {"truth": []}, {"condition": "32px"}):
            bad = copy.deepcopy(source); bad["records"][0].update(fields)
            with self.assertRaises(ValueError):
                baseline_records(bad, source["records"])

    def test_legacy_candidate_predictions_keep_separate_authority(self):
        source = benchmark_fixture()
        old = {"records": [{"id": row["id"], "condition": row["condition"], "kind": row["kind"],
            "group": row["sourceGroupId"], "truth": [box["bbox"] for box in row["truth"]],
            "imageSha256": row["derivedImageSha256"], "predictions": []} for row in source["records"]]}
        self.assertEqual(26, len(baseline_records(old, source["records"], True)))
        old["records"][0]["imageSha256"] = "different"
        with self.assertRaises(ValueError):
            baseline_records(old, source["records"], True)

    def metadata(self, recipe="balanced-640"):
        config = config_for(recipe)
        provenance = {"manifestSha256": "export", "inventorySha256": "inventory", "ledgerSha256": "ledger",
                      "exportedRecords": 94, "acceptedFrames": 94, "skipped": {}}
        metadata = {"schema": CHECKPOINT_SCHEMA, "architecture": MODEL_NAME, "modelConfig": config,
            "reviewerKind": "ai", "humanVerified": False, "activationEligible": False, "scoreThreshold": .3,
            "epoch": 6, "trainingManifestSha256": "export", "inventorySha256": "inventory", "ledgerSha256": "ledger"}
        fit = {"status": "completed", "activationEligible": False, "productionModelChanged": False,
               "labelAuthority": "ai-visual-review", "selectedEpoch": 6, "protocol": {**provenance, "modelConfig": copy.deepcopy(config)}}
        return metadata, fit, provenance

    def test_checkpoint_restores_both_frozen_inference_recipes(self):
        for recipe in ("balanced-320", "balanced-640"):
            self.assertEqual(config_for(recipe), validate_metadata(*self.metadata(recipe)))

    def test_threshold_human_authority_epoch_and_provenance_drift_rejected(self):
        for fields in ({"scoreThreshold": .2}, {"humanVerified": True}, {"activationEligible": True},
                       {"epoch": 5}, {"trainingManifestSha256": "other"}, {"schema": "old"}):
            metadata, fit, provenance = self.metadata(); metadata.update(fields)
            with self.assertRaises(ValueError):
                validate_metadata(metadata, fit, provenance)

    def test_inference_config_cannot_revert_to_default_320(self):
        metadata, fit, provenance = self.metadata()
        fit["protocol"]["modelConfig"] = config_for("balanced-320")
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_metadata(metadata, fit, provenance)
        metadata, fit, provenance = self.metadata()
        metadata["modelConfig"]["diagnosticScoreFloor"] = .001
        with self.assertRaises(ValueError):
            validate_metadata(metadata, fit, provenance)

    def test_fit_report_must_remain_inactive_with_exact_inputs(self):
        for fields in ({"productionModelChanged": True}, {"labelAuthority": "human"}, {"activationEligible": True}):
            metadata, fit, provenance = self.metadata(); fit.update(fields)
            with self.assertRaises(ValueError):
                validate_metadata(metadata, fit, provenance)
        metadata, fit, provenance = self.metadata(); fit["protocol"]["ledgerSha256"] = "changed"
        with self.assertRaises(ValueError):
            validate_metadata(metadata, fit, provenance)

    def test_prediction_threshold_nonfinite_and_image_coordinates_checked(self):
        self.assertEqual([], checked_predictions([], 1280, 720))
        good = {"bbox": [10, 20, 30, 40], "score": .3}
        self.assertEqual([good], checked_predictions([good], 1280, 720))
        for bad in ({**good, "score": .29}, {**good, "score": float("nan")},
                    {**good, "bbox": [-1, 20, 30, 40]}, {**good, "bbox": [10, 20, 1300, 40]}):
            with self.assertRaises(ValueError):
                checked_predictions([bad], 1280, 720)

    def test_completed_evidence_cannot_be_overwritten(self):
        root = self.temporary_root()
        path = root/"report.json"
        write_new(path, {"first": True})
        with self.assertRaises(FileExistsError):
            write_new(path, {"replacement": True})
        self.assertEqual({"first": True}, json.loads(path.read_text()))


if __name__ == "__main__":
    unittest.main()
