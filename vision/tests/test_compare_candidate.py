"""Offline integrity, denominator, release-split and conservative-gate tests."""
import copy
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from PIL import Image

from adapters.export_reviewed_detection_dataset import SCHEMA as DATASET_SCHEMA, canonical, digest, write_json
from vision.compare_candidate import (
    MODEL_NAME, POLICY, SCHEMA, assert_no_training_overlap, checked_predictions,
    decision_for, load_dataset, read_json, recompute, support_for, truth_record,
    validate_comparison, validate_job_metadata,
)
from vision.training_pipeline import release_evaluation


class CandidateComparisonTests(unittest.TestCase):
    def setUp(self):
        self.parent = Path(__file__).resolve().parents[2] / "experiments/tmp-tests"
        self.parent.mkdir(parents=True, exist_ok=True)
        self.root = self.parent / ("comparison-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "images").mkdir()
        rows = []
        for index, split in enumerate(["train", "validation", "validation", "test"]):
            image = Image.new("RGB", (80, 60), (10+index*30, 20, 30))
            image_path = "images/" + str(index) + ".png"
            image.save(self.dataset / image_path)
            pixels = digest(canonical(list(image.size))+image.tobytes())
            raw_sha = digest((self.dataset / image_path).read_bytes())
            boxes = [] if index == 2 else [{"bboxPixelsXYXY": [5, 6, 35, 40]}]
            rows.append({"id": str(index), "sourceId": "frame-"+str(index), "split": split,
                         "leakageGroup": "group-"+str(index), "groupKeys": ["source-"+str(index)],
                         "evaluationQuarantine": split == "test", "evaluationOnly": split == "test",
                         "trainingAllowed": split != "test", "complete": True, "humanVerified": True,
                         "status": "visible" if boxes else "absent", "width": 80, "height": 60,
                         "imagePath": image_path, "imageSha256": raw_sha, "pixelSha256": pixels,
                         "originalSha256": raw_sha, "boxes": boxes})
        self.manifest = {"schema": DATASET_SCHEMA, "status": "READY", "records": rows}
        self.manifest["datasetId"] = "dataset-" + digest(canonical(self.manifest))[:24]
        write_json(self.dataset / "manifest.json", self.manifest)
        self.sha = digest((self.dataset / "manifest.json").read_bytes())
        write_json(self.dataset / "READY.json", {"manifestSha256": self.sha})

    def tearDown(self):
        assert self.root.parent == self.parent
        shutil.rmtree(self.root)

    def evidence(self):
        records = []
        for row in self.manifest["records"]:
            if row["split"] != "validation":
                continue
            record = truth_record(row)
            predictions = [{"bbox": b, "score": .7} for b in record["truth"]]
            record.update(baseline={"predictions": copy.deepcopy(predictions), "seconds": .4},
                          candidate={"predictions": copy.deepcopy(predictions), "seconds": .2})
            records.append(record)
        return {"schema": "mycoscout.candidate-comparison-evidence.v1", "split": "validation", "datasetId": self.manifest["datasetId"],
                "datasetManifestSha256": self.sha, "scoreThreshold": .3, "records": records,
                "models": {"candidate": {"checkpointSha256": "c"*64, "recipeScoreThreshold": .3},
                           "baseline": {"kind": "installed_grounding_dino", "filesSha256": {"fixture": "a"*64}}}}

    def save_comparison(self, evidence=None):
        evidence = evidence or self.evidence()
        output = self.root / "comparison"
        output.mkdir()
        write_json(output / "evidence.json", evidence)
        protocol = {"models": evidence["models"], "policy": POLICY, "split": "validation", "scoreThreshold": .3,
                    "iouThreshold": .5, "commonNmsIou": .5, "maximumPredictionsPerFrame": 100, "datasetManifestSha256": self.sha}
        write_json(output / "protocol.json", protocol)
        report = {"schema": SCHEMA, "status": "completed", "split": "validation", "datasetId": self.manifest["datasetId"],
                  "datasetManifestSha256": self.sha, "models": evidence["models"], "policy": POLICY,
                  "candidateCheckpointSha256": "c"*64, "scoreThreshold": .3, "iouThreshold": .5,
                  "testEvaluationPerformed": False, "evidenceSha256": digest((output / "evidence.json").read_bytes()),
                  **recompute(evidence, self.manifest, self.sha)}
        write_json(output / "comparison.json", report)
        self.bind_output(output)
        return output, report

    def bind_output(self, output):
        (output / "COMPLETE.json").write_text(json.dumps({"schema": SCHEMA,
            "comparisonSha256": digest((output / "comparison.json").read_bytes()),
            "evidenceSha256": digest((output / "evidence.json").read_bytes()),
            "protocolSha256": digest((output / "protocol.json").read_bytes())}), encoding="utf-8")

    def test_frozen_dataset_and_decoded_pixels_are_checked(self):
        manifest, actual = load_dataset(self.dataset)
        self.assertEqual(actual, self.sha)
        self.assertEqual(manifest["datasetId"], self.manifest["datasetId"])
        (self.dataset / "images/1.png").write_bytes(b"tampered")
        with self.assertRaisesRegex(ValueError, "image changed"):
            load_dataset(self.dataset)

    def test_rehashed_manifest_with_changed_human_truth_keeps_no_old_identity(self):
        self.manifest["records"][1]["boxes"][0]["bboxPixelsXYXY"][0] = 7
        (self.dataset / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.dataset / "READY.json").write_text(json.dumps({"manifestSha256": digest((self.dataset / "manifest.json").read_bytes())}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content identity"):
            load_dataset(self.dataset)

    def rewrite_dataset(self):
        self.manifest.pop("datasetId", None)
        self.manifest["datasetId"] = "dataset-" + digest(canonical(self.manifest))[:24]
        (self.dataset / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        (self.dataset / "READY.json").write_text(json.dumps({"manifestSha256": digest((self.dataset / "manifest.json").read_bytes())}), encoding="utf-8")

    def test_even_rehashed_evaluation_only_cannot_be_validation_or_train(self):
        self.manifest["records"][1].update(evaluationOnly=True, evaluationQuarantine=False, trainingAllowed=False)
        self.rewrite_dataset()
        with self.assertRaisesRegex(ValueError, "Evaluation-only"):
            load_dataset(self.dataset)

    def test_source_alias_cannot_inflate_evidence_group_counts_within_validation(self):
        self.manifest["records"][2]["groupKeys"] = self.manifest["records"][1]["groupKeys"]
        self.rewrite_dataset()
        with self.assertRaisesRegex(ValueError, "different leakage groups"):
            load_dataset(self.dataset)

    def test_recompute_uses_only_validation_and_separates_negatives(self):
        evidence = self.evidence()
        evidence["records"][1]["candidate"]["predictions"] = [{"bbox": [40, 20, 50, 30], "score": .9}]
        result = recompute(evidence, self.manifest, self.sha)
        self.assertEqual(result["candidate"]["frames"], 2)
        self.assertEqual(result["candidate"]["truePositives"], 1)
        self.assertEqual(result["candidate"]["negativeFalsePositives"], 1)
        self.assertEqual(result["support"]["targets"], 1)
        self.assertEqual(result["decision"], "candidate_not_better")
        self.assertFalse(result["evidenceSufficient"])
        self.assertFalse(result["activationAllowed"])

    def test_tampered_truth_or_test_frame_cannot_join_validation(self):
        evidence = self.evidence()
        evidence["records"][0]["truth"][0][0] = 8
        with self.assertRaisesRegex(ValueError, "truth"):
            recompute(evidence, self.manifest, self.sha)
        evidence = self.evidence()
        evidence["records"].append(truth_record(self.manifest["records"][-1]))
        with self.assertRaisesRegex(ValueError, "cohort"):
            recompute(evidence, self.manifest, self.sha)

    def test_predictions_reject_nonfinite_invalid_and_duplicate_boxes(self):
        for score in [float("nan"), float("inf"), -1, 1.1, True]:
            with self.assertRaises(ValueError):
                checked_predictions([{"bbox": [1, 2, 5, 6], "score": score}], 80, 60, .3)
        with self.assertRaises(ValueError):
            checked_predictions([{"bbox": [1, 2, 90, 6], "score": .9}], 80, 60, .3)
        evidence = self.evidence()
        evidence["records"][0]["candidate"]["predictions"] *= 2
        with self.assertRaisesRegex(ValueError, "NMS"):
            recompute(evidence, self.manifest, self.sha)

    def test_source_identity_overlap_is_rejected_even_if_group_names_change(self):
        evaluation = copy.deepcopy(self.manifest["records"][1])
        evaluation["groupKeys"] = self.manifest["records"][0]["groupKeys"]
        with self.assertRaisesRegex(ValueError, "training sources"):
            assert_no_training_overlap([evaluation], self.manifest)

    def test_positive_gate_requires_small_targets_groups_negatives_and_no_fp_regression(self):
        support = {"targets": 20, "targetGroups": 3, "smallTargets": 10, "smallTargetGroups": 2, "negativeFrames": 5}
        baseline = {"truePositives": 10, "falsePositives": 3, "negativeFalsePositives": 1, "recall": .5, "smallTargets": {"recall": .3}}
        candidate = {"truePositives": 12, "falsePositives": 3, "negativeFalsePositives": 1, "recall": .6, "smallTargets": {"recall": .5}}
        self.assertEqual(decision_for(baseline, candidate, support)["decision"], "candidate_better")
        low = {**support, "smallTargets": 9}
        self.assertEqual(decision_for(baseline, candidate, low)["decision"], "insufficient_evidence")
        worse = {**candidate, "falsePositives": 4}
        self.assertEqual(decision_for(baseline, worse, support)["decision"], "candidate_not_better")

    def test_validation_only_never_calls_reserved_test_evaluation(self):
        with patch("vision.training_pipeline.evaluate") as evaluate:
            self.assertEqual(release_evaluation(object(), "unused", ["reserved"], .3, True), (None, None))
            evaluate.assert_not_called()
            release_evaluation(object(), "unused", ["reserved"], .3, False)
            self.assertEqual(evaluate.call_args.args[2], ["reserved"])

    def test_model_card_and_checkpoint_recipe_integrity(self):
        card = {"schema": "mycoscout.detector-model-card.v1", "architecture": MODEL_NAME, "checkpoint": "candidate.pt",
                "checkpointSha256": "c"*64, "datasetId": self.manifest["datasetId"], "datasetManifestSha256": self.sha,
                "bestEpoch": 2, "scoreThreshold": .3, "validationOnly": True, "testEvaluationPerformed": False,
                "activationEligible": False, "test": None, "testMetrics": None}
        complete = {"checkpointSha256": "c"*64, "datasetId": self.manifest["datasetId"], "modelCardSha256": "d"*64}
        meta = {"schema": "mycoscout.detector-checkpoint.v1", "architecture": MODEL_NAME, "datasetId": self.manifest["datasetId"],
                "datasetManifestSha256": self.sha, "epoch": 2, "scoreThreshold": .3, "validationOnly": True,
                "classes": {0: "background", 1: "mushroom"}}
        self.assertEqual(validate_job_metadata(card, complete, meta, "c"*64, self.manifest, self.sha, "d"*64), .3)
        for changed in [{**card, "scoreThreshold": .2}, {**card, "test": {"recall": 1}}, {**card, "activationEligible": True}]:
            with self.assertRaises(ValueError):
                validate_job_metadata(changed, complete, meta, "c"*64, self.manifest, self.sha, "d"*64)

    def test_output_validation_recomputes_metrics_even_when_attacker_rehashes_summary(self):
        output, report = self.save_comparison()
        with patch("vision.compare_candidate.installed_model_files", return_value={"fixture": "a"*64}):
            self.assertEqual(validate_comparison(output, self.dataset)["decision"], "insufficient_evidence")
        report["candidate"]["truePositives"] = 900
        (output / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
        self.bind_output(output)
        with self.assertRaisesRegex(ValueError, "metrics or decision"):
            validate_comparison(output, self.dataset)

    def test_output_hash_change_and_json_duplicate_keys_are_rejected(self):
        output, report = self.save_comparison()
        report["decision"] = "candidate_better"
        (output / "comparison.json").write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "integrity"):
            validate_comparison(output, self.dataset)
        duplicate = self.root / "duplicate.json"
        duplicate.write_text('{"x":1,"x":2}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            read_json(duplicate)


if __name__ == "__main__":
    unittest.main()
