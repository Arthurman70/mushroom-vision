from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
from adapters.tests.test_merge_video_collections import fixture, save
from vision.assistant_video_pilot import blocked_by_holdout, digest, prepare, validated_audit


class AssistantPilotTests(unittest.TestCase):
    def setUp(self):
        self.frame = {"id": "video-001", "sha256": "a" * 64, "width": 1280, "height": 720}
        self.proposal = {"collection": "main", "frameId": self.frame["id"], "imageSha256": self.frame["sha256"],
                         "status": "completed", "boxes": [{"id": "ai-1", "bbox": [20, 40, 100, 200], "score": .8}]}
        self.review = {"collection": "main", "frameId": self.frame["id"], "imageSha256": self.frame["sha256"],
                       "reviewer": "assistant-visual-audit", "complete": True, "humanVerified": False,
                       "status": "visible", "acceptedBoxIds": ["ai-1"]}
        self.signature = {"pixelSha256": "b" * 64, "averageHash": "f" * 64, "colorMean": [120, 100, 80]}
        self.held = {"groupKeys": ["source:frozen"], "originalSha256": "c" * 64,
                     "pixelSha256": "d" * 64, "averageHash": "0" * 64, "colorMean": [10, 20, 30]}

    def test_positive_audit_preserves_boxes_with_explicit_nonhuman_authority(self):
        boxes = validated_audit(self.review, self.frame, self.proposal)
        self.assertEqual(boxes[0]["bboxPixelsXYXY"], [20, 40, 100, 200])
        self.assertFalse(boxes[0]["humanVerified"])
        self.assertEqual(boxes[0]["labelAuthority"], "assistant-reviewed-ai")

    def test_explicit_absent_audit_is_required_for_negative(self):
        absent = {**self.review, "status": "absent", "acceptedBoxIds": []}
        self.assertEqual(validated_audit(absent, self.frame, self.proposal), [])
        for change in ({"complete": False}, {"status": "uncertain"}, {"status": "visible"}, {"humanVerified": True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validated_audit({**absent, **change}, self.frame, self.proposal)

    def test_changed_image_and_unaudited_proposal_cannot_train(self):
        for change in ({"imageSha256": "x" * 64}, {"acceptedBoxIds": ["ai-missing"]},
                       {"acceptedBoxIds": ["ai-1", "ai-1"]}, {"collection": "research"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validated_audit({**self.review, **change}, self.frame, self.proposal)

    def test_corrected_boxes_replace_merged_proposal_without_invented_score(self):
        review = {**self.review, "acceptedBoxIds": [], "rejectedBoxIds": ["ai-1"],
                  "correctedBoxes": [{"bbox": [20, 40, 60, 150]}, {"bbox": [55, 60, 100, 200]}]}
        boxes = validated_audit(review, self.frame, self.proposal)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[1]["bboxPixelsXYXY"], [55, 60, 100, 200])
        self.assertTrue(all(box["teacherScore"] is None and box["humanVerified"] is False for box in boxes))
        self.assertTrue(all(box["labelOrigin"] == "assistant_corrected_coordinates" for box in boxes))
        for change in ({"status": "absent"}, {"acceptedBoxIds": ["ai-1"]},
                       {"correctedBoxes": [{"bbox": [0, 0, float("nan"), 10]}]},
                       {"correctedBoxes": [{"bbox": [0, 0, 1300, 10]}]},
                       {"correctedBoxes": [{"bbox": [0, 0, 10, 10]}] * 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validated_audit({**review, **change}, self.frame, self.proposal)

    def test_independent_audit_can_draw_without_model_but_cannot_claim_model_agreement(self):
        review = {**self.review, "acceptedBoxIds": [], "correctedBoxes": [{"bbox": [20, 40, 60, 150]}]}
        boxes = validated_audit(review, self.frame, None)
        self.assertEqual(len(boxes), 1)
        self.assertIsNone(boxes[0]["teacherScore"])
        for change in ({"acceptedBoxIds": ["ai-1"]}, {"rejectedBoxIds": ["ai-1"]}, {"imageSha256": "b" * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validated_audit({**review, **change}, self.frame, None)

    def test_frozen_source_and_identical_pixels_are_always_excluded(self):
        self.assertEqual(blocked_by_holdout(["source:frozen"], "e" * 64, self.signature, [self.held]), "frozen_source_group")
        signature = {**self.signature, "pixelSha256": self.held["pixelSha256"]}
        self.assertEqual(blocked_by_holdout(["source:other"], "e" * 64, signature, [self.held]), "frozen_identical_image")
        self.assertEqual(blocked_by_holdout(["source:other"], self.held["originalSha256"], self.signature, [self.held]), "frozen_identical_image")

    def test_near_duplicate_holdouts_are_excluded_without_blocking_different_imagery(self):
        near = {**self.signature, "averageHash": "0" * 63 + "3", "colorMean": [12, 18, 34]}
        self.assertEqual(blocked_by_holdout(["source:other"], "e" * 64, near, [self.held]), "frozen_near_duplicate")
        self.assertIsNone(blocked_by_holdout(["source:other"], "e" * 64, self.signature, [self.held]))

    def test_prepare_retains_incomplete_audit_evidence_without_training_on_it(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection = root / "collection"
            manifest = fixture(collection, "source", "brown")
            extra = {**manifest["frames"][0], "id": "source-001", "path": "frames/source-001.jpg", "timestampSeconds": 3}
            Image.new("RGB", (16, 12), "yellow").save(collection / extra["path"])
            extra["sha256"] = digest((collection / extra["path"]).read_bytes())
            manifest["frames"].append(extra)
            save(collection, manifest)
            original = (collection / "manifest.json").read_bytes()
            proposals = root / "proposals"
            (proposals / "frames").mkdir(parents=True)
            reviews = []
            for index, frame in enumerate(manifest["frames"]):
                proposal = {"collection": "main", "frameId": frame["id"], "imageSha256": frame["sha256"], "status": "completed",
                            "boxes": [{"id": "ai-1", "bbox": [1, 1, 8, 10], "score": .7}]}
                (proposals / "frames" / ("main--" + frame["id"] + ".json")).write_text(json.dumps(proposal), encoding="utf-8")
                reviews.append({**self.review, "frameId": frame["id"], "imageSha256": frame["sha256"], "complete": index == 0,
                                "notes": "Uncertain scene: reject tree proposal but cannot assert absence" if index else "Visible specimen"})
            audit = root / "audit.json"
            audit.write_text(json.dumps({"schema": "mycoscout.assistant-audit.v1", "reviews": reviews}), encoding="utf-8")
            with patch("vision.assistant_video_pilot.collect_holdouts", return_value=([], [])):
                result = prepare(audit, proposals, collection, root / "holdouts", root / "dataset")
            self.assertEqual(result["images"], 1)
            self.assertEqual(result["explicitNegativeImages"], 0)
            self.assertEqual(len(result["reviewEvidence"]), 2)
            self.assertEqual(result["excluded"][0]["reason"], "incomplete_or_uncertain_audit")
            self.assertFalse(result["records"][0]["humanVerified"])
            self.assertEqual((collection / "manifest.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
