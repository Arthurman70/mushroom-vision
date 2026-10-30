from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from vision.evaluate_video_ai import evaluate_snapshot, matched_pairs, sanitized_snapshot


class VideoEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"sources": [{"id": "source", "sourceGroupId": "group"}],
                         "frames": [{"id": "frame-1", "sourceId": "source", "sha256": "a" * 64, "width": 100, "height": 100}]}
        self.review = {"humanVerified": True, "complete": True, "status": "visible", "reviewer": "private",
                       "notes": "private", "boxes": [{"id": "human-1", "humanVerified": True, "bboxPixelsXYXY": [10, 10, 20, 30]}]}
        self.grades = {"schema": "mycoscout.video-grades.v1", "collectionId": "video-discovery-v1",
                       "reviews": {"frame-1": {"frameId": "frame-1", "revision": 2, "review": self.review}}}
        self.result = {"frameId": "frame-1", "collection": "main", "imageSha256": "a" * 64, "status": "completed",
                       "boxes": [{"id": "ai-1", "bbox": [10, 10, 20, 30], "score": .8}]}

    def test_partial_or_uncertain_human_reviews_cannot_supply_truth(self):
        for changes in ({"complete": False}, {"status": "uncertain"}, {"humanVerified": False}):
            grades = deepcopy(self.grades)
            grades["reviews"]["frame-1"]["review"].update(changes)
            result = sanitized_snapshot(grades, self.manifest, "manifest-sha")
            self.assertEqual(result["records"], [])
            self.assertEqual(len(result["excluded"]), 1)

    def test_snapshot_omits_identity_and_free_text_but_retains_revision_and_hash(self):
        result = sanitized_snapshot(self.grades, self.manifest, "manifest-sha")
        self.assertNotIn("private", json.dumps(result))
        self.assertEqual(result["records"][0]["revision"], 2)
        self.assertEqual(result["records"][0]["imageSha256"], "a" * 64)

    def test_real_matches_separate_from_predictions_and_missing_predictions_not_negative(self):
        snapshot = sanitized_snapshot(self.grades, self.manifest, "manifest-sha")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "frames").mkdir()
            report = evaluate_snapshot(snapshot, root)
            self.assertEqual(report["status"], "partial")
            self.assertEqual(report["metrics"]["frames"], 0)
            path = root / "frames/main--frame-1.json"
            original = json.dumps(self.result)
            path.write_text(original, encoding="utf-8")
            report = evaluate_snapshot(snapshot, root)
            self.assertEqual(report["metrics"]["truePositives"], 1)
            self.assertEqual(report["frames"][0]["humanMatches"][0]["humanBoxId"], "human-1")
            self.assertEqual(path.read_text(encoding="utf-8"), original)
            path.write_text(json.dumps({**self.result, "imageSha256": "b" * 64}), encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_snapshot(snapshot, root)

    def test_one_proposal_cannot_match_two_human_objects(self):
        truth = [{"id": "one", "bbox": [0, 0, 20, 20]}, {"id": "two", "bbox": [1, 1, 21, 21]}]
        predictions = [{"id": "model", "bbox": [0, 0, 20, 20], "score": .9}]
        self.assertEqual(len(matched_pairs(truth, predictions)), 1)


if __name__ == "__main__":
    unittest.main()
