import unittest
import numpy as np

from vision.proposal_verifier_pilot import explicit_negative_crop, select_training_threshold


class ProposalVerifierTests(unittest.TestCase):
    def setUp(self):
        self.frame = {"id": "frame-1", "sha256": "a" * 64, "width": 100, "height": 100}
        self.proposal = {"status": "completed", "collection": "main", "frameId": "frame-1", "imageSha256": "a" * 64,
                         "boxes": [{"id": "ai-1", "bbox": [10, 20, 30, 40]}]}
        self.audit = {"negative": True, "humanVerified": False, "reviewer": "assistant-visual-audit", "collection": "main",
                      "frameId": "frame-1", "imageSha256": "a" * 64, "proposalBoxId": "ai-1", "bbox": [10, 20, 30, 40]}

    def test_explicit_absence_crop_is_bound_to_exact_image_and_coordinates(self):
        self.assertEqual(explicit_negative_crop(self.audit, self.frame, self.proposal), [10, 20, 30, 40])
        for change in ({"imageSha256": "b" * 64}, {"bbox": [11, 20, 30, 40]}, {"humanVerified": True}, {"proposalBoxId": "missing"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                explicit_negative_crop({**self.audit, **change}, self.frame, self.proposal)

    def test_rejected_localization_or_ambiguous_contents_never_implies_negative(self):
        self.assertIsNone(explicit_negative_crop({**self.audit, "negative": False}, self.frame, self.proposal))
        audit = {key: value for key, value in self.audit.items() if key != "negative"}
        audit["rejectedBoxIds"] = ["ai-1"]
        self.assertIsNone(explicit_negative_crop(audit, self.frame, self.proposal))

    def test_threshold_policy_operates_only_on_supplied_training_fold_scores(self):
        threshold, policy = select_training_threshold(np.asarray([0, 0, 1, 1]), np.asarray([.1, .2, .8, .9]))
        self.assertGreater(threshold, .2)
        self.assertLessEqual(threshold, .8)
        self.assertEqual(policy["metrics"]["f1"], 1)
        self.assertFalse(policy["calibratedProbability"])
        self.assertIn("training-source", policy["selectionData"])


if __name__ == "__main__":
    unittest.main()
