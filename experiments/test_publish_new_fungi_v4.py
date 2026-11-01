import copy
import json
import unittest

from experiments.publish_new_fungi_v4 import joined_notes, public_result, reviewed_suggestions


def fixture():
    image_hash = "a" * 64
    frame = {"id": "inat-photo-1", "sourceId": "photo-one", "sha256": image_hash, "originalSha256": image_hash,
             "width": 400, "height": 300}
    source = {"id": "photo-one", "title": "Public photo", "attribution": "Test photographer, CC BY",
              "license": "CC-BY", "sourceUrl": "https://www.inaturalist.org/observations/1",
              "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"}
    note = {"id": frame["id"], "imageSha256": image_hash, "width": 400, "height": 300,
            "humanVerified": False, "complete": False, "status": "visible", "observation": {"mode": "original-image"},
            "selectedTargetPixelsXYXY": [20, 20, 150, 150], "notes": "A selected mushroom is visible.",
            "groupHintFromSource": "source_hint_only"}
    suggestions = {"frames": [{"frameId": "old", "status": "completed", "audit": {"retained": True}},
        {"frameId": "inat-photo-1", "sourceId": "photo-one", "imageSha256": image_hash, "status": "pending", "boxes": [], "audit": None},
        {"frameId": "other-new", "status": "pending", "audit": None}],
        "run": {"status": "in-progress", "reviewedFrames": 1, "totalFrames": 3, "humanVerified": False}}
    return {"frames": [frame], "sources": [source]}, {"records": [note]}, suggestions


class PublicRoundTests(unittest.TestCase):
    def test_only_selected_entry_changes(self):
        manifest, notes, suggestions = fixture()
        before = copy.deepcopy(suggestions)
        updated, changed = reviewed_suggestions(suggestions, notes, manifest)
        self.assertEqual(suggestions, before)
        self.assertEqual(changed, {"inat-photo-1"})
        self.assertEqual(updated["frames"][0], before["frames"][0])
        self.assertEqual(updated["frames"][2], before["frames"][2])
        self.assertEqual(updated["run"]["reviewedFrames"], 2)

    def test_new_box_never_invents_confidence_or_training_permission(self):
        manifest, notes, suggestions = fixture()
        updated, _ = reviewed_suggestions(suggestions, notes, manifest)
        audit = updated["frames"][1]["audit"]
        self.assertEqual(audit["reviewId"], "round4-inat-photo-1")
        self.assertEqual(audit["revision"], 1)
        self.assertEqual(audit["observation"]["mode"], "original-image")
        self.assertFalse(audit["complete"])
        self.assertFalse(audit["humanVerified"])
        self.assertFalse(audit["trainingAllowed"])
        self.assertEqual(audit["boxes"][0]["groupId"], "unknown")
        self.assertNotIn("confidence", audit["boxes"][0])

    def test_existing_reviews_cannot_be_overwritten(self):
        manifest, notes, suggestions = fixture()
        suggestions["frames"][1]["status"] = "completed"
        with self.assertRaisesRegex(ValueError, "previously pending"):
            reviewed_suggestions(suggestions, notes, manifest)

    def test_original_identity_change_rejected(self):
        manifest, notes, _ = fixture()
        manifest["frames"][0]["originalSha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "coordinate system"):
            joined_notes(notes, manifest)

    def test_published_hash_is_used_when_metadata_stripping_changes_bytes(self):
        manifest, notes, _ = fixture()
        manifest["frames"][0]["sha256"] = "b" * 64
        _, frame, _ = joined_notes(notes, manifest)[0]
        self.assertEqual(frame["sha256"], "b" * 64)

    def test_public_payload_excludes_private_paths_and_untested_negative_zeroes(self):
        manifest, notes, _ = fixture()
        note = notes["records"][0]
        methods = {name: {"metrics": {"selectedTargetHit": True, "predictions": 2}, "seconds": 1.}
                   for name in ("native", "low32", "low16", "low32tiles")}
        row = {"id": note["id"], "kind": "new_ai_selected_target", "sha256": note["imageSha256"],
               "selectedTargetPixelsXYXY": note["selectedTargetPixelsXYXY"], "methods": methods, "path": "PRIVATE-PATH"}
        negatives = [{"id": f"negative-{i}", "kind": "prior_ai_negative", "methods": {"native": {"metrics": {"aiReferenceNegativeAlerts": n}}}}
                     for i, n in enumerate((2, 1, 2, 3))]
        summary = [{"method": name, "selectedTargetHits": 1, "selectedTargets": 1, "meanSecondsPerFrame": 1.,
                    "frames": 5 if name == "native" else 1} for name in methods]
        report = {"status": "completed", "inputIntegrityVerified": True, "trainingPerformed": False,
                  "records": [row] + negatives, "summary": summary, "completedAt": "2026-11-01T03:00:00Z",
                  "model": {"model": "Saved local detector"}, "private": "PRIVATE-PATH"}
        result = public_result(report, notes, manifest, {"iouThreshold": .5, "scoreThreshold": .25})
        self.assertNotIn("PRIVATE-PATH", json.dumps(result))
        self.assertEqual(result["conditions"][0]["negativeFramesWithAlerts"], 4)
        self.assertEqual(result["conditions"][0]["negativeAlertCount"], 8)
        for condition in result["conditions"][1:]:
            self.assertIsNone(condition["negativeFrameCount"])
            self.assertIsNone(condition["negativeAlertCount"])
        self.assertEqual(len(result["frames"]), 1)


if __name__ == "__main__":
    unittest.main()
