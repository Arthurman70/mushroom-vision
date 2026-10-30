from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from adapters.merge_video_collections import merge_collections


def fixture(root, sid, color):
    root.mkdir()
    (root / "frames").mkdir()
    path = root / "frames" / (sid + "-000.jpg")
    Image.new("RGB", (16, 12), color).save(path, quality=95)
    vh = hashlib.sha256(sid.encode()).hexdigest()
    source = {"id": sid, "title": sid, "sourceUrl": "https://example.org/" + sid,
              "attribution": "fixture author", "license": "CC0", "licenseUrl": "https://example.org/license",
              "videoHash": vh, "sourceGroupId": "video:" + vh,
              "licenseAllowsTraining": True, "evaluationOnly": False}
    frame = {"id": sid + "-000", "sourceId": sid, "path": "frames/" + path.name,
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "width": 16, "height": 12,
             "timestampSeconds": 1.5, "blurScore": 10, "blurMetric": "fixture",
             "sourceGroupId": source["sourceGroupId"], "partitionGroup": source["sourceGroupId"],
             "evaluationOnly": False, "labels": [], "annotation": {"complete": False, "boxes": []}}
    data = {"schema": "mycoscout.video-collection.v1", "createdAt": "existing timestamp",
            "sources": [source], "frames": [frame], "settings": {"preserve": "exactly"}}
    save(root, data)
    return data


def save(root, data):
    (root / "manifest.json").write_text(json.dumps(data, indent=3) + "\n", encoding="utf-8")


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="video-merge-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.target, self.incoming = self.root / "target", self.root / "incoming"
        self.original = fixture(self.target, "old", "green")
        self.extra = fixture(self.incoming, "morel", "brown")
        self.before = (self.target / "manifest.json").read_bytes()

    def assertUnchanged(self):
        self.assertEqual((self.target / "manifest.json").read_bytes(), self.before)

    def test_dry_run_does_not_write_anything(self):
        before = sorted(p.relative_to(self.target).as_posix() for p in self.target.rglob("*"))
        report = merge_collections(self.target, [self.incoming])
        self.assertTrue(report["dryRun"])
        self.assertEqual(report["newFrames"], 1)
        self.assertEqual(before, sorted(p.relative_to(self.target).as_posix() for p in self.target.rglob("*")))
        self.assertUnchanged()

    def test_append_preserves_metadata_image_bytes_grades_and_backup(self):
        grades = self.root / "video-grades.sqlite"
        grades.write_bytes(b"Existing immutable human review database fixture")
        old_image = (self.target / self.original["frames"][0]["path"]).read_bytes()
        report = merge_collections(self.target, [self.incoming], apply=True)
        merged = json.loads((self.target / "manifest.json").read_bytes())
        self.assertEqual(merged["sources"][0], self.original["sources"][0])
        self.assertEqual(merged["frames"][0], self.original["frames"][0])
        self.assertEqual(merged["settings"], self.original["settings"])
        self.assertEqual(merged["createdAt"], self.original["createdAt"])
        self.assertEqual((self.target / self.original["frames"][0]["path"]).read_bytes(), old_image)
        self.assertEqual((self.target / self.extra["frames"][0]["path"]).read_bytes(),
                         (self.incoming / self.extra["frames"][0]["path"]).read_bytes())
        self.assertEqual((self.target / report["backup"]).read_bytes(), self.before)
        self.assertEqual(grades.read_bytes(), b"Existing immutable human review database fixture")
        self.assertFalse(report["gradesModified"])

    def test_exact_repeat_preserves_manifest_bytes_and_is_noop(self):
        merge_collections(self.target, [self.incoming], apply=True)
        after = (self.target / "manifest.json").read_bytes()
        report = merge_collections(self.target, [self.incoming], apply=True)
        self.assertTrue(report["unchanged"])
        self.assertEqual(report["unchangedDuplicateFrames"], 1)
        self.assertEqual((self.target / "manifest.json").read_bytes(), after)

    def test_existing_source_metadata_cannot_change_permission(self):
        clone = deepcopy(self.original)
        clone["sources"][0]["evaluationOnly"] = True
        (self.incoming / self.original["frames"][0]["path"]).write_bytes(
            (self.target / self.original["frames"][0]["path"]).read_bytes())
        save(self.incoming, clone)
        with self.assertRaisesRegex(ValueError, "source metadata is immutable"):
            merge_collections(self.target, [self.incoming], apply=True)
        self.assertUnchanged()

    def test_existing_frame_identity_cannot_change_timestamp(self):
        clone = deepcopy(self.original)
        clone["frames"][0]["timestampSeconds"] = 100
        (self.incoming / self.original["frames"][0]["path"]).write_bytes(
            (self.target / self.original["frames"][0]["path"]).read_bytes())
        save(self.incoming, clone)
        with self.assertRaisesRegex(ValueError, "frame metadata is immutable"):
            merge_collections(self.target, [self.incoming], apply=True)
        self.assertUnchanged()

    def test_video_alias_and_url_regrouping_rejected(self):
        self.extra["sources"][0]["videoHash"] = self.original["sources"][0]["videoHash"]
        save(self.incoming, self.extra)
        with self.assertRaisesRegex(ValueError, "Repeated video bytes"):
            merge_collections(self.target, [self.incoming])
        self.extra["sources"][0]["videoHash"] = "b" * 64
        self.extra["sources"][0]["sourceUrl"] = self.original["sources"][0]["sourceUrl"]
        save(self.incoming, self.extra)
        with self.assertRaisesRegex(ValueError, "source URL"):
            merge_collections(self.target, [self.incoming])

    def test_cross_group_duplicate_images_rejected(self):
        original = self.original["frames"][0]
        (self.incoming / self.extra["frames"][0]["path"]).write_bytes((self.target / original["path"]).read_bytes())
        self.extra["frames"][0]["sha256"] = original["sha256"]
        save(self.incoming, self.extra)
        with self.assertRaisesRegex(ValueError, "Identical image"):
            merge_collections(self.target, [self.incoming], apply=True)
        self.assertUnchanged()

    def test_related_clips_share_group_but_cannot_escape_quarantine(self):
        sg = self.original["sources"][0]["sourceGroupId"]
        self.extra["sources"][0]["sourceGroupId"] = sg
        self.extra["frames"][0].update(sourceGroupId=sg, partitionGroup=sg)
        save(self.incoming, self.extra)
        report = merge_collections(self.target, [self.incoming])
        self.assertEqual(report["sharedGroups"], [sg])
        self.extra["sources"][0]["evaluationOnly"] = True
        save(self.incoming, self.extra)
        with self.assertRaisesRegex(ValueError, "conflicting training/evaluation"):
            merge_collections(self.target, [self.incoming])

    def test_traversal_and_dimension_mismatch_rejected(self):
        self.extra["frames"][0]["path"] = "../escape.jpg"
        save(self.incoming, self.extra)
        with self.assertRaises(ValueError):
            merge_collections(self.target, [self.incoming], apply=True)
        self.extra["frames"][0]["path"] = "frames/morel-000.jpg"
        self.extra["frames"][0]["width"] = 17
        save(self.incoming, self.extra)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            merge_collections(self.target, [self.incoming], apply=True)
        self.assertUnchanged()

    def test_changed_image_and_incomplete_extraction_rejected(self):
        (self.incoming / self.extra["frames"][0]["path"]).write_bytes(b"Changed bytes")
        with self.assertRaisesRegex(ValueError, "integrity mismatch"):
            merge_collections(self.target, [self.incoming], apply=True)
        (self.incoming / "INCOMPLETE.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            merge_collections(self.target, [self.incoming])
        self.assertUnchanged()

    def test_existing_unreferenced_file_is_not_overwritten(self):
        collision = self.target / self.extra["frames"][0]["path"]
        collision.write_bytes(b"Keep existing file")
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            merge_collections(self.target, [self.incoming], apply=True)
        self.assertEqual(collision.read_bytes(), b"Keep existing file")
        self.assertUnchanged()

    def test_limits_and_concurrent_lock_rejected_before_publish(self):
        with patch("adapters.merge_video_collections.MAX_FRAMES", 1):
            with self.assertRaisesRegex(ValueError, "exceeds 200 sources"):
                merge_collections(self.target, [self.incoming], apply=True)
        (self.target / ".video-merge.lock").write_text("existing owner")
        with self.assertRaisesRegex(ValueError, "Another merge"):
            merge_collections(self.target, [self.incoming], apply=True)
        self.assertEqual((self.target / ".video-merge.lock").read_text(), "existing owner")
        self.assertUnchanged()

    def test_failed_atomic_publish_leaves_original_and_retry_recovers(self):
        with patch("adapters.merge_video_collections.os.replace", side_effect=OSError("fixture publish failure")):
            with self.assertRaisesRegex(OSError, "publish failure"):
                merge_collections(self.target, [self.incoming], apply=True)
        self.assertUnchanged()
        self.assertFalse((self.target / ".video-merge.lock").exists())
        report = merge_collections(self.target, [self.incoming], apply=True)
        self.assertEqual(report["framesAfter"], 2)


if __name__ == "__main__":
    unittest.main()
