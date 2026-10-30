import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adapters.tests.test_merge_video_collections import fixture
from vision.audit_video_ai_pass import audit
from vision.tests.test_video_ai_pass import Detector
from vision.video_ai_pass import run


class PassIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.main, self.research, self.output = self.root / "main", self.root / "research", self.root / "output"
        fixture(self.main, "main", "brown")
        fixture(self.research, "research", "green")
        with patch("builtins.print"):
            run([("main", self.main), ("research", self.research)], self.output, Detector())

    def check(self):
        return audit(self.output, self.main, self.research, self.root / "holdouts", expected_frames=2, expected_sources=2)

    def test_complete_pass_checks_native_hashes_and_exact_identity_coverage(self):
        result = self.check()
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["frames"], 2)
        self.assertEqual(result["duplicateIds"], 0)
        self.assertTrue(result["allOriginalImageHashesVerified"])
        path = self.output / "frames/main--main-000.json"
        frame = json.loads(path.read_text())
        frame["sourceId"] = "other-source"
        path.write_text(json.dumps(frame), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source, geometry"):
            self.check()

    def test_missing_frame_or_duplicate_aggregate_identity_cannot_pass(self):
        report_path = self.output / "report.json"
        original = report_path.read_text()
        report = json.loads(original)
        report["frames"][1] = report["frames"][0]
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "repeated frame"):
            self.check()
        report_path.write_text(original, encoding="utf-8")
        (self.output / "frames/main--main-000.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.check()


if __name__ == "__main__":
    unittest.main()
