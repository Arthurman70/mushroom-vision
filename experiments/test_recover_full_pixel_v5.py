import json
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid
from experiments.recover_full_pixel_v5 import atomic_json_retry, copy_bound, digest


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(__file__).resolve().parent / ("tmp-recovery-test-" + uuid.uuid4().hex)
        self.folder.mkdir()

    def tearDown(self):
        for path in self.folder.iterdir(): path.unlink()
        self.folder.rmdir()

    def test_transient_windows_replace_denial_is_retried_without_corrupting_json(self):
        target = self.folder / "report.json"; target.write_text('{"old":true}')
        original = Path.replace; attempts = []
        def intermittent(path, destination):
            attempts.append(path)
            if len(attempts) < 3: raise PermissionError("shared read handle")
            return original(path, destination)
        with patch.object(Path, "replace", intermittent), patch("time.sleep"):
            atomic_json_retry(target, {"complete": True}, attempts=3)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(json.loads(target.read_text()), {"complete": True})

    def test_persistent_denial_preserves_original_and_leaves_complete_temporary(self):
        target = self.folder / "report.json"; target.write_text('{"old":true}')
        with patch.object(Path, "replace", side_effect=PermissionError("locked")), patch("time.sleep"):
            with self.assertRaises(PermissionError): atomic_json_retry(target, {"next": True}, attempts=2)
        self.assertEqual(json.loads(target.read_text()), {"old": True})
        self.assertEqual(json.loads(target.with_suffix(".json.tmp").read_text()), {"next": True})

    def test_evidence_copy_refuses_wrong_hash_and_overwrite(self):
        source, target = self.folder / "source", self.folder / "copy"
        source.write_bytes(b"frozen evidence")
        with self.assertRaises(ValueError): copy_bound(source, target, "0" * 64)
        self.assertFalse(target.exists())
        copy_bound(source, target, digest(source))
        self.assertEqual(source.read_bytes(), target.read_bytes())
        with self.assertRaises(FileExistsError): copy_bound(source, target, digest(source))


if __name__ == "__main__": unittest.main()
