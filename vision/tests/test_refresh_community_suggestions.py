"""Isolated fake-predictor tests; never load a real model or modify public data."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import shutil
import unittest
import uuid
from unittest.mock import patch

from PIL import Image

from adapters.export_reviewed_detection_dataset import canonical, digest
from vision.refresh_community_suggestions import load_public_collection, refresh


class CommunitySuggestionRefreshTests(unittest.TestCase):
    def setUp(self):
        self.parent = Path(__file__).resolve().parents[2] / "experiments/tmp-tests"
        self.parent.mkdir(parents=True, exist_ok=True)
        self.root = self.parent / ("refresh-"+uuid.uuid4().hex)
        self.root.mkdir()
        self.collection = self.root / "collection"
        (self.collection / "images").mkdir(parents=True)
        frames = []
        for index in range(3):
            image = Image.new("RGB", (80, 60), (30+(index//2)*60, 40, 50))
            stream = io.BytesIO(); image.save(stream, "PNG")
            raw = stream.getvalue(); sha = digest(raw)
            relative = "images/"+sha+".png"
            (self.collection / relative).write_bytes(raw)
            frames.append({"id": "frame-"+str(index), "sourceId": "source", "sha256": sha,
                           "width": 80, "height": 60, "path": relative, "sourceGroupId": "video:source",
                           "videoHash": "a"*64, "evaluationOnly": False, "sourceLicenseAllowsTraining": True})
        self.manifest = {"schema": "mycoscout.community-collection.v1", "frames": frames,
                         "sources": [{"id": "source", "license": "CC BY 4.0", "licenseAllowsTraining": True,
                                      "videoHash": "a"*64, "sourceGroupId": "video:source", "evaluationOnly": False}]}
        self.write_manifest()
        self.model = {"name": "Fixture model", "checkpointSha256": "b"*64, "boxThreshold": .3,
                      "version": "reviewed-fixture", "humanVerified": False, "trainingAllowed": False}
        self.output = self.root / "new-suggestions.json"
        self.cancel = self.root / "cancel"

    def tearDown(self):
        assert self.root.parent == self.parent
        shutil.rmtree(self.root)

    def write_manifest(self):
        (self.collection / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def mocks(self, predict=None):
        stack = ExitStack()
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(patch("vision.refresh_community_suggestions.active_job_identity", return_value=(self.model, self.root / "fixture.pt")))
        stack.enter_context(patch("vision.refresh_community_suggestions.load_detector", return_value=(object(), {"scoreThreshold": .3})))
        stack.enter_context(patch("vision.refresh_community_suggestions.tensor_image", side_effect=lambda p: p))
        inference = stack.enter_context(patch("vision.refresh_community_suggestions.predictions_for",
                  side_effect=predict or (lambda *_: [{"bbox": [5, 6, 30, 40], "score": .8}])))
        return stack, inference

    def test_complete_refresh_reuses_identical_image_and_preserves_unverified_flags(self):
        stack, inference = self.mocks()
        with stack:
            result = refresh(self.collection, "unused", self.output)
            self.assertEqual(inference.call_count, 2)
            before = self.output.read_bytes()
            resumed = refresh(self.collection, "unused", self.output)
            self.assertTrue(resumed["reusedCompletedOutput"])
            self.assertEqual(inference.call_count, 2)
        data = json.loads(before)
        self.assertEqual(result["completedFrames"], 3)
        self.assertEqual(data["run"]["reviewedFrames"], 3)
        self.assertEqual(data["run"]["cacheHits"], 1)
        for frame in data["frames"]:
            for box in frame["boxes"]:
                self.assertFalse(box["humanVerified"])
                self.assertFalse(box["trainingAllowed"])
        self.assertEqual(before, self.output.read_bytes())

    def test_cancel_keeps_final_absent_and_resume_uses_saved_cache(self):
        def predictor(*_):
            self.cancel.write_text("cancel", encoding="utf-8")
            return [{"bbox": [5, 6, 30, 40], "score": .8}]
        stack, inference = self.mocks(predictor)
        with stack:
            result = refresh(self.collection, "unused", self.output, cancel_file=self.cancel)
        self.assertEqual(result["status"], "cancelled")
        self.assertFalse(self.output.exists())
        self.cancel.unlink()
        stack, inference = self.mocks()
        with stack:
            result = refresh(self.collection, "unused", self.output, cancel_file=self.cancel)
            self.assertEqual(inference.call_count, 1)
        self.assertEqual(result["cacheHits"], 2)

    def test_rehashed_cache_cannot_claim_human_verification(self):
        def predictor(*_):
            self.cancel.write_text("cancel", encoding="utf-8")
            return [{"bbox": [5, 6, 30, 40], "score": .8}]
        stack, _ = self.mocks(predictor)
        with stack:
            refresh(self.collection, "unused", self.output, cancel_file=self.cancel)
        cache = next((self.root / "reviewed-suggestion-cache").glob("*.json"))
        data = json.loads(cache.read_text())
        data["payload"]["boxes"][0]["humanVerified"] = True
        data["payloadSha256"] = digest(canonical(data["payload"]))
        cache.write_text(json.dumps(data), encoding="utf-8")
        self.cancel.unlink()
        stack, _ = self.mocks()
        with stack, self.assertRaisesRegex(ValueError, "rights flags"):
            refresh(self.collection, "unused", self.output)
        self.assertFalse(self.output.exists())

    def test_unbound_existing_output_is_never_overwritten(self):
        self.output.write_text("existing public suggestions", encoding="utf-8")
        stack, inference = self.mocks()
        with stack, self.assertRaisesRegex(ValueError, "overwrite"):
            refresh(self.collection, "unused", self.output)
        inference.assert_not_called()
        self.assertEqual(self.output.read_text(), "existing public suggestions")

    def test_changed_source_rights_bytes_and_frame_budget_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "frame/source bound"):
            load_public_collection(self.collection, max_frames=2)
        self.manifest["sources"][0]["license"] = "CC BY-NC 4.0"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "eligible"):
            load_public_collection(self.collection)
        self.manifest["sources"][0]["license"] = "CC BY 4.0"
        self.write_manifest()
        (self.collection / self.manifest["frames"][0]["path"]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            load_public_collection(self.collection)

    def test_crash_after_final_write_can_recover_without_inference(self):
        stack, _ = self.mocks()
        with stack:
            refresh(self.collection, "unused", self.output)
        state_path = self.output.with_suffix(".json.state.json")
        state = json.loads(state_path.read_text())
        state.update(status="running"); state.pop("outputSha256")
        state_path.write_text(json.dumps(state), encoding="utf-8")
        stack, inference = self.mocks()
        with stack:
            result = refresh(self.collection, "unused", self.output)
            inference.assert_not_called()
        self.assertTrue(result["reusedCompletedOutput"])

    def test_model_change_requires_new_output_identity(self):
        stack, _ = self.mocks()
        with stack:
            refresh(self.collection, "unused", self.output)
        self.model = {**self.model, "checkpointSha256": "c"*64}
        stack, inference = self.mocks()
        with stack, self.assertRaisesRegex(ValueError, "inputs changed"):
            refresh(self.collection, "unused", self.output)
        inference.assert_not_called()

    def test_last_forward_exceeding_budget_leaves_only_resumable_cache(self):
        clock = [0.0]
        def predictor(*_):
            clock[0] += .6
            return [{"bbox": [5, 6, 30, 40], "score": .8}]
        stack, inference = self.mocks(predictor)
        with stack, patch("vision.refresh_community_suggestions.time.monotonic", side_effect=lambda: clock[0]):
            result = refresh(self.collection, "unused", self.output, max_seconds=1)
        self.assertEqual(inference.call_count, 2)
        self.assertEqual(result["completedFrames"], 3)
        self.assertEqual(result["status"], "paused_budget")
        self.assertFalse(self.output.exists())
        stack, inference = self.mocks()
        with stack:
            result = refresh(self.collection, "unused", self.output)
            inference.assert_not_called()
        self.assertEqual(result["cacheHits"], 3)


if __name__ == "__main__":
    unittest.main()
