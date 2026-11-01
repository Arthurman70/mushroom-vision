from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

from PIL import Image
from experiments import evaluate_new_fungi_v4 as evaluation


@contextmanager
def temporary_directory():
    base = Path(__file__).resolve().parent
    path = base / ("new-fungi-eval-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != base or not path.name.startswith("new-fungi-eval-test-"):
            raise ValueError("Cleanup escaped test workspace")
        shutil.rmtree(path)


class NewFungiEvaluationTests(unittest.TestCase):
    def test_full_image_information_loss_uses_target_width_and_original_canvas(self):
        image = Image.new("RGB", (400, 200), "red")
        view, metadata = evaluation.degraded_image(image, [100, 20, 200, 100], 32)
        self.assertEqual(view.size, image.size)
        self.assertEqual((metadata["reducedWidth"], metadata["reducedHeight"]), (128, 64))
        self.assertEqual(metadata["effectiveTargetWidthPixels"], 32)
        self.assertFalse(metadata["cropOrPaste"])
        self.assertIsNone(metadata["actualDistanceM"])
        self.assertEqual(view.getpixel((0, 0)), (255, 0, 0))

    def test_variant_pixel_values_match_two_whole_frame_resizes(self):
        image = Image.new("RGB", (400, 200))
        image.putdata([(x % 256, (x // 400) % 256, 127) for x in range(80000)])
        view, _ = evaluation.degraded_image(image, [50, 20, 250, 180], 16)
        expected = image.resize((32, 16), Image.Resampling.LANCZOS).resize(image.size, Image.Resampling.LANCZOS)
        self.assertEqual(view.tobytes(), expected.tobytes())

    def test_invalid_reference_boxes_rejected(self):
        for box in ([0, 0, float("nan"), 10], [0, 0, 500, 100], [-1, 0, 50, 50], [0, 0, 20, 10], [True, 0, 90, 90]):
            with self.assertRaises(ValueError):
                evaluation.validate_target(box, (400, 200))
        self.assertIsNone(evaluation.validate_target(None, (400, 200)))

    def test_no_target_only_native_and_exact_80_forward_budget(self):
        positive = {"selectedTargetPixelsXYXY": [0, 0, 100, 100]}
        unknown = {"selectedTargetPixelsXYXY": None}
        self.assertEqual([m["id"] for m in evaluation.methods_for(unknown)], ["native"])
        references = [positive] * 9 + [unknown] * 8
        self.assertEqual(sum(m["passes"] for r in references for m in evaluation.methods_for(r)), 80)

    def test_multiple_boxes_cannot_count_selected_target_twice(self):
        reference = {"kind": "new_ai_selected_target", "selectedTargetPixelsXYXY": [0, 0, 100, 100]}
        candidates = [{"bboxPixelsXYXY": [0, 0, 100, 100]}, {"bboxPixelsXYXY": [0, 0, 90, 90]}, {"bboxPixelsXYXY": [200, 0, 300, 100]}]
        result = evaluation.score(reference, candidates)
        self.assertIs(result["selectedTargetHit"], True)
        self.assertEqual(result["overlappingPredictionIndexes"], [0, 1])
        self.assertEqual(result["unmatchedAlerts"], 1)
        self.assertIsNone(result["falsePositiveRate"])

    def test_iou_threshold_is_point_five(self):
        reference = {"kind": "new_ai_selected_target", "selectedTargetPixelsXYXY": [0, 0, 100, 100]}
        self.assertTrue(evaluation.score(reference, [{"bboxPixelsXYXY": [0, 0, 50, 100]}])["selectedTargetHit"])
        self.assertFalse(evaluation.score(reference, [{"bboxPixelsXYXY": [0, 0, 49, 100]}])["selectedTargetHit"])

    def test_uncertain_rows_have_no_accuracy_or_absence_score(self):
        reference = {"kind": "new_ai_qualitative", "selectedTargetPixelsXYXY": None}
        result = evaluation.score(reference, [{"bboxPixelsXYXY": [0, 0, 100, 100]}])
        self.assertIsNone(result["selectedTargetHit"])
        self.assertIsNone(result["aiReferenceNegativeAlerts"])
        self.assertTrue(result["qualitativeOnly"])

    def test_ai_negative_alerts_remain_separate(self):
        reference = {"kind": "prior_ai_negative", "selectedTargetPixelsXYXY": None}
        result = evaluation.score(reference, [{"bboxPixelsXYXY": [0, 0, 100, 100]}])
        self.assertEqual(result["aiReferenceNegativeAlerts"], 1)
        self.assertIsNone(result["selectedTargetHit"])
        self.assertFalse(result["humanVerified"])

    def test_input_hash_tampering_rejected(self):
        with temporary_directory() as root:
            path = root / "source.json"
            path.write_text("first", encoding="utf-8")
            protocol = {"inputSha256": {"source.json": evaluation.digest(path)}, "softwareVersions": {"test": "1"}}
            with patch.object(evaluation, "ROOT", root), patch.object(evaluation, "versions", return_value={"test": "1"}):
                evaluation.verify_inputs(protocol)
                path.write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "Frozen input changed"):
                    evaluation.verify_inputs(protocol)

    def test_new_json_refuses_silent_overwrite(self):
        with temporary_directory() as root:
            path = root / "claim.json"
            evaluation.new_json(path, {"first": True})
            with self.assertRaises(FileExistsError):
                evaluation.new_json(path, {"second": True})
            self.assertEqual(json.loads(path.read_text()), {"first": True})


if __name__ == "__main__":
    unittest.main()
