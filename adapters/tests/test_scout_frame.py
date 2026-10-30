import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image, ImageDraw
from adapters.scout_frame import inspect_frame, LocalDetector


class Predictor:
    info = {"model": "software fixture", "resolvedRevision": "test"}

    def __init__(self):
        self.calls = []

    def predict(self, image):
        self.calls.append(image.copy())
        return [{"bboxPixelsXYXY": [18, 18, 43, 43], "confidence": .4 if len(self.calls) == 1 else .7,
                 "species": "caller species must be stripped", "measuredRange": {"valueM": 20}}]


class StationFixture:
    def __init__(self):
        self.calls = []

    def request(self, path, data=None, **kwargs):
        self.calls.append((path, data))
        if path == "/api/finds":
            return {"find": {"id": "test-find-" + str(len(self.calls))}}
        return {"result": {"guesses": [], "speciesConfirmed": False}}


def photo(folder):
    path = Path(folder) / "source.png"
    image = Image.new("RGB", (80, 60), (25, 45, 30))
    ImageDraw.Draw(image).ellipse((20, 20, 40, 40), fill=(60, 90, 90))
    image.save(path)
    return path


class ScoutFrameTests(unittest.TestCase):
    def test_classification_failure_is_reported_while_uploaded_photo_is_retained(self):
        class UnavailableClassifier(StationFixture):
            def request(self, path, data=None, **kwargs):
                if path == "/api/vision/classify":
                    raise RuntimeError("Classifier unavailable in test fixture")
                return super().request(path, data, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            result = inspect_frame(photo(folder), Path(folder) / "result", Predictor(), station=UnavailableClassifier())
            self.assertEqual(result["reviewUploads"][0]["uploadedCount"], 1)
            self.assertEqual(result["reviewUploads"][0]["classificationErrorCount"], 1)
            self.assertEqual(result["reviewUploads"][0]["classifiedCount"], 0)
            self.assertEqual(result["reviewUploads"][0]["uploadErrorCount"], 0)

    def test_dual_view_merges_candidates_without_changing_source_pixels_or_creating_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = photo(folder); output = Path(folder) / "result"; predictor = Predictor()
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            summary = inspect_frame(path, output, predictor, color_sensitive=True, max_color_cues=2, evaluation_only=True)
            report = json.loads((output / "detector-candidates.json").read_text())
            self.assertEqual(len(predictor.calls), 2)
            self.assertEqual(summary["candidateCount"], 1)
            self.assertEqual(summary["colorCueCount"], 2)
            self.assertEqual(report["candidates"][0]["confidence"], .7)
            self.assertEqual(report["candidates"][0]["inferenceView"], "color_enhanced")
            self.assertIsNone(report["candidates"][0]["species"])
            self.assertNotIn("measuredRange", report["candidates"][0])
            self.assertEqual(report["image"]["sha256"], before)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)
            self.assertNotEqual(predictor.calls[0].tobytes(), predictor.calls[1].tobytes())
            leads = json.loads((output / "archive-leads.json").read_text())["observations"]
            self.assertEqual(sum(row["kind"] == "habitat" for row in leads), 2)
            self.assertTrue(all(row["capturedMs"] is None and "metric" not in row for row in leads))

    def test_default_remains_single_original_detector_pass_without_color_proposals(self):
        with tempfile.TemporaryDirectory() as folder:
            path = photo(folder); predictor = Predictor()
            result = inspect_frame(path, Path(folder) / "result", predictor)
            self.assertEqual(result["detectorPasses"], 1)
            self.assertEqual(result["colorCueCount"], 0)
            self.assertFalse(result["colorSensitive"])

    def test_explicit_station_upload_uses_original_crops_and_skips_color_classification(self):
        with tempfile.TemporaryDirectory() as folder:
            path = photo(folder); station = StationFixture()
            result = inspect_frame(path, Path(folder) / "result", Predictor(), color_sensitive=True,
                                   max_color_cues=2, station=station, session_id="fixture-survey", evaluation_only=True)
            uploads = [payload for endpoint, payload in station.calls if endpoint == "/api/finds"]
            self.assertEqual(len(uploads), 3)
            self.assertEqual(sum(endpoint == "/api/vision/classify" for endpoint, _ in station.calls), 1)
            self.assertEqual(result["reviewUploads"][1]["classifiedCount"], 0)
            self.assertEqual(len({row["groupId"] for row in uploads}), 1)
            self.assertTrue(all(row["rights"]["trainingOptIn"] is False for row in uploads))
            self.assertTrue(all(row["captureTimestamp"] is None for row in uploads))
            self.assertTrue(all(row["metadata"]["frameCaptureGeometry"]["physicalCloseCapture"] is False for row in uploads))

    def test_budgets_existing_outputs_and_remote_model_names_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = photo(folder); predictor = Predictor(); output = Path(folder) / "result"
            for options in ({"max_candidates": 51}, {"max_color_cues": 0}, {"min_delta_e": float("nan")},
                            {"color_sensitive": 1}, {"evaluation_only": 1}):
                with self.assertRaises(ValueError):
                    inspect_frame(path, output, predictor, **options)
            self.assertEqual(predictor.calls, [])
            output.mkdir()
            with self.assertRaises(FileExistsError):
                inspect_frame(path, output, predictor)
            with self.assertRaises(ValueError):
                LocalDetector(Path(folder) / "nonexistent-remote-model")


if __name__ == "__main__":
    unittest.main()
