import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from adapters.detections_to_review import INPUT_SCHEMA, COLOR_SCHEMA, prepare_candidates, send_candidates, main


class FakeStation:
    def __init__(self):
        self.calls = []

    def request(self, path, data=None, **kwargs):
        self.calls.append((path, data))
        if path == "/api/finds":
            return {"find": {"id": "fixture-find-" + str(len(self.calls))}}
        return {"result": {"guesses": [{"classId": "black_trumpets", "modelScore": 0.4}], "speciesConfirmed": False}}


def source(folder, orientation=None):
    path = Path(folder) / "source.jpg"
    image = Image.new("RGB", (100, 80), "green")
    options = {}
    if orientation:
        exif = Image.Exif(); exif[274] = orientation; options["exif"] = exif
    image.save(path, **options)
    return path


def document(path, width=100, height=80):
    def candidate(box, score):
        return {"bboxPixelsXYXY": box, "confidence": score, "status": "UNLOCATED", "poseKnown": False, "depthKnown": False}
    return {"schema": INPUT_SCHEMA, "image": {"path": "DO-NOT-OPEN.jpg", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "width": width, "height": height, "pixelConvention": "EXIF-oriented RGB image; x right, y down"},
        "candidates": [candidate([0, 0, 40, 30], 0.9), candidate([50, 35, 100, 80], 0.6), candidate([-1, 0, 40, 30], 0.8)]}


class DetectionReviewTests(unittest.TestCase):
    def test_document_evaluation_and_synthetic_flags_cannot_be_downgraded(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder)
            for index, flags in enumerate(({"evaluationOnly": True}, {"synthetic": True, "evaluationOnly": False})):
                data = document(path); data.update(flags)
                output = Path(folder) / f"evaluation-{index}"
                prepared = prepare_candidates(data, path, output, evaluation_only=False)
                self.assertTrue(prepared["evaluationOnly"])
                station = FakeStation(); send_candidates(station, prepared, output)
                for endpoint, payload in station.calls:
                    if endpoint == "/api/finds":
                        self.assertTrue(payload["evaluationOnly"])
                        self.assertEqual(payload["metadata"]["synthetic"], flags.get("synthetic", False))
            for key in ("evaluationOnly", "synthetic"):
                data = document(path); data[key] = "true"
                with self.assertRaises(ValueError):
                    prepare_candidates(data, path, Path(folder) / ("invalid-" + key))

    def test_cli_classification_failure_returns_nonzero_without_losing_uploaded_crop(self):
        class MissingClassifier(FakeStation):
            def request(self, path, data=None, **kwargs):
                if path == "/api/vision/classify":
                    raise RuntimeError("Fixture classifier unavailable")
                return super().request(path, data, **kwargs)
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder); report_path = Path(folder) / "detector.json"; output = Path(folder) / "out"
            report_path.write_text(json.dumps(document(path)), encoding="utf-8")
            with patch("adapters.detections_to_review.Station", return_value=MissingClassifier()):
                code = main(["--detections", str(report_path), "--image", str(path), "--output-dir", str(output)])
            self.assertEqual(code, 1)
            result = json.loads((output / "review-results.json").read_text())
            self.assertEqual(result["uploadedCount"], 2)
            self.assertEqual(result["classificationErrorCount"], 2)

    def test_color_cues_remain_habitat_and_do_not_request_species_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder); data = document(path); output = Path(folder) / "color"
            data.update(schema=COLOR_SCHEMA, colorOnly=True)
            data["candidates"] = data["candidates"][:1]
            data["candidates"][0].update(contrastScore=0.7, deltaE=14, measuredRange={"valueM": 20},
                illustrativeDistanceM=100, species="False supplied label", cueType="detector",
                investigation={"routingCategory": "immediate_investigation"})
            report = prepare_candidates(data, path, output, evaluation_only=True)
            station = FakeStation(); sent = send_candidates(station, report, output)
            self.assertEqual(sent["classifiedCount"], 0)
            self.assertEqual(len(station.calls), 1)
            payload = station.calls[0][1]; meta = payload["metadata"]
            self.assertEqual(meta["cueType"], "color_salience")
            self.assertEqual(meta["investigation"]["routingCategory"], "habitat_followup")
            self.assertEqual(meta["investigation"]["rangeEvidence"]["status"], "unknown")
            self.assertEqual(meta["colorEvidence"]["maxDeltaE"], 14)
            self.assertFalse(payload["rights"]["trainingOptIn"])
            self.assertNotIn("measuredRange", meta); self.assertNotIn("illustrativeDistanceM", meta)
            self.assertNotIn("species", meta)
            self.assertFalse(meta["frameCaptureGeometry"]["physicalCloseCapture"])
            self.assertFalse(meta["frameCaptureGeometry"]["closeFrameRequest"]["digitalCropCountsAsPhysicalVisit"])

    def test_detector_color_evidence_is_bounded_and_range_is_never_inferred_from_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder); data = document(path); output = Path(folder) / "evidence"
            data["candidates"] = data["candidates"][:1]
            data["candidates"][0].update(colorEvidence={"method": "lab-local-contrast-v1", "maxDeltaE": 20,
                "meanDeltaE": 8, "contrastScore": 0.5, "enhancedViewUsed": True,
                "scoreMeaning": "Species probability", "latitude": 44.1}, measuredRange={"valueM": 10})
            report = prepare_candidates(data, path, output)
            meta = report["crops"][0]["metadata"]
            self.assertEqual(meta["investigation"]["routingCategory"], "unlocated_followup")
            self.assertEqual(meta["frameCaptureGeometry"]["targetWidthFractionOfOriginalFrame"], 0.4)
            self.assertNotIn("latitude", meta["colorEvidence"])
            self.assertIn("not mushroom/species probability", meta["colorEvidence"]["scoreMeaning"])
            data["candidates"][0]["colorEvidence"]["maxDeltaE"] = float("nan")
            rejected = prepare_candidates(data, path, Path(folder) / "bad-evidence")
            self.assertEqual(rejected["preparedCount"], 0)
            self.assertEqual(rejected["rejections"][0]["reason"], "invalid_cue_evidence")

    def test_color_schema_requires_declaration_and_unknown_cue_types_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder); data = document(path)
            data.update(schema=COLOR_SCHEMA)
            with self.assertRaises(ValueError):
                prepare_candidates(data, path, Path(folder) / "missing-color-declaration")
            data = document(path); data["candidates"] = data["candidates"][:1]
            data["candidates"][0]["cueType"] = "confirmed_mushroom"
            report = prepare_candidates(data, path, Path(folder) / "bad-type")
            self.assertEqual(report["preparedCount"], 0)

    def test_parent_manifest_credit_survives_crop_without_location_or_label_transfer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder); data = document(path); output = Path(folder) / "credited"
            manifest = Path(folder) / "manifest.json"
            manifest.write_text(json.dumps({"images": [{"sha256": data["image"]["sha256"],
                "photoLicense": {"attribution": "Fixture Photographer", "url": "https://creativecommons.org/licenses/by/4.0/"},
                "sourceUrl": "https://example.org/photo/1", "latitude": 44.1, "longitude": -85.2,
                "taxon": {"name": "Not a human label"}}]}), encoding="utf-8")
            prepared = prepare_candidates(data, path, output, source_manifest=manifest)
            station = FakeStation(); send_candidates(station, prepared, output, classify=False)
            for endpoint, payload in station.calls:
                if endpoint != "/api/finds": continue
                metadata = payload["metadata"]
                self.assertEqual(metadata["attribution"], "Fixture Photographer")
                self.assertEqual(metadata["photoLicense"]["url"], metadata["licenseUrl"])
                self.assertEqual(metadata["sourceUrl"], "https://example.org/photo/1")
                self.assertNotIn("latitude", metadata); self.assertNotIn("longitude", metadata); self.assertNotIn("taxon", metadata)
                self.assertFalse(payload["rights"]["trainingOptIn"])
            manifest.write_text(json.dumps({"images": [{"sha256": "b" * 64}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_candidates(document(path), path, Path(folder) / "unmatched", source_manifest=manifest)

    def test_padding_is_clamped_score_ranked_bounded_and_rejects_invalid_boxes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder); output = Path(folder) / "crops"
            report = prepare_candidates(document(path), path, output, max_candidates=1, padding=0.5)
            self.assertEqual(report["preparedCount"], 1)
            self.assertEqual(report["validCandidatesOmittedByLimit"], 1)
            self.assertEqual(len(report["rejections"]), 1)
            crop = report["crops"][0]
            self.assertEqual(crop["metadata"]["cropBoxPixelsXYXY"], [0, 0, 60, 45])
            with Image.open(output / crop["file"]) as image:
                self.assertEqual(image.size, (60, 45)); self.assertEqual(image.format, "JPEG"); self.assertFalse(image.getexif())
            with self.assertRaises(FileExistsError):
                prepare_candidates(document(path), path, output)

    def test_hash_dimensions_schema_and_nonfinite_boxes_are_checked_before_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder)
            for index, mutate in enumerate((lambda d: d.update(schema="unsupported"),
                    lambda d: d["image"].update(sha256="a" * 64), lambda d: d["image"].update(width=99))):
                data = document(path); mutate(data)
                with self.subTest(index=index), self.assertRaises(ValueError):
                    prepare_candidates(data, path, Path(folder) / f"invalid-{index}")
            data = document(path); data["candidates"][0]["confidence"] = float("nan")
            report = prepare_candidates(data, path, Path(folder) / "valid")
            self.assertEqual(report["preparedCount"], 1)
            self.assertEqual(len(report["rejections"]), 2)

    def test_exif_oriented_dimensions_match_detector_and_source_path_is_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder, orientation=6)
            data = document(path, width=80, height=100)
            data["candidates"] = data["candidates"][:1]
            report = prepare_candidates(data, path, Path(folder) / "oriented", padding=0)
            self.assertEqual(report["crops"][0]["metadata"]["sourceImageWidth"], 80)
            self.assertEqual(report["crops"][0]["metadata"]["sourceImageHeight"], 100)

    def test_real_bridge_payload_preserves_parent_grouping_and_never_creates_human_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            path = source(folder); output = Path(folder) / "crops"
            prepared = prepare_candidates(document(path), path, output, session_id="survey-fixture", evaluation_only=True)
            station = FakeStation(); report = send_candidates(station, prepared, output)
            self.assertEqual(report["uploadedCount"], 2); self.assertEqual(report["classifiedCount"], 2)
            uploads = [data for endpoint, data in station.calls if endpoint == "/api/finds"]
            self.assertEqual(len({row["groupId"] for row in uploads}), 1)
            for row in uploads:
                self.assertEqual(row["sessionId"], "survey-fixture")
                self.assertEqual(row["metadata"]["parentImageSha256"], prepared["parentImageSha256"])
                self.assertEqual(row["metadata"]["sharedGroupId"], row["groupId"])
                self.assertIsNone(row["captureTimestamp"]); self.assertTrue(row["evaluationOnly"])
                self.assertFalse(row["rights"]["trainingOptIn"]); self.assertFalse(row["metadata"]["depthKnown"])
            self.assertFalse(any(endpoint.endswith("/reviews") for endpoint, _ in station.calls))
            self.assertEqual(json.loads((output / "review-results.json").read_text())["uploadedCount"], 2)


if __name__ == "__main__":
    unittest.main()
