"""Public export contracts using generated geometry, never private run artifacts."""
import copy
import hashlib
from pathlib import Path
import unittest
import uuid
from PIL import Image

from experiments.evaluate_full_pixel_v5 import fractional_canvas, pass_geometry
from experiments.publish_full_pixel_v5 import (
    copy_exact, digest, public_frames, public_geometry, public_record, public_recovery, public_result, validate_receipt,
)


def protocol_fixture():
    """Only metadata used by projection contracts; no image or model files needed."""
    source = Image.new("RGB", (200, 100), "red")
    box = [50., 10., 150., 90.]
    source_hash = hashlib.sha256(source.tobytes()).hexdigest()
    reference = {"frameId": "synthetic-export-fixture", "cohort": "ai-photo",
        "referenceAuthority": "ai", "referenceComplete": False, "isNegativeControl": False,
        "imageSha256": source_hash, "width": source.width, "height": source.height,
        "notes": "Generated fixture, not a mushroom annotation.", "targetUnit": "synthetic rectangle"}
    jobs = []
    for condition, pixels in (("native", None), ("32px", 32.), ("0p69px", .69)):
        if pixels is None:
            image, projected = source, list(box)
            geometry = {"sourceSize": list(source.size), "canvasSize": list(source.size),
                "supersampling": None, "sceneScaleXY": [1., 1.],
                "sceneBoundsPixelsXYXY": [0., 0., *source.size],
                "sceneReEnlarged": False, "actualDistanceM": None}
        else:
            image, projected, geometry = fractional_canvas(source, box, pixels, [.25, .75])
        geometry["processorAndTileCoverage"] = pass_geometry(image.size, projected)
        jobs.append({k: copy.deepcopy(reference[k]) for k in ("frameId", "cohort",
            "referenceAuthority", "referenceComplete", "isNegativeControl")})
        jobs[-1].update(conditionId=condition, imagePath=f"fixture/{condition}.png",
            sourceImageSha256=source_hash, derivedImageSha256=hashlib.sha256(image.tobytes()).hexdigest(),
            width=image.width, height=image.height, geometry=geometry,
            originalTargetPixelsXYXY=list(box), targetPixelsXYXY=projected,
            realizedTargetWidthPixels=projected[2]-projected[0],
            realizedTargetHeightPixels=projected[3]-projected[1])
    return {"references": [reference], "jobs": jobs}


class PublishPixelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = protocol_fixture()

    def record_fixture(self, native=False):
        job = next(j for j in self.protocol["jobs"] if (j["conditionId"] == "native") == native)
        record = copy.deepcopy(job)
        record["uniqueModelForwards"] = 5
        record["predictionPath"] = "private/prediction.json"
        record["methods"] = {name: {"candidates": [{"bbox": [1., 2., 3., 4.], "score": .5, "privatePath": "secret"}],
            "metrics": {"hits": 0, "targetCount": 1, "unmatchedPositiveBoxes": 1, "privatePath": "secret"},
            "seconds": 1., "modelPasses": passes, "sharedContextPasses": shared,
            "privatePath": "secret"} for name, passes, shared in (("raw", 1, 0), ("tiles", 5, 1))}
        return record, job

    def test_public_projection_strips_private_paths_and_unknown_metadata(self):
        record, job = self.record_fixture()
        result = public_record(record, job)
        self.assertNotIn("imagePath", result)
        self.assertNotIn("predictionPath", result)
        self.assertEqual(set(result["methods"]["raw"]["candidates"][0]), {"bbox", "score"})
        self.assertNotIn("privatePath", result["methods"]["raw"])
        self.assertNotIn("privatePath", result["methods"]["raw"]["metrics"])
        self.assertEqual(result["assetName"], f"pixel-view-{record['frameId']}--{record['conditionId']}.png")

    def test_native_uses_exact_published_original(self):
        record, job = self.record_fixture(native=True)
        result = public_record(record, job)
        self.assertIsNone(result["assetName"])
        self.assertEqual(result["derivedImageSha256"], record["sourceImageSha256"])

    def test_changed_input_geometry_is_rejected(self):
        record, job = self.record_fixture()
        record["targetPixelsXYXY"][0] += 1
        with self.assertRaises(ValueError): public_record(record, job)

    def test_missing_shared_context_comparison_is_rejected(self):
        record, job = self.record_fixture()
        del record["methods"]["raw"]
        with self.assertRaises(ValueError): public_record(record, job)

    def test_geometry_is_bounded_to_known_numeric_metadata(self):
        geometry = copy.deepcopy(next(j["geometry"] for j in self.protocol["jobs"] if j["conditionId"] == "0p69px"))
        geometry["privatePath"] = "secret"
        geometry["processorAndTileCoverage"]["passes"][0]["privatePath"] = "secret"
        result = public_geometry(geometry)
        self.assertNotIn("privatePath", result)
        self.assertNotIn("privatePath", result["processorAndTileCoverage"]["passes"][0])
        self.assertLess(result["realizedTargetWidthPixels"], 1)
        self.assertEqual(result["canvasSize"], [1920, 1080])

    def test_source_hash_and_credits_must_match(self):
        ref = self.protocol["references"][0]
        source = {"id": "one", "title": "Source", "attribution": "Photographer", "license": "CC BY",
                  "sourceUrl": "https://example.org/photo", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/"}
        frame = {"id": ref["frameId"], "sourceId": "one", "sha256": ref["imageSha256"], "width": ref["width"], "height": ref["height"]}
        manifest = {"sources": [source], "frames": [frame]}
        self.assertEqual(public_frames([ref], manifest)[0]["source"]["attribution"], "Photographer")
        frame["sha256"] = "0" * 64
        with self.assertRaises(ValueError): public_frames([ref], manifest)
        frame["sha256"] = ref["imageSha256"]; del source["licenseUrl"]
        with self.assertRaises(ValueError): public_frames([ref], manifest)

    def test_partial_or_unverified_run_cannot_be_published(self):
        for report in ({"status": "running"}, {"status": "completed", "inputIntegrityVerified": False},
                       {"status": "completed", "inputIntegrityVerified": True, "trainingPerformed": True}):
            with self.assertRaises(ValueError): public_result(report, self.protocol, {})

    def test_independent_receipt_must_bind_completed_report_and_protocol(self):
        receipt = {"schema": "mushroom-vision.full-pixel-independent-verification.v1", "status": "verified",
            "protocolSha256": "a" * 64, "reportSha256": "b" * 64, "scenes": 101, "uniqueModelForwards": 505,
            "uniquePredictionFiles": 101, "trainingPerformed": False, "weightsChanged": False}
        validate_receipt(receipt, "a" * 64, "b" * 64)
        for key, value in (("status", "running"), ("reportSha256", "c" * 64), ("protocolSha256", "c" * 64),
                           ("scenes", 100), ("uniqueModelForwards", 500), ("weightsChanged", True)):
            changed = {**receipt, key: value}
            with self.assertRaises(ValueError): validate_receipt(changed, "a" * 64, "b" * 64)

    def test_recovery_public_note_preserves_unknown_attempted_work(self):
        report = {"records": [{}] * 101, "recovery": {"reusedScenes": 92, "newScenes": 9,
            "uniquePersistedModelForwards": 505, "totalAttemptedModelForwards": None, "disclosure": "Interrupted run."}}
        result = public_recovery(report)
        self.assertEqual(result["totalScenes"], 101)
        self.assertEqual(result["uniquePersistedModelPasses"], 505)
        self.assertIsNone(result["totalAttemptedModelPasses"])
        self.assertFalse(result["totalAttemptedWorkKnown"])
        report["recovery"]["totalAttemptedModelForwards"] = 505
        with self.assertRaises(ValueError): public_recovery(report)

    def test_copy_preserves_bytes_and_refuses_different_existing_asset(self):
        folder = Path(__file__).resolve().parent / ("tmp-pixel-export-test-" + uuid.uuid4().hex)
        folder.mkdir()
        try:
            source, target = Path(folder) / "source", Path(folder) / "public"
            source.write_bytes(b"exact frozen image bytes")
            expected = digest(source)
            copy_exact(source, target, expected)
            self.assertEqual(target.read_bytes(), source.read_bytes())
            copy_exact(source, target, expected)
            target.write_bytes(b"different")
            with self.assertRaises(FileExistsError): copy_exact(source, target, expected)
        finally:
            for path in (folder / "source", folder / "public"):
                if path.exists(): path.unlink()
            folder.rmdir()


if __name__ == "__main__": unittest.main()
