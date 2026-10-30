from contextlib import contextmanager
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
from adapters.export_testbench_dataset import export_dataset, records_from

ROOT = Path(__file__).resolve().parents[2]
RUN_ID = "11111111-1111-4111-8111-111111111111"
ANNOTATION_ID = "22222222-2222-4222-8222-222222222222"


def fixture(*, negative=False, exif=False, color=(30, 70, 90)):
    stream = io.BytesIO()
    im = Image.new("RGB", (100, 80), color)
    if exif:
        metadata = Image.Exif(); metadata[274] = 6
        im.save(stream, format="JPEG", exif=metadata)
    else:
        im.save(stream, format="PNG")
    payload = stream.getvalue(); digest = hashlib.sha256(payload).hexdigest()
    width, height = (80, 100) if exif else (100, 80)
    row = {"runId": RUN_ID, "sourceRunId": None, "sceneId": "scene-" + digest,
        "evaluationOnly": False, "partition": "test",
        "image": {"sha256": digest, "width": width, "height": height,
                  "mimeType": "image/jpeg" if exif else "image/png", "bytes": len(payload),
                  "orientationVerified": True, "url": f"/api/testbench/runs/{RUN_ID}/image"},
        "annotation": {"id": ANNOTATION_ID, "revision": 1, "sequence": 1,
            "createdAt": "2026-10-28T14:00:00Z", "reviewer": "Software test fixture", "complete": True,
            "trainingAllowed": True, "trainingPermissionRequested": True,
            "rightsAttestation": "Test fixture permits photo and annotation training export.",
            "labelsFrom": "explicit_human_annotations", "emptySceneConfirmed": negative,
            "boxes": [] if negative else [{"id": "human-box-1", "bboxPixelsXYXY": [10, 20, 60, 60],
                                          "groupId": "chanterelles_trumpets", "label": "Human black-trumpet guess"}],
            "notes": "Synthetic software test; not a field identification", "measuredDistanceM": None},
        "sourceCredits": {"attribution": "Software test image"},
        "provenance": {"originalImageSha256": digest, "relatedRunIds": [RUN_ID],
                       "labelsFrom": "explicit_human_annotations", "sceneGrouping": "exact-original-sha256",
                       "nearDuplicateGroupingVerified": False}}
    return {"schema": "mycoscout.testbench-dataset.v1", "releaseId": "testbench-" + "a" * 64,
            "records": [row]}, payload


class FakeStation:
    def __init__(self, manifest, payload, final=None):
        self.initial, self.final, self.payload = manifest, final if final is not None else manifest, payload
        self.calls = []
        self.exports = 0

    def request(self, path, data=None, binary=False):
        self.calls.append((path, binary))
        if path == "/api/testbench/export":
            self.exports += 1
            return deepcopy(self.initial if self.exports == 1 else self.final)
        if path == f"/api/testbench/runs/{RUN_ID}/image" and binary:
            return self.payload
        raise AssertionError("Unexpected request path")


@contextmanager
def collection_path():
    with tempfile.TemporaryDirectory(prefix="testbench-export-", dir=ROOT / "data") as folder:
        path = Path(folder).resolve()
        assert path.parent == (ROOT / "data").resolve()
        yield path / "collection"


class TestbenchDatasetExportTests(unittest.TestCase):
    def test_video_group_provenance_survives_export_and_inconsistent_rights_fail(self):
        manifest, payload = fixture()
        row = manifest['records'][0]
        group = 'video:' + 'b' * 64
        row['sceneId'] = group
        row['provenance']['sceneGrouping'] = 'source-video-group'
        row['videoProvenance'] = {'id': 'frame-1', 'sha256': row['image']['sha256'], 'sceneId': group,
            'timestampSeconds': 12.5, 'evaluationOnly': False, 'source': {'sourceGroupId': group,
            'videoHash': 'b' * 64, 'evaluationOnly': False, 'licenseAllowsTraining': True}}
        with collection_path() as output:
            export_dataset(FakeStation(manifest, payload), output)
            result = json.loads((output / 'collection-manifest.json').read_text())['records'][0]
            self.assertEqual(result['sceneId'], group)
            self.assertEqual(result['videoProvenance']['timestampSeconds'], 12.5)
            self.assertIsNone(result['split'])
        for key, value in [('sceneId', 'other-group'), ('sha256', 'c' * 64), ('evaluationOnly', True)]:
            invalid = deepcopy(manifest)
            invalid['records'][0]['videoProvenance'][key] = value
            with self.assertRaisesRegex(ValueError, 'Video provenance'):
                records_from(invalid)
        row['videoProvenance']['source']['licenseAllowsTraining'] = False
        with self.assertRaisesRegex(ValueError, 'Video provenance'):
            records_from(manifest)

    def test_coco_yolo_human_metadata_and_no_automatic_split(self):
        manifest, payload = fixture()
        station = FakeStation(manifest, payload)
        with collection_path() as output:
            result = export_dataset(station, output)
            self.assertEqual(result["status"], "READY")
            self.assertTrue((output / "READY.json").is_file())
            self.assertFalse((output / "INCOMPLETE.json").exists())
            coco = json.loads((output / "dataset-all.json").read_text())
            self.assertEqual(coco["annotations"][0]["bbox"], [10, 20, 50, 40])
            self.assertEqual(coco["annotations"][0]["category_id"], 0)
            self.assertEqual(coco["annotations"][0]["source_annotation_id"], ANNOTATION_ID)
            self.assertEqual(coco["annotations"][0]["human_label"], "Human black-trumpet guess")
            digest = manifest["records"][0]["image"]["sha256"]
            self.assertEqual((output / "labels" / (digest + ".txt")).read_text(), "0 0.35 0.5 0.5 0.5\n")
            collection = json.loads((output / "collection-manifest.json").read_text())
            self.assertIsNone(collection["records"][0]["split"])
            self.assertEqual(collection["records"][0]["sourceSuggestedPartition"], "test")
            self.assertFalse(collection["policies"]["automaticSplit"])
            self.assertFalse(collection["policies"]["automaticTraining"])
            self.assertEqual(collection["records"][0]["originalImageSha256"], digest)
            self.assertEqual(station.exports, 2)
            self.assertEqual([call[0] for call in station.calls], ["/api/testbench/export",
                              f"/api/testbench/runs/{RUN_ID}/image", "/api/testbench/export"])

    def test_explicit_empty_scene_generates_empty_yolo_file_and_coco_negative(self):
        manifest, payload = fixture(negative=True)
        with collection_path() as output:
            export_dataset(FakeStation(manifest, payload), output)
            coco = json.loads((output / "dataset-all.json").read_text())
            self.assertEqual(len(coco["images"]), 1)
            self.assertEqual(coco["annotations"], [])
            self.assertEqual(next((output / "labels").glob("*.txt")).read_bytes(), b"")
            collection = json.loads((output / "collection-manifest.json").read_text())
            self.assertEqual(collection["confirmedNegativeImages"], 1)

    def test_unconfirmed_empty_annotations_and_permission_or_completion_denials_reject(self):
        changes = [{"trainingAllowed": False}, {"complete": False}, {"rightsAttestation": None},
                   {"labelsFrom": "model_guesses"}, {"boxes": [], "emptySceneConfirmed": False}]
        for fields in changes:
            with self.subTest(fields=fields), collection_path() as output:
                manifest, payload = fixture(); manifest["records"][0]["annotation"].update(fields)
                station = FakeStation(manifest, payload)
                with self.assertRaises(ValueError):
                    export_dataset(station, output)
                self.assertTrue((output / "INCOMPLETE.json").exists())
                self.assertFalse((output / "READY.json").exists())
                self.assertEqual(station.calls, [("/api/testbench/export", False)])

    def test_evaluation_or_missing_quarantine_status_rejected(self):
        for state in (True, None, "false"):
            with self.subTest(state=state), collection_path() as output:
                manifest, payload = fixture()
                if state is None:
                    manifest["records"][0].pop("evaluationOnly")
                else:
                    manifest["records"][0]["evaluationOnly"] = state
                with self.assertRaises(ValueError):
                    export_dataset(FakeStation(manifest, payload), output)
                self.assertFalse((output / "READY.json").exists())

    def test_cross_origin_query_encoded_traversal_and_wrong_id_paths_rejected(self):
        for url in ("https://elsewhere.example/image", "//elsewhere.example/image", "/api/testbench/runs/../image",
                    f"/api/testbench/runs/{RUN_ID}/image?token=secret", f"/api/testbench/runs/{RUN_ID}/%2e%2e/image",
                    "/api/testbench/runs/33333333-3333-4333-8333-333333333333/image"):
            with self.subTest(url=url), collection_path() as output:
                manifest, payload = fixture(); manifest["records"][0]["image"]["url"] = url
                station = FakeStation(manifest, payload)
                with self.assertRaises(ValueError):
                    export_dataset(station, output)
                self.assertEqual(station.calls, [("/api/testbench/export", False)])

    def test_image_hash_and_dimensions_are_checked_before_ready(self):
        for corrupt in ("bytes", "dimensions"):
            with self.subTest(corrupt=corrupt), collection_path() as output:
                manifest, payload = fixture()
                if corrupt == "bytes":
                    payload = b"corrupt"
                else:
                    manifest["records"][0]["image"]["width"] += 1
                with self.assertRaises(ValueError):
                    export_dataset(FakeStation(manifest, payload), output)
                self.assertTrue((output / "INCOMPLETE.json").is_file())
                self.assertFalse((output / "dataset-all.json").exists())

    def test_exif_is_applied_and_removed_without_changing_box_coordinates(self):
        manifest, payload = fixture(exif=True)
        with collection_path() as output:
            export_dataset(FakeStation(manifest, payload), output)
            image_file = next((output / "images").glob("*.png"))
            with Image.open(image_file) as image:
                self.assertEqual(image.size, (80, 100))
                self.assertFalse(image.getexif())
            coco = json.loads((output / "dataset-all.json").read_text())
            self.assertEqual(coco["annotations"][0]["bbox"], [10, 20, 50, 40])

    def test_permission_revocation_or_annotation_revision_change_leaves_quarantine(self):
        for change in ("removed", "rights", "revision", "box"):
            with self.subTest(change=change), collection_path() as output:
                manifest, payload = fixture(); final = deepcopy(manifest)
                if change == "removed":
                    final["records"] = []
                elif change == "rights":
                    final["records"][0]["annotation"]["trainingAllowed"] = False
                elif change == "revision":
                    final["records"][0]["annotation"]["revision"] = 2
                else:
                    final["records"][0]["annotation"]["boxes"][0]["bboxPixelsXYXY"][0] = 11
                station = FakeStation(manifest, payload, final)
                with self.assertRaises(ValueError):
                    export_dataset(station, output)
                self.assertEqual(station.exports, 2)
                self.assertTrue((output / "INCOMPLETE.json").is_file())
                self.assertFalse((output / "READY.json").exists())
                self.assertFalse((output / "dataset-all.json").exists())
                self.assertTrue((output / "failure.json").is_file())

    def test_output_limit_and_existing_directory_never_overwrite(self):
        manifest, payload = fixture()
        for count in (0, 101, True):
            with self.subTest(count=count), collection_path() as output:
                with self.assertRaises(ValueError):
                    export_dataset(FakeStation(manifest, payload), output, count)
                self.assertFalse(output.exists())
        with collection_path() as output:
            output.mkdir(); (output / "keep.txt").write_text("preserve")
            station = FakeStation(manifest, payload)
            with self.assertRaises(FileExistsError):
                export_dataset(station, output)
            self.assertEqual((output / "keep.txt").read_text(), "preserve")
            self.assertEqual(station.calls, [])

    def test_duplicate_sources_bad_ids_and_invalid_boxes_reject(self):
        manifest, _ = fixture()
        repeated = deepcopy(manifest); repeated["records"].append(deepcopy(repeated["records"][0]))
        with self.assertRaises(ValueError):
            records_from(repeated)
        for change in ("id", "box", "scene", "provenance"):
            item = deepcopy(manifest)
            if change == "id": item["records"][0]["runId"] = "../../escape"
            if change == "box": item["records"][0]["annotation"]["boxes"][0]["bboxPixelsXYXY"] = [0, 0, 200, 200]
            if change == "scene": item["records"][0]["sceneId"] = "untrusted-scene"
            if change == "provenance": item["records"][0]["provenance"]["originalImageSha256"] = "b" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                records_from(item)

    def test_empty_export_is_not_marked_ready_and_budget_failure_is_quarantined(self):
        manifest, payload = fixture(); manifest["records"] = []
        with collection_path() as output:
            with self.assertRaises(ValueError):
                export_dataset(FakeStation(manifest, payload), output)
            self.assertFalse((output / "READY.json").exists())
        manifest, payload = fixture()
        with collection_path() as output, patch("adapters.export_testbench_dataset.MAX_COLLECTION_BYTES", 1):
            with self.assertRaises(ValueError):
                export_dataset(FakeStation(manifest, payload), output)
            self.assertTrue((output / "INCOMPLETE.json").exists())
            self.assertFalse((output / "READY.json").exists())


if __name__ == "__main__":
    unittest.main()
