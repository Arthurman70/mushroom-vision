import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import unittest
import uuid

from scripts.acquire_fungi_round4 import (
    TARGETS, add_index, duplicate_reason, identity_reason, prior_index,
    select_record, verify_manifest,
)
from vision.acquire_public import digest


def empty_index():
    return {key: set() for key in ("photos", "observations", "observers", "bytes", "pixels", "perceptual")}


def observation():
    return {"id": 11, "quality_grade": "research", "license_code": "cc-by", "user": {"id": 22},
            "taxon": {"id": 101, "ancestor_ids": [100], "name": "Test fungus", "rank": "species"},
            "photos": [{"id": 31, "license_code": "cc-by", "attribution": "Test observer (CC BY)",
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/31/square.jpg"},
                       {"id": 32, "license_code": "cc0", "attribution": "Test observer (CC0)",
                        "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/32/square.jpg"}]}


QUERY = {"taxon_id": 100, "classId": "test_fungi"}


@contextmanager
def temporary_directory():
    # Windows sandbox ACLs do not support TemporaryDirectory's private mode.
    base = Path(__file__).resolve().parent
    path = base / ("acquire-v4-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        if path.resolve().parent != base or not path.name.startswith("acquire-v4-test-"):
            raise ValueError("Cleanup outside the test workspace")
        shutil.rmtree(path)


class AcquisitionTests(unittest.TestCase):
    def test_default_budget_is_200(self):
        self.assertEqual(sum(q[2] for q in TARGETS), 200)

    def test_secondary_original_with_no_truth_created(self):
        row = select_record(observation(), QUERY)
        self.assertEqual(row["photoId"], 32)
        self.assertTrue(row["downloadUrl"].endswith("/32/original.jpg"))
        self.assertEqual(row["annotations"], [])
        self.assertFalse(row["humanVerified"])
        self.assertFalse(row["trainingEligible"])
        self.assertIsNone(row["distanceM"])
        self.assertEqual(row["context"], "unknown")
        self.assertEqual(row["acquisitionPhotoOrdinal"], 1)

    def test_nonderivative_and_noncommercial_excluded(self):
        for code in ("cc-by-nc", "cc-by-nd", "all-rights-reserved", None):
            obs = observation()
            for photo in obs["photos"]:
                photo["license_code"] = code
            self.assertIsNone(select_record(obs, QUERY))

    def test_unsafe_host_and_flagged_photo_excluded(self):
        obs = observation()
        obs["photos"][0]["url"] = "https://evil.example/photos/31/square.jpg"
        obs["photos"][1]["flags"] = ["copyright"]
        self.assertIsNone(select_record(obs, QUERY))

    def test_source_observation_license_required_separately(self):
        obs = observation()
        obs["license_code"] = "cc-by-nc"
        self.assertIsNone(select_record(obs, QUERY))

    def test_prior_manifest_variants_and_alias_hashes(self):
        with temporary_directory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"images": [{"photoId": 3, "observationId": 4, "observerId": 5,
                "sha256": "bytes", "originalSha256": "original", "normalizedPixelSha256": "pixels", "perceptualDifferenceHash": "000000000000000f"}]}))
            index, evidence = prior_index([path])
            self.assertEqual(index["bytes"], {"bytes", "original"})
            self.assertEqual(index["observers"], {5})
            self.assertEqual(evidence[0]["recordsIndexed"], 1)
            path.write_text(json.dumps({"images": 500}))
            self.assertEqual(prior_index([path])[1], [])

    def test_photographer_exclusion_protects_earlier_holdouts(self):
        index = empty_index()
        index["observers"].add(22)
        self.assertEqual(identity_reason(select_record(observation(), QUERY), index), "existing_observerId")

    def test_perceptual_alias_is_excluded_at_distance_two(self):
        index = empty_index()
        index["perceptual"].add(0)
        row = {"sha256": "new", "normalizedPixelSha256": "new", "perceptualDifferenceHash": "0000000000000003"}
        self.assertEqual(duplicate_reason(row, index), "conservative_perceptual_alias")
        row["perceptualDifferenceHash"] = "0000000000000007"
        self.assertIsNone(duplicate_reason(row, index))

    def test_freeze_verifies_bytes_and_rejects_invented_ground_truth(self):
        with temporary_directory() as directory:
            root = Path(directory)
            (root / "images").mkdir()
            row = select_record(observation(), QUERY)
            payload = b"test image bytes"
            (root / row["path"]).write_bytes(payload)
            row.update(sha256=digest(payload), normalizedPixelSha256="pixels", perceptualDifferenceHash="1234567890abcdef",
                       bytes=len(payload), split="train")
            manifest = {"images": [row]}
            self.assertTrue(verify_manifest(manifest, root, empty_index())["valid"])
            unsafe = copy.deepcopy(manifest)
            unsafe["images"][0]["humanVerified"] = True
            with self.assertRaisesRegex(ValueError, "training truth"):
                verify_manifest(unsafe, root, empty_index())
            (root / row["path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_manifest(manifest, root, empty_index())


if __name__ == "__main__":
    unittest.main()
