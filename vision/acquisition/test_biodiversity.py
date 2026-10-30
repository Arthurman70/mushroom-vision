"""Boundary checks for source labels and split-preserving expansion."""
import unittest

from adapters.acquire_wildlife_context import select
from vision.acquire_biodiversity import classify


class BiodiversityTests(unittest.TestCase):
    def test_specific_species_takes_priority_over_genus(self):
        queries = [{"taxon_id": 1, "classId": "genus", "rank": "genus"},
                   {"taxon_id": 2, "classId": "species", "rank": "species"}]
        self.assertEqual(classify({1, 2}, queries), "species")
        self.assertEqual(classify({99}, queries), "other_fungi")

    def test_ambiguous_same_rank_is_not_silently_assigned(self):
        with self.assertRaises(ValueError):
            classify({1}, [{"taxon_id": 1, "classId": "a", "rank": "genus"},
                           {"taxon_id": 1, "classId": "b", "rank": "genus"}])

    def test_empty_wildlife_is_not_a_mushroom_negative(self):
        metadata = {"categories": [{"id": 1, "name": "empty"}],
                    "annotations": [{"image_id": "sample", "category_id": 1}],
                    "images": [{"id": "sample", "location": "camera1", "seq_id": "seq1",
                                "rights_holder": "Example", "file_name": "00000000-0000-0000-0000-000000000000.jpg"}]}
        rows = select(metadata, {"splits": {"train": [], "val": ["camera1"]}}, 6)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["split"], "validation")
        self.assertEqual(rows[0]["mushroomVisibility"], "unknown")
        self.assertFalse(rows[0]["trainingEligibleForMushroomDetection"])
        self.assertEqual(rows[0]["annotations"], [])

    def test_source_camera_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            select({"categories": [], "annotations": [], "images": []},
                   {"splits": {"train": ["same"], "val": ["same"]}}, 2)


if __name__ == "__main__":
    unittest.main()
