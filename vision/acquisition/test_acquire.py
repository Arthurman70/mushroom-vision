from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from vision.acquire_public import (BLACK_TAXA, QUERIES, SCHEMA, acquire, download_one,
                                   expansion_queries, freeze_splits, photo_record, summary, write_json)


def observation():
    return {"id": 5, "quality_grade": "research", "license_code": "cc-by", "user": {"id": 8},
            "taxon": {"id": 194231, "name": "Craterellus fallax", "rank": "species", "ancestor_ids": [47170, 194231]},
            "photos": [{"id": 12, "license_code": "cc-by", "attribution": "Test author, CC BY", "url": "https://inaturalist-open-data.s3.amazonaws.com/photos/12/square.jpg"}]}


class AcquisitionTests(unittest.TestCase):
    def test_open_photo_preserves_provenance_without_coordinates(self):
        item = observation()
        item.update(location="private-looking coordinates", description="not retained")
        result = photo_record(item, QUERIES[0])
        self.assertEqual(result["weakLabel"], "black_trumpet")
        self.assertEqual(result["downloadUrl"], "https://inaturalist-open-data.s3.amazonaws.com/photos/12/large.jpg")
        self.assertEqual(result["annotations"], [])
        self.assertNotIn("location", result)
        self.assertNotIn("description", result)

    def test_observation_license_does_not_override_photo_license(self):
        item = observation()
        item["photos"][0]["license_code"] = None
        self.assertIsNone(photo_record(item, QUERIES[0]))
        item["photos"][0]["license_code"] = "cc-by-nc"
        self.assertIsNone(photo_record(item, QUERIES[0]))

    def test_closed_host_and_hidden_or_flagged_images_excluded(self):
        item = observation()
        item["photos"][0]["url"] = "https://static.inaturalist.org/photos/12/square.jpg"
        self.assertIsNone(photo_record(item, QUERIES[0]))
        item = observation()
        item["photos"][0]["hidden"] = True
        self.assertIsNone(photo_record(item, QUERIES[0]))
        item = observation()
        item["photos"][0]["flags"] = [{"flag": "copyright"}]
        self.assertIsNone(photo_record(item, QUERIES[0]))

    def test_wrong_taxon_not_mislabeled_as_target(self):
        item = observation()
        item["taxon"] = {"id": 350511, "name": "Craterellus tubaeformis", "rank": "species", "ancestor_ids": [47170, 350511]}
        self.assertIsNone(photo_record(item, QUERIES[0]))
        result = photo_record(item, QUERIES[3])
        self.assertEqual(result["weakLabel"], "other_fungi")
        self.assertEqual(result["annotationStatus"], "unreviewed")

    def test_observer_and_near_duplicate_links_share_frozen_split(self):
        rows = [{"id": "a", "groupId": "observer-1", "perceptualDifferenceHash": "0000000000000000"},
                {"id": "b", "groupId": "observer-2", "perceptualDifferenceHash": "0000000000000001"},
                {"id": "c", "groupId": "observer-2", "perceptualDifferenceHash": "ffffffffffffffff"}]
        first, second = deepcopy(rows), deepcopy(rows)
        policy = freeze_splits(first)
        self.assertEqual(policy, freeze_splits(second))
        self.assertEqual(first, second)
        self.assertEqual(len({row["splitGroupId"] for row in first}), 1)
        self.assertEqual(len({row["split"] for row in first}), 1)

    def test_expansion_preserves_inherited_split_even_when_group_root_changes(self):
        rows = [{"id": "old", "groupId": "observer-z", "perceptualDifferenceHash": "0000000000000000"},
                {"id": "new", "groupId": "observer-a", "perceptualDifferenceHash": "0000000000000001"}]
        policy = freeze_splits(rows, prior_splits={"observer-z": "test"})
        self.assertEqual([row['split'] for row in rows], ['test', 'test'])
        self.assertTrue(policy['priorSplitsPreserved'])

    def test_expansion_refuses_links_between_previously_frozen_splits(self):
        rows = [{"id": "a", "groupId": "a", "perceptualDifferenceHash": "0000000000000000"},
                {"id": "b", "groupId": "b", "perceptualDifferenceHash": "0000000000000001"}]
        with self.assertRaisesRegex(ValueError, 'previously frozen splits'):
            freeze_splits(rows, prior_splits={'a': 'train', 'b': 'test'})
        self.assertTrue(all('split' not in row for row in rows))

    def test_expansion_quota_is_bounded_and_exact(self):
        for count in (1, 9, 100, 1200, 1500):
            queries = expansion_queries(count)
            self.assertEqual(sum(query['quota'] for query in queries), count)
            self.assertTrue(all(query['quota'] > 0 for query in queries))

    def test_expansion_keeps_parent_bytes_and_splits_immutable(self):
        # No network: an empty API demonstrates copy, freeze, and provenance logic.
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder) / 'parent'
            (parent / 'images').mkdir(parents=True)
            row = photo_record(observation(), QUERIES[0])
            Image.new('RGB', (160, 120), 'brown').save(parent / row['path'])
            row = download_one(None, parent, row)
            policy = freeze_splits([row])
            manifest = {'schema': SCHEMA, 'title': 'fixture', 'status': 'frozen', 'images': [row],
                        'summary': summary([row]), 'splitPolicy': policy}
            write_json(parent / 'manifest.json', manifest)
            original = (parent / 'manifest.json').read_bytes()
            original_photo = (parent / row['path']).read_bytes()
            write_json(parent / 'manifest-sha256.json', {'sha256': hashlib.sha256(original).hexdigest()})
            destination = Path(folder) / 'expanded'
            with patch('vision.acquire_public.Client.api', return_value=({'results': []}, 'fixture:empty-api')):
                result = acquire(destination, resume_from=parent / 'manifest.json', more=9)
            self.assertEqual((parent / 'manifest.json').read_bytes(), original)
            self.assertEqual((parent / row['path']).read_bytes(), original_photo)
            self.assertEqual(result['images'][0]['split'], row['split'])
            self.assertEqual(result['parentDataset']['manifestSha256'], hashlib.sha256(original).hexdigest())
            self.assertEqual(result['acquisition']['newImagesAccepted'], 0)
            self.assertEqual((destination / row['path']).read_bytes(), original_photo)
            with self.assertRaisesRegex(ValueError, 'Frozen manifest already exists'):
                acquire(parent, resume_from=parent / 'manifest.json', more=1)

    def test_expansion_requires_paired_bounded_arguments(self):
        for kwargs in ({'more': 1}, {'resume_from': 'missing.json'}, {'resume_from': 'missing.json', 'more': 0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, '--resume-from'):
                acquire('unused', **kwargs)


if __name__ == "__main__":
    unittest.main()
