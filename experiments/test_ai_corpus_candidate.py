"""Integrity checks for the provisional AI corpus fitting path."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
import uuid

from PIL import Image

from experiments.train_ai_corpus_candidate import (
    SCHEMA, balanced_order, digest, load_records, negative_regions,
    pixel_digest, unmatched_indices,
)


class CorpusFixture(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent/('ai-corpus-test-'+uuid.uuid4().hex)
        if not self.root.resolve().is_relative_to(Path(__file__).resolve().parent):
            raise ValueError('Test cleanup path is outside experiments workspace')
        self.root.mkdir()
        self.addCleanup(lambda: shutil.rmtree(self.root))
        (self.root/'data').mkdir()
        self.inventory = {'schema': 'mycoscout.bulk-ai-inventory.v1', 'records': []}
        self.ledger = {'schema': 'mycoscout.bulk-ai-review-ledger.v1', 'reviewerKind': 'ai',
                       'humanVerified': False, 'records': []}
        self.export = {'schema': SCHEMA, 'reviewerKind': 'ai', 'humanVerified': False,
                       'activationEligible': False, 'records': []}
        for i in range(9):
            image = Image.new('RGB', (64, 48), (i*20, 20, 50))
            path = self.root/'data'/f'{i}.png'; image.save(path)
            split = 'train' if i < 8 else 'validation'
            group = f'group-{i%2}' if i < 8 else 'validation-group'
            identity = f'image-{i}'
            source = {'id': identity, 'sha256': digest(path), 'pixelSha256': pixel_digest(image),
                      'path': f'data/{i}.png', 'width': 64, 'height': 48, 'sourceGroupId': group,
                      'split': split, 'groupKeys': [group, f'image-key-{i}'],
                      'rights': {'experimentalTrainingPermission': True},
                      'experimentalTrainingEligible': split == 'train'}
            self.inventory['records'].append(source)
            review = {'id': f'review-{i}', 'imageId': identity,
                      'ledgerRecordHash': hashlib.sha256(str(i).encode()).hexdigest(),
                      'createdAt': '2026-10-29T12:00:00Z', 'reviewerKind': 'ai', 'humanVerified': False,
                      'status': 'visible' if i < 4 else 'absent', 'complete': True,
                      'viewedOriginal': True, 'observation': {'mode': 'original-image'},
                      'evidence': 'Directly inspected original image.',
                      'boxes': [{'bboxPixelsXYXY': [10, 10, 30, 35]}] if i < 4 else [],
                      'confounders': []}
            self.ledger['records'].append(review)
            self.export['records'].append({**copy.deepcopy(review), 'id': identity,
                'imagePath': source['path'], 'imageSha256': source['sha256'],
                'pixelSha256': source['pixelSha256'], 'split': split, 'sourceGroupId': group,
                'groupKeys': source['groupKeys'], 'activationEligible': False})

    def write(self):
        def save(name, value):
            path = self.root/name
            path.write_text(json.dumps(value), encoding='utf-8')
            return path
        inventory_path = save('inventory.json', self.inventory)
        self.ledger['inventorySha256'] = digest(inventory_path)
        ledger_path = save('ledger.json', self.ledger)
        self.export['originalInventorySha256'] = digest(inventory_path)
        self.export['ledgerSha256'] = digest(ledger_path)
        export_path = save('export.json', self.export)
        return export_path, inventory_path, ledger_path

    def load(self):
        return load_records(*self.write(), root=self.root)

    def test_complete_originals_and_source_holdout(self):
        rows, provenance = self.load()
        self.assertEqual(9, len(rows))
        self.assertEqual(8, sum(row['split'] == 'train' for row in rows))
        self.assertEqual({}, provenance['skipped'])

    def test_labels_cannot_differ_from_immutable_review(self):
        self.export['records'][0]['boxes'][0]['bboxPixelsXYXY'][0] = 9
        with self.assertRaisesRegex(ValueError, 'differs from immutable'):
            self.load()

    def test_newer_uncertain_review_invalidates_old_complete_export(self):
        newer = {**copy.deepcopy(self.ledger['records'][0]), 'id': 'newer', 'ledgerRecordHash': 'f'*64,
                 'createdAt': '2026-10-30T12:00:00Z', 'status': 'uncertain', 'complete': False, 'boxes': []}
        self.ledger['records'].append(newer)
        with self.assertRaisesRegex(ValueError, 'superseded'):
            self.load()

    def test_link_to_reserved_source_cannot_fit(self):
        reserved = copy.deepcopy(self.inventory['records'][0])
        reserved.update(id='reserved', split='test')
        self.inventory['records'].append(reserved)
        with self.assertRaisesRegex(ValueError, 'crosses frozen'):
            self.load()

    def test_contact_sheet_cannot_fit(self):
        for item in (self.ledger['records'][0], self.export['records'][0]):
            item['viewedOriginal'] = False
            item['observation']['mode'] = 'contact-sheet'
        with self.assertRaisesRegex(ValueError, 'superseded'):
            self.load()

    def test_file_bytes_cannot_change(self):
        (self.root/'data/0.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.load()


class NegativeAndMatchingTests(unittest.TestCase):
    def test_empty_negative_proposals_get_background_without_changing_positives(self):
        import torch
        from experiments.train_ai_corpus_candidate import ensure_background_proposals
        empty = torch.zeros((0, 4))
        existing = torch.tensor([[1., 2., 20., 30.]])
        result, count = ensure_background_proposals([empty, empty, existing], [(40, 60)]*3,
            [{"boxes": empty}, {"boxes": existing}, {"boxes": empty}])
        self.assertEqual(1, count)
        self.assertEqual([[0., 0., 60., 40.]], result[0].tolist())
        self.assertEqual([], result[1].tolist())
        self.assertIs(existing, result[2])

    def test_rounding_crop_must_not_cover_fungus(self):
        row = {'id': 'x', 'width': 100, 'height': 100,
               'boxes': [{'bboxPixelsXYXY': [30.5, 0, 45, 30]}],
               'confounders': [{'bboxPixelsXYXY': [0, 0, 30.4, 30], 'kind': 'stump', 'explicitNonFungal': True}]}
        regions, rejected = negative_regions(row)
        self.assertEqual([], regions)
        self.assertEqual(1, rejected)

    def test_tags_do_not_create_negatives(self):
        row = {'id': 'x', 'width': 100, 'height': 100, 'boxes': [], 'confounderTags': ['hat', 'tree'],
               'confounders': [{'bboxPixelsXYXY': [0, 0, 30, 30], 'kind': 'tree', 'explicitNonFungal': False}]}
        self.assertEqual(([], 0), negative_regions(row))

    def test_balancing_visits_every_example_once(self):
        examples = [{'id': str(i), 'positive': i < 3, 'row': {'sourceGroupId': str(i%2)}} for i in range(8)]
        result = balanced_order(examples, 4)
        self.assertEqual(set(str(i) for i in range(8)), {row['id'] for row in result})
        self.assertEqual(8, len(result))
        self.assertEqual([True, False, True, False, True, False], [row['positive'] for row in result[:6]])

    def test_duplicate_detections_count_as_unmatched(self):
        predictions = [{'bbox': [0, 0, 20, 20]}, {'bbox': [0, 0, 20, 20]}]
        self.assertEqual(1, len(unmatched_indices([[0, 0, 20, 20]], predictions)))


if __name__ == '__main__':
    unittest.main()
