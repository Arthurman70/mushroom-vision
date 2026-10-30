"""Contract and holdout checks for the isolated AI candidate; no model training."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image, PngImagePlugin

from experiments.train_ai_review_candidate import digest, load_records


class AiCandidateContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='mushroom-ai-contract-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.collection = self.root / 'collection'
        (self.collection / 'frames').mkdir(parents=True)
        self.learning = self.root / 'learning'
        (self.learning / 'datasets').mkdir(parents=True)
        self.policy_file = self.root / 'partitions.json'
        self.export_file = self.root / 'export.json'
        self.sources = [dict(id=key, sourceGroupId='source-' + key, videoHash=character * 64,
                             licenseAllowsTraining=True, evaluationOnly=False)
                        for key, character in [('a', 'a'), ('b', 'b'), ('v', 'c'), ('t', 'd')]]
        self.policy = {'schema': 'mycoscout.ai-review-partitions.v1',
                       'sourceGroups': {'source-a': 'train', 'source-b': 'train', 'source-v': 'validation', 'source-t': 'test'}}
        self.frames = []
        for i, source in enumerate(['a', 'a', 'b', 'b', 'v', 't']):
            image_path = self.collection / 'frames' / (str(i) + '.png')
            Image.new('RGB', (8, 6), (i * 35, 100, 170)).save(image_path)
            self.frames.append({'id': 'f' + str(i), 'sourceId': source, 'path': 'frames/' + image_path.name,
                                'sha256': digest(image_path.read_bytes()), 'width': 8, 'height': 6})
        self.manifest = {'schema': 'mycoscout.video-collection.v1', 'sources': self.sources, 'frames': self.frames}
        self.reviewer = {'kind': 'ai', 'name': 'Test fixture assistant', 'model': 'fixture-model'}
        self.data = {'schema': 'mycoscout.ai-visual-review-export.v1', 'runId': 'ai-fixture',
                     'reviewer': self.reviewer, 'reviewerKind': 'ai', 'humanVerified': False,
                     'productionTrainingAllowed': False, 'automaticActivation': False,
                     'labelAuthority': 'ai-visual-review', 'records': [self.record(i) for i in range(4)]}

    def record(self, index, split='train'):
        frame = self.frames[index]
        source = next(s for s in self.sources if s['id'] == frame['sourceId'])
        positive = index % 2 == 0
        return {'schema': 'mycoscout.ai-visual-review.v1', 'runId': 'ai-fixture', 'frameId': frame['id'],
                'reviewer': copy.deepcopy(self.reviewer), 'reviewerKind': 'ai', 'humanVerified': False,
                'trainingAllowed': False, 'automaticActivation': False, 'labelAuthority': 'ai-visual-review',
                'experimentalTrainingAllowed': split == 'train', 'sourceId': source['id'],
                'sourceGroupId': source['sourceGroupId'], 'sha256': frame['sha256'], 'videoHash': source['videoHash'],
                'split': split, 'width': 8, 'height': 6, 'status': 'visible' if positive else 'absent',
                'complete': True, 'viewedOriginal': True, 'modelSuggestionsViewed': False,
                'evidence': 'Synthetic fixture only; never a real mushroom judgment.',
                'boxes': [{'bboxPixelsXYXY': [1, 1, 4, 5], 'reviewerKind': 'ai', 'humanVerified': False,
                           'confidence': .6, 'groupId': 'unknown', 'evidence': 'Synthetic box.'}] if positive else []}

    def save(self):
        (self.collection / 'manifest.json').write_text(json.dumps(self.manifest), encoding='utf-8')
        self.policy_file.write_text(json.dumps(self.policy), encoding='utf-8')
        self.export_file.write_text(json.dumps(self.data), encoding='utf-8')

    def load(self):
        self.save()
        return load_records([self.export_file], self.collection, self.policy_file, self.learning)

    def frozen(self, row):
        folder = self.learning / 'datasets' / 'release-fixture'
        folder.mkdir(exist_ok=True)
        (folder / 'manifest.json').write_text(json.dumps({'schema': 'mycoscout.reviewed-detection-dataset.v1',
                                                        'records': [row]}), encoding='utf-8')

    def test_valid_ai_records_and_validation_keep_separate_authority(self):
        self.data['records'].append(self.record(4, 'validation'))
        rows, evidence = self.load()
        self.assertEqual(len(rows), 5)
        self.assertEqual(len(evidence), 1)
        self.assertTrue(all(row['humanVerified'] is False and row['trainingAllowed'] is False for row in rows))
        self.assertFalse(rows[-1]['experimentalTrainingAllowed'])

    def test_frozen_test_source_cannot_be_renamed_validation(self):
        self.data['records'].append(self.record(4, 'validation'))
        self.frozen({'split': 'test', 'groupKeys': ['source:source-v'], 'originalSha256': 'e' * 64})
        with self.assertRaisesRegex(ValueError, 'reservation'):
            self.load()

    def test_frozen_validation_source_cannot_enter_training(self):
        self.frozen({'split': 'validation', 'groupKeys': ['source:source-a'], 'originalSha256': 'e' * 64})
        with self.assertRaisesRegex(ValueError, 'reservation'):
            self.load()

    def test_frozen_hash_blocks_whole_related_source(self):
        self.frozen({'split': 'test', 'groupKeys': [], 'originalSha256': self.frames[1]['sha256']})
        # f0 must be blocked too, before the explicitly reserved f1 is reached.
        with self.assertRaisesRegex(ValueError, 'reservation'):
            self.load()

    def test_manifest_duplicate_connects_entire_test_source(self):
        self.frames[-1]['sha256'] = self.frames[0]['sha256']
        with self.assertRaisesRegex(ValueError, 'reservation'):
            self.load()

    def test_decoded_frozen_pixels_survive_a_different_original_hash(self):
        with Image.open(self.collection / self.frames[0]['path']) as image:
            image = image.convert('RGB')
            pixel_hash = digest(json.dumps(list(image.size), separators=(',', ':')).encode() + image.tobytes())
        self.frozen({'split': 'test', 'groupKeys': [], 'originalSha256': 'e' * 64, 'pixelSha256': pixel_hash})
        with self.assertRaisesRegex(ValueError, 'Decoded pixels'):
            self.load()

    def test_reencoded_duplicates_within_experiment_are_rejected(self):
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text('fixture', 'Different encoding, same decoded RGB')
        second_path = self.collection / self.frames[2]['path']
        with Image.open(self.collection / self.frames[0]['path']) as image:
            image.save(second_path, pnginfo=metadata)
        self.frames[2]['sha256'] = digest(second_path.read_bytes())
        self.data['records'][2]['sha256'] = self.frames[2]['sha256']
        with self.assertRaisesRegex(ValueError, 'Duplicate decoded pixels'):
            self.load()

    def test_authority_observation_and_provenance_are_strict(self):
        original = copy.deepcopy(self.data)
        changes = [{'humanVerified': True}, {'reviewerKind': 'human'}, {'trainingAllowed': True},
                   {'automaticActivation': True}, {'complete': 'yes'}, {'viewedOriginal': False},
                   {'modelSuggestionsViewed': 'false'}, {'runId': 'another-run'}, {'experimentalTrainingAllowed': False},
                   {'width': 9}, {'evidence': ''}]
        for change in changes:
            with self.subTest(change=change):
                self.data = copy.deepcopy(original)
                self.data['records'][0].update(change)
                with self.assertRaises(ValueError):
                    self.load()

    def test_invalid_boxes_nonfinite_numbers_and_negative_truth_rejected(self):
        original = copy.deepcopy(self.data)
        changes = [{'bboxPixelsXYXY': [True, 1, 4, 5]}, {'bboxPixelsXYXY': [1, 1, 9, 5]},
                   {'bboxPixelsXYXY': [1, 1, float('nan'), 5]}, {'confidence': float('inf')},
                   {'confidence': True}, {'humanVerified': True}, {'reviewerKind': 'human'}, {'evidence': ''}]
        for change in changes:
            with self.subTest(change=change):
                self.data = copy.deepcopy(original)
                self.data['records'][0]['boxes'][0].update(change)
                with self.assertRaises(ValueError):
                    self.load()
        self.data = copy.deepcopy(original)
        self.data['records'][0]['status'] = 'absent'
        with self.assertRaisesRegex(ValueError, 'Presence'):
            self.load()

    def test_changed_pixels_and_collection_path_escape_rejected(self):
        path = self.collection / self.frames[0]['path']
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes changed'):
            self.load()
        self.frames[0]['path'] = '../outside.png'
        with self.assertRaisesRegex(ValueError, 'escaped'):
            self.load()

    def test_missing_or_unreadable_frozen_dataset_fails_closed(self):
        self.save()
        with self.assertRaisesRegex(ValueError, 'directory is missing'):
            load_records([self.export_file], self.collection, self.policy_file, self.root / 'missing')
        self.frozen({'split': 'mystery', 'groupKeys': []})
        with self.assertRaisesRegex(ValueError, 'Invalid frozen partition'):
            self.load()

    def test_tiny_or_single_source_sample_is_rejected(self):
        self.data['records'] = self.data['records'][:2]
        with self.assertRaisesRegex(ValueError, 'four frames'):
            self.load()

    def test_actual_production_loader_rejects_ai_checkpoint_before_model_creation(self):
        from vision import training_pipeline
        fake_torch = SimpleNamespace(load=Mock(return_value={
            'schema': 'mushroom-vision.ai-candidate-checkpoint.v1',
            'architecture': training_pipeline.MODEL_NAME, 'humanVerified': False,
            'activationEligible': False, 'state_dict': {}}))
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(training_pipeline, 'build_model') as builder:
            with self.assertRaisesRegex(ValueError, 'Unsupported reviewed detector checkpoint'):
                training_pipeline.load_detector(self.root / 'fixture-candidate.pt')
            builder.assert_not_called()
        fake_torch.load.assert_called_once_with(self.root / 'fixture-candidate.pt', map_location='cpu', weights_only=True)


if __name__ == '__main__':
    unittest.main()
