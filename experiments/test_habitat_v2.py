import unittest
import numpy as np
from vision.habitat import summarize_semantics


class HabitatCuesTest(unittest.TestCase):
    def test_uncertain_labels_cannot_supply_habitat_evidence(self):
        r = summarize_semantics(np.ones((8, 8), dtype=int), np.full((8, 8), 0.4), {1: 'grass'}, 'context')
        self.assertEqual(r['signals']['vegetation_ground'], 0)
        self.assertFalse(r['inspectionLeadEligible'])

    def test_closeup_cannot_earn_scene_lead(self):
        r = summarize_semantics(np.ones((8, 8), dtype=int), np.ones((8, 8)), {1: 'grass'}, 'closeup')
        self.assertFalse(r['inspectionLeadEligible'])

    def test_water_is_context_not_moisture_or_clearance(self):
        r = summarize_semantics(np.ones((8, 8), dtype=int), np.ones((8, 8)), {1: 'water'}, 'context')
        self.assertEqual(r['fractions']['water'], 1)
        self.assertFalse(r['inspectionLeadEligible'])
        self.assertNotIn('damp_ground', r['signals'])
        self.assertNotIn('metric', r)

    def test_generic_ground_creates_only_weak_signals(self):
        r = summarize_semantics(np.ones((8, 8), dtype=int), np.ones((8, 8)), {1: 'earth'}, 'context')
        self.assertTrue(r['inspectionLeadEligible'])
        self.assertEqual(r['signals']['exposed_ground'], 1)
        self.assertNotIn('hardwood', r['signals'])

    def test_invalid_confidence_rejected(self):
        with self.assertRaises(ValueError):
            summarize_semantics(np.ones((8, 8), dtype=int), np.full((8, 8), np.nan), {1: 'earth'}, 'context')

    def test_invalid_threshold_rejected(self):
        for threshold in (float('nan'), float('inf'), -0.1, 1.1, True, '0.5', None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                summarize_semantics(np.ones((2, 2), dtype=int), np.ones((2, 2)), {1: 'earth'}, 'context', threshold)

    def test_invalid_or_unknown_label_ids_rejected(self):
        for labels in (np.ones((2, 2)), np.full((2, 2), -1), np.full((2, 2), 7),
                       np.full((2, 2), np.nan), np.ones((2, 2), dtype=bool)):
            with self.subTest(dtype=labels.dtype), self.assertRaises(ValueError):
                summarize_semantics(labels, np.ones((2, 2)), {1: 'earth'}, 'context')

    def test_mapping_rejects_malformed_or_duplicate_ids(self):
        for mapping in ({}, None, {1.5: 'earth'}, {-1: 'earth'}, {True: 'earth'},
                        {'bad': 'earth'}, {1: ''}, {1: None}, {1: 'earth', '1': 'grass'}):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                summarize_semantics(np.ones((2, 2), dtype=int), np.ones((2, 2)), mapping, 'context')

    def test_string_mapping_and_threshold_boundaries_are_supported(self):
        for threshold in (0, 1, np.float32(0.55)):
            result = summarize_semantics(np.ones((2, 2), dtype=int), np.ones((2, 2)), {'1': 'earth'}, 'context', threshold)
            self.assertTrue(result['inspectionLeadEligible'])
            self.assertEqual(result['fractions']['ground'], 1)


if __name__ == '__main__':
    unittest.main()
