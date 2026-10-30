import unittest
from vision.export_scout_leads import candidate_leads, habitat_leads, archive_cue_evidence, COLOR_SCHEMA


class ScoutBridgeTest(unittest.TestCase):
    def test_synthetic_evaluation_provenance_is_preserved_and_strictly_boolean(self):
        sources = {'a'*64: {'width': 100, 'height': 100}}
        p = {'schema': 'mycoscout.unlocated-candidates.v1', 'synthetic': True, 'evaluationOnly': False,
             'image': {'sha256': 'a'*64, 'width': 100, 'height': 100},
             'candidates': [{'confidence': .4, 'bboxPixelsXYXY': [10, 10, 30, 40]}]}
        result = candidate_leads(p, sources)[0]
        self.assertTrue(result['synthetic']); self.assertTrue(result['evaluationOnly'])
        for key in ('synthetic', 'evaluationOnly'):
            broken = {**p, key: 1}
            with self.assertRaises(ValueError):
                candidate_leads(broken, sources)

    def test_color_only_cues_are_habitat_and_synthetic_distances_cannot_route_as_real_range(self):
        sources = {'a'*64: {'width': 100, 'height': 100}}
        p = {'schema': COLOR_SCHEMA, 'colorOnly': True, 'image': {'sha256': 'a'*64, 'width': 100, 'height': 100},
             'candidates': [{'contrastScore': .7, 'deltaE': 30, 'bboxPixelsXYXY': [1, 2, 10, 12],
                             'illustrativeDistanceM': 100, 'measuredRange': {'valueM': 10}, 'capturedMs': 50}]}
        result = candidate_leads(p, sources)[0]
        self.assertEqual(result['kind'], 'habitat')
        self.assertEqual(result['cueType'], 'color_salience')
        self.assertEqual(result['investigation']['routingCategory'], 'habitat_followup')
        self.assertEqual(result['investigation']['rangeEvidence']['status'], 'unknown')
        self.assertFalse(result['classificationEligible'])
        self.assertIsNone(result['capturedMs'])
        self.assertNotIn('measuredRange', result)
        self.assertNotIn('illustrativeDistanceM', result)

    def test_color_evidence_cannot_smuggle_species_location_or_unbounded_values(self):
        evidence = {'method': 'lab-local-contrast-v1', 'maxDeltaE': 10, 'meanDeltaE': 5,
                    'contrastScore': .5, 'enhancedViewUsed': True, 'species': 'not accepted'}
        result = archive_cue_evidence({'colorEvidence': evidence}, 'mycoscout.unlocated-candidates.v1')
        self.assertNotIn('species', result['colorEvidence'])
        self.assertEqual(result['investigation']['routingCategory'], 'unlocated_followup')
        for mutate in (lambda e: e.update(meanDeltaE=11), lambda e: e.update(contrastScore=1.1),
                       lambda e: e.update(maxDeltaE=float('inf')), lambda e: e.update(enhancedViewUsed=1)):
            broken = dict(evidence); mutate(broken)
            with self.assertRaises(ValueError):
                archive_cue_evidence({'colorEvidence': broken}, 'mycoscout.unlocated-candidates.v1')

    def test_supplied_metric_claims_cannot_leak_into_archive_queue(self):
        sources = {'a'*64: {'width': 100, 'height': 100}}
        p = {'schema': 'mycoscout.unlocated-candidates.v1', 'image': {'sha256': 'a'*64, 'width': 100, 'height': 100},
             'candidates': [{'confidence': .4, 'bboxPixelsXYXY': [10, 10, 30, 40], 'metric': {'depthKnown': True},
                             'capturedMs': 50, 'reviewRanking': {'blackTrumpetPatternScore': .99, 'prioritizeHumanReview': True}}]}
        result = candidate_leads(p, sources)[0]
        self.assertIsNone(result['capturedMs'])
        self.assertNotIn('metric', result)
        self.assertEqual(result['confidence'], .4)
        self.assertFalse(result['reviewMetadata']['flightGateEnabled'])

    def test_unknown_photo_source_rejected(self):
        with self.assertRaises(ValueError):
            candidate_leads({'schema': 'mycoscout.unlocated-candidates.v1', 'image': {}, 'candidates': []}, {})

    def test_generic_semantics_cannot_claim_moss_or_moisture(self):
        row = habitat_leads({'schema': 'mycoscout.habitat-cues.v2', 'observations': [
            {'confidence': .6, 'signals': {'moss': 1, 'damp_ground': 1, 'forest_context': .3}, 'metric': {}}]})[0]
        self.assertEqual(row['signals'], {'forest_context': .3})
        self.assertNotIn('metric', row)


if __name__ == '__main__':
    unittest.main()
