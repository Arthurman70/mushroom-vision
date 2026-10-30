import unittest

from experiments.evaluate_ai_corpus_distance import summarize, verify_no_overlap


class FrozenRegressionTests(unittest.TestCase):
    def test_false_alerts_and_misses_remain_separate(self):
        rows = [{"condition": "16px", "kind": "positive", "group": "test",
                 "truth": [[0, 0, 10, 10], [40, 40, 50, 50]],
                 "predictions": [{"bbox": [0, 0, 10, 10], "score": .8},
                                 {"bbox": [0, 0, 10, 10], "score": .7}]},
                {"condition": "16px", "kind": "negative", "group": "test", "truth": [],
                 "predictions": [{"bbox": [0, 0, 10, 10], "score": .8}]}]
        row = summarize(rows)[0]
        self.assertEqual((1, 1, 1, 1), (row['matched'], row['missed'], row['unmatchedDetections'], row['negativeFalsePositives']))

    def test_reencoded_duplicate_group_is_rejected(self):
        fit = [{"split": "train", "imageSha256": "different-bytes", "sourceGroupId": "training", "groupKeys": ["same-decoded-pixels"]}]
        source = [{"imageSha256": "test", "sourceGroupId": "reserved"}]
        inventory = {"records": [{"sha256": "test", "split": "test", "groupKeys": ["same-decoded-pixels"]}]}
        with self.assertRaisesRegex(ValueError, 'overlap'):
            verify_no_overlap(fit, source, inventory)

    def test_evaluation_original_must_be_reserved(self):
        source = [{"imageSha256": "test", "sourceGroupId": "reserved"}]
        inventory = {"records": [{"sha256": "test", "split": "unassigned", "groupKeys": []}]}
        with self.assertRaisesRegex(ValueError, 'frozen test'):
            verify_no_overlap([], source, inventory)

    def test_separate_sources_accepted(self):
        fit = [{"split": "train", "imageSha256": "train", "sourceGroupId": "training", "groupKeys": ["train"]}]
        source = [{"imageSha256": "test", "sourceGroupId": "reserved"}]
        inventory = {"records": [{"sha256": "test", "split": "test", "groupKeys": ["test"]}]}
        verify_no_overlap(fit, source, inventory)


if __name__ == '__main__':
    unittest.main()
