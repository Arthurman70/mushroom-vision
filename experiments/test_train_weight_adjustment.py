"""Partition and supervision checks for the bounded paired-training driver."""
from collections import Counter
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.train_weight_adjustment import examples_for, sanity_row, train
from experiments.weight_recipe import paired_epoch


def record(identity, split="train", boxes=None, source="source-1", confounders=None):
    return {"id": identity, "split": split, "sourceGroupId": source,
            "width": 100, "height": 100,
            "boxes": [{"bboxPixelsXYXY": box} for box in (boxes or [])],
            "confounders": confounders or []}


def nonfungal(box, kind="hat", explicit=True):
    return {"bboxPixelsXYXY": box, "kind": kind, "explicitNonFungal": explicit}


class TrainingExampleTests(unittest.TestCase):
    def test_validation_and_test_rows_never_enter_training_or_crop_expansion(self):
        rows = [record("positive", boxes=[[10, 10, 30, 30]]), record("negative"),
                record("validation", split="validation", boxes=[[0, 0, 80, 80]],
                       confounders=[nonfungal([0, 0, 90, 90])]),
                record("test", split="test", confounders=[nonfungal([0, 0, 90, 90])])]
        before = copy.deepcopy(rows)
        examples, crops, rejected = examples_for(rows)
        self.assertEqual(["positive", "negative"], [item["id"] for item in examples])
        self.assertEqual({}, crops)
        self.assertEqual(0, rejected)
        self.assertEqual(before, rows)

    def test_rounding_negative_crop_cannot_cover_part_of_positive(self):
        row = record("tree-fungus", boxes=[[30.5, 0, 50, 40]],
                     confounders=[nonfungal([0, 0, 30.4, 30], "tree")])
        examples, crops, rejected = examples_for([row])
        self.assertEqual(1, len(examples))
        self.assertTrue(examples[0]["positive"])
        self.assertEqual({}, crops)
        self.assertEqual(1, rejected)

    def test_explicit_nonfungal_regions_are_limited_without_erasing_positive_boxes(self):
        row = record("mixed", boxes=[[80, 80, 100, 100]], confounders=[
            nonfungal([0, 0, 20, 20], "hat"),
            nonfungal([21, 0, 41, 20], "wood"),
            nonfungal([42, 0, 62, 20], "tree"),
            nonfungal([0, 30, 20, 50], "rock", explicit=False)])
        before = copy.deepcopy(row)
        examples, crops, rejected = examples_for([row])
        self.assertEqual(3, len(examples))
        self.assertEqual([True, False, False], [item["positive"] for item in examples])
        self.assertEqual({"hat": 1, "wood": 1}, crops)
        self.assertEqual(0, rejected)
        self.assertEqual([[0, 0, 20, 20], [21, 0, 41, 20]], [item["crop"] for item in examples[1:]])
        self.assertEqual(before, row)
        # Crop labels become empty only in tensor_example; the authoritative row remains intact.
        self.assertEqual(1, len(examples[0]["row"]["boxes"]))

    def test_mixed_original_and_crop_examples_keep_complete_paired_coverage(self):
        rows = [record(f"positive-{i}", boxes=[[60, 60, 80, 80]], source=f"video-{i%2}")
                for i in range(3)]
        rows += [record(f"negative-{i}", source=f"video-{i%2}") for i in range(4)]
        rows[0]["confounders"] = [nonfungal([0, 0, 20, 20])]
        examples, _, _ = examples_for(rows)
        batches = paired_epoch(examples, 8252)
        self.assertEqual(5, len(batches))
        positives = Counter(pair[0]["id"] for pair in batches)
        negatives = Counter(pair[1]["id"] for pair in batches)
        self.assertEqual(3, len(positives))
        self.assertEqual(5, len(negatives))
        self.assertEqual({1}, set(negatives.values()))
        self.assertEqual({item["id"] for item in examples}, set(positives) | set(negatives))


class SanitySelectionTests(unittest.TestCase):
    def test_sanity_choice_uses_training_single_target_and_fractional_size_only(self):
        rows = [record("small", boxes=[[0, 0, 20, 20]]),
                record("large", boxes=[[10, 10, 70, 70]]),
                record("validation-easy", "validation", boxes=[[0, 0, 100, 100]]),
                record("test-easy", "test", boxes=[[0, 0, 100, 100]]),
                record("multi", boxes=[[0, 0, 90, 90], [1, 1, 99, 99]]),
                record("absent")]
        # Absolute pixel size alone must not win: this target occupies only 10% of the image.
        big_pixels = record("large-image", boxes=[[0, 0, 100, 100]])
        big_pixels.update(width=1000, height=1000)
        rows.append(big_pixels)
        self.assertEqual("large", sanity_row(rows)["id"])

    def test_sanity_selection_cannot_fall_back_to_validation_or_test(self):
        rows = [record("absent"), record("multi", boxes=[[0, 0, 20, 20], [40, 40, 60, 60]]),
                record("validation", "validation", boxes=[[0, 0, 90, 90]]),
                record("test", "test", boxes=[[0, 0, 90, 90]])]
        with self.assertRaisesRegex(ValueError, "training original"):
            sanity_row(rows)

    def test_same_size_sanity_choice_is_deterministic_under_reordering(self):
        rows = [record("a", boxes=[[0, 0, 20, 20]]), record("b", boxes=[[0, 0, 20, 20]])]
        self.assertEqual(sanity_row(rows)["id"], sanity_row(list(reversed(rows)))["id"])


class TrainingBoundsTests(unittest.TestCase):
    def test_unknown_recipe_rejected_before_reading_sources_or_creating_output(self):
        with patch("experiments.train_weight_adjustment.load_records") as load, \
             patch.object(Path, "mkdir") as mkdir, self.assertRaisesRegex(ValueError, "recipe"):
            train("unused", "unused", "unused", "unused", recipe="balanced-800")
        load.assert_not_called()
        mkdir.assert_not_called()

    def test_invalid_epoch_and_learning_rate_rejected_before_data_access(self):
        for values in ({"epochs": 0}, {"epochs": 13}, {"epochs": True},
                       {"learning_rate": 0}, {"learning_rate": .006},
                       {"learning_rate": float("nan")}):
            with self.subTest(values=values), patch("experiments.train_weight_adjustment.load_records") as load, \
                 self.assertRaises(ValueError):
                train("unused", "unused", "unused", "unused", **values)
            load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
