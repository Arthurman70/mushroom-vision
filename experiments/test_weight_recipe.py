"""Coverage, authority and reload checks for the weight-recipe ablation."""
from collections import Counter
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from experiments.weight_recipe import (
    CHECKPOINT_SCHEMA, MODEL_NAME, build_recipe_model, config_for,
    load_recipe_checkpoint, paired_epoch, validate_config,
)


def example(identity, positive, source):
    return {"id": identity, "positive": positive,
            "row": {"split": "train", "sourceGroupId": source,
                    "boxes": [{"bboxPixelsXYXY": [10, 10, 20, 20]}] if positive else []}}


class PairedBatchTests(unittest.TestCase):
    def setUp(self):
        self.examples = [example(f"p-{i:02d}", True, f"source-{i % 3}") for i in range(26)]
        self.examples += [example(f"n-{i:02d}", False, f"source-{i % 3}") for i in range(76)]

    def test_every_example_is_seen_and_every_batch_is_balanced(self):
        before = copy.deepcopy(self.examples)
        batches = paired_epoch(self.examples, 8252)
        self.assertEqual(76, len(batches))
        self.assertTrue(all(positive["positive"] and not negative["positive"] for positive, negative in batches))
        negatives = Counter(negative["id"] for _, negative in batches)
        positives = Counter(positive["id"] for positive, _ in batches)
        self.assertEqual(76, len(negatives))
        self.assertEqual({1}, set(negatives.values()))
        self.assertEqual(26, len(positives))
        self.assertEqual({2, 3}, set(positives.values()))
        self.assertEqual(before, self.examples)

    def test_seeded_order_is_repeatable_and_independent_of_input_order(self):
        first = paired_epoch(self.examples, 5)
        self.assertEqual(first, paired_epoch(list(reversed(self.examples)), 5))
        self.assertNotEqual(first, paired_epoch(self.examples, 6))

    def test_source_groups_rotate_and_positive_cycles_cover_every_item(self):
        batches = paired_epoch(self.examples, 7)
        for side in (0, 1):
            self.assertEqual(3, len({batch[side]["row"]["sourceGroupId"] for batch in batches[:3]}))
        self.assertEqual(26, len({batch[0]["id"] for batch in batches[:26]}))
        self.assertEqual(26, len({batch[0]["id"] for batch in batches[26:52]}))

    def test_duplicate_unknown_source_nontraining_and_contradictory_labels_rejected(self):
        mutations = [lambda xs: xs.append(copy.deepcopy(xs[0])),
                     lambda xs: xs[0]["row"].update(sourceGroupId=""),
                     lambda xs: xs[0]["row"].update(split="test"),
                     lambda xs: xs[0].update(positive=False),
                     lambda xs: xs[0].update(positive=1),
                     lambda xs: xs[0].update(crop=[0, 0, 30, 30])]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                rows = copy.deepcopy(self.examples); mutate(rows)
                with self.assertRaises(ValueError):
                    paired_epoch(rows, 5)

    def test_negative_crop_may_not_cover_even_part_of_positive_box(self):
        crop = example("crop", False, "source-1")
        crop["row"]["boxes"] = [{"bboxPixelsXYXY": [10, 10, 20, 20]}]
        crop["crop"] = [0, 0, 11, 11]
        with self.assertRaisesRegex(ValueError, "overlaps"):
            paired_epoch(self.examples + [crop], 5)
        crop["crop"] = [0, 0, 10, 10]
        self.assertEqual(77, len(paired_epoch(self.examples + [crop], 5)))

    def test_empty_single_class_majority_positive_and_boolean_seed_rejected(self):
        for rows in ([], self.examples[:26], self.examples[26:], self.examples[:27]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                paired_epoch(rows, 5)
        with self.assertRaises(ValueError):
            paired_epoch(self.examples, True)


def stub_model():
    parameter = SimpleNamespace(requires_grad=True)
    model = Mock()
    model.transform = SimpleNamespace(min_size=(320,), max_size=640)
    model.roi_heads = SimpleNamespace(score_thresh=.05)
    model.backbone.body.parameters.return_value = [parameter]
    return model


class RecipeRestorationTests(unittest.TestCase):
    def checkpoint(self, recipe="balanced-640"):
        return {"schema": CHECKPOINT_SCHEMA, "architecture": MODEL_NAME,
                "modelConfig": config_for(recipe), "reviewerKind": "ai",
                "humanVerified": False, "activationEligible": False,
                "scoreThreshold": .3, "state_dict": {"synthetic-test-key": 1},
                "trainingManifestSha256": "a"*64}

    def test_configs_are_independent_and_strict(self):
        one, two = config_for("balanced-320"), config_for("balanced-320")
        one["minSize"] = 640
        self.assertEqual(320, two["minSize"])
        for mutate in (lambda x: x.update(minSize=640), lambda x: x.update(extra=True),
                       lambda x: x.update(backboneFrozen=1),
                       lambda x: x.update(diagnosticScoreFloor=float("nan")),
                       lambda x: x.update(decisionScoreThreshold=.1),
                       lambda x: x.update(rpnPolicy="more-proposals")):
            config = config_for("balanced-320"); mutate(config)
            with self.assertRaises(ValueError):
                validate_config(config)

    def test_model_build_applies_resize_floor_and_freeze_without_other_changes(self):
        for recipe, size in (("balanced-320", 320), ("balanced-640", 640)):
            model = stub_model()
            with patch("experiments.weight_recipe.build_model", return_value=model) as build:
                self.assertIs(model, build_recipe_model(config_for(recipe), pretrained=False))
            build.assert_called_once_with(pretrained=False)
            self.assertEqual((size,), model.transform.min_size)
            self.assertEqual(size*2, model.transform.max_size)
            self.assertEqual(.01, model.roi_heads.score_thresh)
            self.assertFalse(next(iter(model.backbone.body.parameters())).requires_grad)

    def test_checkpoint_reloads_config_and_strict_weights_and_keeps_metadata(self):
        model = stub_model()
        data = self.checkpoint()
        with patch("torch.load", return_value=data) as load, \
             patch("experiments.weight_recipe.build_model", return_value=model) as build:
            restored, metadata = load_recipe_checkpoint(Path("synthetic-unused.pt"))
        load.assert_called_once_with(Path("synthetic-unused.pt"), map_location="cpu", weights_only=True)
        build.assert_called_once_with(pretrained=False)
        model.load_state_dict.assert_called_once_with(data["state_dict"], strict=True)
        model.eval.assert_called_once_with()
        self.assertIs(model, restored)
        self.assertEqual((640,), restored.transform.min_size)
        self.assertEqual(1280, restored.transform.max_size)
        self.assertEqual(.3, metadata["scoreThreshold"])
        self.assertEqual("a"*64, metadata["trainingManifestSha256"])
        self.assertNotIn("state_dict", metadata)

    def test_wrong_authority_or_inference_config_rejected_before_build(self):
        mutations = [lambda x: x.update(schema="mycoscout.detector-checkpoint.v1"),
                     lambda x: x.update(reviewerKind="human"), lambda x: x.update(humanVerified=True),
                     lambda x: x.update(activationEligible=True), lambda x: x.update(scoreThreshold=.1),
                     lambda x: x.update(productionModelChanged=True),
                     lambda x: x.update(labelAuthority="human"), lambda x: x.pop("modelConfig"),
                     lambda x: x["modelConfig"].update(maxSize=640),
                     lambda x: x.update(state_dict={})]
        for mutate in mutations:
            data = self.checkpoint(); mutate(data)
            with self.subTest(mutate=mutate), patch("torch.load", return_value=data), \
                 patch("experiments.weight_recipe.build_model") as build, self.assertRaises(ValueError):
                load_recipe_checkpoint("synthetic-unused.pt")
            build.assert_not_called()


if __name__ == "__main__":
    unittest.main()
