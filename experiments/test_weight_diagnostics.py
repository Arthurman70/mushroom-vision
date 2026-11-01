"""Small deterministic checks for diagnostic geometry and evidence separation."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.weight_diagnostics import (
    diagnose, map_proposals_to_original, summarize_partition, target_evidence,
)


def row(identity="positive", split="train", positive=True):
    return {"id": identity, "sourceGroupId": identity+"-source", "groupKeys": [identity+"-source"],
            "split": split, "reviewerKind": "ai", "humanVerified": False, "complete": True,
            "viewedOriginal": True, "status": "visible" if positive else "absent",
            "width": 1000, "height": 600, "imagePath": "data/unused.png",
            "boxes": [{"bboxPixelsXYXY": [100, 60, 200, 120]}] if positive else []}


class Values:
    def __init__(self, values): self.values = values
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.values


class FakeRpn:
    training = True
    def __init__(self): self.hooks = []
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


class FakeModel:
    training = True
    def __init__(self):
        self.rpn = FakeRpn()
        self.roi_heads = SimpleNamespace(training=False, score_thresh=.01)
    def modules(self): return [self, self.rpn, self.roi_heads]
    def eval(self):
        for module in self.modules(): module.training = False


def fake_predictions(model, tensor, threshold):
    model.eval()
    # The unpadded size is 192x320; padding could be 224x320. Only the former
    # maps this proposal back onto the exact original target.
    for hook in model.rpn.hooks:
        hook(model.rpn, [SimpleNamespace(image_sizes=[(192, 320)])],
             ([Values([[32, 19.2, 64, 38.4]])], {}))
    return [{"bbox": [100, 60, 200, 120], "score": .2}]


class WeightDiagnosticTests(unittest.TestCase):
    def test_unpadded_resize_maps_original_coordinates(self):
        result = map_proposals_to_original([[32, 19.2, 64, 38.4]], (600, 1000), (192, 320))
        self.assertEqual([[100, 60, 200, 120]], result)
        with self.assertRaisesRegex(ValueError, "outside"):
            map_proposals_to_original([[0, 0, 320, 224]], (600, 1000), (192, 320))

    def test_nonuniform_rounding_and_empty_proposals(self):
        self.assertEqual([[0, 0, 1001, 601]], map_proposals_to_original([[0, 0, 533, 320]], (601, 1001), (320, 533)))
        self.assertEqual([], map_proposals_to_original([], (600, 1000), (192, 320)))
        with self.assertRaisesRegex(ValueError, "positive"):
            map_proposals_to_original([], (0, 1000), (192, 320))

    def test_low_score_match_does_not_become_default_true_positive(self):
        truth = [[100, 60, 200, 120]]
        predictions = [{"bbox": truth[0], "score": .2}]
        evidence = target_evidence(truth, predictions, truth, (600, 1000), (192, 320))
        self.assertEqual(.2, evidence[0]["topScoreAtIoU0_5"])
        self.assertEqual(32, evidence[0]["transformedWidthPixels"])
        frame = {"id": "p", "group": "s", "status": "visible", "truth": truth,
                 "predictions": predictions, "targets": evidence}
        negative = {"id": "n", "group": "s2", "status": "absent", "truth": [],
                    "predictions": [{"bbox": truth[0], "score": .8}], "targets": []}
        report = summarize_partition([frame, negative], .3)
        self.assertEqual(0, report["positiveFrames"]["truePositives"])
        self.assertEqual(1, report["explicitNegativeFrames"]["falsePositives"])
        self.assertEqual(1, report["thresholdCurve"][1]["positiveFrames"]["truePositives"])
        self.assertEqual(1, report["rpnProposalRecall"][1]["matchedTargets"])

    @patch("experiments.weight_diagnostics.tensor_example", return_value=(SimpleNamespace(shape=(3, 600, 1000)), None))
    @patch("experiments.weight_diagnostics.predictions_for", side_effect=fake_predictions)
    def test_modes_hooks_authority_and_partitions_preserved(self, predict, tensor):
        model = FakeModel()
        report = diagnose(model, [row(), row("negative", "validation", False)], Path("."))
        self.assertEqual([True, True, False], [module.training for module in model.modules()])
        self.assertEqual([], model.rpn.hooks)
        self.assertEqual(1, report["perPartition"]["train"]["positiveFrames"]["frames"])
        self.assertEqual(1, report["perPartition"]["validation"]["explicitNegativeFrames"]["frames"])
        self.assertEqual(1, report["frames"][0]["targets"][0]["bestProposalIoU"])
        self.assertEqual(.01, predict.call_args.args[2])
        self.assertFalse(report["thresholdCurveUsedForSelection"])
        self.assertFalse(report["frames"][0]["humanVerified"])
        self.assertEqual("model-prediction", report["frames"][0]["predictions"][0]["evidenceKind"])

    @patch("experiments.weight_diagnostics.tensor_example", return_value=(SimpleNamespace(shape=(3, 600, 1000)), None))
    def test_hook_and_mixed_modes_restored_on_failure(self, tensor):
        model = FakeModel()
        def fail(model, *args):
            model.eval()
            raise RuntimeError("inference failed")
        with patch("experiments.weight_diagnostics.predictions_for", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "inference failed"):
                diagnose(model, [row()], Path("."))
        self.assertEqual([], model.rpn.hooks)
        self.assertEqual([True, True, False], [module.training for module in model.modules()])

    @patch("experiments.weight_diagnostics.tensor_example")
    def test_reserved_test_rejected_before_any_images_are_opened(self, tensor):
        with self.assertRaisesRegex(ValueError, "reserved test"):
            diagnose(FakeModel(), [row(), row("reserved", "test")], Path("."))
        tensor.assert_not_called()

    @patch("experiments.weight_diagnostics.tensor_example")
    def test_linked_sources_cannot_cross_partitions(self, tensor):
        first, second = row(), row("other", "validation")
        second["groupKeys"].append(first["sourceGroupId"])
        with self.assertRaisesRegex(ValueError, "Linked source"):
            diagnose(FakeModel(), [first, second], Path("."))
        tensor.assert_not_called()

    def test_roi_floor_is_not_silently_changed(self):
        model = FakeModel(); model.roi_heads.score_thresh = .05
        with self.assertRaisesRegex(ValueError, "will not change"):
            diagnose(model, [], Path("."))
        self.assertEqual(.05, model.roi_heads.score_thresh)

    def test_empty_or_missing_prediction_evidence_is_not_zero_confidence_truth(self):
        target = target_evidence([[1, 1, 2, 2]], [], [], (10, 10), (5, 5))[0]
        self.assertIsNone(target["topScoreAtIoU0_5"])
        self.assertEqual(0, target["bestProposalIoU"])


if __name__ == "__main__":
    unittest.main()
