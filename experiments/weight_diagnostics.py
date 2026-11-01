"""Read-only fit diagnostics for provisional AI-reviewed detector candidates.

Run on training or validation originals only. Low-score curves explain missed
targets; they never select a deployment threshold or consume reserved test data.
RPN evidence describes retained, post-NMS proposals, not every anchor examined.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

from experiments.train_ai_corpus_candidate import tensor_example
from vision.training_pipeline import iou, predictions_for, score_frames


SCHEMA = "mushroom-vision.weight-diagnostics.v1"
LOW_SCORE_FLOOR = .01
CURVE_THRESHOLDS = (.05, .1, .2, .3, .5)
LABEL_AUTHORITY = "ai-visual-review"


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate_rows(rows):
    """Validate all partitions before opening even the first original image."""
    identities, groups = set(), {}
    for row in rows:
        split = row.get("split")
        if split not in {"train", "validation"}:
            raise ValueError("Weight diagnostics accept train/validation only; reserved test rows are forbidden")
        if (row.get("reviewerKind") != "ai" or row.get("humanVerified") is not False
            or row.get("complete") is not True or row.get("viewedOriginal") is not True):
            raise ValueError("Diagnostics require complete original AI reviews with explicit provisional authority")
        if row.get("status") not in {"visible", "absent"} or bool(row.get("boxes")) != (row["status"] == "visible"):
            raise ValueError("Original visible/absent labels and target boxes disagree")
        identity, source = row.get("id"), row.get("sourceGroupId")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Diagnostic original identities must be unique")
        if not isinstance(source, str) or not source:
            raise ValueError("A source group is required for separated evidence")
        identities.add(identity)
        for key in [source, *row.get("groupKeys", [])]:
            if key in groups and groups[key] != split:
                raise ValueError("Linked source groups cross train/validation partitions")
            groups[key] = split


def map_proposals_to_original(proposals, original_size_hw, transformed_size_hw):
    """Undo the actual unpadded image resize captured at the RPN boundary."""
    original_h, original_w = original_size_hw
    transformed_h, transformed_w = transformed_size_hw
    if any(not _finite_number(v) or v <= 0 for v in (*original_size_hw, *transformed_size_hw)):
        raise ValueError("Original and transformed image dimensions must be positive and finite")
    result = []
    for box in proposals:
        if (len(box) != 4 or any(not _finite_number(v) for v in box)
            or not 0 <= box[0] < box[2] <= transformed_w
            or not 0 <= box[1] < box[3] <= transformed_h):
            raise ValueError("RPN proposal is outside its unpadded transformed image")
        result.append([box[0]*original_w/transformed_w, box[1]*original_h/transformed_h,
                       box[2]*original_w/transformed_w, box[3]*original_h/transformed_h])
    return result


def target_evidence(truths, predictions, proposals_original, original_size_hw, transformed_size_hw):
    """Per-target overlap evidence; score_frames separately enforces one-to-one matches."""
    original_h, original_w = original_size_hw
    transformed_h, transformed_w = transformed_size_hw
    targets = []
    for index, truth in enumerate(truths):
        overlaps = [(prediction["score"], iou(truth, prediction["bbox"])) for prediction in predictions]
        transformed = [truth[0]*transformed_w/original_w, truth[1]*transformed_h/original_h,
                       truth[2]*transformed_w/original_w, truth[3]*transformed_h/original_h]
        targets.append({
            "targetIndex": index, "bboxPixelsXYXY": list(truth),
            "originalWidthPixels": truth[2]-truth[0], "originalHeightPixels": truth[3]-truth[1],
            "transformedBboxPixelsXYXY": transformed,
            "transformedWidthPixels": transformed[2]-transformed[0],
            "transformedHeightPixels": transformed[3]-transformed[1],
            "bestProposalIoU": max((iou(truth, proposal) for proposal in proposals_original), default=0),
            "bestPredictionIoU": max((overlap for _, overlap in overlaps), default=0),
            "topScoreAtIoU0_5": max((score for score, overlap in overlaps if overlap >= .5), default=None),
            "topScoreAtIoU0_3": max((score for score, overlap in overlaps if overlap >= .3), default=None),
            "labelAuthority": LABEL_AUTHORITY, "humanVerified": False,
        })
    return targets


def summarize_partition(frames, threshold):
    positives = [frame for frame in frames if frame["status"] == "visible"]
    negatives = [frame for frame in frames if frame["status"] == "absent"]
    targets = [target for frame in frames for target in frame["targets"]]
    def scores(cutoff):
        return {"overall": score_frames(frames, cutoff),
                "positiveFrames": score_frames(positives, cutoff),
                "explicitNegativeFrames": score_frames(negatives, cutoff)}
    return {
        **scores(threshold),
        "rpnProposalRecall": [{"iouThreshold": cutoff, "targets": len(targets),
            "matchedTargets": sum(target["bestProposalIoU"] >= cutoff for target in targets),
            "recall": sum(target["bestProposalIoU"] >= cutoff for target in targets)/len(targets) if targets else None,
            "proposalStage": "retained-post-NMS", "originalPixels": True} for cutoff in (.3, .5)],
        "thresholdCurve": [{"scoreThreshold": cutoff, **scores(cutoff)} for cutoff in CURVE_THRESHOLDS],
        "labelAuthority": LABEL_AUTHORITY, "humanVerified": False,
    }


def diagnose(model, rows, root, threshold=.3):
    """Return JSON-compatible diagnostics without changing weights or thresholds.

    The caller supplies provenance-validated rows and a model configured with an
    ROI score floor <= .01. All existing module training flags are restored, even
    after an inference failure; the temporary RPN hook is always removed.
    """
    rows = list(rows)
    _validate_rows(rows)
    if not _finite_number(threshold) or not LOW_SCORE_FLOOR <= threshold <= 1:
        raise ValueError("Diagnostic evaluation threshold must be finite and between .01 and 1")
    roi_floor = getattr(model.roi_heads, "score_thresh", None)
    if not _finite_number(roi_floor) or not 0 <= roi_floor <= LOW_SCORE_FLOOR:
        raise ValueError("Configure the model ROI score floor <= .01 before diagnostics; diagnostics will not change it")
    training_flags = [(module, module.training) for module in model.modules()]
    frames = []
    started = time.perf_counter()
    try:
        for row in rows:
            tensor, _ = tensor_example(Path(root), {"row": row}, augment=False)
            original_size_hw = tuple(int(v) for v in tensor.shape[-2:])
            if original_size_hw != (row["height"], row["width"]):
                raise ValueError("Diagnostic original dimensions differ from the frozen review")
            captured = []
            def capture_rpn(module, inputs, output):
                image_sizes = inputs[0].image_sizes
                proposals = output[0]
                if len(image_sizes) != 1 or len(proposals) != 1:
                    raise ValueError("Diagnostics require one original per inference")
                captured.append({"imageSizeHW": tuple(int(v) for v in image_sizes[0]),
                                 "proposals": proposals[0].detach().cpu().tolist()})
            handle = model.rpn.register_forward_hook(capture_rpn)
            frame_started = time.perf_counter()
            try:
                predictions = predictions_for(model, tensor, LOW_SCORE_FLOOR)
            finally:
                handle.remove()
            elapsed = time.perf_counter()-frame_started
            if len(captured) != 1:
                raise ValueError("RPN evidence was not captured exactly once for this original")
            transformed_size_hw = captured[0]["imageSizeHW"]
            proposals = map_proposals_to_original(captured[0]["proposals"], original_size_hw, transformed_size_hw)
            truths = [box["bboxPixelsXYXY"] for box in row["boxes"]]
            frames.append({
                "id": row["id"], "group": row["sourceGroupId"], "split": row["split"], "status": row["status"],
                "truth": truths,
                "predictions": [{**prediction, "label": "mushroom", "evidenceKind": "model-prediction",
                                 "humanVerified": False} for prediction in predictions],
                "targets": target_evidence(truths, predictions, proposals, original_size_hw, transformed_size_hw),
                "originalSizeHW": list(original_size_hw), "transformedSizeHW": list(transformed_size_hw),
                "rpnProposalCount": len(proposals), "retainedPredictionCount": len(predictions),
                "inferenceSeconds": elapsed, "labelAuthority": LABEL_AUTHORITY,
                "humanVerified": False, "originalPixels": True,
            })
    finally:
        # Assign individually: module.train() recurses and would erase mixed
        # backbone/head modes intentionally used while fitting this candidate.
        for module, training in training_flags:
            module.training = training
    return {
        "schema": SCHEMA, "fixedEvaluationThreshold": threshold, "retainedScoreFloor": LOW_SCORE_FLOOR,
        "configuredRoiScoreFloor": roi_floor,
        "perPartition": {split: summarize_partition([frame for frame in frames if frame["split"] == split], threshold)
                         for split in ("train", "validation")},
        "frames": frames, "elapsedSeconds": time.perf_counter()-started,
        "thresholdCurveUsedForSelection": False, "productionThresholdChanged": False,
        "modelWeightsChanged": False, "testUsed": False, "activationEligible": False,
        "labelAuthority": LABEL_AUTHORITY, "humanVerified": False,
        "limitations": [
            "AI-reviewed targets are provisional labels, not independent human truth.",
            "Training fit and validation evidence are reported separately; neither measures held-out test performance.",
            "RPN recall measures retained post-NMS proposals. Missing proposals can arise before or during RPN filtering.",
            "Low-score predictions are limited by the configured ROI postprocessing and detection-count cap.",
            "Per-target top scores may share a prediction; aggregate metrics use one-to-one matching at IoU >= .5.",
            "Threshold curves are diagnostic only and do not choose a production threshold.",
        ],
    }
