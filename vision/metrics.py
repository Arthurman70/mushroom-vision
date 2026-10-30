"""Box-level evaluation without inventing labels or missing-inference negatives."""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path

from .dataset import read_json, valid_box


def iou(a, b):
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - overlap
    return overlap / union if union else 0.0


def size_bin(box):
    pixels = math.sqrt((box[2] - box[0]) * (box[3] - box[1]))
    return "under16px" if pixels < 16 else "16to31px" if pixels < 32 else "32to63px" if pixels < 64 else "64pxPlus"


def class_agnostic_nms(candidates, threshold):
    """Optional explicit suppression for duplicate mushroom/fungus prompt detections."""
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("NMS IoU threshold must be greater than zero and at most one")
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if all(iou(candidate["bboxPixelsXYXY"], previous["bboxPixelsXYXY"]) <= threshold for previous in kept):
            kept.append(candidate)
    return kept


def load_predictions(path, records):
    path = Path(path)
    if path.is_file():
        data = read_json(path)
        if data.get("schema") == "mycoscout.vision-predictions.v1":
            rows = data["images"]
            if not isinstance(rows, list):
                raise ValueError("Prediction images must be a list")
            result = {}
            known = {r["id"] for r in records}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("Each prediction image must be an object")
                image_id = row.get("imageId")
                if image_id not in known or image_id in result:
                    raise ValueError(f"Unknown or duplicate prediction imageId: {image_id}")
                result[image_id] = row
            return result, data.get("detector", {})
        files = [path]
    else:
        files = sorted(path.glob("*.json"))
    hashes = defaultdict(list)
    for record in records:
        hashes[record["sha256"]].append(record["id"])
    result, detector, config_fingerprint = {}, {}, None
    for file in files:
        data = read_json(file)
        if data.get("schema") != "mycoscout.unlocated-candidates.v1":
            continue
        matched_ids = hashes.get(data.get("image", {}).get("sha256"), [])
        if not matched_ids:
            continue
        config = data.get("detector", {})
        fingerprint = json.dumps(config, sort_keys=True, allow_nan=False)
        if config_fingerprint is not None and fingerprint != config_fingerprint:
            raise ValueError("Matched inference files use different detector configurations; separate experiments before evaluation")
        detector, config_fingerprint = config, fingerprint
        if not isinstance(data.get("candidates"), list):
            raise ValueError(f"Inference file {file.name} must contain an explicit candidates list")
        for image_id in matched_ids:
            if image_id in result:
                raise ValueError(f"Multiple prediction files for {image_id}; use separate run directories")
            result[image_id] = {"imageId": image_id, "status": "ok", "candidates": data["candidates"]}
    return result, detector


def counts():
    return {"images": 0, "negativeImages": 0, "groundTruth": 0, "predictions": 0,
            "truePositive": 0, "falsePositive": 0, "falseNegative": 0}


def finish_counts(row):
    row = dict(row)
    tp, fp, fn = row["truePositive"], row["falsePositive"], row["falseNegative"]
    row["precision"] = tp / (tp + fp) if tp + fp else None
    row["recall"] = tp / (tp + fn) if tp + fn else None
    row["f1"] = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    row["falsePositivesPerImage"] = fp / row["images"] if row["images"] else None
    return row


def evaluate(data, predictions, detector=None, confidence=0.35, iou_threshold=0.5, split=None, nms_iou=None):
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (confidence, iou_threshold)):
        raise ValueError("Thresholds must be between 0 and 1")
    if iou_threshold == 0:
        raise ValueError("IoU threshold must be greater than zero")
    if nms_iou is not None and (not math.isfinite(nms_iou) or not 0 < nms_iou <= 1):
        raise ValueError("NMS IoU threshold must be greater than zero and at most one")
    export_floor = (detector or {}).get("boxThreshold")
    if export_floor is not None:
        if not isinstance(export_floor, (int, float)) or not math.isfinite(export_floor) or not 0 <= export_floor <= 1:
            raise ValueError("Detector export boxThreshold must be a finite score from zero to one")
        if confidence < export_floor:
            raise ValueError("Evaluation confidence is below the detector export threshold; rerun inference at the lower threshold instead of estimating missing detections")
    total, real_total, synthetic_total = counts(), counts(), counts()
    buckets, contexts = defaultdict(counts), defaultdict(counts)
    image_results = []
    skipped = {"unreviewed": 0, "partial": 0, "missingInference": 0, "failedInference": 0}
    selected = [r for r in data["images"] if split is None or r.get("split") == split]
    for record in selected:
        row = predictions.get(record["id"])
        if row is None:
            skipped["missingInference"] += 1
            continue
        if row.get("status") != "ok":
            skipped["failedInference"] += 1
            continue
        if not isinstance(row.get("candidates"), list):
            raise ValueError(f"Successful inference for {record['id']} must contain an explicit candidates list")
        candidates = []
        for candidate in row["candidates"]:
            if not isinstance(candidate, dict):
                raise ValueError(f"Invalid prediction candidate for {record['id']}")
            score, box = candidate.get("confidence"), candidate.get("bboxPixelsXYXY")
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid prediction confidence for {record['id']}")
            if not valid_box(box, record["width"], record["height"]):
                raise ValueError(f"Invalid prediction bbox for {record['id']}")
            if score >= confidence:
                candidates.append(candidate)
        candidates.sort(key=lambda r: r["confidence"], reverse=True)
        raw_count = len(candidates)
        if nms_iou is not None:
            candidates = class_agnostic_nms(candidates, nms_iou)
        result = {"imageId": record["id"], "context": record["context"], "synthetic": record["synthetic"],
                  "split": record.get("split"), "candidateCount": len(candidates), "rawCandidateCount": raw_count, "metrics": None}
        if record["annotationStatus"] != "complete":
            skipped["partial" if record["annotationStatus"] == "partial" else "unreviewed"] += 1
            image_results.append(result)
            continue
        gt = [a["bboxPixelsXYXY"] for a in record["annotations"] if a["label"] == "mushroom"]
        matched, assignments = set(), []
        for candidate in candidates:
            options = [(iou(candidate["bboxPixelsXYXY"], box), i) for i, box in enumerate(gt) if i not in matched]
            overlap, index = max(options, default=(0, -1))
            is_match = index >= 0 and overlap >= iou_threshold
            if is_match:
                matched.add(index)
            assignments.append({"bboxPixelsXYXY": candidate["bboxPixelsXYXY"], "confidence": candidate["confidence"],
                                "match": is_match, "iou": overlap, "groundTruthIndex": index if is_match else None})
            if not is_match:
                buckets[size_bin(candidate["bboxPixelsXYXY"])]["falsePositive"] += 1
        local = counts()
        local.update(images=1, negativeImages=int(not gt), groundTruth=len(gt), predictions=len(candidates),
                     truePositive=len(matched), falsePositive=len(candidates) - len(matched), falseNegative=len(gt) - len(matched))
        context_key = ("synthetic:" if record["synthetic"] else "real:") + record["context"]
        for target in (total, synthetic_total if record["synthetic"] else real_total, contexts[context_key]):
            for key, value in local.items():
                target[key] += value
        for i, box in enumerate(gt):
            bucket = buckets[size_bin(box)]
            bucket["groundTruth"] += 1
            bucket["truePositive"] += int(i in matched)
            bucket["falseNegative"] += int(i not in matched)
        result.update(metrics=finish_counts(local), assignments=assignments,
                      missedGroundTruth=[box for i, box in enumerate(gt) if i not in matched])
        image_results.append(result)
    # Size recall is based on GT size; unmatched predictions are separately sized.
    # Precision by size is deliberately omitted: a predicted box can be a different size to its matched GT.
    pixel_summary = {}
    for key, row in buckets.items():
        pixel_summary[key] = {"groundTruth": row["groundTruth"], "matched": row["truePositive"],
                              "missed": row["falseNegative"],
                              "recall": row["truePositive"] / row["groundTruth"] if row["groundTruth"] else None,
                              "unmatchedPredictionsByPredictedSize": row["falsePositive"]}
    return {"schema": "mycoscout.vision-evaluation.v1", "status": "evaluated" if total["images"] else "labels-or-inference-required",
            "detector": detector or {}, "confidenceThreshold": confidence, "iouThreshold": iou_threshold,
            "detectorExportScoreFloor": export_floor,
            "postprocessing": {"classAgnosticNmsIou": nms_iou},
            "split": split, "selectedImages": len(selected), "evaluatedImages": total["images"],
            "skipped": skipped, "all": finish_counts(total), "real": finish_counts(real_total),
            "synthetic": finish_counts(synthetic_total), "byContext": {key: finish_counts(value) for key, value in contexts.items()},
            "byObjectPixelSize": pixel_summary, "images": image_results,
            "limitations": ["Scores measure mushroom candidate boxes against supplied annotations, not species or edibility.",
                            "Synthetic scale/degradation is a controlled stress test, not evidence of detection at physical distance.",
                            "Complete annotation is required: an unannotated mushroom would otherwise count as a false positive.",
                            "Missing/failed inference is excluded and counted explicitly, never treated as a negative prediction.",
                            "Thresholds must be selected on validation data; report the frozen threshold on untouched test groups."]}
