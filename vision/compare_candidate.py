"""Immutable same-footage candidate comparison; never trains or activates a model.

Daily optimization uses validation only. --split test is an explicit separate
release check. The calling station must recheck current label/rights consent
before and after this process; a frozen export cannot establish current consent.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from PIL import Image, ImageOps

from adapters.export_reviewed_detection_dataset import canonical, digest, inside, write_json
from vision.training_pipeline import MODEL_NAME, iou, load_detector, predictions_for, read_dataset, score_frames, tensor_image

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mycoscout.candidate-baseline-comparison.v1"
POLICY = {"minimumTargets": 20, "minimumTargetGroups": 3, "minimumSmallTargets": 10,
          "minimumSmallTargetGroups": 2, "minimumNegativeFrames": 5, "smallWidthBelowPixels": 64,
          "minimumRecallGain": .05, "minimumSmallRecallGain": .05,
          "maximumFalsePositiveIncrease": 0, "maximumNegativeFalsePositiveIncrease": 0}


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def installed_model_files(directory):
    directory = Path(directory)
    names = ["model.safetensors", "config.json", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "added_tokens.json"]
    return {name: file_sha(inside(directory, name)) for name in names}


def read_json(path):
    raw = Path(path).read_bytes()
    if len(raw) > 32*1024*1024:
        raise ValueError("JSON artifact exceeds the bounded comparison limit")
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Non-finite JSON"))), digest(raw)


def number(value, low=0, high=float("inf")):
    return type(value) in (int, float) and math.isfinite(value) and low <= value <= high


def checked_box(box, width, height):
    if (not isinstance(box, list) or len(box) != 4 or not all(number(x) for x in box)
            or not 0 <= box[0] < box[2] <= width or not 0 <= box[1] < box[3] <= height):
        raise ValueError("Box is outside original image coordinates")
    return list(box)


def load_dataset(directory):
    directory = Path(directory).resolve()
    manifest = read_dataset(directory, require_ready=False)
    raw, manifest_sha = read_json(directory / "manifest.json")
    if raw != manifest or manifest.get("status") != "READY" or not isinstance(manifest.get("datasetId"), str):
        raise ValueError("Dataset identity is invalid")
    expected_id = "dataset-" + digest(canonical({k: v for k, v in manifest.items() if k != "datasetId"}))[:24]
    if manifest["datasetId"] != expected_id:
        raise ValueError("Dataset content identity changed")
    rows = manifest["records"]
    if not isinstance(rows, list) or len(rows) > 1000 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate or unbounded dataset records")
    keys_to_split, keys_to_group = {}, {}
    for row in rows:
        if (row.get("complete") is not True or row.get("humanVerified") is not True or
                row.get("status") not in {"visible", "absent"} or bool(row["boxes"]) != (row["status"] == "visible") or
                type(row.get("width")) is not int or type(row.get("height")) is not int or min(row["width"], row["height"]) < 1):
            raise ValueError("Only complete human-reviewed frames are comparison truth")
        if not row.get("evaluationOnly") and row.get("trainingAllowed") is not True:
            raise ValueError("Frozen permission missing; caller must also recheck current consent")
        if type(row.get("evaluationOnly")) is not bool or any(not isinstance(row.get(key), str) or len(row[key]) != 64 or
                any(c not in "0123456789abcdef" for c in row[key]) for key in ("imageSha256", "pixelSha256", "originalSha256")):
            raise ValueError("Invalid frozen rights/hash metadata")
        if type(row.get("evaluationQuarantine")) is not bool or (row["evaluationOnly"] and
                (row["split"] != "test" or row["evaluationQuarantine"] is not True)):
            raise ValueError("Evaluation-only sources must remain quarantined in the test partition")
        if not row.get("groupKeys") or not row.get("leakageGroup"):
            raise ValueError("Missing source/session grouping")
        for key in row["groupKeys"]:
            if not isinstance(key, str) or (key in keys_to_split and keys_to_split[key] != row["split"]):
                raise ValueError("Source/session identity crosses partitions")
            keys_to_split[key] = row["split"]
            if key in keys_to_group and keys_to_group[key] != row["leakageGroup"]:
                raise ValueError("One source/session alias maps to different leakage groups")
            keys_to_group[key] = row["leakageGroup"]
        if len(row["boxes"]) > 200:
            raise ValueError("Too many human boxes")
        for box in row["boxes"]:
            checked_box(box["bboxPixelsXYXY"], row["width"], row["height"])
        with Image.open(inside(directory, row["imagePath"])) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            if list(rgb.size) != [row["width"], row["height"]] or digest(canonical(list(rgb.size)) + rgb.tobytes()) != row["pixelSha256"]:
                raise ValueError("Decoded image identity changed")
    return manifest, manifest_sha


def validate_job_metadata(card, complete, metadata, checkpoint_sha, manifest, manifest_sha, card_sha):
    """Validate recipe identity, never trust saved card accuracy or eligibility."""
    if (card.get("schema") != "mycoscout.detector-model-card.v1" or card.get("architecture") != MODEL_NAME or
            card.get("checkpoint") != "candidate.pt" or card.get("checkpointSha256") != checkpoint_sha or
            complete.get("checkpointSha256") != checkpoint_sha or complete.get("datasetId") != manifest["datasetId"] or
            card.get("datasetId") != manifest["datasetId"] or card.get("datasetManifestSha256") != manifest_sha or
            (complete.get("modelCardSha256") is not None and complete["modelCardSha256"] != card_sha)):
        raise ValueError("Model card/checkpoint/dataset integrity mismatch")
    if (metadata.get("schema") != "mycoscout.detector-checkpoint.v1" or metadata.get("architecture") != MODEL_NAME or
            metadata.get("datasetId") != manifest["datasetId"] or metadata.get("epoch") != card.get("bestEpoch") or
            metadata.get("scoreThreshold") != card.get("scoreThreshold") or not number(card.get("scoreThreshold"), .05, .95) or
            metadata.get("classes") != {0: "background", 1: "mushroom"}):
        raise ValueError("Checkpoint recipe disagrees with model card")
    if metadata.get("datasetManifestSha256") is not None and metadata["datasetManifestSha256"] != manifest_sha:
        raise ValueError("Checkpoint dataset hash mismatch")
    if card.get("validationOnly") is True:
        if (metadata.get("validationOnly") is not True or card.get("testEvaluationPerformed") is not False or
                card.get("test") is not None or card.get("testMetrics") is not None or card.get("activationEligible") is not False or
                complete.get("modelCardSha256") != card_sha or metadata.get("datasetManifestSha256") != manifest_sha):
            raise ValueError("Validation-only model artifact claims test evaluation or lacks integrity binding")
    return card["scoreThreshold"]


def load_job(directory, manifest, manifest_sha):
    import torch
    directory = Path(directory).resolve()
    card, card_sha = read_json(inside(directory, "model-card.json"))
    complete, complete_sha = read_json(inside(directory, "COMPLETE.json"))
    checkpoint = inside(directory, "candidate.pt")
    checkpoint_sha = file_sha(checkpoint)
    data = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = {k: v for k, v in data.items() if k != "state_dict"}
    threshold = validate_job_metadata(card, complete, metadata, checkpoint_sha, manifest, manifest_sha, card_sha)
    return {"kind": "reviewed_detector", "architecture": MODEL_NAME, "checkpointSha256": checkpoint_sha,
            "modelCardSha256": card_sha, "completeSha256": complete_sha, "datasetId": manifest["datasetId"],
            "datasetManifestSha256": manifest_sha, "recipeScoreThreshold": threshold,
            "modelCardHashBound": complete.get("modelCardSha256") == card_sha,
            "testEvaluationPerformedDuringTraining": card.get("testEvaluationPerformed", card.get("test") is not None)}, checkpoint


def assert_no_training_overlap(evaluation, training_manifest):
    training = [row for row in training_manifest["records"] if row["split"] == "train"]
    train_groups = {row["leakageGroup"] for row in training}
    train_keys = {key for row in training for key in row["groupKeys"]}
    train_pixels = {row["pixelSha256"] for row in training}
    train_originals = {row["originalSha256"] for row in training}
    for row in evaluation:
        if (row["leakageGroup"] in train_groups or train_keys.intersection(row["groupKeys"]) or
                row["pixelSha256"] in train_pixels or row["originalSha256"] in train_originals):
            raise ValueError("Comparison footage overlaps a model's reviewed training sources")


def truth_record(row):
    return {"id": row["id"], "split": row["split"], "sourceId": row["sourceId"], "group": row["leakageGroup"],
            "groupKeys": sorted(row["groupKeys"]), "imageSha256": row["imageSha256"], "pixelSha256": row["pixelSha256"],
            "originalSha256": row["originalSha256"], "width": row["width"], "height": row["height"],
            "truth": [list(b["bboxPixelsXYXY"]) for b in row["boxes"]], "humanVerified": True}


def checked_predictions(predictions, width, height, threshold, apply_nms=False):
    if not isinstance(predictions, list) or len(predictions) > 1000:
        raise ValueError("Invalid or unbounded predictions")
    valid = []
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("Prediction must be an object")
        box = checked_box(item.get("bbox"), width, height)
        if not number(item.get("score"), 0, 1):
            raise ValueError("Invalid detection score")
        if item["score"] >= threshold:
            valid.append({"bbox": box, "score": item["score"]})
    if not apply_nms:
        return valid
    kept = []
    for item in sorted(valid, key=lambda row: row["score"], reverse=True):
        if all(iou(item["bbox"], prior["bbox"]) <= .5 for prior in kept):
            kept.append(item)
    return kept[:100]


def support_for(records):
    small = [(r, b) for r in records for b in r["truth"] if b[2]-b[0] < 64]
    return {"frames": len(records), "targets": sum(len(r["truth"]) for r in records),
            "sourceGroups": len({r["group"] for r in records}), "targetGroups": len({r["group"] for r in records if r["truth"]}),
            "smallTargets": len(small), "smallTargetGroups": len({r["group"] for r, _ in small}),
            "negativeFrames": sum(not r["truth"] for r in records), "smallWidthBelowPixels": 64}


def summarize(records, model, threshold):
    scored, seconds = [], []
    for row in records:
        result = row[model]
        if not number(result.get("seconds")):
            raise ValueError("Invalid inference timing")
        predictions = checked_predictions(result.get("predictions"), row["width"], row["height"], threshold)
        if predictions != result["predictions"] or predictions != checked_predictions(predictions, row["width"], row["height"], threshold, apply_nms=True):
            raise ValueError("Saved evidence violates the fixed threshold, NMS or prediction cap")
        scored.append({"id": row["id"], "group": row["group"], "truth": row["truth"], "predictions": predictions})
        seconds.append(result["seconds"])
    metrics = score_frames(scored, threshold)
    negatives = [row for row in scored if not row["truth"]]
    negative_metrics = score_frames(negatives, threshold)
    bins = metrics["pixelWidthBins"]
    small_total = sum(value["total"] for key, value in bins.items() if key != "64plus")
    small_matched = sum(value["matched"] for key, value in bins.items() if key != "64plus")
    metrics.update(smallTargets={"matched": small_matched, "total": small_total, "recall": small_matched/small_total if small_total else None},
                   negativeFrames=len(negatives), negativeFalsePositives=negative_metrics["falsePositives"],
                   negativeFramesWithDetections=sum(bool(row["predictions"]) for row in negatives),
                   timing={"frames": len(seconds), "totalSeconds": sum(seconds), "meanSeconds": statistics.mean(seconds) if seconds else None,
                           "medianSeconds": statistics.median(seconds) if seconds else None,
                           "p95Seconds": sorted(seconds)[max(0, math.ceil(len(seconds)*.95)-1)] if seconds else None,
                           "includes": "Image decoding, preprocessing, forward pass and fixed postprocessing; excludes model load"})
    return metrics


def decision_for(baseline, candidate, support):
    missing, performance = [], []
    for name, policy_name in [("targets", "minimumTargets"), ("targetGroups", "minimumTargetGroups"),
                              ("smallTargets", "minimumSmallTargets"), ("smallTargetGroups", "minimumSmallTargetGroups"),
                              ("negativeFrames", "minimumNegativeFrames")]:
        if support[name] < POLICY[policy_name]:
            missing.append(f"{name}: {support[name]} observed; at least {POLICY[policy_name]} required")
    gain = candidate["recall"]-baseline["recall"] if baseline["recall"] is not None and candidate["recall"] is not None else None
    cs, bs = candidate["smallTargets"]["recall"], baseline["smallTargets"]["recall"]
    small_gain = cs-bs if cs is not None and bs is not None else None
    if gain is None or gain + 1e-12 < POLICY["minimumRecallGain"]:
        performance.append("Overall recall gain is below five percentage points or undefined")
    if small_gain is None or small_gain + 1e-12 < POLICY["minimumSmallRecallGain"]:
        performance.append("Small-target recall gain is below five percentage points or undefined")
    if candidate["falsePositives"] > baseline["falsePositives"]:
        performance.append("Candidate increases total false detections")
    if candidate["negativeFalsePositives"] > baseline["negativeFalsePositives"]:
        performance.append("Candidate increases false detections on explicit negative frames")
    dominated = (candidate["truePositives"] <= baseline["truePositives"] and candidate["falsePositives"] >= baseline["falsePositives"]
                 and (candidate["truePositives"] < baseline["truePositives"] or candidate["falsePositives"] > baseline["falsePositives"]))
    decision = "candidate_not_better" if dominated or (not missing and performance) else "insufficient_evidence" if missing else "candidate_better"
    return {"decision": decision, "reasons": missing+performance, "supportReasons": missing, "performanceReasons": performance,
            "recallGain": gain, "smallRecallGain": small_gain, "clearlyDominated": dominated,
            "evidenceSufficient": not missing, "activationAllowed": False, "automaticActivation": False}


def recompute(evidence, manifest, manifest_sha):
    split, threshold = evidence.get("split"), evidence.get("scoreThreshold")
    if (evidence.get("schema") != "mycoscout.candidate-comparison-evidence.v1" or split not in {"validation", "test"} or
            evidence.get("datasetManifestSha256") != manifest_sha or not number(threshold, .05, .95)):
        raise ValueError("Comparison evidence identity mismatch")
    expected = [truth_record(row) for row in sorted(manifest["records"], key=lambda r: r["id"]) if row["split"] == split]
    records = evidence.get("records")
    if not isinstance(records, list) or len(records) != len(expected):
        raise ValueError("Comparison evidence cohort mismatch")
    for actual, reference in zip(records, expected):
        if {key: actual.get(key) for key in reference} != reference:
            raise ValueError("Saved truth, source identity or frame order changed")
    support = support_for(records)
    baseline, candidate = (summarize(records, model, threshold) for model in ("baseline", "candidate"))
    return {"support": support, "baseline": baseline, "candidate": candidate, **decision_for(baseline, candidate, support)}


def compare(dataset, candidate_job, output, baseline_job=None, baseline_dataset=None, split="validation"):
    if split not in {"validation", "test"} or bool(baseline_job) != bool(baseline_dataset):
        raise ValueError("Choose validation or test; reviewed baseline requires its frozen dataset")
    output, dataset = Path(output), Path(dataset)
    if output.exists():
        raise ValueError("Comparison output must be a new directory")
    manifest, manifest_sha = load_dataset(dataset)
    candidate_identity, candidate_path = load_job(candidate_job, manifest, manifest_sha)
    rows = [row for row in sorted(manifest["records"], key=lambda row: row["id"]) if row["split"] == split]
    assert_no_training_overlap(rows, manifest)
    threshold = candidate_identity["recipeScoreThreshold"]
    import torch
    torch.set_num_threads(4)
    candidate_model, _ = load_detector(candidate_path, device="cpu")
    def reviewed(model, path):
        return predictions_for(model, tensor_image(path), threshold)
    def settings(model):
        return {"architecture": MODEL_NAME, "cropPolicy": "full_frame", "tiles": False,
                "minSize": list(model.transform.min_size), "maxSize": model.transform.max_size,
                "modelNmsIou": model.roi_heads.nms_thresh, "device": "cpu", "threads": 4}
    candidate_identity["settings"] = settings(candidate_model)
    if baseline_job:
        base_manifest, base_sha = load_dataset(baseline_dataset)
        baseline_identity, baseline_path = load_job(baseline_job, base_manifest, base_sha)
        assert_no_training_overlap(rows, base_manifest)
        baseline_model, _ = load_detector(baseline_path, device="cpu")
        baseline_identity["settings"] = settings(baseline_model)
        baseline_predict = lambda path: reviewed(baseline_model, path)
    else:
        from adapters.scout_frame import LocalDetector
        dino_directory = ROOT / "models/grounding-dino-tiny"
        baseline_files = installed_model_files(dino_directory)
        detector = LocalDetector(dino_directory, device="cpu", threads=4, box_threshold=threshold, text_threshold=.25)
        baseline_identity = {"kind": "installed_grounding_dino", "settings": {**detector.info, "cropPolicy": "full_frame", "tiles": False,
                             "processorSize": detector.processor.image_processor.size},
                             "filesSha256": baseline_files,
                             "reviewedTrainingLineage": "Not applicable; pretrained source overlap is unknown"}
        def baseline_predict(path):
            with Image.open(path) as source:
                predictions = detector.predict(ImageOps.exif_transpose(source).convert("RGB"))
            return [{"bbox": list(row["bboxPixelsXYXY"]), "score": row["confidence"]} for row in predictions]
    output.mkdir(parents=True, exist_ok=False)
    evidence = {"schema": "mycoscout.candidate-comparison-evidence.v1", "datasetId": manifest["datasetId"],
                "datasetManifestSha256": manifest_sha, "split": split, "scoreThreshold": threshold,
                "models": {"candidate": candidate_identity, "baseline": baseline_identity}, "records": []}
    write_json(output / "protocol.json", {"schema": SCHEMA, "split": split, "scoreThreshold": threshold, "iouThreshold": .5,
               "commonNmsIou": .5, "maximumPredictionsPerFrame": 100, "policy": POLICY, "datasetManifestSha256": manifest_sha,
               "models": evidence["models"], "consentRecheckResponsibility": "calling station before and after the process",
               "inferenceOrder": "alternate candidate/baseline order by frame; no warmup, timings descriptive", "activationAllowed": False})
    for index, row in enumerate(rows):
        path = inside(dataset, row["imagePath"])
        record = truth_record(row)
        callbacks = {"candidate": lambda: reviewed(candidate_model, path), "baseline": lambda: baseline_predict(path)}
        for name in (["candidate", "baseline"] if index % 2 == 0 else ["baseline", "candidate"]):
            started = time.perf_counter()
            predictions = checked_predictions(callbacks[name](), row["width"], row["height"], threshold, apply_nms=True)
            record[name] = {"predictions": predictions, "seconds": time.perf_counter()-started}
        evidence["records"].append(record)
        print(json.dumps({"status": "progress", "phase": "compare", "split": split, "completed": index+1, "total": len(rows)}), flush=True)
    # Recheck input bytes after inference; do not complete a comparison whose inputs changed.
    _, final_manifest_sha = load_dataset(dataset)
    final_candidate_identity, _ = load_job(candidate_job, manifest, manifest_sha)
    if final_manifest_sha != manifest_sha or final_candidate_identity != {k: v for k, v in candidate_identity.items() if k != "settings"}:
        raise ValueError("Candidate or dataset changed during comparison")
    if baseline_job:
        _, final_base_sha = load_dataset(baseline_dataset)
        final_baseline_identity, _ = load_job(baseline_job, base_manifest, base_sha)
        if final_base_sha != base_sha or final_baseline_identity != {k: v for k, v in baseline_identity.items() if k != "settings"}:
            raise ValueError("Baseline changed during comparison")
    elif installed_model_files(dino_directory) != baseline_files:
        raise ValueError("Installed baseline changed during comparison")
    write_json(output / "evidence.json", evidence)
    report = {"schema": SCHEMA, "status": "completed", "generatedAt": datetime.now(timezone.utc).isoformat(),
              "split": split, "datasetId": manifest["datasetId"], "datasetManifestSha256": manifest_sha,
              "candidateCheckpointSha256": candidate_identity["checkpointSha256"], "models": evidence["models"],
              "evidenceSha256": digest((output / "evidence.json").read_bytes()), "scoreThreshold": threshold,
              "iouThreshold": .5, "policy": POLICY, **recompute(evidence, manifest, manifest_sha),
              "testEvaluationPerformed": split == "test", "operationallyValidated": False,
              "limitations": ["Validation is development feedback, not independent release proof; repeated comparisons reuse this footage.",
                  "Small targets are native human-box widths below 64 pixels, not measured camera distances.",
                  "Generic mushroom localization is not species identification. No model is activated by this process.",
                  "Timing is descriptive CPU wall time with alternating model order, no warmup; not an onboard realtime benchmark.",
                  "Current consent and the current baseline pointer must be rechecked by the caller before using this result."]}
    write_json(output / "comparison.json", report)
    write_json(output / "COMPLETE.json", {"schema": SCHEMA, "comparisonSha256": digest((output / "comparison.json").read_bytes()),
               "evidenceSha256": report["evidenceSha256"], "protocolSha256": digest((output / "protocol.json").read_bytes())})
    return report


def validate_comparison(output, dataset, candidate_job=None, baseline_job=None, baseline_dataset=None):
    """Recompute saved metrics/gate from bound predictions and current frozen truth."""
    output = Path(output)
    manifest, manifest_sha = load_dataset(dataset)
    report, report_sha = read_json(inside(output, "comparison.json"))
    evidence, evidence_sha = read_json(inside(output, "evidence.json"))
    complete, _ = read_json(inside(output, "COMPLETE.json"))
    protocol, protocol_sha = read_json(inside(output, "protocol.json"))
    if (report.get("schema") != SCHEMA or report.get("status") != "completed" or complete.get("schema") != SCHEMA or
            complete.get("comparisonSha256") != report_sha or complete.get("evidenceSha256") != evidence_sha or
            complete.get("protocolSha256") != protocol_sha or report.get("evidenceSha256") != evidence_sha or
            report.get("models") != evidence.get("models") or report.get("models") != protocol.get("models") or
            report.get("policy") != POLICY or protocol.get("policy") != POLICY or
            report.get("datasetManifestSha256") != manifest_sha or protocol.get("datasetManifestSha256") != manifest_sha or
            report.get("split") != evidence.get("split") or report.get("split") != protocol.get("split") or
            report.get("scoreThreshold") != evidence.get("scoreThreshold") or report.get("scoreThreshold") != protocol.get("scoreThreshold") or
            report.get("scoreThreshold") != evidence.get("models", {}).get("candidate", {}).get("recipeScoreThreshold") or
            report.get("iouThreshold") != .5 or protocol.get("iouThreshold") != .5 or protocol.get("commonNmsIou") != .5 or
            protocol.get("maximumPredictionsPerFrame") != 100 or report.get("datasetId") != manifest["datasetId"] or
            evidence.get("datasetId") != manifest["datasetId"] or report.get("testEvaluationPerformed") != (report.get("split") == "test") or
            report.get("candidateCheckpointSha256") != evidence.get("models", {}).get("candidate", {}).get("checkpointSha256") or
            report.get("activationAllowed") is not False or report.get("automaticActivation") is not False):
        raise ValueError("Saved comparison artifact integrity mismatch")
    recalculated = recompute(evidence, manifest, manifest_sha)
    if any(report.get(key) != value for key, value in recalculated.items()):
        raise ValueError("Saved metrics or decision disagree with prediction evidence")
    if candidate_job is not None:
        current, _ = load_job(candidate_job, manifest, manifest_sha)
        if current != {k: v for k, v in report["models"]["candidate"].items() if k != "settings"}:
            raise ValueError("Candidate job changed after comparison")
    if report["models"]["baseline"]["kind"] == "reviewed_detector":
        if not baseline_job or not baseline_dataset:
            raise ValueError("Verification requires the reviewed baseline job and frozen dataset")
        base_manifest, base_sha = load_dataset(baseline_dataset)
        current, _ = load_job(baseline_job, base_manifest, base_sha)
        if current != {k: v for k, v in report["models"]["baseline"].items() if k != "settings"}:
            raise ValueError("Baseline job changed after comparison")
        assert_no_training_overlap([row for row in manifest["records"] if row["split"] == report["split"]], base_manifest)
    elif report["models"]["baseline"]["kind"] == "installed_grounding_dino":
        if baseline_job or baseline_dataset or report["models"]["baseline"].get("filesSha256") != installed_model_files(ROOT / "models/grounding-dino-tiny"):
            raise ValueError("Installed baseline files changed after comparison")
    else:
        raise ValueError("Unsupported comparison baseline")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--candidate-job", type=Path)
    parser.add_argument("--baseline-job", type=Path)
    parser.add_argument("--baseline-dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.validate_only:
            if not args.candidate_job:
                parser.error("--candidate-job is required for current-artifact verification")
            result = validate_comparison(args.output, args.dataset, args.candidate_job, args.baseline_job, args.baseline_dataset)
        else:
            if not args.candidate_job:
                parser.error("--candidate-job is required for inference")
            result = compare(args.dataset, args.candidate_job, args.output, args.baseline_job, args.baseline_dataset, args.split)
        print(json.dumps({"status": "completed", "decision": result["decision"], "split": result["split"],
                          "comparison": str(args.output / "comparison.json"), "activationAllowed": False}), flush=True)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
