"""Freeze and run an actual-photo pilot without retuning the held-out groups."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import time

from PIL import Image, ImageOps

from .dataset import image_path, load_valid, read_json, sha256, split_groups, validate, write_json
from .metrics import evaluate, load_predictions
from .verifier import score_image, train


def overlap_conflicts(data):
    result = []
    for row in data["images"]:
        positives = [a["bboxPixelsXYXY"] for a in row["annotations"] if a["label"] == "mushroom"]
        negatives = [a["bboxPixelsXYXY"] for a in row["annotations"] if a["label"] == "background"]
        for positive in positives:
            for negative in negatives:
                overlap = max(0, min(positive[2], negative[2]) - max(positive[0], negative[0])) * max(0, min(positive[3], negative[3]) - max(positive[1], negative[1]))
                if overlap:
                    result.append({"imageId": row["id"], "positive": positive, "background": negative, "overlapPixels": overlap})
    return result


def run(manifest, predictions_dir, output_dir):
    manifest, predictions_dir, output_dir = Path(manifest), Path(predictions_dir), Path(output_dir)
    data = load_valid(manifest)
    conflicts = overlap_conflicts(data)
    if conflicts:
        raise ValueError(f"Explicit background boxes overlap positive boxes: {conflicts}")
    source_hash = sha256(manifest)
    frozen_dir = output_dir / "dataset"
    source_copy = frozen_dir / "manifest-source.json"
    split_copy = frozen_dir / "manifest-split.json"
    if source_copy.exists():
        if sha256(source_copy) != source_hash:
            raise ValueError("Original labels changed after experiment freeze; choose a new output directory")
        if not split_copy.exists():
            raise ValueError("Incomplete existing experiment; choose a new output directory")
        data = load_valid(split_copy)
    else:
        frozen_dir.mkdir(parents=True, exist_ok=True)
        for row in data["images"]:
            destination = frozen_dir / row["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path(manifest, row), destination)
        shutil.copy2(manifest, source_copy)
        data = split_groups(data)
        write_json(split_copy, data)
        load_valid(split_copy)
    predictions, detector = load_predictions(predictions_dir, data["images"])
    if len(predictions) != len(data["images"]):
        raise ValueError("This pilot requires successful inference records for every original image")
    frozen_prediction_hashes = {path.name: sha256(path) for path in sorted(predictions_dir.glob("*.json"))}
    checkpoint, training = train(data, split_copy, epochs=1000)
    write_json(output_dir / "verifier-checkpoint.json", checkpoint)
    write_json(output_dir / "verifier-training.json", training)
    baseline = evaluate(data, predictions, detector, confidence=0.25, iou_threshold=0.5)
    heldout = evaluate(data, predictions, detector, confidence=0.25, iou_threshold=0.5, split="test")
    write_json(output_dir / "detector-all-complete.json", baseline)
    write_json(output_dir / "detector-heldout-complete.json", heldout)
    scores, filtered = [], deepcopy(predictions)
    for row in data["images"]:
        prediction = predictions[row["id"]]
        with Image.open(image_path(split_copy, row)) as source:
            values = score_image(ImageOps.exif_transpose(source).convert("RGB"),
                                 [candidate["bboxPixelsXYXY"] for candidate in prediction["candidates"]], checkpoint)
        scores.append({"imageId": row["id"], "split": row["split"], "status": "ok", "candidateScores": values})
        filtered[row["id"]]["candidates"] = [candidate for candidate, value in zip(prediction["candidates"], values) if value["passesVerifier"]]
    write_json(output_dir / "verifier-candidate-scores.json", {"schema": "mycoscout.verifier-scores.v1", "images": scores,
                                                               "threshold": checkpoint["threshold"], "featureVersion": checkpoint["featureVersion"]})
    filtered_detector = deepcopy(detector)
    filtered_detector["candidateVerifier"] = {"featureVersion": checkpoint["featureVersion"], "threshold": checkpoint["threshold"],
                                              "trainingManifestSha256": checkpoint["trainingManifestSha256"]}
    paired = evaluate(data, filtered, filtered_detector, confidence=0.25, iou_threshold=0.5, split="test")
    write_json(output_dir / "detector-plus-verifier-heldout.json", paired)
    report = {"schema": "mycoscout.vision-pilot-study.v1", "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "status": "actual-photo-experiment-completed", "sourceManifestSha256": source_hash,
              "splitManifestFileSha256": sha256(split_copy), "splitPolicy": data["splitPolicy"],
              "validation": validate(split_copy), "positiveBackgroundOverlapConflicts": conflicts,
              "predictionDirectory": str(predictions_dir.resolve()), "predictionFileHashes": frozen_prediction_hashes,
              "detector": detector, "annotationStatusCounts": {status: sum(r["annotationStatus"] == status for r in data["images"]) for status in ("complete", "partial", "unreviewed")},
              "detectorAllComplete": baseline["real"], "detectorHeldoutComplete": heldout["real"],
              "verifierCropTraining": training, "detectorPlusVerifierHeldout": paired["real"],
              "pairedComparisonStatus": "exploratory-only: held-out complete-image sample is too small for a general performance claim",
              "limitations": ["Annotations are approximate first-pass visual labels, not expert-reviewed species labels.",
                              "Partial dense-colony images contribute only explicit crop labels, not whole-image detector metrics.",
                              "No real distant mushroom-positive calibration was collected; a distant negative does not establish detection range.",
                              "Model weights for GroundingDINO remain pretrained; only the small candidate-crop verifier was fitted.",
                              "Single deterministic source-group split; no test-set threshold tuning or split selection.",
                              "Zero overlap of labeled boxes does not prove the remaining image background contains no unannotated fungi.",
                              "Data and derived images remain local; source-group content is not licensed for redistribution."]}
    write_json(output_dir / "study.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--predictions", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    args = p.parse_args()
    report = run(args.manifest, args.predictions, args.output_dir)
    import json
    print(json.dumps({key: report[key] for key in ("status", "sourceManifestSha256", "detectorAllComplete", "detectorHeldoutComplete", "verifierCropTraining", "detectorPlusVerifierHeldout")}, indent=2))


if __name__ == "__main__":
    main()
