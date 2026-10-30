"""Experimental frozen-feature verifier using explicitly audited weak crops.

Training and threshold choice use eligible assistant-audited source groups only.
The frozen human video snapshot is evaluated once after the checkpoint is saved.
Nothing changes the live detector, human labels or flight behavior.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from adapters.merge_video_collections import read_collection
from vision.assistant_video_pilot import blocked_by_holdout, checked_box, collect_holdouts, signatures
from vision.evaluate_video_ai import evaluate_snapshot, write_new
from vision.transfer import EXPECTED_WEIGHT_SHA, fit_head, head_scores, load_backbone, summarize
from vision.video_ai_pass import sha

ROOT = Path(__file__).resolve().parents[1]


def explicit_negative_crop(review, frame, proposal):
    if review.get("negative") is not True:
        return None
    if (review.get("reviewer") != "assistant-visual-audit" or review.get("humanVerified") is not False or
            review.get("collection") != "main" or review.get("frameId") != frame["id"] or
            review.get("imageSha256") != frame["sha256"] or proposal.get("status") != "completed" or
            proposal.get("collection") != "main" or proposal.get("frameId") != frame["id"] or
            proposal.get("imageSha256") != frame["sha256"]):
        raise ValueError("Negative crop audit identity/authority mismatch")
    box = next((box for box in proposal["boxes"] if box["id"] == review.get("proposalBoxId")), None)
    if box is None or review.get("bbox") != box["bbox"]:
        raise ValueError("Negative crop must bind to exact original proposal coordinates")
    return checked_box(box["bbox"], frame)


def select_training_threshold(labels, scores):
    """Use only leave-source-out predictions from the weak training collection."""
    trials = [(round(float(t), 2), summarize(labels, scores, float(t))) for t in np.linspace(.05, .95, 19)]
    threshold, metrics = max(trials, key=lambda item: (item[1]["f1"] or 0, item[1]["recall"] or 0, item[0]))
    return threshold, {"selectionData": "leave-one-training-source-group-out weak labels only", "fixedGrid": [.05, .95, .05],
                       "criterion": "Maximum weak-label F1, then recall, then higher threshold", "metrics": metrics,
                       "calibratedProbability": False, "futurePerformanceGuarantee": False}


def native_crop(image, bbox):
    return image.crop((math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3]))).convert("RGB")


def prepare_crops(dataset, negative_audit, collection, proposals, learning_directory, output):
    dataset, output = Path(dataset), Path(output)
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    pilot = json.loads(manifest_bytes)
    if (pilot.get("schema") != "mycoscout.assistant-video-training.v1" or
            json.loads((dataset / "READY.json").read_text())["manifestSha256"] != sha(manifest_bytes)):
        raise ValueError("Weak training dataset changed")
    root, raw, manifest = read_collection(collection)
    frames = {frame["id"]: frame for frame in manifest["frames"]}
    sources = {source["id"]: source for source in manifest["sources"]}
    holdouts, versions = collect_holdouts(learning_directory)
    negative_data = json.loads(Path(negative_audit).read_text(encoding="utf-8-sig"))
    if negative_data.get("schema") != "mycoscout.assistant-negative-crop-audit.v1":
        raise ValueError("Explicit separate negative crop audit required")
    specs, exclusions = [], []
    for row in pilot["records"]:
        if row.get("humanVerified") is not False or row.get("labelAuthority") != "assistant-reviewed-ai":
            raise ValueError("Weak-label authority changed")
        frame = frames[row["id"]]
        if frame["sha256"] != row["originalSha256"]:
            raise ValueError("Training frame changed")
        if row["boxes"]:
            specs.extend({"frameId": frame["id"], "bbox": box["bboxPixelsXYXY"], "target": 1,
                          "origin": "assistant_positive_box", "reference": box["id"]} for box in row["boxes"])
        else:
            if row.get("status") != "absent" or row["audit"].get("complete") is not True:
                raise ValueError("Background crops require an explicit complete absence audit")
            width, height = frame["width"], frame["height"]
            bounds = [[0, 0, width, height]] + [[x, y, x+width/2, y+height/2] for x in (0, width/2) for y in (0, height/2)]
            specs.extend({"frameId": frame["id"], "bbox": box, "target": 0, "origin": "explicit_absent_frame_crop",
                          "reference": "full" if index == 0 else "quadrant-" + str(index)} for index, box in enumerate(bounds))
    for review in negative_data["reviews"]:
        if review.get("negative") is not True:
            exclusions.append({"frameId": review["frameId"], "proposalBoxId": review["proposalBoxId"], "reason": "crop_not_confidently_negative"})
            continue
        frame = frames[review["frameId"]]
        proposal = json.loads((Path(proposals) / "frames" / ("main--" + frame["id"] + ".json")).read_text())
        bbox = explicit_negative_crop(review, frame, proposal)
        specs.append({"frameId": frame["id"], "bbox": bbox, "target": 0, "origin": "explicit_negative_crop_audit",
                      "reference": review["proposalBoxId"], "cropAudit": review})
    output.mkdir(parents=True, exist_ok=False)
    (output / "crops").mkdir()
    records, seen, validated_frames = [], {}, {}
    for spec in specs:
        frame = frames[spec["frameId"]]
        source = sources[frame["sourceId"]]
        if frame["id"] not in validated_frames:
            if source.get("licenseAllowsTraining") is not True or source.get("evaluationOnly") is not False or frame.get("evaluationOnly") is True:
                raise ValueError("Crop source is not training eligible")
            with Image.open(root / frame["path"]) as image:
                signature = signatures(image)
            keys = ["source:" + source["sourceGroupId"], "video:" + source["videoHash"]]
            if blocked_by_holdout(keys, frame["sha256"], signature, holdouts):
                raise ValueError("Crop source intersects a frozen human holdout")
            validated_frames[frame["id"]] = True
        checked_box(spec["bbox"], frame)
        with Image.open(root / frame["path"]) as image:
            crop = native_crop(image, spec["bbox"])
        fingerprint = sha(str(crop.size).encode() + crop.tobytes())
        if fingerprint in seen:
            if seen[fingerprint] != spec["target"]:
                raise ValueError("Conflicting weak labels for identical crop pixels")
            continue
        seen[fingerprint] = spec["target"]
        name = "crops/" + fingerprint + ".png"
        crop.save(output / name)
        records.append({**spec, "imagePath": name, "cropSha256": sha((output / name).read_bytes()), "originalSha256": frame["sha256"],
                        "width": crop.width, "height": crop.height, "sourceId": source["id"], "sourceGroupId": source["sourceGroupId"],
                        "sourceCredits": source, "humanVerified": False, "labelAuthority": "assistant-reviewed-ai"})
    if len({row["sourceGroupId"] for row in records}) < 3 or {row["target"] for row in records} != {0, 1}:
        raise ValueError("Verifier requires both classes and at least three eligible source groups")
    report = {"schema": "mycoscout.assistant-crop-training.v1", "records": records, "excluded": exclusions,
              "pilotDatasetManifestSha256": sha(manifest_bytes), "negativeAuditSha256": sha(Path(negative_audit).read_bytes()),
              "sourceManifestSha256": sha(raw), "frozenHumanDatasets": versions,
              "positiveCrops": sum(row["target"] == 1 for row in records), "negativeCrops": sum(row["target"] == 0 for row in records),
              "humanVerified": False, "automaticActivation": False}
    write_new(output / "manifest.json", report)
    return report


def vectors(images, model, transform):
    import torch
    result = []
    for start in range(0, len(images), 16):
        batch = torch.stack([transform(image) for image in images[start:start+16]])
        with torch.inference_mode():
            result.extend(model(batch).cpu().numpy())
    return np.asarray(result, dtype=np.float64).reshape(-1, 512)


def run(dataset, negative_audit, collection, proposals, learning_directory, snapshot_path, output):
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    crops = prepare_crops(dataset, negative_audit, collection, proposals, learning_directory, output / "training")
    torch.set_num_threads(2)
    model, transform = load_backbone(ROOT / "models/newfeaturecache/resnet18-f37072fd.pth")
    images = []
    for row in crops["records"]:
        with Image.open(output / "training" / row["imagePath"]) as image:
            images.append(image.convert("RGB"))
    features = vectors(images, model, transform)
    labels = np.asarray([row["target"] for row in crops["records"]])
    groups = np.asarray([row["sourceGroupId"] for row in crops["records"]])
    oof = np.zeros(len(labels))
    folds = []
    for group in sorted(set(groups)):
        train_indices, validation_indices = groups != group, groups == group
        if len(set(labels[train_indices])) != 2:
            raise ValueError("Each leave-source-out training fold needs both weak classes")
        head = fit_head(features[train_indices], labels[train_indices], regularization=.1)
        oof[validation_indices] = head_scores(features[validation_indices], head)
        folds.append({"heldoutTrainingGroup": group, "fitGroups": sorted(set(groups[train_indices])),
                      "fitCrops": int(train_indices.sum()), "validationCrops": int(validation_indices.sum())})
    threshold, policy = select_training_threshold(labels, oof)
    head = fit_head(features, labels, regularization=.1)
    checkpoint = {"schema": "mycoscout.assistant-proposal-verifier.v1", "head": head, "threshold": threshold,
                  "officialWeightSha256": EXPECTED_WEIGHT_SHA, "regularization": .1, "thresholdSelection": policy,
                  "trainingManifestSha256": sha((output / "training/manifest.json").read_bytes()),
                  "trainingGroups": sorted(set(groups)), "humanVerified": False, "labelAuthority": "assistant-reviewed-ai",
                  "probabilityCalibrated": False, "activationAllowed": False, "flightGateEnabled": False}
    write_new(output / "candidate.json", checkpoint)
    # Human evaluation is deliberately read only after model/policy are frozen.
    snapshot_path = Path(snapshot_path)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if any(row["sourceGroupId"] in set(groups) for row in snapshot["records"]):
        raise ValueError("Human evaluation source overlaps weak training sources")
    baseline = evaluate_snapshot(snapshot, proposals, .3)
    if baseline["status"] != "completed":
        raise ValueError("Original proposals incomplete for frozen human evaluation")
    root, raw, manifest = read_collection(collection)
    if sha(raw) != snapshot["sourceManifestSha256"]:
        raise ValueError("Human evaluation collection changed")
    frames = {frame["id"]: frame for frame in manifest["frames"]}
    for row in snapshot["records"]:
        frame = frames[row["frameId"]]
        if frame["sha256"] != row["imageSha256"]:
            raise ValueError("Human evaluation image hash changed")
        result = json.loads((Path(proposals) / "frames" / ("main--" + frame["id"] + ".json")).read_text())
        candidates = [box for box in result["boxes"] if box["score"] >= .3]
        with Image.open(root / frame["path"]) as image:
            crop_images = [native_crop(image, checked_box(box["bbox"], frame)) for box in candidates]
        scores = head_scores(vectors(crop_images, model, transform), head)
        accepted = [{**box, "verifierScore": float(score)} for box, score in zip(candidates, scores) if score >= threshold]
        write_new(output / "verified-proposals/frames" / ("main--" + frame["id"] + ".json"),
                  {**result, "boxes": accepted, "verifier": {"checkpointSha256": sha((output / "candidate.json").read_bytes()),
                    "threshold": threshold, "probabilityCalibrated": False, "humanVerified": False, "activationAllowed": False}})
    evaluated = evaluate_snapshot(snapshot, output / "verified-proposals", .3)
    report = {"schema": "mycoscout.assistant-verifier-model-card.v1", "status": "completed", "humanVerified": False,
              "positiveCrops": crops["positiveCrops"], "negativeCrops": crops["negativeCrops"], "trainingGroups": sorted(set(groups)),
              "featureFamily": "frozen ResNet18 ImageNet1K V1", "regularization": .1, "folds": folds,
              "threshold": threshold, "thresholdPolicy": policy, "scoreThreshold": .3, "humanFrames": len(snapshot["records"]),
              "baseline": baseline, "candidate": evaluated, "checkpointSha256": sha((output / "candidate.json").read_bytes()),
              "snapshotSha256": sha(snapshot_path.read_bytes()), "activationEligible": False, "automaticActivation": False,
              "limitations": ["Assistant crop labels remain weak and can be wrong; no human labels were added.",
                "Threshold uses leave-source-out weak training scores, never human test tuning; tiny training groups can still overfit.",
                "The 19 complete human frames are a selected single-creator sample, not field or low-pixel validation.",
                "Filtering can remove real mushrooms and cannot recover missed proposals or establish species, distance or safety."]}
    write_new(output / "model-card.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/assistant-video-pilot-dataset-v1")
    parser.add_argument("--negative-audit", type=Path, default=ROOT / "data/ai-video-review-v1/assistant-negative-crop-audit.json")
    parser.add_argument("--collection", type=Path, default=ROOT / "data/video-discovery-v1")
    parser.add_argument("--proposals", type=Path, default=ROOT / "data/ai-video-review-v1")
    parser.add_argument("--learning-datasets", type=Path, default=ROOT / "data/learning/datasets")
    parser.add_argument("--snapshot", type=Path, default=ROOT / "data/ai-video-review-v1/human-grades-snapshot.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.dataset, args.negative_audit, args.collection, args.proposals, args.learning_datasets, args.snapshot, args.output)
    print(json.dumps({"status": result["status"], "positiveCrops": result["positiveCrops"], "negativeCrops": result["negativeCrops"],
                      "threshold": result["threshold"], "baseline": result["baseline"]["metrics"], "candidate": result["candidate"]["metrics"],
                      "activationEligible": False}), flush=True)


if __name__ == "__main__":
    main()
