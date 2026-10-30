"""V2 frozen ResNet18 transfer experiment for weakly labeled mushroom photos.

This ranks black-trumpet-labelled photos/crops against other fungi. It is not
object detection, verified species identification, edibility advice or a flight
clearance system. Extraction is local; only official weights are ever downloaded
by the separate preparation command.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image, ImageOps

from .verifier import features as handcrafted_features

SCHEMA = "mycoscout.weak-fungi-dataset.v2"
ROOT = Path(__file__).resolve().parents[1]
EXPECTED_WEIGHT_SHA = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
LIMITATIONS = [
    "Targets are weak observation labels, not independent species or image-content confirmation.",
    "Photo/crop classification does not discover or locate mushrooms in woodland scenes.",
    "Black-trumpet-vs-other-fungi scores are uncalibrated, not probabilities of correct identification or edibility.",
    "Official preprocessing uses a central crop; off-center fungi may be omitted.",
    "Observer, observation, photo and content hashes are disjoint across splits; perceptual duplicates may remain.",
    "Empirical validation precision targets do not guarantee future precision.",
    "No physical detection range, depth accuracy, obstacle avoidance or automatic flight gate is validated.",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def photo_path(manifest, row):
    base = Path(manifest).resolve().parent
    path = (base / row["path"]).resolve()
    if not path.is_relative_to(base):
        raise ValueError("Photo escapes the manifest directory")
    return path


def validate_manifest(manifest, verify_files=True):
    data = read_json(manifest)
    if data.get("schema") != SCHEMA or not isinstance(data.get("images"), list):
        raise ValueError(f"Expected {SCHEMA} with images list")
    ids, group_splits, hashes = set(), defaultdict(set), set()
    for row in data["images"]:
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("Missing or repeated image identifier")
        ids.add(row["id"])
        if row.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Split must be assigned before feature extraction")
        if row.get("weakLabel") not in {"black_trumpet", "other_fungi"}:
            raise ValueError("Unexpected weak classification label")
        if row.get("synthetic") is not False:
            raise ValueError("Training/evaluation manifest must contain original photos only")
        if row.get("sha256") in hashes:
            raise ValueError("Exact duplicate image content in original dataset")
        hashes.add(row.get("sha256"))
        if not row.get("photoLicense") or not row.get("sourceUrl"):
            raise ValueError("License/source provenance is required")
        for field in ("groupId", "observerId", "observationId", "photoId", "sha256", "originalSha256"):
            value = row.get(field)
            if value is None or value == "":
                raise ValueError(f"Required grouping field missing: {field}")
            group_splits[(field, str(value))].add(row["split"])
        path = photo_path(manifest, row)
        if verify_files and checksum(path) != row["sha256"]:
            raise ValueError(f"Image checksum changed: {row['id']}")
    leaking = [(field, value) for (field, value), splits in group_splits.items() if len(splits) > 1]
    if leaking:
        raise ValueError(f"Grouping leakage across splits: {leaking[:5]}")
    counts = {}
    for split in ("train", "validation", "test"):
        rows = [row for row in data["images"] if row["split"] == split]
        labels = Counter(row["weakLabel"] for row in rows)
        counts[split] = {"photos": len(rows), "observers": len({row["observerId"] for row in rows}),
                         "observations": len({row["observationId"] for row in rows}), "labels": dict(labels),
                         "cohorts": dict(Counter(row.get("cohort", "unknown") for row in rows))}
        minimum = 20 if split == "train" else 10
        if min(labels.get("black_trumpet", 0), labels.get("other_fungi", 0)) < minimum:
            raise ValueError(f"Need at least {minimum} photos per weak class in {split}")
        for label in ("black_trumpet", "other_fungi"):
            groups = {row["groupId"] for row in rows if row["weakLabel"] == label}
            if len(groups) < 5:
                raise ValueError(f"Need at least five independent observer groups for {label} in {split}")
    return data, counts


def load_backbone(weights_path):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18
    if checksum(weights_path) != EXPECTED_WEIGHT_SHA:
        raise ValueError("ResNet18 official weights checksum mismatch")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, ResNet18_Weights.IMAGENET1K_V1.transforms()


def extract_features(data, manifest, weights_path, cache_dir, batch_size=16, threads=4):
    import torch
    torch.set_num_threads(threads)
    digest = checksum(manifest)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"features-v2-{digest[:16]}-{EXPECTED_WEIGHT_SHA[:12]}.npz"
    ids = np.array([row["id"] for row in data["images"]])
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            if not np.array_equal(cached["ids"], ids) or str(cached["manifest_sha"].item()) != digest:
                raise ValueError("Feature cache provenance mismatch")
            return {"resnet18": cached["resnet18"].copy(), "handcrafted": cached["handcrafted"].copy()}, read_json(cache_path.with_suffix(".json"))
    model, transform = load_backbone(weights_path)
    deep, simple, batch = [], [], []
    started, inference_seconds = time.perf_counter(), 0.0
    for index, row in enumerate(data["images"]):
        with Image.open(photo_path(manifest, row)) as source:
            if source.width * source.height > 40_000_000:
                raise ValueError("Photo exceeds 40 MP")
            image = ImageOps.exif_transpose(source).convert("RGB")
            if image.size != (row["width"], row["height"]):
                raise ValueError(f"Image dimensions changed: {row['id']}")
            simple.append(handcrafted_features(image))
            batch.append(transform(image))
        if len(batch) == batch_size or index == len(data["images"]) - 1:
            inputs = torch.stack(batch)
            then = time.perf_counter()
            with torch.inference_mode():
                deep.extend(model(inputs).cpu().numpy())
            inference_seconds += time.perf_counter() - then
            batch = []
            if (index + 1) % 128 < batch_size or index == len(data["images"]) - 1:
                print(f"Extracted frozen features {index + 1}/{len(data['images'])}", flush=True)
    arrays = {"resnet18": np.asarray(deep, dtype=np.float64), "handcrafted": np.asarray(simple, dtype=np.float64)}
    for array in arrays.values():
        if not np.isfinite(array).all():
            raise ValueError("Feature extraction produced non-finite values")
    np.savez_compressed(cache_path, ids=ids, manifest_sha=np.array(digest), **arrays)
    metadata = {"schema": "mycoscout.frozen-features.v2", "manifestSha256": digest,
                "officialWeightSha256": EXPECTED_WEIGHT_SHA, "photos": len(ids),
                "extractionSeconds": time.perf_counter() - started,
                "backboneInferenceSeconds": inference_seconds, "batchSize": batch_size, "threads": threads,
                "featureDimensions": {key: array.shape[1] for key, array in arrays.items()},
                "preprocessing": "Official resize256/center-crop224/ImageNet normalization; handcrafted32x32 full photo",
                "cachePath": str(cache_path.resolve())}
    write_json(cache_path.with_suffix(".json"), metadata)
    return arrays, metadata


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def average_precision(labels, scores):
    if not np.sum(labels):
        return None
    order = np.argsort(-scores, kind="stable")
    labels, scores = labels[order], scores[order]
    ends = np.r_[np.where(np.diff(scores) != 0)[0], len(scores) - 1]
    true_positives = np.cumsum(labels)[ends]
    recall = true_positives / np.sum(labels)
    precision = true_positives / (ends + 1)
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def auc_roc(labels, scores):
    positives, negatives = int(np.sum(labels)), int(np.sum(labels == 0))
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((np.sum(ranks[labels == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def summarize(labels, scores, threshold):
    predicted = scores >= threshold
    tp, fp = int(np.sum(predicted & (labels == 1))), int(np.sum(predicted & (labels == 0)))
    fn, tn = int(np.sum(~predicted & (labels == 1))), int(np.sum(~predicted & (labels == 0)))
    return {"photos": len(labels), "positivePhotos": int(np.sum(labels)), "predictedPositive": int(np.sum(predicted)),
            "truePositive": tp, "falsePositive": fp, "falseNegative": fn, "trueNegative": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "accuracy": (tp + tn) / len(labels) if len(labels) else None,
            "balancedAccuracy": ((tp / (tp + fn)) + (tn / (tn + fp))) / 2 if tp + fn and tn + fp else None,
            "averagePrecision": average_precision(labels, scores), "rocAuc": auc_roc(labels, scores)}


def choose_threshold(labels, scores, precision_target=0.8):
    choices = [(float(value), summarize(labels, scores, float(value))) for value in np.linspace(.01, .99, 99)]
    valid = [(value, metrics) for value, metrics in choices
             if metrics["predictedPositive"] >= 10 and (metrics["precision"] or 0) >= precision_target]
    if valid:
        value, metrics = max(valid, key=lambda item: (item[1]["recall"] or 0, item[1]["precision"] or 0, item[0]))
        status = "empirical-validation-target-met"
    else:
        value, metrics = max(choices, key=lambda item: (item[1]["f1"] or 0, item[0]))
        status = "precision-target-not-met-fallback-to-validation-f1"
    return value, {"status": status, "precisionTarget": precision_target, "minimumPredictedPositive": 10,
                   "selectionSplit": "validation", "thresholdGrid": "0.01 through 0.99 in increments of 0.01",
                   "validationAtThreshold": metrics, "guaranteesFuturePrecision": False}


def fit_head(features, targets, regularization):
    import torch
    mean, std = features.mean(axis=0), features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    x = torch.tensor((features - mean) / std, dtype=torch.float64)
    y = torch.tensor(targets, dtype=torch.float64)
    weight = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros((), dtype=torch.float64, requires_grad=True)
    sample_weight = torch.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))
    optimizer = torch.optim.LBFGS([weight, bias], lr=1, max_iter=150, tolerance_grad=1e-7, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        logits = x @ weight + bias
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, y, reduction="none") * sample_weight).mean()
        loss = loss + 0.5 * regularization * torch.sum(weight * weight)
        loss.backward()
        return loss

    optimizer.step(closure)
    return {"mean": mean.tolist(), "std": std.tolist(), "weights": weight.detach().numpy().tolist(),
            "bias": float(bias.detach()), "regularization": regularization}


def head_scores(features, head):
    return sigmoid(((features - np.asarray(head["mean"])) / np.asarray(head["std"])) @ np.asarray(head["weights"]) + head["bias"])


def validate_checkpoint(checkpoint):
    if checkpoint.get("schema") != "mycoscout.black-trumpet-photo-ranker.v2":
        raise ValueError("Unsupported photo-ranker checkpoint")
    family = checkpoint.get("featureFamily")
    if family not in {"handcrafted", "resnet18"}:
        raise ValueError("Unexpected feature family")
    if checkpoint.get("officialWeightSha256") != EXPECTED_WEIGHT_SHA:
        raise ValueError("Unexpected backbone provenance")
    expected = 512 if family == "resnet18" else 130
    head = checkpoint["head"]
    arrays = [np.asarray(head[key], dtype=float) for key in ("weights", "mean", "std")]
    if any(array.shape != (expected,) or not np.isfinite(array).all() for array in arrays):
        raise ValueError("Invalid head dimensions or non-finite parameters")
    if not np.all(arrays[2] > 0) or not math.isfinite(head["bias"]) or not math.isfinite(head["threshold"]) or not 0 <= head["threshold"] <= 1:
        raise ValueError("Invalid standardization, bias or threshold")
    if checkpoint.get("probabilityCalibrated") is not False or checkpoint.get("flightGateEnabled") is not False:
        raise ValueError("Photo ranker must remain uncalibrated and disconnected from flight clearance")
    return family


def rank_photos(images, checkpoint, backbone=None, transform=None):
    """Score local PIL images/crops for review; never create locations or flight gates."""
    import torch
    family = validate_checkpoint(checkpoint)
    if family == "resnet18":
        if backbone is None or transform is None:
            raise ValueError("Supply the verified, frozen ResNet18 model and transform")
        with torch.inference_mode():
            vectors = backbone(torch.stack([transform(ImageOps.exif_transpose(image).convert("RGB")) for image in images])).cpu().numpy()
    else:
        vectors = np.asarray([handcrafted_features(ImageOps.exif_transpose(image).convert("RGB")) for image in images])
    scores = head_scores(vectors, checkpoint["head"])
    return [{"blackTrumpetPatternScore": float(score), "prioritizeHumanReview": bool(score >= checkpoint["head"]["threshold"]),
             "scoreCalibrated": False, "speciesConfirmed": False, "edibility": "not assessed", "flightGateEnabled": False}
            for score in scores]


def group_bootstrap(rows, labels, scores, threshold, iterations=200):
    groups = np.array([str(row["observerId"]) for row in rows])
    unique = np.unique(groups)
    rng = np.random.default_rng(20261026)
    values = defaultdict(list)
    for _ in range(iterations):
        selection = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([np.flatnonzero(groups == group) for group in selection])
        metric = summarize(labels[indices], scores[indices], threshold)
        for name in ("precision", "recall", "f1", "averagePrecision", "rocAuc"):
            if metric[name] is not None:
                values[name].append(metric[name])
    return {"method": "observer-group percentile bootstrap", "iterations": iterations, "seed": 20261026,
            "observerCount": len(unique), "intervals95": {key: [float(v) for v in np.quantile(items, [.025, .975])]
                                                            for key, items in values.items()},
            "limitation": "Reflects this weak-label sample and source groups; does not account for label error or deployment shift."}


def train_experiment(manifest, output_dir, weights_path, cache_dir, threads=4, precision_target=.8):
    import torch
    import torchvision
    if not 0 < precision_target <= 1 or not 1 <= threads <= 64:
        raise ValueError("Invalid precision target or CPU thread count")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Choose a new output directory to preserve prior experiment results")
    data, counts = validate_manifest(manifest)
    manifest_sha = checksum(manifest)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Freeze parameters before reading features or performance outcomes.
    protocol = {"schema": "mycoscout.transfer-protocol.v2", "manifestSha256": manifest_sha,
                "featureFamilies": ["handcrafted", "resnet18"], "l2Candidates": [.001, .01, .1],
                "architectureSelection": "Validation average precision, then validation ROC AUC; test not consulted",
                "precisionTarget": precision_target, "labelTask": "weak black_trumpet versus other_fungi",
                "splitCounts": counts, "officialWeightSha256": EXPECTED_WEIGHT_SHA, "limitations": LIMITATIONS}
    write_json(output_dir / "protocol.json", protocol)
    arrays, extraction = extract_features(data, manifest, weights_path, cache_dir, threads=threads)
    labels = np.array([int(row["weakLabel"] == "black_trumpet") for row in data["images"]])
    split_masks = {split: np.array([row["split"] == split for row in data["images"]]) for split in counts}
    training, reports, trial_rows, score_rows = {}, {}, [], []
    for family in protocol["featureFamilies"]:
        vectors = arrays[family]
        trials = []
        for l2 in protocol["l2Candidates"]:
            head = fit_head(vectors[split_masks["train"]], labels[split_masks["train"]], l2)
            scores = head_scores(vectors[split_masks["validation"]], head)
            threshold, selection = choose_threshold(labels[split_masks["validation"]], scores, precision_target)
            metric = selection["validationAtThreshold"]
            trials.append((metric["averagePrecision"], metric["rocAuc"], -l2, head, threshold, selection))
            trial_rows.append({"family": family, "l2": l2, "threshold": threshold, "thresholdSelection": selection})
        winner = max(trials, key=lambda value: value[:3])
        _, _, _, head, threshold, selection = winner
        training[family] = {**head, "threshold": threshold, "thresholdSelection": selection}
        reports[family] = {"selectedL2": head["regularization"], "threshold": threshold, "thresholdSelection": selection,
                           "metrics": {}, "testByCohort": {}}
    # Architecture selection happens before accessing test scores or metrics.
    selected_family = max(training, key=lambda family: (
        training[family]["thresholdSelection"]["validationAtThreshold"]["averagePrecision"],
        training[family]["thresholdSelection"]["validationAtThreshold"]["rocAuc"]))
    for family, head in training.items():
        scores = head_scores(arrays[family], head)
        for split, mask in split_masks.items():
            reports[family]["metrics"][split] = summarize(labels[mask], scores[mask], head["threshold"])
        test_rows = [row for row in data["images"] if row["split"] == "test"]
        reports[family]["testUncertainty"] = group_bootstrap(test_rows, labels[split_masks["test"]], scores[split_masks["test"]], head["threshold"])
        for cohort in sorted({row.get("cohort", "unknown") for row in test_rows}):
            mask = np.array([row["split"] == "test" and row.get("cohort", "unknown") == cohort for row in data["images"]])
            reports[family]["testByCohort"][cohort] = summarize(labels[mask], scores[mask], head["threshold"])
        for index, row in enumerate(data["images"]):
            score_rows.append({"imageId": row["id"], "split": row["split"], "weakLabel": row["weakLabel"],
                               "observerId": row["observerId"], "family": family, "score": float(scores[index]),
                               "selectedForReview": bool(scores[index] >= head["threshold"]), "sha256": row["sha256"]})
        checkpoint = {"schema": "mycoscout.black-trumpet-photo-ranker.v2", "featureFamily": family,
                      "trainingManifestSha256": manifest_sha, "officialWeightSha256": EXPECTED_WEIGHT_SHA,
                      "backbone": "resnet18.IMAGENET1K_V1-frozen" if family == "resnet18" else "handcrafted-rgb-hsv-texture",
                      "head": head, "classLabels": ["other_fungi", "black_trumpet"],
                      "pretrainedBackbone": {"source": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                                             "weights": "IMAGENET1K_V1", "sha256": EXPECTED_WEIGHT_SHA,
                                             "loading": "torch.load(weights_only=True)"},
                      "trainingData": {"source": "https://www.inaturalist.org", "manifestSha256": manifest_sha,
                                       "photos": len(data["images"]), "splitCounts": counts,
                                       "labels": "weak observation taxa; image content and species not independently verified",
                                       "licenses": sorted({row["photoLicense"]["code"] for row in data["images"]})},
                      "probabilityCalibrated": False, "flightGateEnabled": False, "limitations": LIMITATIONS}
        write_json(output_dir / f"{family}-checkpoint.json", checkpoint)
    report = {"schema": "mycoscout.transfer-study.v2", "status": "trained-and-heldout-evaluated",
              "manifestSha256": manifest_sha, "selectedFamilyByValidation": selected_family,
              "splitCounts": counts, "protocol": protocol, "extraction": extraction, "models": reports,
              "modelSelectionTrialsValidationOnly": trial_rows,
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "torchvision": torchvision.__version__, "threads": threads},
              "limitations": LIMITATIONS}
    write_json(output_dir / "photo-scores.json", {"schema": "mycoscout.photo-ranker-scores.v2", "scores": score_rows})
    write_json(output_dir / "study.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=ROOT / "models/newfeaturecache/resnet18-f37072fd.pth")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "models/newfeaturecache")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision-target", type=float, default=.8)
    args = parser.parse_args()
    report = train_experiment(args.manifest, args.output_dir, args.weights, args.cache_dir, args.threads, args.precision_target)
    print(json.dumps({"status": report["status"], "selectedFamilyByValidation": report["selectedFamilyByValidation"],
                      "splitCounts": report["splitCounts"], "testMetrics": {name: value["metrics"]["test"] for name, value in report["models"].items()}}, indent=2))


if __name__ == "__main__":
    main()
