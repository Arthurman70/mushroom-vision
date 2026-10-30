"""Small trainable candidate-crop baseline, deliberately distinct from a detector."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math

import numpy as np
from PIL import Image, ImageOps

from .dataset import image_path

FEATURE_VERSION = "rgb-hsv-hist-spatial-edges-v1"


def features(image):
    """Color distribution, spatial appearance, and gradients; no location metadata."""
    small = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    rgb = np.asarray(small, dtype=np.float64) / 255.0
    hsv = np.asarray(small.convert("HSV"), dtype=np.float64) / 255.0
    vectors = [rgb.mean(axis=(0, 1)), rgb.std(axis=(0, 1))]
    for data in (rgb, hsv):
        for channel in range(3):
            vectors.append(np.histogram(data[:, :, channel], bins=12, range=(0, 1))[0] / (32 * 32))
    # Spatial averages retain shape/texture information absent from color-only counts.
    vectors.append(rgb.reshape(4, 8, 4, 8, 3).mean(axis=(1, 3)).flatten())
    gray = rgb.mean(axis=2)
    dx, dy = np.diff(gray, axis=1), np.diff(gray, axis=0)
    vectors.append(np.array([np.mean(np.abs(dx)), np.mean(np.abs(dy)), np.std(dx), np.std(dy)]))
    return np.concatenate(vectors)


def samples(data, manifest_path):
    result = {split: [] for split in ("train", "validation", "test")}
    for record in data["images"]:
        if record["synthetic"] or record["annotationStatus"] not in ("complete", "partial"):
            continue
        split = record.get("split")
        if split not in result:
            raise ValueError("Every labeled real image must have a group-aware split before training")
        with Image.open(image_path(manifest_path, record)) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            for annotation in record["annotations"]:
                bbox = annotation["bboxPixelsXYXY"]
                crop = image.crop((math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2]), math.ceil(bbox[3])))
                result[split].append({"features": features(crop), "target": int(annotation["label"] == "mushroom"),
                                      "groupId": record["groupId"], "originalSha256": record["originalSha256"], "imageId": record["id"]})
    return result


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -40, 40)))


def summarize(probabilities, labels, threshold):
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (labels == 1)))
    fp = int(np.sum(predicted & (labels == 0)))
    tn = int(np.sum(~predicted & (labels == 0)))
    fn = int(np.sum(~predicted & (labels == 1)))
    return {"samples": len(labels), "truePositive": tp, "falsePositive": fp, "trueNegative": tn, "falseNegative": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "accuracy": (tp + tn) / len(labels) if len(labels) else None}


def train(data, manifest_path, epochs=1000, learning_rate=0.05, l2=0.01):
    if not isinstance(epochs, int) or not 1 <= epochs <= 100_000:
        raise ValueError("Epochs must be between 1 and 100000")
    if not 0 < learning_rate <= 1 or not 0 <= l2 <= 10:
        raise ValueError("Invalid optimizer settings")
    rows = samples(data, manifest_path)
    minimum_groups = {"train": 6, "validation": 2, "test": 2}
    datasets, stats = {}, {}
    for split, items in rows.items():
        groups = set(r["groupId"] for r in items)
        originals = set(r["originalSha256"] for r in items)
        labels = Counter(r["target"] for r in items)
        required = minimum_groups[split]
        if len(groups) < required or len(originals) < required:
            raise ValueError(f"Verifier needs at least {required} independent labeled original-photo groups in {split}; found {len(groups)} groups / {len(originals)} originals")
        minimum_class = 3 if split == "train" else 1
        if min(labels.get(0, 0), labels.get(1, 0)) < minimum_class:
            raise ValueError(f"Verifier needs mushroom and explicitly reviewed background crops in {split}")
        datasets[split] = np.array([r["features"] for r in items]), np.array([r["target"] for r in items])
        stats[split] = {"groups": len(groups), "originals": len(originals), "mushroomCrops": labels[1], "backgroundCrops": labels[0]}
    x, y = datasets["train"]
    mean, std = x.mean(axis=0), x.std(axis=0)
    std = np.where(std < 1e-6, 1, std)
    x = (x - mean) / std
    weights, bias = np.zeros(x.shape[1]), 0.0
    # Equal total class weight makes explicit background examples matter even when positives dominate.
    sample_weights = np.where(y == 1, len(y) / (2 * np.sum(y == 1)), len(y) / (2 * np.sum(y == 0)))
    for _ in range(epochs):
        error = (sigmoid(x @ weights + bias) - y) * sample_weights
        weights -= learning_rate * (x.T @ error / len(y) + l2 * weights)
        bias -= learning_rate * error.mean()
    val_x, val_y = datasets["validation"]
    val_probabilities = sigmoid(((val_x - mean) / std) @ weights + bias)
    thresholds = np.linspace(0.05, 0.95, 19)
    threshold = max(thresholds, key=lambda t: (summarize(val_probabilities, val_y, t)["f1"] or 0, t))
    metrics = {}
    for split, (split_x, split_y) in datasets.items():
        probabilities = sigmoid(((split_x - mean) / std) @ weights + bias)
        metrics[split] = summarize(probabilities, split_y, float(threshold))
    digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    checkpoint = {"schema": "mycoscout.candidate-verifier.v1", "featureVersion": FEATURE_VERSION,
                  "weights": weights.tolist(), "bias": float(bias), "mean": mean.tolist(), "std": std.tolist(),
                  "threshold": float(threshold), "trainingManifestSha256": digest,
                  "trainingGroups": stats, "optimizer": {"epochs": epochs, "learningRate": learning_rate, "l2": l2},
                  "thresholdSelection": "Maximum crop-classification F1 on validation groups only; tie chooses stricter threshold",
                  "limitations": ["Candidate crop classifier: it cannot discover objects, identify species, or measure edibility.",
                                  "Uncalibrated linear baseline; reported crop metrics are not end-to-end detector metrics.",
                                  "Minimum data checks only permit an experiment; they do not establish deployment accuracy.",
                                  "Synthetic variants excluded from training, validation, and test.",
                                  "A mushroom-vs-background score is never a flight clearance signal."]}
    return checkpoint, {"schema": "mycoscout.verifier-training-report.v1", "status": "trained", "featureVersion": FEATURE_VERSION,
                        "trainingManifestSha256": digest, "threshold": float(threshold), "sampleCounts": stats,
                        "cropMetrics": metrics, "limitations": checkpoint["limitations"]}


def score_image(image, boxes, checkpoint):
    if checkpoint.get("schema") != "mycoscout.candidate-verifier.v1" or checkpoint.get("featureVersion") != FEATURE_VERSION:
        raise ValueError("Unsupported verifier checkpoint")
    weights, mean, std = (np.array(checkpoint[key], dtype=float) for key in ("weights", "mean", "std"))
    if not (weights.shape == mean.shape == std.shape) or not np.all(std > 0):
        raise ValueError("Invalid checkpoint dimensions")
    if not all(np.all(np.isfinite(arr)) for arr in (weights, mean, std)):
        raise ValueError("Checkpoint contains non-finite values")
    scores = []
    for box in boxes:
        x1, y1, x2, y2 = box
        vector = features(image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))))
        value = float(sigmoid(((vector - mean) / std) @ weights + checkpoint["bias"]))
        scores.append({"bboxPixelsXYXY": box, "verifierScore": value,
                       "passesVerifier": value >= checkpoint["threshold"], "species": None})
    return scores
