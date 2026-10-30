"""Human-reviewed mushroom detector training, evaluation and local inference.

Pretrained initialization: torchvision Faster R-CNN MobileNetV3 Large 320 FPN,
official COCO weights. The COCO classification head is replaced with two classes
(background/mushroom). No species, edibility or flight-performance claims follow.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import time
import uuid

from PIL import Image, ImageOps

from adapters.export_reviewed_detection_dataset import SCHEMA, digest, inside, readiness, write_json

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "fasterrcnn_mobilenet_v3_large_320_fpn"
WEIGHT_URL = "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
WEIGHT_FILE = ROOT / "models" / "reviewed-detector" / "fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"
WEIGHT_HASH_PREFIX = "907ea3f9"
WEIGHT_SHA256 = "907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5"
BINS = ["under4", "4to8", "8to16", "16to32", "32to64", "64plus"]


def emit(**data):
    print(json.dumps(data, allow_nan=False), flush=True)


def width_bin(width):
    return BINS[next((i for i, maximum in enumerate([4, 8, 16, 32, 64]) if width < maximum), 5)]


def iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union > 0 else 0


def score_frames(frames, threshold=.3):
    """Maximum one-to-one cardinality matching at IoU >= .5, original pixels."""
    tp = fp = fn = 0
    bins = {key: {"matched": 0, "total": 0} for key in BINS}
    by_source = {}
    for frame in frames:
        truths = frame["truth"]
        predictions = [p["bbox"] for p in frame["predictions"] if p["score"] >= threshold]
        edges = [[j for j, b in enumerate(predictions) if iou(a, b) >= .5] for a in truths]
        owners = {}
        def augment(index, seen):
            for prediction in edges[index]:
                if prediction in seen:
                    continue
                seen.add(prediction)
                if prediction not in owners or augment(owners[prediction], seen):
                    owners[prediction] = index
                    return True
            return False
        for index in range(len(truths)):
            augment(index, set())
        matched = set(owners.values())
        tp += len(matched); fp += len(predictions) - len(matched); fn += len(truths) - len(matched)
        source = by_source.setdefault(frame.get("group", "unknown"), {"frames": 0, "truePositives": 0, "falsePositives": 0, "falseNegatives": 0})
        source["frames"] += 1; source["truePositives"] += len(matched)
        source["falsePositives"] += len(predictions) - len(matched); source["falseNegatives"] += len(truths) - len(matched)
        for index, box in enumerate(truths):
            bucket = bins[width_bin(box[2]-box[0])]
            bucket["total"] += 1; bucket["matched"] += index in matched
    precision = tp / (tp+fp) if tp+fp else None
    recall = tp / (tp+fn) if tp+fn else None
    return {"frames": len(frames), "truePositives": tp, "falsePositives": fp, "falseNegatives": fn,
            "precision": precision, "recall": recall, "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "falsePositivesPerFrame": fp/len(frames) if frames else None, "scoreThreshold": threshold, "iouThreshold": .5,
            "pixelWidthBins": {key: value | {"recall": value["matched"]/value["total"] if value["total"] else None} for key, value in bins.items()},
            "sourceGroups": by_source, "originalPixels": True, "actualDistanceMeasured": False}


def read_dataset(directory, require_ready=True):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ready = json.loads((directory / "READY.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or ready.get("manifestSha256") != digest(manifest_path.read_bytes()):
        raise ValueError("Frozen dataset manifest integrity failed")
    groups, pixels = {}, set()
    for row in manifest["records"]:
        split = row["split"]
        if split not in {"train", "validation", "test"} or (row["evaluationQuarantine"] and split != "test"):
            raise ValueError("Evaluation quarantine or partition integrity failed")
        if row["leakageGroup"] in groups and groups[row["leakageGroup"]] != split:
            raise ValueError("A source/session group crosses partitions")
        groups[row["leakageGroup"]] = split
        if row["pixelSha256"] in pixels:
            raise ValueError("Repeated pixels in frozen dataset")
        pixels.add(row["pixelSha256"])
        if digest(inside(directory, row["imagePath"]).read_bytes()) != row["imageSha256"]:
            raise ValueError("Frozen image changed")
    computed = readiness(manifest["records"])
    if require_ready and not computed["ready"]:
        raise ValueError("Insufficient reviewed data: " + "; ".join(computed["reasons"]))
    return manifest


def build_model(pretrained=True):
    import torch
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    # Explicit None prevents incidental backbone downloads. Only a fixed,
    # separately requested official checkpoint may initialize this detector.
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None)
    # Official detection weights use FrozenBatchNorm. Creating the model with
    # weights=None otherwise chooses BatchNorm (different state-dict buffers).
    # Match the official architecture without triggering implicit downloads.
    from torchvision.ops.misc import FrozenBatchNorm2d
    def freeze_norm(module):
        for name, child in list(module.named_children()):
            if isinstance(child, torch.nn.BatchNorm2d):
                module._modules[name] = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            else:
                freeze_norm(child)
    freeze_norm(model.backbone)
    if pretrained:
        if not WEIGHT_FILE.is_file() or digest(WEIGHT_FILE.read_bytes()) != WEIGHT_SHA256:
            raise ValueError("Official pretrained weights are missing or invalid. Run download-pretrained first.")
        model.load_state_dict(torch.load(WEIGHT_FILE, map_location="cpu", weights_only=True))
    model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
    # Reduces work and overfitting on early small collections; FPN/RPN/head learn.
    for parameter in model.backbone.body.parameters():
        parameter.requires_grad = False
    return model


def tensor_image(path):
    from torchvision.transforms.functional import pil_to_tensor
    with Image.open(path) as image:
        return pil_to_tensor(ImageOps.exif_transpose(image).convert("RGB")).float() / 255


def training_example(directory, row, augment=False):
    import torch
    image = tensor_image(inside(Path(directory), row["imagePath"]))
    boxes = torch.tensor([b["bboxPixelsXYXY"] for b in row["boxes"]], dtype=torch.float32).reshape(-1, 4)
    if augment:
        from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation
        # Mild photometric changes preserve plausible color variation; no hue
        # replacement or generated mushrooms are used as real observations.
        image = adjust_saturation(adjust_contrast(adjust_brightness(image, random.uniform(.85, 1.15)), random.uniform(.85, 1.15)), random.uniform(.85, 1.15))
        if random.random() < .5:
            image = image.flip(-1)
            if len(boxes):
                left = image.shape[-1] - boxes[:, 2].clone()
                boxes[:, 2] = image.shape[-1] - boxes[:, 0]
                boxes[:, 0] = left
    return image, {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}


def predictions_for(model, image, threshold=.05):
    import torch
    model.eval()
    device = next(model.parameters()).device
    with torch.inference_mode():
        prediction = model([image.to(device)])[0]
    return [{"bbox": box.tolist(), "score": float(score)} for box, score, label in
            zip(prediction["boxes"].cpu(), prediction["scores"].cpu(), prediction["labels"].cpu()) if int(label) == 1 and float(score) >= threshold]


def evaluate(model, directory, rows, threshold):
    frames = [{"id": row["id"], "group": row["leakageGroup"], "truth": [b["bboxPixelsXYXY"] for b in row["boxes"]],
               "predictions": predictions_for(model, tensor_image(inside(Path(directory), row["imagePath"])), min(.05, threshold))} for row in rows]
    return score_frames(frames, threshold), frames


def release_evaluation(model, directory, rows, threshold, validation_only=False):
    """Daily optimization never evaluates the reserved release-test footage."""
    return (None, None) if validation_only else evaluate(model, directory, rows, threshold)


def train(directory, output, epochs=3, learning_rate=.005, score_threshold=.3, seed=4127, validation_only=False):
    if type(epochs) is not int or not 1 <= epochs <= 30 or not .00001 <= learning_rate <= .02 or not .05 <= score_threshold <= .95:
        raise ValueError("Training parameters are outside supported limits")
    manifest = read_dataset(directory)
    if type(validation_only) is not bool:
        raise ValueError("validation_only must be a boolean")
    manifest_sha = digest((Path(directory) / "manifest.json").read_bytes())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    import torch
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)
    splits = {split: [row for row in manifest["records"] if row["split"] == split] for split in {"train", "validation", "test"}}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, momentum=.9, weight_decay=.0005)
    history, best_f1, best_epoch = [], -1, None
    started = time.time()
    for epoch in range(1, epochs+1):
        random.shuffle(splits["train"])
        model.train()
        # Frozen backbone BatchNorm buffers also remain frozen.
        model.backbone.body.eval()
        losses = []
        for index, row in enumerate(splits["train"]):
            image, target = training_example(directory, row, augment=True)
            loss = sum(model([image.to(device)], [{k: v.to(device) for k, v in target.items()}]).values())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss; candidate was not released")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10)
            optimizer.step(); losses.append(float(loss.detach().cpu()))
            emit(status="progress", phase="train", epoch=epoch, epochs=epochs, image=index+1, images=len(splits["train"]), loss=losses[-1])
        validation, _ = evaluate(model, directory, splits["validation"], score_threshold)
        epoch_record = {"epoch": epoch, "meanLoss": sum(losses)/len(losses), "validation": validation}
        history.append(epoch_record)
        emit(status="progress", phase="validation", **epoch_record)
        if (validation["f1"] or 0) > best_f1:
            best_f1, best_epoch = validation["f1"] or 0, epoch
            checkpoint = {"schema": "mycoscout.detector-checkpoint.v1", "architecture": MODEL_NAME,
                          "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                          "datasetId": manifest["datasetId"], "epoch": epoch, "scoreThreshold": score_threshold,
                          "datasetManifestSha256": manifest_sha, "validationOnly": validation_only,
                          "classes": {0: "background", 1: "mushroom"}}
            torch.save(checkpoint, output / "candidate.pt")
    model, metadata = load_detector(output / "candidate.pt", device=str(device))
    # Automatic daily jobs leave test scoring to a separate explicit release check.
    test_metrics, predictions = release_evaluation(model, directory, splits["test"], score_threshold, validation_only)
    validation = history[best_epoch-1]["validation"]
    card = {"schema": "mycoscout.detector-model-card.v1", "architecture": MODEL_NAME,
            "datasetId": manifest["datasetId"], "datasetManifestSha256": manifest_sha,
            "checkpoint": "candidate.pt", "checkpointSha256": digest((output / "candidate.pt").read_bytes()),
            "bestEpoch": best_epoch, "epochs": epochs, "learningRate": learning_rate, "scoreThreshold": score_threshold,
            "seed": seed, "device": str(device), "elapsedSeconds": time.time()-started, "history": history,
            "validation": validation, "test": test_metrics, "testMetrics": test_metrics,
            "validationOnly": validation_only, "testEvaluationPerformed": not validation_only,
            "activationEligible": not validation_only and validation["truePositives"] > 0 and test_metrics["truePositives"] > 0,
            "pretrained": {"source": WEIGHT_URL, "sha256": digest(WEIGHT_FILE.read_bytes()),
                "trainingData": "COCO", "codeLicense": "BSD-3-Clause", "license": "torchvision pretrained model terms; COCO source-image rights remain with their owners",
                "termsUrl": "https://pytorch.org/vision/stable/models.html"},
            "automaticActivation": False, "operationallyValidated": False,
            "limitations": ["Experimental mushroom/non-mushroom detector; no species determination or flight certification.",
                "A positive heldout match only permits explicit experimental activation; it is not an operational performance threshold.",
                "This is a new detector family, not a demonstrated improvement over GroundingDINO. Compare both on the same heldout frames."]}
    write_json(output / "model-card.json", card)
    if not validation_only:
        write_json(output / "test-predictions.json", predictions)
    write_json(output / "COMPLETE.json", {"checkpointSha256": card["checkpointSha256"], "datasetId": manifest["datasetId"],
                                        "modelCardSha256": digest((output / "model-card.json").read_bytes())})
    return {"status": "completed", "modelCard": card, "checkpoint": str(output / "candidate.pt")}


def load_detector(checkpoint, device="cpu"):
    """Used by live capture after station activation; weights_only blocks pickle code."""
    import torch
    data = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if data.get("schema") != "mycoscout.detector-checkpoint.v1" or data.get("architecture") != MODEL_NAME:
        raise ValueError("Unsupported reviewed detector checkpoint")
    model = build_model(pretrained=False)
    model.load_state_dict(data["state_dict"], strict=True)
    model.to(device).eval()
    return model, {key: value for key, value in data.items() if key != "state_dict"}


def download_pretrained():
    import torch
    WEIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not WEIGHT_FILE.exists():
        torch.hub.download_url_to_file(WEIGHT_URL, str(WEIGHT_FILE), hash_prefix=WEIGHT_HASH_PREFIX, progress=False)
    actual = digest(WEIGHT_FILE.read_bytes())
    if actual != WEIGHT_SHA256:
        raise ValueError("Official pretrained checkpoint hash verification failed")
    return {"source": WEIGHT_URL, "sha256": actual, "path": str(WEIGHT_FILE)}


def synthetic_smoke(output):
    """One fabricated example verifies plumbing, never recognition accuracy."""
    import torch
    torch.manual_seed(71); torch.set_num_threads(min(4, torch.get_num_threads()))
    model = build_model(pretrained=False)
    model.train(); model.backbone.body.eval()
    before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    image = torch.rand(3, 320, 320)
    target = {"boxes": torch.tensor([[110., 100., 190., 220.]]), "labels": torch.tensor([1])}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.005)
    losses = model([image], [target]); loss = sum(losses.values())
    if not torch.isfinite(loss):
        raise ValueError("Synthetic gradient smoke produced a non-finite loss")
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    change = float((model.roi_heads.box_predictor.cls_score.weight.detach()-before).abs().max())
    if not change > 0:
        raise ValueError("Synthetic gradient smoke did not update detector weights")
    temporary = ROOT / "experiments" / "tmp-tests" / ("synthetic-detector-" + uuid.uuid4().hex + ".pt")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": "mycoscout.detector-checkpoint.v1", "architecture": MODEL_NAME,
                "state_dict": model.state_dict(), "datasetId": "synthetic-toolchain-only", "scoreThreshold": .3}, temporary)
    try:
        restored, _ = load_detector(temporary)
        predictions = predictions_for(restored, image)
    finally:
        temporary.unlink(missing_ok=True)
    result = {"status": "passed", "syntheticOnly": True, "realMushroomTraining": False, "recognitionAccuracy": None,
              "pretrainedWeightsUsed": False, "optimizerSteps": 1, "loss": float(loss.detach()),
              "lossComponents": {k: float(v.detach()) for k, v in losses.items()},
              "maxClassifierWeightChange": change, "checkpointReloaded": True, "inferenceRan": True,
              "predictionCountHasNoAccuracyMeaning": len(predictions),
              "notice": "Random pixels and a fabricated box verify gradient, checkpoint and inference plumbing only. No deployed model was changed."}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("download-pretrained")
    smoke = sub.add_parser("synthetic-smoke")
    smoke.add_argument("--output", type=Path, required=True)
    training = sub.add_parser("train")
    training.add_argument("--dataset", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--learning-rate", type=float, default=.005)
    training.add_argument("--score-threshold", type=float, default=.3)
    training.add_argument("--validation-only", action="store_true", help="Skip reserved test scoring for daily optimization")
    prediction = sub.add_parser("predict")
    prediction.add_argument("--checkpoint", type=Path, required=True)
    prediction.add_argument("--image", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "download-pretrained":
            emit(status="completed", pretrained=download_pretrained())
        elif args.command == "synthetic-smoke":
            emit(**synthetic_smoke(args.output))
        elif args.command == "predict":
            model, metadata = load_detector(args.checkpoint)
            emit(status="completed", model=metadata, predictions=predictions_for(model, tensor_image(args.image), metadata["scoreThreshold"]))
        else:
            emit(**train(args.dataset, args.output, args.epochs, args.learning_rate, args.score_threshold, validation_only=args.validation_only))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        emit(status="failed", error=str(exc))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
