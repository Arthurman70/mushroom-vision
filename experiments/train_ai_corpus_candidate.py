"""Fit an inactive detector from a frozen, explicitly AI-reviewed corpus.

All reviews retain AI authority. Uncertain/incomplete frames and reserved test
sources never become fit examples. Explicit nonfungal regions are optional hard
negatives only when they cannot cover any labeled mushroom.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.training_pipeline import build_model, MODEL_NAME, WEIGHT_SHA256, iou, predictions_for, score_frames
from vision.video_ai_pass import atomic_json

SCHEMA = "mushroom-vision.ai-corpus-training.v1"
CHECKPOINT_SCHEMA = "mushroom-vision.ai-corpus-candidate-checkpoint.v1"
KINDS = {"hat", "stump", "tree", "leaf", "rock", "wood", "person", "animal", "other"}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    if Path(path).stat().st_size > 64*1024*1024:
        raise ValueError("Metadata exceeds 64 MiB")
    def invalid(value):
        raise ValueError("Non-finite metadata: "+value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


def pixel_digest(image):
    return hashlib.sha256(json.dumps(list(image.size), separators=(",", ":")).encode()+image.tobytes()).hexdigest()


def valid_box(value, width, height):
    if (not isinstance(value, list) or len(value) != 4
        or any(type(n) not in (int, float) or not math.isfinite(n) for n in value)
        or not 0 <= value[0] < value[2] <= width or not 0 <= value[1] < value[3] <= height):
        raise ValueError("Invalid original-pixel region")
    return list(value)


def overlap_area(a, b):
    return max(0, min(a[2], b[2])-max(a[0], b[0]))*max(0, min(a[3], b[3])-max(a[1], b[1]))


def negative_regions(row):
    """A stump/tree containing even a sliver of positive truth is never negative."""
    positives = [box["bboxPixelsXYXY"] for box in row["boxes"]]
    result, rejected = [], 0
    for index, region in enumerate(row.get("confounders", [])):
        box = valid_box(region["bboxPixelsXYXY"], row["width"], row["height"])
        if region.get("explicitNonFungal") is not True or region.get("kind") not in KINDS:
            continue
        bounds = [math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3])]
        if any(overlap_area(bounds, positive) > 0 for positive in positives):
            rejected += 1
            continue
        if min(bounds[2]-bounds[0], bounds[3]-bounds[1]) < 16:
            continue
        result.append({"id": row["id"]+"--nonfungal-"+str(index), "kind": region["kind"], "bounds": bounds})
    return result, rejected


def balanced_order(examples, seed):
    """Interleave class and source groups while visiting every example once."""
    rng = random.Random(seed)
    groups = {True: defaultdict(list), False: defaultdict(list)}
    for example in examples:
        groups[example["positive"]][example["row"]["sourceGroupId"]].append(example)
    queues = {}
    for positive, sources in groups.items():
        keys = list(sources); rng.shuffle(keys)
        for values in sources.values():
            rng.shuffle(values)
        queue = []
        while keys:
            for key in keys[:]:
                queue.append(sources[key].pop())
                if not sources[key]:
                    keys.remove(key)
        queues[positive] = deque(queue)
    result = []
    while queues[True] or queues[False]:
        for positive in (True, False):
            if queues[positive]:
                result.append(queues[positive].popleft())
    return result


def load_records(manifest_path, inventory_path, ledger_path, root=ROOT):
    manifest, inventory, ledger = read(manifest_path), read(inventory_path), read(ledger_path)
    if (manifest.get("schema") != SCHEMA or manifest.get("reviewerKind") != "ai"
        or manifest.get("activationEligible") is not False
        or manifest.get("originalInventorySha256") != digest(inventory_path)
        or manifest.get("ledgerSha256") != digest(ledger_path)
        or inventory.get("schema") != "mycoscout.bulk-ai-inventory.v1"):
        raise ValueError("Frozen AI export provenance mismatch")
    sources = {row["id"]: row for row in inventory["records"]}
    if len(sources) != len(inventory["records"]):
        raise ValueError("Duplicate inventory identities")
    if (ledger.get("schema") != "mycoscout.bulk-ai-review-ledger.v1"
        or ledger.get("reviewerKind") != "ai" or ledger.get("humanVerified") is not False
        or ledger.get("inventorySha256") != digest(inventory_path)
        or not isinstance(ledger.get("records"), list)):
        raise ValueError("AI ledger must contain immutable records")
    ledger_records = {row["ledgerRecordHash"]: row for row in ledger["records"]}
    if len(ledger_records) != len(ledger["records"]):
        raise ValueError("Repeated ledger hashes")
    latest_originals = {}
    for record in ledger["records"]:
        if record.get("viewedOriginal") and record.get("observation", {}).get("mode") == "original-image":
            previous = latest_originals.get(record["imageId"])
            if previous is None or record["createdAt"] >= previous["createdAt"]:
                latest_originals[record["imageId"]] = record
    inventory_groups = {}
    for source in inventory["records"]:
        for key in source["groupKeys"]:
            if key in inventory_groups and inventory_groups[key] != source["split"]:
                raise ValueError("Inventory-linked source crosses frozen partitions")
            inventory_groups[key] = source["split"]
    rows, skipped, seen, group_splits, pixels = [], Counter(), set(), {}, {}
    for row in manifest["records"]:
        if row["id"] in seen:
            raise ValueError("Repeated exported identity")
        seen.add(row["id"])
        source = sources.get(row["id"])
        immutable = ledger_records.get(row.get("ledgerRecordHash"))
        if source is None or immutable is None or immutable.get("imageId") != row["id"]:
            raise ValueError("Export has no matching immutable original review")
        if immutable is not latest_originals.get(row["id"]):
            raise ValueError("Export uses superseded original review")
        # The export renames only its public id to the inventory identity. Every
        # ledger field that affects labels or authority must otherwise be exact.
        if any(row.get(key) != value for key, value in immutable.items() if key != "id"):
            raise ValueError("Export label or authority differs from immutable ledger")
        if (row.get("reviewerKind") != "ai" or row.get("humanVerified") is not False
            or row.get("activationEligible") is not False
            or row.get("imageSha256") != source["sha256"] or row.get("pixelSha256") != source["pixelSha256"]
            or row.get("imagePath") != source["path"] or row.get("split") != source["split"]
            or row.get("sourceGroupId") != source["sourceGroupId"] or row.get("groupKeys") != source["groupKeys"]):
            raise ValueError("Review identity, authority or frozen partition changed")
        if row.get("split") == "test":
            skipped["reserved_test"] += 1; continue
        if row.get("split") not in {"train", "validation"}:
            skipped["unassigned"] += 1; continue
        if row.get("complete") is not True or row.get("viewedOriginal") is not True or row.get("status") not in {"visible", "absent"}:
            skipped["uncertain_or_incomplete"] += 1; continue
        if not isinstance(row.get("evidence"), str) or not row["evidence"].strip():
            raise ValueError("Original-view evidence is required")
        if source.get("rights", {}).get("experimentalTrainingPermission") is not True:
            skipped["rights_not_eligible"] += 1; continue
        if row["split"] == "train" and source.get("experimentalTrainingEligible") is not True:
            raise ValueError("Source is not eligible for experimental training")
        path = (root/row["imagePath"]).resolve()
        if not path.is_relative_to((root/"data").resolve()) or path.stat().st_size > 16*1024*1024 or digest(path) != row["imageSha256"]:
            raise ValueError("Original image path or bytes changed")
        with Image.open(path) as image_source:
            image = ImageOps.exif_transpose(image_source).convert("RGB")
        if image.size != (source["width"], source["height"]) or pixel_digest(image) != source["pixelSha256"]:
            raise ValueError("Original oriented pixels changed")
        if source["pixelSha256"] in pixels:
            raise ValueError("Duplicate decoded pixels in fit/validation")
        pixels[source["pixelSha256"]] = row["split"]
        for key in source["groupKeys"]:
            if key in group_splits and group_splits[key] != row["split"]:
                raise ValueError("Linked sources cross fit/validation partitions")
            group_splits[key] = row["split"]
        boxes = row.get("boxes")
        if not isinstance(boxes, list) or len(boxes) > 128 or bool(boxes) != (row["status"] == "visible"):
            raise ValueError("Visible/absent labels and mushroom boxes disagree")
        for box in boxes:
            valid_box(box["bboxPixelsXYXY"], image.width, image.height)
        rows.append({**row, "width": image.width, "height": image.height, "leakageGroup": row["sourceGroupId"]})
    train = [row for row in rows if row["split"] == "train"]
    if len(train) < 8 or sum(bool(row["boxes"]) for row in train) < 4 or len({row["sourceGroupId"] for row in train}) < 2:
        raise ValueError("Need at least eight complete training frames, four positive frames and two source groups")
    return rows, {"exportedRecords": len(manifest["records"]), "acceptedFrames": len(rows), "skipped": dict(skipped),
                  "manifestSha256": digest(manifest_path), "inventorySha256": digest(inventory_path), "ledgerSha256": digest(ledger_path)}


def tensor_example(root, example, augment=False):
    import torch
    from torchvision.transforms.functional import pil_to_tensor, adjust_brightness, adjust_contrast, adjust_saturation
    row = example["row"]
    with Image.open(root/row["imagePath"]) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    boxes = [box["bboxPixelsXYXY"] for box in row["boxes"]]
    if example.get("crop"):
        image = image.crop(example["crop"])
        boxes = []
    tensor = pil_to_tensor(image).float()/255
    truth = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
    if augment:
        tensor = adjust_saturation(adjust_contrast(adjust_brightness(tensor, random.uniform(.9, 1.1)), random.uniform(.9, 1.1)), random.uniform(.9, 1.1))
        if random.random() < .5:
            tensor = tensor.flip(-1)
            if len(truth):
                left = tensor.shape[-1]-truth[:, 2].clone()
                truth[:, 2] = tensor.shape[-1]-truth[:, 0]
                truth[:, 0] = left
    return tensor, {"boxes": truth, "labels": torch.ones(len(truth), dtype=torch.int64)}


def unmatched_indices(truths, predictions):
    edges = [[j for j, prediction in enumerate(predictions) if iou(truth, prediction["bbox"]) >= .5] for truth in truths]
    owners = {}
    def assign(index, seen):
        for prediction in edges[index]:
            if prediction in seen:
                continue
            seen.add(prediction)
            if prediction not in owners or assign(owners[prediction], seen):
                owners[prediction] = index; return True
        return False
    for index in range(len(truths)):
        assign(index, set())
    return [index for index in range(len(predictions)) if index not in owners]


def evaluate(model, rows, root=ROOT):
    frames, negative_frames, confounder_counts = [], [], Counter()
    for row in rows:
        tensor, _ = tensor_example(root, {"row": row})
        predictions = predictions_for(model, tensor, .3)
        truths = [box["bboxPixelsXYXY"] for box in row["boxes"]]
        frame = {"id": row["id"], "group": row["sourceGroupId"], "truth": truths, "predictions": predictions}
        frames.append(frame)
        if row["status"] == "absent":
            negative_frames.append(frame)
        regions, _ = negative_regions(row)
        for index in unmatched_indices(truths, predictions):
            box = predictions[index]["bbox"]
            area = (box[2]-box[0])*(box[3]-box[1])
            kinds = {region["kind"] for region in regions if overlap_area(box, region["bounds"])/max(area, 1e-12) >= .5}
            for kind in kinds:
                confounder_counts[kind] += 1
    return {"overall": score_frames(frames, .3), "explicitNegativeFrames": score_frames(negative_frames, .3),
            "falseAlertsOnExplicitConfounders": dict(confounder_counts), "labelAuthority": "ai-visual-review"}, frames


def selection_key(metrics):
    return (metrics["overall"]["f1"] or 0, -metrics["explicitNegativeFrames"]["falsePositives"], metrics["overall"]["truePositives"])


def ensure_background_proposals(proposals, image_sizes, targets):
    """Keep blank confirmed-negative frames trainable when RPN emits no boxes.

    ROI cross-entropy on an empty tensor is NaN. A whole-image proposal on an
    explicitly empty target supplies background supervision; it is not a new
    annotation and never alters inference or positive-frame target boxes.
    """
    import torch
    result, inserted = list(proposals), 0
    for index, (proposal, (height, width), target) in enumerate(zip(proposals, image_sizes, targets)):
        if len(proposal) == 0 and len(target["boxes"]) == 0:
            result[index] = torch.tensor([[0., 0., float(width), float(height)]], device=proposal.device, dtype=proposal.dtype)
            inserted += 1
    return result, inserted


def train(manifest, inventory, ledger, output, epochs=8, step_budget=1200, learning_rate=.001, seed=8251, root=ROOT):
    import torch
    if not 1 <= epochs <= 12 or not 100 <= step_budget <= 10000 or not .0001 <= learning_rate <= .005:
        raise ValueError("Training bounds exceeded")
    rows, provenance = load_records(manifest, inventory, ledger, root)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    inputs = output/"inputs"; inputs.mkdir()
    for source, name, key in ((manifest, "training-export.json", "manifestSha256"),
                              (inventory, "inventory.json", "inventorySha256"),
                              (ledger, "ledger.json", "ledgerSha256")):
        shutil.copyfile(source, inputs/name)
        if digest(inputs/name) != provenance[key]:
            raise ValueError("Review export changed while freezing training inputs")
    atomic_json(output/"accepted-records.json", rows)
    training = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    examples, rejected, confounders = [], 0, Counter()
    for row in training:
        examples.append({"id": row["id"], "row": row, "positive": bool(row["boxes"])})
        regions, count = negative_regions(row); rejected += count
        for region in regions[:2]:
            examples.append({"id": region["id"], "row": row, "positive": False, "crop": region["bounds"]})
            confounders[region["kind"]] += 1
    effective_epochs = max(1, min(epochs, step_budget//len(examples)))
    steps = effective_epochs*len(examples)
    protocol = {"schema": "mushroom-vision.ai-corpus-fit-protocol.v1", "createdAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeTraining": True, **provenance, "model": MODEL_NAME, "initializationSha256": WEIGHT_SHA256,
        "initialization": "official COCO weights plus new mushroom/background head; not prior failed candidate",
        "epochs": effective_epochs, "requestedEpochs": epochs, "stepBudget": step_budget, "plannedSteps": steps,
        "oneCompletePassMinimum": True, "learningRate": learning_rate, "optimizer": "SGD momentum0.9 weight_decay0.0005, clip10",
        "sampling": "Every eligible training example once per epoch, interleaving positive/negative and source groups",
        "emptyNegativeProposalPolicy": "One full-frame background proposal only during fitting when a confirmed-negative image has no RPN proposals",
        "seed": seed, "scoreThreshold": .3, "iouThreshold": .5, "cpuThreads": 4,
        "trainingFrames": len(training), "trainingSources": len({row["sourceGroupId"] for row in training}),
        "trainingBoxes": sum(len(row["boxes"]) for row in training), "trainingExamples": len(examples),
        "hardNegativeCrops": dict(confounders), "negativeRegionsRejectedForMushroomOverlap": rejected,
        "validationFrames": len(validation), "selection": "maximum AI-validation F1; then fewer negative-frame alerts; then more matched targets; earlier epoch ties" if validation else "fixed final epoch; no validation available",
        "labelAuthority": "ai-visual-review", "humanVerified": False, "activationEligible": False, "testUsedForSelection": False}
    atomic_json(output/"protocol.json", protocol)
    random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)
    model = build_model(pretrained=True)
    empty_background_fallbacks = 0
    def background_hook(module, inputs, result):
        nonlocal empty_background_fallbacks
        if not module.training:
            return result
        proposals, inserted = ensure_background_proposals(result[0], inputs[0].image_sizes, inputs[2])
        empty_background_fallbacks += inserted
        return proposals, result[1]
    model.rpn.register_forward_hook(background_hook)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=learning_rate, momentum=.9, weight_decay=.0005)
    history, best, best_epoch, completed = [], None, None, 0
    started = time.perf_counter()
    for epoch in range(1, effective_epochs+1):
        model.train(); model.backbone.body.eval()
        losses = []
        components = Counter()
        for example in balanced_order(examples, seed+epoch):
            image, target = tensor_example(root, example, augment=True)
            values = model([image], [target]); loss = sum(values.values())
            if not torch.isfinite(loss):
                raise ValueError("Non-finite loss; no candidate can be released")
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10)
            optimizer.step(); completed += 1
            losses.append(float(loss.detach()))
            components.update({key: float(value.detach()) for key, value in values.items()})
            if completed % 20 == 0:
                atomic_json(output/"progress.json", {"completedSteps": completed, "plannedSteps": steps, "epoch": epoch, "epochs": effective_epochs})
                print(json.dumps({"steps": completed, "total": steps, "epoch": epoch, "loss": losses[-1]}), flush=True)
        metrics, _ = evaluate(model, validation, root) if validation else (None, [])
        record = {"epoch": epoch, "meanLoss": sum(losses)/len(losses), "lossComponents": {k: v/len(losses) for k, v in components.items()}, "validation": metrics}
        history.append(record)
        key = selection_key(metrics) if metrics else (epoch,)
        if best is None or key > best:
            best, best_epoch = key, epoch
            checkpoint = {"schema": CHECKPOINT_SCHEMA, "architecture": MODEL_NAME, "state_dict": model.state_dict(),
                "reviewerKind": "ai", "humanVerified": False, "activationEligible": False, "epoch": epoch,
                "scoreThreshold": .3, "trainingManifestSha256": provenance["manifestSha256"], "inventorySha256": provenance["inventorySha256"], "ledgerSha256": provenance["ledgerSha256"]}
            torch.save(checkpoint, output/"candidate.pt")
        atomic_json(output/"history.json", history)
        print(json.dumps(record), flush=True)
    selected = torch.load(output/"candidate.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected["state_dict"])
    metrics, predictions = evaluate(model, validation, root) if validation else (None, [])
    atomic_json(output/"validation-predictions.json", predictions)
    report = {"schema": "mushroom-vision.ai-corpus-fit-report.v1", "status": "completed", "protocol": protocol,
        "optimizerSteps": completed, "selectedEpoch": best_epoch, "history": history, "validation": metrics,
        "emptyNegativeProposalFallbacks": empty_background_fallbacks,
        "checkpointSha256": digest(output/"candidate.pt"), "elapsedSeconds": time.perf_counter()-started,
        "labelAuthority": "ai-visual-review", "humanVerified": False, "activationEligible": False, "productionModelChanged": False,
        "testEvaluationPerformed": False, "limitations": ["AI boxes and negative labels are provisional, not independent human truth.",
            "Validation chooses the research checkpoint only; it does not authorize deployment.",
            "Every eligible training item was visited; uncertain/incomplete/test/rights-restricted items were excluded with counts.",
            "Explicit nonfungal crops intersecting any mushroom truth were refused, preserving fungi attached to trees or stumps."]}
    atomic_json(output/"report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--step-budget", type=int, default=1200)
    args = parser.parse_args()
    result = train(args.manifest, args.inventory, args.ledger, args.output, args.epochs, args.step_budget)
    print(json.dumps({"status": result["status"], "optimizerSteps": result["optimizerSteps"], "selectedEpoch": result["selectedEpoch"], "validation": result["validation"]}), flush=True)
