"""Separate experimental training on visually audited AI labels; never activation.

This does not bypass the human-reviewed training service. Assistant audits remain
weak labels, and frozen human validation/test groups cannot become training data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

from PIL import Image, ImageOps

from adapters.export_reviewed_detection_dataset import canonical, digest, inside, write_json
from adapters.merge_video_collections import read_collection
from vision.training_pipeline import build_model, evaluate, read_dataset, training_example, WEIGHT_SHA256

ROOT = Path(__file__).resolve().parents[1]


def signatures(image):
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    small = rgb.resize((16, 16))
    values = list(small.convert("L").getdata())
    average = sum(values) / len(values)
    return {"pixelSha256": digest(canonical(list(rgb.size)) + rgb.tobytes()),
            "averageHash": format(sum((value >= average) << index for index, value in enumerate(values)), "064x"),
            "colorMean": [round(sum(small.getdata(band)) / 256, 3) for band in range(3)]}


def blocked_by_holdout(group_keys, original_sha, signature, holdouts):
    for held in holdouts:
        if set(group_keys).intersection(held["groupKeys"]):
            return "frozen_source_group"
        if original_sha == held["originalSha256"] or signature["pixelSha256"] == held["pixelSha256"]:
            return "frozen_identical_image"
        if ((int(signature["averageHash"], 16) ^ int(held["averageHash"], 16)).bit_count() <= 6 and
                max(abs(a-b) for a, b in zip(signature["colorMean"], held["colorMean"])) <= 8):
            return "frozen_near_duplicate"
    return None


def collect_holdouts(learning_directory):
    rows, versions = [], []
    for manifest_file in sorted(Path(learning_directory).glob("*/manifest.json")):
        manifest = read_dataset(manifest_file.parent, require_ready=False)
        versions.append({"datasetId": manifest["datasetId"], "manifestSha256": digest(manifest_file.read_bytes())})
        rows.extend(row for row in manifest["records"] if row["split"] in {"test", "validation"})
    return rows, versions


def validate_audit_identity(review, frame, proposal):
    if (review.get("collection") != "main" or review.get("frameId") != frame["id"] or
            review.get("imageSha256") != frame["sha256"]):
        raise ValueError("Audit/frame/proposal identity mismatch")
    if proposal is not None and (proposal.get("imageSha256") != frame["sha256"] or
            proposal.get("frameId") != frame["id"] or proposal.get("collection") != "main" or
            proposal.get("status") != "completed"):
        raise ValueError("Audit/frame/proposal identity mismatch")
    if proposal is None and (review.get("acceptedBoxIds") or review.get("rejectedBoxIds")):
        raise ValueError("Cannot accept or reject a model box without its completed proposal")


def checked_box(coords, frame):
    if (not isinstance(coords, list) or len(coords) != 4 or
            any(type(value) not in {int, float} or not math.isfinite(value) for value in coords) or
            not 0 <= coords[0] < coords[2] <= frame["width"] or
            not 0 <= coords[1] < coords[3] <= frame["height"]):
        raise ValueError("Audited box is outside native image or not finite")
    return coords


def validated_audit(review, frame, proposal):
    if (review.get("reviewer") != "assistant-visual-audit" or review.get("complete") is not True or
            review.get("humanVerified") is not False or review.get("status") not in {"visible", "absent"}):
        raise ValueError("Pilot requires explicit complete assistant audits, never claimed human verification")
    validate_audit_identity(review, frame, proposal)
    accepted = review.get("acceptedBoxIds")
    rejected = review.get("rejectedBoxIds", [])
    corrected = review.get("correctedBoxes", [])
    if (not isinstance(accepted, list) or any(not isinstance(value, str) for value in accepted) or
            len(accepted) != len(set(accepted)) or not isinstance(rejected, list) or
            any(not isinstance(value, str) for value in rejected) or len(rejected) != len(set(rejected)) or
            set(accepted).intersection(rejected) or not isinstance(corrected, list) or len(corrected) > 100):
        raise ValueError("Audit box IDs must be a unique list")
    if bool(accepted or corrected) != (review["status"] == "visible"):
        raise ValueError("Visible audits need boxes; explicit absent audits must have none")
    by_id = {box["id"]: box for box in proposal["boxes"]} if proposal is not None else {}
    if any(box_id not in by_id for box_id in rejected):
        raise ValueError("Audit rejected a missing proposal")
    boxes = []
    for box_id in accepted:
        if box_id not in by_id:
            raise ValueError("Audit accepted a missing proposal")
        box = by_id[box_id]
        coords = checked_box(box["bbox"], frame)
        boxes.append({"id": box_id, "bboxPixelsXYXY": coords, "teacherScore": box["score"],
                      "labelOrigin": "accepted_model_proposal", "labelAuthority": "assistant-reviewed-ai", "humanVerified": False})
    for index, correction in enumerate(corrected):
        if not isinstance(correction, dict):
            raise ValueError("Corrected boxes must be objects with native bbox coordinates")
        coords = checked_box(correction.get("bbox"), frame)
        boxes.append({"id": "corrected-" + str(index + 1), "bboxPixelsXYXY": coords, "teacherScore": None,
                      "labelOrigin": "assistant_corrected_coordinates", "labelAuthority": "assistant-reviewed-ai", "humanVerified": False})
    if len({tuple(box["bboxPixelsXYXY"]) for box in boxes}) != len(boxes):
        raise ValueError("Repeated audited coordinates")
    return boxes


def prepare(audit_path, proposals, collection, learning_directory, output):
    root, raw, data = read_collection(collection)
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8-sig"))
    if audit.get("schema") != "mycoscout.assistant-audit.v1" or not isinstance(audit.get("reviews"), list):
        raise ValueError("Expected assistant audit manifest")
    if not 1 <= len(audit["reviews"]) <= 300:
        raise ValueError("Provide between 1 and 300 independently audited frames")
    sources = {source["id"]: source for source in data["sources"]}
    frames = {frame["id"]: frame for frame in data["frames"]}
    holdouts, versions = collect_holdouts(learning_directory)
    rows, excluded, review_evidence, seen = [], [], [], set()
    for review in audit["reviews"]:
        frame = frames.get(review.get("frameId"))
        if not frame or frame["id"] in seen:
            raise ValueError("Missing or repeated audited frame")
        seen.add(frame["id"])
        source = sources[frame["sourceId"]]
        if source.get("licenseAllowsTraining") is not True or source.get("evaluationOnly") is not False or frame.get("evaluationOnly") is True:
            excluded.append({"frameId": frame["id"], "reason": "source_permission_or_quarantine"})
            continue
        proposal_path = Path(proposals) / "frames" / ("main--" + frame["id"] + ".json")
        proposal = json.loads(proposal_path.read_text(encoding="utf-8")) if proposal_path.exists() else None
        validate_audit_identity(review, frame, proposal)
        if review.get("reviewer") != "assistant-visual-audit" or review.get("humanVerified") is not False:
            raise ValueError("Audit authority must remain explicitly nonhuman")
        review_evidence.append(review)
        if review.get("complete") is not True or review.get("status") not in {"visible", "absent"}:
            excluded.append({"frameId": frame["id"], "reason": "incomplete_or_uncertain_audit"})
            continue
        boxes = validated_audit(review, frame, proposal)
        with Image.open(root / frame["path"]) as image:
            signature = signatures(image)
        keys = ["source:" + source["sourceGroupId"], "video:" + source.get("videoHash", source.get("sha256"))]
        blocked = blocked_by_holdout(keys, frame["sha256"], signature, holdouts)
        if blocked:
            excluded.append({"frameId": frame["id"], "reason": blocked})
            continue
        rows.append({"id": frame["id"], "sourceId": source["id"], "groupKeys": keys,
                     "leakageGroup": source["sourceGroupId"], "width": frame["width"], "height": frame["height"],
                     "originalSha256": frame["sha256"], "imageSha256": frame["sha256"], **signature,
                     "imagePath": "images/" + frame["sha256"] + ".jpg", "boxes": boxes, "status": review["status"],
                     "audit": review, "proposalSha256": digest(proposal_path.read_bytes()) if proposal is not None else None,
                     "proposalStatus": "completed" if proposal is not None else "unavailable_at_audit",
                     "sourceCredits": source, "labelAuthority": "assistant-reviewed-ai", "humanVerified": False,
                     "split": "train", "originalPath": frame["path"]})
    if not any(row["boxes"] for row in rows):
        raise ValueError("No eligible, nonheldout assistant-audited positive frames")
    if len({row["pixelSha256"] for row in rows}) != len(rows):
        raise ValueError("Repeated training pixels; retain one audited record per image")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    for row in rows:
        raw_image = (root / row.pop("originalPath")).read_bytes()
        if digest(raw_image) != row["imageSha256"]:
            raise ValueError("Source image changed during pilot preparation")
        with (output / row["imagePath"]).open("xb") as stream:
            stream.write(raw_image)
    manifest = {"schema": "mycoscout.assistant-video-training.v1", "createdAt": datetime.now(timezone.utc).isoformat(),
                "labelAuthority": "assistant-reviewed-ai", "humanVerified": False, "automaticActivation": False,
                "auditSha256": digest(Path(audit_path).read_bytes()), "sourceManifestSha256": digest(raw),
                "frozenHumanDatasets": versions, "records": rows, "excluded": excluded, "reviewEvidence": review_evidence,
                "images": len(rows), "positiveImages": sum(bool(row["boxes"]) for row in rows),
                "explicitNegativeImages": sum(not row["boxes"] for row in rows),
                "sourceGroups": sorted({row["leakageGroup"] for row in rows}),
                "notice": "Weak assistant labels are isolated from the human-reviewed training pipeline. No model-empty negatives are used."}
    write_json(output / "manifest.json", manifest)
    write_json(output / "READY.json", {"manifestSha256": digest((output / "manifest.json").read_bytes())})
    return manifest


def train(dataset, human_test, learning_directory, output, seed=291026):
    dataset, output = Path(dataset), Path(output)
    manifest_file = dataset / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if (manifest.get("schema") != "mycoscout.assistant-video-training.v1" or
            json.loads((dataset / "READY.json").read_text())["manifestSha256"] != digest(manifest_file.read_bytes())):
        raise ValueError("Pilot dataset integrity failed")
    holdouts, _ = collect_holdouts(learning_directory)
    for row in manifest["records"]:
        if (row.get("humanVerified") is not False or row.get("labelAuthority") != "assistant-reviewed-ai" or
                digest(inside(dataset, row["imagePath"]).read_bytes()) != row["imageSha256"] or
                blocked_by_holdout(row["groupKeys"], row["originalSha256"], row, holdouts)):
            raise ValueError("Training inputs changed or now intersect a frozen human holdout")
    human = read_dataset(human_test, require_ready=False)
    test_rows = [row for row in human["records"] if row["split"] == "test"]
    if not test_rows:
        raise ValueError("A frozen human test set is required for this pilot")
    output.mkdir(parents=True, exist_ok=False)
    import torch
    torch.set_num_threads(2)
    random.seed(seed); torch.manual_seed(seed)
    model = build_model(pretrained=True)
    initial, _ = evaluate(model, human_test, test_rows, .3)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.005, momentum=.9, weight_decay=.0005)
    order = list(manifest["records"])
    random.shuffle(order)
    before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    losses, started = [], time.time()
    for index, row in enumerate(order):
        model.train(); model.backbone.body.eval()
        image, target = training_example(dataset, row, augment=True)
        loss = sum(model([image], [target]).values())
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite pilot loss")
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10)
        optimizer.step(); losses.append(float(loss.detach()))
        print(json.dumps({"phase": "assistant-pilot-train", "step": index+1, "steps": len(order), "loss": losses[-1]}), flush=True)
    checkpoint = {"schema": "mycoscout.assistant-pilot-checkpoint.v1", "architecture": "fasterrcnn_mobilenet_v3_large_320_fpn",
                  "state_dict": model.state_dict(), "labelAuthority": "assistant-reviewed-ai", "humanVerified": False,
                  "epochs": 1, "scoreThreshold": .3, "activationAllowed": False,
                  "datasetManifestSha256": digest(manifest_file.read_bytes())}
    torch.save(checkpoint, output / "candidate.pt")
    restored = torch.load(output / "candidate.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(restored["state_dict"], strict=True)
    metrics, predictions = evaluate(model, human_test, test_rows, .3)
    report = {"schema": "mycoscout.assistant-pilot-model-card.v1", "status": "completed", "epochs": 1,
              "labelAuthority": "assistant-reviewed-ai", "humanVerified": False, "trainingImages": len(order),
              "positiveImages": manifest["positiveImages"], "explicitNegativeImages": manifest["explicitNegativeImages"],
              "trainingSourceGroups": manifest["sourceGroups"], "optimizerSteps": len(order), "meanLoss": sum(losses)/len(losses),
              "maxClassifierWeightChange": float((model.roi_heads.box_predictor.cls_score.weight.detach()-before).abs().max()),
              "checkpointSha256": digest((output / "candidate.pt").read_bytes()), "pretrainedSha256": WEIGHT_SHA256,
              "checkpointReloaded": True, "scoreThreshold": .3, "learningRate": .005, "seed": seed,
              "selection": "Fixed one epoch and parameters; no test-based checkpoint selection",
              "initialRandomMushroomHeadHumanTest": initial, "humanTest": metrics,
              "humanTestDatasetId": human["datasetId"], "elapsedSeconds": time.time()-started,
              "activationEligible": False, "automaticActivation": False, "operationallyValidated": False,
              "limitations": ["Assistant-reviewed AI boxes remain imperfect weak labels, not human ground truth.",
                  "Unmarked regions of weakly annotated positive images can contain missed mushrooms.",
                  "The tiny reused human holdout does not establish woodland, species or distance accuracy.",
                  "Initial-head comparison is not a GroundingDINO comparison or evidence of deployment readiness."]}
    write_json(output / "model-card.json", report)
    write_json(output / "human-test-predictions.json", predictions)
    write_json(output / "COMPLETE.json", {"checkpointSha256": report["checkpointSha256"], "activationEligible": False})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--audit", type=Path, required=True)
    prep.add_argument("--proposals", type=Path, default=ROOT / "data/ai-video-review-v1")
    prep.add_argument("--collection", type=Path, default=ROOT / "data/video-discovery-v1")
    prep.add_argument("--learning-datasets", type=Path, default=ROOT / "data/learning/datasets")
    prep.add_argument("--output", type=Path, required=True)
    fit = sub.add_parser("train")
    fit.add_argument("--dataset", type=Path, required=True)
    fit.add_argument("--human-test", type=Path, required=True)
    fit.add_argument("--learning-datasets", type=Path, default=ROOT / "data/learning/datasets")
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        report = prepare(args.audit, args.proposals, args.collection, args.learning_datasets, args.output)
    else:
        report = train(args.dataset, args.human_test, args.learning_datasets, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in {"records", "reviewEvidence"}}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
