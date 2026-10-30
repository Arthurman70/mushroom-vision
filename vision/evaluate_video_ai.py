"""Freeze sanitized human video grades and score separate AI proposals against them.

No production data is changed. Incomplete, uncertain and unreviewed frames are
excluded from accuracy denominators; model-empty output never supplies truth.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import urllib.parse
import urllib.request

from adapters.merge_video_collections import read_collection
from vision.assistant_video_pilot import checked_box
from vision.training_pipeline import iou, score_frames
from vision.video_ai_pass import sha

ROOT = Path(__file__).resolve().parents[1]


def write_new(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded)


def sanitized_snapshot(grades, manifest, manifest_hash, collection_id="video-discovery-v1"):
    if grades.get("schema") != "mycoscout.video-grades.v1" or grades.get("collectionId") != collection_id:
        raise ValueError("Human grade collection mismatch")
    by_id = {frame["id"]: frame for frame in manifest["frames"]}
    sources = {source["id"]: source for source in manifest["sources"]}
    records, excluded = [], []
    for frame_id, entry in grades["reviews"].items():
        frame = by_id.get(frame_id)
        if frame is None or entry.get("frameId") != frame_id:
            raise ValueError("Human grade frame is not in collection")
        review = entry.get("review")
        if not review:
            continue
        if review.get("humanVerified") is not True or review.get("complete") is not True or review.get("status") not in {"visible", "absent"}:
            excluded.append({"frameId": frame_id, "reason": "incomplete_or_uncertain_human_review"})
            continue
        boxes = review.get("boxes")
        if not isinstance(boxes, list) or bool(boxes) != (review["status"] == "visible"):
            raise ValueError("Complete human frame has inconsistent boxes/status")
        truth = []
        for box in boxes:
            if box.get("humanVerified") is not True:
                raise ValueError("Nonhuman box in claimed human review")
            truth.append({"id": box["id"], "bbox": checked_box(box["bboxPixelsXYXY"], frame)})
        records.append({"frameId": frame_id, "collection": "main", "sourceId": frame["sourceId"],
                        "sourceGroupId": sources[frame["sourceId"]]["sourceGroupId"], "imageSha256": frame["sha256"],
                        "width": frame["width"], "height": frame["height"], "revision": entry["revision"],
                        "humanVerified": True, "complete": True, "status": review["status"], "truth": truth})
    return {"schema": "mycoscout.human-video-evaluation-snapshot.v1", "createdAt": datetime.now(timezone.utc).isoformat(),
            "sourceManifestSha256": manifest_hash, "collectionId": collection_id,
            "records": records, "excluded": excluded,
            "notice": "Only complete saved human reviews; identities, free text and track names omitted. No labels propagated."}


def matched_pairs(truth, predictions):
    edges = [[j for j, pred in enumerate(predictions) if iou(box["bbox"], pred["bbox"]) >= .5] for box in truth]
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
    for index in range(len(truth)):
        augment(index, set())
    return [{"humanBoxId": truth[t]["id"], "proposalBoxId": predictions[p]["id"],
             "iou": iou(truth[t]["bbox"], predictions[p]["bbox"])} for p, t in sorted(owners.items())]


def evaluate_snapshot(snapshot, proposals, threshold=.25):
    if snapshot.get("schema") != "mycoscout.human-video-evaluation-snapshot.v1" or not 0 <= threshold <= 1:
        raise ValueError("Invalid evaluation snapshot or score threshold")
    matched, pending, failed, scoring = [], [], [], []
    for row in snapshot["records"]:
        path = Path(proposals) / "frames" / ("main--" + row["frameId"] + ".json")
        if not path.exists():
            pending.append(row["frameId"])
            continue
        result = json.loads(path.read_text(encoding="utf-8"))
        if (result.get("frameId") != row["frameId"] or result.get("imageSha256") != row["imageSha256"] or
                result.get("collection") != "main"):
            raise ValueError("Prediction and human review image identity mismatch")
        if result.get("status") != "completed":
            failed.append(row["frameId"])
            continue
        predictions = [box for box in result["boxes"] if box["score"] >= threshold]
        for box in predictions:
            checked_box(box["bbox"], row)
        pairs = matched_pairs(row["truth"], predictions)
        used_truth = {pair["humanBoxId"] for pair in pairs}
        used_predictions = {pair["proposalBoxId"] for pair in pairs}
        matched.append({"frameId": row["frameId"], "collection": "main", "imageSha256": row["imageSha256"],
                        "humanReviewRevision": row["revision"], "humanReviewComplete": True,
                        "humanMatches": pairs, "unmatchedHumanBoxIds": [b["id"] for b in row["truth"] if b["id"] not in used_truth],
                        "unmatchedProposalBoxIds": [b["id"] for b in predictions if b["id"] not in used_predictions],
                        "predictionFileSha256": sha(path.read_bytes())})
        scoring.append({"id": row["frameId"], "group": row["sourceGroupId"],
                        "truth": [box["bbox"] for box in row["truth"]], "predictions": predictions})
    metrics = score_frames(scoring, threshold)
    if sum(len(row["humanMatches"]) for row in matched) != metrics["truePositives"]:
        raise ValueError("Per-frame matches and aggregate metrics disagree")
    return {"schema": "mycoscout.ai-video-human-evaluation.v1", "status": "partial" if pending or failed else "completed",
            "metrics": metrics, "frames": matched, "pendingFrames": pending, "failedFrames": failed,
            "humanSnapshotFrames": len(snapshot["records"]), "excludedReviews": snapshot["excluded"],
            "predictionAuthority": "ai-unverified", "truthAuthority": "human-complete-frame-review",
            "actualDistanceMeasured": False, "automaticActivation": False,
            "limitations": ["Only the available complete human reviews are scored; this small selected sample is not representative of whole videos.",
                "Unreviewed, partial and uncertain frames supply no truth or negative evidence.",
                "Localization IoU matching measures generic mushroom boxes, not species identification or safety."]}


def compare_pilot(snapshot_path, dataset, checkpoint, collection, proposals, output):
    """One fixed-threshold comparison of the inactive pilot and original model."""
    from vision.training_pipeline import MODEL_NAME, build_model, predictions_for, tensor_image
    import torch
    threshold = .3
    snapshot_path, dataset, checkpoint, output = map(Path, (snapshot_path, dataset, checkpoint, output))
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    baseline = evaluate_snapshot(snapshot, proposals, threshold)
    if baseline["status"] != "completed":
        raise ValueError("Wait for original AI predictions for every frozen human evaluation frame")
    root, raw, manifest = read_collection(collection)
    if sha(raw) != snapshot["sourceManifestSha256"]:
        raise ValueError("Collection changed since human snapshot")
    train_manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    train_hash = sha((dataset / "manifest.json").read_bytes())
    if (train_manifest.get("schema") != "mycoscout.assistant-video-training.v1" or
            json.loads((dataset / "READY.json").read_text())["manifestSha256"] != train_hash):
        raise ValueError("Pilot training manifest changed")
    training_groups = set(train_manifest["sourceGroups"])
    if any(row["sourceGroupId"] in training_groups for row in snapshot["records"]):
        raise ValueError("Video evaluation source group overlaps pilot training")
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    checkpoint_hash = sha(checkpoint.read_bytes())
    if (metadata.get("schema") != "mycoscout.assistant-pilot-checkpoint.v1" or
            metadata.get("architecture") != MODEL_NAME or metadata.get("activationAllowed") is not False or
            metadata.get("humanVerified") is not False or metadata.get("datasetManifestSha256") != train_hash):
        raise ValueError("Unsupported pilot checkpoint or training identity mismatch")
    torch.set_num_threads(2)
    model = build_model(pretrained=False)
    model.load_state_dict(metadata["state_dict"], strict=True)
    model.eval()
    frame_map = {frame["id"]: frame for frame in manifest["frames"]}
    output.mkdir(parents=True, exist_ok=False)
    pilot_output = output / "pilot-proposals"
    for row in snapshot["records"]:
        frame = frame_map[row["frameId"]]
        if frame["sha256"] != row["imageSha256"]:
            raise ValueError("Video evaluation image changed")
        boxes = predictions_for(model, tensor_image(root / frame["path"]), threshold)
        result = {"schema": "mycoscout.ai-video-frame.v1", "collection": "main", "frameId": row["frameId"],
                  "sourceId": row["sourceId"], "sourceGroupId": row["sourceGroupId"], "imageSha256": row["imageSha256"],
                  "status": "completed", "model": {"architecture": MODEL_NAME, "checkpointSha256": checkpoint_hash},
                  "humanVerified": False, "trainingAllowed": False, "actuationEnabled": False,
                  "boxes": [{"id": "pilot-" + str(index+1), **box} for index, box in enumerate(boxes)]}
        write_new(pilot_output / "frames" / ("main--" + row["frameId"] + ".json"), result)
    candidate = evaluate_snapshot(snapshot, pilot_output, threshold)
    report = {"schema": "mycoscout.assistant-pilot-comparison.v1", "status": "completed", "scoreThreshold": threshold,
              "snapshotSha256": sha(snapshot_path.read_bytes()), "checkpointSha256": checkpoint_hash,
              "trainingManifestSha256": train_hash, "humanFrames": len(snapshot["records"]),
              "baseline": baseline, "candidate": candidate, "activationEligible": False, "automaticActivation": False,
              "selection": "One predetermined epoch and threshold; this comparison did not select hyperparameters or a checkpoint",
              "notice": "Both models evaluated on identical saved human frames. Tiny selected sample from one heldout creator is not a reliable field-performance estimate."}
    write_new(output / "comparison.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("snapshot")
    freeze.add_argument("--collection", type=Path, default=ROOT / "data/video-discovery-v1")
    freeze.add_argument("--station", default="http://127.0.0.1:8787")
    freeze.add_argument("--output", type=Path, required=True)
    score = sub.add_parser("evaluate")
    score.add_argument("--snapshot", type=Path, required=True)
    score.add_argument("--proposals", type=Path, default=ROOT / "data/ai-video-review-v1")
    score.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare-pilot")
    compare.add_argument("--snapshot", type=Path, required=True)
    compare.add_argument("--dataset", type=Path, required=True)
    compare.add_argument("--checkpoint", type=Path, required=True)
    compare.add_argument("--collection", type=Path, default=ROOT / "data/video-discovery-v1")
    compare.add_argument("--proposals", type=Path, default=ROOT / "data/ai-video-review-v1")
    compare.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "snapshot":
        parsed = urllib.parse.urlsplit(args.station)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Snapshot only reads the local station over loopback HTTP")
        _, raw, manifest = read_collection(args.collection)
        with urllib.request.urlopen(args.station.rstrip("/") + "/api/video-grades", timeout=15) as response:
            grades = json.load(response)
        report = sanitized_snapshot(grades, manifest, sha(raw))
    elif args.command == "evaluate":
        report = evaluate_snapshot(json.loads(args.snapshot.read_text(encoding="utf-8")), args.proposals)
        report["snapshotSha256"] = sha(args.snapshot.read_bytes())
    else:
        report = compare_pilot(args.snapshot, args.dataset, args.checkpoint, args.collection, args.proposals, args.output)
        print(json.dumps({"status": report["status"], "humanFrames": report["humanFrames"],
                          "baseline": report["baseline"]["metrics"], "candidate": report["candidate"]["metrics"],
                          "activationEligible": False}), flush=True)
        return
    write_new(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"records", "frames"}}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
