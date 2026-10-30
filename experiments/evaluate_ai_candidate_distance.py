"""One fixed evaluation of an inactive AI-trained detector on frozen video pixels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.video_distance_retest import digest, read
from experiments.train_ai_review_candidate import load_records
from vision.training_pipeline import MODEL_NAME, build_model, predictions_for, score_frames, tensor_image
from vision.video_ai_pass import atomic_json


def verify_no_overlap(training, evaluation):
    hashes = {row["sha256"] for row in training}
    groups = {row["sourceGroupId"] for row in training}
    if any(row["imageSha256"] in hashes or row["sourceGroupId"] in groups for row in evaluation):
        raise ValueError("AI fit sources overlap frozen test sources")


def run(output, candidate, benchmark, exports):
    import torch
    output, candidate, benchmark = Path(output), Path(candidate), Path(benchmark)
    if output.exists():
        raise ValueError("This fixed evaluation already exists; do not repeatedly tune on test frames")
    previous = read(benchmark/"report.json")
    training_report = read(candidate.parent/"report.json")
    if previous.get("status") != "completed" or previous.get("sample", {}).get("labelAuthority") != "human":
        raise ValueError("Wait for the complete frozen human-box distance benchmark")
    checkpoint_hash = digest(candidate)
    if checkpoint_hash != training_report.get("checkpointSha256") or training_report.get("activationEligible") is not False:
        raise ValueError("AI candidate provenance changed")
    training, provenance = load_records(exports, ROOT/"data/video-discovery-v1", ROOT/"config/ai-review-partitions-v1.json", ROOT/"data/learning")
    if provenance != training_report["sourceExports"]:
        raise ValueError("AI export hashes differ from training")
    verify_no_overlap(training, previous["records"])
    for row in previous["records"]:
        if digest(benchmark/row["imagePath"]) != row["derivedImageSha256"]:
            raise ValueError("Frozen evaluation pixels changed")
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != "mushroom-vision.ai-candidate-checkpoint.v1" or checkpoint.get("architecture") != MODEL_NAME
        or checkpoint.get("reviewerKind") != "ai" or checkpoint.get("humanVerified") is not False
        or checkpoint.get("activationEligible") is not False or checkpoint.get("scoreThreshold") != .3
        or checkpoint.get("sourceExports") != provenance):
        raise ValueError("Unsupported or operationally eligible candidate")
    output.mkdir(parents=True)
    protocol = {"schema": "mushroom-vision.ai-candidate-distance-protocol.v1", "frozenBeforeInference": True,
        "checkpointSha256": checkpoint_hash, "benchmarkReportSha256": digest(benchmark/"report.json"),
        "scoreThreshold": .3, "iouThreshold": .5, "methods": ["raw"], "sourceExports": provenance,
        "trainingLabelAuthority": "ai-visual-review", "evaluationBoxAuthority": "human", "epochs": 1,
        "model": MODEL_NAME, "plannedForwards": len(previous["records"]), "cpuThreads": 4,
        "thresholdSelectionOnTest": False, "additionalTraining": False, "activationEligible": False,
        "trainingFrames": training_report["trainFrames"], "trainingSourceGroups": training_report["trainSources"],
        "trainingBoxes": training_report["trainBoxes"], "trainingAndValidationOverlap": False,
        "actualDistanceM": None}
    atomic_json(output/"protocol.json", protocol)
    torch.set_num_threads(4)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    started = time.perf_counter()
    records = []
    for source in previous["records"]:
        predictions = predictions_for(model, tensor_image(benchmark/source["imagePath"]), .3)
        records.append({"id": source["id"], "condition": source["condition"], "group": source["sourceGroupId"],
            "kind": source["kind"], "truth": [b["bbox"] for b in source["truth"]], "predictions": predictions,
            "synthetic": source["synthetic"], "imageSha256": source["derivedImageSha256"], "actualDistanceM": None})
    summaries = []
    for condition in dict.fromkeys(row["condition"] for row in records):
        selected = [row for row in records if row["condition"] == condition]
        positives = [row for row in selected if row["kind"] == "positive"]
        negatives = [row for row in selected if row["kind"] == "negative"]
        scored, negative = score_frames(positives, .3), score_frames(negatives, .3)
        summaries.append({"condition": condition, "frames": len(positives), "targets": sum(len(r["truth"]) for r in positives),
            "matched": scored["truePositives"], "unmatchedDetections": scored["falsePositives"],
            "negativeFrames": len(negatives), "negativeFalsePositives": negative["falsePositives"]})
    if digest(candidate) != checkpoint_hash:
        raise ValueError("Candidate changed during evaluation")
    report = {"schema": "mushroom-vision.ai-candidate-distance-report.v1", "status": "completed", "generatedAt": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol, "rows": summaries, "records": records, "elapsedSeconds": time.perf_counter()-started,
        "modelActivation": False, "productionModelChanged": False,
        "limitations": ["Candidate was trained for one pass on only six AI-reviewed frames; training boxes are not human verified.",
            "This is a separate Faster R-CNN candidate at fixed 0.3, not the GroundingDINO 0.25 experiment. Do not pool these scores.",
            "Frozen human presence boxes support annotation matching; they do not verify a species.",
            "Three correlated positive source frames and one negative from one creator are not independent population accuracy.",
            "Scaled pixel footprints are synthetic and do not measure camera range in meters.",
            "No threshold selection, further fitting or activation is performed from these evaluation results."]}
    atomic_json(output/"report.json", report)
    print(json.dumps({"rows": summaries, "elapsedSeconds": report["elapsedSeconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT/"experiments/results/ai-review-candidate-v1/ai-candidate.pt")
    parser.add_argument("--benchmark", type=Path, default=ROOT/"experiments/results/video-distance-v2")
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/results/ai-candidate-distance-v1")
    parser.add_argument("--export", type=Path, action="append", dest="exports")
    args = parser.parse_args()
    exports = args.exports or [ROOT/"data/ai-visual-review-v1/train-export.json", ROOT/"data/ai-visual-review-v1/validation-export.json"]
    run(args.output, args.candidate, args.benchmark, exports)
