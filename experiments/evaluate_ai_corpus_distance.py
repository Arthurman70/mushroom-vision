"""One fixed regression evaluation of the inactive corpus-trained detector."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
import time
import json

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_ai_corpus_candidate import CHECKPOINT_SCHEMA, digest, read, load_records
from vision.training_pipeline import MODEL_NAME, build_model, predictions_for, score_frames, tensor_image
from vision.video_ai_pass import atomic_json


def verify_no_overlap(rows, sources, inventory):
    fit = [row for row in rows if row["split"] == "train"]
    used_keys = {key for row in fit for key in row["groupKeys"]}
    used_groups = {row["sourceGroupId"] for row in fit}
    used_hashes = {row["imageSha256"] for row in fit}
    indexed = {row["sha256"]: row for row in inventory["records"]}
    for source in sources:
        original = indexed.get(source["imageSha256"])
        if original is None or original["split"] != "test":
            raise ValueError("Evaluation original is not in the frozen test inventory")
        if (source["imageSha256"] in used_hashes or source["sourceGroupId"] in used_groups
            or used_keys.intersection(original["groupKeys"])):
            raise ValueError("Training source or decoded pixels overlap the frozen test")


def summarize(records):
    result = []
    for condition in dict.fromkeys(row["condition"] for row in records):
        selected = [row for row in records if row["condition"] == condition]
        positive = [row for row in selected if row["kind"] == "positive"]
        negative = [row for row in selected if row["kind"] == "negative"]
        scored, neg = score_frames(positive, .3), score_frames(negative, .3)
        targets = sum(len(row["truth"]) for row in positive)
        result.append({"condition": condition, "frames": len(positive), "targets": targets,
            "matched": scored["truePositives"], "missed": scored["falseNegatives"],
            "recall": scored["truePositives"]/targets if targets else None,
            "unmatchedDetections": scored["falsePositives"], "negativeFrames": len(negative),
            "negativeFalsePositives": neg["falsePositives"]})
    return result


def evidence_sheet(output, benchmark, records):
    chosen = [row for row in records if row["condition"] == "native"]
    chosen += [row for row in records if row["condition"] == "16px"][:2]
    sheet = Image.new("RGB", (1280, 3*408), "#171d20")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(chosen):
        image = Image.open(benchmark/row["imagePath"]).convert("RGB")
        marks = ImageDraw.Draw(image)
        for box in row["truth"]:
            marks.rectangle(box, outline="#16e49c", width=3)
        for prediction in row["predictions"]:
            marks.rectangle(prediction["bbox"], outline="#ff6159", width=3)
        image.thumbnail((630, 356))
        x, y = (index % 2)*640, (index//2)*408
        sheet.paste(image, (x, y+44))
        draw.text((x+8, y+6), row["id"], fill="white")
        draw.text((x+8, y+23), f"Green = human box; red = candidate alert ({len(row['predictions'])})", fill="white")
    sheet.save(output/"failure-examples.jpg", quality=92)


def run(output, candidate, benchmark, previous_candidate):
    import torch
    output, candidate, benchmark = Path(output), Path(candidate), Path(benchmark)
    if output.exists():
        raise ValueError("Fixed evaluation already exists; do not tune or repeatedly inspect test thresholds")
    source = read(benchmark/"report.json")
    fit = read(candidate.parent/"report.json")
    inputs = candidate.parent/"inputs"
    rows, provenance = load_records(inputs/"training-export.json", inputs/"inventory.json", inputs/"ledger.json")
    if source.get("status") != "completed" or source.get("sample", {}).get("labelAuthority") != "human":
        raise ValueError("Complete human presence-box benchmark required")
    if (fit.get("status") != "completed" or fit.get("activationEligible") is not False
        or fit.get("labelAuthority") != "ai-visual-review" or fit.get("productionModelChanged") is not False
        or any(fit["protocol"].get(key) != value for key, value in provenance.items())):
        raise ValueError("Candidate training provenance changed")
    candidate_hash = digest(candidate)
    if candidate_hash != fit["checkpointSha256"]:
        raise ValueError("Candidate checkpoint changed")
    verify_no_overlap(rows, source["records"], read(inputs/"inventory.json"))
    for record in source["records"]:
        path = (benchmark/record["imagePath"]).resolve()
        if not path.is_relative_to(benchmark.resolve()) or digest(path) != record["derivedImageSha256"]:
            raise ValueError("Frozen benchmark pixels changed")
    checkpoint = torch.load(candidate, map_location="cpu", weights_only=True)
    if (checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("architecture") != MODEL_NAME
        or checkpoint.get("reviewerKind") != "ai" or checkpoint.get("humanVerified") is not False
        or checkpoint.get("activationEligible") is not False or checkpoint.get("scoreThreshold") != .3
        or checkpoint.get("trainingManifestSha256") != provenance["manifestSha256"]
        or checkpoint.get("inventorySha256") != provenance["inventorySha256"]
        or checkpoint.get("ledgerSha256") != provenance["ledgerSha256"]
        or checkpoint.get("epoch") != fit["selectedEpoch"]):
        raise ValueError("Candidate provenance or research-only schema is invalid")
    old = read(previous_candidate)
    if (old.get("schema") != "mushroom-vision.ai-candidate-distance-report.v1"
        or old["protocol"]["benchmarkReportSha256"] != digest(benchmark/"report.json")
        or old["protocol"]["scoreThreshold"] != .3 or old["protocol"]["iouThreshold"] != .5):
        raise ValueError("Prior candidate used different test or threshold")
    output.mkdir(parents=True)
    protocol = {"schema": "mushroom-vision.ai-corpus-distance-protocol.v1", "frozenBeforeInference": True,
        "checkpointSha256": candidate_hash, "benchmarkReportSha256": digest(benchmark/"report.json"),
        "previousCandidateReportSha256": digest(previous_candidate), "scoreThreshold": .3, "iouThreshold": .5,
        "trainingLabelAuthority": "ai-visual-review", "evaluationBoxAuthority": "human",
        "thresholdSelectionOnTest": False, "additionalTraining": False, "activationEligible": False,
        "model": MODEL_NAME, "epochs": fit["protocol"]["epochs"], "selectedEpoch": fit["selectedEpoch"],
        "trainingFrames": fit["protocol"]["trainingFrames"], "trainingSourceGroups": fit["protocol"]["trainingSources"],
        "trainingBoxes": fit["protocol"]["trainingBoxes"], "optimizerSteps": fit["optimizerSteps"],
        "hardNegativeCrops": fit["protocol"]["hardNegativeCrops"], "plannedForwards": len(source["records"]),
        "methods": ["raw"], "cpuThreads": 4, "actualDistanceM": None,
        "evaluationPurpose": "Previously inspected frozen regression benchmark; not fresh field validation", **provenance}
    atomic_json(output/"protocol.json", protocol)
    torch.set_num_threads(4)
    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    started, records = time.perf_counter(), []
    for row in source["records"]:
        predictions = predictions_for(model, tensor_image(benchmark/row["imagePath"]), .3)
        records.append({"id": row["id"], "condition": row["condition"], "group": row["sourceGroupId"],
            "kind": row["kind"], "truth": [box["bbox"] for box in row["truth"]], "predictions": predictions,
            "imagePath": row["imagePath"], "imageSha256": row["derivedImageSha256"], "synthetic": row["synthetic"],
            "actualDistanceM": None})
    if digest(candidate) != candidate_hash:
        raise ValueError("Checkpoint changed during evaluation")
    report = {"schema": "mushroom-vision.ai-corpus-distance-report.v1", "status": "completed",
        "generatedAt": datetime.now(timezone.utc).isoformat(), "protocol": protocol,
        "rows": summarize(records), "previousCandidateRows": old["rows"], "records": records,
        "elapsedSeconds": time.perf_counter()-started, "modelActivation": False, "productionModelChanged": False,
        "limitations": ["Training labels are AI observations, not human verification.",
            "The same frozen test was previously viewed, so this is a regression diagnostic rather than independent validation.",
            "Only three correlated positive frames and one negative original from one creator are represented.",
            "Resized pixel widths do not measure physical camera range; actual distance remains unknown.",
            "The prior six-frame candidate uses the same architecture, thresholds and saved pixels; compare its counts separately from GroundingDINO.",
            "Model selection used only AI validation frames, whose only complete positive frame is an editorial chanterelle inset, not woodland field footage.",
            "No test-guided fitting, threshold changes, species claims or production activation follow this evaluation."]}
    evidence_sheet(output, benchmark, records)
    atomic_json(output/"report.json", report)
    print(json.dumps({"rows": report["rows"], "elapsedSeconds": report["elapsedSeconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT/"experiments/results/ai-corpus-candidate-v2/candidate.pt")
    parser.add_argument("--benchmark", type=Path, default=ROOT/"experiments/results/video-distance-v2")
    parser.add_argument("--previous-candidate", type=Path, default=ROOT/"experiments/results/ai-candidate-distance-v1/report.json")
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/results/ai-corpus-distance-v1")
    args = parser.parse_args()
    run(args.output, args.candidate, args.benchmark, args.previous_candidate)
