"""One immutable regression pass for an inactive, weight-adjusted candidate.

The 26 derived views are four previously inspected originals, not 26 independent
observations. Training/validation diagnostics and saved DINO results stay
separate. This script never trains, selects thresholds, or activates weights.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
import sys
import time

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.evaluate_ai_corpus_distance import verify_no_overlap
from experiments.train_ai_corpus_candidate import digest, load_records, read, valid_box
from vision.compare_candidate import POLICY
from vision.training_pipeline import MODEL_NAME, predictions_for, score_frames, tensor_image

CHECKPOINT_SCHEMA = "mushroom-vision.weight-adjusted-checkpoint.v1"
SCHEMA = "mushroom-vision.weight-adjustment-evaluation.v1"
BENCHMARK_SHA256 = "280594aafa1d6bd01656c6389f46568ceecd0c681333c70456508461325c7dc7"
CONDITIONS = {"native": 4, "64px": 3, "32px": 3, "16px": 3, "6p9px": 4,
              "3p45px": 3, "1p38px": 3, "0p69px": 3}
POSITIVE_IDS = ["youtube-m9aGEh33w0I-043", "youtube-m9aGEh33w0I-005", "youtube-m9aGEh33w0I-022"]
NEGATIVE_ID = "youtube-m9aGEh33w0I-040"
GROUND_REGION = [0., .35, 1., 1.]


def write_new(path, value):
    """Never replace an existing protocol, claim, or completed result."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def validate_benchmark(source):
    if (source.get("schema") != "mycoscout.video-distance-report.v1"
        or source.get("status") != "completed"
        or source.get("sample") != {"positiveFrames": 3, "negativeFrames": 1, "targets": 3,
                                    "sourceVideos": 1, "sourceGroups": 1, "labelAuthority": "human"}):
        raise ValueError("The fixed human regression benchmark is required")
    records = source.get("records", [])
    if (len(records) != 26 or len({r["id"] for r in records}) != 26
        or Counter(r["condition"] for r in records) != Counter(CONDITIONS)):
        raise ValueError("Exactly the 26 frozen regression records are required")
    expected = {(frame, condition) for condition in CONDITIONS for frame in POSITIVE_IDS}
    expected |= {(NEGATIVE_ID, condition) for condition in ("native", "6p9px")}
    if {(r["frameId"], r["condition"]) for r in records} != expected:
        raise ValueError("Frozen regression identities or conditions changed")
    for row in records:
        positive = row["frameId"] in POSITIVE_IDS
        if (row.get("humanVerifiedTruth") is not True or row.get("actualDistanceM") is not None
            or row.get("kind") != ("positive" if positive else "negative")
            or len(row.get("truth", [])) != int(positive)
            or row.get("synthetic") is not (row["condition"] != "native")
            or row["id"] != row["frameId"]+"--"+row["condition"]):
            raise ValueError("Frozen reference authority, identity, or distance claim changed")
    return records


def validate_metadata(metadata, fit, provenance):
    from experiments.weight_recipe import validate_config
    config = metadata.get("modelConfig")
    validate_config(config)
    if (metadata.get("schema") != CHECKPOINT_SCHEMA or metadata.get("architecture") != MODEL_NAME
        or metadata.get("reviewerKind") != "ai" or metadata.get("humanVerified") is not False
        or metadata.get("activationEligible") is not False or metadata.get("scoreThreshold") != .3
        or metadata.get("epoch") != fit.get("selectedEpoch")
        or metadata.get("trainingManifestSha256") != provenance["manifestSha256"]
        or metadata.get("inventorySha256") != provenance["inventorySha256"]
        or metadata.get("ledgerSha256") != provenance["ledgerSha256"]
        or fit.get("protocol", {}).get("modelConfig") != config):
        raise ValueError("Checkpoint authority, threshold, selected epoch, or model configuration changed")
    if (fit.get("status") != "completed" or fit.get("activationEligible") is not False
        or fit.get("productionModelChanged") is not False or fit.get("labelAuthority") != "ai-visual-review"
        or any(fit.get("protocol", {}).get(key) != value for key, value in provenance.items())):
        raise ValueError("Training report provenance or authority changed")
    return config


def source_support(records):
    """Count native independent units only; synthetic scales add no support."""
    native = [row for row in records if row["condition"] == "native"]
    identities = [row.get("frameId", row["id"].split("--")[0]) for row in native]
    if len(set(identities)) != len(identities):
        raise ValueError("Repeated native original cannot increase source support")
    positive = [row for row in native if row["truth"]]
    small = [(row, box) for row in positive for box in row["truth"]
             if box[2]-box[0] < POLICY["smallWidthBelowPixels"]]
    return {"nativeOriginals": len(native), "targets": sum(len(r["truth"]) for r in positive),
        "targetGroups": len({r["group"] for r in positive}), "sourceGroups": len({r["group"] for r in native}),
        "smallTargets": len(small), "smallTargetGroups": len({r["group"] for r, _ in small}),
        "negativeFrames": sum(r["kind"] == "negative" for r in native),
        "countingPolicy": "Distinct native originals only; resized views never increase independent support"}


def rows_for(records, threshold):
    rows = []
    for condition in CONDITIONS:
        chosen = [r for r in records if r["condition"] == condition]
        positive = [r for r in chosen if r["kind"] == "positive"]
        negative = [r for r in chosen if r["kind"] == "negative"]
        p, n = score_frames(positive, threshold), score_frames(negative, threshold)
        rows.append({"condition": condition, "frames": len(positive), "targets": p["truePositives"]+p["falseNegatives"],
            "matched": p["truePositives"], "missed": p["falseNegatives"], "recall": p["recall"],
            "unmatchedPositiveAlerts": p["falsePositives"], "negativeFrames": len(negative),
            "negativeFalseAlerts": n["falsePositives"], "scoreThreshold": threshold, "iouThreshold": .5})
    return rows


def native_metrics(records, threshold):
    chosen = [r for r in records if r["condition"] == "native"]
    result = score_frames(chosen, threshold)
    neg = score_frames([r for r in chosen if r["kind"] == "negative"], threshold)
    small = [v for k, v in result["pixelWidthBins"].items() if k != "64plus"]
    total, found = sum(v["total"] for v in small), sum(v["matched"] for v in small)
    return {**result, "negativeFalsePositives": neg["falsePositives"],
            "smallTargets": {"matched": found, "total": total, "recall": found/total if total else None}}


def gate_for(candidate, baseline, support):
    missing = []
    for key, policy in (("targets", "minimumTargets"), ("targetGroups", "minimumTargetGroups"),
                       ("smallTargets", "minimumSmallTargets"), ("smallTargetGroups", "minimumSmallTargetGroups"),
                       ("negativeFrames", "minimumNegativeFrames")):
        if support[key] < POLICY[policy]:
            missing.append(f"{key}: {support[key]} observed; at least {POLICY[policy]} required")
    recall_gain = candidate["recall"]-baseline["recall"] if None not in (candidate["recall"], baseline["recall"]) else None
    cs, bs = candidate["smallTargets"]["recall"], baseline["smallTargets"]["recall"]
    small_gain = cs-bs if None not in (cs, bs) else None
    performance = []
    if recall_gain is None or recall_gain+1e-12 < POLICY["minimumRecallGain"]:
        performance.append("Native recall gain is below five percentage points or undefined")
    if small_gain is None or small_gain+1e-12 < POLICY["minimumSmallRecallGain"]:
        performance.append("Native small-target recall gain is below five percentage points or undefined")
    if candidate["falsePositives"] > baseline["falsePositives"]:
        performance.append("More unmatched native alerts than baseline")
    if candidate["negativeFalsePositives"] > baseline["negativeFalsePositives"]:
        performance.append("More alerts on the explicit native negative than baseline")
    return {"decision": "insufficient_evidence" if missing else "candidate_not_better" if performance else "candidate_better",
        "support": support, "supportReasons": missing, "performanceReasons": performance,
        "recallGain": recall_gain, "smallRecallGain": small_gain, "evidenceSufficient": not missing,
        "activationAllowed": False, "automaticActivation": False}


def baseline_records(saved, frozen, previous_candidate=False):
    rows = saved.get("records", [])
    if len(rows) != len(frozen) or len({r["id"] for r in rows}) != len(frozen):
        raise ValueError("Baseline does not contain exactly the frozen records")
    indexed = {r["id"]: r for r in rows}
    result = []
    for source in frozen:
        row = indexed.get(source["id"])
        truth = [box["bbox"] for box in source["truth"]]
        candidate_truth = row.get("truth") if row and previous_candidate else [b["bbox"] for b in row.get("truth", [])] if row else None
        if (row is None or candidate_truth != truth or row.get("condition") != source["condition"]
            or row.get("kind") != source["kind"]
            or row.get("group" if previous_candidate else "sourceGroupId") != source["sourceGroupId"]
            or row.get("imageSha256" if previous_candidate else "derivedImageSha256") != source["derivedImageSha256"]):
            raise ValueError("Baseline identity, truth, source group, or image hash changed")
        predictions = row["predictions"] if previous_candidate else row["predictions"]["raw"]
        result.append({"id": source["id"], "frameId": source["frameId"], "condition": source["condition"],
            "group": source["sourceGroupId"], "kind": source["kind"], "truth": truth, "predictions": predictions})
    return result


def checked_predictions(predictions, width, height):
    for row in predictions:
        score = row.get("score")
        if type(score) not in (int, float) or not math.isfinite(score) or not .3 <= score <= 1:
            raise ValueError("Prediction violates frozen score threshold")
        valid_box(row.get("bbox"), width, height)
    return predictions


def native_records(records):
    selected = [row for row in records if row["condition"] == "native"]
    if (len(selected) != 4 or {r["frameId"] for r in selected} != set(POSITIVE_IDS+[NEGATIVE_ID])
        or any(row.get("synthetic") is not False for row in selected)):
        raise ValueError("Ground scanning requires exactly the four native originals")
    return selected


def evaluation_plan(frozen):
    selected = native_records(frozen)
    if len(frozen) != 26:
        raise ValueError("The fixed raw evaluation requires 26 views")
    return {"recordOrder": [r["id"] for r in frozen], "groundRecordOrder": [r["id"] for r in selected],
        "plannedRawForwards": 26, "plannedGroundForwards": 20, "plannedForwards": 46,
        "methods": ["raw", "native-ground-tiles"],
        "groundSettings": {"profile": "ground-tiles", "tileRegion": GROUND_REGION,
            "colorSensitive": False, "fullFrameContext": True, "passesPerNativeFrame": 5,
            "overlapFraction": .2, "nmsIou": .5, "scoreThreshold": .3}}


class CandidateViewDetector:
    """Reuse live view projection/NMS without changing the candidate threshold."""
    def __init__(self, model, checkpoint_sha):
        self.model = model
        self.box_threshold = .3
        self.forward_count = 0
        self.info = {"model": "inactive weight-adjusted candidate", "resolvedRevision": "sha256:"+checkpoint_sha,
                     "boxThreshold": .3, "activationEligible": False}

    def predict(self, image):
        from torchvision.transforms.functional import pil_to_tensor
        tensor = pil_to_tensor(image).float()/255
        predictions = predictions_for(self.model, tensor, .3)
        self.forward_count += 1
        return [{"bboxPixelsXYXY": row["bbox"], "confidence": row["score"]} for row in predictions]


def scan_native_ground(image, detector):
    from adapters.live_capture import scan_frame
    before = detector.forward_count
    selected, scan = scan_frame(image, detector, profile="ground-tiles", color_sensitive=False)
    actual = detector.forward_count-before
    timings = scan.get("passTimings", [])
    if (actual != 5 or scan.get("modelPasses") != actual or len(timings) != actual
        or [r.get("kind") for r in timings] != ["context", "tile", "tile", "tile", "tile"]
        or scan.get("fullFrameContext") is not True or scan.get("tileRegion") != GROUND_REGION
        or scan.get("settings", {}).get("boxThreshold") != .3 or scan.get("nmsIou") != .5):
        raise ValueError("Live ground scanning differs from the frozen five-pass protocol")
    predictions = [{"bbox": row["bboxPixelsXYXY"], "score": row["confidence"],
                    "sourcePass": row.get("sourcePass")} for row in selected]
    return checked_predictions(predictions, image.width, image.height), {**scan, "actualForwards": actual}


def ground_baseline_records(tiled, frozen, root=ROOT):
    """Read only the four human native references; no AI train/validation rows."""
    human = [r for r in tiled.get("records", []) if r.get("referenceAuthority") == "human"]
    indexed = {r["frameId"]: r for r in human}
    if len(human) != 4 or set(indexed) != set(POSITIVE_IDS+[NEGATIVE_ID]):
        raise ValueError("Ground baseline must contain the same four human originals")
    records, paths = [], []
    for source in native_records(frozen):
        row = indexed[source["frameId"]]
        truth = [b["bbox"] for b in source["truth"]]
        if (row.get("referenceComplete") is not True or row.get("sha256") != source["imageSha256"]
            or row.get("sourceGroupId") != source["sourceGroupId"]
            or row.get("referenceStatus") != ("visible" if truth else "absent")
            or [b["bbox"] for b in row.get("truth", [])] != truth):
            raise ValueError("Ground baseline truth, original pixels, or source differs")
        method = row.get("methods", {}).get("ground_tiles", {})
        path = (root/method.get("predictionPath", "")).resolve()
        if (not path.is_relative_to(root.resolve()) or not path.is_file()
            or digest(path) != method.get("predictionSha256") or method.get("modelPasses") != 5):
            raise ValueError("Ground baseline predictions changed")
        saved = read(path)
        settings = saved.get("settings", {})
        if (settings.get("boxThreshold") != .25 or settings.get("tileRegion") != GROUND_REGION
            or settings.get("methods") != ["tiles"] or saved.get("image", {}).get("sha256") != source["imageSha256"]
            or len(saved.get("methods", [])) != 1 or saved["methods"][0].get("id") != "tiles"):
            raise ValueError("Ground baseline profile or threshold changed")
        predictions = [{"bbox": r["bboxPixelsXYXY"], "score": r["confidence"],
                        "sourcePass": r.get("sourcePass")} for r in saved["methods"][0]["candidates"]]
        records.append({"id": source["id"], "frameId": source["frameId"], "condition": "native",
            "group": source["sourceGroupId"], "kind": source["kind"], "truth": truth,
            "predictions": predictions, "seconds": method["seconds"], "modelPasses": 5,
            "predictionSha256": method["predictionSha256"]})
        paths.append(path)
    return records, paths


def timing_summary(seconds):
    return {"meanSeconds": statistics.mean(seconds), "medianSeconds": statistics.median(seconds),
        "p95Seconds": sorted(seconds)[math.ceil(len(seconds)*.95)-1], "totalSeconds": sum(seconds)}


def run(candidate, output, benchmark=ROOT/"experiments/results/video-distance-v2",
        previous=ROOT/"experiments/results/ai-corpus-distance-v1/report.json",
        dino=ROOT/"experiments/results/video-distance-v3/report.json",
        ground=ROOT/"experiments/results/ground-tiles-v1/report.json"):
    import torch
    from experiments.weight_recipe import load_recipe_checkpoint
    candidate, output, benchmark = Path(candidate).resolve(), Path(output).resolve(), Path(benchmark).resolve()
    previous, dino, ground = Path(previous).resolve(), Path(dino).resolve(), Path(ground).resolve()
    claim_path = candidate.parent/"weight-adjustment-evaluation-claim.json"
    if output.exists() or claim_path.exists():
        raise FileExistsError("This candidate regression is immutable and may be attempted only once")
    benchmark_path = benchmark/"report.json"
    if digest(benchmark_path) != BENCHMARK_SHA256:
        raise ValueError("Frozen video-distance-v2 report hash changed")
    source = read(benchmark_path); frozen = validate_benchmark(source); plan = evaluation_plan(frozen)
    inputs = candidate.parent/"inputs"
    rows, provenance = load_records(inputs/"training-export.json", inputs/"inventory.json", inputs/"ledger.json")
    if read(candidate.parent/"accepted-records.json") != rows:
        raise ValueError("Accepted training/validation records differ from frozen input export")
    inventory = read(inputs/"inventory.json")
    verify_no_overlap(rows, frozen, inventory)
    # Validation also influences checkpoint selection; it must not overlap test.
    verify_no_overlap([{**row, "split": "train"} for row in rows], frozen, inventory)
    tracked = [candidate, candidate.parent/"report.json", candidate.parent/"accepted-records.json",
        inputs/"training-export.json", inputs/"inventory.json", inputs/"ledger.json", benchmark_path,
        previous, dino, ground, ground.parent/"protocol.json", Path(__file__),
        ROOT/"experiments/weight_recipe.py", ROOT/"adapters/live_capture.py", ROOT/"vision/testbench.py"]
    image_paths = []
    for row in frozen:
        path = (benchmark/row["imagePath"]).resolve()
        if not path.is_relative_to(benchmark) or digest(path) != row["derivedImageSha256"]:
            raise ValueError("Frozen regression pixels changed or escaped benchmark")
        image_paths.append(path)
    tracked += image_paths
    hashes = {str(path): digest(path) for path in tracked}
    fit = read(candidate.parent/"report.json")
    if fit.get("checkpointSha256") != hashes[str(candidate)]:
        raise ValueError("Candidate checkpoint hash differs from training report")
    old, active, tiled, tile_protocol = read(previous), read(dino), read(ground), read(ground.parent/"protocol.json")
    if (old.get("schema") != "mushroom-vision.ai-corpus-distance-report.v1" or old.get("status") != "completed"
        or old.get("protocol", {}).get("benchmarkReportSha256") != BENCHMARK_SHA256
        or old["protocol"].get("scoreThreshold") != .3 or old["protocol"].get("iouThreshold") != .5):
        raise ValueError("Previous candidate did not use this fixed benchmark and threshold")
    if (active.get("status") != "completed" or active.get("protocol", {}).get("frozenPreviousReportSha256") != BENCHMARK_SHA256
        or active["protocol"].get("scoreThreshold") != .25 or active["protocol"].get("iouThreshold") != .5):
        raise ValueError("Saved DINO reference differs from frozen benchmark or operating threshold")
    old_records, dino_records = baseline_records(old, frozen, True), baseline_records(active, frozen)
    if (tiled.get("schema") != "mycoscout.ground-tiles-report.v1" or tiled.get("status") != "completed"
        or tiled.get("protocolSha256") != digest(ground.parent/"protocol.json")
        or tile_protocol.get("inputSha256", {}).get("experiments/results/video-distance-v2/report.json") != BENCHMARK_SHA256
        or tile_protocol.get("scoreThreshold") != .25 or tile_protocol.get("iouThreshold") != .5):
        raise ValueError("Saved ground-tile reference provenance or threshold changed")
    dino_ground_records, ground_prediction_paths = ground_baseline_records(tiled, frozen)
    hashes.update({str(path): digest(path) for path in ground_prediction_paths})
    torch.set_num_threads(4)
    load_started = time.perf_counter()
    model, metadata = load_recipe_checkpoint(candidate)
    config = validate_metadata(metadata, fit, provenance)
    model_load_seconds = time.perf_counter()-load_started
    protocol = {"schema": "mushroom-vision.weight-adjustment-evaluation-protocol.v1",
        "createdAt": datetime.now(timezone.utc).isoformat(), "frozenBeforeInference": True,
        "checkpointSha256": hashes[str(candidate)], "modelConfig": config, **provenance,
        "selectedEpoch": fit["selectedEpoch"], "benchmarkReportSha256": BENCHMARK_SHA256,
        "inputSha256": hashes, **plan, "scoreThreshold": .3, "iouThreshold": .5, "cpuThreads": 4,
        "thresholdSelectionOnTest": False, "evaluationPurpose": "Previously inspected correlated regression only",
        "trainingLabelAuthority": "ai-visual-review", "evaluationBoxAuthority": "human",
        "sourceGatePolicy": POLICY, "activationAllowed": False, "productionModelChanged": False,
        "actualDistanceM": None, "modelLoadSeconds": model_load_seconds}
    # An attempted run remains claimed even after an interruption: no silent
    # test repetition, replacement output, or after-the-fact threshold selection.
    write_new(claim_path, {"schema": "mushroom-vision.weight-adjustment-evaluation-claim.v1",
        "checkpointSha256": hashes[str(candidate)], "benchmarkReportSha256": BENCHMARK_SHA256,
        "output": str(output), "createdAt": protocol["createdAt"], "automaticRetryAllowed": False})
    output.mkdir(parents=True, exist_ok=False)
    write_new(output/"protocol.json", protocol)
    (output/"records").mkdir()
    (output/"ground-records").mkdir()
    records, started = [], time.perf_counter()
    for index, (row, path) in enumerate(zip(frozen, image_paths)):
        frame_started = time.perf_counter()
        image = tensor_image(path)
        predictions = checked_predictions(predictions_for(model, image, .3), image.shape[-1], image.shape[-2])
        record = {"id": row["id"], "frameId": row["frameId"], "condition": row["condition"],
            "group": row["sourceGroupId"], "kind": row["kind"], "truth": [b["bbox"] for b in row["truth"]],
            "predictions": predictions, "seconds": time.perf_counter()-frame_started,
            "imagePath": row["imagePath"], "imageSha256": row["derivedImageSha256"],
            "synthetic": row["synthetic"], "actualDistanceM": None, "humanVerifiedTruth": True}
        write_new(output/"records"/f"{index:02d}.json", record); records.append(record)
    ground_records = []
    view_detector = CandidateViewDetector(model, hashes[str(candidate)])
    for index, row in enumerate(native_records(frozen)):
        frame_started = time.perf_counter()
        with Image.open(benchmark/row["imagePath"]) as image_source:
            image = ImageOps.exif_transpose(image_source).convert("RGB")
        predictions, scan = scan_native_ground(image, view_detector)
        record = {"id": row["id"], "frameId": row["frameId"], "condition": "native",
            "group": row["sourceGroupId"], "kind": row["kind"], "truth": [b["bbox"] for b in row["truth"]],
            "predictions": predictions, "scan": scan, "seconds": time.perf_counter()-frame_started,
            "imagePath": row["imagePath"], "imageSha256": row["derivedImageSha256"],
            "synthetic": False, "actualDistanceM": None, "humanVerifiedTruth": True}
        write_new(output/"ground-records"/f"{index:02d}.json", record); ground_records.append(record)
    completed_forwards = len(records)+view_detector.forward_count
    if completed_forwards != plan["plannedForwards"]:
        raise ValueError("Actual model forwards differ from the frozen 46-forward protocol")
    if any(digest(path) != expected for path, expected in hashes.items()):
        raise ValueError("Evaluation input changed during inference; no completed report is valid")
    support = source_support(records)
    native = native_metrics(records, .3)
    ground_metrics = native_metrics(ground_records, .3)
    dino_ground_metrics = native_metrics(dino_ground_records, .25)
    seconds = [r["seconds"] for r in records]
    report = {"schema": SCHEMA, "status": "completed", "generatedAt": datetime.now(timezone.utc).isoformat(),
        "protocolSha256": digest(output/"protocol.json"), "protocol": protocol, "rows": rows_for(records, .3),
        "native": native, "records": records, "completedForwards": completed_forwards,
        "completedRawForwards": len(records), "completedGroundForwards": view_detector.forward_count, "sourceSupport": support,
        "timing": {**timing_summary(seconds),
            "modelLoadSeconds": model_load_seconds, "includes": "Image decoding, preprocessing, forward and fixed postprocessing; excludes model load"},
        "nativeGroundTiles": {"profile": "ground-tiles", "scoreThreshold": .3, "iouThreshold": .5,
            "tileRegion": GROUND_REGION, "nativeOriginals": 4, "completedForwards": view_detector.forward_count,
            "records": ground_records, "metrics": ground_metrics, "row": rows_for(ground_records, .3)[0],
            "timing": {**timing_summary([r["seconds"] for r in ground_records]),
                "includes": "Image decoding, five model passes, coordinate projection and NMS; excludes model load"},
            "comparisonToSavedDinoGround": gate_for(ground_metrics, dino_ground_metrics, support)},
        "comparisons": {
            "previousCandidate": {"reportSha256": hashes[str(previous)], "scoreThreshold": .3,
                "rows": rows_for(old_records, .3), "native": native_metrics(old_records, .3),
                "gate": gate_for(native, native_metrics(old_records, .3), support)},
            "savedDinoRaw": {"reportSha256": hashes[str(dino)], "scoreThreshold": .25,
                "model": active["protocol"].get("model"), "rows": rows_for(dino_records, .25),
                "native": native_metrics(dino_records, .25), "gate": gate_for(native, native_metrics(dino_records, .25), support),
                "scoreComparison": "Different uncalibrated model scores; outcomes at separately frozen operating thresholds"},
            "savedDinoGroundTiles": {"reportSha256": hashes[str(ground)], "protocolSha256": tiled["protocolSha256"],
                "scoreThreshold": .25, "native": dino_ground_metrics, "nativeRecords": dino_ground_records,
                "row": rows_for(dino_ground_records, .25)[0], "usedForCandidateSelection": False,
                "timing": {**timing_summary([r["seconds"] for r in dino_ground_records]),
                    "includes": "Saved five-pass model/scanning time; excludes model load and image decoding"},
                "scope": "Exactly the same four human native references; AI train/validation negatives excluded from this comparison",
                "scoreComparison": "Ground candidates use .3; saved DINO ground uses .25. Scores are not calibrated equivalences."}},
        "decision": "insufficient_evidence", "activationAllowed": False, "activationEligible": False,
        "automaticActivation": False, "productionModelChanged": False, "trainingPerformed": False,
        "fitAndValidation": {"reportSha256": hashes[str(candidate.parent/"report.json")], "labelAuthority": "ai-visual-review",
            "includedInTestMetrics": False, "trainFrames": sum(r["split"] == "train" for r in rows),
            "validationFrames": sum(r["split"] == "validation" for r in rows)},
        "actualDistanceM": None, "elapsedSeconds": time.perf_counter()-started, "inputIntegrityVerified": True,
        "limitations": ["Only three positive originals and one negative original from one creator; resized versions do not add independent observations.",
            "The regression was previously inspected and cannot establish independent field performance or calibrated distance.",
            "Unmatched detections on positive frames are alerts outside saved boxes, not automatically verified false mushrooms.",
            "AI training labels remain provisional. The only positive AI validation frame is an editorial inset.",
            "Candidate raw and ground-tiled native outcomes are separate methods. Ground mode uses four native originals only, never resized views or AI train/validation negatives.",
            "The lower 65% is a manually selected image rectangle, not semantic ground. Full-frame context and the same model threshold are retained.",
            "No checkpoint selection, threshold tuning, species identification, flight claim, or activation follows this test."]}
    write_new(output/"report.json", report)
    print(json.dumps({"status": report["status"], "rows": report["rows"], "decision": report["decision"], "timing": report["timing"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, default=ROOT/"experiments/results/video-distance-v2")
    parser.add_argument("--previous-candidate", type=Path, default=ROOT/"experiments/results/ai-corpus-distance-v1/report.json")
    args = parser.parse_args()
    run(args.candidate, args.output, args.benchmark, args.previous_candidate)
