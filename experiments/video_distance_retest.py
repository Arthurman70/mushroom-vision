"""Rerun frozen video pixels and compare two fixed category heads, without training.

Detection truth is the existing human presence boxes. Category references are
explicit assistant visual opinions, never human species truth. No active model,
review, training split or threshold is changed by this experiment.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.video_distance_benchmark import metric_counts
from vision.testbench import compare
from vision.training_pipeline import iou
from vision.video_ai_pass import atomic_json

METHODS = {"raw": "original", "color": "color", "tiles": "tiles"}
TILED_CONDITIONS = {"native", "64px", "32px", "16px"}
CLASSIFIERS = {
    "baseline": ("vision/checkpoints/groups-v1.json", "data/public-fungi-v3/manifest.json"),
    "expanded": ("vision/checkpoints/groups-expanded-v1.json", "data/biodiversity-fungi-v1/manifest.json"),
}
REFERENCE_CLASS = "true_morels"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def protected_sources(manifest, references):
    """Reject exact-content or source-group overlap with any fitted partition."""
    hashes = {row["imageSha256"] for row in references}
    groups = {row["sourceGroupId"] for row in references}
    fitted = [row for row in manifest["images"] if row.get("split") in {"train", "validation"}]
    for row in fitted:
        if hashes.intersection({row.get("sha256"), row.get("originalSha256")}) or groups.intersection(
                {row.get("groupId"), row.get("sourceGroupId"), row.get("splitGroupId")}):
            raise ValueError("Evaluation content or source group overlaps a fitted partition")
    return {"trainingAndValidationRowsChecked": len(fitted), "exactContentOverlap": 0,
            "sourceGroupOverlap": 0, "nearDuplicateGuarantee": False}


def known_box_crop(image, box):
    """Outward integer crop; fractional truth never becomes a larger native object."""
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
        raise ValueError("Truth box lies outside the frozen image")
    bounds = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
    return image.crop(bounds), {"boundsPixelsXYXY": bounds, "nativeTargetWidth": x2-x1,
        "nativeTargetHeight": y2-y1, "productionEligible": min(x2-x1, y2-y1) >= 32,
        "knownTargetLocation": True, "localizationEvaluated": False}


def summarize_classification(records):
    result = {"targets": len(records), "referenceAuthority": "assistant", "referenceClass": REFERENCE_CLASS,
              "humanSpeciesGroundTruth": False, "productionEligible": sum(r["crop"]["productionEligible"] for r in records)}
    for name in CLASSIFIERS:
        guesses = [r["classification"][name] for r in records]
        retained = [guess for guess in guesses if not guess["abstained"]]
        result[name] = {"top1Agreement": sum(g["guesses"][0]["classId"] == REFERENCE_CLASS for g in guesses),
            "top3Agreement": sum(any(c["classId"] == REFERENCE_CLASS for c in g["guesses"]) for g in guesses),
            "retained": len(retained), "retainedAgreement": sum(g["guesses"][0]["classId"] == REFERENCE_CLASS for g in retained),
            "abstained": len(guesses)-len(retained)}
    return result


def write_examples(output, records):
    """Pick failures by fixed condition/frame ordering, not a flattering score."""
    failures = [row for row in records if row["kind"] == "positive"
              and any(max((iou(truth["bbox"], pred["bbox"]) for pred in row["predictions"]["raw"]), default=0) < .5 for truth in row["truth"])]
    failed = [next((row for row in failures if row["condition"] == condition), None)
              for condition in ["native", "32px", "16px", "6p9px", "3p45px", "0p69px"]]
    failed = [row for row in failed if row is not None]
    canvas = Image.new("RGB", (1200, 310 * math.ceil(len(failed)/2)), "#11211a")
    draw = ImageDraw.Draw(canvas)
    examples = []
    for index, row in enumerate(failed):
        with Image.open(output / row["imagePath"]) as source:
            source = source.convert("RGB")
            overlay = ImageDraw.Draw(source)
            for truth in row["truth"]:
                overlay.rectangle(truth["bbox"], outline="#6dffac", width=3)
            for pred in row["predictions"]["raw"]:
                overlay.rectangle(pred["bbox"], outline="#ff9568", width=2)
            thumb = ImageOps.contain(source, (590, 260))
            x, y = (index % 2)*600, (index//2)*310
            canvas.paste(thumb, (x, y+40))
            draw.text((x+8, y+6), row["condition"]+" / "+row["frameId"], fill="white")
            draw.text((x+8, y+22), "Green: human region. Orange: detector. Synthetic scale is not meters.", fill="#c1d4c6")
        examples.append({"id": row["id"], "condition": row["condition"], "reason": "At least one human region missed at IoU 0.5", "imagePath": row["imagePath"]})
    if failed:
        canvas.save(output / "failure-examples.jpg", quality=90)
    atomic_json(output / "failure-examples.json", {"examples": examples, "selection": "First raw miss per native/32/16/6.9/3.45/0.69 px condition in frozen record order", "notIndependentSamples": True})


def run(output, previous, collection):
    output, previous, collection = Path(output), Path(previous), Path(collection)
    if output.exists():
        raise ValueError("Choose a new output directory; frozen runs cannot be overwritten")
    prior = read(previous / "report.json")
    if prior["status"] != "completed" or prior["sample"]["labelAuthority"] != "human":
        raise ValueError("Requires the completed frozen human-box video benchmark")
    sources = read(collection / "manifest.json")
    source_by_id = {row["id"]: row for row in sources["frames"]}
    small = read(previous / "native-evidence.json")["smallPositiveOnly"]
    references = prior["records"] + small
    immutable = {str(previous / "report.json"): digest(previous / "report.json"),
                 str(collection / "manifest.json"): digest(collection / "manifest.json")}
    for row in references:
        frame = source_by_id[row["frameId"]]
        if frame["sha256"] != row["imageSha256"] or digest(collection/frame["path"]) != row["imageSha256"]:
            raise ValueError("Original source frame changed")
        immutable[str(collection/frame["path"])] = row["imageSha256"]
    for row in prior["records"]:
        if digest(previous/row["imagePath"]) != row["derivedImageSha256"]:
            raise ValueError("Frozen resized pixels changed")
    provenance = {}
    for name, (checkpoint_path, manifest_path) in CLASSIFIERS.items():
        checkpoint = read(ROOT/checkpoint_path)
        if REFERENCE_CLASS not in {row["classId"] for row in checkpoint["classes"]}:
            raise ValueError("Reference category is absent from the checkpoint")
        if digest(ROOT/manifest_path) != checkpoint["training"]["manifestSha256"]:
            raise ValueError("Classifier training provenance does not match its dataset")
        provenance[name] = {"checkpointSha256": digest(ROOT/checkpoint_path), "evidenceVersion": checkpoint["evidenceVersion"],
            "manifestSha256": digest(ROOT/manifest_path), "splitAudit": protected_sources(read(ROOT/manifest_path), references)}
        immutable[str(ROOT/checkpoint_path)] = digest(ROOT/checkpoint_path)
        immutable[str(ROOT/manifest_path)] = digest(ROOT/manifest_path)
    output.mkdir(parents=True)
    (output/"images").mkdir(); (output/"predictions").mkdir()
    total = sum(6 if row["condition"] in TILED_CONDITIONS else 2 for row in prior["records"]) + len(small)
    protocol = {**copy.deepcopy(prior["protocol"]), "schema": "mycoscout.video-distance-retest-protocol.v1",
        "frozenPreviousReportSha256": digest(previous/"report.json"), "reuseInference": False, "cpuThreads": 4,
        "tiledConditions": sorted(TILED_CONDITIONS), "plannedModelForwards": total, "plannedNewForwards": total,
        "plannedReusedNativeForwards": 0, "classifierProvenance": provenance,
        "classification": {"referenceAuthority": "assistant", "referenceClass": REFERENCE_CLASS, "humanSpeciesGroundTruth": False,
            "referenceEvidence": "Assistant inspected native images of frozen 043, 005 and 022: pitted morel-like caps. Human reviews identify mushroom presence only, with group unknown.",
            "input": "Exact outward-rounded human box from the already scaled frame; no padding, reconstruction, deblurring or new detail",
            "diagnosticOnly": True, "productionMinNativeSidePixels": 32,
            "smallCropPolicy": "Diagnostic guesses are measured below 32 pixels, but the production guard remains ineligible regardless of confidence"},
        "trainingPerformed": False, "modelActivation": False, "actualDistanceM": None}
    atomic_json(output/"protocol.json", protocol)
    report = {"schema": "mycoscout.video-distance-report.v1", "runVersion": "video-distance-v2", "status": "running",
        "generatedAt": datetime.now(timezone.utc).isoformat(), "sample": copy.deepcopy(prior["sample"]),
        "progress": {"completed": 0, "total": total, "unit": "model-forwards", "newCompleted": 0, "reused": 0},
        "rows": [{**{k: r[k] for k in ("id", "label", "native", "targetPixels", "realizedTargetWidths", "realizedTargetWidthPixels")},
                  "raw": None, "color": None, "tiles": None} for r in prior["rows"]],
        "protocol": protocol, "nativeBaseline": None, "nativeSmallPositiveOnly": None, "records": [],
        "classificationRows": [], "previousComparison": [], "actualDistanceM": None, "actuationEnabled": False,
        "limitations": ["Same frozen human presence boxes and exact image hashes as v1; all displayed v2 inference is fresh.",
            "Three correlated morel-like views from one creator are a stress test, not a population detection rate.",
            "Synthetic pixel footprints are not measured 10/20/50/100 metre performance; camera optics and subject sizes are unknown.",
            "Tiling is rerun at native, 64, 32 and 16 px only. Smaller-scale tiling is not remeasured or silently copied.",
            "Classification compares known human-box crops against a provisional assistant morel-group opinion; it is not human species accuracy or a localization result.",
            "Diagnostic tiny-crop scores can be confidently wrong. Production still requires at least 32 original pixels on each crop side.",
            "Training added category-head examples, not new detector weights. No test image is used to fit or tune either head.",
            "Whole-frame resize on gray padding differs from real camera motion, blur, clutter, occlusion and lighting.",
            "Partial lawn-mushroom annotations support positive recall only; unmarked regions cannot be scored as negatives.",
            "No model activation, human-review changes, split changes or robot commands occur."]}
    atomic_json(output/"report.json", report)
    from adapters.scout_frame import LocalDetector
    from vision.multiclass import Predictor
    detector = LocalDetector(ROOT/"models/grounding-dino-tiny", device="cpu", threads=4, box_threshold=.25, text_threshold=.25)
    old_model = read(previous/"predictions"/(prior["records"][0]["id"]+".json"))["model"]
    for key in ("model", "resolvedRevision", "prompt"):
        if detector.info[key] != old_model[key]:
            raise ValueError("Detector differs from original benchmark: "+key)
    predictors = {name: Predictor(ROOT/values[0]) for name, values in CLASSIFIERS.items()}
    started = time.perf_counter()
    def progress(event):
        report["progress"]["completed"] = done + event["completed"]
        report["progress"]["newCompleted"] = report["progress"]["completed"]
        report["elapsedSeconds"] = time.perf_counter()-started
        atomic_json(output/"report.json", report)
    done = 0
    for source_row in prior["records"]:
        row = copy.deepcopy(source_row)
        image_path = output/row["imagePath"]
        shutil.copyfile(previous/row["imagePath"], image_path)
        with Image.open(image_path) as image_source:
            image = image_source.convert("RGB")
        methods = ["original", "color"] + (["tiles"] if row["condition"] in TILED_CONDITIONS else [])
        result = compare(image, row["derivedImageSha256"], {"methods": methods, "tileGrid": 2, "scale": 1, "boxThreshold": .25}, detector, progress)
        done += result["modelPasses"]
        atomic_json(output/"predictions"/(row["id"]+".json"), result)
        row["predictions"] = {name: [{"id": b["id"], "bbox": b["bboxPixelsXYXY"], "score": b["confidence"]}
            for b in next(m for m in result["methods"] if m["id"] == method)["candidates"]]
            for name, method in METHODS.items() if method in methods}
        row.pop("cachedNativeRawSha256", None); row.pop("nearestPredictions", None)
        row["reusedInference"] = False
        if row["kind"] == "positive":
            crop, row["crop"] = known_box_crop(image, row["truth"][0]["bbox"])
            row["classification"] = {name: predictor.predict(crop) for name, predictor in predictors.items()}
        report["records"].append(row)
        condition_records = [r for r in report["records"] if r["condition"] == row["condition"]]
        expected_count = sum(r["condition"] == row["condition"] for r in prior["records"])
        if len(condition_records) == expected_count:
            summary = next(r for r in report["rows"] if r["id"] == row["condition"])
            for method in METHODS:
                summary[method] = metric_counts(condition_records, method) if method in row["predictions"] else None
            summary["complete"] = True
            classification = {"id": row["condition"], **summarize_classification([r for r in condition_records if r["kind"] == "positive"])}
            report["classificationRows"].append(classification)
            old = next(r for r in prior["rows"] if r["id"] == row["condition"])
            report["previousComparison"].append({"id": row["condition"], "methods": {m: {"matchedDelta": summary[m]["matched"]-old[m]["matched"],
                "falsePositiveDelta": summary[m]["falsePositives"]-old[m]["falsePositives"], "negativeFalsePositiveDelta": summary[m]["negativeFalsePositives"]-old[m]["negativeFalsePositives"]}
                for m in METHODS if summary[m] is not None}})
            print(json.dumps({"condition": row["condition"], "raw": summary["raw"], "color": summary["color"], "tiles": summary["tiles"], "classification": classification}), flush=True)
        atomic_json(output/"report.json", report)
    small_details = []
    for reference in small:
        frame = source_by_id[reference["frameId"]]
        with Image.open(collection/frame["path"]) as source:
            image = source.convert("RGB")
        result = compare(image, frame["sha256"], {"methods": ["original"], "tileGrid": 2, "scale": 1, "boxThreshold": .25}, detector, progress)
        done += result["modelPasses"]
        predictions = [{"bbox": b["bboxPixelsXYXY"], "score": b["confidence"]} for b in result["methods"][0]["candidates"]]
        used = set(); matched = 0
        for truth in reference["truth"]:
            candidates = [(iou(truth["bbox"], pred["bbox"]), i) for i, pred in enumerate(predictions) if i not in used]
            overlap, index = max(candidates, default=(0, -1))
            if overlap >= .5:
                matched += 1; used.add(index)
        small_details.append({**reference, "predictions": predictions, "matched": matched, "reused": False, "falsePositives": None})
    widths = [b["bbox"][2]-b["bbox"][0] for row in small for b in row["truth"]]
    report["nativeSmallPositiveOnly"] = {"frames": len(small), "matched": sum(r["matched"] for r in small_details),
        "targets": len(widths), "falsePositives": None, "sourceVideos": len({r["sourceId"] for r in small}),
        "sourceGroups": len({r["sourceGroupId"] for r in small}), "targetWidthPixels": {"min": min(widths), "max": max(widths)},
        "scoreThreshold": .25, "iouThreshold": .5, "labelAuthority": "human", "reused": False, "actualDistanceM": None, "positiveOnly": True}
    atomic_json(output/"native-small-evidence.json", small_details)
    if done != total or any(digest(path) != expected for path, expected in immutable.items()):
        raise ValueError("Forward count mismatch or immutable inputs changed")
    report.update(status="completed", completedAt=datetime.now(timezone.utc).isoformat(), elapsedSeconds=time.perf_counter()-started,
                  inputIntegrityVerified=True, trainingPerformed=False, thresholdsChanged=False, weightsChanged=False)
    atomic_json(output/"report.json", report)
    write_examples(output, report["records"])
    print(json.dumps({"status": "completed", "forwards": done, "elapsedSeconds": report["elapsedSeconds"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/results/video-distance-v2")
    parser.add_argument("--previous", type=Path, default=ROOT/"experiments/results/video-distance-v1")
    parser.add_argument("--collection", type=Path, default=ROOT/"data/video-discovery-v1")
    args = parser.parse_args()
    run(args.output, args.previous, args.collection)
