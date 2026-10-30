"""Fixed crop-framing diagnostic on frozen video frames; no detector or training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.video_distance_retest import CLASSIFIERS, REFERENCE_CLASS, digest, known_box_crop, protected_sources, read
from vision.video_ai_pass import atomic_json


def context_crop(image, box):
    """Square 1.25x context, with explicit neutral padding only at image edges."""
    x1, y1, x2, y2 = box
    if not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
        raise ValueError("Candidate outside original image")
    side = max(1, math.ceil(max(x2-x1, y2-y1)*1.25))
    left, top = math.floor((x1+x2-side)/2), math.floor((y1+y2-side)/2)
    src = [max(0, left), max(0, top), min(image.width, left+side), min(image.height, top+side)]
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(image.crop(src), (src[0]-left, src[1]-top))
    return canvas


def fitted_square(crop):
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), (127, 127, 127))
    canvas.paste(crop, ((side-crop.width)//2, (side-crop.height)//2))
    return canvas


def run(output, previous):
    output, previous = Path(output), Path(previous)
    if output.exists():
        raise ValueError("Choose a new output; do not overwrite a frozen diagnostic")
    prior = read(previous/"report.json")
    references = [row for row in prior["records"] if row["kind"] == "positive"]
    for row in references:
        if digest(previous/row["imagePath"]) != row["derivedImageSha256"]:
            raise ValueError("Frozen view changed")
    provenance = {}
    for name, (checkpoint, manifest) in CLASSIFIERS.items():
        data = read(ROOT/checkpoint)
        if digest(ROOT/manifest) != data["training"]["manifestSha256"] or REFERENCE_CLASS not in {c["classId"] for c in data["classes"]}:
            raise ValueError("Classifier provenance mismatch")
        provenance[name] = {"checkpointSha256": digest(ROOT/checkpoint), "splitAudit": protected_sources(read(ROOT/manifest), references)}
    output.mkdir(parents=True)
    protocol = {"schema": "mycoscout.crop-context-protocol.v1", "createdAt": datetime.now(timezone.utc).isoformat(),
        "frozenBeforeInference": True, "priorReportSha256": digest(previous/"report.json"), "classifiers": provenance,
        "policies": {"tight": "Exact outward-rounded known human box, current diagnostic baseline",
                     "fitted": "Same tight crop centered on a neutral square without resizing",
                     "context": "Square crop centered on known box; side 1.25 times longer box dimension, outward rounded; neutral padding only at source edges"},
        "referenceAuthority": "assistant", "referenceClass": REFERENCE_CLASS, "humanSpeciesGroundTruth": False,
        "knownTargetLocation": True, "localizationEvaluated": False, "modelActivation": False, "trainingPerformed": False,
        "actualDistanceM": None, "productionMinimumNativeSidePixels": 32}
    atomic_json(output/"protocol.json", protocol)
    from vision.multiclass import Predictor
    predictors = {name: Predictor(ROOT/config[0]) for name, config in CLASSIFIERS.items()}
    results = []
    for row in references:
        with Image.open(previous/row["imagePath"]) as source:
            image = source.convert("RGB")
        crop, evidence = known_box_crop(image, row["truth"][0]["bbox"])
        views = {"tight": crop, "fitted": fitted_square(crop), "context": context_crop(image, row["truth"][0]["bbox"])}
        results.append({"id": row["id"], "condition": row["condition"], "sourceGroupId": row["sourceGroupId"],
            "imageSha256": row["derivedImageSha256"], "productionEligible": evidence["productionEligible"],
            "policies": {policy: {name: predictor.predict(view) for name, predictor in predictors.items()} for policy, view in views.items()}})
    summaries = []
    for condition in dict.fromkeys(r["condition"] for r in results):
        selected = [r for r in results if r["condition"] == condition]
        summary = {"condition": condition, "targets": len(selected), "productionEligible": sum(r["productionEligible"] for r in selected), "policies": {}}
        for policy in protocol["policies"]:
            summary["policies"][policy] = {}
            for name in predictors:
                predictions = [r["policies"][policy][name] for r in selected]
                summary["policies"][policy][name] = {
                    "top1Agreement": sum(p["guesses"][0]["classId"] == REFERENCE_CLASS for p in predictions),
                    "retained": sum(not p["abstained"] for p in predictions),
                    "retainedAgreement": sum(not p["abstained"] and p["guesses"][0]["classId"] == REFERENCE_CLASS for p in predictions)}
        summaries.append(summary)
    report = {"schema": "mycoscout.crop-context-report.v1", "status": "completed", "protocol": protocol, "rows": summaries, "records": results,
        "limitations": ["Three correlated views from one video, resized to eight sizes; not independent accuracy.",
            "Human presence boxes supply target locations; morel category is a provisional assistant visual opinion.",
            "Context framing is an exploratory diagnostic motivated by native tight-crop failures. Do not tune or promote it from this test alone.",
            "The same underlying pixels are retained. Framing adds no optical detail, depth, GPS or detection capability.",
            "The live 32-native-pixel guard remains in effect regardless of diagnostic score."]}
    atomic_json(output/"report.json", report)
    print(__import__("json").dumps(summaries), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"experiments/results/video-distance-context-v1")
    parser.add_argument("--previous", type=Path, default=ROOT/"experiments/results/video-distance-v1")
    args = parser.parse_args()
    run(args.output, args.previous)
