"""Frozen fractional-pixel color/discovery development test; not distance accuracy."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import platform
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageOps
from vision.far_color import enhance, proposals, color_evidence, contrast_map, analyze
from adapters.detect_fungi import candidate_records
from experiments.run_grounding_dino_batch import class_agnostic_nms, intersection_over_union

MANIFEST_SHA = "334b8c209db455e4b7d32dc5bd633b0a410638b8c05d10e8295722a1e02542a0"
SOURCE_SPEC = [
    ("inat-photo-745660170", [400, 700, 760, 1000]),
    ("inat-photo-734698732", [30, 500, 700, 1000]),
    ("inat-photo-718433879", [530, 500, 760, 720]),
    ("inat-photo-717487456", [0, 520, 1024, 768]),
]
LEVELS = [(6.9, 10), (3.45, 20), (1.38, 50), (.69, 100), (32, None)]
OUT = ROOT / "experiments/results/far-pixel-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")


def quarantine_color_report(payload, source_prediction_sha256=None):
    """Mark every benchmark cue as synthetic evaluation evidence, never a field find."""
    if payload.get("schema") != "mycoscout.color-proposals.v1":
        raise ValueError("Expected a color-proposals report")
    if source_prediction_sha256 is not None and (len(source_prediction_sha256) != 64 or
            any(c not in "0123456789abcdef" for c in source_prediction_sha256)):
        raise ValueError("Expected a lowercase SHA-256 source prediction digest")
    result = deepcopy(payload)
    result["synthetic"] = True
    result["evaluationOnly"] = True
    if source_prediction_sha256 is not None:
        result["derivedFromPredictionSha256"] = source_prediction_sha256
    return result


def protocol():
    return {"schema": "mycoscout.far-pixel-protocol.v1", "frozenBeforeInference": True,
            "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "benchmarkRole": "Fixed synthetic development audit; reused sources, not independent field validation",
            "auditAmendment": "Initial pilot was stopped after seven first-scene forwards: enlarged visual QA found an ambiguous pale body in source718433879 negative crop. Crop narrowed before any inference for that source. Initial protocol and partial outputs retained in invalidated-background-pilot; no thresholds changed.",
            "sourceManifest": "data/black-trumpet-discovery-v2/manifest.json", "sourceManifestSha256": MANIFEST_SHA,
            "sources": [{"id": id_, "negativeBackgroundCropXYXY": box} for id_, box in SOURCE_SPEC],
            "levels": [{"targetPixels": n, "illustrativeDistanceM": m} for n, m in LEVELS],
            "primaryAnnotationIndex": 0, "supersampling": 16,
            "framePolicy": "Retain original source dimensions; source scene is reduced onto a stretched target-free crop from that scene",
            "negativeSelection": "Four source crops visually reviewed by assistant before inference; moss, bark, roots and needles, no visible mushrooms. Not expert adjudicated.",
            "color": {"proposalMethod": "lab-local-contrast-v1", "thresholdDeltaE": 8, "maxProposals": 12,
                      "nmsRadiusPixels": 16, "localBlurSigma": 8,
                      "enhancement": {"localBlurSigma": 4, "luminanceResidualGain": .75, "chromaResidualGain": 2}},
            "detector": {"model": "IDEA-Research/grounding-dino-tiny", "revision": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
                         "prompt": "a mushroom. a fungus.", "boxThreshold": .25, "textThreshold": .25, "nmsIou": .5,
                         "matchingIou": .5, "rawAndEnhancedFullFrameOnly": True, "cpuThreads": 4},
            "colorCueHitRule": "Peak point within focal target rectangle expanded by two native pixels; this is cue proximity, not object detection",
            "thresholdTuning": "None; all constants and source/background choices frozen before inference",
            "limitations": ["Not a camera-distance simulator: no calibrated optics, atmosphere, motion, occlusion or sensor model.",
                            "Fractional target widths apply to an existing whole-body annotation, not a measured 5 cm cap.",
                            "Synthetic scene-island edges, resampling and enlarged background texture can create artificial contrast.",
                            "Four source scenes/observers; 24 derivatives are not 24 independent source scenes.",
                            "Negative controls are selected source crops, not representative mushroom-free flight footage.",
                            "A color anomaly is not evidence of mushroom presence or a species identification."]}


def fractional_scene(background, source, annotations, target_width, position):
    """Area-integrated subpixel coverage, never clamp a 0.69-pixel body to 1px."""
    supersample = 16
    primary = annotations[0]["bboxPixelsXYXY"]
    scale = target_width / (primary[2] - primary[0])
    sw, sh = max(1, round(source.width * scale * supersample)), max(1, round(source.height * scale * supersample))
    sx, sy = sw / supersample / source.width, sh / supersample / source.height
    patch = source.resize((sw, sh), Image.Resampling.LANCZOS)
    x = round((background.width * position[0] - sw / supersample / 2) * supersample)
    y = round((background.height * position[1] - sh / supersample / 2) * supersample)
    left, top = math.floor(x / supersample), math.floor(y / supersample)
    right, bottom = math.ceil((x + sw) / supersample), math.ceil((y + sh) / supersample)
    if left < 0 or top < 0 or right > background.width or bottom > background.height:
        raise ValueError("Synthetic island exceeds retained canvas")
    local = background.crop((left, top, right, bottom)).resize(((right - left) * supersample,
                                (bottom - top) * supersample), Image.Resampling.NEAREST)
    local.paste(patch, (x - left * supersample, y - top * supersample))
    result = background.copy()
    result.paste(local.resize((right - left, bottom - top), Image.Resampling.BOX), (left, top))
    boxes = [[a["bboxPixelsXYXY"][0] * sx + x / supersample,
              a["bboxPixelsXYXY"][1] * sy + y / supersample,
              a["bboxPixelsXYXY"][2] * sx + x / supersample,
              a["bboxPixelsXYXY"][3] * sy + y / supersample] for a in annotations]
    return result, boxes, {"requestedTargetWidthPixels": target_width,
                          "realizedTargetWidthPixels": boxes[0][2] - boxes[0][0],
                          "supersampling": supersample, "sceneScaleXY": [sx, sy],
                          "sceneBoundsPixelsXYXY": [x / supersample, y / supersample,
                                                     (x + sw) / supersample, (y + sh) / supersample]}


def prepare(out):
    manifest_path = ROOT / "data/black-trumpet-discovery-v2/manifest.json"
    if digest(manifest_path) != MANIFEST_SHA:
        raise ValueError("Frozen source annotations changed")
    manifest = json.loads(manifest_path.read_text())
    source_by_id = {r["id"]: r for r in manifest["images"]}
    (out / "images").mkdir()
    records = []
    for index, (source_id, crop) in enumerate(SOURCE_SPEC):
        row = source_by_id[source_id]
        path = manifest_path.parent / row["path"]
        if digest(path) != row["sha256"]:
            raise ValueError("Frozen source photo changed")
        with Image.open(path) as original:
            source = ImageOps.exif_transpose(original).convert("RGB")
        background = source.crop(crop).resize(source.size, Image.Resampling.LANCZOS)
        common = {"sourceId": source_id, "observerId": row["observerId"], "originalSha256": row["sha256"],
                  "sourceUrl": row["sourceUrl"], "photoLicense": row["photoLicense"],
                  "frameWidth": source.width, "frameHeight": source.height, "synthetic": True,
                  "evaluationOnly": True, "actualRangeM": None, "backgroundCropXYXY": crop}
        for target, distance in LEVELS:
            name = f"{index}-{str(target).replace('.', 'p')}px"
            image, boxes, transform = fractional_scene(background, source, row["annotations"], target,
                                                       [(0.43, .37), (.61, .43), (.47, .64), (.63, .58)][index])
            dest = out / "images" / f"{name}.png"
            image.save(dest)
            records.append({**common, "id": name, "image": f"images/{name}.png", "sha256": digest(dest),
                            "kind": "positive-control" if distance is None else "far-pixel",
                            "targetPixels": target, "illustrativeDistanceM": distance,
                            "annotations": boxes, "transform": transform})
        name = f"{index}-negative"
        dest = out / "images" / f"{name}.png"
        background.save(dest)
        records.append({**common, "id": name, "image": f"images/{name}.png", "sha256": digest(dest),
                        "kind": "negative-control", "targetPixels": None, "illustrativeDistanceM": None,
                        "annotations": [], "transform": None})
    write(out / "inputs.json", {"schema": "mycoscout.far-pixel-inputs.v1", "images": records})
    return records


def score(candidates, boxes):
    used = set()
    hits = []
    false = 0
    for c in candidates:
        eligible = [(intersection_over_union(c["bboxPixelsXYXY"], box), i) for i, box in enumerate(boxes) if i not in used]
        best = max(eligible, default=(0, -1))
        if best[0] >= .5:
            used.add(best[1]); hits.append(best[1])
        else:
            false += 1
    return {"focalHit": 0 in used, "truePositive": len(hits), "falsePositive": false,
            "falseNegative": len(boxes) - len(hits), "matchedAnnotationIndices": sorted(hits),
            "candidateCount": len(candidates)}


def cue_score(candidates, boxes):
    def hit(c, box):
        x, y = c["pointPixelsXY"]
        return box[0] - 2 <= x <= box[2] + 2 and box[1] - 2 <= y <= box[3] + 2
    return {"focalCueHit": bool(boxes) and any(hit(c, boxes[0]) for c in candidates),
            "cueCount": len(candidates), "offTargetCues": sum(not any(hit(c, b) for b in boxes) for c in candidates)}


def run(out, records):
    import torch
    import transformers
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    torch.set_num_threads(4)
    model_dir = ROOT / "models/grounding-dino-tiny"
    started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True,
                  trust_remote_code=False, use_safetensors=True).eval()
    load_seconds = time.perf_counter() - started
    print(f"Loaded local GroundingDINO in {load_seconds:.2f}s; protocol {digest(out / 'protocol.json')}", flush=True)
    (out / "predictions").mkdir()
    results = []
    for record in records:
        with Image.open(out / record["image"]) as im:
            original = im.convert("RGB")
        delta = contrast_map(original)
        color_cues = proposals(original)
        colors = quarantine_color_report(analyze(out / record["image"]))
        write(out / "predictions" / f"{record['id']}-color-cues.json", colors)
        methods = {}
        for name, im in [("raw", original), ("color", enhance(original))]:
            start = time.perf_counter()
            inputs = processor(images=im, text="a mushroom. a fungus.", return_tensors="pt")
            with torch.inference_mode():
                output = model(**inputs)
            detections = processor.post_process_grounded_object_detection(output, inputs.input_ids,
                              box_threshold=.25, text_threshold=.25, target_sizes=[(im.height, im.width)])[0]
            labels = detections.get("text_labels", detections.get("labels"))
            candidates = candidate_records(detections["boxes"].tolist(), detections["scores"].tolist(),
                                            labels, im.width, im.height)
            candidates = class_agnostic_nms(candidates, .5)
            for c in candidates:
                c["colorEvidence"] = color_evidence(delta, c["bboxPixelsXYXY"], enhanced=name == "color")
            seconds = time.perf_counter() - start
            payload = {"schema": "mycoscout.unlocated-candidates.v1", "image": colors["image"],
                       "detector": {**protocol()["detector"], "variant": name},
                       "candidates": candidates, "runtimeSeconds": seconds,
                       "evaluationOnly": True, "synthetic": True, "inputId": record["id"]}
            write(out / "predictions" / f"{record['id']}-{name}.json", payload)
            methods[name] = {**score(candidates, record["annotations"]), "seconds": seconds,
                             "processorTensorShape": list(inputs.pixel_values.shape)}
            print(json.dumps({"image": record["id"], "method": name, **methods[name]}), flush=True)
        results.append({"id": record["id"], "kind": record["kind"], "targetPixels": record["targetPixels"],
                        "sourceId": record["sourceId"], "methods": methods,
                        "colorOnly": cue_score(color_cues, record["annotations"])})
    write(out / "results.json", {"records": results})
    rows = []
    for pixels, meters in LEVELS:
        selected = [r for r in results if r["targetPixels"] == pixels]
        row = {"label": f"{meters} m illustration" if meters else "32 px positive control", "targetPixels": pixels,
               "illustrativeDistanceM": meters}
        for name in ("raw", "color"):
            hits = sum(r["methods"][name]["focalHit"] for r in selected)
            row[name] = {"hits": hits, "targets": len(selected), "recall": hits / len(selected),
                         "falseDetections": sum(r["methods"][name]["falsePositive"] for r in selected)}
        row["colorOnly"] = {"focalCueHits": sum(r["colorOnly"]["focalCueHit"] for r in selected),
                            "targets": len(selected), "cues": sum(r["colorOnly"]["cueCount"] for r in selected),
                            "offTargetCues": sum(r["colorOnly"]["offTargetCues"] for r in selected)}
        rows.append(row)
    negatives = [r for r in results if r["kind"] == "negative-control"]
    negative_metrics = {name: {"frames": len(negatives),
         "falseDetections": sum(r["methods"][name]["candidateCount"] for r in negatives),
         "framesWithFalseAlerts": sum(r["methods"][name]["candidateCount"] > 0 for r in negatives)} for name in ("raw", "color")}
    negative_metrics["colorOnly"] = {"frames": len(negatives), "cues": sum(r["colorOnly"]["cueCount"] for r in negatives),
                                    "framesWithCues": sum(r["colorOnly"]["cueCount"] > 0 for r in negatives)}
    p = json.loads((out / "protocol.json").read_text())
    report = {"schema": "mycoscout.far-pixel-report.v1", "version": "far-pixel-v1",
              "status": "completed", "kind": "synthetic-pixel-development-audit", "actualDistanceDetectionRates": None,
              "protocolSha256": digest(out / "protocol.json"), "inputManifestSha256": digest(out / "inputs.json"),
              "sourceScenes": 4, "sourceObservers": len({r["observerId"] for r in records}),
              "images": len(records), "modelInferences": len(records) * 2,
              "frameDimensions": sorted({f"{r['frameWidth']}x{r['frameHeight']}" for r in records}),
              "rows": rows, "negativeControls": negative_metrics,
              "controlsNote": "Four manually inspected clutter-only source crops and four 32-pixel targets. All 24 images share only four original source scenes.",
              "colorNote": "Color-enhanced GroundingDINO and color-only anomaly proposals are measured separately. Color cues never count as mushroom detections.",
              "limitations": p["limitations"], "environment": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "device": "cpu", "threads": 4},
              "runtime": {"modelLoadSeconds": load_seconds, "totalSeconds": time.perf_counter() - started,
                          "meanInferenceSeconds": statistics.mean(r["methods"][n]["seconds"] for r in results for n in ("raw", "color"))},
              "sourceAttributions": [{"id": r["sourceId"], "sourceUrl": r["sourceUrl"], "photoLicense": r["photoLicense"]}
                                     for r in records if r["kind"] == "negative-control"]}
    write(out / "report.json", report)
    print(json.dumps({"report": str(out / "report.json"), "rows": rows, "negativeControls": negative_metrics}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "protocol.json").exists():
        parser.error("Frozen output already exists; use a new version directory")
    write(args.output / "protocol.json", protocol())
    records = prepare(args.output)
    run(args.output, records)


if __name__ == "__main__":
    main()
