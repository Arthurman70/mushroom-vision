"""Fixed video-derived pixel-size stress test; no real distance or training.

Complete saved human boxes supply transformed truth. Whole frames are resized
and centered on a fixed native-size neutral canvas, retaining within-frame
context. Original media, reviews, active models and thresholds are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import urllib.request

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.merge_video_collections import read_collection
from vision.evaluate_video_ai import matched_pairs, sanitized_snapshot, write_new
from vision.testbench import compare
from vision.training_pipeline import iou, score_frames
from vision.video_ai_pass import atomic_json, sha

POSITIVE_IDS = ["youtube-m9aGEh33w0I-043", "youtube-m9aGEh33w0I-005", "youtube-m9aGEh33w0I-022"]
NEGATIVE_ID = "youtube-m9aGEh33w0I-040"
LEVELS = [("native", None), ("64px", 64), ("32px", 32), ("16px", 16),
          ("6p9px", 6.9), ("3p45px", 3.45), ("1p38px", 1.38), ("0p69px", .69)]
METHODS = {"raw": "original", "color": "color", "tiles": "tiles"}
SETTINGS = {"methods": ["original", "color", "tiles"], "tileGrid": 2, "scale": 1, "boxThreshold": .25}


def transform_frame(image, truths, target_pixels, anchor_width):
    """Exact integer-resize geometry; never round fractional truth up to a pixel."""
    if target_pixels is None:
        return image.copy(), [dict(box) for box in truths], {"scaleX": 1, "scaleY": 1, "offset": [0, 0], "contentSize": list(image.size)}
    if not 0 < target_pixels <= anchor_width:
        raise ValueError("Synthetic targets must downscale existing native detail")
    factor = target_pixels / anchor_width
    width, height = max(1, round(image.width * factor)), max(1, round(image.height * factor))
    left, top = (image.width-width)//2, (image.height-height)//2
    canvas = Image.new("RGB", image.size, (127, 127, 127))
    canvas.paste(image.resize((width, height), Image.Resampling.LANCZOS), (left, top))
    sx, sy = width/image.width, height/image.height
    transformed = [{**box, "bbox": [box["bbox"][0]*sx+left, box["bbox"][1]*sy+top,
                                    box["bbox"][2]*sx+left, box["bbox"][3]*sy+top]} for box in truths]
    return canvas, transformed, {"scaleX": sx, "scaleY": sy, "offset": [left, top], "contentSize": [width, height]}


def metric_counts(records, method):
    positives = [record for record in records if record["kind"] == "positive"]
    negatives = [record for record in records if record["kind"] == "negative"]
    def score(items):
        return score_frames([{"id": row["id"], "group": row["sourceGroupId"], "truth": [b["bbox"] for b in row["truth"]],
                              "predictions": row["predictions"][method]} for row in items], .25)
    positive_metrics, negative_metrics = score(positives), score(negatives)
    return {"frames": len(positives), "matched": positive_metrics["truePositives"],
            "targets": positive_metrics["truePositives"] + positive_metrics["falseNegatives"],
            "falsePositives": positive_metrics["falsePositives"], "negativeFrames": len(negatives),
            "negativeFalsePositives": negative_metrics["falsePositives"],
            "negativeFramesWithDetections": sum(bool(row["predictions"][method]) for row in negatives)}


def review_rows(grades, manifest):
    frames = {frame["id"]: frame for frame in manifest["frames"]}
    sources = {source["id"]: source for source in manifest["sources"]}
    result = []
    for frame_id, entry in grades["reviews"].items():
        review = entry.get("review")
        if not review or review.get("humanVerified") is not True:
            continue
        frame = frames[frame_id]
        result.append({"frameId": frame_id, "sourceId": frame["sourceId"], "sourceGroupId": sources[frame["sourceId"]]["sourceGroupId"],
                       "imageSha256": frame["sha256"], "width": frame["width"], "height": frame["height"],
                       "revision": entry["revision"], "status": review["status"], "complete": review["complete"],
                       "humanVerified": True, "truth": [{"id": b["id"], "bbox": b["bboxPixelsXYXY"]} for b in review["boxes"] if b.get("humanVerified") is True]})
    return result


def cached_result(row, cache):
    path = Path(cache) / "frames" / ("main--" + row["frameId"] + ".json")
    result = json.loads(path.read_text(encoding="utf-8"))
    if (result.get("collection") != "main" or result.get("frameId") != row["frameId"] or
            result.get("sourceId") != row["sourceId"] or result.get("imageSha256") != row["imageSha256"] or
            result.get("status") != "completed"):
        raise ValueError("Cached native result does not match frozen human frame")
    return result, sha(path.read_bytes())


def native_summary(rows, cache, positive_only=False):
    scoring, evidence = [], []
    for row in rows:
        result, result_sha = cached_result(row, cache)
        boxes = [box for box in result["boxes"] if box["score"] >= .25]
        pairs = matched_pairs(row["truth"], boxes)
        scoring.append({"id": row["frameId"], "group": row["sourceGroupId"], "truth": [box["bbox"] for box in row["truth"]], "predictions": boxes})
        evidence.append({**row, "predictionFileSha256": result_sha, "matched": len(pairs), "matches": pairs,
                         "predictions": boxes, "falsePositives": None if positive_only else len(boxes)-len(pairs),
                         "positiveOnly": positive_only, "reused": True, "actualDistanceM": None})
    metrics = score_frames(scoring, .25)
    widths = [box["bbox"][2]-box["bbox"][0] for row in rows for box in row["truth"]]
    summary = {"frames": len(rows), "matched": metrics["truePositives"], "targets": metrics["truePositives"]+metrics["falseNegatives"],
               "falsePositives": None if positive_only else metrics["falsePositives"], "precision": None if positive_only else metrics["precision"],
               "sourceVideos": len({row["sourceId"] for row in rows}), "sourceGroups": len({row["sourceGroupId"] for row in rows}),
               "targetWidthPixels": {"min": min(widths) if widths else None, "max": max(widths) if widths else None},
               "scoreThreshold": .25, "iouThreshold": .5, "labelAuthority": "human", "reused": True, "actualDistanceM": None,
               "targetUnit": "saved human annotation region", "positiveOnly": positive_only}
    return summary, evidence


class CountedDetector:
    def __init__(self, detector, cached, progress):
        self.detector, self.cached, self.progress = detector, cached, progress
        self.info = detector.info
        self.box_threshold, self.text_threshold = .25, .25
        self.first = True

    def predict(self, image):
        self.detector.box_threshold, self.detector.text_threshold = self.box_threshold, self.text_threshold
        if self.first and self.cached is not None:
            self.first = False
            if {key: value for key, value in self.cached["model"].items() if key != "threads"} != {key: value for key, value in self.info.items() if key != "threads"}:
                raise ValueError("Cached native detector configuration differs from current detector")
            predictions = [{"bboxPixelsXYXY": box["bbox"], "confidence": box["score"]} for box in self.cached["boxes"]]
            self.progress(True)
            return predictions
        self.first = False
        result = self.detector.predict(image)
        self.progress(False)
        return result


def run(output, collection, cache, station="http://127.0.0.1:8787"):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir(); (output / "predictions").mkdir()
    root, original_manifest, manifest = read_collection(collection)
    with urllib.request.urlopen(station.rstrip("/") + "/api/video-grades", timeout=20) as response:
        grades = json.load(response)
    complete = sanitized_snapshot(grades, manifest, sha(original_manifest))
    reviewed = review_rows(grades, manifest)
    by_review = {row["frameId"]: row for row in reviewed}
    frame_map = {frame["id"]: frame for frame in manifest["frames"]}
    selected = [by_review[frame_id] for frame_id in POSITIVE_IDS]
    negative = by_review[NEGATIVE_ID]
    if any(row["complete"] is not True or row["status"] != "visible" or len(row["truth"]) != 1 for row in selected):
        raise ValueError("Selected positive requires one complete saved human box")
    if negative["complete"] is not True or negative["status"] != "absent" or negative["truth"]:
        raise ValueError("Selected negative requires complete explicit human absence")
    small = [{**row, "truth": [box for box in row["truth"] if 32 <= box["bbox"][2]-box["bbox"][0] < 64]}
             for row in reviewed if row["status"] == "visible" and not row["complete"]]
    small = [row for row in small if row["truth"]]
    native, native_evidence = native_summary(complete["records"], cache)
    native["labelCautions"] = ["Frame 024 has a saved absent label disputed by independent native-image inspection; excluded from the clean negative-control test.",
                               "Frame 038 has one saved annotation region containing two visible fungi; excluded from single-target scale testing. Human grades remain unchanged."]
    small_native, small_evidence = native_summary(small, cache, positive_only=True)
    snapshot = {"schema": "mycoscout.video-distance-human-snapshot.v1", "sourceManifestSha256": sha(original_manifest),
                "complete": complete, "visiblePartial": [row for row in reviewed if row["status"] == "visible" and not row["complete"]]}
    write_new(output / "human-snapshot.json", snapshot)
    native_evidence_file = {"complete": native_evidence, "smallPositiveOnly": small_evidence}
    write_new(output / "native-evidence.json", native_evidence_file)
    protocol = {"schema": "mycoscout.video-distance-protocol.v1", "positiveFrameIds": POSITIVE_IDS, "negativeFrameIds": [NEGATIVE_ID],
                "selection": "Small, middle-range and large native human boxes with independently inspected single-body contents; selection does not use model outcomes",
                "selectedNativeWidths": [row["truth"][0]["bbox"][2]-row["truth"][0]["bbox"][0] for row in selected],
                "sourceVideos": 1, "sourceGroups": 1, "sourceGroupIds": sorted({row["sourceGroupId"] for row in selected}),
                "scoreThreshold": .25, "textThreshold": .25, "iouThreshold": .5, "nmsIou": .5, "tileGrid": 2,
                "methods": SETTINGS["methods"], "tilesIncludeRawFullFrame": True, "tileOverlap": .2,
                "targetPixels": [target for _, target in LEVELS], "negativeConditions": ["native", "6p9px"],
                "negativeScaleReferenceWidth": selected[1]["truth"][0]["bbox"][2]-selected[1]["truth"][0]["bbox"][0],
                "resize": "Whole original frame, PIL Lanczos, rounded integer dimensions; exact realized XY scale maps all human truth",
                "canvas": "Original native dimensions, centered content on fixed neutral RGB 127 padding", "color": "Existing fixed luminance/chroma residual enhancement after resizing",
                "plannedModelForwards": 156, "plannedReusedNativeForwards": 4, "plannedNewForwards": 152,
                "illustrativeDistancesOnly": [{"meters": 10, "targetPixels": 6.9}, {"meters": 20, "targetPixels": 3.45},
                                             {"meters": 50, "targetPixels": 1.38}, {"meters": 100, "targetPixels": .69}],
                "distanceAssumption": "Prior illustrative 5 cm mushroom projection; no video lens, physical size or distance calibration exists",
                "actualDistanceM": None, "trainingPerformed": False, "modelActivation": False,
                "excludedControlFrames": [{"frameId": "youtube-m9aGEh33w0I-024", "reason": "Saved human absence disputed by independent native inspection"},
                    {"frameId": "youtube-m9aGEh33w0I-041", "reason": "Near-duplicate view of 040; do not count as independent"},
                    {"frameId": "youtube-m9aGEh33w0I-042", "reason": "Motion/occlusion makes clean absence uncertain"},
                    {"frameId": "youtube-m9aGEh33w0I-038", "reason": "One human region encloses two fungi; unsuitable single-body pixel calibration"}],
                "nativeBaselineCaveat": "Native 19 matches saved annotation regions, including a merged pair and disputed absence; not 17 distinct mushroom truth instances",
                "snapshotSha256": sha((output / "human-snapshot.json").read_bytes())}
    write_new(output / "protocol.json", protocol)
    report = {"schema": "mycoscout.video-distance-report.v1", "status": "running", "generatedAt": datetime.now(timezone.utc).isoformat(),
              "sample": {"positiveFrames": 3, "negativeFrames": 1, "targets": 3, "sourceVideos": 1, "sourceGroups": 1, "labelAuthority": "human"},
              "progress": {"completed": 0, "total": 156, "unit": "model-forwards", "newCompleted": 0, "reused": 0},
              "rows": [{"id": name, "label": "Native video frames" if target is None else "Requested " + str(target) + " pixels", "native": target is None,
                        "targetPixels": target, "raw": None, "color": None, "tiles": None} for name, target in LEVELS],
              "protocol": protocol, "nativeBaseline": native, "nativeSmallPositiveOnly": small_native,
              "limitations": ["Synthetic pixel sizes are not measured 10/20/50/100 metre detection rates.",
                  "Three correlated morel-like targets come from one video/creator group; this is a small stress test, not population accuracy.",
                  "The entire frame shrinks on neutral padding: local context is retained, but scenery scale, padding and model resizing differ from real camera motion.",
                  "Fractional subpixel boxes are geometric projections; resizing cannot preserve or recover reliable fungal texture below one pixel.",
                  "Tiles and color enhancement add no optical detail and can create false positives; strict IoU 0.5 is especially unforgiving for subpixel boxes.",
                  "Negative controls exist only at native and 6.9 px-matched scene scale, using one correlated background frame.",
                  "The 3 genuine 41–61 px human partial boxes are white lawn mushrooms from another evaluation-only source, not morels; only positive recall is valid.",
                  "Native 19 cached annotation matching includes a merged pair and a disputed absent label; its 17 regions are not 17 confirmed individual mushrooms.",
                  "No training, threshold tuning, live-model activation, human-grade changes or flight commands occur."],
              "actualDistanceM": None, "actuationEnabled": False, "records": []}
    atomic_json(output / "report.json", report)
    from adapters.scout_frame import LocalDetector
    detector = LocalDetector(ROOT / "models/grounding-dino-tiny", device="cpu", threads=8, box_threshold=.25, text_threshold=.25)
    started = time.perf_counter()
    def progress(reused):
        report["progress"]["completed"] += 1
        report["progress"]["reused" if reused else "newCompleted"] += 1
        report["elapsedSeconds"] = time.perf_counter()-started
        atomic_json(output / "report.json", report)
    for name, target in LEVELS:
        views = [(row, "positive") for row in selected]
        if name in {"native", "6p9px"}:
            views.append((negative, "negative"))
        condition_records = []
        for row, kind in views:
            frame = frame_map[row["frameId"]]
            source_path = root / frame["path"]
            if sha(source_path.read_bytes()) != row["imageSha256"]:
                raise ValueError("Original frame changed after snapshot")
            with Image.open(source_path) as source:
                original = source.convert("RGB")
            anchor = row["truth"][0]["bbox"][2]-row["truth"][0]["bbox"][0] if row["truth"] else protocol["negativeScaleReferenceWidth"]
            image, truth, geometry = transform_frame(original, row["truth"], target, anchor)
            key = row["frameId"] + "--" + name
            image_path = output / "images" / (key + ".png")
            image.save(image_path)
            cached, cached_sha = cached_result(row, cache) if target is None else (None, None)
            wrapped = CountedDetector(detector, cached, progress)
            result = compare(image, sha(image_path.read_bytes()), SETTINGS, wrapped)
            write_new(output / "predictions" / (key + ".json"), result)
            predictions = {name: [{"id": box["id"], "bbox": box["bboxPixelsXYXY"], "score": box["confidence"]}
                                  for box in next(method for method in result["methods"] if method["id"] == identifier)["candidates"]]
                           for name, identifier in METHODS.items()}
            detail = {"id": key, "condition": name, "frameId": row["frameId"], "sourceId": row["sourceId"], "sourceGroupId": row["sourceGroupId"],
                      "imageSha256": row["imageSha256"], "reviewRevision": row["revision"], "kind": kind, "humanVerifiedTruth": True,
                      "truth": truth, "originalTruth": row["truth"], "targetPixels": target,
                      "realizedTargetWidths": [box["bbox"][2]-box["bbox"][0] for box in truth], "transform": geometry,
                      "imagePath": "images/" + image_path.name, "derivedImageSha256": sha(image_path.read_bytes()),
                      "cachedNativeRawSha256": cached_sha, "synthetic": target is not None, "actualDistanceM": None,
                      "predictions": predictions, "nearestPredictions": {method: [{"truthId": box["id"],
                          "bestIoU": max((iou(box["bbox"], pred["bbox"]) for pred in preds), default=0)} for box in truth]
                          for method, preds in predictions.items()}}
            condition_records.append(detail); report["records"].append(detail)
            atomic_json(output / "report.json", report)
        summary = next(row for row in report["rows"] if row["id"] == name)
        summary.update({method: metric_counts(condition_records, method) for method in METHODS})
        summary["realizedTargetWidths"] = [width for row in condition_records for width in row["realizedTargetWidths"]]
        summary["realizedTargetWidthPixels"] = {"min": min(summary["realizedTargetWidths"]), "max": max(summary["realizedTargetWidths"])}
        summary["complete"] = True
        atomic_json(output / "report.json", report)
        print(json.dumps({"condition": name, "progress": report["progress"], "raw": summary["raw"], "color": summary["color"], "tiles": summary["tiles"]}), flush=True)
    if report["progress"] != {"completed": 156, "total": 156, "unit": "model-forwards", "newCompleted": 152, "reused": 4}:
        raise ValueError("Detector forward count differs from frozen protocol")
    if (Path(collection) / "manifest.json").read_bytes() != original_manifest:
        raise ValueError("Source collection changed during experiment")
    report.update(status="completed", completedAt=datetime.now(timezone.utc).isoformat(), elapsedSeconds=time.perf_counter()-started,
                  sourceManifestUnchanged=True, originalFramesUnchanged=all(sha((root / frame_map[row["frameId"]]["path"]).read_bytes()) == row["imageSha256"] for row in selected + [negative]))
    atomic_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/video-distance-v1")
    parser.add_argument("--collection", type=Path, default=ROOT / "data/video-discovery-v1")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/ai-video-review-v1")
    args = parser.parse_args()
    run(args.output, args.collection, args.cache)


if __name__ == "__main__":
    main()
