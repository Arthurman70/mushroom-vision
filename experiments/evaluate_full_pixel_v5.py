"""Frozen tiny-footprint stress test: unchanged DINO, raw plus overlapping tiles.

Whole scenes shrink onto a fixed 1920x1080 gray canvas at 16x fractional coverage.
This is development evidence, not a calibrated camera-distance or flight test.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import multiprocessing
from pathlib import Path
import sys
import time
from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.acquire_public import write_json
from vision.metrics import iou
from vision.testbench import compare, tile_boxes

OUT = ROOT / "experiments/results/full-pixel-v5"
CANVAS = (1920, 1080)
SS = 16
LEVELS = [("native", None), ("64px", 64.), ("32px", 32.), ("16px", 16.),
          ("6p9px", 6.9), ("3p45px", 3.45), ("1p38px", 1.38), ("0p69px", .69)]
SETTINGS = {"methods": ["original", "tiles"], "tileGrid": 2, "scale": 1, "boxThreshold": .25}
COHORTS = {"human-video": "Saved human video references", "ai-photo": "Incomplete AI photo references"}
LIMITATIONS = [
    "Previously inspected reference regions are a development regression, not independent field accuracy.",
    "Three human targets share one video; nine AI photo targets remain approximate and incomplete, including clusters as well as individual bodies.",
    "All synthetic views shrink the whole source onto a fixed 1920x1080 RGB127 canvas with 16x coverage integration. They are never re-enlarged to native footprint.",
    "Gray borders, scene edges, fixed placement, resampling and model resizing differ from real camera motion; no new optical detail is created.",
    "Placement is SHA-derived before inference; focal positions are clamped only to keep the entire scene inside the canvas. Actual tile coverage is recorded.",
    "Fractional widths and heights describe projected annotation geometry, not guaranteed recoverable fungal texture. A subpixel width may have a height above one pixel.",
    "10/20/50/100 metres are illustrative only: widthPixels = 1380px * 0.05m / distance. The sources have no measured lens, organism width or range.",
    "The old round-4 downsample-then-re-enlarge test is a separate information-loss experiment and is not the small-footprint curve reported here.",
    "Each region can match at most one prediction. Extra overlaps and other unmatched boxes are reported; incomplete AI positives cannot establish false-positive rates.",
    "Five negative controls are native-only: one saved human absence and four earlier AI absence opinions. They establish no distance-dependent false-positive rate.",
    "Two isolated CPU workers use four Torch threads each. Tiled timing includes its shared raw context pass; method times must not be added as elapsed runtime.",
    "Concurrent-worker latency includes CPU contention and is not directly comparable to earlier sequential-worker timing.",
    "No training, score tuning, active-model changes, source-label changes or aircraft commands occur.",
]


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text(encoding="utf-8"))
def digest(path):
    with Path(path).open("rb") as stream: return hashlib.file_digest(stream, "sha256").hexdigest()
def rel(path): return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def checked(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file(): raise ValueError("Input outside project or not a file")
    return path
def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write("\n")
def versions(): return {k: importlib.metadata.version(k) for k in ("torch", "torchvision", "transformers", "Pillow")}


def position_for(identity):
    hashed = hashlib.sha256(identity.encode("utf-8")).digest()
    grid = (.25, .50, .75)
    return [grid[hashed[0] % 3], grid[hashed[1] % 3]]


def valid_box(box, size):
    if (not isinstance(box, (list, tuple)) or len(box) != 4 or
        any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
        not 0 <= box[0] < box[2] <= size[0] or not 0 <= box[1] < box[3] <= size[1]):
        raise ValueError("Invalid selected original-coordinate region")


def fractional_canvas(source, box, target_width, position):
    """Adapt fractional_scene's 16x local integration without a huge full canvas."""
    valid_box(box, source.size)
    if type(target_width) not in (int, float) or not math.isfinite(target_width) or not 0 < target_width <= box[2] - box[0]:
        raise ValueError("Target must shrink existing native detail")
    if len(position) != 2 or any(v not in (.25, .5, .75) for v in position): raise ValueError("Unfrozen placement")
    scale = target_width / (box[2] - box[0])
    sw, sh = max(1, round(source.width * scale * SS)), max(1, round(source.height * scale * SS))
    if sw > CANVAS[0] * SS or sh > CANVAS[1] * SS: raise ValueError("Whole reduced scene exceeds fixed canvas")
    sx, sy = sw / SS / source.width, sh / SS / source.height
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    requested_x = round((CANVAS[0] * position[0] - cx * sx) * SS)
    requested_y = round((CANVAS[1] * position[1] - cy * sy) * SS)
    x, y = min(max(0, requested_x), CANVAS[0] * SS - sw), min(max(0, requested_y), CANVAS[1] * SS - sh)
    left, top, right, bottom = math.floor(x / SS), math.floor(y / SS), math.ceil((x + sw) / SS), math.ceil((y + sh) / SS)
    patch = source.resize((sw, sh), Image.Resampling.LANCZOS)
    local = Image.new("RGB", ((right - left) * SS, (bottom - top) * SS), (127, 127, 127))
    local.paste(patch, (x - left * SS, y - top * SS))
    result = Image.new("RGB", CANVAS, (127, 127, 127))
    result.paste(local.resize((right - left, bottom - top), Image.Resampling.BOX), (left, top))
    projected = [box[0] * sx + x / SS, box[1] * sy + y / SS, box[2] * sx + x / SS, box[3] * sy + y / SS]
    return result, projected, {"sourceSize": list(source.size), "canvasSize": list(CANVAS), "supersampling": SS,
        "requestedTargetWidthPixels": target_width, "realizedTargetWidthPixels": projected[2] - projected[0],
        "realizedTargetHeightPixels": projected[3] - projected[1], "sceneScaleXY": [sx, sy],
        "sceneBoundsPixelsXYXY": [x / SS, y / SS, (x + sw) / SS, (y + sh) / SS],
        "requestedFocalPositionNormalized": position, "focalPositionClamped": x != requested_x or y != requested_y,
        "realizedFocalPositionNormalized": [(projected[0] + projected[2]) / (2 * CANVAS[0]), (projected[1] + projected[3]) / (2 * CANVAS[1])],
        "patchSupersampledSize": [sw, sh], "sceneReEnlarged": False, "actualDistanceM": None}


def pass_geometry(size, box):
    from transformers.models.grounding_dino.image_processing_grounding_dino import get_size_with_aspect_ratio
    passes = [("context", [0, 0, *size])] + [(f"tile-{i + 1}", list(b)) for i, b in enumerate(tile_boxes(*size, 2))]
    result = []
    for name, bounds in passes:
        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        ih, iw = get_size_with_aspect_ratio((height, width), 800, 1333)
        visible_width = 0. if box is None else max(0., min(box[2], bounds[2]) - max(box[0], bounds[0]))
        visible_height = 0. if box is None else max(0., min(box[3], bounds[3]) - max(box[1], bounds[1]))
        area = visible_width * visible_height
        fraction = None if box is None else area / ((box[2] - box[0]) * (box[3] - box[1]))
        result.append({"kind": name, "boundsPixelsXYXY": bounds, "inputWidth": iw, "inputHeight": ih,
            "processorScaleXY": [iw / width, ih / height], "selectedTargetCoverageFraction": fraction,
            "selectedTargetInputWidthPixels": None if box is None else (visible_width * iw / width if area else 0.),
            "selectedTargetInputHeightPixels": None if box is None else (visible_height * ih / height if area else 0.)})
    return {"passes": result, "intersectingTileCount": None if box is None else sum(p["selectedTargetCoverageFraction"] > 0 for p in result[1:]),
            "fullyCoveringTileCount": None if box is None else sum(p["selectedTargetCoverageFraction"] >= 1 - 1e-9 for p in result[1:])}


def score(box, candidates, kind, authority):
    if box is None:
        return {"hits": None, "targetCount": 0, "unmatchedPositiveBoxes": None, "extraOverlappingAlerts": None,
                "negativeAlerts": len(candidates), "negativeHasAlerts": bool(candidates),
                "humanNegativeFalseAlerts": len(candidates) if authority == "human" else None,
                "aiNegativeOpinionAlerts": len(candidates) if authority == "ai" else None}
    overlaps = [iou(box, p["bbox"]) for p in candidates]
    eligible = [i for i, overlap in enumerate(overlaps) if overlap >= .5]
    selected = max(eligible, key=lambda i: (candidates[i]["score"], overlaps[i], -i)) if eligible else None
    return {"hits": int(selected is not None), "targetCount": 1, "matchedPredictionIndex": selected,
            "bestIoU": max(overlaps, default=0.), "unmatchedPositiveBoxes": len(candidates) - int(selected is not None),
            "extraOverlappingAlerts": max(0, len(eligible) - 1),
            "humanCompletePositiveFalseAlerts": len(candidates) - int(selected is not None) if authority == "human" else None,
            "negativeAlerts": None, "negativeHasAlerts": None}


def aggregate(records):
    result = []
    for cohort, label in COHORTS.items():
        rows = [r for r in records if r["cohort"] == cohort]
        conditions = []
        for level, pixels in LEVELS:
            level_rows = [r for r in rows if r["conditionId"] == level]
            for method in ("raw", "tiles"):
                available = [r for r in level_rows if method in r.get("methods", {})]
                if not available: continue
                positives = [r for r in available if not r["isNegativeControl"]]
                negatives = [r for r in available if r["isNegativeControl"]]
                metric = lambda row: row["methods"][method]["metrics"]
                conditions.append({"id": level, "mode": method, "targetPixels": pixels,
                    "hits": sum(metric(r)["hits"] for r in positives), "targetCount": len(positives),
                    "unmatchedPositiveBoxes": sum(metric(r)["unmatchedPositiveBoxes"] for r in positives),
                    "extraOverlappingAlerts": sum(metric(r)["extraOverlappingAlerts"] for r in positives),
                    "negativeAlertCount": sum(metric(r)["negativeAlerts"] for r in negatives) if negatives else None,
                    "negativeFramesWithAlerts": sum(metric(r)["negativeHasAlerts"] for r in negatives) if negatives else None,
                    "negativeFrameCount": len(negatives) if negatives else None,
                    "meanSecondsPerFrame": sum(r["methods"][method]["seconds"] for r in available) / len(available),
                    "measurementFrames": len(available), "realizedTargetWidths": [r["realizedTargetWidthPixels"] for r in positives],
                    "realizedTargetHeights": [r["realizedTargetHeightPixels"] for r in positives]})
        result.append({"id": cohort, "label": label, "referenceAuthority": "human" if cohort == "human-video" else "ai-incomplete",
            "targetCount": 3 if cohort == "human-video" else 9, "negativeFrameCount": 1 if cohort == "human-video" else 4, "conditions": conditions})
    return result


def validate_references(refs):
    expected = {("human", False): 3, ("ai", False): 9, ("human", True): 1, ("ai", True): 4}
    if Counter((r["referenceAuthority"], r["isNegativeControl"]) for r in refs) != expected:
        raise ValueError("Expected three human and nine AI positives, one human and four AI negatives")
    if len({r["frameId"] for r in refs}) != len(refs): raise ValueError("Duplicate reference frame")
    for ref in refs:
        expected_complete = ref["referenceAuthority"] == "human" or ref["isNegativeControl"]
        if ref["referenceComplete"] is not expected_complete: raise ValueError("Unexpected reference completeness")
        if (ref["originalTargetPixelsXYXY"] is None) != ref["isNegativeControl"]:
            raise ValueError("Target/absence reference conflict")


def references():
    ai_path = ROOT / "experiments/results/new-fungi-v4/evaluation/protocol.json"
    old_path = ROOT / "experiments/results/ground-tiles-v1/protocol.json"
    ai, old = read(ai_path), read(old_path)
    refs = []
    for row in ai["references"]:
        if row["kind"] != "new_ai_selected_target": continue
        refs.append({"frameId": row["id"], "cohort": "ai-photo", "referenceAuthority": "ai", "isNegativeControl": False,
            "path": row["path"], "imageSha256": row["sha256"], "width": row["width"], "height": row["height"],
            "sourceGroupId": row["sourceGroupId"], "originalTargetPixelsXYXY": row["selectedTargetPixelsXYXY"],
            "referenceComplete": False, "notes": row["referenceEvidence"]["notes"], "sourceClassId": row["sourceClassId"],
            "targetUnit": "selected AI region; can be a body or cluster", "referenceEvidence": row["referenceEvidence"]})
    for row in old["references"]:
        negative = row["referenceStatus"] == "absent"
        if not negative and (row["referenceAuthority"] != "human" or len(row["truth"]) != 1): raise ValueError("Expected one human region")
        refs.append({"frameId": row["frameId"], "cohort": "human-video" if row["referenceAuthority"] == "human" else "ai-photo",
            "referenceAuthority": row["referenceAuthority"], "isNegativeControl": negative, "path": row["path"],
            "imageSha256": row["sha256"], "width": row["width"], "height": row["height"], "sourceGroupId": row["sourceGroupId"],
            "originalTargetPixelsXYXY": None if negative else row["truth"][0]["bbox"], "referenceComplete": row["referenceComplete"],
            "notes": "Earlier complete human annotation" if row["referenceAuthority"] == "human" else row["referenceEvidence"].get("evidence", "Earlier AI absence opinion"),
            "targetUnit": "saved human annotation region" if not negative else "whole native absence opinion", "referenceEvidence": row["referenceEvidence"]})
    validate_references(refs)
    for row in refs:
        path = checked(row["path"])
        if digest(path) != row["imageSha256"]: raise ValueError("Reference image bytes changed")
        with Image.open(path) as opened: size = ImageOps.exif_transpose(opened).size
        if size != (row["width"], row["height"]): raise ValueError("Reference orientation/dimensions changed")
        if row["originalTargetPixelsXYXY"] is not None: valid_box(row["originalTargetPixelsXYXY"], size)
        row["fixedFocalPositionNormalized"] = position_for(row["frameId"])
    return refs, [ai_path, old_path], old["expectedModel"]


def prepare(output=OUT):
    output = Path(output).resolve()
    if output.exists(): raise FileExistsError("Choose a new immutable output directory")
    if not output.is_relative_to(ROOT): raise ValueError("Output outside project")
    refs, paths, expected = references()
    paths += [Path(__file__), ROOT / "vision/testbench.py", ROOT / "vision/metrics.py", ROOT / "vision/far_color.py",
              ROOT / "adapters/scout_frame.py", ROOT / "adapters/detect_fungi.py"]
    paths += [p for p in (ROOT / "models/grounding-dino-tiny").iterdir() if p.is_file()]
    inputs = {rel(p): digest(p) for p in paths}
    output.mkdir(parents=True); (output / "images").mkdir(); (output / "predictions").mkdir()
    jobs = []
    for level, pixels in LEVELS:
        for ref in refs:
            if pixels is not None and ref["isNegativeControl"]: continue
            inputs[ref["path"]] = ref["imageSha256"]
            with Image.open(checked(ref["path"])) as opened: image = ImageOps.exif_transpose(opened).convert("RGB")
            box = ref["originalTargetPixelsXYXY"]
            if pixels is None:
                path, projected = checked(ref["path"]), box
                geometry = {"sourceSize": list(image.size), "canvasSize": list(image.size), "sceneScaleXY": [1., 1.],
                    "sceneBoundsPixelsXYXY": [0., 0., *image.size], "supersampling": None, "sceneReEnlarged": False,
                    "actualDistanceM": None}
            else:
                image, projected, geometry = fractional_canvas(image, box, pixels, ref["fixedFocalPositionNormalized"])
                path = output / "images" / f"{ref['frameId']}--{level}.png"
                image.save(path, format="PNG")
            geometry["processorAndTileCoverage"] = pass_geometry(image.size, projected)
            image_hash = digest(path); inputs[rel(path)] = image_hash
            jobs.append({"order": len(jobs), "frameId": ref["frameId"], "cohort": ref["cohort"],
                "referenceAuthority": ref["referenceAuthority"], "referenceComplete": ref["referenceComplete"],
                "isNegativeControl": ref["isNegativeControl"], "conditionId": level, "targetPixels": pixels,
                "imagePath": rel(path), "derivedImageSha256": image_hash, "sourceImageSha256": ref["imageSha256"],
                "width": image.width, "height": image.height, "geometry": geometry,
                "originalTargetPixelsXYXY": box, "targetPixelsXYXY": projected,
                "realizedTargetWidthPixels": None if projected is None else projected[2] - projected[0],
                "realizedTargetHeightPixels": None if projected is None else projected[3] - projected[1],
                "synthetic": pixels is not None, "actualDistanceM": None, "methods": {}})
            print(json.dumps({"preparedScene": len(jobs), "frameId": ref["frameId"], "level": level}), flush=True)
    if len(jobs) != 101: raise ValueError("Expected 101 fixed scenes")
    protocol = {"schema": "mushroom-vision.full-pixel-v5-protocol.v1", "status": "geometry-review-pending", "createdAt": now(),
        "references": refs, "jobs": jobs, "levels": [{"id": a, "targetPixels": b} for a, b in LEVELS],
        "settings": SETTINGS, "expectedModel": expected, "inputSha256": inputs, "softwareVersions": versions(),
        "device": "cpu", "workers": 2, "torchThreadsPerWorker": 4, "plannedModelForwards": 505,
        "sharedRawContext": True, "canvasSize": list(CANVAS), "supersampling": SS, "scoreThreshold": .25,
        "iouThreshold": .5, "nmsIou": .5, "focalLengthPixels": 1380, "assumedTargetWidthMetres": .05,
        "illustrativeDistancesOnly": [{"metres": d, "targetPixels": 1380 * .05 / d} for d in (10, 20, 50, 100)],
        "trainingPerformed": False, "weightsChanged": False, "actuationEnabled": False, "actualDistanceM": None, "limitations": LIMITATIONS}
    new_json(output / "draft-protocol.json", protocol)
    # Clearly marked geometry preview; exact unannotated inputs remain separate.
    selected = [j for j in jobs if (j["frameId"], j["conditionId"]) in {
        ("youtube-m9aGEh33w0I-043", "64px"), ("inat-photo-757471312", "0p69px"),
        ("inat-photo-753536713", "6p9px"), ("inat-photo-340813928", "32px")}]
    sheet = Image.new("RGB", (960, 320 * len(selected)), "white"); draw = ImageDraw.Draw(sheet)
    for i, job in enumerate(selected):
        with Image.open(checked(job["imagePath"])) as opened: thumb = opened.convert("RGB").resize((480, 270))
        sheet.paste(thumb, (0, i * 320 + 35))
        b = job["targetPixelsXYXY"]; cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        draw.text((5, i * 320 + 5), f"{job['frameId']} {job['conditionId']}; target {b[2]-b[0]:.4f} x {b[3]-b[1]:.4f}px", fill="black")
        draw.ellipse((cx / 4 - 4, i * 320 + 35 + cy / 4 - 4, cx / 4 + 4, i * 320 + 35 + cy / 4 + 4), outline="red", width=1)
        with Image.open(checked(job["imagePath"])) as opened:
            left, top = max(0, min(1920 - 96, round(cx - 48))), max(0, min(1080 - 96, round(cy - 48)))
            zoom = opened.crop((left, top, left + 96, top + 96)).resize((270, 270), Image.Resampling.NEAREST)
        sheet.paste(zoom, (540, i * 320 + 35)); draw.text((815, i * 320 + 40), "96px crop\nnearest-neighbor\nQA only", fill="black")
    sheet.save(output / "geometry-preview.png")
    print(json.dumps({"prepared": rel(output), "scenes": len(jobs), "forwards": 505, "status": "review geometry then freeze"}), flush=True)
    return protocol


def verify(protocol):
    for path, expected in protocol["inputSha256"].items():
        if digest(checked(path)) != expected: raise ValueError("Frozen input changed: " + path)
    if versions() != protocol["softwareVersions"]: raise ValueError("Software versions changed")


def freeze(output=OUT):
    output = Path(output); protocol = read(output / "draft-protocol.json"); verify(protocol)
    protocol.update(status="frozen", frozenAt=now(), geometryReview={"reviewerKind": "ai", "completedBeforeInference": True,
        "scope": "Geometry tests, source/patch boundaries, exact fractional dimensions and generated preview inspected before inference"})
    new_json(output / "protocol.json", protocol)
    new_json(output / "protocol-sha256.json", {"sha256": digest(output / "protocol.json")})
    print(json.dumps({"frozen": True, "protocolSha256": digest(output / "protocol.json"), "forwards": 505}), flush=True)


_DETECTOR = None
def worker_init():
    global _DETECTOR
    from adapters.scout_frame import LocalDetector
    _DETECTOR = LocalDetector(ROOT / "models/grounding-dino-tiny", device="cpu", threads=4, box_threshold=.25, text_threshold=.25)


def evaluate_job(job):
    with Image.open(checked(job["imagePath"])) as opened: image = ImageOps.exif_transpose(opened).convert("RGB")
    if digest(checked(job["imagePath"])) != job["derivedImageSha256"]: raise ValueError("Scene changed before inference")
    result = compare(image, job["derivedImageSha256"], SETTINGS, _DETECTOR)
    if result["modelPasses"] != 5: raise ValueError("Expected five unique shared-context forwards")
    methods = {}
    for method in result["methods"]:
        name = "raw" if method["id"] == "original" else "tiles"
        candidates = [{"bbox": p["bboxPixelsXYXY"], "score": p["confidence"], "sourcePass": p["sourcePass"]} for p in method["candidates"]]
        methods[name] = {"candidates": candidates, "metrics": score(job["targetPixelsXYXY"], candidates, job["isNegativeControl"], job["referenceAuthority"]),
            "seconds": method["seconds"], "modelPasses": method["passes"], "sharedContextPasses": method["reusedPasses"],
            "passTimings": method.get("passTimings", [])}
    return {**job, "methods": methods, "uniqueModelForwards": 5, "sceneWallSeconds": result["seconds"]}, result


def run(output=OUT):
    output = Path(output).resolve(); protocol = read(output / "protocol.json"); phash = digest(output / "protocol.json")
    if protocol.get("status") != "frozen" or read(output / "protocol-sha256.json")["sha256"] != phash: raise ValueError("Protocol is not frozen or changed")
    if (output / "report.json").exists(): raise FileExistsError("Evaluation already started; no overwrite or silent resume")
    new_json(output / "run-claim.json", {"startedAt": now(), "protocolSha256": phash})
    verify(protocol)
    report = {"schema": "mushroom-vision.full-pixel-v5-report.v1", "status": "running", "startedAt": now(),
        "protocolSha256": phash, "frames": protocol["references"], "records": [], "cohorts": [],
        "progress": {"completed": 0, "total": 505, "scenesCompleted": 0, "scenesTotal": 101},
        "workers": 2, "torchThreadsPerWorker": 4, "trainingPerformed": False, "weightsChanged": False,
        "actualDistanceM": None, "limitations": LIMITATIONS}
    write_json(output / "report.json", report); started = time.perf_counter(); pending = []
    executor = ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context("spawn"), initializer=worker_init)
    try:
        pending = [executor.submit(evaluate_job, job) for job in protocol["jobs"]]
        for future in as_completed(pending):
            record, prediction = future.result()
            for key, expected in protocol["expectedModel"].items():
                if prediction["model"].get(key) != expected: raise ValueError("Model identity changed: " + key)
            path = output / "predictions" / f"{record['frameId']}--{record['conditionId']}.json"
            new_json(path, prediction)
            record.update(predictionPath=rel(path), predictionSha256=digest(path))
            report["records"].append(record); report["records"].sort(key=lambda r: r["order"])
            report["progress"]["completed"] += 5; report["progress"]["scenesCompleted"] += 1
            report["cohorts"] = aggregate(report["records"])
            report.update(elapsedWallSeconds=time.perf_counter() - started,
                summedSceneInferenceSeconds=sum(r["sceneWallSeconds"] for r in report["records"]), model=prediction["model"])
            write_json(output / "report.json", report)
            print(json.dumps({"frameId": record["frameId"], "level": record["conditionId"], "progress": report["progress"],
                "rawHits": record["methods"]["raw"]["metrics"]["hits"], "tileHits": record["methods"]["tiles"]["metrics"]["hits"],
                "sceneSeconds": round(record["sceneWallSeconds"], 2), "elapsedWallSeconds": round(report["elapsedWallSeconds"], 1)}), flush=True)
        verify(protocol)
        if digest(output / "protocol.json") != phash or report["progress"]["completed"] != 505: raise ValueError("Final protocol/progress integrity failed")
        report.update(status="completed", completedAt=now(), inputIntegrityVerified=True, elapsedWallSeconds=time.perf_counter() - started)
    except Exception as exc:
        for future in pending: future.cancel()
        report.update(status="failed", failedAt=now(), error={"type": type(exc).__name__, "message": str(exc)})
        write_json(output / "report.json", report)
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    write_json(output / "report.json", report)
    print(json.dumps({"completed": True, "report": rel(output / "report.json"), "elapsedWallSeconds": report["elapsedWallSeconds"], "cohorts": report["cohorts"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("action", choices=("prepare", "freeze", "run")); parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(); {"prepare": prepare, "freeze": freeze, "run": run}[args.action](args.output)
