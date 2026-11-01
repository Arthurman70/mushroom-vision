"""Frozen GroundingDINO evaluation against selected, explicitly AI-only references.

Native photos and whole-frame synthetic information loss are separate conditions.
No annotations are created here; partial references cannot measure false positives.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
import time

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.metrics import iou
from vision.acquire_public import write_json

SCHEMA = "mushroom-vision.new-fungi-v4-protocol.v1"
METHODS = [
    {"id": "native", "targetPixels": None, "method": "original", "passes": 1},
    {"id": "low32", "targetPixels": 32, "method": "original", "passes": 1},
    {"id": "low16", "targetPixels": 16, "method": "original", "passes": 1},
    {"id": "low32tiles", "targetPixels": 32, "method": "tiles", "passes": 5},
]
LIMITATIONS = [
    "Selected boxes are manually observed AI references, not human ground truth; image annotations are incomplete.",
    "The selected-target hit criterion is IoU >= 0.5. Unmatched alerts are not automatically false positives.",
    "Native rows without a selected target are qualitative checks only, with no accuracy score.",
    "Prior AI absence references are reported separately and remain AI opinions, not human negatives.",
    "Low-pixel variants reduce the entire image and resize it back with LANCZOS; no crop/paste or selective target modification.",
    "Synthetic information loss has no measured distance in metres and is not new independent field evidence.",
    "Weights, prompt, thresholds, images, references and code are frozen before inference; no training or activation occurs.",
]


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def checked_path(value):
    path = (ROOT / value).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError("Input must be an existing file inside the project")
    return path


def new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "transformers", "Pillow")}


def validate_target(box, size):
    if box is None:
        return None
    if (not isinstance(box, list) or len(box) != 4 or
        any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
        not 0 <= box[0] < box[2] <= size[0] or not 0 <= box[1] < box[3] <= size[1]):
        raise ValueError("Selected target must be a finite, in-frame original-coordinate XYXY box")
    if box[2] - box[0] <= 32:
        raise ValueError("Selected target must exceed 32 native pixels to test both 32px and 16px information loss")
    return list(map(float, box))


def degraded_image(image, box, requested_pixels):
    if requested_pixels not in (16, 32):
        raise ValueError("Only predeclared 16px and 32px conditions are supported")
    box = validate_target(box, image.size)
    if box is None:
        raise ValueError("Synthetic target-size views need a selected target")
    scale = requested_pixels / (box[2] - box[0])
    reduced_size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    reduced = image.resize(reduced_size, Image.Resampling.LANCZOS)
    restored = reduced.resize(image.size, Image.Resampling.LANCZOS)
    return restored, {"synthetic": True, "requestedTargetWidthPixels": requested_pixels,
        "sourceTargetWidthPixels": box[2] - box[0], "scaleRequested": scale,
        "reducedWidth": reduced_size[0], "reducedHeight": reduced_size[1],
        "effectiveTargetWidthPixels": (box[2] - box[0]) * reduced_size[0] / image.width,
        "restoredWidth": image.width, "restoredHeight": image.height,
        "transform": "Whole-frame LANCZOS downsample then LANCZOS restore to original dimensions",
        "cropOrPaste": False, "actualDistanceM": None}


def methods_for(reference):
    return METHODS if reference.get("selectedTargetPixelsXYXY") is not None else METHODS[:1]


def score(reference, candidates):
    box = reference.get("selectedTargetPixelsXYXY")
    selected = box is not None
    overlaps = [iou(c["bboxPixelsXYXY"], box) for c in candidates] if selected else []
    match_indexes = [i for i, overlap in enumerate(overlaps) if overlap >= .5]
    best = max(overlaps, default=0.) if selected else None
    negative = reference["kind"] == "prior_ai_negative"
    return {"predictions": len(candidates), "selectedTargetEvaluated": selected,
        "selectedTargetHit": bool(match_indexes) if selected else None,
        "bestSelectedTargetIoU": best, "overlappingPredictionIndexes": match_indexes,
        "unmatchedAlerts": sum(overlap < .5 for overlap in overlaps) if selected else None,
        "aiReferenceNegativeAlerts": len(candidates) if negative else None,
        "qualitativeOnly": not selected and not negative, "humanVerified": False,
        "falsePositiveRate": None, "criterion": "IoU >= 0.5 against one selected AI reference; other image content is unannotated"}


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        for name, result in record.get("methods", {}).items():
            groups[name].append((record, result))
    result = []
    for method in METHODS:
        pairs = groups.get(method["id"], [])
        if not pairs:
            continue
        selected = [p for p in pairs if p[1]["metrics"]["selectedTargetEvaluated"]]
        negatives = [p for p in pairs if p[0]["kind"] == "prior_ai_negative"]
        seconds = sum(p[1]["seconds"] for p in pairs)
        result.append({"method": method["id"], "frames": len(pairs), "selectedTargets": len(selected),
            "selectedTargetHits": sum(p[1]["metrics"]["selectedTargetHit"] for p in selected),
            "unmatchedAlertsOnPartiallyAnnotatedImages": sum(p[1]["metrics"]["unmatchedAlerts"] for p in selected),
            "qualitativeOnlyFrames": sum(p[1]["metrics"]["qualitativeOnly"] for p in pairs),
            "priorAiNegativeFrames": len(negatives), "priorAiNegativeAlerts": sum(p[1]["metrics"]["aiReferenceNegativeAlerts"] for p in negatives),
            "modelForwards": sum(p[1]["modelPasses"] for p in pairs), "modelSeconds": seconds,
            "meanSecondsPerFrame": seconds / len(pairs), "humanVerified": False, "actualDistanceM": None})
    return result


def verify_inputs(protocol):
    for value, wanted in protocol["inputSha256"].items():
        if digest(checked_path(value)) != wanted:
            raise ValueError("Frozen input changed: " + value)
    if versions() != protocol["softwareVersions"]:
        raise ValueError("Frozen inference software versions changed")


def prepare(references_path, output, include_prior_negatives=True):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Evaluation output already exists; no overwrite or silent resume")
    if not output.is_relative_to(ROOT.resolve()):
        raise ValueError("Evaluation output must be inside the project")
    references_path = checked_path(references_path)
    observations = read(references_path)
    notes = observations.get("records")
    if not isinstance(notes, list) or not 1 <= len(notes) <= 100:
        raise ValueError("References require 1–100 manually inspected records")
    manifest_path = ROOT / "data/public-fungi-v4/manifest.json"
    manifest = read(manifest_path)
    if manifest.get("status") != "frozen":
        raise ValueError("Acquisition must be frozen before selecting evaluation references")
    source_rows = {row["id"]: row for row in manifest["images"]}
    inputs = {relative(p): digest(p) for p in [references_path, manifest_path, Path(__file__),
        ROOT / "vision/testbench.py", ROOT / "vision/metrics.py", ROOT / "vision/far_color.py",
        ROOT / "adapters/scout_frame.py", ROOT / "adapters/detect_fungi.py"]}
    model_dir = ROOT / "models/grounding-dino-tiny"
    for path in model_dir.iterdir():
        if path.is_file():
            inputs[relative(path)] = digest(path)
    references, identities = [], set()
    for note in notes:
        if not isinstance(note, dict):
            raise ValueError("Each observed reference must be an object")
        identity = note.get("id", note.get("imageId"))
        if not isinstance(identity, str) or not re.fullmatch(r"inat-photo-[0-9]+", identity) or identity in identities:
            raise ValueError("References require unique V4 image IDs")
        if identity not in source_rows or "selectedTargetPixelsXYXY" not in note or note.get("humanVerified") is True:
            raise ValueError("Reference must name a V4 photo and an explicit AI-selected target or null")
        identities.add(identity)
        row = source_rows[identity]
        path = checked_path(relative(manifest_path.parent / row["path"]))
        if digest(path) != row["sha256"]:
            raise ValueError("Source photograph changed")
        with Image.open(path) as opened:
            size = ImageOps.exif_transpose(opened).size
        if size != (row["width"], row["height"]):
            raise ValueError("Source dimensions disagree with frozen acquisition")
        box = validate_target(note["selectedTargetPixelsXYXY"], size)
        references.append({"id": identity, "kind": "new_ai_selected_target" if box else "new_ai_qualitative",
            "path": relative(path), "sha256": row["sha256"], "width": size[0], "height": size[1],
            "sourceGroupId": row["sourceGroupId"], "sourceUrl": row["sourceUrl"], "photoLicense": row["photoLicense"],
            "sourceClassId": row["classId"], "split": row["split"], "referenceAuthority": "ai", "humanVerified": False,
            "selectedTargetPixelsXYXY": box, "referenceComplete": False, "referenceEvidence": note})
    prior_protocol_path = ROOT / "experiments/results/ground-tiles-v1/protocol.json"
    prior_protocol = read(prior_protocol_path)
    inputs[relative(prior_protocol_path)] = digest(prior_protocol_path)
    if include_prior_negatives:
        prior = [r for r in prior_protocol["references"] if r["referenceAuthority"] == "ai"]
        if len(prior) != 4:
            raise ValueError("Expected four frozen earlier AI negatives")
        for row in prior:
            if row["referenceStatus"] != "absent" or row["referenceComplete"] is not True or row["truth"]:
                raise ValueError("Earlier AI negative reference is not complete absence")
            path = checked_path(row["path"])
            if digest(path) != row["sha256"]:
                raise ValueError("Earlier AI reference bytes changed")
            references.append({"id": row["frameId"], "kind": "prior_ai_negative", "path": row["path"],
                "sha256": row["sha256"], "width": row["width"], "height": row["height"],
                "sourceGroupId": row["sourceGroupId"], "selectedTargetPixelsXYXY": None,
                "referenceAuthority": "ai", "humanVerified": False, "referenceComplete": True,
                "referenceEvidence": row["referenceEvidence"], "confounders": row["confounders"]})
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "predictions").mkdir()
    planned = 0
    for reference in references:
        inputs[reference["path"]] = reference["sha256"]
        reference["views"] = {}
        with Image.open(checked_path(reference["path"])) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        for method in methods_for(reference):
            planned += method["passes"]
            pixels = method["targetPixels"]
            if pixels is None:
                view = {"path": reference["path"], "sha256": reference["sha256"], "synthetic": False, "actualDistanceM": None}
            else:
                image_path = output / "images" / (reference["id"] + f"--low{pixels}.png")
                if not image_path.exists():
                    degraded, transform = degraded_image(image, reference["selectedTargetPixelsXYXY"], pixels)
                    degraded.save(image_path, format="PNG")
                else:
                    _, transform = degraded_image(image, reference["selectedTargetPixelsXYXY"], pixels)
                view = {"path": relative(image_path), "sha256": digest(image_path), **transform}
            inputs[view["path"]] = view["sha256"]
            reference["views"][method["id"]] = view
    protocol = {"schema": SCHEMA, "createdAt": now(), "references": references, "methods": METHODS,
        "inputSha256": inputs, "softwareVersions": versions(), "expectedModel": prior_protocol["expectedModel"],
        "cpuThreads": 4, "device": "cpu", "scoreThreshold": .25, "textThreshold": .25, "iouThreshold": .5,
        "tileGrid": 2, "tileRegion": [0, 0, 1, 1], "fullFrameContext": True, "overlapFraction": .2,
        "plannedModelForwards": planned, "annotationAuthority": "ai", "trainingPerformed": False,
        "modelActivation": False, "actuationEnabled": False, "actualDistanceM": None, "limitations": LIMITATIONS}
    new_json(output / "protocol.json", protocol)
    new_json(output / "protocol-sha256.json", {"sha256": digest(output / "protocol.json")})
    print(json.dumps({"prepared": relative(output), "frames": len(references), "plannedModelForwards": planned}), flush=True)
    return protocol


def run(output):
    from adapters.scout_frame import LocalDetector
    from vision.testbench import compare
    output = Path(output).resolve()
    protocol_path = checked_path(output / "protocol.json")
    protocol = read(protocol_path)
    protocol_hash = digest(protocol_path)
    if protocol.get("schema") != SCHEMA or read(output / "protocol-sha256.json")["sha256"] != protocol_hash:
        raise ValueError("Unknown or modified frozen protocol")
    if (output / "report.json").exists():
        raise FileExistsError("Evaluation already started; no overwrite or silent resume")
    # Exclusive claim precedes model loading so concurrent runs cannot duplicate work.
    new_json(output / "run-claim.json", {"startedAt": now(), "protocolSha256": protocol_hash})
    verify_inputs(protocol)
    started = time.perf_counter()
    report = {"schema": "mushroom-vision.new-fungi-v4-report.v1", "status": "loading", "startedAt": now(),
        "protocolSha256": protocol_hash, "protocolPath": relative(protocol_path), "records": [], "summary": [],
        "progress": {"completed": 0, "total": protocol["plannedModelForwards"], "unit": "model-forwards"},
        "trainingPerformed": False, "modelActivation": False, "actuationEnabled": False, "humanVerified": False,
        "actualDistanceM": None, "limitations": LIMITATIONS}
    write_json(output / "report.json", report)
    try:
        detector = LocalDetector(ROOT / "models/grounding-dino-tiny", device="cpu", threads=4, box_threshold=.25, text_threshold=.25)
        for key, wanted in protocol["expectedModel"].items():
            if detector.info.get(key) != wanted:
                raise ValueError("Frozen model identity changed: " + key)
        report.update(status="running", model=detector.info, modelLoadSeconds=time.perf_counter() - started)
        write_json(output / "report.json", report)
        for reference in protocol["references"]:
            record = {**reference, "methods": {}}
            report["records"].append(record)
            for method in methods_for(reference):
                view = reference["views"][method["id"]]
                with Image.open(checked_path(view["path"])) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                settings = {"methods": [method["method"]], "tileGrid": 2, "scale": 1, "boxThreshold": .25}
                result = compare(image, view["sha256"], settings, detector)
                if result["modelPasses"] != method["passes"]:
                    raise ValueError("Model forward count differs from frozen condition")
                observed = result["methods"][0]
                path = output / "predictions" / (reference["id"] + "--" + method["id"] + ".json")
                new_json(path, result)
                metrics = score(reference, observed["candidates"])
                record["methods"][method["id"]] = {"metrics": metrics, "seconds": observed["seconds"],
                    "wallSeconds": result["seconds"], "modelPasses": result["modelPasses"],
                    "predictionPath": relative(path), "predictionSha256": digest(path)}
                report["progress"]["completed"] += result["modelPasses"]
                report["elapsedSeconds"] = time.perf_counter() - started
                report["summary"] = summarize(report["records"])
                write_json(output / "report.json", report)
                print(json.dumps({"image": reference["id"], "condition": method["id"], "seconds": round(observed["seconds"], 2),
                    "targetHit": metrics["selectedTargetHit"], "alerts": metrics["predictions"], "progress": report["progress"]}), flush=True)
        verify_inputs(protocol)
        if digest(protocol_path) != protocol_hash or report["progress"]["completed"] != protocol["plannedModelForwards"]:
            raise ValueError("Protocol integrity or completed inference budget mismatch")
        report.update(status="completed", completedAt=now(), inputIntegrityVerified=True)
    except Exception as exc:
        report.update(status="failed", failedAt=now(), error={"type": type(exc).__name__, "message": str(exc)})
        write_json(output / "report.json", report)
        raise
    write_json(output / "report.json", report)
    print(json.dumps({"completed": True, "report": relative(output / "report.json"), "summary": report["summary"]}), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--references", type=Path, default=ROOT / "experiments/results/new-fungi-v4/ai-observations.json")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/results/new-fungi-v4/evaluation")
    parser.add_argument("--without-prior-negatives", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.references, args.output, not args.without_prior_negatives)
    else:
        run(args.output)
