"""Local still-photo comparisons with original-coordinate evidence, never flight control."""
from __future__ import annotations

import base64
import binascii
import hashlib
import io
import math
import time

from PIL import Image, ImageOps, UnidentifiedImageError
from vision.far_color import enhance, proposals
from vision.metrics import iou

MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_CANDIDATES = 100
MAX_PASS_CANDIDATES = 1000
METHOD_ORDER = ("original", "color", "tiles")
FULL_TILE_REGION = (0., 0., 1., 1.)
GROUND_TILE_REGION = (0., .35, 1., 1.)


class RequestError(ValueError):
    """Only fixed, client-safe validation messages are carried by this type."""


def settings_for(value):
    if not isinstance(value, dict):
        raise RequestError("Settings must be an object.")
    if set(value) - {"methods", "tileGrid", "scale", "boxThreshold", "tileRegion"}:
        raise RequestError("Unsupported test settings.")
    methods = value.get("methods", ["original"])
    if (not isinstance(methods, list) or not 1 <= len(methods) <= 3 or
            any(not isinstance(m, str) or m not in METHOD_ORDER for m in methods) or len(set(methods)) != len(methods)):
        raise RequestError("Choose unique methods from original, color and tiles.")
    grid = value.get("tileGrid", 2)
    scale = value.get("scale", 1)
    threshold = value.get("boxThreshold", .25)
    if type(grid) is not int or grid not in (2, 3):
        raise RequestError("Tile grid must be 2 or 3.")
    if type(scale) not in (int, float) or scale not in (1, .5, .25, .125):
        raise RequestError("Scale must be 1, 0.5, 0.25 or 0.125.")
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not .1 <= threshold <= .8:
        raise RequestError("Detection threshold must be between 0.1 and 0.8.")
    result = {"methods": [m for m in METHOD_ORDER if m in methods], "tileGrid": grid,
              "scale": float(scale), "boxThreshold": float(threshold)}
    if "tileRegion" in value:
        if "tiles" not in methods:
            raise RequestError("A tile region requires the tiles method.")
        result["tileRegion"] = normalized_tile_region(value["tileRegion"])
    return result


def normalized_tile_region(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 4 or
            any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in value) or
            value[0] >= value[2] or value[1] >= value[3]):
        raise RequestError("Tile region must be normalized [left, top, right, bottom] within 0..1 with positive area.")
    return [float(v) for v in value]


def tile_region_bounds(width, height, region=None):
    """Selected image rectangle, not inferred ground; include boundary pixels."""
    if type(width) is not int or type(height) is not int or min(width, height) < 1:
        raise ValueError("Invalid region dimensions")
    left, top, right, bottom = normalized_tile_region(FULL_TILE_REGION if region is None else region)
    return (math.floor(left * width), math.floor(top * height),
            math.ceil(right * width), math.ceil(bottom * height))


def region_tile_boxes(width, height, grid, region=None):
    left, top, right, bottom = tile_region_bounds(width, height, region)
    return [(x1 + left, y1 + top, x2 + left, y2 + top)
            for x1, y1, x2, y2 in tile_boxes(right - left, bottom - top, grid)]


def original_bounds(bounds, original_size, working_size):
    sx, sy = original_size[0] / working_size[0], original_size[1] / working_size[1]
    return [bounds[0] * sx, bounds[1] * sy, bounds[2] * sx, bounds[3] * sy]


def decode_image(encoded, mime_type):
    if not isinstance(encoded, str) or len(encoded) > ((MAX_IMAGE_BYTES + 2) // 3) * 4:
        raise RequestError("Image must be base64 containing at most 8 MiB.")
    if mime_type not in ("image/jpeg", "image/png"):
        raise RequestError("Only still JPEG and PNG images are supported.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise RequestError("Image base64 is invalid.") from exc
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise RequestError("Image must contain between 1 byte and 8 MiB.")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            formats = {"JPEG": "image/jpeg", "PNG": "image/png"}
            if formats.get(source.format) != mime_type or getattr(source, "n_frames", 1) != 1:
                raise RequestError("Image format does not match a supported still JPEG or PNG.")
            if min(source.size) < 16 or source.width * source.height > MAX_IMAGE_PIXELS:
                raise RequestError("Image must be at least 16 pixels per side and at most 20 megapixels.")
            image = ImageOps.exif_transpose(source).convert("RGB")
    except RequestError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise RequestError("Image could not be decoded.") from exc
    return image, hashlib.sha256(payload).hexdigest()


def tile_boxes(width, height, grid):
    """Edge-anchored grid, approximately 20% tile overlap and complete coverage."""
    if type(width) is not int or type(height) is not int or min(width, height) < 1 or grid not in (2, 3):
        raise ValueError("Invalid tile dimensions")

    def intervals(length):
        tile = min(length, math.ceil(length / (grid - .2 * (grid - 1))))
        starts = sorted({round(i * (length - tile) / (grid - 1)) for i in range(grid)})
        return [(start, start + tile) for start in starts]

    return [(left, top, right, bottom) for top, bottom in intervals(height)
            for left, right in intervals(width)]


def projected_candidates(predictions, bounds, original_size, working_size, threshold, source_pass):
    if not isinstance(predictions, list) or len(predictions) > MAX_PASS_CANDIDATES:
        raise RuntimeError("Invalid detector output")
    left, top, right, bottom = bounds
    sx, sy = original_size[0] / working_size[0], original_size[1] / working_size[1]
    valid = []
    for row in predictions:
        if not isinstance(row, dict):
            continue
        box, confidence = row.get("bboxPixelsXYXY"), row.get("confidence")
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in box) or
                type(confidence) not in (int, float) or not math.isfinite(confidence) or not threshold <= confidence <= 1):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = max(0, min(right - left, x1)), max(0, min(right - left, x2))
        y1, y2 = max(0, min(bottom - top, y1)), max(0, min(bottom - top, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        valid.append({"bboxPixelsXYXY": [min(original_size[0], (x1 + left) * sx), min(original_size[1], (y1 + top) * sy),
                                          min(original_size[0], (x2 + left) * sx), min(original_size[1], (y2 + top) * sy)],
                      "confidence": float(confidence), "sourcePass": source_pass})
    return valid


def bounded_nms(candidates, method):
    ordered = sorted(candidates, key=lambda row: row["confidence"], reverse=True)
    kept = []
    for row in ordered:
        if all(iou(row["bboxPixelsXYXY"], previous["bboxPixelsXYXY"]) <= .5 for previous in kept):
            kept.append({"id": f"{method}-{len(kept) + 1}", **row})
            if len(kept) == MAX_CANDIDATES:
                break
    return kept


def work_dimensions(size, scale):
    return (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))


def pass_count(size, settings):
    methods = settings["methods"]
    working = work_dimensions(size, settings["scale"])
    return (int("original" in methods or "tiles" in methods) + int("color" in methods) +
            (len(region_tile_boxes(*working, settings["tileGrid"], settings.get("tileRegion"))) if "tiles" in methods else 0))


def compare(image, source_sha256, settings, detector, progress=None):
    """Sequential request; detector can stay warm, but all thresholds reset here."""
    settings = settings_for(settings)
    started = time.perf_counter()
    original_size = image.size
    if image.mode != "RGB" or min(original_size) < 16 or image.width * image.height > MAX_IMAGE_PIXELS:
        raise RequestError("Provide an oriented RGB image within the image limits.")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_sha256):
        raise RequestError("Image checksum is invalid.")
    working_size = work_dimensions(original_size, settings["scale"])
    working = image if working_size == original_size else image.resize(working_size, Image.Resampling.LANCZOS)
    methods = settings["methods"]
    # LocalDetector exposes threshold fields; reset them for every request rather
    # than mutating the global model defaults or retaining prior UI settings.
    detector.box_threshold = settings["boxThreshold"]
    detector.text_threshold = .25
    total, completed = pass_count(original_size, settings), 0
    full_bounds = (0, 0, *working_size)
    warnings = ["Detection scores are uncalibrated model scores, not mushroom or species probabilities.",
                "These photo comparisons establish no actual distance, location or flight clearance.",
                "More candidates do not establish improvement; annotate misses and false positives to compare methods."]
    if settings["scale"] != 1:
        warnings.append("Scale reduces the entire image; boxes are projected back to original coordinates. This is not a measured distance simulation.")
    if min(working_size) < 16:
        warnings.append("The scaled image is extremely small; model resizing cannot restore missing detail.")

    def notify(stage):
        if progress is not None:
            progress({"stage": stage, "completed": completed, "total": total})

    def forward(view, bounds, name):
        nonlocal completed
        notify(name)
        before = time.perf_counter()
        prediction = detector.predict(view)
        candidates = projected_candidates(prediction, bounds, original_size, working_size, settings["boxThreshold"], name)
        seconds = time.perf_counter() - before
        completed += 1
        notify(name)
        return candidates, seconds

    full, full_seconds, by_method = [], 0, {}
    if "original" in methods or "tiles" in methods:
        full, full_seconds = forward(working, full_bounds, "original")
        if "original" in methods:
            by_method["original"] = {"id": "original", "label": "Original photo", "candidates": bounded_nms(full, "original"),
                                     "seconds": full_seconds, "passes": 1, "reusedPasses": 0, "preNmsCandidateCount": len(full)}
    if "color" in methods:
        adjusted = enhance(working)
        colors, seconds = forward(adjusted, full_bounds, "color")
        by_method["color"] = {"id": "color", "label": "Color-sensitive view", "candidates": bounded_nms(colors, "color"),
                              "seconds": seconds, "passes": 1, "reusedPasses": 0, "preNmsCandidateCount": len(colors)}
        warnings.append("Color enhancement is experimental and can increase false positives. Original pixels remain the review evidence.")
    if "tiles" in methods:
        candidates, additional_seconds = list(full), 0
        boxes = region_tile_boxes(*working_size, settings["tileGrid"], settings.get("tileRegion"))
        timings = [{"kind": "context", "bboxPixelsXYXY": list((0, 0, *original_size)), "seconds": full_seconds}]
        for index, bounds in enumerate(boxes):
            extra, seconds = forward(working.crop(bounds), bounds, f"tile_{index + 1}")
            candidates.extend(extra)
            additional_seconds += seconds
            timings.append({"kind": "tile", "bboxPixelsXYXY": original_bounds(bounds, original_size, working_size), "seconds": seconds})
        by_method["tiles"] = {"id": "tiles", "label": "Overlapping detail tiles", "candidates": bounded_nms(candidates, "tiles"),
                              "seconds": full_seconds + additional_seconds, "additionalSeconds": additional_seconds,
                              "passes": 1 + len(boxes), "reusedPasses": int("original" in methods),
                              "preNmsCandidateCount": len(candidates),
                              "tileBoundsWorkingPixelsXYXY": [list(b) for b in boxes], "overlapFraction": .2,
                              "tileBoundsPixelsXYXY": [original_bounds(b, original_size, working_size) for b in boxes],
                              "tileRegion": list(settings.get("tileRegion", FULL_TILE_REGION)), "fullFrameContext": True,
                              "passTimings": timings}
        warnings.append("Tiles preserve the working image pixels and include the original full-frame pass; they do not create new optical detail.")
        if "tileRegion" in settings:
            warnings.append("The tile region is a selected image rectangle, not detected ground or evidence of safe terrain. The context pass still covers the entire frame.")
    cue_results = []
    if "color" in methods:
        notify("color_cues")
        sx, sy = image.width / working.width, image.height / working.height
        for index, cue in enumerate(proposals(working, max_proposals=12, min_delta_e=8)):
            b = cue["bboxPixelsXYXY"]
            cue_results.append({"id": f"color_cues-{index + 1}", "bboxPixelsXYXY": [b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy],
                                "pointPixelsXY": [cue["pointPixelsXY"][0] * sx, cue["pointPixelsXY"][1] * sy],
                                "contrastScore": cue["contrastScore"], "colorEvidence": cue["colorEvidence"],
                                "cueType": "color_salience", "colorOnly": True, "species": None})
        warnings.append("Separate color cues are contrast anomalies, not mushroom detections; leaves and shadows can trigger them.")
    if any(len(row["candidates"]) == MAX_CANDIDATES for row in by_method.values()):
        warnings.append("At least one method reached the 100-candidate display limit.")
    notify("complete")
    info = getattr(detector, "info", {})
    allowed_info = {key: info.get(key) for key in ("model", "resolvedRevision", "device", "threads", "prompt")}
    model = {**allowed_info, "boxThreshold": settings["boxThreshold"], "textThreshold": .25,
             "downloadsEnabled": False, "nmsIou": .5, "scoreMeaning": "Uncalibrated detection score, not a species probability"}
    return {"schema": "mycoscout.testbench-result.v1",
            "image": {"width": image.width, "height": image.height, "workingWidth": working.width, "workingHeight": working.height,
                      "sha256": source_sha256, "pixelConvention": "EXIF-oriented RGB image; x right, y down"},
            "settings": settings, "model": model, "methods": [by_method[m] for m in METHOD_ORDER if m in by_method],
            "colorCues": cue_results, "seconds": time.perf_counter() - started, "modelPasses": completed,
            "warnings": warnings, "actualDistanceM": None, "actuationEnabled": False}
