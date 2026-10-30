"""Crop saved unlocated detector candidates and send them to human review.

This consumes adapters.detect_fungi output; it is not a live camera driver.
Candidate crops remain unlocated and never become human labels automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import urllib.parse

from PIL import Image, ImageOps
from .review_stream import MAX_IMAGE, Station, send_frame
from vision.export_scout_leads import COLOR_SCHEMA, archive_cue_evidence, archive_flags

INPUT_SCHEMA = "mycoscout.unlocated-candidates.v1"
OUTPUT_SCHEMA = "mycoscout.detection-review-bridge.v1"
MAX_PARENT_BYTES = 50_000_000
MAX_PARENT_PIXELS = 40_000_000


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def identifier(value, name):
    if not isinstance(value, str) or not value.strip() or len(value) > 160 or any(ord(ch) < 32 for ch in value):
        raise ValueError(f"{name} must be nonempty text of at most 160 characters")
    return value.strip()


def read_document(path):
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Detector JSON exceeds 2 MiB")
    return json.loads(path.read_text(encoding="utf-8"))


def source_credits(path, parent_sha256):
    """Read explicitly selected local credit fields, never location/taxon data."""
    if path is None:
        return {}
    if not path.is_file() or path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Source manifest must be local JSON no larger than 32 MiB")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("images") if isinstance(manifest, dict) else None
    if not isinstance(rows, list) or len(rows) > 100000:
        raise ValueError("Source manifest requires images[] with at most 100000 rows")
    matches = [row for row in rows if isinstance(row, dict) and row.get("sha256") == parent_sha256]
    if not matches:
        raise ValueError("No source manifest image matches the exact parent SHA256")

    def bounded_text(value, name, is_url=False):
        if value is None: return None
        if not isinstance(value, str) or not value.strip() or len(value) > (2048 if is_url else 1000) or any(ord(ch) < 32 for ch in value):
            raise ValueError(f"Invalid bounded source {name}")
        value = value.strip()
        if is_url:
            parsed = urllib.parse.urlsplit(value)
            if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError(f"Source {name} must be HTTPS without credentials")
        return value

    credits = []
    for row in matches:
        license_info = row.get("photoLicense", {})
        if not isinstance(license_info, dict): raise ValueError("photoLicense must be an object")
        attribution = bounded_text(license_info.get("attribution"), "attribution")
        license_url = bounded_text(license_info.get("url"), "license URL", True)
        source_url = bounded_text(row.get("sourceUrl"), "source URL", True)
        item = {}
        if attribution is not None: item["attribution"] = attribution
        if license_url is not None: item["licenseUrl"] = license_url
        if source_url is not None: item["sourceUrl"] = source_url
        if attribution is not None or license_url is not None:
            item["photoLicense"] = {key: value for key, value in (("attribution", attribution), ("url", license_url)) if value is not None}
        credits.append(item)
    if any(item != credits[0] for item in credits[1:]):
        raise ValueError("Matching parent rows have conflicting attribution")
    return credits[0]


def prepare_candidates(document, image_path, output_dir, *, max_candidates=10, padding=0.15,
                       max_edge=2048, session_id=None, source="detector_candidate",
                       device_id="saved-detector", evaluation_only=False, source_manifest=None):
    if not isinstance(document, dict) or document.get("schema") not in {INPUT_SCHEMA, COLOR_SCHEMA}:
        raise ValueError(f"Only {INPUT_SCHEMA} and {COLOR_SCHEMA} are supported")
    schema = document["schema"]
    flags = archive_flags(document)
    if schema == COLOR_SCHEMA and document.get("colorOnly") is not True:
        raise ValueError("Color proposals must explicitly declare colorOnly:true")
    info = document.get("image")
    candidates = document.get("candidates")
    if not isinstance(info, dict) or not isinstance(candidates, list) or len(candidates) > 1000:
        raise ValueError("Detector image metadata and at most 1000 candidates are required")
    digest = info.get("sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Detector image.sha256 must identify the original source bytes")
    if any(not isinstance(info.get(key), int) or isinstance(info[key], bool) or info[key] < 1 for key in ("width", "height")):
        raise ValueError("Detector dimensions must be positive integer pixels")
    if info.get("pixelConvention") != "EXIF-oriented RGB image; x right, y down":
        raise ValueError("Detector pixel convention must be EXIF-oriented RGB image; x right, y down")
    if not isinstance(max_candidates, int) or isinstance(max_candidates, bool) or not 1 <= max_candidates <= 50:
        raise ValueError("max_candidates must be 1–50")
    if not finite(padding) or not 0 <= padding <= 0.5 or not isinstance(max_edge, int) or not 64 <= max_edge <= 2048:
        raise ValueError("Use padding 0–0.5 and max_edge 64–2048")
    if not isinstance(evaluation_only, bool):
        raise ValueError("evaluation_only must be boolean")
    evaluation_only = evaluation_only or flags["evaluationOnly"]
    source, device_id = identifier(source, "source"), identifier(device_id, "device_id")
    group_id = "detector-frame-" + digest
    session_id = identifier(session_id or group_id, "session_id")
    if not image_path.is_file() or image_path.stat().st_size > MAX_PARENT_BYTES:
        raise ValueError("Explicit source image must exist and be at most 50 MB")
    raw = image_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Explicit source image SHA256 does not match the detector report")
    credits = source_credits(source_manifest, digest)
    # The detector's embedded path is never opened. Source bytes come only from --image.
    with Image.open(io.BytesIO(raw)) as original:
        if original.format not in {"JPEG", "PNG"} or getattr(original, "n_frames", 1) != 1:
            raise ValueError("Source must be a still JPEG or PNG")
        if original.width * original.height > MAX_PARENT_PIXELS:
            raise ValueError("Source image exceeds 40 megapixels")
        oriented = ImageOps.exif_transpose(original)
        if oriented.size != (info["width"], info["height"]):
            raise ValueError("Source EXIF-oriented dimensions do not match the detector report")
        image = oriented.convert("RGB")
    width, height = image.size
    valid, rejections = [], []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            rejections.append({"index": index, "reason": "malformed_candidate"}); continue
        score = candidate.get("contrastScore") if schema == COLOR_SCHEMA else candidate.get("confidence")
        box = candidate.get("bboxPixelsXYXY")
        if not finite(score) or not 0 <= score <= 1 or not isinstance(box, list) or len(box) != 4 or not all(finite(value) for value in box):
            rejections.append({"index": index, "reason": "invalid_score_or_pixel_box"}); continue
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            rejections.append({"index": index, "reason": "pixel_box_outside_image_or_empty"}); continue
        if candidate.get("status") != "UNLOCATED" or candidate.get("poseKnown") is not False or candidate.get("depthKnown") is not False:
            rejections.append({"index": index, "reason": "unsupported_candidate_location_contract"}); continue
        try:
            evidence = archive_cue_evidence(candidate, schema)
        except ValueError as error:
            rejections.append({"index": index, "reason": "invalid_cue_evidence", "detail": str(error)[:300]}); continue
        valid.append((index, float(score), [float(value) for value in box], evidence))
    valid.sort(key=lambda item: (-item[1], item[0]))
    selected = valid[:max_candidates]
    # Create a new directory only after source validation; never replace prior crops/reports.
    output_dir.mkdir(parents=True, exist_ok=False)
    detector = document.get("detector") if isinstance(document.get("detector"), dict) else {}
    rows = []
    for rank, (index, score, box, evidence) in enumerate(selected, 1):
        x1, y1, x2, y2 = box
        dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
        crop_box = [max(0, math.floor(x1 - dx)), max(0, math.floor(y1 - dy)),
                    min(width, math.ceil(x2 + dx)), min(height, math.ceil(y2 + dy))]
        crop = image.crop(crop_box)
        source_crop_size = list(crop.size)
        crop.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        encoded = io.BytesIO()
        # EXIF and source file metadata are deliberately absent from the derived image.
        crop.save(encoded, format="JPEG", quality=90, optimize=True)
        photo = encoded.getvalue()
        if len(photo) > MAX_IMAGE:
            raise ValueError("Candidate JPEG exceeds the review upload limit")
        filename = f"candidate-{rank:02d}-box-{index:04d}.jpg"
        with (output_dir / filename).open("xb") as stream:
            stream.write(photo)
        metadata = {
            **credits,
            **evidence,
            "synthetic": flags["synthetic"], "evaluationOnly": evaluation_only,
            "parentImageSha256": digest, "sourceFrameId": "frame-sha256-" + digest,
            "sourceFrameIdIsContentDerived": True, "sharedGroupId": group_id,
            "sourceImageWidth": width, "sourceImageHeight": height,
            "detectorCandidateIndex": index, "detectorScore": score,
            "detectorScoreMeaning": "Local color contrast, not mushroom/species probability" if evidence["cueType"] == "color_salience" else "Uncalibrated candidate score; not a species probability",
            "detectorBoxPixelsXYXY": box, "cropBoxPixelsXYXY": crop_box,
            "cropSourceSize": source_crop_size, "cropOutputSize": list(crop.size),
            "cropPaddingFraction": padding, "captureTimeKnown": False,
            "poseKnown": False, "depthKnown": False, "actualLiveHardware": False,
            "frameCaptureGeometry": {
                "source": "derived_archive_crop", "originalFrameSizePixels": [width, height],
                "targetBoxPixelsXYXY": box, "targetWidthFractionOfOriginalFrame": (x2 - x1) / width,
                "targetHeightFractionOfOriginalFrame": (y2 - y1) / height,
                "physicalCloseCapture": False, "physicalRangeKnown": False,
                "closeFrameRequest": {"desiredNativeWidthFraction": 0.7,
                    "requireFreshPhoto": True, "requiresSafeStandoff": True,
                    "digitalCropCountsAsPhysicalVisit": False}},
            "detectorModel": str(detector.get("model", "unspecified"))[:300],
            "detectorRevision": str(detector.get("resolvedRevision", "unspecified"))[:160],
        }
        rows.append({"file": filename, "sha256": hashlib.sha256(photo).hexdigest(), "metadata": metadata})
    report = {"schema": OUTPUT_SCHEMA, "inputSchema": schema, "parentImageSha256": digest,
              "sessionId": session_id, "groupId": group_id, "source": source, "deviceId": device_id,
              "evaluationOnly": evaluation_only, "synthetic": flags["synthetic"],
              "captureTimestamp": None, "actuationEnabled": False,
              "humanReviewRequired": True, "inputCandidateCount": len(candidates), "preparedCount": len(rows),
              "validCandidatesOmittedByLimit": max(0, len(valid) - len(selected)), "rejections": rejections,
              "colorOnlyCount": sum(row["metadata"]["cueType"] == "color_salience" for row in rows),
              "crops": rows, "uploads": [], "notes": [
                  "Saved detector output was replayed; no live camera or flight driver was used.",
                  "Every crop remains unlocated, with unknown original capture time and no human label.",
                  "All crops share their parent-frame group and declared survey session.",
                  "Keep parentImageSha256 in exported provenance so prior held-out frames remain held out.",
                  "Color-only cues are habitat follow-ups and bypass mushroom classification.",
                  "A derived crop does not demonstrate a physical visit or frame-filling close capture.",
                  "Synthetic pixel-scale test distances and caller-supplied range are not measurement evidence."]}
    with (output_dir / "prepared.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    return report


class GroupedStation:
    """Add only a fixed source-frame grouping ID to the existing bridge payload."""
    def __init__(self, station, group_id):
        self.station, self.group_id = station, group_id

    def request(self, path, data=None, **kwargs):
        if path == "/api/finds":
            data = {**data, "groupId": self.group_id}
        return self.station.request(path, data, **kwargs)


def send_candidates(station, prepared, output_dir, *, classify=True):
    report = json.loads(json.dumps(prepared))
    grouped = GroupedStation(station, prepared["groupId"])
    for row in prepared["crops"]:
        try:
            result = send_frame(grouped, output_dir / row["file"], session_id=prepared["sessionId"],
                                source=prepared["source"], device_id=prepared["deviceId"],
                                evaluation_only=prepared["evaluationOnly"], metadata=row["metadata"],
                                classify=classify and row["metadata"].get("classificationEligible", True))
            if row["metadata"].get("classificationEligible") is False:
                result["classificationSkippedReason"] = "Habitat/color cue; fungal presence requires closer evidence"
            report["uploads"].append({"file": row["file"], **result})
        except (ValueError, RuntimeError, OSError) as error:
            report["uploads"].append({"file": row["file"], "uploadError": str(error)[:1500]})
    report["uploadedCount"] = sum("findId" in row for row in report["uploads"])
    report["classifiedCount"] = sum(row.get("classified") is True for row in report["uploads"])
    report["classificationErrorCount"] = sum("classificationError" in row for row in report["uploads"])
    report["uploadErrorCount"] = sum("uploadError" in row for row in report["uploads"])
    with (output_dir / "review-results.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True, help="Explicit original image; embedded detector paths are ignored")
    parser.add_argument("--source-manifest", type=Path, help="Local images[] manifest for parent SHA256-matched photo attribution")
    parser.add_argument("--output-dir", type=Path, required=True, help="New crop/report directory; must not already exist")
    parser.add_argument("--station", default="http://127.0.0.1:8787")
    parser.add_argument("--session-id")
    parser.add_argument("--device-id", default="saved-detector")
    parser.add_argument("--source", default="detector_candidate")
    parser.add_argument("--evaluation-only", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--padding", type=float, default=0.15, help="Fraction of each box dimension added on each side, 0–0.5")
    parser.add_argument("--max-edge", type=int, default=2048)
    parser.add_argument("--prepare-only", action="store_true", help="Write crops without uploading")
    parser.add_argument("--no-classify", action="store_true", help="Upload for human review without model guesses")
    args = parser.parse_args(argv)
    try:
        station = None if args.prepare_only else Station(args.station)
        prepared = prepare_candidates(read_document(args.detections), args.image, args.output_dir,
            max_candidates=args.max_candidates, padding=args.padding, max_edge=args.max_edge,
            session_id=args.session_id, source=args.source, device_id=args.device_id, evaluation_only=args.evaluation_only,
            source_manifest=args.source_manifest)
        report = prepared if station is None else send_candidates(station, prepared, args.output_dir, classify=not args.no_classify)
        print(json.dumps({key: report[key] for key in ("schema", "preparedCount", "uploadedCount", "classifiedCount", "classificationErrorCount", "uploadErrorCount") if key in report}))
        return 1 if report.get("uploadErrorCount", 0) or report.get("classificationErrorCount", 0) else 0
    except (ValueError, OSError, RuntimeError, Image.DecompressionBombError) as error:
        print(f"Detection review bridge failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
