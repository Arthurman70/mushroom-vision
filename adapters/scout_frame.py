"""Inspect a saved RGB frame with an optional additional color-sensitive pass.

Uses local detector weights only. Original source pixels feed review crops;
enhancement is only an inference view. The output cannot command aircraft motion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from PIL import Image, ImageOps
from adapters.detect_fungi import candidate_records
from adapters.detections_to_review import prepare_candidates, send_candidates, source_credits
from adapters.review_stream import Station
from vision.export_scout_leads import candidate_leads
from vision.far_color import color_evidence, contrast_map, enhance, proposals
from vision.metrics import class_agnostic_nms

PROMPT = "a mushroom. a fungus."


def bounded_integer(value, minimum, maximum, name):
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in {minimum}..{maximum}")


class LocalDetector:
    """Load once for sequential frames; no downloads or remote model code."""
    def __init__(self, model_path, *, device="cpu", threads=4, box_threshold=.25, text_threshold=.25):
        bounded_integer(threads, 1, 8, "threads")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) or not 0 <= v <= 1
               for v in (box_threshold, text_threshold)):
            raise ValueError("Detector thresholds must be finite in 0..1")
        if not Path(model_path).is_dir():
            raise ValueError("Provide an existing local model directory; downloads are disabled")
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        torch.set_num_threads(threads)
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable")
        self.torch, self.device = torch, device
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path, local_files_only=True,
            trust_remote_code=False, use_safetensors=True).to(device).eval()
        local_weights = Path(model_path) / "model.safetensors"
        with local_weights.open("rb") as weights:
            weights_hash = hashlib.file_digest(weights, "sha256").hexdigest()
        self.info = {"model": "GroundingDINO local model", "resolvedRevision": getattr(self.model.config, "_commit_hash", None) or "sha256:" + weights_hash,
            "device": device, "threads": threads, "boxThreshold": box_threshold, "textThreshold": text_threshold,
            "prompt": PROMPT, "downloadsEnabled": False, "scoreMeaning": "Uncalibrated detection score, not a species probability"}

    def predict(self, image):
        inputs = self.processor(images=image, text=PROMPT, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs)
        result = self.processor.post_process_grounded_object_detection(output, inputs.input_ids,
            box_threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[(image.height, image.width)])[0]
        labels = result.get("text_labels", result.get("labels"))
        if labels is None:
            raise RuntimeError("Detector output omitted labels")
        return candidate_records(result["boxes"].detach().cpu().tolist(), result["scores"].detach().cpu().tolist(),
                                 labels, image.width, image.height)


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")


def inspect_frame(image_path, output_dir, detector, *, color_sensitive=False, max_candidates=10,
                  max_color_cues=12, min_delta_e=8, source_manifest=None, evaluation_only=False,
                  station=None, session_id=None):
    """Reusable bounded sequential pass. A caller may reuse the loaded detector.

    This is a saved-frame adapter; the caller must supply a synchronized live
    geometry contract elsewhere before any queue can plan a physical approach.
    """
    bounded_integer(max_candidates, 1, 50, "max_candidates")
    bounded_integer(max_color_cues, 1, 50, "max_color_cues")
    if not isinstance(color_sensitive, bool) or not isinstance(evaluation_only, bool):
        raise ValueError("color_sensitive and evaluation_only must be boolean")
    if not isinstance(min_delta_e, (int, float)) or isinstance(min_delta_e, bool) or not math.isfinite(min_delta_e) or not 0 < min_delta_e <= 200:
        raise ValueError("min_delta_e must be finite in (0, 200]")
    image_path, output_dir = Path(image_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; preserve the prior frame report")
    if not image_path.is_file() or image_path.stat().st_size > 50_000_000:
        raise ValueError("Source must be a local image no larger than 50 MB")
    source_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
    credits = source_credits(source_manifest, source_hash)
    with Image.open(image_path) as opened:
        if opened.format not in {"JPEG", "PNG"} or getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Source must be a still JPEG or PNG")
        if opened.width * opened.height > 8_000_000:
            raise ValueError("Scout frame exceeds 8 megapixels; provide a native camera frame within the limit")
        original = ImageOps.exif_transpose(opened).convert("RGB")
    meta = {"path": str(image_path.resolve()), "sha256": source_hash, "width": original.width,
            "height": original.height, "pixelConvention": "EXIF-oriented RGB image; x right, y down"}
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    delta = contrast_map(original) if color_sensitive else None
    candidates, comparisons = [], []
    for variant in ("original", "color_enhanced") if color_sensitive else ("original",):
        view = original if variant == "original" else enhance(original)
        before = time.perf_counter()
        predicted = detector.predict(view)
        if not isinstance(predicted, list) or len(predicted) > 1000 or any(not isinstance(item, dict) for item in predicted):
            raise ValueError("Detector must return at most 1000 candidates")
        # Re-validate even a caller-supplied predictor; discard any invented labels/geometry.
        valid = candidate_records([c.get("bboxPixelsXYXY", []) for c in predicted],
            [c.get("confidence", float("nan")) for c in predicted], ["mushroom candidate"] * len(predicted), original.width, original.height)
        for item in valid:
            item["cueType"] = "detector"
            item["inferenceView"] = variant
            if delta is not None:
                item["colorEvidence"] = color_evidence(delta, item["bboxPixelsXYXY"], enhanced=variant == "color_enhanced")
        candidates.extend(valid)
        comparisons.append({"variant": variant, "candidateCount": len(valid), "seconds": time.perf_counter() - before})
    combined = class_agnostic_nms(candidates, .5)
    selected = combined[:max_candidates]
    colors = proposals(original, max_color_cues, min_delta_e) if color_sensitive else []
    detector_report = {"schema": "mycoscout.unlocated-candidates.v1", "image": meta,
        "detector": {**detector.info, "colorSensitive": color_sensitive, "nmsThreshold": .5},
        "candidates": selected, "evaluationOnly": evaluation_only,
        "comparison": comparisons, "omittedByLimit": max(0, len(combined) - len(selected)),
        "limitations": ["Additional color inference is experimental and may increase false positives.",
            "Candidates from both views share original RGB coordinates; review crops use original pixels.",
            "No physical distance, capture time, pose, species or flight clearance is inferred."]}
    color_report = {"schema": "mycoscout.color-proposals.v1", "image": meta, "colorOnly": True,
        "detector": {"model": "lab-local-contrast-v1", "resolvedRevision": "v1", "minDeltaE": min_delta_e,
                     "maxProposals": max_color_cues, "enabled": color_sensitive},
        "candidates": colors, "evaluationOnly": evaluation_only,
        "limitations": ["Color cues are habitat follow-ups; fungal presence is unknown."]}
    sources = {source_hash: {"width": original.width, "height": original.height}}
    leads = candidate_leads(detector_report, sources) + candidate_leads(color_report, sources)
    archive = {"schema": "mycoscout.archival-scout-leads.v2", "simulationOnly": True, "actuationEnabled": False,
               "observations": leads, "sourceCredits": credits,
               "summary": {"visualCandidates": len(selected), "colorOnlyCues": len(colors), "metricTargets": 0}}
    for filename, report in (("detector-candidates.json", detector_report), ("color-proposals.json", color_report),
                             ("archive-leads.json", archive)):
        write_new(output_dir / filename, report)
    summary = {"schema": "mycoscout.scout-frame.v1", "originalImageSha256": source_hash,
        "colorSensitive": color_sensitive, "detectorPasses": len(comparisons), "candidateCount": len(selected),
        "colorCueCount": len(colors), "evaluationOnly": evaluation_only, "metricTargets": 0,
        "actuationEnabled": False, "humanConfirmationRequired": True,
        "analysisSeconds": time.perf_counter() - started, "sourceCredits": credits, "reviewUploads": []}
    if station is not None:
        for name, report in (("detector", detector_report), ("color", color_report)):
            prepared = prepare_candidates(report, image_path, output_dir / f"{name}-review", max_candidates=max_candidates if name == "detector" else max_color_cues,
                session_id=session_id, source="saved_color_cue" if name == "color" else "saved_scout_frame",
                evaluation_only=evaluation_only, source_manifest=source_manifest)
            sent = send_candidates(station, prepared, output_dir / f"{name}-review")
            summary["reviewUploads"].append({"kind": name, **{key: sent[key] for key in ("uploadedCount", "classifiedCount", "classificationErrorCount", "uploadErrorCount")}})
    write_new(output_dir / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/grounding-dino-tiny"))
    parser.add_argument("--color-sensitive", action="store_true", help="Add an experimental enhanced detector pass and separate color-only cues")
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--max-color-cues", type=int, default=12)
    parser.add_argument("--min-delta-e", type=float, default=8)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--station", help="Optional explicit review upload destination; omitted means local files only")
    parser.add_argument("--session-id")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        station = Station(args.station) if args.station else None
        if args.output_dir.exists():
            raise FileExistsError("Output already exists")
        detector = LocalDetector(args.model, device=args.device, threads=args.threads)
        summary = inspect_frame(args.image, args.output_dir, detector, color_sensitive=args.color_sensitive,
            max_candidates=args.max_candidates, max_color_cues=args.max_color_cues, min_delta_e=args.min_delta_e,
            source_manifest=args.source_manifest, evaluation_only=args.evaluation_only, station=station, session_id=args.session_id)
        print(json.dumps(summary))
        return 1 if any(row["uploadErrorCount"] or row["classificationErrorCount"] for row in summary["reviewUploads"]) else 0
    except (ValueError, OSError, RuntimeError, Image.DecompressionBombError) as error:
        print(f"Scout frame failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
