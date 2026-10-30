"""Detect unverified fungi candidates in one image with GroundingDINO.

This CLI writes pixel boxes for review only. A single RGB image cannot supply
metric world coordinates, species certainty, edibility or a safe walking route.
Optional model dependencies are loaded only after command-line validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

DEFAULT_MODEL = "IDEA-Research/grounding-dino-tiny"


def candidate_records(boxes, scores, labels, width, height):
    """Validate model outputs; omit invalid/degenerate boxes instead of inventing a pose."""
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError("Model output lengths do not match")
    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        if len(box) != 4 or not all(math.isfinite(float(v)) for v in [*box, score]):
            continue
        if not 0 <= float(score) <= 1:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        x1, x2 = max(0.0, min(width, x1)), max(0.0, min(width, x2))
        y1, y2 = max(0.0, min(height, y1)), max(0.0, min(height, y2))
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append({
            "label": "Unverified fungi candidate",
            "modelLabel": str(label),
            "confidence": float(score),
            "bboxPixelsXYXY": [x1, y1, x2, y2],
            "status": "UNLOCATED",
            "reviewed": False,
            "poseKnown": False,
            "depthKnown": False,
            "species": None,
            "edibility": "not assessed",
        })
    candidates.sort(key=lambda item: item["confidence"], reverse=True)
    return candidates


def detect(args):
    try:
        import torch
        from PIL import Image, ImageOps
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Optional detector dependencies are missing. Install adapters/requirements-vision.txt in a separate Python 3.10/3.11/3.12 environment.") from exc
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")
    load_options = {
        "local_files_only": not args.allow_model_download,
        "revision": args.revision,
        "trust_remote_code": False,
    }
    if args.cache_dir:
        load_options["cache_dir"] = str(args.cache_dir)
    try:
        processor = AutoProcessor.from_pretrained(args.model, **load_options)
        # Require safetensors. No remote model Python or pickle checkpoints.
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            args.model, use_safetensors=True, **load_options
        ).to(device).eval()
    except OSError as exc:
        raise RuntimeError("Model unavailable. Supply a local model directory or explicitly allow its first download with --allow-model-download.") from exc
    with Image.open(args.image) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    width, height = image.size
    if width * height > 40_000_000:
        raise ValueError("Image exceeds 40 megapixels; provide a smaller image or meaningful crops")
    # GroundingDINO text queries use lowercase phrases ending with a period.
    prompt = args.prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        output, inputs.input_ids, box_threshold=args.box_threshold,
        text_threshold=args.text_threshold, target_sizes=[(height, width)]
    )[0]
    labels = result.get("text_labels")
    if labels is None:
        labels = result.get("labels")
    if labels is None:
        raise RuntimeError("Detector returned no label field; check the pinned Transformers version")
    candidates = candidate_records(
        result["boxes"].detach().cpu().tolist(),
        result["scores"].detach().cpu().tolist(), labels, width, height,
    )
    return {
        "schema": "mycoscout.unlocated-candidates.v1",
        "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "image": {
            "path": str(args.image.resolve()),
            "sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
            "width": width, "height": height,
            "pixelConvention": "EXIF-oriented RGB image; x right, y down",
        },
        "detector": {
            "model": args.model, "requestedRevision": args.revision,
            "resolvedRevision": getattr(model.config, "_commit_hash", None),
            "prompt": prompt, "device": device,
            "boxThreshold": args.box_threshold, "textThreshold": args.text_threshold,
            "scoreMeaning": "Uncalibrated model score, not probability of species correctness",
        },
        "candidates": candidates,
        "limitations": [
            "Candidate presence may be wrong; no field validation has been performed.",
            "No camera intrinsics, depth or synchronized pose was used; map coordinates are unknown.",
            "Species and edibility are not determined. These results are not food-safety advice.",
            "This file is not a /api/observations payload and is never uploaded automatically.",
        ],
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--image", type=Path, required=True, help="Local RGB image; no image is uploaded")
    result.add_argument("--output", type=Path, required=True, help="New JSON review file (will not overwrite)")
    result.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model ID or local snapshot directory")
    result.add_argument("--revision", default="main", help="Model revision; use a verified commit hash for repeatability")
    result.add_argument("--cache-dir", type=Path, help="Model download cache")
    result.add_argument("--allow-model-download", action="store_true",
                        help="Permit downloading the selected pretrained model; otherwise local cache only")
    result.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    result.add_argument("--prompt", default="a mushroom. a fungus.", help="Lowercase object phrases; output remains unverified")
    result.add_argument("--box-threshold", type=float, default=0.35)
    result.add_argument("--text-threshold", type=float, default=0.25)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        for value in (args.box_threshold, args.text_threshold):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Detection thresholds must be finite values from 0 to 1")
        if not args.prompt.strip() or len(args.prompt) > 1000:
            raise ValueError("Provide an object prompt between 1 and 1000 characters")
        if not args.image.is_file():
            raise ValueError("Input image does not exist")
        if args.image.stat().st_size > 50_000_000:
            raise ValueError("Image file exceeds 50 MB")
        if args.output.exists():
            raise ValueError("Output exists; choose a new review filename")
        if not args.output.parent.is_dir():
            raise ValueError("Output parent directory does not exist")
        payload = detect(args)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Saved {len(payload['candidates'])} UNLOCATED candidates to {args.output}")
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Detection failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

