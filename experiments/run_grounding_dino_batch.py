"""Run local-only pretrained mushroom candidate detection over a photo folder.

Loads the model once; writes one review JSON per image and an auditable summary.
No weight training is performed and model scores are not calibrated accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.detect_fungi import candidate_records


def write_json(path, payload):
    with path.open("x", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, allow_nan=False)
        output.write("\n")


def scan_views(width, height, tile_size=0, overlap=0.25):
    """Always include the full image, then edge-anchored overlapping crops."""
    views = [(0, 0, width, height)]
    if tile_size <= 0 or (width <= tile_size and height <= tile_size):
        return views
    stride = max(1, int(tile_size * (1 - overlap)))

    def starts(length):
        if length <= tile_size:
            return [0]
        result = list(range(0, length - tile_size + 1, stride))
        if result[-1] != length - tile_size:
            result.append(length - tile_size)
        return result

    for y in starts(height):
        for x in starts(width):
            view = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            if view != views[0]:
                views.append(view)
    return views


def to_original(candidate, view, view_index):
    x, y, _, _ = view
    local = candidate["bboxPixelsXYXY"]
    return {**candidate, "bboxPixelsXYXY": [local[0] + x, local[1] + y, local[2] + x, local[3] + y],
            "sourceViewIndex": view_index, "sourceBboxPixelsXYXY": list(local)}


def intersection_over_union(first, second):
    left, top = max(first[0], second[0]), max(first[1], second[1])
    right, bottom = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area = (first[2] - first[0]) * (first[3] - first[1])
    area += (second[2] - second[0]) * (second[3] - second[1])
    union = area - intersection
    return intersection / union if union > 0 else 0.0


def class_agnostic_nms(candidates, threshold):
    """Stable score-ordered NMS across related prompt labels and source views."""
    kept = []
    for candidate in sorted(candidates, key=lambda item: item["confidence"], reverse=True):
        if all(intersection_over_union(candidate["bboxPixelsXYXY"], old["bboxPixelsXYXY"]) <= threshold
               for old in kept):
            kept.append(candidate)
    return kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/grounding-dino-tiny")
    parser.add_argument("--prompt", default="a mushroom. a fungus.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--tile-size", type=int, default=0,
                        help="Add full-resolution overlapping crops of this size; 0 means full frame only")
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, choices=[0.5], default=None,
                        help="Explicit class-agnostic NMS baseline; rawCandidates remain available")
    parser.add_argument("--include", nargs="*", help="Optional exact source basenames to compare a fixed subset")
    args = parser.parse_args()
    if not args.input_dir.is_dir():
        parser.error("Input folder does not exist")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Output folder must be empty or new")
    if not 1 <= args.threads <= 64:
        parser.error("Threads must be between 1 and 64")
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in [args.box_threshold, args.text_threshold]):
        parser.error("Thresholds must be between 0 and 1")
    if args.tile_size != 0 and not 256 <= args.tile_size <= 4096:
        parser.error("Tile size must be zero or from 256 to 4096 pixels")
    if not math.isfinite(args.tile_overlap) or not 0 <= args.tile_overlap <= 0.75:
        parser.error("Tile overlap must be from 0 to 0.75")
    paths = sorted(path for path in args.input_dir.rglob("*")
                   if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and path.is_file())
    if args.include is not None:
        paths = [path for path in paths if path.name in args.include]
        if set(args.include) - {path.name for path in paths}:
            parser.error("At least one --include image was not found")
    if not paths:
        parser.error("No supported image files found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    import transformers
    from PIL import Image, ImageOps
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    provenance_path = args.model_dir / "mycoscout-model-provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}
    load_start = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False, use_safetensors=True).to(args.device).eval()
    model_load_seconds = time.perf_counter() - load_start
    prompt = args.prompt.strip().lower()
    if not prompt or len(prompt) > 1000:
        parser.error("Prompt must have 1 to 1000 characters")
    if not prompt.endswith("."):
        prompt += "."
    records, errors = [], []
    print(f"Loaded model in {model_load_seconds:.2f}s; running {len(paths)} images on {args.device}", flush=True)
    for index, path in enumerate(paths):
        try:
            if path.stat().st_size > 50_000_000:
                raise ValueError("Image exceeds 50 MB")
            started = time.perf_counter()
            with Image.open(path) as original:
                if original.width * original.height > 40_000_000:
                    raise ValueError("Image exceeds 40 MP")
                image = ImageOps.exif_transpose(original).convert("RGB")
            width, height = image.size
            view_boxes = scan_views(width, height, args.tile_size, args.tile_overlap)
            if len(view_boxes) > 100:
                raise ValueError("Tiling would exceed 100 views per photo; choose larger tiles")
            raw_candidates, view_records = [], []
            inference_seconds = 0
            for view_index, view_box in enumerate(view_boxes):
                view_image = image if view_index == 0 else image.crop(view_box)
                view_width, view_height = view_image.size
                preprocessing_started = time.perf_counter()
                inputs = processor(images=view_image, text=prompt, return_tensors="pt").to(args.device)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                preprocessing_seconds = time.perf_counter() - preprocessing_started
                inference_start = time.perf_counter()
                with torch.inference_mode():
                    outputs = model(**inputs)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                view_inference_seconds = time.perf_counter() - inference_start
                inference_seconds += view_inference_seconds
                result = processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids, box_threshold=args.box_threshold,
                    text_threshold=args.text_threshold, target_sizes=[(view_height, view_width)])[0]
                labels = result.get("text_labels")
                if labels is None:
                    labels = result.get("labels")
                if labels is None:
                    raise RuntimeError("Detector returned no text label field")
                local_candidates = candidate_records(result["boxes"].detach().cpu().tolist(),
                    result["scores"].detach().cpu().tolist(), labels, view_width, view_height)
                raw_candidates.extend(to_original(candidate, view_box, view_index) for candidate in local_candidates)
                view_records.append({"index": view_index, "kind": "full-frame" if view_index == 0 else "tile",
                                     "boundsOriginalPixelsXYXY": list(view_box),
                                     "tensorShape": list(inputs.pixel_values.shape),
                                     "preprocessingSeconds": preprocessing_seconds,
                                     "inferenceSeconds": view_inference_seconds, "candidateCount": len(local_candidates)})
            raw_candidates.sort(key=lambda candidate: candidate["confidence"], reverse=True)
            candidates = class_agnostic_nms(raw_candidates, args.nms_iou) if args.nms_iou is not None else raw_candidates
            elapsed = time.perf_counter() - started
            source_id = path.relative_to(args.input_dir).as_posix()
            image_sha = hashlib.sha256(path.read_bytes()).hexdigest()
            output_name = f"{index:03d}-{path.stem}-{image_sha[:8]}.json"
            payload = {
                "schema": "mycoscout.unlocated-candidates.v1",
                "createdAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "image": {"path": str(path.resolve()), "sourceId": source_id,
                          "sha256": image_sha, "width": width, "height": height,
                          "pixelConvention": "EXIF-oriented RGB image; x right, y down"},
                "detector": {"model": provenance.get("modelId", str(args.model_dir)),
                             "resolvedRevision": provenance.get("resolvedRevision"),
                             "prompt": prompt, "device": args.device, "boxThreshold": args.box_threshold,
                             "textThreshold": args.text_threshold,
                             "tileSize": args.tile_size, "tileOverlap": args.tile_overlap,
                             "postprocessing": {"method": "class-agnostic-nms" if args.nms_iou is not None else "none",
                                                "iouThreshold": args.nms_iou, "rawCandidatesPreserved": True},
                             "scoreMeaning": "Uncalibrated phrase matching score; not accuracy or species probability"},
                "runtime": {"inferenceSeconds": inference_seconds, "totalImageSeconds": elapsed,
                            "views": view_records, "cpuThreads": args.threads},
                "candidates": candidates,
                "rawCandidates": raw_candidates,
                "limitations": ["Pretrained inference only; no local weight training.",
                                "No species or edibility identification.",
                                "No pose or metric depth; detections are UNLOCATED.",
                                "Detection count and scores alone do not establish accuracy."],
            }
            write_json(args.output_dir / output_name, payload)
            record = {"sourceId": source_id, "sha256": image_sha, "output": output_name,
                      "width": width, "height": height, "candidateCount": len(candidates),
                      "rawCandidateCount": len(raw_candidates), "viewCount": len(view_records),
                      "maxScore": max((c["confidence"] for c in candidates), default=None),
                      "inferenceSeconds": inference_seconds, "totalImageSeconds": elapsed}
            records.append(record)
            print(json.dumps(record), flush=True)
        except (OSError, ValueError, RuntimeError) as exc:
            error = {"sourceId": str(path.relative_to(args.input_dir)), "error": str(exc)}
            errors.append(error)
            print(json.dumps(error), flush=True)
    summary = {
        "schema": "mycoscout.vision-run.v1", "kind": "pretrained-inference-not-training",
        "model": provenance, "modelLoadSeconds": model_load_seconds,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "device": args.device, "cpuThreads": args.threads},
        "prompt": prompt, "boxThreshold": args.box_threshold, "textThreshold": args.text_threshold,
        "tileSize": args.tile_size, "tileOverlap": args.tile_overlap, "nmsIou": args.nms_iou,
        "imageCount": len(records), "imagesWithCandidates": sum(r["candidateCount"] > 0 for r in records),
        "candidateCount": sum(r["candidateCount"] for r in records),
        "rawCandidateCount": sum(r["rawCandidateCount"] for r in records),
        "meanInferenceSeconds": statistics.mean(r["inferenceSeconds"] for r in records) if records else None,
        "medianInferenceSeconds": statistics.median(r["inferenceSeconds"] for r in records) if records else None,
        "accuracy": None, "accuracyReason": "Requires independent human boxes including negatives; scores are not accuracy.",
        "images": records, "errors": errors,
    }
    write_json(args.output_dir / "summary.json", summary)
    print(f"Wrote {len(records)} detection files and summary; {len(errors)} errors", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
