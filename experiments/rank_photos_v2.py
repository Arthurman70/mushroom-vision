"""Rank local photos or existing candidate crops for review without removing boxes.

Examples:
  python experiments/rank_photos_v2.py --input-dir photos --output-dir ranked
  python experiments/rank_photos_v2.py --input-dir photos --detections detections --output-dir ranked-candidates
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.dataset import valid_box
from vision.transfer import checksum, load_backbone, rank_photos, read_json, validate_checkpoint, write_json


def load_image(path):
    if path.stat().st_size > 50_000_000:
        raise ValueError("Image file exceeds 50 MB")
    with Image.open(path) as source:
        if source.width * source.height > 40_000_000:
            raise ValueError("Image exceeds 40 MP")
        return ImageOps.exif_transpose(source).convert("RGB")


def rank_batch(images, checkpoint, backbone, transform):
    results = []
    for start in range(0, len(images), 16):
        results.extend(rank_photos(images[start:start + 16], checkpoint, backbone, transform))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--detections", type=Path, help="GroundingDINO JSON file or directory; all candidate boxes are preserved")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "vision/checkpoints/black-trumpet-ranker-v2.json")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/newfeaturecache/resnet18-f37072fd.pth")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not args.input_dir.is_dir() or not 1 <= args.threads <= 64:
        parser.error("Provide an existing input directory and1–64CPUthreads")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Choose a new or empty output directory")
    files = sorted(path for path in args.input_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        parser.error("No supported photos found")
    checkpoint = read_json(args.checkpoint)
    family = validate_checkpoint(checkpoint)
    checkpoint_sha = checksum(args.checkpoint)
    torch.set_num_threads(args.threads)
    backbone, transform = load_backbone(args.weights) if family == "resnet18" else (None, None)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started, records, candidate_count = time.perf_counter(), [], 0
    provenance = {"featureFamily": family, "checkpointSha256": checkpoint_sha,
                  "trainingManifestSha256": checkpoint["trainingManifestSha256"],
                  "reviewThreshold": checkpoint["head"]["threshold"], "probabilityCalibrated": False,
                  "flightGateEnabled": False, "candidateFilteringEnabled": False,
                  "distributionWarning": "Trained on weak whole-photo labels of fungi; detector crops and forest background are a different distribution."}
    if args.detections is None:
        for path in files:
            image = load_image(path)
            result = rank_photos([image], checkpoint, backbone, transform)[0]
            records.append({"sourceId": path.relative_to(args.input_dir).as_posix(), "sha256": checksum(path),
                            "width": image.width, "height": image.height, **result})
        write_json(args.output_dir / "photo-scores.json", {"schema": "mycoscout.ranked-photos.v2", "ranker": provenance, "photos": records})
    else:
        by_sha = {checksum(path): path for path in files}
        prediction_files = [args.detections] if args.detections.is_file() else sorted(args.detections.glob("*.json"))
        for prediction_path in prediction_files:
            payload = read_json(prediction_path)
            if payload.get("schema") != "mycoscout.unlocated-candidates.v1":
                continue
            image_meta = payload.get("image", {})
            path = by_sha.get(image_meta.get("sha256"))
            if path is None:
                raise ValueError(f"Prediction source hash not found inside the authorized input directory: {prediction_path.name}")
            image = load_image(path)
            if (image.width, image.height) != (image_meta.get("width"), image_meta.get("height")):
                raise ValueError("Prediction image dimensions differ from the verified source")
            candidates = payload.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError("Predictions must contain an explicit candidates list")
            crops = []
            for candidate in candidates:
                box = candidate.get("bboxPixelsXYXY")
                if not valid_box(box, image.width, image.height):
                    raise ValueError("Invalid candidate box")
                crops.append(image.crop((math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3]))))
            scores = rank_batch(crops, checkpoint, backbone, transform) if crops else []
            ranked = deepcopy(payload)
            ranked["reviewRanker"] = provenance
            for candidate, score in zip(ranked["candidates"], scores):
                candidate["reviewRanking"] = score
            if [row["bboxPixelsXYXY"] for row in ranked["candidates"]] != [row["bboxPixelsXYXY"] for row in candidates]:
                raise RuntimeError("Candidate boxes were changed unexpectedly")
            write_json(args.output_dir / prediction_path.name, ranked)
            records.append({"sourceId": path.name, "predictionFile": prediction_path.name, "candidateCountBefore": len(candidates),
                            "candidateCountAfter": len(ranked["candidates"]),
                            "prioritizedForReview": sum(row["prioritizeHumanReview"] for row in scores)})
            candidate_count += len(candidates)
        if not records:
            raise ValueError("No supported candidate prediction JSON found")
    report = {"schema": "mycoscout.review-ranking-run.v2", "mode": "candidate-crop-review" if args.detections else "whole-photo-review",
              "ranker": provenance, "images": len(records), "preservedCandidates": candidate_count,
              "elapsedSeconds": time.perf_counter() - started, "records": records, "limitations": checkpoint["limitations"]}
    write_json(args.output_dir / "ranking-summary.json", report)
    print(json.dumps({key: report[key] for key in ("mode", "images", "preservedCandidates", "elapsedSeconds")}, indent=2))


if __name__ == "__main__":
    main()
