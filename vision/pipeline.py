"""Local vision research CLI. Run from the project directory: python -m vision.pipeline --help."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from .dataset import SCHEMA, image_path, load_valid, read_json, sha256, split_groups, validate, write_json


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    add = commands.add_parser("ingest", help="Copy one authorized local photo and preserve source/post provenance")
    add.add_argument("--manifest", type=Path, required=True)
    add.add_argument("--image", type=Path, required=True)
    add.add_argument("--id", required=True)
    add.add_argument("--source-url", required=True)
    add.add_argument("--group-id", required=True)
    add.add_argument("--context", choices=("closeup", "context", "far", "unknown"), default="unknown")
    add.add_argument("--source-note", default="User-authorized local research; not licensed for redistribution")
    check = commands.add_parser("validate", help="Verify hashes, labels, geometry, and group split leakage")
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--output", type=Path)
    split = commands.add_parser("split", help="Assign train/validation/test by connected source groups")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", default="mycoscout-v1")
    scale = commands.add_parser("curriculum", help="Generate explicitly synthetic pixel-scale/blur stress-test images")
    scale.add_argument("--manifest", type=Path, required=True)
    scale.add_argument("--output-dir", type=Path, required=True)
    scale.add_argument("--scales", type=float, nargs="+", default=[1, 0.5, 0.25, 0.125])
    scale.add_argument("--blur", type=float, default=0.6)
    scale.add_argument("--jpeg-quality", type=int, default=80)
    scale.add_argument("--mode", choices=("fixed-canvas", "downsample"), default="fixed-canvas")
    evaluate = commands.add_parser("evaluate", help="Score boxes against complete annotations; otherwise show counts only")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--confidence", type=float, default=0.35)
    evaluate.add_argument("--iou", type=float, default=0.5)
    evaluate.add_argument("--split", choices=("train", "validation", "test"))
    evaluate.add_argument("--nms-iou", type=float, help="Optional explicit class-agnostic duplicate-box suppression; omitted by default")
    train = commands.add_parser("train-verifier", help="Train a candidate-crop baseline from explicit positive/background labels")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--checkpoint", type=Path, required=True)
    train.add_argument("--report", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=1000)
    score = commands.add_parser("score-verifier", help="Score existing detector candidates; does not discover mushrooms")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--checkpoint", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    return p


def ingest(args):
    from PIL import Image, ImageOps
    data = load_valid(args.manifest) if args.manifest.exists() else {"schema": SCHEMA, "images": []}
    if any(record["id"] == args.id for record in data["images"]):
        raise ValueError("Duplicate image ID")
    digest = sha256(args.image)
    if any(record["sha256"] == digest for record in data["images"]):
        raise ValueError("This exact image is already imported")
    with Image.open(args.image) as image:
        width, height = ImageOps.exif_transpose(image).size
        if width * height > 40_000_000:
            raise ValueError("Image exceeds 40 megapixels")
        extension = "." + (image.format or "jpg").lower().replace("jpeg", "jpg")
    relative = Path("images") / (digest[:24] + extension)
    destination = args.manifest.resolve().parent / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != args.image.resolve():
        shutil.copy2(args.image, destination)
    data["images"].append({"id": args.id, "path": relative.as_posix(), "sha256": digest,
                           "originalSha256": digest, "groupId": args.group_id, "sourceUrl": args.source_url,
                           "sourceNote": args.source_note, "context": args.context, "width": width, "height": height,
                           "synthetic": False, "distanceM": None, "annotationStatus": "unreviewed", "annotations": []})
    write_json(args.manifest, data)
    return {"status": "imported", "imageId": args.id, "sha256": digest, "annotationStatus": "unreviewed"}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "ingest":
            result = ingest(args)
        elif args.command == "validate":
            result = validate(args.manifest)
            if args.output:
                write_json(args.output, result)
        else:
            data = load_valid(args.manifest)
            if args.command == "split":
                if args.output.resolve().parent != args.manifest.resolve().parent:
                    raise ValueError("Split output must share the source manifest directory so relative image paths remain valid")
                result = split_groups(data, args.seed)
                write_json(args.output, result)
                result = {"status": "split", "images": len(result["images"]), "splitPolicy": result["splitPolicy"]}
            elif args.command == "curriculum":
                from .curriculum import make_curriculum
                result = make_curriculum(data, args.manifest, args.output_dir, args.scales, args.blur, args.jpeg_quality, args.mode)
                result = {"status": "synthetic-stress-set-created", "images": len(result["images"]), "manifest": str(args.output_dir / "manifest.json")}
            elif args.command == "evaluate":
                from .metrics import evaluate, load_predictions
                predictions, detector = load_predictions(args.predictions, data["images"])
                result = evaluate(data, predictions, detector, args.confidence, args.iou, args.split, args.nms_iou)
                write_json(args.output, result)
                result = {key: result[key] for key in ("status", "selectedImages", "evaluatedImages", "skipped", "real", "synthetic")}
            elif args.command == "train-verifier":
                from .verifier import train
                checkpoint, result = train(data, args.manifest, args.epochs)
                write_json(args.checkpoint, checkpoint)
                write_json(args.report, result)
            elif args.command == "score-verifier":
                from PIL import Image, ImageOps
                from .metrics import load_predictions
                from .verifier import score_image
                from .dataset import valid_box
                checkpoint = read_json(args.checkpoint)
                predictions, detector = load_predictions(args.predictions, data["images"])
                rows = []
                for record in data["images"]:
                    prediction = predictions.get(record["id"])
                    if prediction is None or prediction.get("status") != "ok":
                        continue
                    boxes = [c["bboxPixelsXYXY"] for c in prediction["candidates"]]
                    if not all(valid_box(box, record["width"], record["height"]) for box in boxes):
                        raise ValueError(f"Invalid candidate box for {record['id']}")
                    with Image.open(image_path(args.manifest, record)) as image:
                        scores = score_image(ImageOps.exif_transpose(image).convert("RGB"), boxes, checkpoint)
                    rows.append({"imageId": record["id"], "status": "ok", "candidateScores": scores})
                result = {"schema": "mycoscout.verifier-scores.v1", "images": rows, "detector": detector,
                          "trainingManifestSha256": checkpoint["trainingManifestSha256"],
                          "limitations": checkpoint["limitations"]}
                write_json(args.output, result)
        import json
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0 if result.get("valid", True) else 2
    except (ValueError, OSError, KeyError, TypeError, ImportError) as exc:
        print(f"Vision pipeline failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
