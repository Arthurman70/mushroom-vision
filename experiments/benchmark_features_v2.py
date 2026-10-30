"""Time frozen feature extraction only; never fit/evaluate on a partial dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.transfer import EXPECTED_WEIGHT_SHA, load_backbone, photo_path, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    source = args.manifest.read_bytes()
    data = json.loads(source.decode("utf-8-sig"))
    rows = data["images"][:args.limit]
    torch.set_num_threads(4)
    model, transform = load_backbone(ROOT / "models/newfeaturecache/resnet18-f37072fd.pth")
    batch, total_inference, total_photos = [], 0.0, 0
    started = time.perf_counter()
    for index, row in enumerate(rows):
        path = photo_path(args.manifest, row)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Photo hash changed")
        with Image.open(path) as source_image:
            batch.append(transform(ImageOps.exif_transpose(source_image).convert("RGB")))
        if len(batch) == 16 or index == len(rows) - 1:
            tensor = torch.stack(batch)
            then = time.perf_counter()
            with torch.inference_mode():
                output = model(tensor)
            total_inference += time.perf_counter() - then
            if tuple(output.shape) != (len(batch), 512) or not torch.isfinite(output).all():
                raise ValueError("Invalid frozen features")
            total_photos += len(batch)
            batch = []
    report = {"kind": "partial-dataset-runtime-check-not-training-or-accuracy",
              "sourceManifestSnapshotSha256": hashlib.sha256(source).hexdigest(), "photos": total_photos,
              "imageIds": [row["id"] for row in rows], "officialWeightSha256": EXPECTED_WEIGHT_SHA,
              "featureDimensions": 512, "backboneInferenceSeconds": total_inference,
              "inferenceSecondsPerPhoto": total_inference / total_photos,
              "completeSeconds": time.perf_counter() - started, "batchSize": 16, "cpuThreads": 4}
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "imageIds"}, indent=2))


if __name__ == "__main__":
    main()
