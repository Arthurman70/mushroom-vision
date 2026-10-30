"""Download a bounded LILA Caltech sample; animal-empty never means fungi-absent."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import urllib.request
import zipfile

from PIL import Image, ImageOps

SOURCE = "https://lila.science/datasets/caltech-camera-traps"
LICENSE = "https://cdla.io/permissive-1-0/"
ANNOTATIONS = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/labels/caltech_camera_traps.json.zip"
SPLITS = "https://storage.googleapis.com/public-datasets-lila/caltechcameratraps/CaltechCameraTrapsSplits_v0.json"
IMAGES = "https://lilawildlife.blob.core.windows.net/lila-wildlife/caltech-unzipped/cct_images/"


def fetch(url, limit):
    request = urllib.request.Request(url, headers={"User-Agent": "MushroomVision/0.1 bounded-open-data-research"})
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Bounded response limit exceeded")
    return content


def select(metadata, split_data, per_category):
    categories = {c["id"]: c["name"] for c in metadata["categories"]}
    labels = defaultdict(set)
    for row in metadata["annotations"]:
        labels[row["image_id"]].add(categories[row["category_id"]])
    seen_sequences, counts, locations, selected = set(), Counter(), Counter(), []
    train = set(split_data["splits"]["train"])
    validation = set(split_data["splits"]["val"])
    if train & validation:
        raise ValueError("Source camera splits overlap")
    # Stable selection deliberately spans different camera locations and sequences.
    for row in sorted(metadata["images"], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest()):
        classes = sorted(labels.get(row["id"], []))
        if len(classes) != 1 or classes[0] in {"car", "person", "human"}:
            continue
        category, location = classes[0], str(row["location"])
        if counts[category] >= per_category or locations[(category, location)] >= 2 or row["seq_id"] in seen_sequences:
            continue
        if not re.fullmatch(r"[a-f0-9-]{36}\.jpg", row["file_name"]):
            raise ValueError("Unexpected image path in source metadata")
        split = "train" if location in train else "validation" if location in validation else None
        if split is None:
            continue
        selected.append({"id": "cct-" + row["id"], "path": "images/" + row["file_name"],
                         "sourceImageId": row["id"], "sequenceId": row["seq_id"], "groupId": "cct-camera-" + location,
                         "split": split, "sourceCategory": category, "sourceUrl": SOURCE,
                         "downloadUrl": IMAGES + row["file_name"], "rightsHolder": row.get("rights_holder"),
                         "license": {"code": "CDLA-Permissive-1.0", "url": LICENSE},
                         "annotationStatus": "unreviewed", "mushroomVisibility": "unknown", "annotations": [],
                         "usage": "wildlife-context-review", "trainingEligibleForMushroomDetection": False})
        counts[category] += 1; locations[(category, location)] += 1; seen_sequences.add(row["seq_id"])
    return selected


def acquire(root, per_category=6):
    root = Path(root).resolve()
    if (root / "manifest.json").exists():
        raise ValueError("Frozen collection exists; choose a new directory")
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    metadata_files = {}
    for name, url, limit in (("annotations.zip", ANNOTATIONS, 15_000_000), ("splits.json", SPLITS, 100_000)):
        path = root / name
        if not path.exists():
            path.write_bytes(fetch(url, limit))
        metadata_files[name] = {"url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
    with zipfile.ZipFile(root / "annotations.zip") as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size > 200_000_000:
            raise ValueError("Unexpected source metadata archive")
        metadata = json.loads(archive.read(members[0]))
    rows = select(metadata, json.loads((root / "splits.json").read_text()), per_category)
    def download(row):
        path = root / row["path"]
        payload = path.read_bytes() if path.exists() else fetch(row["downloadUrl"], 8_000_000)
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > 20_000_000:
                raise ValueError("Unexpected image dimensions")
            rgb = ImageOps.exif_transpose(image).convert("RGB")
            row.update(width=rgb.width, height=rgb.height, pixelSha256=hashlib.sha256(rgb.tobytes()).hexdigest())
        if not path.exists():
            path.write_bytes(payload)
        row.update(bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest())
        return row
    accepted, skipped = [], []
    with ThreadPoolExecutor(max_workers=3) as executor:
        for candidate, future in [(r, executor.submit(download, r)) for r in rows]:
            try:
                accepted.append(future.result())
            except (OSError, ValueError) as exc:
                skipped.append({"id": candidate["id"], "reason": str(exc)})
    if len({r["sha256"] for r in accepted}) != len(accepted):
        raise ValueError("Duplicate bytes require grouping review before freezing")
    pixel_partitions = defaultdict(set)
    for row in accepted:
        pixel_partitions[row["pixelSha256"]].add(row["split"])
    if any(len(partitions) > 1 for partitions in pixel_partitions.values()):
        raise ValueError("Identical image pixels cross source camera partitions")
    manifest = {"schema": "mushroom-vision.wildlife-context.v1", "status": "frozen",
                "createdAt": datetime.now(timezone.utc).isoformat(), "source": SOURCE,
                "citation": "Sara Beery, Grant Van Horn, Pietro Perona. Recognition in Terra Incognita. ECCV 2018.",
                "license": {"code": "CDLA-Permissive-1.0", "url": LICENSE},
                "metadata": metadata_files, "images": accepted, "skipped": skipped,
                "summary": {"images": len(accepted), "bytes": sum(r["bytes"] for r in accepted),
                            "categories": dict(Counter(r["sourceCategory"] for r in accepted)),
                            "cameraGroups": len({r["groupId"] for r in accepted}),
                            "splitCounts": dict(Counter(r["split"] for r in accepted))},
                "limitations": ["Empty means no labeled camera-trap animal; it does not mean no mushroom.",
                    "All mushroom visibility remains unknown until a human examines the frame.",
                    "Static camera images do not establish drone flight performance or physical detection distance.",
                    "Original official camera train/validation assignments preserved; no independent test set is fabricated."]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (root / "manifest-sha256.json").write_text(json.dumps({"sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()}) + "\n")
    (root / "validation.json").write_text(json.dumps({"valid": True, "errors": [], "imageCount": len(accepted),
        "verifiedImageBytes": sum(r["bytes"] for r in accepted), "allMushroomVisibilityUnknown": True,
        "sourceCameraAndPixelPartitionsPreserved": True}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/wildlife-context-v1"))
    parser.add_argument("--per-category", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.per_category <= 10:
        parser.error("Choose 1–10 images per animal category")
    acquire(args.output, args.per_category)
