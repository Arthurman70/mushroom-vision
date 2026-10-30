"""Freeze reviewed detection labels into leakage-grouped, versioned datasets.

This adapter consumes only a station-produced local snapshot, never remote URLs.
Evaluation-only sources remain test-only, even when duplicate pixels or manually
linked sessions also occur in training-eligible records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps

SCHEMA = "mycoscout.reviewed-detection-dataset.v1"
MINIMUMS = {"train": 20, "validation": 5, "test": 5}
POSITIVE_MINIMUMS = {"train": 10, "validation": 2, "test": 2}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_json(path, data):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def inside(root, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("Expected a relative staged image path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Staged image escaped its snapshot directory")
    return path


def readiness(rows):
    counts = Counter(row["split"] for row in rows)
    positives = Counter(row["split"] for row in rows if row["boxes"])
    groups = {split: len({r["leakageGroup"] for r in rows if r["split"] == split}) for split in MINIMUMS}
    reasons = []
    for split, minimum in MINIMUMS.items():
        if counts[split] < minimum:
            reasons.append(f"{split} needs at least {minimum} complete images; currently {counts[split]}")
        if positives[split] < POSITIVE_MINIMUMS[split]:
            reasons.append(f"{split} needs at least {POSITIVE_MINIMUMS[split]} mushroom-positive images; currently {positives[split]}")
        if groups[split] < 1:
            reasons.append(f"{split} needs an independent source/session group")
    return {"ready": not reasons, "reasons": reasons, "images": {k: counts[k] for k in MINIMUMS},
            "positiveImages": {k: positives[k] for k in MINIMUMS}, "groups": groups,
            "minimumImages": MINIMUMS, "minimumPositiveImages": POSITIVE_MINIMUMS,
            "notice": "These minimums prevent empty or trivial runs; they do not establish woodland performance."}


class Union:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        self.parent[self.find(b)] = self.find(a)


def freeze(snapshot_path, destination):
    snapshot_path, destination = Path(snapshot_path).resolve(), Path(destination).resolve()
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if snapshot.get("schema") != "mycoscout.reviewed-detection-snapshot.v1":
        raise ValueError("Expected a station-reviewed detection snapshot")
    source_rows = snapshot.get("records")
    if not isinstance(source_rows, list) or len(source_rows) > 1000:
        raise ValueError("Expected at most 1000 reviewed records")
    if destination.exists():
        raise ValueError("Dataset destinations are immutable; choose a new release directory")
    destination.mkdir(parents=True)
    write_json(destination / "INCOMPLETE.json", {"status": "INCOMPLETE"})
    (destination / "images").mkdir()
    rows, image_bytes, seen_ids = [], {}, set()
    total_bytes = 0
    evaluation_pixels, evaluation_groups = set(), set()
    evaluations = snapshot.get("evaluationSources", [])
    if not isinstance(evaluations, list) or len(evaluations) > 3000:
        raise ValueError("Evaluation quarantine exceeds its limit")
    for source in evaluations:
        raw = inside(snapshot_path.parent, source["stagedPath"]).read_bytes()
        total_bytes += len(raw)
        if len(raw) > 8*1024*1024 or total_bytes > 512*1024*1024 or digest(raw) != source["originalSha256"]:
            raise ValueError("Evaluation quarantine image integrity failed")
        with Image.open(__import__("io").BytesIO(raw)) as evaluation:
            if evaluation.width*evaluation.height > 20_000_000:
                raise ValueError("Evaluation quarantine image is too large")
            rgb = ImageOps.exif_transpose(evaluation).convert("RGB")
            evaluation_pixels.add(digest(canonical(list(rgb.size)) + rgb.tobytes()))
        evaluation_groups.update(source.get("groupKeys", []))
    for source in source_rows:
        row = dict(source)
        if not isinstance(row.get("id"), str) or row["id"] in seen_ids:
            raise ValueError("Duplicate or missing reviewed record identity")
        seen_ids.add(row["id"])
        if row.get("complete") is not True or row.get("humanVerified") is not True or not row.get("reviewer"):
            raise ValueError("Only complete human-reviewed records can be frozen")
        if row.get("status") not in {"visible", "absent"}:
            raise ValueError("Uncertain or missing visibility is not a label")
        if not isinstance(row.get("evaluationOnly"), bool) or (not row["evaluationOnly"] and row.get("trainingAllowed") is not True):
            raise ValueError("Training permission is missing")
        if not isinstance(row.get("groupKeys"), list) or not row["groupKeys"] or any(not isinstance(k, str) or not k for k in row["groupKeys"]):
            raise ValueError("Source/session grouping is required")
        raw = inside(snapshot_path.parent, row.pop("stagedPath")).read_bytes()
        total_bytes += len(raw)
        if len(raw) > 8 * 1024 * 1024 or total_bytes > 512 * 1024 * 1024 or digest(raw) != row.get("originalSha256"):
            raise ValueError("Staged image hash or size is invalid")
        with Image.open(__import__("io").BytesIO(raw)) as source_image:
            if source_image.width * source_image.height > 20_000_000 or getattr(source_image, "n_frames", 1) != 1:
                raise ValueError("Invalid still-image dimensions")
            rgb = ImageOps.exif_transpose(source_image).convert("RGB")
            if list(rgb.size) != [row.get("width"), row.get("height")]:
                raise ValueError("Image dimensions disagree with label coordinates")
            # Decoded oriented-pixel hashes catch duplicate images with different JPEG metadata.
            row["pixelSha256"] = digest(canonical(list(rgb.size)) + rgb.tobytes())
            small = rgb.resize((16, 16)).convert("L")
            values = list(small.getdata())
            mean = sum(values) / len(values)
            row["averageHash"] = format(sum((v >= mean) << i for i, v in enumerate(values)), "064x")
            row["colorMean"] = [round(sum(rgb.resize((16, 16)).getdata(band)) / 256, 3) for band in range(3)]
            row["imagePath"] = "images/" + row["pixelSha256"] + ".png"
            if row["pixelSha256"] not in image_bytes:
                rgb.info.clear()
                rgb.save(destination / row["imagePath"], "PNG")
                image_bytes[row["pixelSha256"]] = digest((destination / row["imagePath"]).read_bytes())
            row["imageSha256"] = image_bytes[row["pixelSha256"]]
        boxes = row.get("boxes")
        if not isinstance(boxes, list) or len(boxes) > 200 or bool(boxes) != (row["status"] == "visible"):
            raise ValueError("Visibility and complete boxes disagree")
        for box in boxes:
            b = box.get("bboxPixelsXYXY")
            if (not isinstance(b, list) or len(b) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in b)
                    or not 0 <= b[0] < b[2] <= row["width"] or not 0 <= b[1] < b[3] <= row["height"]):
                raise ValueError("Box lies outside the original oriented frame")
        rows.append(row)
    union, index = Union(len(rows)), {}
    for i, row in enumerate(rows):
        for key in row["groupKeys"] + ["pixels:" + row["pixelSha256"], "original:" + row["originalSha256"]]:
            if key in index:
                union.join(i, index[key])
            else:
                index[key] = i
    # Conservative near-duplicate grouping is not a proof of scene independence.
    near_pairs = 0
    for i, row in enumerate(rows):
        for j in range(i):
            other = rows[j]
            if ((int(row["averageHash"], 16) ^ int(other["averageHash"], 16)).bit_count() <= 6 and
                    max(abs(a - b) for a, b in zip(row["colorMean"], other["colorMean"])) <= 8):
                union.join(i, j)
                near_pairs += 1
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[union.find(i)].append(row)
    blocked = set(snapshot.get("evaluationHashes", []))
    previous = snapshot.get("frozenPartitions", {})
    for members in components.values():
        group = "group-" + digest(canonical(sorted({key for row in members for key in row["groupKeys"]})))[:24]
        restricted = any(row["evaluationOnly"] or row["originalSha256"] in blocked or row["pixelSha256"] in evaluation_pixels or
                         evaluation_groups.intersection(row["groupKeys"]) for row in members)
        number = int(digest(group.encode())[:8], 16) % 100
        prior = {previous.get(key) for row in members for key in row["groupKeys"]}
        split = "test" if restricted or "test" in prior else "validation" if "validation" in prior else "train" if number < 70 else "validation" if number < 85 else "test"
        for row in members:
            row.update(leakageGroup=group, split=split, evaluationQuarantine=restricted)
    # One pixel-identical image contributes once. Conflicting human boxes are
    # excluded instead of inventing a winner or silently duplicating targets.
    by_pixels = defaultdict(list)
    for row in rows:
        by_pixels[row["pixelSha256"]].append(row)
    selected, conflicts, duplicates = [], [], 0
    for pixels, members in by_pixels.items():
        signatures = {digest(canonical(sorted([b["bboxPixelsXYXY"] for b in r["boxes"]]))) for r in members}
        if len(signatures) != 1:
            conflicts.append({"pixelSha256": pixels, "recordIds": [r["id"] for r in members]})
            continue
        selected.append(members[0] | {"relatedRecordIds": [r["id"] for r in members]})
        duplicates += len(members) - 1
    selected.sort(key=lambda row: row["id"])
    manifest = {"schema": SCHEMA, "status": "READY", "snapshotSha256": digest(snapshot_path.read_bytes()),
                "createdAt": snapshot["createdAt"], "records": selected, "readiness": readiness(selected),
                "classes": {"0": "background", "1": "mushroom"}, "labelType": "human-boxes-class-agnostic",
                "excludedConflictingImages": conflicts, "duplicatesCollapsed": duplicates, "nearDuplicatePairsGrouped": near_pairs,
                "sourceRevisionSignatures": snapshot.get("sourceRevisionSignatures", {}),
                "policies": {"evaluationSourcesTestOnly": True, "sourceSessionGrouping": True,
                    "exactPixelDuplicatesGrouped": True, "nearDuplicateHeuristic": "256-bit average hash distance <=6 and RGB mean difference <=8",
                    "nearDuplicateGroupingIsProof": False, "automaticModelActivation": False},
                "limitations": ["Separate camera angles of one site may escape the duplicate heuristic. Link related sessions before export.",
                    "Measured image detection scores do not establish flight distance or species identification accuracy."]}
    manifest["datasetId"] = "dataset-" + digest(canonical(manifest))[:24]
    write_json(destination / "manifest.json", manifest)
    for split in MINIMUMS:
        subset = [row for row in selected if row["split"] == split]
        coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "mushroom"}]}
        for n, row in enumerate(subset, 1):
            coco["images"].append({"id": n, "file_name": row["imagePath"], "width": row["width"], "height": row["height"]})
            for box in row["boxes"]:
                x1, y1, x2, y2 = box["bboxPixelsXYXY"]
                coco["annotations"].append({"id": len(coco["annotations"]) + 1, "image_id": n, "category_id": 1,
                    "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0})
        write_json(destination / (split + ".coco.json"), coco)
    write_json(destination / "READY.json", {"manifestSha256": digest((destination / "manifest.json").read_bytes())})
    (destination / "INCOMPLETE.json").unlink()
    return {"datasetId": manifest["datasetId"], "readiness": manifest["readiness"], "images": len(selected),
            "excludedConflictingImages": len(conflicts), "duplicatesCollapsed": duplicates, "status": "READY"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(freeze(args.snapshot, args.output)))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Reviewed dataset export failed: {exc}\n")


if __name__ == "__main__":
    main()
