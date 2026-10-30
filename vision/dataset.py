"""Dataset provenance, labels, and leakage-resistant image-group splitting."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

SCHEMA = "mycoscout.vision-dataset.v1"
CONTEXTS = {"closeup", "context", "far", "unknown"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_path(manifest_path, record):
    # Dataset files are portable and cannot refer to arbitrary files outside their tree.
    base = Path(manifest_path).resolve().parent
    path = (base / record["path"]).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Image {record.get('id')} escapes the manifest directory")
    return path


def valid_box(box, width, height):
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in box)
            and 0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height)


def validate(manifest_path, verify_files=True):
    data = read_json(manifest_path)
    errors, warnings = [], []
    if data.get("schema") != SCHEMA:
        errors.append(f"schema must be {SCHEMA}")
    records = data.get("images")
    if not isinstance(records, list):
        raise ValueError("Dataset images must be a list")
    seen, originals, groups, hashes = set(), defaultdict(set), defaultdict(set), defaultdict(set)
    for record in records:
        if not isinstance(record, dict):
            errors.append("Each image record must be an object")
            continue
        image_id = record.get("id")
        prefix = f"{image_id}: "
        if not isinstance(image_id, str) or not image_id:
            errors.append(prefix + "missing or duplicate id")
        elif image_id in seen:
            errors.append(prefix + "missing or duplicate id")
        else:
            seen.add(image_id)
        for key in ("path", "groupId", "sourceUrl", "sha256", "originalSha256"):
            if not isinstance(record.get(key), str) or not record[key].strip():
                errors.append(prefix + f"missing {key}")
        for key in ("sha256", "originalSha256"):
            value = record.get(key, "")
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                errors.append(prefix + f"invalid {key}")
        if record.get("context") not in tuple(CONTEXTS):
            errors.append(prefix + "context must be closeup, context, far, or unknown")
        if record.get("annotationStatus") not in ("complete", "partial", "unreviewed"):
            errors.append(prefix + "annotationStatus must be complete, partial, or unreviewed")
        if not isinstance(record.get("synthetic"), bool):
            errors.append(prefix + "synthetic flag must be explicit")
        width, height = record.get("width", 0), record.get("height", 0)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in (width, height)):
            errors.append(prefix + "width/height must be positive integers")
            width = height = 0
        elif width * height > 40_000_000:
            errors.append(prefix + "image exceeds 40 megapixel limit")
        annotations = record.get("annotations", [])
        if not isinstance(annotations, list):
            errors.append(prefix + "annotations must be a list")
            annotations = []
        for box in annotations:
            if not isinstance(box, dict):
                errors.append(prefix + "annotations must contain objects")
                continue
            if box.get("label") not in ("mushroom", "background"):
                errors.append(prefix + "annotation label must be mushroom or background")
            if not valid_box(box.get("bboxPixelsXYXY"), width, height):
                errors.append(prefix + "invalid annotation box")
        if record.get("annotationStatus") in ("complete", "partial") and not record.get("reviewer"):
            errors.append(prefix + "reviewed annotations must name a reviewer or annotation method")
        if record.get("synthetic") and (not record.get("originalId") or not record.get("transform")):
            errors.append(prefix + "synthetic variants require originalId and transform")
        if not record.get("synthetic") and record.get("originalSha256") != record.get("sha256"):
            errors.append(prefix + "original image hash must equal originalSha256")
        if record.get("distanceM") is not None:
            distance = record["distanceM"]
            if not isinstance(distance, (float, int)) or not math.isfinite(distance) or distance <= 0:
                errors.append(prefix + "distanceM must be a measured positive distance or null")
        split = record.get("split")
        if split is not None and split not in ("train", "validation", "test"):
            errors.append(prefix + "invalid split")
        for field, target in (("originalSha256", originals), ("groupId", groups), ("sha256", hashes)):
            value = record.get(field)
            if isinstance(value, str) and isinstance(split, (str, type(None))):
                target[value].add(split)
        if verify_files:
            try:
                from PIL import Image, ImageOps
                path = image_path(manifest_path, record)
                if sha256(path) != record.get("sha256"):
                    errors.append(prefix + "file hash mismatch")
                with Image.open(path) as image:
                    oriented = ImageOps.exif_transpose(image)
                    if oriented.size != (width, height):
                        errors.append(prefix + "EXIF-oriented dimensions mismatch")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append(prefix + str(exc))
    for label, mapping in (("original photo", originals), ("post/group", groups), ("duplicate file", hashes)):
        for key, splits in mapping.items():
            if len(splits) > 1:
                errors.append(f"{label} {key} crosses splits (including partially assigned variants)")
    if not records:
        warnings.append("No images have been imported")
    valid_records = [r for r in records if isinstance(r, dict)]
    if not any(r.get("annotationStatus") == "complete" for r in valid_records):
        warnings.append("No complete annotations: detector precision/recall cannot be computed")
    far_positive = any(not r.get("synthetic") and r.get("context") == "far"
                       and isinstance(r.get("annotations"), list)
                       and any(isinstance(a, dict) and a.get("label") == "mushroom" for a in r["annotations"])
                       for r in valid_records)
    if not far_positive:
        warnings.append("No labeled genuine distant positive images: synthetic reductions cannot establish woodland detection range")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "imageCount": len(records), "groupCount": len(groups),
            "originalCount": len(originals), "contextCounts": dict(Counter(str(r.get("context")) for r in valid_records)),
            "syntheticCount": sum(r.get("synthetic") is True for r in valid_records)}


def load_valid(manifest_path):
    result = validate(manifest_path)
    if not result["valid"]:
        raise ValueError("Invalid dataset: " + "; ".join(result["errors"]))
    return read_json(manifest_path)


def split_groups(data, seed="mycoscout-v1", train_fraction=0.6, validation_fraction=0.2):
    """Join both source-post and original-photo relationships before assigning splits."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must leave positive train, validation, and test fractions")
    records = data["images"]
    parent = {r["id"]: r["id"] for r in records}

    def root(item):
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    indexes = ({}, {}, {})
    for r in records:
        for field, index in zip(("groupId", "originalSha256", "sha256"), indexes):
            previous = index.setdefault(r[field], r["id"])
            parent[root(r["id"])] = root(previous)
    components = defaultdict(list)
    for r in records:
        components[root(r["id"])].append(r)
    if len(components) < 3:
        raise ValueError("At least three independent photo/post groups are required for a held-out split")
    ordered = sorted(components.values(), key=lambda rows: hashlib.sha256(
        (str(seed) + "|" + min(r["originalSha256"] for r in rows)).encode()).hexdigest())
    count = len(ordered)
    train_end = min(count - 2, max(1, round(count * train_fraction)))
    val_end = min(count - 1, train_end + max(1, round(count * validation_fraction)))
    for i, rows in enumerate(ordered):
        split = "train" if i < train_end else "validation" if i < val_end else "test"
        for row in rows:
            row["split"] = split
    data["splitPolicy"] = {"method": "connected post/original/file groups", "seed": seed,
                           "groupCount": count, "trainFractionRequested": train_fraction,
                           "validationFractionRequested": validation_fraction,
                           "warning": "Hash assignment does not guarantee class balance; check per-split counts before training"}
    return data
