"""Verify the frozen public weak-label corpus and train/test grouping."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from vision.acquire_public import BLACK_TAXA, LICENSES, SCHEMA, summary, write_json


def validate(manifest_path, verify_files=True):
    manifest_path = Path(manifest_path)
    root = manifest_path.resolve().parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    if data.get("schema") != SCHEMA or data.get("status") != "frozen":
        errors.append("Only a frozen v2 weak-label manifest is suitable for training")
    records = data.get("images", [])
    relationships = {key: defaultdict(set) for key in ("groupId", "observerId", "observationId", "photoId", "sha256", "normalizedPixelSha256", "splitGroupId")}
    seen_ids = set()
    for row in records:
        image_id = row["id"]
        if image_id in seen_ids:
            errors.append(f"{image_id}: duplicate photo record")
        seen_ids.add(image_id)
        if row.get("weakLabel") not in ("black_trumpet", "other_fungi"):
            errors.append(f"{image_id}: invalid weak label")
        if row.get("annotationStatus") != "unreviewed" or row.get("annotations") != []:
            errors.append(f"{image_id}: weak labels must not silently become detector annotations")
        if row.get("synthetic") is not False or row.get("originalSha256") != row.get("sha256"):
            errors.append(f"{image_id}: original-photo hash mismatch or synthetic record")
        if row.get("photoLicense", {}).get("code") not in LICENSES or row.get("observationLicense", {}).get("code") not in LICENSES:
            errors.append(f"{image_id}: photo or observation license is outside the allowlist")
        if not row.get("photoLicense", {}).get("attribution"):
            errors.append(f"{image_id}: missing attribution")
        if row.get("sourceUrl") != f"https://www.inaturalist.org/observations/{row['observationId']}":
            errors.append(f"{image_id}: source URL does not match observation")
        if any(key in row for key in ("latitude", "longitude", "location", "geojson", "place_guess", "private_location")):
            errors.append(f"{image_id}: unnecessary location information")
        split = row.get("split")
        if split not in ("train", "validation", "test"):
            errors.append(f"{image_id}: missing frozen split")
        for key, mapping in relationships.items():
            mapping[row.get(key)].add(split)
        if row.get("weakLabel") == "black_trumpet" and row.get("queryTaxonId") not in BLACK_TAXA:
            errors.append(f"{image_id}: target taxon is outside explicit black-trumpet label map")
        if verify_files:
            path = (root / row["path"]).resolve()
            if not path.is_relative_to(root):
                errors.append(f"{image_id}: image path escapes dataset")
                continue
            try:
                if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
                    errors.append(f"{image_id}: file SHA-256 mismatch")
                with Image.open(path) as image:
                    if ImageOps.exif_transpose(image).size != (row["width"], row["height"]):
                        errors.append(f"{image_id}: dimensions mismatch")
            except OSError as exc:
                errors.append(f"{image_id}: {exc}")
    for field, mapping in relationships.items():
        for key, splits in mapping.items():
            if key is None or len(splits) != 1:
                errors.append(f"{field} {key}: missing group or cross-split leakage")
    image_lookup = {row["id"]: row for row in records}
    for link in data.get("splitPolicy", {}).get("nearDuplicateLinks", []):
        if image_lookup[link["photoA"]]["split"] != image_lookup[link["photoB"]]["split"]:
            errors.append("Perceptually similar cross-observer images cross splits")
    calculated = summary(records)
    if calculated != data.get("summary"):
        errors.append("Manifest summary does not match its image records")
    return {"schema": "mycoscout.public-dataset-validation.v1", "valid": not errors, "errors": errors,
            "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "summary": calculated,
            "filesVerified": verify_files, "limitations": ["License codes and taxonomy reflect the recorded acquisition snapshot.",
                "Research-grade observation labels do not verify every visible object or provide mushroom-free background ground truth."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.manifest)
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["valid"] else 2)
