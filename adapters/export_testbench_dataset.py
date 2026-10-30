"""Export approved human testbench boxes to an unsplit COCO/YOLO collection.

No model training, uploads, species classes or automatic train/test split. A
READY.json marker is written only after a second rights/revision check succeeds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

from PIL import Image, ImageOps, UnidentifiedImageError
from adapters.review_stream import Station

MAX_IMAGES = 100
MAX_IMAGE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MAX_COLLECTION_BYTES = 1024 * 1024 * 1024
UUID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
SHA_PATTERN = r"[0-9a-f]{64}"


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha(value):
    return hashlib.sha256(value).hexdigest()


def write_new(path, value):
    encoded = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as output:
        output.write(encoded)
    return sha(encoded)


def require_id(value, name):
    if not isinstance(value, str) or not re.fullmatch(UUID_PATTERN, value):
        raise ValueError(f"Invalid {name}")


def records_from(manifest):
    if not isinstance(manifest, dict) or manifest.get("schema") != "mycoscout.testbench-dataset.v1":
        raise ValueError("Unexpected testbench export schema")
    if not isinstance(manifest.get("releaseId"), str) or not re.fullmatch(r"testbench-" + SHA_PATTERN, manifest["releaseId"]):
        raise ValueError("Invalid testbench export release ID")
    rows = manifest.get("records")
    if not isinstance(rows, list) or len(rows) > 1000:
        raise ValueError("Export records must be a list of at most 1000 images")
    seen_hashes, seen_runs = set(), set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Invalid export record")
        require_id(row.get("runId"), "run ID")
        if row["runId"] in seen_runs:
            raise ValueError("Duplicate immutable run ID")
        seen_runs.add(row["runId"])
        if row.get("sourceRunId") is not None:
            require_id(row["sourceRunId"], "source run ID")
        if row.get("evaluationOnly") is not False:
            raise ValueError("Evaluation photos, or missing quarantine status, cannot enter this collection")
        info, annotation = row.get("image"), row.get("annotation")
        if not isinstance(info, dict) or not isinstance(annotation, dict):
            raise ValueError("Image metadata and explicit human annotations are required")
        digest = info.get("sha256")
        if not isinstance(digest, str) or not re.fullmatch(SHA_PATTERN, digest) or digest in seen_hashes:
            raise ValueError("Invalid or repeated original image SHA-256")
        seen_hashes.add(digest)
        if (info.get("mimeType") not in {"image/jpeg", "image/png"} or type(info.get("bytes")) is not int or
                not 1 <= info["bytes"] <= MAX_IMAGE_BYTES or info.get("orientationVerified") is not True):
            raise ValueError("Invalid or unverified original image metadata")
        w, h = info.get("width"), info.get("height")
        if type(w) is not int or type(h) is not int or min(w, h) < 1 or w * h > MAX_IMAGE_PIXELS:
            raise ValueError("Invalid oriented image dimensions")
        expected_url = f"/api/testbench/runs/{row['runId']}/image"
        if info.get("url") != expected_url:
            raise ValueError("Image reference must be the exact same-station run image API path")
        require_id(annotation.get("id"), "annotation ID")
        if type(annotation.get("revision")) is not int or not 1 <= annotation["revision"] <= 200:
            raise ValueError("Invalid annotation revision")
        if annotation.get("complete") is not True or annotation.get("trainingAllowed") is not True:
            raise ValueError("Only complete human annotations with explicit training permission may be exported")
        if annotation.get("labelsFrom") != "explicit_human_annotations":
            raise ValueError("Model guesses cannot become training labels")
        if not isinstance(annotation.get("reviewer"), str) or not annotation["reviewer"].strip():
            raise ValueError("Human reviewer identity is required")
        if not isinstance(annotation.get("rightsAttestation"), str) or not annotation["rightsAttestation"].strip():
            raise ValueError("Explicit photo and annotation rights attestation is required")
        provenance = row.get("provenance")
        if (not isinstance(provenance, dict) or provenance.get("originalImageSha256") != digest or
                provenance.get("labelsFrom") != "explicit_human_annotations"):
            raise ValueError("Original source-scene provenance is inconsistent")
        video = row.get("videoProvenance")
        if video is None:
            if row.get("sceneId") != "scene-" + digest:
                raise ValueError("Original source-scene provenance is inconsistent")
        else:
            source = video.get("source") if isinstance(video, dict) else None
            if (not isinstance(source, dict) or provenance.get("sceneGrouping") != "source-video-group" or
                    not isinstance(row.get("sceneId"), str) or not re.fullmatch(r"[A-Za-z0-9:_-]{1,160}", row["sceneId"]) or
                    row["sceneId"] != video.get("sceneId") or row["sceneId"] != source.get("sourceGroupId") or
                    video.get("sha256") != digest or video.get("evaluationOnly") is not False or
                    source.get("evaluationOnly") is not False or source.get("licenseAllowsTraining") is not True or
                    not isinstance(source.get("videoHash"), str) or not re.fullmatch(SHA_PATTERN, source["videoHash"]) or
                    type(video.get("timestampSeconds")) not in (int, float) or not 0 <= video["timestampSeconds"] <= 86400):
                raise ValueError("Video provenance or training permission is inconsistent")
        related = provenance.get("relatedRunIds")
        if not isinstance(related, list) or not 1 <= len(related) <= 1000 or row["runId"] not in related:
            raise ValueError("Invalid related-run provenance")
        for run_id in related:
            require_id(run_id, "related run ID")
        boxes = annotation.get("boxes")
        if not isinstance(boxes, list) or len(boxes) > 200:
            raise ValueError("Expected at most 200 human boxes per image")
        if not boxes and annotation.get("emptySceneConfirmed") is not True:
            raise ValueError("An empty label set requires explicit empty-scene confirmation")
        box_ids = set()
        for box in boxes:
            if not isinstance(box, dict) or not isinstance(box.get("id"), str) or not 1 <= len(box["id"]) <= 128 or box["id"] in box_ids:
                raise ValueError("Annotation box IDs must be distinct immutable strings")
            box_ids.add(box["id"])
            b = box.get("bboxPixelsXYXY")
            if (not isinstance(b, list) or len(b) != 4 or
                    any(type(v) not in (float, int) or not math.isfinite(v) for v in b) or
                    not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h)):
                raise ValueError("Human box must lie within the original oriented image")
            if not isinstance(box.get("groupId"), str) or not 1 <= len(box["groupId"]) <= 100:
                raise ValueError("Human group metadata is required")
            if not isinstance(box.get("label", ""), str) or len(box.get("label", "")) > 240:
                raise ValueError("Invalid human label metadata")
        # Also reject non-finite numbers anywhere in the exported source metadata.
        canonical(row)
    return rows


def normalized_png(payload, info):
    if not isinstance(payload, bytes) or len(payload) != info["bytes"] or sha(payload) != info["sha256"]:
        raise ValueError("Downloaded image failed its original length or SHA-256 check")
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if ({"JPEG": "image/jpeg", "PNG": "image/png"}.get(source.format) != info["mimeType"] or
                    getattr(source, "n_frames", 1) != 1 or source.width * source.height > MAX_IMAGE_PIXELS):
                raise ValueError("Downloaded image is not a supported bounded still photo")
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            if rgb.size != (info["width"], info["height"]):
                raise ValueError("Downloaded image dimensions disagree with annotation coordinates")
            # PNG encodes the decoded, oriented RGB values losslessly and omits EXIF.
            # Clearing info prevents source metadata from being copied by Pillow.
            rgb.info.clear()
            result = io.BytesIO()
            rgb.save(result, format="PNG")
            return result.getvalue()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise ValueError("Downloaded image could not be decoded safely") from exc


def export_dataset(station, destination, max_images=20):
    if type(max_images) is not int or not 1 <= max_images <= MAX_IMAGES:
        raise ValueError("Use a maximum image count from 1 to 100")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Output directory exists; choose a new collection directory")
    destination.mkdir(parents=True, exist_ok=False)
    write_new(destination / "INCOMPLETE.json", {"status": "INCOMPLETE", "createdAt": now(),
        "notice": "Quarantined export in progress. Do not train from this directory unless READY.json exists.",
        "automaticTraining": False})
    stage = "read_approved_export"
    try:
        initial = station.request("/api/testbench/export")
        rows = records_from(initial)
        if not rows:
            raise ValueError("No approved complete non-evaluation images are available")
        selected = rows[:max_images]
        signatures = {r["image"]["sha256"]: sha(canonical(r)) for r in selected}
        images_dir, labels_dir = destination / "images", destination / "labels"
        images_dir.mkdir(); labels_dir.mkdir()
        coco = {"info": {"description": "Human-reviewed MycoScout mushroom detection collection; no split assigned",
                         "version": "testbench-detection-v1"},
                "images": [], "annotations": [], "categories": [{"id": 0, "name": "mushroom", "supercategory": "fungi"}]}
        records, total_bytes, annotation_id = [], 0, 1
        stage = "download_and_verify_images"
        for index, row in enumerate(selected, 1):
            info = row["image"]
            # Only the exact validated API path is ever requested; never fetch source credits or external URLs.
            raw = station.request(info["url"], binary=True)
            png = normalized_png(raw, info)
            total_bytes += len(raw) + len(png)
            if total_bytes > MAX_COLLECTION_BYTES:
                raise ValueError("Collection exceeds the 1 GiB transfer and normalized-image budget")
            filename = info["sha256"] + ".png"
            with (images_dir / filename).open("xb") as output:
                output.write(png)
            image_path = "images/" + filename
            coco["images"].append({"id": index, "file_name": image_path, "width": info["width"], "height": info["height"],
                                    "source_run_id": row["runId"], "original_sha256": info["sha256"]})
            yolo = []
            for box in row["annotation"]["boxes"]:
                x1, y1, x2, y2 = box["bboxPixelsXYXY"]
                bw, bh = x2 - x1, y2 - y1
                coco["annotations"].append({"id": annotation_id, "image_id": index, "category_id": 0,
                    "bbox": [x1, y1, bw, bh], "area": bw * bh, "iscrowd": 0,
                    "source_annotation_id": row["annotation"]["id"], "source_box_id": box["id"],
                    "human_group_id": box["groupId"], "human_label": box.get("label", "")})
                annotation_id += 1
                values = [(x1 + x2) / (2 * info["width"]), (y1 + y2) / (2 * info["height"]), bw / info["width"], bh / info["height"]]
                yolo.append("0 " + " ".join(format(value, ".12g") for value in values))
            label_path = labels_dir / (info["sha256"] + ".txt")
            with label_path.open("x", encoding="utf-8", newline="\n") as output:
                output.write("\n".join(yolo) + ("\n" if yolo else ""))
            records.append({"runId": row["runId"], "sourceRunId": row.get("sourceRunId"), "sceneId": row["sceneId"],
                "originalImageSha256": info["sha256"], "originalImageBytes": info["bytes"], "sourceImageUrl": info["url"],
                "image": {"relativePath": image_path, "sha256": sha(png), "bytes": len(png), "mimeType": "image/png",
                          "width": info["width"], "height": info["height"], "orientationApplied": True, "exifRemoved": True},
                "yoloLabels": {"relativePath": "labels/" + label_path.name, "sha256": sha(label_path.read_bytes())},
                "annotation": deepcopy(row["annotation"]), "sourceCredits": deepcopy(row.get("sourceCredits", {})),
                "sourceProvenance": deepcopy(row["provenance"]), "videoProvenance": deepcopy(row.get("videoProvenance")), "sourceSuggestedPartition": row.get("partition"),
                "sourceRecordSha256": signatures[info["sha256"]], "evaluationOnly": False, "split": None})
        stage = "recheck_permissions_and_revisions"
        current = station.request("/api/testbench/export")
        current_rows = records_from(current)
        current_signatures = {r["image"]["sha256"]: sha(canonical(r)) for r in current_rows}
        if any(current_signatures.get(digest) != signature for digest, signature in signatures.items()):
            raise ValueError("Source permission, annotations or provenance changed during export; collection remains quarantined")
        checked_at = now()
        stage = "write_verified_collection"
        coco_sha = write_new(destination / "dataset-all.json", coco)
        with (destination / "classes.txt").open("x", encoding="utf-8", newline="\n") as output:
            output.write("mushroom\n")
        with (destination / "images-all.txt").open("x", encoding="utf-8", newline="\n") as output:
            output.write("".join(row["image"]["relativePath"] + "\n" for row in records))
        collection = {"schema": "mycoscout.testbench-detection-collection.v1", "status": "READY",
            "createdAt": checked_at, "sourceExportReleaseId": initial["releaseId"], "verifiedExportReleaseId": current["releaseId"],
            "rightsAndRevisionsCheckedAt": checked_at, "selectedSourceRecordsUnchanged": True,
            "images": len(records), "annotations": len(coco["annotations"]),
            "confirmedNegativeImages": sum(not row["annotation"]["boxes"] for row in records),
            "availableEligibleImages": len(rows), "omittedByLimit": max(0, len(rows) - len(records)),
            "classMapping": {"0": "mushroom"}, "coco": {"relativePath": "dataset-all.json", "sha256": coco_sha},
            "records": records, "policies": {"automaticTraining": False, "automaticModelDeployment": False,
                "automaticSplit": False, "speciesClassesCreated": False, "humanSceneSeparationRequired": True,
                "nearDuplicateSceneGroupingVerified": False, "permissionIsPointInTimeSnapshot": True},
            "notes": ["Training and validation/test partitions are deliberately not created. Group related scenes before selecting any split.",
                      "Source-suggested partitions are preserved only as provenance, not used to assign this collection.",
                      "Original upload hashes and immutable human annotation IDs are retained; output PNG hashes differ when pixels were decoded/oriented.",
                      "PNG contains losslessly encoded oriented RGB pixels; EXIF and other source metadata are removed.",
                      "An empty YOLO label file denotes an explicitly confirmed negative, not an unreviewed image.",
                      "Human group and label text are metadata; every training category is class 0 mushroom.",
                      "Recheck current source permissions before later training; this export cannot guarantee consent remains unchanged forever."]}
        manifest_sha = write_new(destination / "collection-manifest.json", collection)
        # READY is the last write. A failure or revocation before this point leaves
        # INCOMPLETE and no training-ready collection marker.
        (destination / "INCOMPLETE.json").rename(destination / "export-start.json")
        write_new(destination / "READY.json", {"status": "READY", "collectionManifestSha256": manifest_sha,
                   "datasetSha256": coco_sha, "rightsAndRevisionsCheckedAt": checked_at, "automaticTraining": False})
        return {"directory": str(destination), "manifest": str(destination / "collection-manifest.json"),
                "images": len(records), "annotations": len(coco["annotations"]), "status": "READY", "automaticTraining": False}
    except Exception:
        # Leave partial artifacts for audit rather than silently deleting them.
        if not (destination / "INCOMPLETE.json").exists():
            write_new(destination / "INCOMPLETE.json", {"status": "INCOMPLETE", "createdAt": now(), "automaticTraining": False})
        if not (destination / "failure.json").exists():
            write_new(destination / "failure.json", {"status": "QUARANTINED", "stage": stage,
                       "notice": "Export did not finish. Do not use these artifacts for training.", "automaticTraining": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="http://127.0.0.1:8787")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing output is never overwritten")
    parser.add_argument("--max-images", type=int, default=20, help="Maximum 1–100 approved images; default 20")
    args = parser.parse_args()
    try:
        result = export_dataset(Station(args.station), args.output_dir, args.max_images)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"Dataset export failed: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
