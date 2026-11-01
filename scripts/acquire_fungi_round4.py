"""Acquire a bounded, NEW-only licensed fungal review cohort from public iNaturalist.

No source label becomes a detector annotation. Existing photographers, observations,
photo identities and conservative perceptual duplicates are excluded before freezing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from vision.acquire_public import Client, LICENSES, digest, download_one, freeze_splits, now, write_json
from vision.acquire_groups import photo_record

TARGETS = [
    ("Morchella", "true_morels", 40),
    ("Laetiporus", "chicken_of_woods", 40),
    ("Pleurotus", "oysters", 40),
    ("Craterellus fallax", "black_trumpets", 10),
    ("Craterellus cornucopioides", "black_trumpets", 10),
    ("Craterellus calicornucopioides", "black_trumpets", 10),
    ("Cantharellus", "chanterelles", 20),
    ("Trametes versicolor", "turkey_tail", 15),
    ("Grifola frondosa", "hen_of_woods", 15),
]
SCHEMA = "mushroom-vision.public-fungi-round.v4"


def prior_index(manifest_paths):
    result = {key: set() for key in ("photos", "observations", "observers", "bytes", "pixels", "perceptual")}
    evidence = []
    for path in manifest_paths:
        payload = path.read_bytes()
        data = json.loads(payload.decode("utf-8-sig"))
        rows = data.get("images", []) if isinstance(data, dict) else []
        if not isinstance(rows, list):
            continue
        count = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            # Include all recorded hashes, including non-iNaturalist observations.
            for field, key in (("photoId", "photos"), ("observationId", "observations"),
                               ("observerId", "observers"), ("sha256", "bytes"),
                               ("originalSha256", "bytes"), ("normalizedPixelSha256", "pixels")):
                value = row.get(field)
                if isinstance(value, (str, int)):
                    result[key].add(value)
            if isinstance(row.get("perceptualDifferenceHash"), str):
                result["perceptual"].add(int(row["perceptualDifferenceHash"], 16))
            count += 1
        if count:
            evidence.append({"path": path.relative_to(PROJECT).as_posix() if path.is_relative_to(PROJECT) else str(path),
                             "sha256": digest(payload), "recordsIndexed": count})
    return result, evidence


def identity_reason(row, index):
    for field, key in (("photoId", "photos"), ("observationId", "observations"), ("observerId", "observers")):
        if row.get(field) in index[key]:
            return "existing_" + field
    return None


def duplicate_reason(row, index):
    if row["sha256"] in index["bytes"]:
        return "exact_image_bytes"
    if row["normalizedPixelSha256"] in index["pixels"]:
        return "normalized_pixel_alias"
    h = int(row["perceptualDifferenceHash"], 16)
    if any((h ^ previous).bit_count() <= 2 for previous in index["perceptual"]):
        return "conservative_perceptual_alias"
    return None


def add_index(row, index):
    for field, key in (("photoId", "photos"), ("observationId", "observations"), ("observerId", "observers"),
                       ("sha256", "bytes"), ("normalizedPixelSha256", "pixels")):
        index[key].add(row[field])
    index["perceptual"].add(int(row["perceptualDifferenceHash"], 16))


def select_record(observation, query):
    # A secondary photo can show the organism's setting; that is a sampling
    # heuristic only, never a claim that the photograph is distant or in situ.
    photos = observation.get("photos", [])
    eligible = []
    for ordinal, photo in enumerate(photos):
        row = photo_record({**observation, "photos": [photo]}, query, [query])
        if row:
            eligible.append((ordinal, row))
    if not eligible:
        return None
    ordinal, row = eligible[-1]
    row["downloadUrl"] = row["downloadUrl"].replace("/large.", "/original.")
    row.update(cohort="new_public_fungi_v4", context="unknown", mushroomVisibility="unknown",
               humanVerified=False, trainingEligible=False, annotationStatus="unreviewed", annotations=[],
               sourceGroupId=row["groupId"], sourceObservationGroupId=f"inat-observation-{row['observationId']}",
               acquisitionPhotoOrdinal=ordinal, observationPhotoCount=len(photos),
               selectionHint="last licensed photo of previously unseen observation and photographer; context unverified")
    return row


def verify_manifest(manifest, root, prior):
    rows = manifest["images"]
    seen = {key: set(values) for key, values in prior.items()}
    observer_splits = {}
    for row in rows:
        if identity_reason(row, seen) or duplicate_reason(row, seen):
            raise ValueError("Duplicate identity or image in frozen cohort")
        if row["annotations"] or row["humanVerified"] or row["trainingEligible"] or row["distanceM"] is not None:
            raise ValueError("Acquisition must not create training truth or distances")
        if row["photoLicense"]["code"] not in LICENSES or not row["photoLicense"].get("attribution"):
            raise ValueError("Missing allowed photo rights")
        if row["split"] not in ("train", "validation", "test"):
            raise ValueError("Invalid frozen split")
        if observer_splits.setdefault(row["observerId"], row["split"]) != row["split"]:
            raise ValueError("Observer crosses partitions")
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path.read_bytes()) != row["sha256"]:
            raise ValueError("Image path or checksum mismatch")
        add_index(row, seen)
    return {"valid": True, "images": len(rows), "allPriorPhotographersExcluded": True,
            "imageBytesVerified": sum(r["bytes"] for r in rows), "detectorAnnotationsCreated": 0}


def acquire(output, per_target_scale=1.0):
    output = output.resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; frozen and partial runs are never overwritten")
    if not 0 < per_target_scale <= 1.5:
        raise ValueError("Scale must be >0 and <=1.5 (maximum 300 images)")
    manifest_paths = sorted(p / "manifest.json" for p in (PROJECT / "data").iterdir()
                            if p.is_dir() and (p / "manifest.json").is_file())
    prior, prior_evidence = prior_index(manifest_paths)
    index = {key: set(values) for key, values in prior.items()}
    source_taxonomy = PROJECT / "data/public-fungi-v3/taxonomy.json"
    known_queries = json.loads(source_taxonomy.read_text(encoding="utf-8"))["queries"]
    queries = []
    for name, class_id, quota in TARGETS:
        match = [q for q in known_queries if q["name"] == name and q["classId"] == class_id]
        if len(match) != 1:
            raise ValueError("Unresolved frozen taxonomy target " + name)
        queries.append({**match[0], "quota": max(1, int(quota * per_target_scale))})
    (output / "images").mkdir(parents=True)
    write_json(output / "protocol.json", {"schema": SCHEMA, "createdAt": now(), "targets": queries,
        "priorManifests": prior_evidence, "taxonomySha256": digest(source_taxonomy.read_bytes()),
        "mediaBudgetBytes": 600_000_000, "maxImages": sum(q["quota"] for q in queries),
        "labelPolicy": "weak source taxa only; no verified boxes or negative labels",
        "dedupePolicy": "Exclude prior and current photo IDs, observation IDs, ALL photographers, exact bytes, normalized pixels, and dHash distance <=2",
        "samplingPolicy": "Last licensed original-resolution photo per independent observation/photographer; environmental view and distance remain unknown",
        "sourceDocumentation": ["https://registry.opendata.aws/inaturalist-open-data/", "https://github.com/inaturalist/inaturalist-open-data", "https://api.inaturalist.org/v1/docs/"]})
    client = Client(max_media_bytes=600_000_000, api_delay=1.2)
    rows, logs, skipped = [], [], Counter()
    for query in queries:
        accepted = 0
        for page in range(1, 7):
            if accepted >= query["quota"]:
                break
            response, url = client.api("observations", {"taxon_id": query["taxon_id"], "quality_grade": "research",
                "photos": "true", "captive": "false", "photo_license": ",".join(LICENSES),
                "license": ",".join(LICENSES), "per_page": 200, "page": page, "order_by": "id", "order": "desc"})
            logs.append({"url": url, "retrievedAt": now(), "classId": query["classId"], "totalAvailable": response.get("total_results")})
            candidates, batch_observers = [], set()
            for observation in response.get("results", []):
                row = select_record(observation, query)
                if not row:
                    skipped["no_eligible_photo"] += 1
                    continue
                reason = identity_reason(row, index)
                if reason or row["observerId"] in batch_observers:
                    skipped[reason or "same_batch_photographer"] += 1
                    continue
                candidates.append(row)
                batch_observers.add(row["observerId"])
                if len(candidates) >= query["quota"] - accepted:
                    break
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = [(row, pool.submit(download_one, client, output, row)) for row in candidates]
                for candidate, future in pending:
                    try:
                        row = future.result()
                        reason = duplicate_reason(row, index)
                        if reason:
                            skipped[reason] += 1
                            (output / row["path"]).unlink()
                            continue
                        add_index(row, index)
                        rows.append(row)
                        accepted += 1
                    except (OSError, ValueError, RuntimeError) as exc:
                        skipped[type(exc).__name__ + ": " + str(exc)[:120]] += 1
            write_json(output / "progress.json", {"status": "not-frozen", "images": rows, "queries": logs, "skipped": dict(skipped)})
            print(json.dumps({"event": "page", "target": query["name"], "page": page, "accepted": accepted, "total": len(rows)}), flush=True)
            if len(response.get("results", [])) < 200:
                break
    split_policy = freeze_splits(rows, seed="mushroom-vision-new-public-round4-20261031")
    summary = {"images": len(rows), "bytes": sum(r["bytes"] for r in rows), "observations": len({r["observationId"] for r in rows}),
        "photographers": len({r["observerId"] for r in rows}), "classes": dict(Counter(r["classId"] for r in rows)),
        "photoLicenses": dict(Counter(r["photoLicense"]["code"] for r in rows)), "splitCounts": dict(Counter(r["split"] for r in rows)),
        "multiPhotoObservations": sum(r["observationPhotoCount"] > 1 for r in rows), "newDetectorAnnotations": 0}
    manifest = {"schema": SCHEMA, "status": "frozen", "title": "Mushroom Vision new public fungi round 4", "createdAt": now(),
        "images": rows, "summary": summary, "splitPolicy": split_policy, "priorManifests": prior_evidence,
        "acquisition": {"apiRequests": client.api_calls, "downloadedBytes": client.media_bytes, "queries": logs, "skipped": dict(skipped),
            "apiMinimumIntervalSeconds": 1.2, "concurrency": 3, "mediaBudgetBytes": 600_000_000},
        "limitations": ["Taxa are source observation guesses, not verified image annotations.",
            "No mushroom distances, detector boxes, negative labels, or measured flight performance.",
            "Images require independent review; train split alone does not confer training eligibility.",
            "All earlier photographers excluded; all source and image aliases screened conservatively.",
            "Each photo retains attribution and CC0/CC BY/CC BY-SA rights; code license does not relicense photos."]}
    validation = verify_manifest(manifest, output, prior)
    write_json(output / "manifest.json", manifest)
    write_json(output / "manifest-sha256.json", {"sha256": digest((output / "manifest.json").read_bytes())})
    write_json(output / "summary.json", summary)
    write_json(output / "validation.json", validation)
    print(json.dumps({"event": "frozen", "summary": summary}), flush=True)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/public-fungi-v4")
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    acquire(args.output, args.scale)
