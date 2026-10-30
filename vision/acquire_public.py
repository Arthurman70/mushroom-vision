"""Bounded, licensed iNaturalist photo acquisition for weak-label mushroom research.

Public API only; no login, cookie, HTML scraping, locations, or detector labels.
"""
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
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageOps

SCHEMA = "mycoscout.weak-fungi-dataset.v2"
API = "https://api.inaturalist.org/v1/"
USER_AGENT = "MycoScoutResearch/0.2 (bounded licensed-photo vision experiment)"
LICENSES = {"cc0": "https://creativecommons.org/publicdomain/zero/1.0/",
            "cc-by": "https://creativecommons.org/licenses/by/4.0/",
            "cc-by-sa": "https://creativecommons.org/licenses/by-sa/4.0/"}
BLACK_TAXA = {194231: "Craterellus fallax", 48607: "Craterellus cornucopioides", 473935: "Craterellus calicornucopioides"}
QUERIES = [
    {"taxon_id": 194231, "name": "Craterellus fallax", "cohort": "black_trumpet", "quota": 250},
    {"taxon_id": 48607, "name": "Craterellus cornucopioides", "cohort": "black_trumpet", "quota": 150},
    {"taxon_id": 473935, "name": "Craterellus calicornucopioides", "cohort": "black_trumpet", "quota": 50},
    {"taxon_id": 350511, "name": "Craterellus tubaeformis", "cohort": "related_fungi", "quota": 100},
    {"taxon_id": 782233, "name": "Craterellus ignicolor", "cohort": "related_fungi", "quota": 60},
    {"taxon_id": 350509, "name": "Craterellus cinereus", "cohort": "related_fungi", "quota": 40},
    {"taxon_id": 1137797, "name": "Craterellus undulatus", "cohort": "related_fungi", "quota": 50},
    {"taxon_id": 49206, "name": "Helvella", "cohort": "related_fungi", "quota": 100},
    {"taxon_id": 47169, "name": "Basidiomycota excluding queried genera", "cohort": "other_fungi", "quota": 400,
     "without_taxon_id": "48611,49206"},
]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class Client:
    def __init__(self, max_media_bytes=1_000_000_000, api_delay=1.1):
        self.max_media_bytes = max_media_bytes
        self.api_delay = max(1.1, api_delay)
        self.api_calls = 0
        self.media_bytes = 0
        self.media_attempts = 0
        self.last_api = 0
        self.lock = threading.Lock()

    def request(self, url, api=False):
        if api:
            if self.api_calls >= 200:
                raise RuntimeError("Bounded pilot API budget exhausted")
            time.sleep(max(0, self.api_delay - (time.monotonic() - self.last_api)))
            self.last_api = time.monotonic()
            self.api_calls += 1
        else:
            with self.lock:
                if self.media_bytes >= self.max_media_bytes:
                    raise RuntimeError("Bounded pilot media byte budget exhausted")
                self.media_attempts += 1
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json" if api else "image/*"})
                with urllib.request.urlopen(request, timeout=45) as response:
                    final_host = urllib.parse.urlparse(response.url).hostname
                    if not api and final_host != "inaturalist-open-data.s3.amazonaws.com":
                        raise ValueError("Image redirected outside the verified open-data media host")
                    maximum = 30_000_000 if api else 8_000_000
                    payload = response.read(maximum + 1)
                    if len(payload) > maximum:
                        raise ValueError("Response exceeds per-request size limit")
                if not api:
                    with self.lock:
                        self.media_bytes += len(payload)
                return payload
            except urllib.error.HTTPError as exc:
                if exc.code not in (429, 500, 502, 503, 504) or attempt == 3:
                    raise
                delay = max(2 ** (attempt + 1), min(60, int(exc.headers.get("Retry-After", "0"))))
                time.sleep(delay)
            except (TimeoutError, urllib.error.URLError):
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        raise RuntimeError("Request retry budget exhausted")

    def api(self, endpoint, params):
        url = API + endpoint + "?" + urllib.parse.urlencode(params)
        return json.loads(self.request(url, api=True)), url


def photo_record(observation, query):
    taxon = observation.get("taxon") or {}
    observer = observation.get("user") or {}
    if observation.get("quality_grade") != "research" or observation.get("license_code") not in LICENSES:
        return None
    if not observer.get("id") or not taxon.get("id") or observer.get("suspended") or observer.get("spam"):
        return None
    ancestors = set(taxon.get("ancestor_ids", [])) | {taxon["id"]}
    if query["taxon_id"] not in ancestors:
        return None
    black = bool(ancestors.intersection(BLACK_TAXA))
    if (query["cohort"] == "black_trumpet") != black:
        return None
    for photo in observation.get("photos", []):
        code = photo.get("license_code")
        url = photo.get("url", "")
        parsed = urllib.parse.urlparse(url)
        if (code not in LICENSES or parsed.scheme != "https" or parsed.hostname != "inaturalist-open-data.s3.amazonaws.com"
                or photo.get("hidden") or photo.get("flags") or not photo.get("attribution")):
            continue
        photo_id = photo.get("id")
        if not isinstance(photo_id, int) or not re.fullmatch(rf"/photos/{photo_id}/square\.[A-Za-z0-9]+", parsed.path):
            continue
        large_url = urllib.parse.urlunparse(parsed._replace(path=parsed.path.replace("/square.", "/large."), query="", fragment=""))
        return {"id": f"inat-photo-{photo_id}", "path": f"images/inat-photo-{photo_id}.jpg",
                "photoId": photo_id, "observationId": observation["id"], "observerId": observer["id"],
                "groupId": f"inat-observer-{observer['id']}", "sourceUrl": f"https://www.inaturalist.org/observations/{observation['id']}",
                "photoUrl": f"https://www.inaturalist.org/photos/{photo_id}", "downloadUrl": large_url,
                "photoLicense": {"code": code, "url": LICENSES[code], "attribution": photo["attribution"],
                                 "licenseVersionNote": "iNaturalist public license code mapped to current Creative Commons 4.0 page (CC0 1.0); source photo attribution retained"},
                "observationLicense": {"code": observation["license_code"], "url": LICENSES[observation["license_code"]]},
                "taxon": {key: taxon.get(key) for key in ("id", "name", "rank", "preferred_common_name")},
                "cohort": query["cohort"], "queryTaxonId": query["taxon_id"],
                "weakLabel": "black_trumpet" if black else "other_fungi", "labelEvidence": "iNaturalist research-grade observation taxon; image content not exhaustively reviewed",
                "annotationStatus": "unreviewed", "annotations": [], "synthetic": False,
                "context": "unknown", "distanceM": None, "retrievedAt": now()}
    return None


def download_one(client, root, row):
    path = root / row["path"]
    if path.exists():
        payload = path.read_bytes()
    else:
        payload = client.request(row["downloadUrl"])
    with Image.open(io.BytesIO(payload)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        if min(width, height) < 120 or width * height > 5_000_000:
            raise ValueError("Photo is too small or unexpectedly large")
        pixel_hash = digest(image.resize((64, 64), Image.Resampling.BILINEAR).tobytes())
        thumbnail = list(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR).tobytes())
        difference = sum((thumbnail[y * 9 + x] > thumbnail[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
    if not path.exists():
        path.write_bytes(payload)
    row = dict(row)
    row.update(width=width, height=height, sha256=digest(payload), originalSha256=digest(payload),
               normalizedPixelSha256=pixel_hash, perceptualDifferenceHash=f"{difference:016x}", bytes=len(payload))
    return row


def freeze_splits(rows, seed="mycoscout-public-fungi-v2-20261026", prior_splits=None):
    """Connect observers and exact/near duplicates before a fixed hash-based split."""
    parents = {row["groupId"]: row["groupId"] for row in rows}

    def find(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group

    near_links = []
    for i, row in enumerate(rows):
        for previous in rows[:i]:
            if row["groupId"] == previous["groupId"]:
                continue
            difference = (int(row["perceptualDifferenceHash"], 16) ^ int(previous["perceptualDifferenceHash"], 16)).bit_count()
            if difference <= 2:
                a, b = find(row["groupId"]), find(previous["groupId"])
                parents[max(a, b)] = min(a, b)
                near_links.append({"photoA": row["id"], "photoB": previous["id"], "hashDistance": difference})
    inherited = defaultdict(set)
    for group, split in (prior_splits or {}).items():
        if split not in ("train", "validation", "test") or group not in parents:
            raise ValueError("Invalid inherited observer split")
        inherited[find(group)].add(split)
    if any(len(splits) > 1 for splits in inherited.values()):
        raise ValueError("New near-duplicate links connect previously frozen splits; expanded corpus requires review and was not frozen")
    for row in rows:
        component = find(row["groupId"])
        value = int(digest((seed + "|" + component).encode())[:16], 16) / 16 ** 16
        row["splitGroupId"] = component
        row["split"] = next(iter(inherited[component])) if inherited[component] else "train" if value < 0.6 else "validation" if value < 0.8 else "test"
    return {"seed": seed, "method": "hash of connected observer/near-duplicate groups; frozen before features or augmentation",
            "requestedFractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
            "nearDuplicateHash": "64-bit difference hash, Hamming distance <=2 connects observers conservatively",
            "nearDuplicateLinks": near_links, "componentCount": len({find(group) for group in parents}),
            **({"inheritedObserverCount": len(prior_splits), "priorSplitsPreserved": True} if prior_splits else {})}


def expansion_queries(count):
    """Allocate a bounded additional batch with deterministic proportional quotas."""
    total = sum(query["quota"] for query in QUERIES)
    quotas = [query["quota"] * count // total for query in QUERIES]
    remainder = count - sum(quotas)
    order = sorted(range(len(QUERIES)), key=lambda index: (-(QUERIES[index]["quota"] * count % total), index))
    for index in order[:remainder]:
        quotas[index] += 1
    return [{**query, "quota": quota} for query, quota in zip(QUERIES, quotas) if quota]


def prepare_expansion(source_manifest, output_root):
    """Verify and copy a frozen source into a separate portable dataset directory."""
    from vision.acquisition.validate import validate

    source_manifest = Path(source_manifest).resolve()
    source_root, output_root = source_manifest.parent, Path(output_root).resolve()
    if output_root == source_root or output_root.is_relative_to(source_root) or source_root.is_relative_to(output_root):
        raise ValueError("Expansion destination must be separate from and outside the frozen source directory")
    frozen_hash = json.loads((source_root / "manifest-sha256.json").read_text(encoding="utf-8"))["sha256"]
    if digest(source_manifest.read_bytes()) != frozen_hash:
        raise ValueError("Frozen source manifest SHA-256 does not match its freeze record")
    checked = validate(source_manifest)
    if not checked["valid"]:
        raise ValueError("Frozen source validation failed: " + "; ".join(checked["errors"][:5]))
    source = json.loads(source_manifest.read_text(encoding="utf-8"))
    if not source["images"]:
        raise ValueError("Cannot expand an empty source corpus")
    for row in source["images"]:
        destination = (output_root / row["path"]).resolve()
        if not destination.is_relative_to(output_root):
            raise ValueError("Inherited image path escapes destination")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination.read_bytes()) != row["sha256"]:
                raise ValueError("Existing destination image differs from frozen source")
        else:
            # Copy rather than hard-link: later edits cannot mutate frozen source bytes.
            shutil.copyfile(source_root / row["path"], destination)
    return source, {"manifestSha256": frozen_hash, "imagesInherited": len(source["images"]),
                    "sourceDatasetTitle": source.get("title"), "splitAssignmentsPreserved": True,
                    "note": "Source manifest and media remain unchanged; inherited test examples are not a new untouched test set."}


def summary(rows):
    return {"images": len(rows), "observers": len({row["observerId"] for row in rows}),
            "observations": len({row["observationId"] for row in rows}), "bytes": sum(row["bytes"] for row in rows),
            "weakLabels": dict(Counter(row["weakLabel"] for row in rows)), "cohorts": dict(Counter(row["cohort"] for row in rows)),
            "photoLicenses": dict(Counter(row["photoLicense"]["code"] for row in rows)),
            "taxonCounts": dict(Counter(row["taxon"]["name"] for row in rows)),
            "splitCounts": {split: dict(Counter(row["weakLabel"] for row in rows if row.get("split") == split)) for split in ("train", "validation", "test")}}


def acquire(root, max_images=1200, workers=3, per_observer=8, resume_from=None, more=None):
    if not 1 <= max_images <= 1500 or not 1 <= workers <= 4 or not 1 <= per_observer <= 20:
        raise ValueError("Bounded acquisition requires 1–1500 images, 1–4 workers, and 1–20 photos per observer")
    if (resume_from is None) != (more is None) or (more is not None and not 1 <= more <= 1500):
        raise ValueError("--resume-from and --more 1–1500 must be provided together")
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError("Frozen manifest already exists; choose a new dataset directory for another acquisition")
    inherited, provenance = prepare_expansion(resume_from, root) if resume_from else (None, None)
    (root / "images").mkdir(parents=True, exist_ok=True)
    client = Client()
    rows, skipped, query_log = inherited["images"] if inherited else [], [], []
    prior_splits = {row["groupId"]: row["split"] for row in rows}
    starting_count = len(rows)
    limit = starting_count + more if more else max_images
    seen_ids = {row["photoId"] for row in rows}
    seen_observations = {row["observationId"] for row in rows}
    seen_hashes = {row["sha256"] for row in rows}
    seen_pixels = {row["normalizedPixelSha256"] for row in rows}
    observer_counts = Counter(row["observerId"] for row in rows)
    for query in expansion_queries(more) if more else QUERIES:
        retained = 0
        for page in range(1, 11):
            if retained >= query["quota"] or len(rows) >= limit:
                break
            params = {"taxon_id": query["taxon_id"], "quality_grade": "research", "photos": "true", "captive": "false",
                      "photo_license": ",".join(LICENSES), "license": ",".join(LICENSES), "per_page": 200,
                      "page": page, "order_by": "id", "order": "desc"}
            if query.get("without_taxon_id"):
                params["without_taxon_id"] = query["without_taxon_id"]
            response, url = client.api("observations", params)
            query_log.append({"url": url, "queryName": query["name"], "retrievedAt": now(), "totalAvailable": response.get("total_results"), "recordsReturned": len(response["results"])})
            candidates = []
            page_observers = observer_counts.copy()
            for observation in response["results"]:
                row = photo_record(observation, query)
                if not row or row["photoId"] in seen_ids or row["observationId"] in seen_observations or page_observers[row["observerId"]] >= per_observer:
                    continue
                candidates.append(row)
                seen_ids.add(row["photoId"])
                seen_observations.add(row["observationId"])
                page_observers[row["observerId"]] += 1
                if len(candidates) >= min(query["quota"] - retained, limit - len(rows)):
                    break
            with ThreadPoolExecutor(max_workers=max(1, min(4, workers))) as executor:
                futures = [(candidate, executor.submit(download_one, client, root, candidate)) for candidate in candidates]
                for candidate, future in futures:
                    try:
                        row = future.result()
                        if row["sha256"] in seen_hashes or row["normalizedPixelSha256"] in seen_pixels:
                            skipped.append({"id": row["id"], "reason": "duplicate content excluded from manifest"})
                            continue
                        seen_hashes.add(row["sha256"])
                        seen_pixels.add(row["normalizedPixelSha256"])
                        rows.append(row)
                        observer_counts[row["observerId"]] += 1
                        retained += 1
                        if len(rows) % 100 == 0:
                            write_json(root / "manifest-progress.json", {"schema": SCHEMA, "status": "downloading-not-frozen", "images": rows, "summary": summary(rows)})
                            print(json.dumps({"event": "download-progress", "images": len(rows), "bytes": client.media_bytes, "weakLabels": summary(rows)["weakLabels"]}), flush=True)
                    except (OSError, ValueError, RuntimeError) as exc:
                        skipped.append({"id": candidate["id"], "reason": str(exc)})
            write_json(root / "acquisition-progress.json", {"queries": query_log, "summary": summary(rows), "skipped": skipped})
            if len(response["results"]) < 200:
                break
        print(json.dumps({"event": "query-complete", "name": query["name"], "retained": retained, "totalImages": len(rows)}), flush=True)
    seed = inherited["splitPolicy"]["seed"] if inherited else "mycoscout-public-fungi-v2-20261026"
    split_policy = freeze_splits(rows, seed=seed, prior_splits=prior_splits)
    manifest = {"schema": SCHEMA, "title": "MycoScout licensed fungi weak-label corpus v2", "createdAt": now(),
                "status": "frozen", "images": rows, "summary": summary(rows), "splitPolicy": split_policy,
                "labelMap": {"black_trumpet": {str(key): value for key, value in BLACK_TAXA.items()},
                             "other_fungi": "Comparison fungi, not mushroom-free background; related trumpet-like taxa included"},
                "acquisition": {"apiRequests": client.api_calls, "mediaAttempts": client.media_attempts, "downloadedBytes": client.media_bytes,
                                "maxImages": limit, "newImagesRequested": more if more else max_images,
                                "newImagesAccepted": len(rows) - starting_count,
                                "perObserverCap": per_observer, "apiMinimumIntervalSeconds": client.api_delay,
                                "mediaConcurrency": workers, "mediaBudgetBytes": client.max_media_bytes,
                                "imageSize": "iNaturalist large variant, maximum1024px", "onePhotoPerObservation": True,
                                "queryLog": query_log, "skipped": skipped},
                "limitations": ["Observation taxonomy is a weak image label; image contents and species identity are not independently verified.",
                                "No bounding boxes, mushroom-free negatives, camera distances, terrain labels, or flight validation are supplied.",
                                "Other fungi are classification comparison examples, not negative images for mushroom detection.",
                                "Observer-group splitting reduces source leakage; it cannot guarantee all same-site or near-duplicate encounters are separated.",
                                "Recent-observation and open-license selection introduce geographic, seasonal, photographer, and taxonomic sampling bias.",
                                "No locations, original observation text, or private account data are retained.",
                                "Retain per-photo attribution and license when displaying or redistributing media; CC BY-SA adaptations carry its license conditions."]}
    if provenance:
        manifest["title"] = "MycoScout licensed fungi weak-label corpus expansion"
        manifest["parentDataset"] = provenance
    write_json(root / "manifest.json", manifest)
    write_json(root / "manifest-sha256.json", {"schema": "mycoscout.dataset-freeze.v1", "sha256": digest((root / "manifest.json").read_bytes()), "createdAt": now()})
    write_json(root / "summary.json", manifest["summary"])
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/public-fungi-v2"))
    parser.add_argument("--max-images", type=int, default=1200)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--per-observer", type=int, default=8)
    parser.add_argument("--resume-from", type=Path, help="Frozen parent manifest; never modified")
    parser.add_argument("--more", type=int, help="Request this many additional photos into a separate output directory (1–1500)")
    args = parser.parse_args()
    if not 1 <= args.max_images <= 1500 or not 1 <= args.workers <= 4 or not 1 <= args.per_observer <= 20:
        parser.error("Bounded pilot limits:1–1500images,1–4workers,1–20photos per observer")
    if (args.resume_from is None) != (args.more is None) or (args.more is not None and not 1 <= args.more <= 1500):
        parser.error("--resume-from and --more 1–1500 must be provided together")
    result = acquire(args.output_dir, args.max_images, args.workers, args.per_observer, args.resume_from, args.more)
    print(json.dumps({"event": "complete", "summary": result["summary"], "apiRequests": result["acquisition"]["apiRequests"]}), flush=True)
