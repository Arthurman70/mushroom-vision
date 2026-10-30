"""Bounded, attributed biodiversity expansion; never creates verified detector labels."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

from .acquire_public import Client, LICENSES, digest, download_one, freeze_splits, now, write_json
from .acquire_groups import normalize, photo_record

TARGETS = [
    ("Amanita", "gilled"), ("Russula", "gilled"), ("Lactarius", "gilled"),
    ("Lactifluus", "gilled"), ("Cortinarius", "gilled"), ("Galerina", "gilled"),
    ("Inocybe", "gilled"), ("Hypholoma", "gilled"), ("Gymnopilus", "gilled"),
    ("Psilocybe", "gilled"), ("Agaricus", "gilled"), ("Mycena", "gilled"),
    ("Lepiota", "gilled"), ("Chlorophyllum", "gilled"), ("Gyromitra", "ascomycetes"),
    ("Verpa", "ascomycetes"), ("Helvella", "ascomycetes"), ("Auricularia", "jelly_fungi"),
    ("Tremella", "jelly_fungi"), ("Ramaria", "coral_fungi"), ("Phallus", "stinkhorns"),
]


def classify(ancestors, queries):
    """Most specific resolved research scope wins over broad inherited groups."""
    rank = {"species": 4, "genus": 3, "family": 2, "order": 1}
    matches = [q for q in queries if q["taxon_id"] in ancestors]
    if not matches:
        return "other_fungi"
    best = max(rank.get(q["rank"], 0) for q in matches)
    ids = {q["classId"] for q in matches if rank.get(q["rank"], 0) == best}
    if len(ids) != 1:
        raise ValueError("Ambiguous research taxonomy mapping")
    return next(iter(ids))


def resolve(client, targets, output):
    if output.exists():
        return json.loads(output.read_text(encoding="utf-8"))["queries"]
    queries = []
    for name, group in targets:
        response, url = client.api("taxa/autocomplete", {"q": name, "per_page": 30})
        rank = "species" if " " in name else "genus"
        choices = [t for t in response.get("results", []) if t.get("rank") == rank
                   and t.get("is_active", True) and t.get("iconic_taxon_name") == "Fungi"
                   and normalize(t.get("name", "")) == normalize(name)]
        if len(choices) != 1:
            raise ValueError(f"Unresolved fungal name {name}")
        taxon = choices[0]
        queries.append({"classId": name.lower().replace(" ", "_") + "_research", "groupId": group,
                        "label": name + " research group", "name": name, "requestedName": name,
                        "taxon_id": taxon["id"], "rank": taxon["rank"], "resolutionUrl": url,
                        "researchTaxonId": name.lower().replace(" ", "_")})
        print(json.dumps({"event": "resolved", "name": name, "id": taxon["id"]}), flush=True)
    write_json(output, {"retrievedAt": now(), "queries": queries})
    return queries


def summarize(rows):
    return {"images": len(rows), "newImages": sum(r.get("cohort") == "new_biodiversity_v1" for r in rows),
            "bytes": sum(r["bytes"] for r in rows), "observers": len({r["observerId"] for r in rows}),
            "taxa": len({r["taxon"]["id"] for r in rows}), "classes": dict(Counter(r["classId"] for r in rows)),
            "photoLicenses": dict(Counter(r["photoLicense"]["code"] for r in rows)),
            "splitCounts": dict(Counter(r["split"] for r in rows if "split" in r))}


def acquire(parent, root, quota=32):
    from .multiclass import validate_manifest
    parent, root = Path(parent).resolve(), Path(root).resolve()
    if root == parent.parent or root.is_relative_to(parent.parent) or parent.parent.is_relative_to(root):
        raise ValueError("Output must be separate from the frozen parent")
    if (root / "manifest.json").exists():
        raise ValueError("Frozen output exists; choose another output directory")
    source = validate_manifest(parent)
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(exist_ok=True)
    rows = source["images"]
    for row in rows:
        destination = (root / row["path"]).resolve()
        if not destination.is_relative_to(root):
            raise ValueError("Inherited path escapes output")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination.read_bytes()) != row["sha256"]:
            raise ValueError("Existing inherited image differs")
        if not destination.exists():
            shutil.copyfile(parent.parent / row["path"], destination)
        row.update(inherited=True, cohort="inherited_v3")
    client = Client(max_media_bytes=1_000_000_000, api_delay=1.1)
    queries = resolve(client, TARGETS, root / "taxonomy.json")
    inherited_queries = json.loads((parent.parent / "taxonomy.json").read_text(encoding="utf-8"))["queries"]
    all_queries = inherited_queries + queries
    for row in rows:
        ancestors = set(row["taxon"].get("ancestor_ids", [])) | {row["taxon"]["id"]}
        row["previousClassId"] = row["classId"]
        row["classId"] = classify(ancestors, all_queries)
        row["weakLabel"] = row["classId"]
    prior = {r["groupId"]: r["split"] for r in rows}
    seen_ids = {r["photoId"] for r in rows}
    seen_obs = {r["observationId"] for r in rows}
    seen_hash = {r["sha256"] for r in rows}
    seen_pixels = {r["normalizedPixelSha256"] for r in rows}
    counts = Counter(r["observerId"] for r in rows)
    logs, skipped = [], []
    for query in queries:
        retained = 0
        for page in range(1, 4):
            if retained >= quota:
                break
            response, url = client.api("observations", {"taxon_id": query["taxon_id"], "quality_grade": "research",
                "photos": "true", "captive": "false", "photo_license": ",".join(LICENSES),
                "license": ",".join(LICENSES), "per_page": 200, "page": page, "order_by": "id", "order": "desc"})
            logs.append({"url": url, "classId": query["classId"], "retrievedAt": now(), "totalAvailable": response.get("total_results")})
            candidates, page_counts = [], counts.copy()
            for obs in response.get("results", []):
                # A single queried genus cannot conflict. Global mapping is applied explicitly below.
                row = photo_record(obs, query, [query])
                if not row or row["photoId"] in seen_ids or row["observationId"] in seen_obs or page_counts[row["observerId"]] >= 8:
                    continue
                row.update(cohort="new_biodiversity_v1", context="unknown", researchTaxonId=query["researchTaxonId"])
                candidates.append(row)
                seen_ids.add(row["photoId"]); seen_obs.add(row["observationId"]); page_counts[row["observerId"]] += 1
                if len(candidates) >= quota - retained:
                    break
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = [(r, pool.submit(download_one, client, root, r)) for r in candidates]
                for candidate, future in pending:
                    try:
                        row = future.result()
                        if row["sha256"] in seen_hash or row["normalizedPixelSha256"] in seen_pixels:
                            raise ValueError("Exact/normalized duplicate excluded")
                        neighbors = {prior[p["groupId"]] for p in rows if p["groupId"] in prior
                                     and (int(row["perceptualDifferenceHash"], 16) ^ int(p["perceptualDifferenceHash"], 16)).bit_count() <= 2}
                        if row["groupId"] in prior:
                            neighbors.add(prior[row["groupId"]])
                        if len(neighbors) > 1:
                            raise ValueError("Near duplicate connects inherited partitions")
                        seen_hash.add(row["sha256"]); seen_pixels.add(row["normalizedPixelSha256"])
                        rows.append(row); counts[row["observerId"]] += 1; retained += 1
                    except (OSError, ValueError, RuntimeError) as exc:
                        skipped.append({"id": candidate["id"], "reason": str(exc)})
            write_json(root / "progress.json", {"status": "not-frozen", "images": rows, "queries": logs, "skipped": skipped})
            if len(response.get("results", [])) < 200:
                break
        print(json.dumps({"event": "query-complete", "name": query["name"], "accepted": retained, "total": len(rows)}), flush=True)
    splits = freeze_splits(rows, seed=source["splitPolicy"]["seed"], prior_splits=prior)
    classes = source["classes"] + [{"classId": q["classId"], "label": q["label"], "groupId": q["groupId"],
                                   "researchTaxonNames": [q["name"]], "certificationScope": "not inferred from model labels"} for q in queries]
    manifest = {"schema": "mycoscout.group-photo-dataset.v3", "status": "frozen", "createdAt": now(),
                "title": "Mushroom Vision biodiversity expansion 1", "images": rows, "classes": classes,
                "summary": summarize(rows), "splitPolicy": splits,
                "parentDataset": {"manifestSha256": digest(parent.read_bytes()), "imagesInherited": len(source["images"]) - sum(r.get("cohort") == "new_biodiversity_v1" for r in rows), "splitAssignmentsPreserved": True},
                "acquisition": {"apiRequests": client.api_calls, "downloadedBytes": client.media_bytes, "queries": logs,
                                "skipped": skipped, "maxNewImages": quota * len(queries), "mediaBudgetBytes": 1_024_000_000,
                                "apiMinimumIntervalSeconds": 1.1, "concurrency": 3},
                "limitations": ["Source observation taxonomy is a weak label; no newly downloaded image is human-verified.",
                    "No detector boxes, distances, mushroom-free negatives, clinical or flight validation.",
                    "Parent holdouts remain evaluation-only; inherited test images are reused evaluation.",
                    "CC attribution and share-alike conditions stay attached to each image; software license does not relicense images."]}
    write_json(root / "manifest.json", manifest)
    write_json(root / "manifest-sha256.json", {"sha256": digest((root / "manifest.json").read_bytes())})
    write_json(root / "summary.json", manifest["summary"])
    validate_manifest(root / "manifest.json")
    write_json(root / "validation.json", {"valid": True, "imageCount": len(rows), "bytesVerified": sum(r["bytes"] for r in rows), "allPriorSplitsPreserved": all(r["split"] == prior[r["groupId"]] for r in rows if r["groupId"] in prior)})
    print(json.dumps({"event": "frozen", "summary": manifest["summary"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=Path("data/public-fungi-v3/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/biodiversity-fungi-v1"))
    parser.add_argument("--per-genus", type=int, default=32)
    args = parser.parse_args()
    if not 1 <= args.per_genus <= 60:
        parser.error("Per-genus limit must be 1–60 (maximum 1260 new images)")
    acquire(args.parent, args.output, args.per_genus)
