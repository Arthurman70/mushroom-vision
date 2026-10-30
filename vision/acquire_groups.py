"""Bounded public licensed-photo expansion, with inherited observer splits."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import urllib.parse
from .acquire_public import Client, LICENSES, now, write_json, digest, download_one, freeze_splits, prepare_expansion

SCHEMA = "mycoscout.group-photo-dataset.v3"
# Taxon scope is for visual research. It is NOT the commercial certification scope.
CLASSES = [
    ("honey_mushrooms", "Honey mushroom group", "gilled", ["Armillaria", "Desarmillaria"], 110),
    ("giant_puffball", "Giant puffball", "puffballs", ["Calvatia gigantea"], 100),
    ("chanterelles", "Chanterelle group", "chanterelles_trumpets", ["Cantharellus"], 130),
    ("shaggy_mane", "Shaggy mane", "gilled", ["Coprinus comatus"], 100),
    ("black_trumpets", "Black trumpet group", "chanterelles_trumpets", ["Craterellus fallax", "Craterellus cornucopioides", "Craterellus calicornucopioides"], 40),
    ("aborted_entoloma", "Entoloma abortivum (form unverified)", "crust_parasitized", ["Entoloma abortivum"], 100),
    ("ganoderma", "Ganoderma conk group", "polypores", ["Ganoderma"], 110),
    ("hen_of_woods", "Hen of the woods", "polypores", ["Grifola frondosa"], 100),
    ("hericium", "Hericium tooth-fungus group", "tooth_fungi", ["Hericium"], 110),
    ("hedgehogs", "Hedgehog mushroom group", "tooth_fungi", ["Hydnum"], 110),
    ("lobster_mushroom", "Lobster mushroom", "crust_parasitized", ["Hypomyces lactifluorum"], 100),
    ("chaga", "Chaga (form unverified)", "polypores", ["Inonotus obliquus"], 100),
    ("chicken_of_woods", "Chicken of the woods group", "polypores", ["Laetiporus"], 110),
    ("true_morels", "Morel group", "morels", ["Morchella"], 130),
    ("oysters", "Oyster mushroom group", "gilled", ["Pleurotus"], 110),
    ("dryads_saddle", "Dryad's saddle", "polypores", ["Cerioporus squamosus"], 100),
    ("umbrella_polypore", "Umbrella polypore", "polypores", ["Polyporus umbellatus"], 100),
    ("wine_cap", "Wine cap", "gilled", ["Stropharia rugosoannulata"], 100),
    ("turkey_tail", "Turkey tail", "polypores", ["Trametes versicolor"], 110),
    ("michigan_truffle", "Michigan truffle (limited evidence)", "truffles", ["Tuber canaliculatum"], 100),
    ("bolete_research", "Bolete family group", "boletes", ["Boletaceae"], 150),
    ("other_fungi", "Other or unsupported fungi", "unknown", [], 0),
]


def normalize(value):
    return re.sub(r"[-\s]+", "", value.lower())


def taxonomy(client, root, prior):
    path = root / "taxonomy.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    resolved, queries = {}, []
    for class_id, label, group_id, names, quota in CLASSES:
        for name in names:
            response, url = client.api("taxa/autocomplete", {"q": name, "per_page": 30})
            rank = "species" if " " in name else "family" if name.endswith("aceae") else "genus"
            found = [t for t in response.get("results", []) if t.get("rank") == rank and t.get("is_active", True) and t.get("iconic_taxon_name") == "Fungi"
                     and (normalize(t.get("name", "")) == normalize(name) or normalize(t.get("matched_term", "")) == normalize(name))]
            if len(found) != 1:
                raise ValueError(f"Ambiguous or missing official taxon resolution for {name}: {[(t.get('id'),t.get('name')) for t in response.get('results', [])]}")
            taxon = found[0]
            queries.append({"classId": class_id, "groupId": group_id, "label": label, "requestedName": name,
                            "name": taxon["name"], "taxon_id": taxon["id"], "rank": taxon["rank"], "resolutionUrl": url,
                            "quota": quota // len(names) + (quota % len(names) if name == names[0] else 0)})
            resolved[str(taxon["id"])] = taxon
            print(json.dumps({"event": "taxon-resolved", "name": name, "resolved": taxon["name"], "id": taxon["id"]}), flush=True)
    ids = sorted({row["taxon"]["id"] for row in prior})
    for start in range(0, len(ids), 30):
        response, url = client.api("taxa/" + ",".join(map(str, ids[start:start+30])), {})
        for taxon in response["results"]:
            resolved[str(taxon["id"])] = taxon
    data = {"retrievedAt": now(), "queries": queries,
            "taxa": {key: {field: val.get(field) for field in ("id", "name", "rank", "ancestor_ids")} for key, val in resolved.items()}}
    missing = set(map(str, ids)) - set(data["taxa"])
    if missing:
        raise ValueError(f"Unresolved inherited taxa: {missing}")
    write_json(path, data)
    return data


def assign_class(ancestors, queries):
    matches = {q["classId"] for q in queries if q["taxon_id"] in ancestors}
    if len(matches) > 1:
        raise ValueError(f"Overlapping research class taxonomy: {matches}")
    return next(iter(matches)) if matches else "other_fungi"


def photo_record(observation, query, queries):
    taxon, observer = observation.get("taxon") or {}, observation.get("user") or {}
    if observation.get("quality_grade") != "research" or observation.get("license_code") not in LICENSES:
        return None
    if not observer.get("id") or not taxon.get("id") or observer.get("suspended") or observer.get("spam"):
        return None
    ancestors = set(taxon.get("ancestor_ids", [])) | {taxon["id"]}
    if query["taxon_id"] not in ancestors:
        return None
    for photo in observation.get("photos", []):
        code, photo_id = photo.get("license_code"), photo.get("id")
        parsed = urllib.parse.urlparse(photo.get("url", ""))
        if (code not in LICENSES or parsed.scheme != "https" or parsed.hostname != "inaturalist-open-data.s3.amazonaws.com"
                or photo.get("hidden") or photo.get("flags") or not photo.get("attribution") or not isinstance(photo_id, int)
                or not re.fullmatch(rf"/photos/{photo_id}/square\.[A-Za-z0-9]+", parsed.path)):
            continue
        large = urllib.parse.urlunparse(parsed._replace(path=parsed.path.replace("/square.", "/large."), query="", fragment=""))
        class_id = assign_class(ancestors, queries)
        return {"id": f"inat-photo-{photo_id}", "path": f"images/inat-photo-{photo_id}.jpg", "photoId": photo_id,
                "observationId": observation["id"], "observerId": observer["id"], "groupId": f"inat-observer-{observer['id']}",
                "sourceUrl": f"https://www.inaturalist.org/observations/{observation['id']}", "photoUrl": f"https://www.inaturalist.org/photos/{photo_id}",
                "downloadUrl": large, "photoLicense": {"code": code, "url": LICENSES[code], "attribution": photo["attribution"]},
                "observationLicense": {"code": observation["license_code"], "url": LICENSES[observation["license_code"]]},
                "taxon": {key: taxon.get(key) for key in ("id", "name", "rank", "ancestor_ids")},
                "classId": class_id, "weakLabel": class_id, "cohort": "new_v3", "inherited": False,
                "queryTaxonId": query["taxon_id"], "labelEvidence": "Weak observation taxonomy; image identity and form unreviewed",
                "annotationStatus": "unreviewed", "annotations": [], "synthetic": False, "distanceM": None, "retrievedAt": now()}
    return None


def summarize(rows):
    return {"images": len(rows), "observers": len({r["observerId"] for r in rows}), "bytes": sum(r["bytes"] for r in rows),
            "newImages": sum(not r["inherited"] for r in rows), "classes": dict(Counter(r["classId"] for r in rows)),
            "splitCounts": {split: dict(Counter(r["classId"] for r in rows if r.get("split") == split)) for split in ("train", "validation", "test")}}


def acquire(root, parent):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise ValueError("Frozen output exists; refusing mutation")
    inherited, provenance = prepare_expansion(parent, root)
    rows = inherited["images"]
    client = Client(max_media_bytes=1_980_000_000, api_delay=1.1)
    taxa = taxonomy(client, root, rows)
    queries = taxa["queries"]
    prior_splits = {r["groupId"]: r["split"] for r in rows}
    for row in rows:
        taxon = taxa["taxa"][str(row["taxon"]["id"])]
        ancestors = set(taxon.get("ancestor_ids") or []) | {taxon["id"]}
        row.update(classId=assign_class(ancestors, queries), inherited=True, cohort="inherited_v2")
        row["previousWeakLabel"] = row["weakLabel"]
        row["weakLabel"] = row["classId"]
        row["taxon"]["ancestor_ids"] = sorted(ancestors)
    seen_ids = {r["photoId"] for r in rows}
    seen_obs = {r["observationId"] for r in rows}
    seen_hashes = {r["sha256"] for r in rows}
    seen_pixels = {r["normalizedPixelSha256"] for r in rows}
    observer_counts = Counter(r["observerId"] for r in rows)
    skipped, log = [], []
    # New photos may join an inherited group, but never move it to a different partition.
    for query in queries:
        retained = 0
        for page in range(1, 6):
            if retained >= query["quota"]:
                break
            response, url = client.api("observations", {"taxon_id": query["taxon_id"], "quality_grade": "research", "photos": "true",
                "captive": "false", "photo_license": ",".join(LICENSES), "license": ",".join(LICENSES), "per_page": 200,
                "page": page, "order_by": "id", "order": "desc"})
            log.append({"url": url, "classId": query["classId"], "totalAvailable": response.get("total_results"), "retrievedAt": now()})
            candidates, page_counts = [], observer_counts.copy()
            for obs in response.get("results", []):
                row = photo_record(obs, query, queries)
                if not row or row["photoId"] in seen_ids or row["observationId"] in seen_obs or page_counts[row["observerId"]] >= 8:
                    continue
                candidates.append(row)
                seen_ids.add(row["photoId"]); seen_obs.add(row["observationId"]); page_counts[row["observerId"]] += 1
                if len(candidates) >= query["quota"] - retained:
                    break
            with ThreadPoolExecutor(max_workers=3) as pool:
                pending = [(r, pool.submit(download_one, client, root, r)) for r in candidates]
                for candidate, future in pending:
                    try:
                        row = future.result()
                        if row["sha256"] in seen_hashes or row["normalizedPixelSha256"] in seen_pixels:
                            skipped.append({"id": row["id"], "reason": "exact or normalized duplicate"}); continue
                        # Quarantine new cross-partition near duplicates rather than alter inherited test membership.
                        neighbors = {prior_splits[p["groupId"]] for p in rows if p["groupId"] in prior_splits
                                     and (int(row["perceptualDifferenceHash"],16)^int(p["perceptualDifferenceHash"],16)).bit_count() <= 2}
                        if row["groupId"] in prior_splits:
                            neighbors.add(prior_splits[row["groupId"]])
                        if len(neighbors) > 1:
                            skipped.append({"id": row["id"], "reason": "near-duplicate connects inherited partitions"}); continue
                        seen_hashes.add(row["sha256"]); seen_pixels.add(row["normalizedPixelSha256"])
                        rows.append(row); observer_counts[row["observerId"]] += 1; retained += 1
                        if len(rows) % 100 == 0:
                            write_json(root / "manifest-progress.json", {"schema": SCHEMA,"status":"not-frozen", "images":rows,"summary":summarize(rows)})
                            print(json.dumps({"event":"progress", "summary":summarize(rows)}),flush=True)
                    except (OSError, ValueError, RuntimeError) as exc:
                        skipped.append({"id":candidate["id"],"reason":str(exc)})
            write_json(root / "acquisition-progress.json", {"summary":summarize(rows), "queries":log,"skipped":skipped})
            if len(response.get("results",[])) < 200:
                break
        print(json.dumps({"event":"query-complete","name":query["name"],"accepted":retained,"total":len(rows)}),flush=True)
    splits = freeze_splits(rows, seed=inherited["splitPolicy"]["seed"], prior_splits=prior_splits)
    classes = [{"classId":cid,"label":label,"groupId":gid,"researchTaxonNames":names,
                "certificationScope": "not inferred from model labels"} for cid,label,gid,names,quota in CLASSES]
    manifest = {"schema":SCHEMA,"status":"frozen","title":"Licensed mushroom-group research photos V3", "createdAt":now(),
                "images":rows,"classes":classes,"summary":summarize(rows),"splitPolicy":splits,"parentDataset":provenance,
                "acquisition":{"queries":log,"skipped":skipped,"apiRequests":client.api_calls,"downloadedBytes":client.media_bytes,
                               "mediaConcurrency":3,"apiMinimumIntervalSeconds":1.1,"mediaBudgetBytes":2_000_000_000},
                "limitations":["Weak observation taxonomy, not verified image contents, species or morphology.",
                    "Research genus/family scope differs from Michigan certification species and exclusions.",
                    "Entoloma abortivum photos may show either form; this classifier cannot approve the aborted form.",
                    "No woodland-background negatives, physical ranges, flight validation, or mushroom detection labels.",
                    "Inherited V2 test examples are reused evaluation, not a newly untouched source.",
                    "License and photographer attribution must accompany redistribution."]}
    write_json(root/"manifest.json",manifest)
    write_json(root/"manifest-sha256.json",{"sha256":digest((root/"manifest.json").read_bytes()),"createdAt":now()})
    write_json(root/"summary.json",manifest["summary"])
    print(json.dumps({"event":"frozen","summary":manifest["summary"]}),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=Path("data/public-fungi-v3"))
    parser.add_argument("--parent",type=Path,default=Path("data/public-fungi-v2/manifest.json"))
    args=parser.parse_args(); acquire(args.output_dir,args.parent)

