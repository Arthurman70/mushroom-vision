"""Build public-only round-4 results and append thirteen actual AI image reviews.

Reads only metadata members from the append archive. Does not deploy, train, edit
frozen experiment artifacts, or change any earlier/remaining review entry.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = [
    ("native", "native", "Whole original image"),
    ("low32", "32px", "Synthetic 32 px, whole frame"),
    ("low16", "16px", "Synthetic 16 px, whole frame"),
    ("low32tiles", "32px-tiled", "Synthetic 32 px, tiles plus context"),
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata(archive):
    with tarfile.open(archive, "r:gz") as source:
        result = []
        for name in ("collection/manifest.json", "collection/ai-suggestions.json"):
            members = [m for m in source.getmembers() if m.name == name]
            if len(members) != 1 or not members[0].isfile() or members[0].size > 25_000_000:
                raise ValueError("Missing, ambiguous or oversized archive metadata: " + name)
            with source.extractfile(members[0]) as stream:
                result.append(json.load(stream))
    return result


def joined_notes(observations, manifest):
    public = {r["id"]: r for r in manifest["frames"]}
    sources = {r["id"]: r for r in manifest["sources"]}
    joined, seen = [], set()
    for note in observations["records"]:
        identity = note["id"]
        if identity in seen or identity not in public or not identity.startswith("inat-photo-"):
            raise ValueError("Reviewed identity is duplicate, unpublished or outside the new photo cohort")
        seen.add(identity)
        frame = public[identity]
        if (frame.get("originalSha256") != note["imageSha256"] or
            (frame["width"], frame["height"]) != (note["width"], note["height"])):
            raise ValueError("Published frame no longer shares the reviewed original coordinate system")
        if note.get("humanVerified") is not False or note.get("complete") is not False or note.get("observation", {}).get("mode") != "original-image":
            raise ValueError("Expected actual incomplete AI original-image observations")
        if note.get("status") not in ("visible", "uncertain"):
            raise ValueError("This public round has no newly established absence labels")
        box = note["selectedTargetPixelsXYXY"]
        if box is not None and (len(box) != 4 or not 0 <= box[0] < box[2] <= frame["width"] or not 0 <= box[1] < box[3] <= frame["height"]):
            raise ValueError("Selected AI box is outside the public image")
        joined.append((note, frame, sources[frame["sourceId"]]))
    return joined


def public_result(report, observations, manifest, protocol):
    if report.get("status") != "completed" or report.get("inputIntegrityVerified") is not True or report.get("trainingPerformed") is not False:
        raise ValueError("Only a completed, unchanged-weight verified report may be published")
    metrics = {r["method"]: r for r in report["summary"]}
    records = {r["id"]: r for r in report["records"]}
    negatives = [r for r in report["records"] if r["kind"] == "prior_ai_negative"]
    conditions = []
    for local_id, public_id, label in CONDITIONS:
        row = metrics[local_id]
        # Unrun negative conditions are null, not invented zeroes.
        measured_negatives = [r["methods"][local_id]["metrics"] for r in negatives if local_id in r["methods"]]
        conditions.append({"id": public_id, "label": label,
            "selectedTargetHits": row["selectedTargetHits"], "selectedTargetCount": row["selectedTargets"],
            "negativeFramesWithAlerts": sum(r["aiReferenceNegativeAlerts"] > 0 for r in measured_negatives) if measured_negatives else None,
            "negativeFrameCount": len(measured_negatives) if measured_negatives else None,
            "negativeAlertCount": sum(r["aiReferenceNegativeAlerts"] for r in measured_negatives) if measured_negatives else None,
            "meanSecondsPerFrame": row["meanSecondsPerFrame"], "measurementFrames": row["frames"]})
    frames = []
    for note, frame, source in joined_notes(observations, manifest):
        result = records[note["id"]]
        if result["sha256"] != note["imageSha256"] or result["selectedTargetPixelsXYXY"] != note["selectedTargetPixelsXYXY"]:
            raise ValueError("Evaluation reference differs from the actually observed image")
        results = []
        for local_id, public_id, _ in CONDITIONS:
            if local_id not in result["methods"]:
                continue
            method = result["methods"][local_id]
            results.append({"conditionId": public_id, "selectedTargetHit": method["metrics"]["selectedTargetHit"],
                            "alertCount": method["metrics"]["predictions"], "elapsedSeconds": method["seconds"]})
        frames.append({"frameId": frame["id"], "imageSha256": frame["sha256"], "width": frame["width"], "height": frame["height"],
            "status": note["status"], "note": note["notes"], "groupHint": note.get("groupHintFromSource", ""),
            "selectedTargetPixelsXYXY": note["selectedTargetPixelsXYXY"], "results": results,
            "source": {key: source.get(key, "") for key in ("title", "attribution", "license", "sourceUrl", "licenseUrl")}})
    return {"schema": "mushroom-vision.public-testing-round4.v1", "title": "Mushroom Vision: new photo testing round",
        "completedAt": report["completedAt"],
        "summary": {"positiveReferences": metrics["native"]["selectedTargets"], "negativeReferences": len(negatives),
                    "referenceAuthority": "ai-incomplete", "weightsChanged": False},
        "protocol": {"detectorName": report["model"]["model"], "iouThreshold": protocol["iouThreshold"],
            "scoreThreshold": protocol["scoreThreshold"],
            "notes": "Thirteen new original photographs were inspected by AI; nine have one approximate selected target, and four remain cropped or uncertain qualitative checks. Four older AI absence references were tested only at native resolution and are summarized separately. The entire image was downsampled until the selected target was approximately 32 or 16 pixels wide, then upsampled back to its original size with LANCZOS. No crop or paste creates the synthetic views; tiled inference uses four overlapping tiles plus whole-image context. These tests measure no actual distance or flight range. At synthetic 32 px, tiled scanning matched eight of nine selected targets versus seven for whole-frame scanning, but produced 49 unmatched alerts versus 11. The incomplete AI references cannot establish that these extra alerts are false positives. Model weights were unchanged; this selected photo test is not a field accuracy estimate."},
        "conditions": conditions, "frames": frames}


def reviewed_suggestions(original, observations, manifest):
    result = copy.deepcopy(original)
    entries = {r["frameId"]: r for r in result["frames"]}
    if len(entries) != len(result["frames"]):
        raise ValueError("Duplicate public suggestion identities")
    changed = set()
    for note, frame, _ in joined_notes(observations, manifest):
        entry = entries[frame["id"]]
        if (entry["imageSha256"] != frame["sha256"] or entry["sourceId"] != frame["sourceId"] or
            entry.get("status") != "pending" or entry.get("audit") is not None):
            raise ValueError("Only matching, previously pending new photos may receive these reviews")
        box = note["selectedTargetPixelsXYXY"]
        boxes = [] if box is None else [{"bboxPixelsXYXY": list(box), "groupId": "unknown",
            "evidence": "One approximate selected region from direct AI original-image inspection; remaining mushrooms were not exhaustively boxed.",
            "humanVerified": False, "trainingAllowed": False}]
        entry.update(status="completed", model="AI original-image visual review; incomplete and unverified",
            audit={"imageId": frame.get("bulkImageId") or "image-" + frame["sha256"], "reviewerKind": "ai",
                "humanVerified": False, "status": note["status"], "complete": False, "boxes": boxes,
                "confounders": [], "confounderTags": [], "observation": {"mode": "original-image", "viewed": True},
                "evidence": note["notes"], "revision": 1, "reviewId": "round4-" + frame["id"], "trainingAllowed": False})
        changed.add(frame["id"])
    for old, new in zip(original["frames"], result["frames"]):
        if old["frameId"] != new["frameId"] or (old["frameId"] not in changed and old != new):
            raise ValueError("An earlier or unreviewed frame was changed")
    result["run"].update(status="in-progress", reviewedFrames=sum(r.get("status") == "completed" for r in result["frames"]),
                         totalFrames=len(result["frames"]), humanVerified=False)
    return result, changed


def write_reproducible(path, value):
    encoded = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise FileExistsError("Refusing to replace a different publication artifact: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(encoded)


def publish(archive):
    manifest, suggestions = metadata(archive)
    observations_path = ROOT / "experiments/results/new-fungi-v4/ai-observations.json"
    report_path = ROOT / "experiments/results/new-fungi-v4/evaluation/report.json"
    protocol_path = report_path.parent / "protocol.json"
    observations, report, protocol = read(observations_path), read(report_path), read(protocol_path)
    if (len(observations["records"]) != 13 or len(manifest["frames"]) != 4660 or
        report["protocolSha256"] != sha(protocol_path) or
        protocol["inputSha256"][observations_path.relative_to(ROOT).as_posix()] != sha(observations_path)):
        raise ValueError("Frozen round-4 publication counts or reference checksums differ")
    public = public_result(report, observations, manifest, protocol)
    updated, changed = reviewed_suggestions(suggestions, observations, manifest)
    if len(changed) != 13 or updated["run"]["reviewedFrames"] != 4480:
        raise ValueError("Unexpected public AI review totals")
    public_path = ROOT / "community/public/testing-round4.json"
    suggestions_path = ROOT / "data/community-round4/reviewed-ai-suggestions.json"
    write_reproducible(public_path, public)
    write_reproducible(suggestions_path, updated)
    result = {"publicFrames": len(public["frames"]), "newReviews": len(changed), "reviewedFrames": 4480,
              "priorRecordsPreserved": 4467, "newPendingPreserved": 180, "humanVerified": False,
              "publicSha256": sha(public_path), "suggestionsSha256": sha(suggestions_path),
              "publicBytes": public_path.stat().st_size, "suggestionsBytes": suggestions_path.stat().st_size}
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "dist/community-round4-append.tar.gz")
    publish(parser.parse_args().archive)
