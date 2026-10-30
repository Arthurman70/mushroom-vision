"""Assemble the reviewed source selection without altering extracted images."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


def prepare(root):
    acquisition = root / "data/morel-video-acquisition-v1"
    chosen = {"youtube-" + value for value in (
        "1Z9FgVpYris", "ng7VOJHum7U", "m9aGEh33w0I", "OhN1BQgK2L8",
        "go_1s6wVScI", "xTJDZ3IkXbU")}
    batches = [root / ("data/morel-video-frames-v1-batch-" + suffix) for suffix in ("1b", "2", "3", "4")]
    sources, frames, copies, used = [], [], [], set()
    for directory in batches:
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        for source in payload["sources"]:
            if source["id"] not in chosen:
                continue
            if source["id"] in used:
                raise ValueError("Duplicate source")
            used.add(source["id"])
            source = deepcopy(source)
            # Group both hunts by this creator conservatively. Future train/test
            # exports must not split reused intros/scenes across those hunts.
            if source["id"] in {"youtube-m9aGEh33w0I", "youtube-xTJDZ3IkXbU"}:
                group = "youtube:creator:JoshuaMcConkeyFishing"
                source["sourceGroupId"] = source["partitionGroup"] = group
            sources.append(source)
            for original in payload["frames"]:
                if original["sourceId"] != source["id"]:
                    continue
                frame = deepcopy(original)
                frame["sourceGroupId"] = frame["partitionGroup"] = source["sourceGroupId"]
                image = directory / frame["path"]
                if hashlib.sha256(image.read_bytes()).hexdigest() != frame["sha256"]:
                    raise ValueError("Input image checksum mismatch")
                if frame["labels"] or frame["annotation"]["complete"] or frame["trainingAllowed"]:
                    raise ValueError("Expected unreviewed frames only")
                frames.append(frame)
                copies.append((image, frame["path"]))
    if used != chosen:
        raise ValueError("Missing selected sources")
    output = root / "data/morel-video-feed-v1"
    if output.exists():
        existing = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if existing["sources"] != sources or existing["frames"] != frames:
            raise ValueError("Prepared collection already exists with different records; preserve it")
        for frame in frames:
            if hashlib.sha256((output / frame["path"]).read_bytes()).hexdigest() != frame["sha256"]:
                raise ValueError("Existing prepared frame failed its checksum")
    else:
        output.mkdir()
        (output / "frames").mkdir()
        for original, relative in copies:
            shutil.copyfile(original, output / relative)
    manifest = {"schema": "mycoscout.video-collection.v1",
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "sources": sources, "frames": frames,
                "settings": {"selection": "Six morel field-hunt sources; 60 unreviewed frames per video",
                             "nativeRenditionPixels": True, "imageChanges": "None since extraction"},
                "warnings": ["Source titles are discovery hints, not species labels.",
                             "Some frames contain people, titles, other fungi or no visible mushrooms; review every frame.",
                             "No physical distance is known. Pixel size and relative sharpness are not meter measurements.",
                             "Same-creator Joshua McConkey videos share a partition group to reduce reuse leakage.",
                             "Human annotation and permission are required before training."],
                "autoTraining": False, "actuationEnabled": False}
    if not (output / "manifest.json").exists():
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(acquisition.glob("download-*.json"))]
    safe_records = [{key: row.get(key) for key in (
        "id", "title", "creator", "sourceUrl", "channelUrl", "retrievedAt", "license", "licenseEvidence",
        "licenseUrl", "sha256", "bytes", "durationSeconds", "width", "height", "formatId", "path")}
        | {"inGradingFeed": "youtube-" + row["id"] in chosen} for row in records if row.get("status") == "downloaded"]
    report = {"schema": "mycoscout.morel-acquisition-report.v1", "sources": safe_records,
              "feedVideoCount": len(sources), "feedFrames": len(frames),
              "feedMinutes": sum(source["video"]["durationSeconds"] for source in sources) / 60,
              "feedResolution": "Five 720p platform renditions and one 360p rendition; extracted at native rendition pixels",
              "referenceOnly": {"videoId": "S_ruFMIM0qs", "reason": "Visual review found mostly slide-style locations/advice; excluded from temporal field detection feed"},
              "researchOnlyCollection": "data/morel-research-frames-v1/manifest.json",
              "researchOnlyFrames": 60, "researchOnlyTrainingEnabled": False,
              "groundTruthCreated": False, "modelTrainingPerformed": False}
    (acquisition / "collection-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"sources": len(sources), "frames": len(frames), "minutes": round(report["feedMinutes"], 2), "output": str(output)}))


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1])
