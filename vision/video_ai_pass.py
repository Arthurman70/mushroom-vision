"""Resumable, local AI proposals for every sampled frame, never human labels."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from PIL import Image

from adapters.merge_video_collections import read_collection

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "mycoscout.ai-video-frame.v1"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_inputs(collections):
    rows, sources, manifests = [], [], []
    seen = set()
    for collection_id, directory in collections:
        # Old research archives retain historical HTTP license attribution
        # strings. These are read as evidence only; this runner never fetches
        # them or changes the evaluation-only/training permission flags.
        root, original, manifest = read_collection(directory, allow_http_attribution=collection_id == "research")
        manifests.append({"collection": collection_id, "manifestSha256": sha(original),
                          "frames": len(manifest["frames"]), "sources": len(manifest["sources"])})
        by_id = {source["id"]: source for source in manifest["sources"]}
        sources.extend({"collection": collection_id, **source} for source in manifest["sources"])
        for frame in manifest["frames"]:
            key = collection_id + "--" + frame["id"]
            if key in seen:
                raise ValueError("Repeated collection/frame identity")
            seen.add(key)
            source = by_id[frame["sourceId"]]
            rows.append({"key": key, "collection": collection_id, "frame": frame,
                         "source": source, "path": root / frame["path"]})
    return rows, sources, manifests


def frame_record(row, detector, predictor=None):
    from adapters.detect_fungi import candidate_records
    from vision.live_groups import classify_candidates
    from vision.metrics import class_agnostic_nms
    started = time.perf_counter()
    frame, source = row["frame"], row["source"]
    image_bytes = row["path"].read_bytes()
    if sha(image_bytes) != frame["sha256"]:
        raise ValueError("Source frame changed after manifest validation")
    with Image.open(row["path"]) as opened:
        image = opened.convert("RGB")
    if image.size != (frame["width"], frame["height"]):
        raise ValueError("Source geometry changed")
    proposals = detector.predict(image)
    valid = candidate_records([p.get("bboxPixelsXYXY", []) for p in proposals],
                              [p.get("confidence", float("nan")) for p in proposals],
                              ["mushroom candidate"] * len(proposals), image.width, image.height)
    candidates = class_agnostic_nms(valid, .5)[:100]
    if predictor is not None:
        candidates = classify_candidates(image, candidates, predictor, max_candidates=5)
    boxes = [{"id": "ai-" + str(index + 1), "bbox": candidate["bboxPixelsXYXY"],
              "score": candidate["confidence"], "label": "Unverified mushroom candidate",
              "classification": candidate.get("classification"), "humanVerified": False,
              "trainingAllowed": False} for index, candidate in enumerate(candidates)]
    evaluation = source.get("evaluationOnly") is not False or source.get("licenseAllowsTraining") is not True or frame.get("evaluationOnly") is True
    return {"schema": SCHEMA, "frameId": frame["id"], "collection": row["collection"],
            "sourceId": source["id"], "sourceGroupId": source.get("sourceGroupId"),
            "videoHash": source.get("videoHash", source.get("sha256")),
            "imageSha256": frame["sha256"], "timestampSeconds": frame["timestampSeconds"],
            "width": frame["width"], "height": frame["height"],
            "status": "completed", "model": detector.info, "boxes": boxes,
            "humanVerified": False, "labelsVerified": False, "trainingAllowed": False,
            "evaluationOnly": evaluation, "sourceLicenseAllowsTraining": source.get("licenseAllowsTraining") is True,
            "completedAt": datetime.now(timezone.utc).isoformat(), "seconds": time.perf_counter() - started,
            "humanMatches": None, "reviewStatus": "awaiting_human_review",
            "emptyPredictionMeaning": "No model proposal does not establish a mushroom-free frame",
            "originalCoordinates": True, "distanceMeters": None, "actuationEnabled": False}


def run(collections, output, detector=None, predictor=None, resume=False, threads=4):
    if type(threads) is not int or not 1 <= threads <= 8:
        raise ValueError("Threads must be an integer from 1 to 8")
    rows, sources, manifests = load_inputs(collections)
    output = Path(output)
    if output.exists() and not resume:
        raise ValueError("Output exists; use --resume with the same immutable input manifests")
    output.mkdir(parents=True, exist_ok=True)
    (output / "frames").mkdir(exist_ok=True)
    identity = {"schema": "mycoscout.ai-video-pass-inputs.v1", "collections": manifests}
    identity_file = output / "inputs.json"
    if identity_file.exists():
        if json.loads(identity_file.read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume inputs changed")
    else:
        atomic_json(identity_file, identity)
    if detector is None:
        from adapters.scout_frame import LocalDetector
        from vision.live_groups import load_group_predictor
        detector = LocalDetector(ROOT / "models/grounding-dino-tiny", device="cpu", threads=threads,
                                 box_threshold=.25, text_threshold=.25)
        predictor = load_group_predictor()
        import torch
        torch.set_num_threads(threads)
    records = []
    started = time.time()
    for index, row in enumerate(rows):
        target = output / "frames" / (row["key"] + ".json")
        if target.exists():
            record = json.loads(target.read_text(encoding="utf-8"))
            if (record.get("schema") != SCHEMA or record.get("imageSha256") != row["frame"]["sha256"] or
                    {k: v for k, v in record.get("model", {}).items() if k != "threads"} !=
                    {k: v for k, v in detector.info.items() if k != "threads"} or record.get("frameId") != row["frame"]["id"] or
                    record.get("collection") != row["collection"]):
                raise ValueError("Existing AI result does not match source/model identity")
        else:
            try:
                record = frame_record(row, detector, predictor)
            except Exception as error:
                record = {"schema": SCHEMA, "collection": row["collection"], "frameId": row["frame"]["id"],
                          "sourceId": row["source"]["id"], "imageSha256": row["frame"]["sha256"],
                          "model": detector.info, "status": "failed", "error": str(error)[:500], "boxes": [],
                          "humanVerified": False, "trainingAllowed": False}
            atomic_json(target, record)
        records.append(record)
        status = {"schema": "mycoscout.ai-video-pass-status.v1", "status": "running", "processedFrames": index + 1,
                  "totalFrames": len(rows), "completedFrames": sum(r["status"] == "completed" for r in records),
                  "failedFrames": sum(r["status"] == "failed" for r in records),
                  "candidateBoxes": sum(len(r["boxes"]) for r in records), "latestFrame": row["frame"]["id"],
                  "elapsedSeconds": sum(r.get("seconds", 0) for r in records), "currentRunWallSeconds": time.time() - started,
                  "humanVerified": False, "automaticTraining": False}
        atomic_json(output / "status.json", status)
        print(json.dumps(status, allow_nan=False), flush=True)
    summary = {"schema": "mycoscout.ai-video-pass.v1", "status": "completed" if not status["failedFrames"] else "completed_with_errors",
               "model": detector.info, "sourceCollections": manifests, "sources": sources,
               "sampledFrames": len(rows), "sourceVideos": len(sources), "processedFrames": len(records),
               "completedFrames": status["completedFrames"], "failedFrames": status["failedFrames"],
               "candidateBoxes": status["candidateBoxes"], "elapsedSeconds": status["elapsedSeconds"],
               "currentRunWallSeconds": time.time() - started,
               "frames": records, "humanVerified": False, "automaticTraining": False,
               "notice": "One AI proposal pass over every imported sampled frame, not every original video frame. Human verification remains separate."}
    atomic_json(output / "report.json", summary)
    atomic_json(output / "status.json", {**status, "status": summary["status"]})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-collection", type=Path, default=ROOT / "data/video-discovery-v1")
    parser.add_argument("--research-collection", type=Path, default=ROOT / "data/morel-research-frames-v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run([("main", args.main_collection), ("research", args.research_collection)], args.output, resume=args.resume, threads=args.threads)


if __name__ == "__main__":
    main()
