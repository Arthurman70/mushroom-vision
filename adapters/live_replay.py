"""Replay real saved detector results or an explicitly simulated RGBD fixture.

Run as python -m adapters.live_replay. JSONL stdout (--emit) contains packets only.
Replay never invents distance, depth, labels, GNSS, or cleared flight corridors.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import uuid

import numpy as np
from PIL import Image

from adapters.review_stream import Station
from vision.live_mapping import LiveMapper, identifier


def read_json(path, limit=32 * 1024 * 1024):
    if path.stat().st_size > limit:
        raise ValueError("JSON file exceeds bounded input size")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def contained(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Frame/sidecar paths must stay inside their collection")
    return result


def image_metadata(path, include_image=True, expected_hash=None):
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Frame must exist and be at most 8 MiB")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and digest != expected_hash:
        raise ValueError("Replay frame checksum mismatch")
    with Image.open(io.BytesIO(raw)) as opened:
        if opened.format not in {"JPEG", "PNG"} or opened.width * opened.height > 8000000 or getattr(opened, "n_frames", 1) != 1:
            raise ValueError("Frame must be a still JPEG/PNG of at most 8 megapixels")
        rgb = opened.convert("RGB")
        meta = {"width": rgb.width, "height": rgb.height, "sha256": digest,
                "mimeType": "image/jpeg" if opened.format == "JPEG" else "image/png"}
    if include_image:
        meta["imageBase64"] = base64.b64encode(raw).decode("ascii")
    return meta, rgb


def completed_runs(database):
    """Exact-frame original-RGB results only; never guess identity from pixels alone."""
    if database is None or not database.is_file():
        return {}
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    connection.row_factory = sqlite3.Row
    records = {}
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(runs)")}
        if "video_provenance" not in columns:
            return records
        rows = connection.execute("SELECT id, photo_hash, settings, result, video_provenance FROM runs WHERE status='completed' AND video_provenance IS NOT NULL ORDER BY created_at DESC LIMIT 1000")
        for row in rows:
            try:
                provenance, result, settings = (json.loads(row[name]) for name in ("video_provenance", "result", "settings"))
                if settings.get("scale") != 1 or not isinstance(provenance, dict):
                    continue
                method = next((m for m in result.get("methods", []) if m.get("id") == "original"), None)
                if not method or result.get("image", {}).get("sha256") != row["photo_hash"] or provenance.get("sha256") != row["photo_hash"]:
                    continue
                key = (provenance.get("id"), row["photo_hash"], provenance.get("timestampSeconds"), provenance.get("sourceId"))
                records.setdefault(key, {"runId": row["id"], "candidates": method["candidates"], "model": result.get("model"),
                                         "seconds": method.get("seconds"), "image": result["image"]})
            except (ValueError, TypeError, KeyError):
                continue
    finally:
        connection.close()
    return records


def replay_collection(collection, database, session_id, *, include_image=True, max_frames=1000, group_predictor=None):
    manifest = read_json(collection / "manifest.json")
    if manifest.get("schema") != "mycoscout.video-collection.v1" or not isinstance(manifest.get("frames"), list) or len(manifest["frames"]) > 100000:
        raise ValueError("Expected a bounded video collection manifest")
    runs = completed_runs(database)
    sources = {s["id"]: s for s in manifest.get("sources", [])}
    frames = sorted(manifest["frames"], key=lambda f: (f["sourceId"], f["timestampSeconds"], f["id"]))
    mappers = {}
    for index, item in enumerate(frames[:max_frames]):
        source_id = item["sourceId"]
        if source_id not in mappers:
            # Every source has a separate capture clock, coordinate frame and track namespace.
            suffix = hashlib.sha256(source_id.encode()).hexdigest()[:8]
            child_session = session_id if len(sources) == 1 else f"{session_id[:110]}-{suffix}"
            mappers[source_id] = LiveMapper(child_session)
        mapper = mappers[source_id]
        meta, rgb = image_metadata(contained(collection, item["path"]), include_image, item["sha256"])
        if (meta["width"], meta["height"]) != (item["width"], item["height"]):
            raise ValueError("Manifest image geometry does not match frame")
        run = runs.get((item["id"], item["sha256"], item["timestampSeconds"], source_id))
        if run and (run["image"].get("width"), run["image"].get("height")) != (meta["width"], meta["height"]):
            run = None
        candidates = run["candidates"] if run else []
        if group_predictor is not None:
            from vision.live_groups import classify_candidates
            candidates = classify_candidates(rgb, candidates, group_predictor)
        frame = {"frameId": item["id"], "sequence": index, "timestampSeconds": item["timestampSeconds"],
            "simulationOnly": True, "mapRevision": "rgb-unlocated-v1", "image": meta,
            "detections": candidates, "detectorStatus": "completed" if run else "unprocessed",
            "detector": {"source": "saved-original-rgb-run", "runId": run["runId"], "model": run["model"], "inferenceSeconds": run["seconds"], "classifyGroups": group_predictor is not None} if run else None,
            "provenance": {"sourceId": source_id, "sourceGroupId": item.get("sourceGroupId"), "frameId": item["id"],
                "videoHash": item.get("videoHash"), "timestampSeconds": item["timestampSeconds"], "sha256": item["sha256"],
                "evaluationOnly": bool(item.get("evaluationOnly", True)), "trainingAllowed": False,
                "source": sources.get(source_id), "replay": True, "synthetic": False, "realRecording": True, "actualDistanceM": None}}
        yield mapper.process(frame, rgb=rgb)


def replay_fixture(path, session_id, *, include_image=True, max_frames=1000):
    fixture = read_json(path)
    if fixture.get("schema") != "mycoscout.live-sensor-fixture.v1" or fixture.get("simulationOnly") is not True:
        raise ValueError("Only explicitly simulated fixtures are accepted")
    mapper = LiveMapper(session_id)
    for index, entry in enumerate(fixture["frames"][:max_frames]):
        frame = {**fixture["defaults"], **entry, "simulationOnly": True, "sequence": index}
        depth_spec = frame.pop("depthFixture", None)
        width, height = fixture["imageSize"]
        # A clearly labeled generated checker image exercises transport, not vision accuracy.
        yy, xx = np.indices((height, width))
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[:] = np.where(((xx // 20 + yy // 20) % 2)[..., None], [50, 65, 45], [75, 85, 65])
        for detection in frame.get("detections", []):
            x1, y1, x2, y2 = map(int, detection["bboxPixelsXYXY"])
            pixels[y1:y2, x1:x2] = [220, 180, 90]
        rgb = Image.fromarray(pixels)
        stream = io.BytesIO(); rgb.save(stream, format="JPEG", quality=90)
        raw = stream.getvalue()
        frame["image"] = {"width": width, "height": height, "mimeType": "image/jpeg", "sha256": hashlib.sha256(raw).hexdigest()}
        if include_image:
            frame["image"]["imageBase64"] = base64.b64encode(raw).decode("ascii")
        frame["provenance"] = {"simulationOnly": True, "syntheticFixture": path.name, "trainingAllowed": False,
                               "evaluationOnly": True, "replay": True, "note": "Synthetic geometry test, not a mushroom-detection result"}
        depth = None
        if depth_spec:
            if depth_spec.get("kind") != "constant-plane":
                raise ValueError("Unknown synthetic depth fixture")
            depth = np.full((height, width), depth_spec["meters"], dtype=np.float32)
        yield mapper.process(frame, depth=depth, rgb=rgb)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--collection", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--testbench-db", type=Path, default=Path("data/testbench/testbench.sqlite"))
    parser.add_argument("--station")
    parser.add_argument("--session-id", default="replay-" + str(uuid.uuid4()))
    parser.add_argument("--interval", type=float, default=0)
    parser.add_argument("--token-env", default="MYCOSCOUT_TOKEN")
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--classify-groups", action="store_true", help="Optional bounded native-crop group guesses on real recorded images")
    args = parser.parse_args(argv)
    if not 0 <= args.interval <= 60 or not 1 <= args.max_frames <= 10000:
        parser.error("Interval must be 0..60 seconds and max-frames 1..10000")
    if not (args.emit or args.station or args.output):
        parser.error("Choose --emit, --output or --station")
    if args.fixture and args.classify_groups:
        parser.error("Group recognition is not evaluated on synthetic geometry fixtures")
    identifier(args.session_id, "sessionId")
    station = Station(args.station, token=os.environ.get(args.token_env)) if args.station else None
    group_predictor = None
    if args.classify_groups:
        from vision.live_groups import load_group_predictor
        group_predictor = load_group_predictor()
    packets = replay_fixture(args.fixture, args.session_id, max_frames=args.max_frames) if args.fixture else replay_collection(args.collection, args.testbench_db, args.session_id, max_frames=args.max_frames, group_predictor=group_predictor)
    output = args.output.open("x", encoding="utf8") if args.output else None
    count, located, processed, track_ids, started_sessions = 0, 0, 0, set(), set()
    try:
        for packet in packets:
            if count and args.interval:
                time.sleep(args.interval)
            line = json.dumps(packet, allow_nan=False, separators=(",", ":"))
            if station:
                if packet["sessionId"] not in started_sessions:
                    station.request("/api/live/session", {"sessionId": packet["sessionId"], "mode": "replay"})
                    started_sessions.add(packet["sessionId"])
                station.request("/api/live/frames", packet)
            if args.emit:
                print(line, flush=True)
            if output:
                output.write(line + "\n"); output.flush()
            count += 1
            processed += packet["detectorStatus"] == "completed"
            located += sum(d["positionEnuM"] is not None for d in packet["detections"])
            track_ids.update((packet["sessionId"], t["trackId"]) for t in packet["tracks"])
    finally:
        if output:
            output.close()
    print(json.dumps({"frames": count, "processedFrames": processed, "locatedObservations": located, "proposedTracks": len(track_ids), "humanVerifiedTracks": 0, "actuationEnabled": False}), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as error:
        print(f"Live replay failed: {error}", file=sys.stderr)
        raise SystemExit(1)
