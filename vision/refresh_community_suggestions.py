"""Resumable local suggestions from an already activated reviewed detector.

Never publishes, trains, changes rights, verifies a label, or selects an active
model. The caller must verify the current active pointer and release approval.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time
import uuid

from PIL import Image, ImageOps

from adapters.export_reviewed_detection_dataset import canonical, digest, inside
from vision.compare_candidate import checked_predictions, file_sha, read_json, validate_job_metadata
from vision.training_pipeline import load_detector, predictions_for, tensor_image

SCHEMA = "mycoscout.public-ai-suggestions.v1"
ID = re.compile(r"[A-Za-z0-9_-]{1,160}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")
LICENSE = re.compile(r"CC BY(?:-SA)?(?:\s+\d+(?:\.\d+)?)?(?: \(YouTube\))?\Z")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+"."+uuid.uuid4().hex+".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(path):
    """OS releases this lock after cancellation/crash; stale files do not block resume."""
    with Path(path).open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0"); stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise ValueError("Another refresh owns this output") from exc
            try:
                yield
            finally:
                stream.seek(0); msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise ValueError("Another refresh owns this output") from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def load_public_collection(directory, max_frames=1000):
    directory = Path(directory).resolve()
    manifest, manifest_sha = read_json(inside(directory, "manifest.json"))
    if manifest.get("schema") != "mycoscout.community-collection.v1" or type(max_frames) is not int or not 1 <= max_frames <= 1000:
        raise ValueError("Expected the bounded packaged public collection")
    sources, frames = manifest.get("sources"), manifest.get("frames")
    if (not isinstance(sources, list) or not 1 <= len(sources) <= 1000 or not isinstance(frames, list) or
            not 1 <= len(frames) <= max_frames):
        raise ValueError("Public collection exceeds the frame/source bound or is empty")
    source_map = {}
    for source in sources:
        if (not ID.fullmatch(str(source.get("id", ""))) or source["id"] in source_map or
                source.get("licenseAllowsTraining") is not True or not LICENSE.fullmatch(str(source.get("license", ""))) or
                not HASH.fullmatch(str(source.get("videoHash", ""))) or not source.get("sourceGroupId") or
                type(source.get("evaluationOnly")) is not bool):
            raise ValueError("Source is not an eligible already-packaged public source")
        source_map[source["id"]] = source
    seen = set()
    for frame in frames:
        source = source_map.get(frame.get("sourceId"))
        if (source is None or not ID.fullmatch(str(frame.get("id", ""))) or frame["id"] in seen or
                not HASH.fullmatch(str(frame.get("sha256", ""))) or frame.get("sourceGroupId") != source["sourceGroupId"] or
                frame.get("videoHash") != source["videoHash"] or frame.get("sourceLicenseAllowsTraining") is not True or
                type(frame.get("evaluationOnly")) is not bool or (source["evaluationOnly"] and not frame["evaluationOnly"]) or
                type(frame.get("width")) is not int or type(frame.get("height")) is not int or
                min(frame["width"], frame["height"]) < 16 or frame["width"]*frame["height"] > 20_000_000):
            raise ValueError("Public frame provenance/dimensions are invalid")
        seen.add(frame["id"])
        path = inside(directory, frame["path"])
        if path.stat().st_size > 8*1024*1024 or file_sha(path) != frame["sha256"]:
            raise ValueError("Public source image bytes changed")
        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"} or getattr(image, "n_frames", 1) != 1:
                raise ValueError("Public frame must be a still PNG or JPEG")
            rgb = ImageOps.exif_transpose(image)
            if rgb.size != (frame["width"], frame["height"]):
                raise ValueError("Public frame geometry changed")
    return manifest, manifest_sha


def active_job_identity(directory):
    import torch
    directory = Path(directory).resolve()
    card, card_sha = read_json(inside(directory, "model-card.json"))
    complete, complete_sha = read_json(inside(directory, "COMPLETE.json"))
    checkpoint = inside(directory, "candidate.pt")
    checkpoint_sha = file_sha(checkpoint)
    if (complete.get("modelCardSha256") != card_sha or not HASH.fullmatch(str(card.get("datasetManifestSha256", ""))) or
            not isinstance(card.get("datasetId"), str)):
        raise ValueError("Active job must have a completed, hash-bound model card")
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=True)
    threshold = validate_job_metadata(card, complete, metadata, checkpoint_sha, {"datasetId": card["datasetId"]},
                                      card["datasetManifestSha256"], card_sha)
    version = "reviewed-"+checkpoint_sha[:16]
    model = {"name": "Reviewed mushroom detector " + checkpoint_sha[:12], "model": card["architecture"],
             "version": version, "resolvedRevision": "sha256:"+checkpoint_sha, "checkpointSha256": checkpoint_sha,
             "modelCardSha256": card_sha, "completeSha256": complete_sha, "boxThreshold": threshold,
             "device": "cpu", "threads": 4, "cropPolicy": "full_frame", "nmsIou": .5, "maxBoxes": 64,
             "downloadsEnabled": False, "humanVerified": False, "trainingAllowed": False,
             "scoreMeaning": "Uncalibrated detection score, not species probability",
             "runtime": {package: importlib.metadata.version(package) for package in ("torch", "torchvision", "Pillow")}}
    return model, checkpoint


def cache_key(model, frame):
    return digest(canonical({"model": model, "imageSha256": frame["sha256"], "width": frame["width"], "height": frame["height"]}))


def public_boxes(predictions, frame, threshold):
    return [{"bbox": box["bbox"], "score": box["score"], "label": "Unverified mushroom candidate",
             "humanVerified": False, "trainingAllowed": False}
            for box in checked_predictions(predictions, frame["width"], frame["height"], threshold, apply_nms=True)[:64]]


def cached_boxes(path, key, model, frame):
    value, _ = read_json(path)
    payload = value.get("payload")
    if (value.get("schema") != "mycoscout.reviewed-suggestion-cache.v1" or not isinstance(payload, dict) or
            value.get("payloadSha256") != digest(canonical(payload)) or payload.get("cacheKey") != key or
            payload.get("model") != model or payload.get("imageSha256") != frame["sha256"] or
            payload.get("width") != frame["width"] or payload.get("height") != frame["height"]):
        raise ValueError("Cached model/image identity or payload changed")
    boxes = payload.get("boxes")
    if not isinstance(boxes, list) or len(boxes) > 64 or any(not isinstance(box, dict) for box in boxes):
        raise ValueError("Invalid cached suggestion boxes")
    raw = [{"bbox": box.get("bbox"), "score": box.get("score")} for box in boxes]
    if public_boxes(raw, frame, model["boxThreshold"]) != boxes:
        raise ValueError("Cached boxes contain invalid labels, rights flags, geometry or scores")
    return boxes


def validate_completed_output(result, manifest, manifest_sha, model):
    run = result.get("run", {})
    if (result.get("schema") != SCHEMA or run.get("status") != "completed" or run.get("manifestSha256") != manifest_sha or
            run.get("checkpointSha256") != model["checkpointSha256"] or run.get("modelVersion") != model["version"] or
            run.get("humanVerified") is not False or run.get("trainingAllowed") is not False or
            run.get("reviewedFrames") != len(manifest["frames"]) or run.get("totalFrames") != len(manifest["frames"]) or
            not isinstance(result.get("frames"), list) or len(result["frames"]) != len(manifest["frames"])):
        raise ValueError("Completed refresh identity/counts are invalid")
    for actual, frame in zip(result["frames"], manifest["frames"]):
        boxes = actual.get("boxes")
        if (actual.get("frameId") != frame["id"] or actual.get("imageSha256") != frame["sha256"] or
                actual.get("sourceId") != frame["sourceId"] or actual.get("model") != model or actual.get("status") != "completed" or
                not isinstance(boxes, list) or len(boxes) > 64 or any(not isinstance(box, dict) for box in boxes)):
            raise ValueError("Completed refresh frame identity is invalid")
        if public_boxes([{"bbox": b.get("bbox"), "score": b.get("score")} for b in boxes], frame, model["boxThreshold"]) != boxes:
            raise ValueError("Completed refresh labels or rights flags changed")


def refresh(collection, active_job, output, cache_dir=None, cancel_file=None, max_frames=1000, max_seconds=3600):
    if not isinstance(max_seconds, (int, float)) or isinstance(max_seconds, bool) or not 1 <= max_seconds <= 86400:
        raise ValueError("Refresh time budget must be 1..86400 seconds")
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir).resolve() if cache_dir else output.parent / "reviewed-suggestion-cache"
    state_path = output.with_suffix(output.suffix+".state.json")
    started = time.monotonic()
    cancelled = lambda: cancel_file is not None and Path(cancel_file).exists()
    if cancelled():
        return {"status": "cancelled", "outputWritten": False, "completedFrames": 0}
    with output_lock(output.with_suffix(output.suffix+".lock")):
        manifest, manifest_sha = load_public_collection(collection, max_frames)
        model, checkpoint = active_job_identity(active_job)
        context = {"schema": SCHEMA, "manifestSha256": manifest_sha, "model": model, "totalFrames": len(manifest["frames"])}
        context_sha = digest(canonical(context))
        had_state = state_path.exists()
        if had_state:
            state, _ = read_json(state_path)
            if state.get("contextSha256") != context_sha:
                raise ValueError("Refresh inputs changed; use a new output name")
        else:
            state = {"contextSha256": context_sha, "context": context, "status": "running", "completedFrames": 0}
        if output.exists():
            if not had_state or state.get("completedFrames") != len(manifest["frames"]):
                raise ValueError("Refusing to overwrite an existing unverified suggestions file")
            if state.get("status") == "completed" and state.get("outputSha256") != file_sha(output):
                raise ValueError("Completed suggestions bytes changed")
            result, _ = read_json(output)
            validate_completed_output(result, manifest, manifest_sha, model)
            # Recover a crash between final JSON replacement and the completion marker.
            state.update(status="completed", outputSha256=file_sha(output))
            atomic_json(state_path, state)
            return {"status": "completed", "outputWritten": True, "reusedCompletedOutput": True,
                    "completedFrames": len(result["frames"]), "outputSha256": state["outputSha256"]}
        atomic_json(state_path, state)
        cache_dir.mkdir(parents=True, exist_ok=True)
        detector, frames, cache_hits = None, [], 0
        for frame in manifest["frames"]:
            stop = "cancelled" if cancelled() else "paused_budget" if time.monotonic()-started >= max_seconds else None
            if stop:
                state.update(status=stop, completedFrames=len(frames), cacheHits=cache_hits)
                atomic_json(state_path, state)
                return {"status": stop, "outputWritten": False, "completedFrames": len(frames)}
            path = inside(Path(collection).resolve(), frame["path"])
            if file_sha(path) != frame["sha256"]:
                raise ValueError("Source changed after collection validation")
            key = cache_key(model, frame)
            cache_path = cache_dir / (key+".json")
            if cache_path.exists():
                boxes = cached_boxes(inside(cache_dir, cache_path.name), key, model, frame)
                cache_hits += 1
            else:
                if detector is None:
                    import torch
                    torch.set_num_threads(4)
                    detector, metadata = load_detector(checkpoint, device="cpu")
                    if metadata["scoreThreshold"] != model["boxThreshold"]:
                        raise ValueError("Loaded active threshold changed")
                boxes = public_boxes(predictions_for(detector, tensor_image(path), model["boxThreshold"]), frame, model["boxThreshold"])
                payload = {"cacheKey": key, "model": model, "imageSha256": frame["sha256"],
                           "width": frame["width"], "height": frame["height"], "boxes": boxes}
                atomic_json(cache_path, {"schema": "mycoscout.reviewed-suggestion-cache.v1", "payload": payload, "payloadSha256": digest(canonical(payload))})
            frames.append({"frameId": frame["id"], "imageSha256": frame["sha256"], "sourceId": frame["sourceId"],
                           "status": "completed", "model": model, "boxes": boxes})
            state.update(status="running", completedFrames=len(frames), cacheHits=cache_hits)
            atomic_json(state_path, state)
            print(json.dumps({"status": "progress", "phase": "refresh_suggestions", "completed": len(frames),
                              "total": len(manifest["frames"]), "cacheHits": cache_hits}), flush=True)
        _, final_sha = load_public_collection(collection, max_frames)
        final_model, _ = active_job_identity(active_job)
        if final_sha != manifest_sha or final_model != model:
            raise ValueError("Source manifest or active model changed during refresh")
        stop = "cancelled" if cancelled() else "paused_budget" if time.monotonic()-started >= max_seconds else None
        if stop:
            state.update(status=stop); atomic_json(state_path, state)
            return {"status": stop, "outputWritten": False, "completedFrames": len(frames)}
        result = {"schema": SCHEMA, "run": {"status": "completed", "reviewedFrames": len(frames), "totalFrames": len(frames),
                  "updatedAt": datetime.now(timezone.utc).isoformat(), "humanVerified": False, "trainingAllowed": False,
                  "checkpointSha256": model["checkpointSha256"], "modelVersion": model["version"], "manifestSha256": manifest_sha,
                  "cacheHits": cache_hits, "elapsedSeconds": time.monotonic()-started}, "frames": frames}
        validate_completed_output(result, manifest, manifest_sha, model)
        atomic_json(output, result)
        state.update(status="completed", outputSha256=file_sha(output), completedFrames=len(frames))
        atomic_json(state_path, state)
        return {"status": "completed", "outputWritten": True, "completedFrames": len(frames),
                "cacheHits": cache_hits, "outputSha256": state["outputSha256"], "checkpointSha256": model["checkpointSha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--active-job", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cancel-file", type=Path)
    parser.add_argument("--max-frames", type=int, default=1000)
    parser.add_argument("--max-seconds", type=float, default=3600)
    args = parser.parse_args()
    try:
        print(json.dumps(refresh(args.collection, args.active_job, args.output, args.cache_dir, args.cancel_file,
                                 args.max_frames, args.max_seconds)), flush=True)
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
