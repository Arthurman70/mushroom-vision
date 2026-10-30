"""Deliver saved native photos for queued live capture requests with sensor evidence.

An actual camera/ROS driver writes JPEG/PNG plus an atomic .capture.json sidecar.
This bridge does not invent pose, settled motion, or a mushroom bounding box.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from adapters.live_replay import contained, image_metadata, read_json
from adapters.review_stream import Station
from vision.live_mapping import number


class CaptureCompletionConflict(RuntimeError):
    """The photo may exist, but its exact queued capture was not completed."""


def verify_capture_completion(response, request, photo_sha256):
    """An HTTP success or saved file alone is not a task-completion acknowledgment."""
    if not isinstance(response, dict):
        raise CaptureCompletionConflict("Capture response was not a completion record")
    action, find, state = response.get("action"), response.get("find"), response.get("state")
    if not isinstance(action, dict) or action.get("type") != "completed" or action.get("taskId") != request["taskId"]:
        raise CaptureCompletionConflict("Photo delivery did not complete the exact requested task")
    if not isinstance(find, dict) or not find.get("id") or not isinstance(find.get("image"), dict) or find["image"].get("sha256") != photo_sha256:
        raise CaptureCompletionConflict("Completion did not confirm the exact saved photo")
    metadata = find.get("metadata", {})
    if not isinstance(metadata, dict) or find.get("sessionId") != request["sessionId"] or any(metadata.get(key) != request[key] for key in ("requestId", "taskId", "landmarkId", "mapRevision")):
        raise CaptureCompletionConflict("Saved photo belongs to another session, task or capture request")
    capture = action.get("capture", {})
    if not isinstance(capture, dict) or capture.get("photoSha256") != photo_sha256 or capture.get("photoRef") != find.get("imageUrl") or not find.get("imageUrl"):
        raise CaptureCompletionConflict("Task completion does not reference the saved native photo")
    for identity in (action, capture):
        for key in ("requestId", "sessionId", "landmarkId", "mapRevision"):
            if key in identity and identity[key] != request[key]:
                raise CaptureCompletionConflict("Completion acknowledgment refers to a different capture request")
    if isinstance(state, dict):
        if state.get("sessionId") != request["sessionId"]:
            raise CaptureCompletionConflict("Active session changed before capture completion")
        completed_task = next((task for task in state.get("tasks", []) if task.get("id") == request["taskId"]), None)
        if completed_task is not None:
            if completed_task.get("status") != "completed":
                raise CaptureCompletionConflict("Capture task is no longer completed")
            current_request = completed_task.get("captureRequest")
            if isinstance(current_request, dict) and current_request.get("requestId") != request["requestId"]:
                raise CaptureCompletionConflict("Capture request changed while the photo was being saved")
    return find["id"]


def photo_payload(request, image_path, evidence, *, min_sharpness, now_ms=None):
    """Validate external capture evidence and compute actual image sharpness."""
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    if evidence.get("schema") != "mycoscout.capture-evidence.v1" or evidence.get("synthetic") is not False or evidence.get("nativeUncropped") is not True:
        raise ValueError("Real uncropped camera capture evidence is required")
    for key in ("requestId", "taskId", "sessionId", "landmarkId", "mapRevision"):
        if evidence.get(key) != request.get(key) or not request.get(key):
            raise ValueError("Capture evidence does not match the pending request")
    stamp = number(evidence.get("capturedMs"), "capture time", request["requestedMs"], now_ms + 30)
    if now_ms - stamp > 8000:
        raise ValueError("Capture exceeds the request acknowledgment freshness window")
    if abs(number(evidence.get("poseTimestampMs"), "pose time", 0) - stamp) > 30:
        raise ValueError("Capture pose is not synchronized within 30 ms")
    position = evidence.get("positionEnuM")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError("Capture needs measured ENU vehicle-body position")
    position = [number(v, "vehicle position") for v in position]
    if math.dist(position, request["positionEnuM"]) > .18:
        raise ValueError("Vehicle left the requested capture position")
    camera_position = None
    if request.get("cameraPositionEnuM") is not None:
        camera_position = evidence.get("cameraPositionEnuM")
        if not isinstance(camera_position, list) or len(camera_position) != 3:
            raise ValueError("Capture needs the measured optical-camera position separately from vehicle origin")
        camera_position = [number(v, "camera position") for v in camera_position]
        if math.dist(camera_position, request["cameraPositionEnuM"]) > .18:
            raise ValueError("Optical camera left the requested capture position")
    yaw, pitch = (number(evidence.get(key), key, -100, 100) for key in ("yawRad", "pitchRad"))
    def angle(a, b): return abs((a - b + math.pi) % (2 * math.pi) - math.pi)
    if angle(yaw, request["aim"]["yawRad"]) > .12 or angle(pitch, request["aim"]["pitchRad"]) > .12:
        raise ValueError("Measured camera aim differs from requested target")
    samples = evidence.get("motionSamples")
    if not isinstance(samples, list) or not 2 <= len(samples) <= 1000:
        raise ValueError("Provide measured motion samples covering the settling interval")
    previous = None
    for sample in samples:
        timestamp = number(sample.get("timestampMs"), "motion sample time", stamp - 3000, stamp + 30)
        number(sample.get("speedMps"), "motion speed", 0, .08)
        if previous is not None and not 0 < timestamp - previous <= 150:
            raise ValueError("Settling telemetry must be ordered and have gaps no greater than 150 ms")
        previous = timestamp
    if stamp - samples[-1]["timestampMs"] > 30 or samples[-1]["timestampMs"] - samples[0]["timestampMs"] < 800:
        raise ValueError("Camera has not supplied 800 ms of settled telemetry through exposure time")
    meta, rgb = image_metadata(image_path, include_image=True, expected_hash=evidence.get("photoSha256"))
    if evidence.get("photoSha256") != meta["sha256"]:
        raise ValueError("Evidence must bind to the exact saved photo SHA256")
    with Image.open(image_path) as original:
        if original.getexif().get(274, 1) != 1:
            raise ValueError("Photo must use native oriented pixels; rotated EXIF needs an explicit camera transform")
    box = evidence.get("targetBboxPixelsXYXY")
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError("Supply a measured target box in this exact native photo")
    x1, y1, x2, y2 = [number(v, "target box", 0, max(rgb.size)) for v in box]
    if not 0 <= x1 < x2 <= rgb.width or not 0 <= y1 < y2 <= rgb.height or min(x2-x1, y2-y1) < 5:
        raise ValueError("Target must occupy a valid patch of at least 5 pixels per side")
    # Luminance Laplacian variance measures this original target patch; it is a
    # focus cue, not a mushroom-presence or identification confidence score.
    gray = np.asarray(rgb.convert("L"), dtype=float)[int(y1):int(y2), int(x1):int(x2)]
    laplacian = gray[1:-1, :-2] + gray[1:-1, 2:] + gray[:-2, 1:-1] + gray[2:, 1:-1] - 4 * gray[1:-1, 1:-1]
    sharpness = float(np.var(laplacian))
    threshold = number(min_sharpness, "minimum sharpness", .001, 1e9)
    if sharpness < threshold:
        raise ValueError(f"Measured target sharpness {sharpness:.3f} is below configured threshold {threshold:.3f}")
    ack = {**{key: request[key] for key in ("requestId", "taskId", "sessionId", "landmarkId", "mapRevision")},
        "capturedMs": stamp, "synthetic": False, "photoSha256": meta["sha256"],
        "photoRef": image_path.resolve().as_uri(), "nativeUncropped": True, "width": rgb.width, "height": rgb.height,
        "positionEnuM": position, "yawRad": yaw, "pitchRad": pitch, "settled": True, "sharpnessAccepted": True,
        "targetBboxPixelsXYXY": box, "qualityEvidence": {"metric": "native target luminance Laplacian variance", "value": sharpness, "threshold": threshold,
            "poseTimestampMs": evidence["poseTimestampMs"], "motionSampleCount": len(samples), "evidenceSource": evidence.get("sourceId")}}
    if camera_position is not None: ack["cameraPositionEnuM"] = camera_position
    return {"imageBase64": meta["imageBase64"], "mimeType": meta["mimeType"], "ack": ack, "rights": {"trainingOptIn": False}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="http://127.0.0.1:8787")
    parser.add_argument("--folder", type=Path, required=True, help="Camera-written native images and atomic *.capture.json evidence")
    parser.add_argument("--min-sharpness", type=float, required=True, help="Camera-specific validated target-patch focus threshold")
    parser.add_argument("--duration-seconds", type=float, default=60)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--token-env", default="MYCOSCOUT_TOKEN")
    args = parser.parse_args(argv)
    if not args.folder.is_dir() or not .2 <= args.interval <= 30 or not 1 <= args.duration_seconds <= 3600:
        parser.error("Use an existing folder, .2..30 second interval, and 1..3600 second duration")
    number(args.min_sharpness, "minimum sharpness", .001, 1e9)
    station = Station(args.station, token=os.environ.get(args.token_env))
    deadline, sent, rejected = time.monotonic() + args.duration_seconds, set(), set()
    while time.monotonic() < deadline:
        state = station.request("/api/live/state")
        if state.get("mode") == "live":
            pending = {task["captureRequest"]["requestId"]: task["captureRequest"] for task in state.get("tasks", []) if task.get("status") == "running" and task.get("captureRequest")}
            paths = sorted(args.folder.glob("*.capture.json"))
            if len(paths) > 1000: raise ValueError("Capture folder exceeds 1000 evidence sidecars; archive completed captures")
            for path in paths:
                evidence = read_json(path, limit=1024 * 1024)
                request = pending.get(evidence.get("requestId"))
                identity = (str(path), path.stat().st_mtime_ns)
                if request is None or request["requestId"] in sent or identity in rejected: continue
                try:
                    payload = photo_payload(request, contained(args.folder, evidence["imagePath"]), evidence, min_sharpness=args.min_sharpness)
                    response = station.request("/api/live/captures", payload)
                    saved_find_id = verify_capture_completion(response, request, payload["ack"]["photoSha256"])
                    sent.add(request["requestId"])
                    print(json.dumps({"requestId": request["requestId"], "savedFindId": saved_find_id, "savedPhotoSha256": payload["ack"]["photoSha256"], "action": response.get("action")}), flush=True)
                except CaptureCompletionConflict as error:
                    # Keep the request undelivered. Re-read the live queue before
                    # any retry; a saved image alone never marks the task done.
                    print(f"Capture completion conflict: {error}", file=sys.stderr)
                except (ValueError, OSError, RuntimeError) as error:
                    rejected.add(identity)
                    print(f"Capture evidence rejected: {error}", file=sys.stderr)
        time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    print(json.dumps({"deliveredCaptures": len(sent), "rejectedEvidenceFiles": len(rejected), "actuationEnabled": False}))


if __name__ == "__main__":
    try: main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Photo bridge failed: {error}", file=sys.stderr)
        raise SystemExit(1)
