"""Read actual OpenCV video/camera frames, detect, track, and send review packets.

RGB cameras provide unlocated candidates. Metric reconstruction requires an
explicit calibrated depth/pose sidecar from the same recorded sensor capture.
No camera, GNSS, or aircraft connection is implied by merely running this CLI.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PIL import Image

from adapters.live_replay import contained, read_json
from adapters.review_stream import Station
from adapters.scout_frame import LocalDetector
from vision.live_mapping import LiveMapper


class ReviewedDetector:
    def __init__(self, pointer_path, device="cpu"):
        import torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        from vision.training_pipeline import load_detector
        pointer = read_json(pointer_path)
        checkpoint = Path(pointer["checkpoint"])
        if not checkpoint.is_file() or checkpoint.stat().st_size > 2 * 1024 ** 3:
            raise ValueError("Activated checkpoint is missing or too large")
        with checkpoint.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != pointer.get("checkpointSha256"):
            raise ValueError("Activated model checksum mismatch")
        self.model, self.metadata = load_detector(checkpoint, device=device)
        threshold = self.metadata.get("scoreThreshold", .3)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Activated detector has an invalid score threshold")
        self.info = {"model": "reviewed-mushroom-detector", "resolvedRevision": "sha256:" + digest,
                     "jobId": pointer.get("jobId"), "scoreMeaning": "Detection score, not a species probability"}

    def predict(self, image):
        import torch
        tensor = torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1).float() / 255
        device = next(self.model.parameters()).device
        with torch.inference_mode():
            result = self.model([tensor.to(device)])[0]
        candidates = []
        for box, score, label in zip(result["boxes"].cpu().tolist(), result["scores"].cpu().tolist(), result["labels"].cpu().tolist()):
            if score >= self.metadata.get("scoreThreshold", .3) and label == 1:
                candidates.append({"bboxPixelsXYXY": box, "confidence": score})
        return candidates[:100]


def sidecar_frames(path):
    if path is None:
        return {}, None
    data = read_json(path)
    if data.get("schema") != "mycoscout.rgbd-sidecar.v1" or data.get("coordinateFrame") != "map-enu":
        raise ValueError("Sidecar must declare mycoscout.rgbd-sidecar.v1 and map-enu coordinates")
    if type(data.get("simulationOnly")) is not bool or len(data.get("frames", [])) > 100000:
        raise ValueError("Sidecar needs explicit simulation flag and bounded frames")
    frames = {}
    for frame in data["frames"]:
        index = frame.get("sourceFrameIndex")
        if type(index) is not int or index < 0 or index in frames:
            raise ValueError("Sidecar frame indices must be unique nonnegative integers")
        frames[index] = frame
    return frames, data


def load_depth(path):
    if path.stat().st_size > 64 * 1024 ** 2:
        raise ValueError("Depth file exceeds 64 MiB")
    if path.suffix.lower() == ".npy":
        values = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".png":
        with Image.open(path) as image:
            if image.width * image.height > 8000000:
                raise ValueError("Depth image exceeds 8 megapixels")
            values = np.array(image)
    else:
        raise ValueError("Depth file must be a numeric NPY array or 16-bit PNG")
    if values.ndim != 2 or values.size > 8000000:
        raise ValueError("Depth must be a bounded 2D array")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Local video file or numeric USB camera index")
    parser.add_argument("--station")
    parser.add_argument("--emit", action="store_true")
    parser.add_argument("--output", type=Path, help="New directory for native-size JPEG frames and packets.jsonl")
    parser.add_argument("--session-id", default="capture-" + str(uuid.uuid4()))
    parser.add_argument("--model", type=Path, default=Path("models/grounding-dino-tiny"))
    parser.add_argument("--active-model", type=Path, help="Explicit approved model pointer instead of baseline detector")
    parser.add_argument("--no-detector", action="store_true", help="View/map only; never emit pretend detections")
    parser.add_argument("--color-sensitive", action="store_true", help="Add an experimental color-enhanced detector pass")
    parser.add_argument("--classify-groups", action="store_true", help="Add group guesses for up to five candidates with at least 32 native pixels on both sides")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--sidecar", type=Path, help="Synchronized calibrated depth+pose sidecar for a recorded video")
    parser.add_argument("--token-env", default="MYCOSCOUT_TOKEN")
    args = parser.parse_args(argv)
    if not (args.station or args.emit or args.output):
        parser.error("Choose --station, --emit or --output")
    if not 1 <= args.sample_every <= 1000 or not 1 <= args.max_frames <= 1000 or not 1 <= args.max_seconds <= 86400:
        parser.error("Use bounded sample-every 1..1000, max-frames 1..1000 and max-seconds 1..86400")
    is_camera = args.input.isdecimal()
    if not is_camera and not Path(args.input).is_file():
        parser.error("Input must be a local video file or numeric camera index")
    if is_camera and args.sidecar:
        parser.error("Pre-recorded sidecars cannot be attached to a live camera clock")
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install optional opencv-python-headless==4.10.0.84 in the vision environment") from error
    mapper = LiveMapper(args.session_id)
    sideframes, sidedata = sidecar_frames(args.sidecar)
    if args.sidecar and sidedata.get("videoSha256"):
        with Path(args.input).open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sidedata["videoSha256"]:
                raise ValueError("Sidecar belongs to a different video")
    elif args.sidecar:
        raise ValueError("RGBD sidecar must bind to the exact videoSha256")
    detector = None if args.no_detector else ReviewedDetector(args.active_model, args.device) if args.active_model else LocalDetector(args.model, device=args.device)
    group_predictor = None
    if args.classify_groups:
        from vision.live_groups import load_group_predictor
        group_predictor = load_group_predictor()
    station = Station(args.station, token=os.environ.get(args.token_env)) if args.station else None
    if station:
        station.request("/api/live/session", {"sessionId": args.session_id, "mode": "live" if is_camera else "replay"})
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
    archive = (args.output / "packets.jsonl").open("x", encoding="utf8") if args.output else None
    capture = cv2.VideoCapture(int(args.input) if is_camera else args.input)
    if not capture.isOpened():
        raise RuntimeError("Camera/video could not be opened")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not is_camera and (not math.isfinite(fps) or fps <= 0):
        raise ValueError("Video has no trustworthy frame rate; use a timestamped extracted collection")
    started, read_index, emitted = time.monotonic(), -1, 0
    try:
        while emitted < args.max_frames and time.monotonic() - started < args.max_seconds:
            success, bgr = capture.read()
            if not success:
                break
            read_index += 1
            if read_index % args.sample_every:
                continue
            if bgr.ndim != 3 or bgr.shape[0] * bgr.shape[1] > 8000000:
                raise ValueError("Decoded color frame exceeds supported geometry")
            # USB timestamps describe host receipt, not exposure; do not use them as
            # synchronized sensor evidence. Sidecars carry real capture timestamps.
            timestamp = time.time() if is_camera else read_index / fps
            sensor = sideframes.get(read_index)
            if sensor:
                timestamp = sensor["timestampSeconds"]
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inference_start = time.monotonic()
            detections = detector.predict(rgb) if detector else []
            if args.color_sensitive and detector:
                from vision.far_color import enhance
                from vision.metrics import class_agnostic_nms
                detections = class_agnostic_nms(detections + detector.predict(enhance(rgb)), .5)[:100]
            if group_predictor is not None:
                from vision.live_groups import classify_candidates
                detections = classify_candidates(rgb, detections[:100], group_predictor)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok or len(encoded) > 8 * 1024 ** 2:
                raise ValueError("Could not encode a bounded native-size JPEG")
            raw = encoded.tobytes()
            frame_id = f"frame-{read_index:08d}"
            frame = {"frameId": frame_id, "sequence": emitted, "timestampSeconds": timestamp,
                "simulationOnly": not is_camera,
                "mapRevision": sidedata["mapRevision"] if sidedata else "rgb-unlocated-v1",
                "image": {"width": rgb.width, "height": rgb.height, "mimeType": "image/jpeg", "sha256": hashlib.sha256(raw).hexdigest(), "imageBase64": base64.b64encode(raw).decode("ascii")},
                "detections": detections[:100], "detectorStatus": "completed" if detector else "unprocessed",
                "detector": {**detector.info, "inferenceSeconds": time.monotonic() - inference_start, "colorSensitive": args.color_sensitive, "classifyGroups": args.classify_groups} if detector else None,
                "provenance": {"source": "usb-camera" if is_camera else "local-video", "sourceFrameIndex": read_index,
                    "captureTimeKnown": bool(sensor), "timestampOrigin": "sensor-sidecar" if sensor else "host-receipt" if is_camera else "frame-index/fps",
                    "trainingAllowed": False, "replay": not is_camera,
                    "synthetic": sidedata["simulationOnly"] if sidedata else False}}
            depth = None
            if sensor:
                for field in ("camera", "depth", "georeference"):
                    if field in sensor:
                        frame[field] = sensor[field]
                depth = load_depth(contained(args.sidecar.parent, sensor["depthPath"]))
            packet = mapper.process(frame, depth=depth, rgb=rgb)
            line = json.dumps(packet, allow_nan=False, separators=(",", ":"))
            if args.output:
                (args.output / f"{frame_id}.jpg").write_bytes(raw)
                archive.write(line + "\n"); archive.flush()
            if station:
                station.request("/api/live/frames", packet)
            if args.emit:
                print(line, flush=True)
            emitted += 1
    finally:
        capture.release()
        if archive:
            archive.close()
    print(json.dumps({"frames": emitted, "inputWasCamera": is_camera, "elapsedSeconds": time.monotonic() - started,
                      "actuationEnabled": False, "note": "Capture/inference loop is not an obstacle-avoidance controller"}), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Live capture failed: {error}", file=sys.stderr)
        raise SystemExit(1)
