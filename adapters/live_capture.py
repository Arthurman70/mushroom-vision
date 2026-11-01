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
import threading
import time
import uuid

import numpy as np
from PIL import Image

from adapters.live_replay import contained, read_json
from adapters.review_stream import Station
from adapters.scout_frame import LocalDetector
from vision.live_mapping import LiveMapper
from vision.hybrid_scan import HybridScanScheduler
from vision.testbench import (FULL_TILE_REGION, GROUND_TILE_REGION, bounded_nms,
                             projected_candidates, region_tile_boxes, settings_for)

VISION_PROFILES = ("hybrid", "ground-tiles", "full-tiles", "fast")


def capture_scan_settings(profile="ground-tiles", tile_region=None):
    if profile not in VISION_PROFILES:
        raise ValueError("Choose hybrid, ground-tiles, full-tiles or fast vision profile")
    settings = {"methods": ["original"] if profile == "fast" else ["tiles"],
                "tileGrid": 2, "scale": 1, "boxThreshold": .25}
    if profile == "fast":
        if tile_region is not None:
            raise ValueError("A tile region requires a tiled vision profile")
    else:
        settings["tileRegion"] = list(GROUND_TILE_REGION if profile in ("ground-tiles", "hybrid") else FULL_TILE_REGION) if tile_region is None else tile_region
    return settings_for(settings)


def scan_frame(image, detector, *, profile="ground-tiles", tile_region=None, color_sensitive=False,
               scheduler=None, timestamp_seconds=None):
    """Apply image views without changing the detector's approved thresholds.

    The selected rectangle is only a pixel region. It makes no semantic ground,
    obstacle, range or flight-clearance claim. The original view always runs.
    """
    # The direct still-frame default remains ground tiles; only a persistent live
    # scheduler can provide hybrid cadence. Invalid inputs never consume a slot.
    capture_scan_settings(profile, tile_region)
    requested_profile, plan = profile, None
    if profile == "hybrid":
        if not isinstance(scheduler, HybridScanScheduler):
            raise ValueError("Hybrid scans require a persistent HybridScanScheduler")
        plan = scheduler.plan(timestamp_seconds)
        profile = plan.profile
        # A lower-image override applies to ground detail only. The periodic
        # whole-image pass must still inspect trunks and logs above that crop.
        effective_region = tile_region if profile == "ground-tiles" else None
    else:
        if scheduler is not None or timestamp_seconds is not None:
            raise ValueError("A scheduler and timestamp are only used by hybrid scans")
        effective_region = tile_region
    settings = capture_scan_settings(profile, effective_region)
    if type(color_sensitive) is not bool:
        raise ValueError("color_sensitive must be boolean")
    if color_sensitive:
        settings["methods"] = ["original", "color"] if profile == "fast" else ["color", "tiles"]
    if image.mode != "RGB" or min(image.size) < 16 or image.width * image.height > 8000000:
        raise ValueError("Provide an RGB frame of at least 16 pixels per side and at most 8 megapixels")
    threshold = getattr(detector, "box_threshold", getattr(detector, "metadata", {}).get("scoreThreshold", .25))
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Detector has an invalid score threshold")
    # Unlike still-photo experiments, this adapter must not reset an activated
    # detector's threshold to the baseline UI default (including valid extremes).
    settings["boxThreshold"] = float(threshold)
    started, timings, candidates = time.perf_counter(), [], []
    full_bounds = (0, 0, image.width, image.height)

    def forward(view, bounds, kind, name):
        before = time.perf_counter()
        predicted = projected_candidates(detector.predict(view), bounds, image.size, image.size, threshold, name)
        seconds = time.perf_counter() - before
        timings.append({"kind": kind, "bboxPixelsXYXY": list(bounds), "seconds": seconds})
        return predicted

    candidates.extend(forward(image, full_bounds, "context", "original"))
    context_count = len(candidates)
    boxes = region_tile_boxes(image.width, image.height, 2, settings["tileRegion"]) if profile != "fast" else []
    for index, bounds in enumerate(boxes):
        candidates.extend(forward(image.crop(bounds), bounds, "tile", f"tile_{index + 1}"))
    methods = [{"id": "tiles" if boxes else "original", "passes": len(timings),
                "seconds": sum(row["seconds"] for row in timings), "preNmsCandidateCount": len(candidates),
                "contextCandidateCount": context_count, "fullFrameContext": True,
                "tileRegion": settings.get("tileRegion"), "tileBoundsPixelsXYXY": [list(bounds) for bounds in boxes],
                "overlapFraction": .2 if boxes else 0., "passTimings": list(timings)}]
    if color_sensitive:
        from vision.far_color import enhance
        colors = forward(enhance(image), full_bounds, "color", "color")
        candidates.extend(colors)
        methods.append({"id": "color", "passes": 1, "seconds": timings[-1]["seconds"],
                        "preNmsCandidateCount": len(colors), "passTimings": [timings[-1]]})
    selected = bounded_nms(candidates, "live")
    seconds = time.perf_counter() - started
    cadence = scheduler.commit(plan, seconds) if plan is not None else None
    return selected, {"profileId": requested_profile, "effectiveProfileId": profile,
                      "cadence": cadence, "settings": settings, "modelPasses": len(timings),
                      "methods": methods, "seconds": seconds,
                      "fullFrameContext": True, "tileRegion": settings.get("tileRegion"),
                      "tileBoundsPixelsXYXY": [list(bounds) for bounds in boxes], "passTimings": timings,
                      "regionMeaning": "Selected image rectangle; not semantic ground or safe terrain",
                      "preNmsCandidateCount": len(candidates), "candidateCount": len(selected), "nmsIou": .5}


class LatestCameraFrames:
    """Continuously drain camera reads into one replaceable frame slot.

    Receipt time is a host observation, not a sensor exposure timestamp. Driver
    buffering can still add unknown delay; CAP_PROP_BUFFERSIZE is only a request.
    """

    def __init__(self, capture):
        self.capture = capture
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._latest = None
        self._error = None
        self._done = False
        self._thread = threading.Thread(target=self._run, name="camera-latest-frame", daemon=True)
        self._thread.start()

    def _run(self):
        index = -1
        try:
            while not self._stop.is_set():
                success, frame = self.capture.read()
                if not success:
                    break
                receipt, monotonic_receipt = time.time(), time.monotonic()
                if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] * frame.shape[1] > 8000000:
                    raise ValueError("Decoded camera frame exceeds supported color geometry")
                index += 1
                with self._condition:
                    self._latest = (index, frame, receipt, monotonic_receipt)
                    self._condition.notify_all()
        except Exception as error:
            with self._condition:
                self._error = error
        finally:
            with self._condition:
                self._done = True
                self._condition.notify_all()

    @property
    def done(self):
        with self._condition:
            return self._done

    def next_frame(self, after_index, timeout=1.):
        if type(after_index) is not int or after_index < -1:
            raise ValueError("Camera frame index must be an integer at least -1")
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 <= timeout <= 5:
            raise ValueError("Camera wait timeout must be finite in 0..5 seconds")
        with self._condition:
            self._condition.wait_for(lambda: self._error is not None or self._done or
                (self._latest is not None and self._latest[0] > after_index), timeout)
            if self._error is not None:
                raise RuntimeError("Camera reader failed") from self._error
            return self._latest if self._latest is not None and self._latest[0] > after_index else None

    def close(self):
        self._stop.set()
        # Releasing is needed to unblock backends waiting for a camera frame.
        self.capture.release()
        self._thread.join(timeout=1.)
        return not self._thread.is_alive()


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
        self.box_threshold = threshold
        self.info = {"model": "reviewed-mushroom-detector", "resolvedRevision": "sha256:" + digest,
                     "jobId": pointer.get("jobId"), "boxThreshold": threshold,
                     "scoreMeaning": "Detection score, not a species probability"}

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


def argument_parser():
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
    parser.add_argument("--vision-profile", choices=VISION_PROFILES, default="hybrid",
                        help="hybrid (default): fast scans between periodic ground/whole-image detail; ground-tiles/full-tiles: every-frame detail; fast: context only")
    parser.add_argument("--detail-interval", type=float, default=5.,
                        help="Hybrid minimum source-time spacing in seconds (0.5..3600); actual scans can take longer")
    parser.add_argument("--full-detail-every", type=int, default=3,
                        help="Inspect the whole image every N detail sweeps (1..100), including trunks and logs")
    parser.add_argument("--min-fast-frames", type=int, default=2,
                        help="Require 2..1000 distinct fast-frame scans between detail sweeps")
    parser.add_argument("--tile-region", type=float, nargs=4, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Override tiled image rectangle in normalized 0..1 coordinates; not inferred ground")
    parser.add_argument("--classify-groups", action="store_true", help="Add group guesses for up to five candidates with at least 32 native pixels on both sides")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-seconds", type=float, default=1800)
    parser.add_argument("--sidecar", type=Path, help="Synchronized calibrated depth+pose sidecar for a recorded video")
    parser.add_argument("--token-env", default="MYCOSCOUT_TOKEN")
    return parser


def main(argv=None):
    parser = argument_parser()
    args = parser.parse_args(argv)
    if not (args.station or args.emit or args.output):
        parser.error("Choose --station, --emit or --output")
    if not 1 <= args.sample_every <= 1000 or not 1 <= args.max_frames <= 1000 or not 1 <= args.max_seconds <= 86400:
        parser.error("Use bounded sample-every 1..1000, max-frames 1..1000 and max-seconds 1..86400")
    try:
        capture_scan_settings(args.vision_profile, args.tile_region)
        scheduler = HybridScanScheduler(args.detail_interval, args.full_detail_every, args.min_fast_frames)
    except ValueError as error:
        parser.error(str(error))
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
    last_packet_index = -1
    camera_reader = LatestCameraFrames(capture) if is_camera else None
    camera_reader_stopped = None
    try:
        while emitted < args.max_frames and time.monotonic() - started < args.max_seconds:
            receipt, monotonic_receipt = None, None
            if camera_reader:
                received = camera_reader.next_frame(read_index, timeout=min(1., max(0., args.max_seconds - (time.monotonic() - started))))
                if received is None:
                    if camera_reader.done:
                        break
                    continue
                read_index, bgr, receipt, monotonic_receipt = received
                if emitted and read_index - last_packet_index < args.sample_every:
                    continue
            else:
                success, bgr = capture.read()
                if not success:
                    break
                read_index += 1
                if read_index % args.sample_every:
                    continue
            if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.shape[0] * bgr.shape[1] > 8000000:
                raise ValueError("Decoded color frame exceeds supported geometry")
            # USB timestamps describe host receipt, not exposure; do not use them as
            # synchronized sensor evidence. Sidecars carry real capture timestamps.
            timestamp = receipt if is_camera else read_index / fps
            sensor = sideframes.get(read_index)
            if sensor:
                timestamp = sensor["timestampSeconds"]
            rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inference_start = time.monotonic()
            detections, scan = scan_frame(rgb, detector, profile=args.vision_profile, tile_region=args.tile_region,
                                          color_sensitive=args.color_sensitive,
                                          scheduler=scheduler if args.vision_profile == "hybrid" else None,
                                          timestamp_seconds=timestamp if args.vision_profile == "hybrid" else None) if detector else ([], None)
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
                "detector": {**detector.info, "inferenceSeconds": time.monotonic() - inference_start,
                    "colorSensitive": args.color_sensitive, "classifyGroups": args.classify_groups, "scan": scan} if detector else None,
                "provenance": {"source": "usb-camera" if is_camera else "local-video", "sourceFrameIndex": read_index,
                    "captureTimeKnown": bool(sensor), "timestampOrigin": "sensor-sidecar" if sensor else "host-receipt" if is_camera else "frame-index/fps",
                    "cameraDelivery": {"policy": "latest-host-received-frame", "bufferSlots": 1,
                        "hostReceivedTimestampSeconds": receipt,
                        "hostReceiptAgeAtInferenceStartSeconds": max(0., inference_start - monotonic_receipt),
                        "hostReceiptAgeAtPacketSeconds": max(0., time.monotonic() - monotonic_receipt),
                        "sourceFramesSkippedSincePreviousPacket": max(0, read_index - last_packet_index - 1),
                        "sensorExposureTimeKnown": False, "driverBufferDelayKnown": False} if is_camera else None,
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
            last_packet_index = read_index
    finally:
        if camera_reader:
            camera_reader_stopped = camera_reader.close()
        else:
            capture.release()
        if archive:
            archive.close()
    print(json.dumps({"frames": emitted, "inputWasCamera": is_camera, "elapsedSeconds": time.monotonic() - started,
                      "cameraReaderStopped": camera_reader_stopped,
                      "actuationEnabled": False, "note": "Capture/inference loop is not an obstacle-avoidance controller"}), file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Live capture failed: {error}", file=sys.stderr)
        raise SystemExit(1)
