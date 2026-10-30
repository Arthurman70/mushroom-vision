"""Outbound-only optional vision worker. No flight commands, listener or model downloads."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
import math
import os
import time
from pathlib import Path

from adapters.review_stream import Station


def decode_job(job):
    from PIL import Image
    if job.get("kind") not in {"image-quality-v1", "fungi-candidates-v1"}:
        raise ValueError("Unsupported task")
    encoded = job["image"]["imageBase64"]
    if not isinstance(encoded, str) or len(encoded) > 5_592_408:
        raise ValueError("Image exceeds 4 MiB")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) > 4 * 1024 * 1024 or hashlib.sha256(raw).hexdigest() != job["imageSha256"]:
        raise ValueError("Image size or hash does not match")
    with Image.open(io.BytesIO(raw)) as opened:
        if opened.format not in {"PNG", "JPEG"} or getattr(opened, "n_frames", 1) != 1:
            raise ValueError("A still PNG or JPEG is required")
        if opened.width * opened.height > 8_000_000 or [opened.width, opened.height] != [job["width"], job["height"]]:
            raise ValueError("Image dimensions do not match")
        # Protocol pixels are stored raster orientation, never inferred GPS or pose.
        return opened.convert("RGB")


def quality_metrics(image):
    import numpy as np
    from PIL import Image
    start = time.perf_counter()
    width, height = image.size
    scale = min(1, 640 / max(width, height))
    sample = image.resize((max(1, math.floor(width * scale + .5)), max(1, math.floor(height * scale + .5))), Image.Resampling.BILINEAR)
    rgb = np.asarray(sample, dtype=np.float64)
    y = rgb @ np.array([.2126, .7152, .0722])
    lap = y[:-2, 1:-1] + y[2:, 1:-1] + y[1:-1, :-2] + y[1:-1, 2:] - 4 * y[1:-1, 1:-1]
    color_distance = np.sqrt(np.mean((rgb - rgb.mean(axis=(0, 1))) ** 2, axis=2))
    return {"algorithm": "rgba-quality-v1", "width": width, "height": height,
            "sampleWidth": sample.width, "sampleHeight": sample.height,
            "meanLuminance": float(y.mean()), "luminanceStd": float(y.std()),
            "laplacianVariance": float(lap.var()) if lap.size else 0.,
            "meanChroma": float((rgb.max(axis=2) - rgb.min(axis=2)).mean()),
            "colorOutlierFraction": float((color_distance > 40).mean()),
            "processingMs": (time.perf_counter() - start) * 1000}


def execute_job(job, detector=None):
    start = time.perf_counter()
    image = decode_job(job)
    if job["kind"] == "image-quality-v1":
        result = quality_metrics(image)
    elif detector is not None:
        candidates = detector.predict(image)[:100]
        result = {"model": "GroundingDINO / " + detector.info["resolvedRevision"], "candidates": [
            {"bboxPixelsXYXY": c["bboxPixelsXYXY"], "confidence": c["confidence"], "modelLabel": c["modelLabel"]}
            for c in candidates]}
    else:
        raise ValueError("Detector capability is not loaded")
    result["processingMs"] = (time.perf_counter() - start) * 1000
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="http://127.0.0.1:8787", help="HTTPS origin or loopback HTTP; no redirects")
    parser.add_argument("--token-env", default="MYCOSCOUT_DEVICE_TOKEN", help="Environment variable containing a paired worker token, never the station owner token")
    parser.add_argument("--detector-model", type=Path, help="Optional existing local GroundingDINO model directory")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--once", action="store_true", help="Claim and execute at most one job")
    args = parser.parse_args(argv)
    token = os.environ.get(args.token_env, "")
    if len(token) < 32:
        parser.error("Set the paired device token in the named environment variable")
    detector = None
    capabilities = ["image-quality-v1"]
    if args.detector_model:
        from adapters.scout_frame import LocalDetector
        detector = LocalDetector(args.detector_model, device=args.device, threads=args.threads)
        capabilities.append("fungi-candidates-v1")
    station = Station(args.station, token)
    path = "/api/device-compute/worker"
    station.request(path, {"op": "announce", "capabilities": capabilities, "runtime": "python-reference-worker-v1"})
    print(json.dumps({"state": "ready", "capabilities": capabilities, "flightAuthority": False}), flush=True)
    while True:
        message = station.request(path, {"op": "claim"})
        job = message.get("job")
        if job:
            start = time.monotonic()
            try:
                result = execute_job(job, detector)
                if (time.monotonic() - start) * 1000 >= job["remainingMs"]:
                    print(json.dumps({"jobId": job["id"], "state": "late_result_discarded"}), flush=True)
                else:
                    station.request(path, {"op": "result", "jobId": job["id"], "leaseToken": job["leaseToken"],
                        "sessionId": job["sessionId"], "frameId": job["frameId"], "imageSha256": job["imageSha256"], "result": result})
                    print(json.dumps({"jobId": job["id"], "state": "completed", "unverified": True}), flush=True)
            except (ValueError, RuntimeError, OSError) as error:
                # Failed jobs expire; never retry under a new frame identity or upload a guessed result.
                print(json.dumps({"jobId": job["id"], "state": "failed", "error": str(error)[:240]}), flush=True)
        if args.once:
            return 0
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
