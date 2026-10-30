"""Send candidate JPEG/PNG frames to the station's human-review inbox.

File/folder replay is implemented. A camera or detector can atomically rename completed
frames into the watched folder; this adapter does not pretend to be a device driver.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

MAX_IMAGE = 8 * 1024 * 1024


def station_origin(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
        raise ValueError("Use HTTPS, or HTTP on loopback only")
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("Use a plain station origin without credentials, path, query or fragment")
    return value.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Station redirects are not followed; configure its final HTTPS origin")


class Station:
    def __init__(self, origin, token=None):
        self.origin = station_origin(origin)
        self.token = token or os.environ.get("MYCOSCOUT_TOKEN", "")
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, path, data=None, binary=False):
        if not path.startswith("/api/") or "?" in path or "#" in path:
            raise ValueError("Only fixed API paths are accepted")
        headers = {"Accept": "application/json", "User-Agent": "MycoScout-review-bridge/0.4"}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        body = None
        if data is not None:
            body = json.dumps(data).encode("utf8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.origin + path, data=body, headers=headers)
        try:
            with self.opener.open(req, timeout=100) as response:
                payload = response.read(32 * 1024 * 1024 + 1)
                if len(payload) > 32 * 1024 * 1024:
                    raise ValueError("Station response exceeds limit")
                return payload if binary else json.loads(payload)
        except urllib.error.HTTPError as error:
            detail = error.read(2048).decode("utf8", errors="replace")
            raise RuntimeError(f"Station request failed ({error.code}): {detail}") from None


def read_frame(path):
    if path.stat().st_size > MAX_IMAGE:
        raise ValueError("Frame exceeds 8 MiB")
    raw = path.read_bytes()
    if raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    else:
        raise ValueError("Frame must be a JPEG or PNG")
    return raw, mime, hashlib.sha256(raw).hexdigest()


def send_frame(station, path, *, session_id, source="camera_frame", device_id="capture-folder", evaluation_only=False, metadata=None, classify=True):
    raw, mime, digest = read_frame(path)
    result = station.request("/api/finds", {"imageBase64": base64.b64encode(raw).decode("ascii"),
        "mimeType": mime, "source": source, "sessionId": session_id, "deviceId": device_id,
        "captureTimestamp": None, "evaluationOnly": evaluation_only,
        "rights": {"trainingOptIn": False}, "metadata": {"inputFile": path.name, "sha256": digest,
            "captureTimeKnown": False, **(metadata or {})}})
    find_id = result["find"]["id"]
    report = {"findId": find_id, "sha256": digest, "humanReviewRequired": True, "classified": False}
    if classify:
        try:
            prediction = station.request("/api/vision/classify", {"findId": find_id})
            report["classified"] = True
            report["prediction"] = prediction["result"]
        except (RuntimeError, ValueError, OSError) as error:
            report["classificationError"] = str(error)
            # Preserve the uploaded image even if the model is busy/unavailable.
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JPEG/PNG or a folder of completed candidate frames")
    parser.add_argument("--station", default="http://127.0.0.1:8787")
    parser.add_argument("--watch", action="store_true", help="Watch a folder; write frames elsewhere then atomically rename into it")
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--session-id", default=None, help="Keep the same session ID for all frames from one survey")
    parser.add_argument("--device-id", default="capture-folder")
    parser.add_argument("--source", default="camera_frame")
    parser.add_argument("--evaluation-only", action="store_true", help="Required for replay of prior evaluation/benchmark photos")
    parser.add_argument("--no-classify", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.max_frames <= 1000 or not 1 <= args.interval <= 3600:
        parser.error("Use 1–1000 frames and an interval of at least one second")
    if not args.input.exists() or args.watch and not args.input.is_dir():
        parser.error("Input must exist; watch requires a directory")
    station = Station(args.station)
    session_id = args.session_id or "capture-" + str(uuid.uuid4())
    seen = set()
    count = 0
    while count < args.max_frames:
        candidates = sorted(args.input.iterdir()) if args.input.is_dir() else [args.input]
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            try:
                _, _, digest = read_frame(path)
                if digest in seen:
                    continue
                report = send_frame(station, path, session_id=session_id, source=args.source,
                    device_id=args.device_id, evaluation_only=args.evaluation_only, classify=not args.no_classify)
                seen.add(digest)
                count += 1
                print(json.dumps(report), flush=True)
                if count >= args.max_frames:
                    break
                time.sleep(args.interval)
            except (ValueError, RuntimeError, OSError) as error:
                print(json.dumps({"file": path.name, "error": str(error)}), flush=True)
                if not args.watch:
                    raise SystemExit(1)
                time.sleep(args.interval)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
