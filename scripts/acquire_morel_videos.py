"""Acquire a fixed, bounded set of public CC-licensed morel field videos.

Downloads platform video renditions without cookies or authentication. Keeps
license evidence and checksums; does not create annotations or train a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

from yt_dlp import YoutubeDL


VIDEO_IDS = (
    "1Z9FgVpYris", "ng7VOJHum7U", "m9aGEh33w0I",
    "OhN1BQgK2L8", "go_1s6wVScI", "S_ruFMIM0qs",
    "xTJDZ3IkXbU",
)
MAX_FILE = 450 * 1024 ** 2
MAX_TOTAL = 2 * 1024 ** 3
LICENSE_HELP = "https://support.google.com/youtube/answer/2797468?hl=en"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def licensed(info, *, incomplete=False):
    if incomplete:
        return None
    if info.get("id") not in VIDEO_IDS:
        return "Video is not in the reviewed acquisition list"
    if info.get("availability") != "public":
        return "Only public footage is accepted"
    if info.get("license") != "Creative Commons Attribution license (reuse allowed)":
        return "The video's actual license is not the verified CC Attribution license"
    return None


def select_format(context):
    candidates = [item for item in context["formats"]
                  if item.get("ext") == "mp4"
                  and str(item.get("vcodec", "")).startswith("avc1")
                  and 0 < (item.get("height") or 0) <= 720
                  and item.get("protocol") == "https"
                  and isinstance(item.get("filesize"), (int, float))
                  and 0 < item["filesize"] <= MAX_FILE]
    if not candidates:
        raise ValueError("No bounded HTTPS H.264 rendition at 720p or lower is available")
    # Prefer video-only at the highest permitted resolution. Audio is unnecessary.
    chosen = max(candidates, key=lambda item: (
        item["height"], item.get("acodec") == "none", item.get("fps") or 0,
        item.get("tbr") or 0))
    yield chosen


def validate_existing(record, video_id, destination):
    """Never silently attribute unrelated or changed local bytes to a video."""
    expected = {"id": video_id, "sourceUrl": "https://www.youtube.com/watch?v=" + video_id,
                "path": "videos/" + video_id + ".mp4", "status": "downloaded",
                "licenseEvidence": "Creative Commons Attribution license (reuse allowed)"}
    if any(record.get(key) != value for key, value in expected.items()):
        raise ValueError("Existing download record has inconsistent source or license identity")
    if (not destination.is_file() or destination.is_symlink()
            or not 0 < destination.stat().st_size <= MAX_FILE
            or destination.stat().st_size != record.get("bytes")
            or sha256(destination) != record.get("sha256")):
        raise ValueError("A previously acquired video failed its checksum or size verification")


def acquire(output, ids):
    output.mkdir(parents=True, exist_ok=True)
    media = output / "videos"
    media.mkdir(exist_ok=True)
    records = []
    for video_id in ids:
        record_path = output / f"download-{video_id}.json"
        destination = media / f"{video_id}.mp4"
        if record_path.exists():
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("status") == "downloaded":
                validate_existing(record, video_id, destination)
                records.append(record)
                print(json.dumps({"id": video_id, "status": "verified-existing"}), flush=True)
                continue
        if destination.exists() or destination.with_suffix(".mp4.part").exists():
            raise ValueError("Unverified prior media exists; retain it separately and use a fresh acquisition directory")
        current_bytes = sum(path.stat().st_size for path in media.glob("*.*") if path.is_file())
        if current_bytes + MAX_FILE > MAX_TOTAL:
            raise ValueError("Acquisition would exceed the 2 GiB media budget")
        last_progress = [0.0]

        def progress(event):
            downloaded = event.get("downloaded_bytes") or 0
            if downloaded > MAX_FILE:
                raise ValueError("Download exceeded the per-video limit")
            now = time.monotonic()
            if now - last_progress[0] >= 20 or event.get("status") == "finished":
                print(json.dumps({"id": video_id, "status": event.get("status"),
                                  "downloadedBytes": downloaded,
                                  "totalBytes": event.get("total_bytes")}), flush=True)
                last_progress[0] = now

        options = {"quiet": True, "no_warnings": True, "noprogress": True,
                   "format": select_format, "match_filter": licensed,
                   "outtmpl": str(media / "%(id)s.%(ext)s"), "noplaylist": True,
                   "socket_timeout": 30, "retries": 0, "extractor_retries": 0,
                   "fragment_retries": 0, "max_filesize": MAX_FILE,
                   "overwrites": False, "continuedl": True, "progress_hooks": [progress]}
        url = "https://www.youtube.com/watch?v=" + video_id
        try:
            with YoutubeDL(options) as downloader:
                info = downloader.extract_info(url, download=True)
            if not info or licensed(info) or not destination.is_file():
                raise ValueError("No licensed video file was downloaded")
            size = destination.stat().st_size
            if not 0 < size <= MAX_FILE:
                raise ValueError("Acquired media is outside the expected size limit")
            record = {"id": video_id, "status": "downloaded", "sourceUrl": url,
                      "retrievedAt": datetime.now(timezone.utc).isoformat(),
                      "title": info["title"], "creator": info.get("uploader"),
                      "channelId": info.get("channel_id"), "channelUrl": info.get("channel_url"),
                      "uploadDate": info.get("upload_date"), "licenseEvidence": info["license"],
                      "license": "CC BY (YouTube)", "licenseUrl": LICENSE_HELP,
                      "licenseNote": "Exact video license from public platform metadata; version not asserted.",
                      "description": info.get("description"), "durationSeconds": info.get("duration"),
                      "formatId": info.get("format_id"), "width": info.get("width"),
                      "height": info.get("height"), "fps": info.get("fps"),
                      "codec": info.get("vcodec"), "path": "videos/" + destination.name,
                      "bytes": size, "sha256": sha256(destination),
                      "rendition": "Platform rendition; not original camera footage",
                      "labelsVerified": False}
        except Exception as error:
            # Omit expiring signed media URLs from persistent download reports.
            message = re.sub(r"https?://\S+", "[URL omitted]", str(error))[:600]
            record = {"id": video_id, "sourceUrl": url, "status": "failed", "error": message}
        save_json(record_path, record)
        records.append(record)
        print(json.dumps({key: record.get(key) for key in ("id", "status", "bytes", "error")}), flush=True)
    save_json(output / "download-summary.json", {"videos": records,
                                                "downloaded": sum(r["status"] == "downloaded" for r in records)})
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/morel-video-acquisition-v1"))
    parser.add_argument("--ids", nargs="+", choices=VIDEO_IDS, default=list(VIDEO_IDS))
    args = parser.parse_args()
    acquire(args.output.resolve(), args.ids)
