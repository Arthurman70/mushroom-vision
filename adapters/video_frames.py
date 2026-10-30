"""Extract real, unlabelled video frames into a bounded human-review collection.

Only explicit local files are decoded. No downloading, synthetic degradation,
object labels, distance estimates, train/test split, or model training occurs.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

import numpy as np
from PIL import Image, ImageDraw, ImageOps

MAX_SOURCES = 12
MAX_VIDEO_BYTES = 2 * 1024 ** 3
MAX_TOTAL_VIDEO_BYTES = 8 * 1024 ** 3
MAX_PIXELS = 20_000_000
MAX_FRAMES = 120
MAX_CANDIDATES_PER_VIDEO = 480
MAX_OUTPUT_BYTES = 1024 ** 3
ID_RE = re.compile(r"[A-Za-z0-9_-]{1,120}\Z")
GROUP_RE = re.compile(r"[A-Za-z0-9:_-]{1,160}\Z")
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".ogv", ".ogg", ".m4v", ".flv", ".wmv"}
BLUR_METRIC = "whole-frame luminance Laplacian variance; relative within source video"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def bounded_number(value, name, minimum, maximum, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def public_url(value, name):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError(f"Invalid {name}")
    parsed = urlsplit(value)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{name} must be a public attribution URL")
    return value


def read_sources(manifest_path):
    manifest_path = Path(manifest_path).resolve(strict=True)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("Sources manifest must be a local JSON file under 1 MiB")
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema") != "mycoscout.video-sources.v1":
        raise ValueError("Expected mycoscout.video-sources.v1 manifest")
    sources = payload.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= MAX_SOURCES:
        raise ValueError(f"Supply 1–{MAX_SOURCES} explicit local video sources")
    result, ids, total = [], set(), 0
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not ID_RE.fullmatch(source["id"]):
            raise ValueError("Source IDs must contain only letters, digits, hyphens and underscores")
        if source["id"] in ids:
            raise ValueError("Repeated source ID")
        ids.add(source["id"])
        raw_path = source.get("path")
        if (not isinstance(raw_path, str) or not raw_path or len(raw_path) > 2048 or
                "://" in raw_path or raw_path.startswith(("\\\\", "//")) or "\x00" in raw_path):
            raise ValueError("Video path must identify a local regular file")
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = manifest_path.parent / candidate
        if candidate.is_symlink():
            raise ValueError("Video symlinks are not accepted")
        path = candidate.resolve(strict=True)
        if str(path).startswith(("\\\\", "//")) or not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError("Unsupported local video file")
        size = path.stat().st_size
        if not 1 <= size <= MAX_VIDEO_BYTES:
            raise ValueError("Video is empty or exceeds the 2 GiB source limit")
        total += size
        if total > MAX_TOTAL_VIDEO_BYTES:
            raise ValueError("Sources exceed the 8 GiB total input limit")
        clean = {"id": source["id"], "localPath": path}
        for key in ("title", "attribution", "license"):
            value = source.get(key)
            if not isinstance(value, str) or not value.strip() or len(value) > 4096:
                raise ValueError(f"Source {key} is required (at most 4096 characters)")
            clean[key] = value
        for key in ("sourceUrl", "licenseUrl"):
            clean[key] = public_url(source.get(key), key)
        for key, default in (("licenseAllowsTraining", False), ("evaluationOnly", True)):
            clean[key] = source.get(key, default)
            if type(clean[key]) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
        if not clean["licenseAllowsTraining"]:
            clean["evaluationOnly"] = True
        group = source.get("sourceGroupId")
        if group is not None and (not isinstance(group, str) or not GROUP_RE.fullmatch(group)):
            raise ValueError("Invalid sourceGroupId")
        clean["sourceGroupId"] = group
        clean["bytes"] = size
        result.append(clean)
    return result


def executable(name, override=None):
    found = str(Path(override).resolve(strict=True)) if override else shutil.which(name)
    if not found or not Path(found).is_file():
        raise ValueError(f"{name} is required on PATH, or supply its executable path")
    return found


def probe_video(path, ffprobe):
    command = [ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-select_streams", "v:0",
               "-show_entries", "stream=width,height,codec_name:format=duration", "-of", "json", str(path)]
    process = subprocess.run(command, capture_output=True, timeout=30, check=False)
    if process.returncode or len(process.stdout) > 1024 * 1024:
        raise ValueError("Could not probe local video")
    try:
        info = json.loads(process.stdout)
        stream = info["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        if width < 16 or height < 16 or width * height > MAX_PIXELS:
            raise ValueError("Video resolution must be 16 pixels or larger and at most 20 megapixels")
        duration = float(info.get("format", {}).get("duration", 0))
        if not math.isfinite(duration) or duration < 0:
            duration = None
        return {"width": width, "height": height, "durationSeconds": duration, "codec": stream.get("codec_name")}
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("No supported video stream was found") from error


@dataclass
class Candidate:
    path: Path
    timestamp: float
    blur: float
    digest: str
    dhash: int
    thumb: np.ndarray
    width: int
    height: int
    stratum: str = "middle"


def frame_metrics(image):
    """A descriptive sharpness statistic, not proof of blur or mushroom presence."""
    rgb = image.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    center = gray[1:-1, 1:-1]
    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * center
    blur = float(np.var(lap, dtype=np.float64))
    small = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.BILINEAR))
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    dhash = sum(int(value) << index for index, value in enumerate(bits))
    thumb = np.asarray(rgb.resize((32, 32), Image.Resampling.BILINEAR), dtype=np.int16)
    digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode() + rgb.tobytes()).hexdigest()
    return blur, digest, dhash, thumb


def decode_candidates(source, directory, ffmpeg, interval, max_seconds, limit):
    # select uses actual presentation timestamps, instead of fabricating time from
    # average FPS. showinfo is before output so its records identify selected frames.
    selector = f"select=isnan(prev_selected_t)+gte(t-prev_selected_t\\,{interval:.6f}),showinfo"
    command = [ffmpeg, "-hide_banner", "-loglevel", "info", "-nostdin", "-protocol_whitelist", "file,pipe",
               "-threads", "2", "-i", str(source["localPath"]), "-t", str(max_seconds), "-map", "0:v:0",
               "-an", "-sn", "-dn", "-vf", selector, "-fps_mode", "vfr", "-frames:v", str(limit),
               "-pix_fmt", "rgb24", "-threads", "2", str(directory / "%06d.png")]
    process = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=240, check=False)
    if process.returncode:
        raise ValueError("Local FFmpeg decoding failed: " + process.stderr.decode("utf-8", "replace")[-1600:])
    if len(process.stderr) > 4 * 1024 * 1024:
        raise ValueError("Unexpectedly large decoder log")
    times = [float(value) for value in re.findall(rb"\bn:\s*\d+\s+pts:\s*-?\d+\s+pts_time:([\d.eE+\-]+)", process.stderr)]
    paths = sorted(directory.glob("*.png"))
    if not paths or len(paths) > limit or len(times) < len(paths):
        raise ValueError("Video produced no frames or lacked decoded frame timestamps")
    result = []
    bytes_read = 0
    for index, path in enumerate(paths):
        bytes_read += path.stat().st_size
        if bytes_read > MAX_OUTPUT_BYTES:
            raise ValueError("Decoded candidate frames exceeded the 1 GiB limit")
        timestamp = times[index]
        if not math.isfinite(timestamp) or timestamp < 0 or timestamp > max_seconds + interval:
            raise ValueError("Decoder returned an unexpected timestamp")
        with Image.open(path) as original:
            if original.width * original.height > MAX_PIXELS:
                raise ValueError("Decoded image exceeded pixel limit")
            image = original.convert("RGB")
            blur, digest, dhash, thumb = frame_metrics(image)
            result.append(Candidate(path, timestamp, blur, digest, dhash, thumb, image.width, image.height))
    return result


def near_duplicate(left, right):
    # Near duplicate suppression stays within a sharpness stratum; a blurry view
    # must not disappear merely because the sharp view shows the same scene.
    return (left.stratum == right.stratum and left.width == right.width and left.height == right.height and
            (left.dhash ^ right.dhash).bit_count() <= 3 and float(np.mean(np.abs(left.thumb - right.thumb))) < 6)


def spread_order(rows):
    """Visit endpoints and then interior time positions before adjacent frames."""
    if not rows:
        return []
    pending, order = [(0, len(rows) - 1)], []
    while pending:
        lo, hi = pending.pop(0)
        if lo > hi:
            continue
        if not order:
            order.append(rows[lo])
            if hi != lo:
                order.append(rows[hi])
            if lo + 1 <= hi - 1:
                pending.append((lo + 1, hi - 1))
        else:
            middle = (lo + hi) // 2
            order.append(rows[middle])
            pending.extend(((lo, middle - 1), (middle + 1, hi)))
    return order


def select_frames(candidates, count):
    if not candidates:
        return []
    low, high = np.quantile([row.blur for row in candidates], [1 / 3, 2 / 3])
    buckets = {"low": [], "middle": [], "high": []}
    for row in sorted(candidates, key=lambda item: item.timestamp):
        row.stratum = "middle" if high == low else "low" if row.blur <= low else "high" if row.blur >= high else "middle"
        buckets[row.stratum].append(row)
    queues = {key: spread_order(rows) for key, rows in buckets.items()}
    selected, exact = [], set()
    while len(selected) < count and any(queues.values()):
        for key in ("low", "middle", "high"):
            while queues[key]:
                row = queues[key].pop(0)
                if row.digest in exact or any(near_duplicate(row, old) for old in selected):
                    continue
                selected.append(row)
                exact.add(row.digest)
                break
            if len(selected) >= count:
                break
    return sorted(selected, key=lambda row: row.timestamp)


def contact_sheet(directory, frames):
    columns, cell_width, cell_height = 4, 320, 224
    sheet = Image.new("RGB", (columns * cell_width, max(1, math.ceil(len(frames) / columns)) * cell_height), "#14231f")
    draw = ImageDraw.Draw(sheet)
    for index, frame in enumerate(frames):
        x, y = index % columns * cell_width, index // columns * cell_height
        with Image.open(directory / frame["path"]) as original:
            tile = ImageOps.contain(original.convert("RGB"), (cell_width - 8, 180), Image.Resampling.LANCZOS)
            sheet.paste(tile, (x + (cell_width - tile.width) // 2, y + (180 - tile.height) // 2))
        # Pillow's default font is intentionally portable.
        title = f"{frame['sourceId'][:31]}  {frame['timestampSeconds']:.2f}s"
        detail = f"Laplacian {frame['blurScore']:.1f} / {frame['blurStratum']} sharpness"
        draw.text((x + 6, y + 184), title, fill="white")
        draw.text((x + 6, y + 201), detail, fill="#b8d1c1")
    target = directory / "contact-sheet.jpg"
    with target.open("xb") as output:
        sheet.save(output, "JPEG", quality=90)


def extract_collection(sources_path, output_dir, frames_per_video=12, sample_interval=1.0,
                       max_seconds=180.0, ffmpeg_path=None, ffprobe_path=None):
    bounded_number(frames_per_video, "frames_per_video", 1, 60, integer=True)
    bounded_number(sample_interval, "sample_interval", 0.25, 30)
    bounded_number(max_seconds, "max_seconds", 1, 3600)
    sources = read_sources(sources_path)
    if frames_per_video * len(sources) > MAX_FRAMES:
        raise ValueError("Requested frames exceed the 120-image collection limit")
    ffmpeg, ffprobe = executable("ffmpeg", ffmpeg_path), executable("ffprobe", ffprobe_path)
    output_dir = Path(output_dir)
    if output_dir.is_symlink() or str(output_dir).startswith(("\\\\", "//")):
        raise ValueError("Output must be a local directory, not a link or network path")
    output_dir = output_dir.resolve()
    # Existing research/attribution files are allowed; collection artifacts never
    # overwrite earlier results or merge partially completed collections.
    for name in ("frames", "manifest.json", "collection.json", "contact-sheet.jpg", "INCOMPLETE.json"):
        if (output_dir / name).exists():
            raise ValueError(f"Output already contains {name}; choose a fresh collection directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir()
    frames, public_sources, total_output = [], [], 0
    try:
        for source in sources:
            video_hash = sha256_file(source["localPath"])
            info = probe_video(source["localPath"], ffprobe)
            group = source["sourceGroupId"] or "video:" + video_hash
            public = {key: value for key, value in source.items() if key != "localPath"}
            public.update({"sha256": video_hash, "videoHash": video_hash, "sourceVideoId": source["id"],
                           "sourceGroupId": group, "partitionGroup": group, "video": info})
            with tempfile.TemporaryDirectory(prefix=".decode-", dir=output_dir) as temp:
                # PNG candidate storage is bounded before decoding, including
                # high-resolution noisy imagery that compresses poorly.
                candidate_limit = min(MAX_CANDIDATES_PER_VIDEO, math.ceil(max_seconds / sample_interval) + 1,
                                      max(1, MAX_OUTPUT_BYTES // (info["width"] * info["height"] * 4 + 65536)))
                clip_seconds = min(max_seconds, info["durationSeconds"] or max_seconds)
                effective_interval = max(sample_interval, clip_seconds / max(1, candidate_limit - 1))
                candidates = decode_candidates(source, Path(temp), ffmpeg, effective_interval, max_seconds, candidate_limit)
                selected = select_frames(candidates, frames_per_video)
                public["candidateFrameCount"] = len(candidates)
                public["selectedFrameCount"] = len(selected)
                public["effectiveSampleIntervalSeconds"] = effective_interval
                public["sharpnessRange"] = {"minimum": min(row.blur for row in candidates), "maximum": max(row.blur for row in candidates)}
                for index, row in enumerate(selected):
                    frame_id = f"{source['id']}-{index:03d}"
                    relative = f"frames/{frame_id}.jpg"
                    destination = output_dir / relative
                    with Image.open(row.path) as decoded:
                        with destination.open("xb") as output:
                            decoded.convert("RGB").save(output, "JPEG", quality=95, subsampling=0)
                    total_output += destination.stat().st_size
                    if total_output > MAX_OUTPUT_BYTES:
                        raise ValueError("Collection frames exceeded the 1 GiB output limit")
                    frames.append({"id": frame_id, "sourceId": source["id"], "sourceVideoId": source["id"],
                                   "sourceGroupId": group, "partitionGroup": group, "path": relative,
                                   "sha256": sha256_file(destination), "decodedPixelSha256": row.digest,
                                   "videoHash": video_hash, "width": row.width, "height": row.height,
                                   "bytes": destination.stat().st_size, "mimeType": "image/jpeg",
                                   "timestampSeconds": row.timestamp, "timestampOrigin": "decoded presentation timestamp",
                                   "blurScore": row.blur, "blurMetric": BLUR_METRIC, "blurStratum": row.stratum,
                                   "actualDistanceM": None, "labels": [],
                                   "annotation": {"status": "unverified", "boxes": [], "complete": False},
                                   "trainingAllowed": False, "evaluationOnly": source["evaluationOnly"],
                                   "imageTransform": "decoded RGB; JPEG quality 95; no enhancement, resizing or synthetic blur"})
            if sha256_file(source["localPath"]) != video_hash:
                raise ValueError("Source video changed during extraction; collection is incomplete")
            public_sources.append(public)
            print(json.dumps({"sourceId": source["id"], "candidateFrames": public["candidateFrameCount"],
                              "selectedFrames": public["selectedFrameCount"]}), flush=True)
        if not frames:
            raise ValueError("No review frames were extracted")
        contact_sheet(output_dir, frames)
        manifest = {"schema": "mycoscout.video-collection.v1", "createdAt": datetime.now(timezone.utc).isoformat(),
                    "sources": public_sources, "frames": frames, "contactSheet": "contact-sheet.jpg",
                    "settings": {"framesPerVideo": frames_per_video, "sampleIntervalSeconds": sample_interval,
                                 "maxSecondsPerVideo": max_seconds, "maxCandidateFramesPerVideo": MAX_CANDIDATES_PER_VIDEO},
                    "warnings": ["Frames have no mushroom ground truth. Empty labels do not mean a negative scene.",
                                 "Sharpness strata are relative to each source and can also reflect texture, focus or compression.",
                                 "Physical camera distance is unknown; a wide view does not establish meters.",
                                 "Keep every frame and related clips from one source group together when later splitting data.",
                                 "Training requires human annotations and explicit rights permission; extraction does not grant permission."],
                    "autoTraining": False, "actuationEnabled": False}
        write_json_new(output_dir / "manifest.json", manifest)
        return manifest
    except Exception as error:
        write_json_new(output_dir / "INCOMPLETE.json", {"status": "incomplete", "error": str(error)[:2000],
                                                       "trainingAllowed": False, "autoTraining": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", required=True, type=Path, help="mycoscout.video-sources.v1 local JSON manifest")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--frames-per-video", type=int, default=12)
    parser.add_argument("--sample-interval", type=float, default=1.0)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--ffmpeg", type=Path)
    parser.add_argument("--ffprobe", type=Path)
    args = parser.parse_args()
    try:
        result = extract_collection(args.sources, args.output_dir, args.frames_per_video, args.sample_interval,
                                    args.max_seconds, args.ffmpeg, args.ffprobe)
    except (ValueError, OSError, subprocess.TimeoutExpired) as error:
        parser.exit(1, f"Video extraction failed: {error}\n")
    print(json.dumps({"manifest": str((args.output_dir / "manifest.json").resolve()), "frames": len(result["frames"])}))


if __name__ == "__main__":
    main()
