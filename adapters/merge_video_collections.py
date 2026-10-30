"""Append extracted video collections without changing existing frames or grades.

Dry run is the default. --apply copies only new frame files, then atomically
publishes a manifest. It never opens a grade database or rewrites source videos.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import uuid
from urllib.parse import urlsplit

from PIL import Image

SCHEMA = "mycoscout.video-collection.v1"
MAX_MANIFEST = 8 * 1024 * 1024
MAX_IMAGE = 8 * 1024 * 1024
MAX_SOURCES, MAX_FRAMES = 200, 2000
ID = re.compile(r"[A-Za-z0-9_-]{1,160}\Z")
GROUP = re.compile(r"[A-Za-z0-9:_-]{1,160}\Z")
HASH = re.compile(r"[a-f0-9]{64}\Z")
FRAME_PATH = re.compile(r"frames/[A-Za-z0-9._-]+\.(?:jpg|png)\Z")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def linked(path):
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def local_root(value):
    path = Path(value)
    if str(path).startswith(("\\\\", "//")) or linked(path):
        raise ValueError("Collections must be local directories, not links or network paths")
    path = path.resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Collection directory does not exist")
    return path


def bounded_path(root, relative, exists=True):
    path = root / relative
    # Check every child component so a frames directory junction cannot redirect writes.
    cursor = path
    while cursor != root:
        if linked(cursor):
            raise ValueError(f"Links inside collections are not accepted: {relative}")
        if cursor.parent == cursor:
            raise ValueError("Path escaped collection")
        cursor = cursor.parent
    actual = path.resolve(strict=exists)
    if not actual.is_relative_to(root) or actual == root:
        raise ValueError("Path escaped collection")
    if exists and not actual.is_file():
        raise ValueError(f"Expected a regular file: {relative}")
    return actual


def finite(value, name, maximum, integer=False, minimum=0):
    if (type(value) not in (int, float) or not math.isfinite(value) or
            value < minimum or value > maximum or (integer and type(value) is not int)):
        raise ValueError(f"Invalid {name}")


def text(value, name, limit):
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"Invalid {name}")


def url(value, name, allow_http=False):
    text(value, name, 2000)
    if not value:
        return
    parsed = urlsplit(value)
    if parsed.scheme not in ({"http", "https"} if allow_http else {"https"}) or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError(f"{name} must be an HTTPS attribution URL")


def video_hash(source):
    return source.get("videoHash", source.get("sha256"))


def group(source):
    return source.get("sourceGroupId", "video:" + video_hash(source))


def permission(source):
    allowed = source.get("licenseAllowsTraining") is True
    return allowed, source.get("evaluationOnly") is not False or not allowed


def read_collection(directory, *, allow_http_attribution=False):
    root = local_root(directory)
    if (root / "INCOMPLETE.json").exists():
        raise ValueError("Incomplete extraction cannot be imported")
    manifest = bounded_path(root, "manifest.json")
    if manifest.stat().st_size > MAX_MANIFEST:
        raise ValueError("Manifest exceeds the grader's 8 MiB limit")
    raw = manifest.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if (not isinstance(data, dict) or data.get("schema") != SCHEMA or
            not isinstance(data.get("sources"), list) or not isinstance(data.get("frames"), list) or
            len(data["sources"]) > MAX_SOURCES or len(data["frames"]) > MAX_FRAMES):
        raise ValueError("Invalid collection schema or source/frame limit")
    sources, frame_ids, paths = {}, set(), set()
    for source in data["sources"]:
        if not isinstance(source, dict) or not ID.fullmatch(str(source.get("id", ""))):
            raise ValueError("Invalid source ID")
        if source["id"] in sources or not HASH.fullmatch(str(video_hash(source))):
            raise ValueError("Duplicate source ID or invalid video hash")
        if "sha256" in source and source["sha256"] != video_hash(source):
            raise ValueError("Source hashes disagree")
        if not GROUP.fullmatch(group(source)) or source.get("partitionGroup", group(source)) != group(source):
            raise ValueError("Source grouping conflict")
        if source.get("sourceVideoId", source["id"]) != source["id"]:
            raise ValueError("Source video identity conflict")
        for key, limit in (("title", 500), ("attribution", 2000), ("license", 160)):
            text(source.get(key), key, limit)
        for key in ("sourceUrl", "licenseUrl"):
            url(source.get(key, ""), key, allow_http=allow_http_attribution)
        for key in ("licenseAllowsTraining", "evaluationOnly"):
            if key in source and type(source[key]) is not bool:
                raise ValueError(f"Invalid source {key}")
        sources[source["id"]] = source
    for frame in data["frames"]:
        if not isinstance(frame, dict) or not ID.fullmatch(str(frame.get("id", ""))):
            raise ValueError("Invalid frame ID")
        source = sources.get(frame.get("sourceId"))
        relative = frame.get("path", "")
        if (not source or frame["id"] in frame_ids or not isinstance(relative, str) or
                not FRAME_PATH.fullmatch(relative) or relative.casefold() in paths or
                not HASH.fullmatch(str(frame.get("sha256", "")))):
            raise ValueError("Duplicate frame ID/path or invalid frame source/hash/path")
        frame_ids.add(frame["id"])
        paths.add(relative.casefold())
        for key in ("sourceGroupId", "partitionGroup"):
            if frame.get(key, group(source)) != group(source):
                raise ValueError("Frame grouping conflicts with source")
        if (frame.get("videoHash", video_hash(source)) != video_hash(source) or
                frame.get("sourceVideoId", source["id"]) != source["id"]):
            raise ValueError("Frame video provenance conflicts with source")
        for key in ("width", "height"):
            finite(frame.get(key), key, 20_000_000, integer=True, minimum=1)
        if frame["width"] * frame["height"] > 20_000_000:
            raise ValueError("Image exceeds 20 megapixels")
        finite(frame.get("timestampSeconds"), "timestamp", 86400)
        finite(frame.get("blurScore"), "blur score", 1e30)
        text(frame.get("blurMetric"), "blur metric", 160)
        if "decodedPixelSha256" in frame and not HASH.fullmatch(str(frame["decodedPixelSha256"])):
            raise ValueError("Invalid decoded pixel hash")
        image = bounded_path(root, relative)
        if not 0 < image.stat().st_size <= MAX_IMAGE:
            raise ValueError("Frame exceeds the grader's 8 MiB image limit")
        image_bytes = image.read_bytes()
        if digest(image_bytes) != frame["sha256"] or frame.get("bytes", len(image_bytes)) != len(image_bytes):
            raise ValueError(f"Frame integrity mismatch: {frame['id']}")
        with Image.open(image) as decoded:
            expected = "PNG" if image.suffix == ".png" else "JPEG"
            if decoded.format != expected or decoded.size != (frame["width"], frame["height"]):
                raise ValueError("Frame format/dimensions conflict")
            decoded.verify()
    return root, raw, data


def merge_plan(destination, incoming):
    root, original, existing = read_collection(destination)
    additions = [read_collection(path) for path in incoming]
    merged = dict(existing)
    merged["sources"], merged["frames"] = list(existing["sources"]), list(existing["frames"])
    source_index = {x["id"]: x for x in merged["sources"]}
    frame_index = {x["id"]: x for x in merged["frames"]}
    hashes, urls, groups, frame_hashes, paths = {}, {}, {}, {}, {}
    shared_groups, duplicate_images, identical = set(), set(), 0

    def register_source(source):
        sid, vh, sg = source["id"], video_hash(source), group(source)
        if vh in hashes and hashes[vh]["id"] != sid:
            raise ValueError("Repeated video bytes under another source ID; retain the original identity")
        if source.get("sourceUrl") in urls and urls[source["sourceUrl"]] != sg:
            raise ValueError("Same source URL assigned to different dataset groups")
        if sg in groups:
            if groups[sg] != permission(source):
                raise ValueError("Dataset group has conflicting training/evaluation permissions")
            shared_groups.add(sg)
        hashes[vh] = source
        if source.get("sourceUrl"):
            urls[source["sourceUrl"]] = sg
        groups[sg] = permission(source)

    def register_frame(frame):
        sg = group(source_index[frame["sourceId"]])
        for kind in ("sha256", "decodedPixelSha256"):
            key = (kind, frame.get(kind))
            if key[1] is None:
                continue
            if key in frame_hashes:
                if frame_hashes[key] != sg:
                    raise ValueError("Identical image bytes/pixels assigned to different dataset groups")
                duplicate_images.add(frame["id"])
            frame_hashes[key] = sg
        key = frame["path"].casefold()
        if key in paths and paths[key] != frame["id"]:
            raise ValueError("Frame file path collision")
        paths[key] = frame["id"]

    for source in merged["sources"]:
        register_source(source)
    for frame in merged["frames"]:
        register_frame(frame)
    copies = []
    for source_root, _, collection in additions:
        for source in collection["sources"]:
            if source["id"] in source_index:
                if source_index[source["id"]] != source:
                    raise ValueError(f"Existing source metadata is immutable: {source['id']}")
                continue
            register_source(source)
            source_index[source["id"]] = source
            merged["sources"].append(source)
        for frame in collection["frames"]:
            if frame["id"] in frame_index:
                if frame_index[frame["id"]] != frame:
                    raise ValueError(f"Existing frame metadata is immutable: {frame['id']}")
                identical += 1
                continue
            register_frame(frame)
            frame_index[frame["id"]] = frame
            merged["frames"].append(frame)
            copies.append((bounded_path(source_root, frame["path"]), frame))
    if len(merged["sources"]) > MAX_SOURCES or len(merged["frames"]) > MAX_FRAMES:
        raise ValueError("Merged collection exceeds 200 sources or 2,000 frames")
    output = encode(merged)
    if len(output) > MAX_MANIFEST:
        raise ValueError("Merged manifest exceeds 8 MiB")
    report = {"schema": "mycoscout.video-merge.v1", "dryRun": True,
              "sourcesBefore": len(existing["sources"]), "framesBefore": len(existing["frames"]),
              "sourcesAfter": len(merged["sources"]), "framesAfter": len(merged["frames"]),
              "newFrames": len(copies), "unchangedDuplicateFrames": identical,
              "sharedGroups": sorted(shared_groups), "duplicateImageFramesInSameGroup": sorted(duplicate_images),
              "manifestBeforeSha256": digest(original), "manifestAfterSha256": digest(output),
              "gradesModified": False, "automaticTraining": False}
    return root, original, output, copies, report


@contextmanager
def merge_lock(root):
    lock = root / ".video-merge.lock"
    try:
        stream = lock.open("x", encoding="utf-8")
    except FileExistsError as error:
        raise ValueError("Another merge is active, or a stale .video-merge.lock needs inspection") from error
    try:
        with stream:
            stream.write(json.dumps({"pid": os.getpid(), "createdAt": datetime.now(timezone.utc).isoformat()}))
        yield
    finally:
        lock.unlink()


def write_new(path, data):
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def merge_collections(destination, incoming, apply=False):
    root = local_root(destination)
    if not apply:
        return merge_plan(root, incoming)[-1]
    with merge_lock(root):
        root, original, output, copies, report = merge_plan(root, incoming)
        # Exact retries are a no-op, including the existing manifest's whitespace.
        if not copies and report["sourcesBefore"] == report["sourcesAfter"]:
            report.update(dryRun=False, manifestAfterSha256=digest(original), unchanged=True)
            return report
        (root / "frames").mkdir(exist_ok=True)
        for source_file, frame in copies:
            target = bounded_path(root, frame["path"], exists=False)
            if target.exists():
                if digest(target.read_bytes()) != frame["sha256"]:
                    raise ValueError(f"Refusing to overwrite a conflicting existing image: {frame['path']}")
                continue  # Resume after an interrupted copy, with unchanged bytes.
            temporary = root / (".video-merge-" + uuid.uuid4().hex)
            try:
                with source_file.open("rb") as source, temporary.open("xb") as dest:
                    shutil.copyfileobj(source, dest, 1024 * 1024)
                    dest.flush()
                    os.fsync(dest.fileno())
                if digest(temporary.read_bytes()) != frame["sha256"]:
                    raise ValueError("Incoming frame changed during import")
                # A hard link publishes the complete file exclusively; it never overwrites.
                os.link(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        if (root / "manifest.json").read_bytes() != original:
            raise ValueError("Destination manifest changed during merge; no manifest was published")
        backup = root / ("manifest.before-" + digest(original) + ".json")
        if backup.exists():
            if backup.read_bytes() != original:
                raise ValueError("Manifest backup integrity mismatch")
        else:
            write_new(backup, original)
        temporary = root / (".manifest-" + uuid.uuid4().hex + ".json")
        try:
            write_new(temporary, output)
            os.replace(temporary, root / "manifest.json")
        finally:
            temporary.unlink(missing_ok=True)
        report.update(dryRun=False, backup=backup.name, unchanged=False)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--incoming", type=Path, action="append", required=True)
    parser.add_argument("--apply", action="store_true", help="Commit the append; otherwise validate only")
    args = parser.parse_args()
    try:
        report = merge_collections(args.destination, args.incoming, apply=args.apply)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Collection merge failed: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
