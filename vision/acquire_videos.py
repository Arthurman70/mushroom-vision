"""Download a bounded, explicitly selected Commons video collection with provenance."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import urllib.error


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Media redirects are not followed")


def verified_source(row):
    if not isinstance(row, dict):
        raise ValueError("Expected a video metadata object")
    if not re.fullmatch(r"[0-9]{1,20}", str(row.get("id", ""))):
        raise ValueError("Expected a Commons page ID")
    for key in ("downloadUrl", "sourceUrl", "licenseUrl", "sha1"):
        if not isinstance(row.get(key), str) or not 1 <= len(row[key]) <= 8192:
            raise ValueError(f"Invalid {key} metadata")
    url = urllib.parse.urlsplit(row.get("downloadUrl", ""))
    if (url.scheme != "https" or url.hostname != "upload.wikimedia.org" or url.port or
            url.username or url.password or url.query or url.fragment or
            not re.fullmatch(r"/wikipedia/commons/[a-f0-9]/[a-f0-9]{2}/[^/]+\.(webm|ogv|mp4)", url.path, re.IGNORECASE) or
            any(part in urllib.parse.unquote(url.path.rsplit("/", 1)[-1]) for part in ("/", "\\", "\x00"))):
        raise ValueError("Only direct Wikimedia video originals are supported")
    source = urllib.parse.urlsplit(row.get("sourceUrl", ""))
    if (source.scheme != "https" or source.netloc != "commons.wikimedia.org" or source.query or source.fragment or
            not source.path.startswith("/wiki/File:") or len(source.path) <= len("/wiki/File:")):
        raise ValueError("Commons file-page evidence required")
    license_url = row.get("licenseUrl", "").rstrip("/")
    if not re.fullmatch(r"https://creativecommons.org/(licenses/(by|by-sa)/(2.0|2.5|3.0|4.0)|publicdomain/zero/1.0)", license_url):
        raise ValueError("Expected verified open-license metadata")
    for key in ("author", "licenseEvidence", "title", "license"):
        if not isinstance(row.get(key), str) or not row[key].strip() or len(row[key]) > 4096:
            raise ValueError(f"Bounded nonempty {key} metadata required")
    normalized_license = re.sub(r"[\s_-]", "", row["license"]).upper()
    if "/publicdomain/zero/" in license_url:
        matches_license = normalized_license in {"CC0", "CC01.0", "CCZERO"}
    else:
        kind, version = license_url.split("/")[-2:]
        matches_license = normalized_license == "CC" + kind.replace("-", "").upper() + version
    if not matches_license or row.get("licenseAllowsTraining") is False:
        raise ValueError("License label and open-license URL must agree")
    if not row["title"].startswith("File:"):
        raise ValueError("Commons file title required")
    group = row.get("sourceGroupId")
    if group is not None and (not isinstance(group, str) or not re.fullmatch(r"[A-Za-z0-9:_-]{1,160}", group)):
        raise ValueError("Invalid source group")
    if type(row.get("bytes")) is not int or not 1 <= row["bytes"] <= 120_000_000:
        raise ValueError("Video exceeds the 120 MB file limit")
    if not re.fullmatch(r"[a-f0-9]{40}", row.get("sha1", "")):
        raise ValueError("Commons SHA1 metadata required")
    return url


def retry_deadline(value, now=None):
    """Record host instructions; acquisition never sleeps or retries itself."""
    if not isinstance(value, str) or not value.strip():
        return None
    now = now or datetime.now(timezone.utc)
    try:
        if re.fullmatch(r"[0-9]{1,9}", value.strip()):
            deadline = now + timedelta(seconds=int(value))
        else:
            deadline = parsedate_to_datetime(value)
            if deadline.tzinfo is None:
                deadline = deadline.replace(tzinfo=timezone.utc)
        return max(now, deadline.astimezone(timezone.utc)).isoformat()
    except (ValueError, TypeError, OverflowError):
        return None


def acquire(manifest, destination, ids, max_bytes=400_000_000):
    if (not isinstance(ids, list) or not 1 <= len(ids) <= 20 or
            any(not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,20}", value) for value in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Select 1–20 unique Commons IDs")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 1_000_000_000:
        raise ValueError("Collection limit must be at most 1 GB")
    if (not isinstance(manifest, dict) or manifest.get("schema") != "mycoscout.video-sources-research.v1" or
            not isinstance(manifest.get("sources"), list) or len(manifest["sources"]) > 1000):
        raise ValueError("Expected bounded mycoscout.video-sources-research.v1 metadata")
    by_id = {}
    for source in manifest["sources"]:
        if not isinstance(source, dict) or "id" not in source or str(source["id"]) in by_id:
            raise ValueError("Source metadata IDs must be unique")
        by_id[str(source["id"])] = {**source, "id": str(source["id"])}
    if any(value not in by_id for value in ids):
        raise ValueError("Selected source ID is absent from verified metadata")
    rows = [by_id[value] for value in ids]
    for row in rows:
        verified_source(row)
    if sum(r["bytes"] for r in rows) > max_bytes:
        raise ValueError("Selected originals exceed the collection byte budget")
    if str(destination).startswith(("\\\\", "//")) or Path(destination).is_symlink():
        raise ValueError("Destination must be a local collection directory")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Choose a new download collection directory")
    destination.mkdir(parents=True)
    (destination / "downloads").mkdir()
    opener = urllib.request.build_opener(NoRedirect())
    sources, failures = [], []
    for row in rows:
        suffix = Path(urllib.parse.urlsplit(row["downloadUrl"]).path).suffix
        relative = f"downloads/commons-{row['id']}{suffix}"
        output = destination / relative
        partial = output.with_suffix(output.suffix + ".partial")
        try:
            h1, h256, count = hashlib.sha1(), hashlib.sha256(), 0
            started = time.monotonic()
            request = urllib.request.Request(row["downloadUrl"], headers={"User-Agent": "MycoScoutResearch/0.7 (bounded licensed video collection)", "Accept": "video/*"})
            with opener.open(request, timeout=45) as response, partial.open("xb") as stream:
                if response.getcode() != 200 or response.geturl() != row["downloadUrl"]:
                    raise ValueError("Expected the selected original URL with HTTP 200")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and (not content_length.isdecimal() or int(content_length) != row["bytes"]):
                    raise ValueError("Response size differs from published size")
                while True:
                    if time.monotonic() - started > 300:
                        raise ValueError("Original download exceeded its five-minute time limit")
                    chunk = response.read(min(1024 * 1024, row["bytes"] - count + 1))
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > row["bytes"]:
                        raise ValueError("Original exceeds its published size")
                    h1.update(chunk); h256.update(chunk); stream.write(chunk)
            if count != row["bytes"] or h1.hexdigest() != row["sha1"]:
                raise ValueError("Original size or SHA1 differs from Commons evidence")
            partial.rename(output)
            # Group repeated shoots by author/date/location when supplied; never split neighboring frames.
            source_group = row.get("sourceGroupId") or "commons-video-" + row["id"]
            sources.append({"id": "commons-" + row["id"], "path": relative,
                "title": row["title"].removeprefix("File:"), "sourceUrl": row["sourceUrl"],
                "downloadUrl": row["downloadUrl"], "attribution": f"{row['author']} — {row['title']} — {row['license']}",
                "license": row["license"], "licenseUrl": row["licenseUrl"],
                "licenseAllowsTraining": True, "evaluationOnly": True,
                "sourceGroupId": source_group, "sha256": h256.hexdigest(), "bytes": count,
                "sourceMetadata": row, "distanceM": None, "annotations": [], "labelStatus": "unreviewed"})
            print(json.dumps({"downloaded": row["id"], "bytes": count}), flush=True)
        except urllib.error.HTTPError as exc:
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            failures.append({"id": row["id"], "httpStatus": exc.code, "retryAfter": retry_after,
                             "retryNotBefore": retry_deadline(retry_after), "error": str(exc)[:200]})
            print(json.dumps(failures[-1]), flush=True)
            if exc.code in (429, 503):
                break  # Respect host backoff; do not fan out more requests.
        except (OSError, ValueError) as exc:
            failures.append({"id": row["id"], "error": str(exc)[:500]})
            print(json.dumps({"failed": row["id"], "error": str(exc)[:200]}), flush=True)
    completed_ids = {row["id"].removeprefix("commons-") for row in sources} | {row["id"] for row in failures}
    result = {"schema": "mycoscout.video-sources.v1", "retrievedAt": datetime.now(timezone.utc).isoformat(),
        "sources": sources, "failures": failures, "downloadedBytes": sum(s["bytes"] for s in sources),
        "status": "complete" if len(sources) == len(rows) else "partial" if sources else "failed",
        "unattemptedIds": [row["id"] for row in rows if row["id"] not in completed_ids],
        "automaticTraining": False, "notice": "Source species are metadata hints only. Starter collection is evaluation-only; preserve attribution and license on extracted frames."}
    (destination / "sources.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ids", required=True, help="Comma-separated verified Commons page IDs")
    parser.add_argument("--max-bytes", type=int, default=400_000_000)
    args = parser.parse_args()
    try:
        if args.manifest.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Source metadata exceeds 8 MiB")
        result = acquire(json.loads(args.manifest.read_text(encoding="utf-8-sig")), args.output_dir, args.ids.split(","), args.max_bytes)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Video acquisition failed: {error}\n")
    print(json.dumps({"videos": len(result["sources"]), "bytes": result["downloadedBytes"], "failed": len(result["failures"])}))
    if not result["sources"]:
        parser.exit(1, "No videos were acquired; see sources.json for failure/backoff details.\n")


if __name__ == "__main__":
    main()
