"""Export the current reviewed dataset and checksum-verified images for later merging."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from .review_stream import Station


def export_bundle(station, destination):
    manifest = station.request("/api/training/export")
    if manifest.get("schema") != "mycoscout.reviewed-training-export.v1" or not re.fullmatch(r"reviewed-[a-f0-9]{64}", manifest.get("releaseId", "")):
        raise ValueError("Unexpected reviewed dataset schema or release ID")
    if len(manifest.get("records", [])) > 10000:
        raise ValueError("Dataset exceeds export limit")
    root = destination.resolve() / manifest["releaseId"]
    root.mkdir(parents=True, exist_ok=True)
    photos = root / "photos"
    photos.mkdir(exist_ok=True)
    for row in manifest["records"]:
        info = row["image"]
        digest = info["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", digest) or info["mimeType"] not in {"image/jpeg", "image/png"}:
            raise ValueError("Invalid image metadata")
        suffix = "jpg" if info["mimeType"] == "image/jpeg" else "png"
        target = photos / (digest + "." + suffix)
        expected_url = "/api/finds/" + row["findId"] + "/image"
        if not re.fullmatch(r"/api/finds/[a-f0-9-]{36}/image", expected_url) or info["url"] != expected_url:
            raise ValueError("Invalid image reference")
        data = target.read_bytes() if target.exists() else station.request(expected_url, binary=True)
        if len(data) != info["bytes"] or hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("Image failed its length or checksum check")
        if not target.exists():
            with target.open("xb") as output:
                output.write(data)
        info["relativePath"] = "photos/" + target.name
    manifest["imagePathBase"] = "directory containing this manifest"
    encoded = (json.dumps(manifest, indent=2) + "\n").encode("utf8")
    output = root / "manifest.json"
    if output.exists() and output.read_bytes() != encoded:
        raise ValueError("An immutable export already exists with different content")
    if not output.exists():
        with output.open("xb") as file:
            file.write(encoded)
    check = root / "manifest-sha256.json"
    check.write_text(json.dumps({"sha256": hashlib.sha256(encoded).hexdigest(), "releaseId": manifest["releaseId"]}, indent=2) + "\n", encoding="utf8")
    return {"manifest": str(output), "records": len(manifest["records"]), "releaseId": manifest["releaseId"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="http://127.0.0.1:8787")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_bundle(Station(args.station), args.output_dir), indent=2))


if __name__ == "__main__":
    main()
