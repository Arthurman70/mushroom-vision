"""Build an explicit, bounded contributor-service release. No private station files."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import re
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def build(collection, output):
    collection = Path(collection).resolve(strict=True)
    manifest = json.loads((collection / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "mycoscout.community-collection.v1":
        raise ValueError("Expected the explicitly exported public collection")
    sources = manifest.get("sources", [])
    if not sources or not 1 <= len(manifest.get("frames", [])) <= 2000:
        raise ValueError("Public sources and a bounded set of frames are required")
    for source in sources:
        license_text = str(source.get("license", ""))
        if not re.search(r"CC[ -]?BY|CC0|public domain", license_text, re.I) or re.search(r"\bNC\b|\bND\b|research.only", license_text, re.I):
            raise ValueError(f"Unapproved publication license for {source.get('id')}")
        for field in ("sourceUrl", "licenseUrl", "attribution"):
            if not source.get(field):
                raise ValueError(f"Missing public attribution field {field}")
    entries = {}

    def add(path, name):
        path = Path(path)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Not a regular release file: {name}")
        if name in entries:
            return
        entries[name] = path

    for name in ("server.mjs", "store.mjs", "export-reviews.mjs"):
        add(ROOT / "community" / name, "community/" + name)
    for path in (ROOT / "community" / "public").rglob("*"):
        if path.is_file():
            if path.suffix.lower() not in {".html", ".js", ".css", ".svg", ".png", ".jpg", ".ico", ".webp"}:
                raise ValueError(f"Unexpected public asset: {path.name}")
            add(path, "community/public/" + path.relative_to(ROOT / "community" / "public").as_posix())
    if "community/public/index.html" not in entries:
        raise ValueError("Public UI is not ready")
    add(collection / "manifest.json", "collection/manifest.json")
    if (collection / "ai-suggestions.json").exists():
        add(collection / "ai-suggestions.json", "collection/ai-suggestions.json")
    for frame in manifest["frames"]:
        name = frame.get("path", "")
        if not re.fullmatch(r"images/[a-f0-9]{64}\.(jpg|png)", name):
            raise ValueError("Only immutable public image files can be packaged")
        path = collection / PurePosixPath(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != frame.get("sha256") or digest != path.stem:
            raise ValueError(f"Image digest mismatch: {frame.get('id')}")
        add(path, "collection/" + name)
    for name in ("Dockerfile", "compose.yaml", "caddy-route.txt", "deploy-server.py"):
        add(HERE / name, name)
    total = sum(path.stat().st_size for path in entries.values())
    if total > 512 * 1024 * 1024:
        raise ValueError("Public release exceeds the 512 MiB bound")
    # Snapshot changing JSON/code once, so a progressive AI update cannot create
    # an archive whose bytes disagree with its integrity manifest.
    snapshots = {name: path.read_bytes() for name, path in entries.items() if not name.startswith("collection/images/")}
    integrity = {}
    for name, path in sorted(entries.items()):
        data = snapshots[name] if name in snapshots else path.read_bytes()
        integrity[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    encoded = json.dumps({"schema": "mycoscout.community-release.v1", "frames": len(manifest["frames"]), "files": integrity}, indent=2).encode()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "x:gz") as archive:
        for name, path in sorted(entries.items()):
            if name in snapshots:
                info = tarfile.TarInfo(name)
                info.size = len(snapshots[name])
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(snapshots[name]))
            else:
                archive.add(path, arcname=name, recursive=False)
        info = tarfile.TarInfo("release-manifest.json")
        info.size = len(encoded)
        info.mode = 0o644
        archive.addfile(info, io.BytesIO(encoded))
    return {"bundle": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "files": len(entries), "frames": len(manifest["frames"]), "uncompressedBytes": total, "archiveBytes": output.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.collection, args.output), indent=2))
