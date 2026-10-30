"""Download one official, revision-pinned safetensors model to a local directory.

This downloads model assets only. It does not load remote Python, pickle weights,
or upload any of the local survey photographs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


MODEL_ID = "IDEA-Research/grounding-dino-tiny"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("models/grounding-dino-tiny"))
    parser.add_argument("--revision", default="main", help="Official model commit hash, tag, or branch")
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(MODEL_ID, revision=args.revision)
    revision = info.sha
    print(f"Downloading {MODEL_ID}@{revision}", flush=True)
    snapshot_download(
        MODEL_ID, revision=revision, local_dir=str(args.output_dir),
        allow_patterns=["*.json", "*.txt", "*.safetensors", "README.md", "LICENSE*"],
        ignore_patterns=["*.bin", "*.pt", "*.pth", "*.py", "*.h5", "*.msgpack"],
    )
    files = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "mycoscout-model-provenance.json":
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            files.append({"name": path.name, "bytes": path.stat().st_size,
                          "sha256": checksum})
    metadata = {
        "modelId": MODEL_ID, "requestedRevision": args.revision, "resolvedRevision": revision,
        "downloadedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{revision}",
        "weightsFormat": "safetensors", "remoteCodeAllowed": False, "files": files,
    }
    (args.output_dir / "mycoscout-model-provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"modelId": MODEL_ID, "revision": revision,
                      "fileCount": len(files), "outputDir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
