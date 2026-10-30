"""Download only the official torchvision ResNet18 weights for local V2 research."""
from pathlib import Path
import hashlib
import json
import torch
from torchvision.models import ResNet18_Weights


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root / "models/newfeaturecache"
    folder.mkdir(parents=True, exist_ok=True)
    url = ResNet18_Weights.IMAGENET1K_V1.url
    if url != "https://download.pytorch.org/models/resnet18-f37072fd.pth":
        raise RuntimeError("Unexpected torchvision model source")
    target = folder / "resnet18-f37072fd.pth"
    if not target.exists():
        torch.hub.download_url_to_file(url, str(target), hash_prefix="f37072fd")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if not digest.startswith("f37072fd"):
        raise RuntimeError("Official weight checksum prefix does not match")
    # Only deserialize tensors; do not execute unrestricted pickle or remote code.
    state = torch.load(target, weights_only=True, map_location="cpu")
    if "fc.weight" not in state or tuple(state["fc.weight"].shape) != (1000, 512):
        raise RuntimeError("Unexpected ResNet18 tensor structure")
    metadata = {"model": "torchvision.resnet18", "weights": "IMAGENET1K_V1", "url": url,
                "sha256": digest, "bytes": target.stat().st_size, "weightsOnlyLoading": True,
                "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms: resize256 center-crop224 ImageNet normalization"}
    (folder / "resnet18-provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
