"""Frozen contextual-photo pixel-scale stress set; no metric distance inference."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.curriculum import make_curriculum
from vision.dataset import load_valid, sha256
from evaluate_discovery_v2 import EXPECTED_MANIFEST_SHA, CONTEXT_IDS


def main():
    path = ROOT / "data/black-trumpet-discovery-v2/manifest.json"
    if sha256(path) != EXPECTED_MANIFEST_SHA:
        raise ValueError("Discovery manifest changed")
    data = load_valid(path)
    data["images"] = [row for row in data["images"] if row["id"] in CONTEXT_IDS]
    target = ROOT / "experiments/results/v2-discovery-stress-inputs"
    result = make_curriculum(data, path, target, scales=[1.0, .5, .25, .125], blur_radius=.6, quality=80, mode="fixed-canvas")
    print(f"Prepared {len(result['images'])} synthetic scale-stress images from four independent source photos: {target}")


if __name__ == "__main__":
    main()
