"""Summarize repeated-source synthetic pixel stress without claiming range."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.dataset import load_valid, sha256, write_json
from vision.metrics import evaluate, load_predictions


def main():
    manifest = ROOT / "experiments/results/v2-discovery-stress-inputs/manifest.json"
    predictions_dir = ROOT / "experiments/results/v2-discovery-generic-stress"
    data = load_valid(manifest)
    predictions, detector = load_predictions(predictions_dir, data["images"])
    if len(predictions) != len(data["images"]):
        raise ValueError("Incomplete scale-stress inference")
    levels = []
    for scale in (1.0, .5, .25, .125):
        selected = deepcopy(data)
        selected["images"] = [row for row in data["images"] if row["transform"]["scaleX"] == scale]
        report = evaluate(selected, predictions, detector, confidence=.25, iou_threshold=.5)
        levels.append({"relativeSceneScale": scale, "metrics": report["synthetic"],
                       "byObjectPixelSize": report["byObjectPixelSize"]})
        write_json(predictions_dir / f"evaluation-scale-{str(scale).replace('.', 'p')}.json", report)
    result = {"schema": "mycoscout.scale-stress-comparison.v2", "manifestSha256": sha256(manifest),
              "sourceOriginalPhotos": 4, "syntheticPhotos": len(data["images"]), "levels": levels,
              "detector": detector, "limitations": [
                  "Four repeated source photos, not sixteen independent field observations.",
                  "Centered scene reduction on an artificial neutral canvas with blur0.6 and JPEG80.",
                  "Relative scene scale changes target pixel footprint; it does not measure real camera range.",
                  "No woodland motion, metric depth, occlusion or safe aircraft approach was validated."]}
    write_json(ROOT / "experiments/results/v2-discovery-scale-stress.json", result)
    print(json.dumps(levels, indent=2))


if __name__ == "__main__":
    main()
