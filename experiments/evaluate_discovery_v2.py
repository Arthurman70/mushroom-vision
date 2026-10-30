"""Paired frozen-prompt/tile comparisons on the selected black-trumpet audit."""
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.dataset import load_valid, sha256, write_json
from vision.metrics import evaluate, load_predictions

EXPECTED_MANIFEST_SHA = "334b8c209db455e4b7d32dc5bd633b0a410638b8c05d10e8295722a1e02542a0"
CONTEXT_IDS = ["inat-photo-745660170", "inat-photo-734698732", "inat-photo-718433879", "inat-photo-717487456"]


def run():
    manifest = ROOT / "data/black-trumpet-discovery-v2/manifest.json"
    if sha256(manifest) != EXPECTED_MANIFEST_SHA:
        raise ValueError("Discovery labels changed after the frozen audit declaration")
    data = load_valid(manifest)
    context = deepcopy(data)
    context["images"] = [row for row in context["images"] if row["id"] in CONTEXT_IDS]
    outputs = {}
    for name in ("generic", "black-aware", "black-aware-tiled", "generic-tiled"):
        folder = ROOT / "experiments/results" / f"v2-discovery-{name}"
        selected = context if name.endswith("tiled") else data
        predictions, detector = load_predictions(folder, selected["images"])
        if len(predictions) != len(selected["images"]):
            raise ValueError(f"Incomplete inference for {name}")
        report = evaluate(selected, predictions, detector, confidence=.25, iou_threshold=.5)
        timing = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        outputs[name] = {"evaluated": report, "meanInferenceSeconds": timing["meanInferenceSeconds"],
                         "meanCompleteSeconds": sum(item["totalImageSeconds"] for item in timing["images"]) / len(timing["images"])}
        write_json(folder / "evaluation.json", report)
        if not name.endswith("tiled"):
            outputs[name + "-context-subset"] = {"evaluated": evaluate(context, predictions, detector, confidence=.25, iou_threshold=.5)}
    result = {"schema": "mycoscout.discovery-comparison.v2", "benchmarkManifestSha256": EXPECTED_MANIFEST_SHA,
              "benchmarkRole": data.get("benchmarkRole"), "annotationPolicy": data.get("annotationPolicy"),
              "protocol": {"boxThreshold": .25, "textThreshold": .25, "nmsIou": .5, "matchingIou": .5,
                           "tileSize": 640, "overlap": .25, "tiledImageIds": CONTEXT_IDS,
                           "thresholdTunedOnAudit": False,
                           "genericTiledStatus": "Exploratory extra comparison added after initial black-aware results; do not treat it as untouched test selection"}, "comparisons": outputs,
              "limitations": ["All images are selected positives; this cannot estimate background specificity or operational false alarms.",
                              "Approximate single-assistant boxes may separate fused bodies differently than the model.",
                              "Source taxon is weak metadata; this tests candidate localization, not species confirmation.",
                              "Unknown camera distances and no flight frames; no physical detection range is established.",
                              "The audit is selected development evidence, not a randomized independent operational test."]}
    output = ROOT / "experiments/results/v2-discovery-comparison.json"
    write_json(output, result)
    print(json.dumps({name: value["evaluated"]["real"] for name, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    run()
