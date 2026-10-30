"""One small CPU-thread comparison; identical model, prompt and image resolution."""
from pathlib import Path
import json
import time
from PIL import Image
import torch
from adapters.scout_frame import LocalDetector

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "data/video-discovery-v1/manifest.json").read_text(encoding="utf-8"))
frame = manifest["frames"][100]
image = Image.open(root / "data/video-discovery-v1" / frame["path"]).convert("RGB")
model = LocalDetector(root / "models/grounding-dino-tiny", threads=4)
model.predict(image)  # warm-up is excluded from the comparison
rows = []
for threads in (4, 8):
    torch.set_num_threads(threads)
    started = time.perf_counter()
    predictions = model.predict(image)
    rows.append({"threads": threads, "seconds": time.perf_counter() - started, "predictions": predictions})
report = {"frameId": frame["id"], "imageSha256": frame["sha256"], "nativeSize": list(image.size),
          "runs": rows, "otherInferenceWasRunning": True, "model": model.info,
          "limitation": "One warmed sample under concurrent inference load; not a general throughput benchmark."}
out = root / "experiments/results/video-thread-benchmark-v1.json"
with out.open("x", encoding="utf-8") as stream:
    json.dump(report, stream, indent=2)
print(json.dumps({"report": str(out), "runs": [{"threads": r["threads"], "seconds": r["seconds"], "candidates": len(r["predictions"])} for r in rows]}))
