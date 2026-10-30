"""Derive quarantined review copies without changing frozen far-pixel predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.far_pixel_benchmark import digest, quarantine_color_report, write


def derive(run_dir):
    run_dir = Path(run_dir).resolve()
    destination = run_dir / "review-ready-color"
    if destination.exists():
        raise ValueError("Review derivation exists; it will not be overwritten")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if digest(run_dir / "inputs.json") != report["inputManifestSha256"] or digest(run_dir / "protocol.json") != report["protocolSha256"]:
        raise ValueError("Frozen input manifest or protocol changed")
    rows = json.loads((run_dir / "inputs.json").read_text(encoding="utf-8"))["images"]
    if not 1 <= len(rows) <= 1000:
        raise ValueError("Expected a bounded benchmark image list")
    prepared = []
    for row in rows:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", row["id"]):
            raise ValueError("Invalid benchmark image identifier")
        name = f"{row['id']}-color-cues.json"
        source = run_dir / "predictions" / name
        source_hash = digest(source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload["image"]["sha256"] != row["sha256"]:
            raise ValueError("Prediction does not match frozen input image")
        derived = quarantine_color_report(payload, source_hash)
        prepared.append((name, source_hash, derived))
    destination.mkdir()
    audit = {"schema": "mycoscout.far-pixel-review-derivation.v1", "reason":
        "Original color predictions lacked top-level synthetic/evaluationOnly flags. These review-ready copies explicitly quarantine benchmark cues; original model evidence is unchanged.",
        "sourceReportSha256": digest(run_dir / "report.json"),
        "inputManifestSha256": report["inputManifestSha256"], "protocolSha256": report["protocolSha256"],
        "inferenceRerun": False, "originalPredictionsModified": False, "files": []}
    for name, source_hash, payload in prepared:
        path = destination / name
        write(path, payload)
        if digest(run_dir / "predictions" / name) != source_hash:
            raise ValueError("Original prediction changed while deriving review copy")
        audit["files"].append({"source": f"predictions/{name}", "sourceSha256": source_hash,
                               "derived": f"review-ready-color/{name}", "derivedSha256": digest(path),
                               "synthetic": True, "evaluationOnly": True})
    write(destination / "derivation-audit.json", audit)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "experiments/results/far-pixel-v1")
    args = p.parse_args()
    audit = derive(args.run)
    print(json.dumps({"copies": len(audit["files"]), "audit": str(args.run / "review-ready-color/derivation-audit.json"),
                      "inferenceRerun": False, "originalPredictionsModified": False}))


if __name__ == "__main__":
    main()
