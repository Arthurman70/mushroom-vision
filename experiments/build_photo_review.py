"""Build a private, self-contained photo review report with no remote assets.

The output embeds reduced photos and belongs under private data/, never public/
or a redistributable package. It reports candidate boxes, not mushroom accuracy.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageOps
from vision.dataset import image_path, load_valid
from vision.metrics import load_predictions


def safe_link(url):
    parsed = urlsplit(url)
    if parsed.scheme == "https" and parsed.hostname in {"facebook.com", "www.facebook.com"}:
        return f'<a href="{html.escape(url, quote=True)}" target="_blank" rel="noopener noreferrer">Source photo on Facebook ↗</a>'
    return ""


def boxes_svg(items, class_name, width, include_score=False):
    rows = []
    stroke = max(2.5, width / 450)
    font = max(16, width / 46)
    for index, item in enumerate(items, 1):
        x1, y1, x2, y2 = item["bboxPixelsXYXY"]
        text = f'D{index} score {item["confidence"]:.2f}' if include_score else f'L{index}'
        text_y = max(font, y1 - font * .25)
        rows.append(f'<g class="{class_name}"><rect x="{x1:.3f}" y="{y1:.3f}" width="{x2-x1:.3f}" height="{y2-y1:.3f}" fill="none" stroke-width="{stroke:.2f}"/>'
                    f'<text x="{x1:.3f}" y="{text_y:.3f}" font-size="{font:.2f}" stroke-width="{max(2, font/6):.2f}" paint-order="stroke" stroke="#092028">{html.escape(text)}</text></g>')
    return "".join(rows)


def render(manifest_path, prediction_dir, output):
    data = load_valid(manifest_path)
    predictions, detector = load_predictions(prediction_dir, data["images"])
    raw_counts = {}
    runtimes = {}
    for path in sorted(Path(prediction_dir).glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("schema") == "mycoscout.unlocated-candidates.v1":
            digest = payload["image"]["sha256"]
            raw_counts[digest] = len(payload.get("rawCandidates", payload["candidates"]))
            runtimes[digest] = payload.get("runtime", {}).get("inferenceSeconds")
    cards = []
    total_candidates = 0
    for record in data["images"]:
        prediction = predictions.get(record["id"])
        if prediction is None or prediction.get("status") != "ok":
            raise ValueError(f"Successful inference missing for {record['id']}; report would be incomplete")
        candidates = prediction["candidates"]
        total_candidates += len(candidates)
        labels = [annotation for annotation in record["annotations"] if annotation["label"] == "mushroom"]
        with Image.open(image_path(manifest_path, record)) as source:
            oriented = ImageOps.exif_transpose(source).convert("RGB")
            oriented.thumbnail((760, 760), Image.Resampling.LANCZOS)
            stream = io.BytesIO()
            oriented.save(stream, format="JPEG", quality=84, optimize=True)
        image_url = "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")
        width, height = record["width"], record["height"]
        status = record["annotationStatus"]
        note = "Partial labels: other fungi may be unmarked; no full-image accuracy metric." if status == "partial" else "First-pass complete labels: approximate and awaiting human confirmation."
        if status == "unreviewed":
            note = "Unreviewed: labels and accuracy unavailable."
        if record["context"] == "far" and not labels:
            note += " This is a distant negative example; it does not establish distant mushroom detection."
        runtime = runtimes.get(record["sha256"])
        runtime_text = f"{runtime:.2f} s CPU" if runtime is not None else "Timing unavailable"
        notes = record.get("annotationNotes", "")
        svg = (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(record["id"], quote=True)} photograph with reviewed labels and model candidate boxes">'
               f'<image href="{image_url}" x="0" y="0" width="{width}" height="{height}"/>'
               + boxes_svg(labels, "label-box", width)
               + boxes_svg(candidates, "detection-box", width, True) + '</svg>')
        cards.append(f'''<article class="photo-card">
<div class="photo">{svg}</div>
<div class="details"><div class="badges"><span>{html.escape(record['context'])}</span><span>{html.escape(status)} labels</span></div>
<h2>{html.escape(record['id'])}</h2>
<p class="counts"><b>{len(candidates)}</b> model candidates · <b>{len(labels)}</b> reviewed regions · {raw_counts.get(record['sha256'], len(candidates))} raw boxes</p>
<p class="technical">{width} × {height} original · {runtime_text}</p>
<p>{html.escape(note)}</p><p class="annotation">{html.escape(notes)}</p>
{safe_link(record['sourceUrl'])}</div></article>''')
    page = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="referrer" content="no-referrer"><title>MycoScout · private mushroom photo review</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:#edf5ef;background:#0b171b}
*{box-sizing:border-box}body{margin:0}header,main{max-width:1500px;margin:auto;padding:26px}header{padding-top:42px}h1{font-size:clamp(28px,4vw,44px);line-height:1.1;margin:8px 0 18px}h2{font-size:17px;overflow-wrap:anywhere}.eyebrow{letter-spacing:.13em;text-transform:uppercase;color:#a3c5bc;font-size:12px;font-weight:700}header p{max-width:1000px;line-height:1.55;color:#c2d7d1}.summary{font-size:20px!important;color:#fff!important}.controls{position:sticky;top:0;z-index:10;display:flex;gap:22px;flex-wrap:wrap;background:#173238f5;padding:16px 26px;border-block:1px solid #35575d}.controls label{cursor:pointer;display:flex;align-items:center;gap:9px}.controls input{width:18px;height:18px;accent-color:#78e89b}.controls small{color:#b9cfca;align-self:center}.legend-dot{width:13px;height:13px;border:2px solid;display:inline-block}.green{color:#76f49d}.orange{color:#ffae54}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:24px}.photo-card{border:1px solid #365159;border-radius:15px;overflow:hidden;background:#13262c}.photo{background:#081216}.photo svg{display:block;width:100%;max-height:680px}.details{padding:22px}.details p{line-height:1.5;color:#c8d8d4;font-size:14px}.details .counts{color:#f0f6f3}.details .technical{color:#a9c0ba;font-size:12px}.badges{display:flex;gap:8px;flex-wrap:wrap}.badges span{padding:4px 9px;border-radius:15px;background:#27454b;font-size:12px;color:#d1e7df}.annotation{color:#9eb7b0!important}a{color:#9fe2d2;font-size:13px}.label-box{stroke:#76f49d;fill:#76f49d}.detection-box{stroke:#ffae54;fill:#ffae54}.detection-box rect{stroke-dasharray:12 6}.label-box text,.detection-box text{font-weight:750}body:has(#labels:not(:checked)) .label-box,body:has(#detections:not(:checked)) .detection-box{display:none}footer{padding:28px;color:#9eb8b0;font-size:13px}
</style></head><body>
<header><div class="eyebrow">MycoScout · private local review · 26 October 2026</div><h1>Mushroom vision: inspect the evidence</h1>
<p class="summary">__SUMMARY__</p>
<p>Green boxes are independent approximate reviewed mushroom regions. Orange boxes are pretrained model proposals after fixed duplicate suppression. Overlapping colonies can be represented differently by the reviewer and model; candidate totals are not mushroom counts or accuracy. Toggle either layer to inspect what the model found, missed, or confused.</p>
<p>No species or edibility identification. Physical photo distances are unknown. These photographs do not validate autonomous woodland flight. This page embeds private thumbnails locally and loads no external assets.</p></header>
<div class="controls"><label><input id="labels" type="checkbox" checked><span class="legend-dot green"></span>Reviewed regions</label><label><input id="detections" type="checkbox" checked><span class="legend-dot orange"></span>Model proposals and scores</label><small>Scores are uncalibrated; they are not probabilities of correct identification.</small></div>
<main class="grid">__CARDS__</main><footer>__MODEL__ · Source-photo links open Facebook only when selected. Photographs are retained privately; redistribution permission is not inferred.</footer>
</body></html>'''
    description = f"{len(cards)} original group photos · {total_candidates} model proposals · full-frame inference"
    model = html.escape(f"{detector.get('model', 'Model unknown')} · revision {detector.get('resolvedRevision', 'unknown')} · box threshold {detector.get('boxThreshold')} · NMS IoU {detector.get('postprocessing', {}).get('iouThreshold')}")
    page = page.replace("__SUMMARY__", html.escape(description)).replace("__CARDS__", "".join(cards)).replace("__MODEL__", model)
    output = Path(output).resolve()
    private_root = (ROOT / "data").resolve()
    if not output.is_relative_to(private_root):
        raise ValueError("Photo reports must remain inside the private data directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return {"path": str(output), "photoCount": len(cards), "candidateCount": total_candidates, "bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.manifest, args.predictions, args.output)))


if __name__ == "__main__":
    main()
