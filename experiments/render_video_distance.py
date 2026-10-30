"""Render immutable experiment evidence; no inference, retuning, or label edits."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from experiments.video_distance_benchmark import LEVELS, POSITIVE_IDS, METHODS
from vision.training_pipeline import iou
from vision.video_ai_pass import sha


def font(size):
    try:
        return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render(directory):
    directory = Path(directory)
    report_path = directory / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["status"] != "completed":
        raise ValueError("Render the finished fixed experiment only")
    records = {(r["frameId"], r["condition"]): r for r in report["records"]}
    cell_w, cell_h = 420, 330
    sheet = Image.new("RGB", (cell_w * 3, 120 + cell_h * len(LEVELS)), (20, 28, 38))
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 12), "Video target evidence - synthetic size test", font=font(26), fill="white")
    draw.text((20, 48), "Green: unchanged human region after projection. Crops are magnified for display only.", font=font(17), fill="#d1d9e6")
    draw.text((20, 76), "R / C / T = raw / color / tiles. HIT requires score >=0.25 and IoU >=0.5; no measured distances.", font=font(17), fill="#d1d9e6")
    misses = []
    for row_index, (condition, nominal) in enumerate(LEVELS):
        for column, frame_id in enumerate(POSITIVE_IDS):
            record = records[(frame_id, condition)]
            image_path = directory / record["imagePath"]
            if sha(image_path.read_bytes()) != record["derivedImageSha256"]:
                raise ValueError("Derived experiment image changed")
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            box = record["truth"][0]["bbox"]
            target_width = box[2]-box[0]
            side = max(64, math.ceil(max(box[2]-box[0], box[3]-box[1]) * 1.5))
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
            left, top = round(cx-side/2), round(cy-side/2)
            crop = image.crop((left, top, left+side, top+side))
            # Nearest-neighbor display exposes retained pixels without invented detail.
            panel = crop.resize((220, 220), Image.Resampling.NEAREST)
            panel_draw = ImageDraw.Draw(panel)
            scale = 220 / side
            panel_draw.rectangle([(box[0]-left)*scale, (box[1]-top)*scale,
                                  (box[2]-left)*scale, (box[3]-top)*scale], outline="#52f097", width=2)
            x, y = column * cell_w, 120 + row_index * cell_h
            draw.text((x+10, y+5), f"Frame {frame_id.rsplit('-', 1)[-1]} / {'native' if nominal is None else str(nominal)+' px requested'}", font=font(18), fill="white")
            draw.text((x+10, y+30), f"Actual width {target_width:.4f} px; crop {side} x {side} px", font=font(16), fill="#d1d9e6")
            sheet.paste(panel, (x+10, y+58))
            for method_index, method in enumerate(METHODS):
                predictions = record["predictions"][method]
                best_iou = max((iou(box, pred["bbox"]) for pred in predictions), default=0)
                hit = best_iou >= .5
                draw.text((x+240, y+70+method_index*48), method.upper()+": "+("HIT" if hit else "MISS"), font=font(17), fill="#52f097" if hit else "#ffab8f")
                draw.text((x+240, y+92+method_index*48), f"Best IoU {best_iou:.3f}", font=font(15), fill="#d1d9e6")
                if not hit:
                    misses.append({"frameId": frame_id, "condition": condition, "method": method,
                                   "requestedTargetPixels": nominal, "realizedTargetPixels": target_width,
                                   "bestIoU": best_iou, "proposalCount": len(predictions),
                                   "imagePath": record["imagePath"], "truth": record["truth"]})
            draw.text((x+10, y+286), "Original image preserved; this view is display-only.", font=font(15), fill="#d1d9e6")
    sheet.save(directory / "evidence-contact-sheet.png")
    context = Image.new("RGB", (1260, 320), (20, 28, 38))
    d = ImageDraw.Draw(context)
    d.text((12, 12), "Native source context - saved human regions", font=font(24), fill="white")
    for column, frame_id in enumerate(POSITIVE_IDS):
        record = records[(frame_id, "native")]
        with Image.open(directory / record["imagePath"]) as source:
            preview = source.convert("RGB").resize((400, 225), Image.Resampling.LANCZOS)
            sx, sy = 400/source.width, 225/source.height
        b = record["truth"][0]["bbox"]
        ImageDraw.Draw(preview).rectangle([b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy], outline="#52f097", width=2)
        context.paste(preview, (10+column*420, 55))
        d.text((10+column*420, 290), f"Frame {frame_id.rsplit('-',1)[-1]} - {b[2]-b[0]:.2f} native pixels", font=font(18), fill="white")
    context.save(directory / "native-context.png")
    artifact_record = records[(POSITIVE_IDS[1], "64px")]
    with Image.open(directory / artifact_record["imagePath"]) as source:
        example = source.convert("RGB")
    artifact = Image.new("RGB", (example.width, example.height+110), (20, 28, 38))
    artifact.paste(example, (0, 110))
    ad = ImageDraw.Draw(artifact)
    ad.text((15, 10), "Synthetic padding artifact - frame 005 at requested 64 px", font=font(25), fill="white")
    ad.text((15, 45), "Green: human target. Orange: unmatched raw-model box. Blue: matched raw box.", font=font(20), fill="white")
    ad.text((15, 76), "Insert-border detections are counted, but do not estimate natural woodland false-alarm rates.", font=font(20), fill="#d1d9e6")
    truth_box = artifact_record["truth"][0]["bbox"]
    for prediction in artifact_record["predictions"]["raw"]:
        b = prediction["bbox"]
        color = "#56b5ff" if iou(b, truth_box) >= .5 else "#ff9b52"
        ad.rectangle([b[0], b[1]+110, b[2], b[3]+110], outline=color, width=3)
        ad.text((b[0], max(110, b[1]+86)), f"{prediction['score']:.3f}", font=font(18), fill=color)
    ad.rectangle([truth_box[0], truth_box[1]+110, truth_box[2], truth_box[3]+110], outline="#52f097", width=3)
    artifact.save(directory / "padding-artifact.png")
    with (directory / "results-table.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["condition", "requestedPixels", "actualMinPixels", "actualMaxPixels", "method", "matched", "targets", "falsePositives", "negativeFrames", "negativeFalsePositives"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in report["rows"]:
            for method in METHODS:
                writer.writerow({"condition": row["id"], "requestedPixels": row["targetPixels"],
                                 "actualMinPixels": row["realizedTargetWidthPixels"]["min"],
                                 "actualMaxPixels": row["realizedTargetWidthPixels"]["max"], "method": method,
                                 **{key: row[method][key] for key in fields[5:]}})
    (directory / "miss-examples.json").write_text(json.dumps({"reportSha256": sha(report_path.read_bytes()),
                "description": "All positive-frame misses; no examples filtered to favor any method.", "records": misses}, indent=2)+"\n", encoding="utf-8")
    return {"contactSheet": str(directory / "evidence-contact-sheet.png"), "misses": len(misses)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(render(parser.parse_args().directory)))
