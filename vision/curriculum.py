"""Explicitly synthetic scale/degradation trials alongside real-context levels."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from .dataset import image_path, sha256, write_json


def make_curriculum(data, manifest_path, output_dir, scales=(1.0, 0.5, 0.25, 0.125), blur_radius=0.6, quality=80, mode="fixed-canvas"):
    import math
    if not scales or len(set(scales)) != len(scales) or any(not math.isfinite(s) or not 0 < s <= 1 for s in scales):
        raise ValueError("Scales must be finite values greater than zero and at most one")
    if not math.isfinite(blur_radius) or not 0 <= blur_radius <= 10 or not 1 <= quality <= 100:
        raise ValueError("Invalid blur or JPEG quality")
    if mode not in ("fixed-canvas", "downsample"):
        raise ValueError("Curriculum mode must be fixed-canvas or downsample")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Curriculum output directory must be empty to avoid mixing experiment runs")
    output_dir.mkdir(parents=True, exist_ok=True)
    originals = [r for r in data["images"] if not r["synthetic"]]
    records = []
    for index, record in enumerate(originals):
        with Image.open(image_path(manifest_path, record)) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            for scale in scales:
                size = max(1, round(source.width * scale)), max(1, round(source.height * scale))
                reduced = source.resize(size, Image.Resampling.LANCZOS)
                if blur_radius:
                    reduced = reduced.filter(ImageFilter.GaussianBlur(blur_radius))
                offset_x = (source.width - size[0]) // 2 if mode == "fixed-canvas" else 0
                offset_y = (source.height - size[1]) // 2 if mode == "fixed-canvas" else 0
                if mode == "fixed-canvas":
                    canvas = Image.new("RGB", source.size, (127, 127, 127))
                    canvas.paste(reduced, (offset_x, offset_y))
                else:
                    canvas = reduced
                variant = deepcopy(record)
                token = str(scale).replace(".", "p")
                name = f"{index:05d}-scale-{token}.jpg"
                path = output_dir / name
                canvas.save(path, quality=quality, optimize=True)
                sx, sy = size[0] / source.width, size[1] / source.height
                variant.update(id=record["id"] + f"-synthetic-{token}", originalId=record["id"], path=name,
                               sha256=sha256(path), width=canvas.width, height=canvas.height, synthetic=True, distanceM=None,
                               transform={"method": "centered scaled scene on fixed canvas + gaussian blur + JPEG" if mode == "fixed-canvas" else "full-frame downsample + gaussian blur + JPEG",
                                          "mode": mode, "contentWidth": size[0], "contentHeight": size[1],
                                          "offsetX": offset_x, "offsetY": offset_y,
                                          "paddingRGB": [127, 127, 127] if mode == "fixed-canvas" else None,
                                          "scaleX": sx, "scaleY": sy, "blurPixels": blur_radius,
                                          "jpegQuality": quality, "physicalDistanceKnown": False})
                for annotation in variant["annotations"]:
                    x1, y1, x2, y2 = annotation["bboxPixelsXYXY"]
                    annotation["bboxPixelsXYXY"] = [x1 * sx + offset_x, y1 * sy + offset_y, x2 * sx + offset_x, y2 * sy + offset_y]
                records.append(variant)
    result = {"schema": data["schema"], "name": data.get("name", "Mushroom dataset") + " synthetic scale stress test",
              "sourceManifest": str(Path(manifest_path).resolve()), "images": records,
              "curriculum": ["Real closeups establish visible mushroom candidate behavior.",
                             "Real habitat/context frames test camouflage and clutter.",
                             "Genuine distant frames must include measured range, blur, occlusion, and explicit negatives.",
                             "This generated set is only a controlled pixel-scale/degradation test; it cannot graduate field autonomy.",
                             "Fixed-canvas mode preserves frame dimensions so model preprocessing does not undo target scale; neutral padding is an artificial cue.",
                             "Downsample mode tests lost resolution; model resizing may restore relative object scale."]}
    write_json(output_dir / "manifest.json", result)
    return result
