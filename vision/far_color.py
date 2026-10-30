"""Bounded local-color investigation cues; never a species classifier.

All coordinates refer to the untouched, EXIF-oriented RGB source image. The
enhanced view is a separate in-memory aid. No scale, range, or pose is inferred.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

METHOD = "lab-local-contrast-v1"
MEANING = "Local color contrast, not mushroom/species probability"
MAX_PIXELS = 20_000_000


def rgb_image(image):
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Color analysis is limited to 20 megapixels")
    return ImageOps.exif_transpose(image).convert("RGB")


def lab(rgb):
    """sRGB (D65) to CIE Lab; Delta E is Euclidean CIE76, not probability."""
    a = np.asarray(rgb, dtype=np.float32) / 255.0
    a = np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055) ** 2.4)
    xyz = a @ np.array([[.4124564, .2126729, .0193339],
                        [.3575761, .7151522, .1191920],
                        [.1804375, .0721750, .9503041]], dtype=np.float32)
    xyz /= np.array([.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], axis=-1)


def enhance(image):
    """Fixed local luminance/chroma gain; preserve the original image object."""
    original = rgb_image(image)
    a = np.asarray(original, dtype=np.float32)
    smooth = np.asarray(original.filter(ImageFilter.GaussianBlur(4)), dtype=np.float32)
    residual = a - smooth
    luma = (residual * np.array([.2126, .7152, .0722])).sum(axis=-1, keepdims=True)
    amplified = a + .75 * luma + 2.0 * (residual - luma)
    return Image.fromarray(np.rint(np.clip(amplified, 0, 255)).astype(np.uint8), "RGB")


def contrast_map(image):
    original = rgb_image(image)
    local = original.filter(ImageFilter.GaussianBlur(8))
    return np.linalg.norm(lab(original) - lab(local), axis=-1)


def color_evidence(delta, box, enhanced=False):
    w, h = delta.shape[1], delta.shape[0]
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("Invalid color evidence box")
    x1, y1 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x2, y2 = min(w, math.ceil(box[2])), min(h, math.ceil(box[3]))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Empty color evidence box")
    area = delta[y1:y2, x1:x2]
    maximum = float(area.max())
    return {"method": METHOD, "channels": ["L*", "a*", "b*"],
            "maxDeltaE": maximum, "meanDeltaE": float(area.mean()),
            "contrastScore": float(math.tanh(maximum / 20)),
            "enhancedViewUsed": bool(enhanced), "scoreMeaning": MEANING}


def proposals(image, max_proposals=12, min_delta_e=8.0):
    if not isinstance(max_proposals, int) or not 1 <= max_proposals <= 50:
        raise ValueError("Proposal budget must be an integer from 1 to 50")
    if not math.isfinite(min_delta_e) or not 0 < min_delta_e <= 200:
        raise ValueError("Delta E threshold must be finite and in (0, 200]")
    delta = contrast_map(image)
    work = delta.copy()
    h, w = delta.shape
    found = []
    for _ in range(max_proposals):
        y, x = np.unravel_index(int(work.argmax()), work.shape)
        peak = float(work[y, x])
        if peak < min_delta_e:
            break
        box = [max(0, int(x) - 4), max(0, int(y) - 4),
               min(w, int(x) + 5), min(h, int(y) + 5)]
        evidence = color_evidence(delta, box)
        found.append({"label": "Unconfirmed color anomaly", "cueType": "color_salience",
                      "bboxPixelsXYXY": box, "pointPixelsXY": [int(x), int(y)],
                      "contrastScore": evidence["contrastScore"], "deltaE": peak,
                      "colorEvidence": evidence, "colorOnly": True,
                      "status": "UNLOCATED", "reviewed": False, "species": None,
                      "poseKnown": False, "depthKnown": False})
        work[max(0, y - 16):min(h, y + 17), max(0, x - 16):min(w, x + 17)] = -1
    return found


def analyze(path, max_proposals=12, min_delta_e=8.0):
    path = Path(path)
    if path.stat().st_size > 50_000_000:
        raise ValueError("Image file exceeds 50 MB")
    with Image.open(path) as opened:
        im = rgb_image(opened)
    return {"schema": "mycoscout.color-proposals.v1", "colorOnly": True,
            "image": {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "width": im.width, "height": im.height,
                      "pixelConvention": "EXIF-oriented RGB image; x right, y down"},
            "detector": {"model": METHOD, "resolvedRevision": "v1", "maxProposals": max_proposals,
                         "minDeltaE": min_delta_e, "scoreMeaning": MEANING},
            "candidates": proposals(im, max_proposals, min_delta_e),
            "limitations": ["Color contrasts also occur on leaves, water, bark and shadows.",
                            "No mushroom presence, species, metric range, or pose is established.",
                            "Only an investigation cue; human confirmation requires original RGB evidence."]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-proposals", type=int, default=12)
    p.add_argument("--min-delta-e", type=float, default=8)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; choose a new filename")
    result = analyze(a.image, a.max_proposals, a.min_delta_e)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(f"Saved {len(result['candidates'])} unconfirmed color cues to {a.output}")


if __name__ == "__main__":
    main()
