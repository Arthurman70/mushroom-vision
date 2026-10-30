"""Local audit contact sheet: original context beside exact proposal crop."""
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from adapters.merge_video_collections import read_collection
from vision.proposal_verifier_pilot import native_crop


def main():
    root, _, manifest = read_collection(ROOT / "data/video-discovery-v1")
    frames = {frame["id"]: frame for frame in manifest["frames"]}
    audit = json.loads((ROOT / "data/ai-video-review-v1/assistant-negative-crop-audit.json").read_text(encoding="utf-8-sig"))
    sheet = Image.new("RGB", (1400, 410 * len(audit["reviews"])), "#111827")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 21)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 17)
    for index, review in enumerate(audit["reviews"]):
        frame = frames[review["frameId"]]
        image = Image.open(root / frame["path"]).convert("RGB")
        top = index * 410
        decision = "AUDITED NEGATIVE" if review["negative"] else "EXCLUDED: AMBIGUOUS CONTENTS"
        draw.text((15, top + 7), review["frameId"] + " / " + review["proposalBoxId"] + " / " + decision, font=font, fill="white")
        context = ImageOps.contain(image, (620, 330))
        scale = context.width / image.width
        ImageDraw.Draw(context).rectangle([value * scale for value in review["bbox"]], outline="#ff00ff", width=3)
        sheet.paste(context, (15, top + 42))
        crop = ImageOps.contain(native_crop(image, review["bbox"]), (720, 330))
        sheet.paste(crop, (660, top + 42))
        draw.text((15, top + 376), "Original: " + frame["path"] + " | native bbox " + str([round(value, 1) for value in review["bbox"]]), font=small, fill="#cbd5e1")
    destination = ROOT / "data/ai-video-review-v1/negative-crops-contact-sheet.jpg"
    sheet.save(destination, quality=94)
    print(destination)


if __name__ == "__main__":
    main()
