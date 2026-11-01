"""Read back the public V5 report and every displayed test image after deployment."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "community/public"
BASE = "https://thegreatlakesgleaner.press/mushroom-lab/"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def fetch(name):
    request = Request(BASE + name, headers={"User-Agent": "MushroomVision-PublicEvidenceCheck/1.0"})
    with urlopen(request, timeout=45) as response:
        if response.status != 200 or not response.url.startswith(BASE):
            raise ValueError("Unexpected public response: " + name)
        raw = response.read(8 * 1024 * 1024 + 1)
        if len(raw) > 8 * 1024 * 1024:
            raise ValueError("Public evidence exceeds asset limit")
        return raw


def main():
    raw = (PUBLIC / "testing-pixels.json").read_bytes()
    report = json.loads(raw)
    if report.get("status") != "completed" or len(report.get("records", [])) != 101:
        raise ValueError("Completed local evidence required")
    if fetch("testing-pixels.json?v=" + digest(raw)) != raw:
        raise ValueError("Public JSON differs from verified local evidence")

    for name in ("testing-pixels.html", "testing-pixels.js", "testing-pixels.css", "testing-round4.html", "index.html", "ai-contribute.html", "llms.txt"):
        local = (PUBLIC / name).read_bytes()
        if fetch(name + "?v=" + digest(local)) != local:
            raise ValueError("Public page differs from local asset: " + name)

    jobs = [(r["assetName"], r["derivedImageSha256"], r["width"], r["height"]) for r in report["records"] if r["assetName"]]
    jobs += [("images/" + f["frameId"], f["imageSha256"], f["width"], f["height"]) for f in report["frames"]]
    if len(jobs) != 101 or len({j[0] for j in jobs}) != 101:
        raise ValueError("Expected 84 exact synthetic inputs and 17 originals")

    def check(job):
        name, expected, width, height = job
        data = fetch(name + "?v=" + expected)
        if digest(data) != expected:
            raise ValueError("Public image hash differs: " + name)
        with Image.open(io.BytesIO(data)) as image:
            if image.size != (width, height):
                raise ValueError("Public image dimensions differ: " + name)
            image.verify()
        return {"asset": name, "sha256": expected, "bytes": len(data), "width": width, "height": height}

    with ThreadPoolExecutor(max_workers=4) as pool:
        assets = list(pool.map(check, jobs))
    receipt = {"schema": "mushroom-vision.public-pixel-readback.v1", "status": "verified",
               "checkedAt": datetime.now(timezone.utc).isoformat(), "url": BASE + "testing-pixels.html",
               "publicReportSha256": digest(raw), "protocolSha256": report["protocol"]["protocolSha256"],
               "syntheticImages": 84, "originalImages": 17, "pageAssets": 7, "assets": assets}
    output = ROOT / "data/community-pixels-v5"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = output / ("public-readback-" + stamp + ".json")
    with path.open("x", encoding="utf8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({k: v for k, v in receipt.items() if k != "assets"} | {"receipt": str(path)}))


if __name__ == "__main__":
    main()
