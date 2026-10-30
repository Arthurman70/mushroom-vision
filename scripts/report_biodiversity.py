"""Write a compact acquisition report without exposing images, locations or reviewers."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def report(root=Path(".")):
    root = root.resolve()
    fungi_path = root / "data/biodiversity-fungi-v1/manifest.json"
    wildlife_path = root / "data/wildlife-context-v1/manifest.json"
    fungi = json.loads(fungi_path.read_text(encoding="utf-8"))
    wildlife = json.loads(wildlife_path.read_text(encoding="utf-8"))
    for path in (fungi_path, wildlife_path):
        validation = json.loads(path.with_name("validation.json").read_text(encoding="utf-8"))
        if validation.get("valid") is not True:
            raise ValueError("A verified frozen collection is required")
        freeze = json.loads(path.with_name("manifest-sha256.json").read_text(encoding="utf-8"))
        if hashlib.sha256(path.read_bytes()).hexdigest() != freeze["sha256"]:
            raise ValueError("Frozen manifest changed")
    new = [r for r in fungi["images"] if r.get("cohort") == "new_biodiversity_v1"]
    by_class = []
    for info in fungi["classes"]:
        rows = [r for r in fungi["images"] if r["classId"] == info["classId"]]
        by_class.append({"classId": info["classId"], "groupId": info["groupId"], "images": len(rows),
                         "newImages": sum(r.get("cohort") == "new_biodiversity_v1" for r in rows),
                         "splits": dict(Counter(r["split"] for r in rows)),
                         "sourceGroups": {s: len({r["splitGroupId"] for r in rows if r["split"] == s}) for s in ("train", "validation", "test")}})
    result = {"schema": "mushroom-vision.acquisition-report.v1", "fungi": {
        "manifestSha256": hashlib.sha256(fungi_path.read_bytes()).hexdigest(), "totalImages": len(fungi["images"]),
        "newImages": len(new), "newImageBytes": sum(r["bytes"] for r in new), "newTaxa": len({r["taxon"]["id"] for r in new}),
        "totalTaxa": len({r["taxon"]["id"] for r in fungi["images"]}), "newImageObservers": len({r["observerId"] for r in new}),
        "previouslyUnseenObservers": len({r["observerId"] for r in new} - {r["observerId"] for r in fungi["images"] if r.get("cohort") != "new_biodiversity_v1"}),
        "newPhotoLicenses": dict(Counter(r["photoLicense"]["code"] for r in new)), "classes": by_class,
        "verification": "Weak source taxonomy only; no new human-verified boxes or physical distances"},
        "wildlife": wildlife["summary"], "newTotalImages": len(new) + len(wildlife["images"]),
        "newTotalImageBytes": sum(r["bytes"] for r in new) + wildlife["summary"]["bytes"],
        "limits": ["No accuracy improvement is claimed by data acquisition alone.",
                   "Wildlife animal-empty frames are not verified mushroom-negative examples.",
                   "Inherited test images remain reused evaluation; no new independent field validation."]}
    output = root / "experiments/results/biodiversity-v1"
    output.mkdir(parents=True, exist_ok=True)
    (output / "acquisition-report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("newTotalImages", "newTotalImageBytes")}))
    return result


if __name__ == "__main__":
    report()
