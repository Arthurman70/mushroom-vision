"""Publish completed V5 pixel evidence with exact inputs and bounded metadata.

This reads the frozen experiment and licensed public collection manifest. It never
changes weights, reference annotations, experiment code, or deployment state.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.evaluate_full_pixel_v5 import ROOT, OUT, LEVELS, aggregate, digest, read, verify
from experiments.publish_new_fungi_v4 import metadata, write_reproducible

ID = re.compile(r"^[A-Za-z0-9_-]{1,160}$")
SOURCE_KEYS = ("title", "attribution", "license", "sourceUrl", "licenseUrl")
RECORD_KEYS = ("frameId", "cohort", "conditionId", "width", "height", "derivedImageSha256",
               "geometry", "originalTargetPixelsXYXY", "targetPixelsXYXY", "realizedTargetWidthPixels",
               "realizedTargetHeightPixels", "isNegativeControl", "referenceAuthority", "referenceComplete")
GEOMETRY_KEYS = ("sourceSize", "canvasSize", "supersampling", "requestedTargetWidthPixels",
                 "realizedTargetWidthPixels", "realizedTargetHeightPixels", "sceneScaleXY",
                 "sceneBoundsPixelsXYXY", "requestedFocalPositionNormalized", "focalPositionClamped",
                 "realizedFocalPositionNormalized", "patchSupersampledSize", "sceneReEnlarged", "actualDistanceM")
PASS_KEYS = ("kind", "boundsPixelsXYXY", "inputWidth", "inputHeight", "processorScaleXY",
             "selectedTargetCoverageFraction", "selectedTargetInputWidthPixels", "selectedTargetInputHeightPixels")
METRIC_KEYS = ("hits", "targetCount", "unmatchedPositiveBoxes", "extraOverlappingAlerts", "bestIoU",
               "negativeAlerts", "negativeHasAlerts", "humanNegativeFalseAlerts", "aiNegativeOpinionAlerts",
               "humanCompletePositiveFalseAlerts", "matchedPredictionIndex")


def public_geometry(geometry):
    result = {k: copy.deepcopy(geometry[k]) for k in GEOMETRY_KEYS if k in geometry}
    coverage = geometry["processorAndTileCoverage"]
    result["processorAndTileCoverage"] = {
        "intersectingTileCount": coverage["intersectingTileCount"],
        "fullyCoveringTileCount": coverage["fullyCoveringTileCount"],
        "passes": [{k: copy.deepcopy(item[k]) for k in PASS_KEYS} for item in coverage["passes"]],
    }
    return result


def public_frames(references, manifest):
    frames = {f["id"]: f for f in manifest["frames"]}
    sources = {s["id"]: s for s in manifest["sources"]}
    if len(frames) != len(manifest["frames"]) or len(sources) != len(manifest["sources"]):
        raise ValueError("Ambiguous public collection identities")
    result = []
    for reference in references:
        identity = reference["frameId"]
        if not ID.fullmatch(identity) or identity not in frames:
            raise ValueError("Reference is not in the licensed public collection")
        frame = frames[identity]
        if (frame["sha256"] != reference["imageSha256"] or
            (frame["width"], frame["height"]) != (reference["width"], reference["height"])):
            raise ValueError("Published original differs from frozen input bytes or dimensions")
        source = sources[frame["sourceId"]]
        if any(not isinstance(source.get(k), str) or not source[k] for k in SOURCE_KEYS):
            raise ValueError("Published source credit is incomplete")
        result.append({k: copy.deepcopy(reference[k]) for k in (
            "frameId", "cohort", "referenceAuthority", "isNegativeControl", "imageSha256",
            "width", "height", "referenceComplete", "notes", "targetUnit")})
        result[-1]["source"] = {k: source[k] for k in SOURCE_KEYS}
    return result


def public_record(record, job):
    for key in ("frameId", "cohort", "conditionId", "imagePath", "derivedImageSha256", "sourceImageSha256",
                "width", "height", "geometry", "originalTargetPixelsXYXY", "targetPixelsXYXY",
                "realizedTargetWidthPixels", "realizedTargetHeightPixels", "isNegativeControl",
                "referenceAuthority", "referenceComplete"):
        if record[key] != job[key]: raise ValueError("Inference input differs from frozen job: " + key)
    if set(record["methods"]) != {"raw", "tiles"} or record["uniqueModelForwards"] != 5:
        raise ValueError("Incomplete paired comparison")
    result = {k: copy.deepcopy(record[k]) for k in RECORD_KEYS}
    result["geometry"] = public_geometry(record["geometry"])
    result["assetName"] = None if record["conditionId"] == "native" else f"pixel-view-{record['frameId']}--{record['conditionId']}.png"
    result["methods"] = {}
    for name in ("raw", "tiles"):
        method = record["methods"][name]
        result["methods"][name] = {
            "candidates": [{"bbox": list(c["bbox"]), "score": c["score"]} for c in method["candidates"]],
            "metrics": {k: copy.deepcopy(method["metrics"][k]) for k in METRIC_KEYS if k in method["metrics"]},
            "seconds": method["seconds"], "modelPasses": method["modelPasses"],
            "sharedContextPasses": method["sharedContextPasses"],
        }
    return result


def public_recovery(report):
    recovery = report.get("recovery")
    if recovery is None: return None
    retained, recovered, total = recovery["reusedScenes"], recovery["newScenes"], len(report["records"])
    if (retained + recovered != total or recovery["uniquePersistedModelForwards"] != total * 5 or
        recovery["totalAttemptedModelForwards"] is not None):
        raise ValueError("Recovery counts or unknown attempted-work disclosure differ")
    return {"occurred": True, "reason": "file-lock", "retainedScenes": retained, "recoveredScenes": recovered,
        "totalScenes": total, "uniquePersistedModelPasses": recovery["uniquePersistedModelForwards"],
        "totalAttemptedModelPasses": None, "totalAttemptedWorkKnown": False,
        "notes": recovery["disclosure"] + " Elapsed wall time includes the interruption and recovery."}


def public_result(report, protocol, manifest):
    if (report.get("status") != "completed" or report.get("inputIntegrityVerified") is not True or
        report.get("trainingPerformed") is not False or report.get("weightsChanged") is not False or
        report.get("progress") != {"completed": 505, "total": 505, "scenesCompleted": 101, "scenesTotal": 101}):
        raise ValueError("Only the completed unchanged-weight 505-forward run can be published")
    jobs = {(j["frameId"], j["conditionId"]): j for j in protocol["jobs"]}
    records = {(r["frameId"], r["conditionId"]): r for r in report["records"]}
    if len(jobs) != 101 or len(records) != 101 or len(report["records"]) != 101 or set(records) != set(jobs):
        raise ValueError("Missing or repeated comparison inputs")
    if report["frames"] != protocol["references"] or report["cohorts"] != aggregate(report["records"]):
        raise ValueError("Reported references or aggregate counts differ")
    for key, value in protocol["expectedModel"].items():
        if report["model"].get(key) != value: raise ValueError("Unexpected model identity")
    frames = public_frames(protocol["references"], manifest)
    cohort_rows = copy.deepcopy(report["cohorts"])
    for cohort in cohort_rows:
        refs = [f for f in protocol["references"] if f["cohort"] == cohort["id"] and not f["isNegativeControl"]]
        cohort["sourceGroupCount"] = len({f["sourceGroupId"] for f in refs})
        cohort["notes"] = ("Three annotated targets share one video. One human absence control was tested at native resolution only."
            if cohort["id"] == "human-video" else
            "Nine approximate selected regions from nine photographs; the regions include individual bodies and clusters. Four older AI video absence opinions were tested only at native resolution. Unmatched boxes on incomplete photographs are not established false positives.")
    result = {"schema": "mushroom-vision.public-pixels.v1", "completedAt": report["completedAt"],
        "status": "completed", "frames": frames,
        "records": [public_record(record, jobs[(record["frameId"], record["conditionId"])]) for record in report["records"]],
        "cohorts": cohort_rows,
        "protocol": {"detectorName": report["model"]["model"], "modelSha256": report["model"]["resolvedRevision"].removeprefix("sha256:"),
            "protocolSha256": report["protocolSha256"], "scoreThreshold": protocol["scoreThreshold"],
            "iouThreshold": protocol["iouThreshold"], "canvasWidth": 1920, "canvasHeight": 1080,
            "focalLengthPixels": 1380, "assumedTargetWidthMetres": .05, "weightsChanged": False,
            "trainingPerformed": False, "actualDistanceM": None, "workers": 2, "torchThreadsPerWorker": 4,
            "uniquePersistedModelPasses": 505, "elapsedWallSeconds": report["elapsedWallSeconds"],
            "summedSceneInferenceSeconds": report["summedSceneInferenceSeconds"],
            "transformDescription": "Each whole source scene shrinks onto a 1920 by 1080 RGB127 gray canvas at a fixed, hash-derived position. Sixteen-times fractional coverage is integrated to the saved pixels. The scene is never re-enlarged to native size. Original inputs retain their native dimensions. Both methods receive the identical saved bytes; tiled scanning adds four overlapping tiles to the shared whole-frame context pass.",
            "notes": "Development stress test, not independent field accuracy or calibrated drone range. The illustrative 10/20/50/100 m labels assume a 5 cm target and a 1380 px focal length; actual organism sizes, camera intrinsics and distances were not measured. A selected region can be a cluster. The previous round-4 downsample-then-restore information-loss test is different and must not be merged with this curve. Two concurrent CPU workers used four Torch threads each; method timings include contention and cannot be compared directly with earlier sequential CPU timings. Tiled timing includes the shared raw pass, so adding raw and tiled time double-counts it. All five negative controls are native-only, separated by human versus AI authority. Extra selected-region overlaps are counted as unmatched alerts; no accuracy conclusion is assigned to unmarked regions in incomplete photographs. An independent read-only audit found broad detector alerts in tiles containing only uniform RGB127 padding. Synthetic extra-box totals therefore include padding artifacts and are not field false-positive rates; no alert was removed or rescored after observing this behavior.",
            "limitations": list(protocol["limitations"])}}
    recovery = public_recovery(report)
    if recovery is not None: result["protocol"]["recovery"] = recovery
    return result


def copy_exact(source, target, expected):
    if digest(source) != expected: raise ValueError("Saved synthetic input bytes changed")
    if target.exists():
        if digest(target) != expected: raise FileExistsError("Different public pixel asset exists")
        return
    with source.open("rb") as src, target.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if digest(target) != expected: raise ValueError("Copied asset hash mismatch")


def validate_receipt(receipt, protocol_hash, report_hash):
    if (receipt.get("schema") != "mushroom-vision.full-pixel-independent-verification.v1" or
        receipt.get("status") != "verified" or receipt.get("protocolSha256") != protocol_hash or
        receipt.get("reportSha256") != report_hash or receipt.get("scenes") != 101 or
        receipt.get("uniqueModelForwards") != 505 or receipt.get("uniquePredictionFiles") != 101 or
        receipt.get("trainingPerformed") is not False or receipt.get("weightsChanged") is not False):
        raise ValueError("Matching independent completed-run verification is required")


def publish(output=OUT, archive=ROOT / "dist/community-round4-append.tar.gz", public_dir=ROOT / "community/public"):
    output, public_dir = Path(output), Path(public_dir)
    protocol_path, report_path = output / "protocol.json", output / "report.json"
    protocol, report = read(protocol_path), read(report_path)
    receipt_path = output / "independent-verification.json"
    receipt = read(receipt_path)
    validate_receipt(receipt, digest(protocol_path), digest(report_path))
    if receipt["verifierSha256"] != digest(ROOT / "experiments/verify_full_pixel_v5.py"):
        raise ValueError("Independent verifier code changed since its receipt")
    if report.get("recovery") is not None:
        recovery_path = output / "recovery-receipt.json"
        if digest(recovery_path) != report["recovery"]["receiptSha256"]:
            raise ValueError("Recovery receipt differs from completed report")
    if (report["protocolSha256"] != digest(protocol_path) or
        read(output / "protocol-sha256.json")["sha256"] != report["protocolSha256"]):
        raise ValueError("Frozen protocol changed")
    verify(protocol)
    for record in report["records"]:
        if digest(ROOT / record["predictionPath"]) != record["predictionSha256"]:
            raise ValueError("Saved model prediction changed")
    manifest, _ = metadata(archive)
    result = public_result(report, protocol, manifest)
    result["protocol"].update(independentVerificationSha256=digest(receipt_path), reportSha256=digest(report_path),
        verificationScope=receipt["scope"])
    public_dir.mkdir(parents=True, exist_ok=True)
    copied, total_bytes = 0, 0
    for record, public in zip(report["records"], result["records"]):
        if not public["assetName"]: continue
        target = public_dir / public["assetName"]
        copy_exact(ROOT / record["imagePath"], target, record["derivedImageSha256"])
        copied += 1; total_bytes += target.stat().st_size
    if copied != 84: raise ValueError("Unexpected number of synthetic assets")
    path = public_dir / "testing-pixels.json"
    write_reproducible(path, result)
    summary = {"publicJson": str(path), "publicSha256": digest(path), "jsonBytes": path.stat().st_size,
               "frames": len(result["frames"]), "records": len(result["records"]),
               "syntheticImages": copied, "imageBytes": total_bytes, "reportSha256": digest(report_path)}
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--archive", type=Path, default=ROOT / "dist/community-round4-append.tar.gz")
    parser.add_argument("--public-dir", type=Path, default=ROOT / "community/public")
    args = parser.parse_args(); publish(args.output, args.archive, args.public_dir)
