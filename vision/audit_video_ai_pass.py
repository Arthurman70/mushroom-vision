"""Read-only integrity audit of a completed sampled-video AI pass."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path

from vision.assistant_video_pilot import checked_box, collect_holdouts
from vision.evaluate_video_ai import write_new
from vision.video_ai_pass import ROOT, SCHEMA, load_inputs, sha


def audit_main_coverage(directory, main):
    """Validate the public collection independently of research/global progress."""
    directory = Path(directory)
    rows, sources, manifests = load_inputs([("main", main)])
    inputs = json.loads((directory / "inputs.json").read_text())
    if len(rows) != 372 or len(sources) != 7 or manifests[0] not in inputs["collections"]:
        raise ValueError("Expected immutable 372-frame public collection")
    expected = {row["key"] for row in rows}
    actual = {path.stem for path in (directory / "frames").glob("main--*.json")}
    if actual != expected or len(expected) != 372:
        raise ValueError("Public collection still has missing or additional records")
    counts, candidates = Counter(), 0
    for row in rows:
        frame, source = row["frame"], row["source"]
        result = json.loads((directory / "frames" / (row["key"] + ".json")).read_text())
        if (result.get("schema") != SCHEMA or result.get("collection") != "main" or result.get("status") != "completed" or
                result.get("frameId") != frame["id"] or result.get("imageSha256") != frame["sha256"] or
                result.get("sourceId") != source["id"] or result.get("sourceGroupId") != source["sourceGroupId"] or
                result.get("videoHash") != source["videoHash"] or result.get("humanVerified") is not False or
                result.get("trainingAllowed") is not False or sha(row["path"].read_bytes()) != frame["sha256"]):
            raise ValueError("Public frame failed completion, identity or native hash check: " + frame["id"])
        counts[source["id"]] += 1
        candidates += len(result["boxes"])
    return {"schema": "mycoscout.ai-public-coverage.v1", "status": "passed", "collection": "main", "frames": 372,
            "sourceVideos": 7, "candidateBoxes": candidates, "missingFrames": 0, "failedFrames": 0,
            "duplicateIds": 0, "allOriginalImageHashesVerified": True, "allRecordIdentitiesVerified": True,
            "sourceManifestSha256": manifests[0]["manifestSha256"], "sourceFrameCounts": dict(counts),
            "researchFramesIncluded": 0, "globalCounterUsedAsEvidence": False,
            "notice": "Public/main subset alone is complete. Research processing remains separate; AI proposals are unverified."}


def audit(directory, main, research, learning, expected_frames=432, expected_sources=9):
    directory = Path(directory)
    rows, sources, manifests = load_inputs([("main", main), ("research", research)])
    report_bytes = (directory / "report.json").read_bytes()
    report = json.loads(report_bytes)
    status = json.loads((directory / "status.json").read_text())
    identity = json.loads((directory / "inputs.json").read_text())
    if (len(rows) != expected_frames or len(sources) != expected_sources or report.get("status") != "completed" or
            status.get("status") != "completed" or report.get("failedFrames") != 0 or status.get("failedFrames") != 0 or
            report.get("sourceCollections") != manifests or identity.get("collections") != manifests):
        raise ValueError("Pass is incomplete or original collection identity changed")
    expected_keys = {row["key"] for row in rows}
    actual_keys = {path.stem for path in (directory / "frames").glob("*.json")}
    if len(expected_keys) != len(rows) or actual_keys != expected_keys:
        raise ValueError("Missing, additional or duplicate per-frame files")
    if len(report.get("frames", [])) != len(rows):
        raise ValueError("Aggregate report frame count mismatch")
    aggregate = {(row["collection"], row["frameId"]): row for row in report["frames"]}
    if len(aggregate) != len(rows):
        raise ValueError("Aggregate report contains repeated frame identities")
    holdouts, versions = collect_holdouts(learning)
    frozen_keys = {key for row in holdouts for key in row["groupKeys"]}
    counts, proposals = Counter(), Counter()
    native_hashes = set()
    for row in rows:
        frame, source = row["frame"], row["source"]
        record = json.loads((directory / "frames" / (row["key"] + ".json")).read_text())
        native_hashes.add(frame["sha256"])
        expected_evaluation = source.get("evaluationOnly") is not False or source.get("licenseAllowsTraining") is not True or frame.get("evaluationOnly") is True
        expected = {"schema": SCHEMA, "status": "completed", "collection": row["collection"], "frameId": frame["id"],
                    "sourceId": source["id"], "sourceGroupId": source["sourceGroupId"],
                    "videoHash": source.get("videoHash", source.get("sha256")), "imageSha256": frame["sha256"],
                    "timestampSeconds": frame["timestampSeconds"], "width": frame["width"], "height": frame["height"],
                    "evaluationOnly": expected_evaluation, "sourceLicenseAllowsTraining": source.get("licenseAllowsTraining") is True,
                    "humanVerified": False, "labelsVerified": False, "trainingAllowed": False,
                    "originalCoordinates": True, "actuationEnabled": False, "distanceMeters": None, "humanMatches": None}
        if any(record.get(key) != value for key, value in expected.items()) or sha(row["path"].read_bytes()) != frame["sha256"]:
            raise ValueError("Per-frame source, geometry, hash or nonhuman authority mismatch: " + row["key"])
        if aggregate[(row["collection"], frame["id"])] != record:
            raise ValueError("Aggregate report differs from immutable per-frame record")
        if len({box["id"] for box in record["boxes"]}) != len(record["boxes"]):
            raise ValueError("Duplicate proposal box IDs")
        for box in record["boxes"]:
            checked_box(box["bbox"], frame)
            if (type(box.get("score")) not in {float, int} or not math.isfinite(box["score"]) or not 0 <= box["score"] <= 1 or
                    box.get("humanVerified") is not False or box.get("trainingAllowed") is not False):
                raise ValueError("Invalid candidate score or authority")
        key = row["collection"], source["id"]
        counts[key] += 1
        proposals[key] += len(record["boxes"])
    total_proposals = sum(proposals.values())
    if (report.get("completedFrames") != len(rows) or report.get("processedFrames") != len(rows) or
            report.get("sampledFrames") != len(rows) or report.get("sourceVideos") != len(sources) or
            report.get("candidateBoxes") != total_proposals or status.get("candidateBoxes") != total_proposals):
        raise ValueError("Aggregate counts do not match verified records")
    breakdown = []
    for source in sources:
        reasons = []
        if source.get("licenseAllowsTraining") is not True:
            reasons.append("source_license_excludes_training")
        if source.get("evaluationOnly") is not False:
            reasons.append("evaluation_only_collection")
        if "source:" + source["sourceGroupId"] in frozen_keys:
            reasons.append("frozen_human_holdout_source_group")
        key = source["collection"], source["id"]
        breakdown.append({"collection": source["collection"], "sourceId": source["id"], "sourceGroupId": source["sourceGroupId"],
                          "frames": counts[key], "candidateBoxes": proposals[key], "trainingEligibleSource": not reasons,
                          "trainingExclusions": reasons, "allProposalsRequireHumanReview": True})
    return {"schema": "mycoscout.ai-video-integrity-audit.v1", "status": "passed", "frames": len(rows), "sourceVideos": len(sources),
            "uniqueFrameIdentities": len(expected_keys), "uniqueOriginalImageHashes": len(native_hashes),
            "candidateBoxes": total_proposals, "missingFrames": 0, "failedFrames": 0, "duplicateIds": 0,
            "sourceManifestHashesVerified": True, "allOriginalImageHashesVerified": True,
            "allRecordProvenanceVerified": True, "aggregateMatchesImmutableFrameFiles": True,
            "humanGradesModified": False, "modelActivationPerformed": False, "reportSha256": sha(report_bytes),
            "frozenHumanDatasets": versions, "sources": breakdown,
            "notice": "One inference pass over every imported sampled frame. This audit establishes integrity and coverage, not mushroom recognition accuracy."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/ai-video-review-v1")
    parser.add_argument("--main", type=Path, default=ROOT / "data/video-discovery-v1")
    parser.add_argument("--research", type=Path, default=ROOT / "data/morel-research-frames-v1")
    parser.add_argument("--learning", type=Path, default=ROOT / "data/learning/datasets")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-only", action="store_true")
    args = parser.parse_args()
    report = audit_main_coverage(args.directory, args.main) if args.main_only else audit(args.directory, args.main, args.research, args.learning)
    write_new(args.output, report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
