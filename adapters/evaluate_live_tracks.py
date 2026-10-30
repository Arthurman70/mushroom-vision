"""Compare proposed video tracks to complete, explicitly linked human annotations.

Only exact source video/frame hashes and timestamps can join. Partial, uncertain,
unreviewed and synthetic examples do not become accuracy ground truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import sys

from vision.live_mapping import iou


def grade_records(database):
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        tracks = {row[0]: json.loads(row[1]) for row in connection.execute("SELECT id,data FROM grade_tracks")}
        rows = connection.execute("""SELECT f.context,r.data FROM grade_frames f JOIN grade_reviews r
            ON f.collection_id=r.collection_id AND f.frame_id=r.frame_id WHERE r.revision=(
            SELECT MAX(r2.revision) FROM grade_reviews r2 WHERE r2.collection_id=r.collection_id AND r2.frame_id=r.frame_id)""")
        return [{"context": json.loads(context), "review": json.loads(review)} for context, review in rows], tracks
    finally:
        connection.close()


def assignment(human, predictions, threshold=.5):
    """Maximum-cardinality one-to-one IoU matches; no duplicate credit per box."""
    adjacency = {h: sorted([p for p in range(len(predictions)) if iou(box["bboxPixelsXYXY"], predictions[p]["bboxPixelsXYXY"]) >= threshold],
                            key=lambda p: iou(box["bboxPixelsXYXY"], predictions[p]["bboxPixelsXYXY"]), reverse=True)
                 for h, box in enumerate(human)}
    owners = {}
    def augment(h, visited):
        for p in adjacency[h]:
            if p in visited: continue
            visited.add(p)
            if p not in owners or augment(owners[p], visited):
                owners[p] = h
                return True
        return False
    for h in range(len(human)): augment(h, set())
    return {h: p for p, h in owners.items()}


def width_bin(width):
    for upper, name in ((4, "under4"), (8, "4to8"), (16, "8to16"), (32, "16to32"), (64, "32to64")):
        if width < upper: return name
    return "64plus"


def evaluate(packets, records, tracks):
    keyed = {}
    model_keys = set()
    for packet in packets:
        provenance = packet.get("provenance", {})
        if packet.get("detectorStatus") != "completed" or provenance.get("synthetic") is True or provenance.get("syntheticFixture"):
            continue
        model_keys.add(json.dumps(packet.get("detector", {}).get("model"), sort_keys=True))
        key = (packet["frameId"], packet["image"]["sha256"], provenance.get("sourceId"), provenance.get("videoHash"), packet["timestampSeconds"])
        if key in keyed:
            raise ValueError("Multiple replay results for the same observation; evaluate one replay session at a time")
        keyed[key] = packet
    if len(model_keys) > 1:
        raise ValueError("Mixed detector models/settings cannot form one tracking evaluation")
    bins = {name: {"humanBoxes": 0, "matchedBoxes": 0, "recall": None} for name in ("under4", "4to8", "8to16", "16to32", "32to64", "64plus")}
    observations = defaultdict(list)
    eligible, incomplete, no_run, true_positive, predictions_total, truth_total = 0, 0, 0, 0, 0, 0
    for row in records:
        ctx, review = row["context"], row["review"]
        if review.get("complete") is not True or review.get("humanVerified") is not True or review.get("status") not in {"visible", "absent"}:
            incomplete += 1; continue
        key = (ctx["frameId"], ctx["sha256"], ctx["sourceId"], ctx["videoHash"], ctx["timestampSeconds"])
        packet = keyed.get(key)
        if packet is None:
            no_run += 1; continue
        boxes, predictions = review.get("boxes", []), packet["detections"]
        matches = assignment(boxes, predictions)
        eligible += 1; true_positive += len(matches); predictions_total += len(predictions); truth_total += len(boxes)
        for index, box in enumerate(boxes):
            width = box["bboxPixelsXYXY"][2] - box["bboxPixelsXYXY"][0]
            bucket = bins[width_bin(width)]; bucket["humanBoxes"] += 1; bucket["matchedBoxes"] += index in matches
            human_track = tracks.get(box.get("trackId"))
            if not human_track or human_track.get("sourceId") != ctx["sourceId"] or human_track.get("videoHash") != ctx["videoHash"] or human_track.get("humanCreated") is not True:
                continue
            predicted = predictions[matches[index]] if index in matches else None
            observations[human_track["id"]].append({"frameId": ctx["frameId"], "timestampSeconds": ctx["timestampSeconds"],
                "humanPixelWidth": width, "detected": predicted is not None,
                "machineTrackId": predicted.get("trackId") if predicted else None,
                "machinePixelWidth": predicted["bboxPixelsXYXY"][2] - predicted["bboxPixelsXYXY"][0] if predicted else None})
    track_reports = []
    for track_id, values in observations.items():
        values.sort(key=lambda row: row["timestampSeconds"])
        detected = [v for v in values if v["detected"]]
        matched_ids = [v["machineTrackId"] for v in detected if v["machineTrackId"]]
        switches = sum(a != b for a, b in zip(matched_ids, matched_ids[1:]))
        segments = sum(v["detected"] and (index == 0 or not values[index - 1]["detected"]) for index, v in enumerate(values))
        track_reports.append({"humanTrackId": track_id, "label": tracks[track_id].get("label"),
            "reviewedVisibleFrames": len(values), "matchedFrames": len(detected),
            "firstHumanVisible": values[0], "firstMachineMatch": detected[0] if detected else None,
            "delayWithinReviewedFramesSeconds": detected[0]["timestampSeconds"] - values[0]["timestampSeconds"] if detected else None,
            "identitySwitches": switches if len(matched_ids) >= 2 else None,
            "matchedSegments": segments, "observations": values})
    for bucket in bins.values():
        if bucket["humanBoxes"]: bucket["recall"] = bucket["matchedBoxes"] / bucket["humanBoxes"]
    return {"schema": "mycoscout.track-evaluation.v1", "status": "awaiting_complete_labels" if not eligible else "awaiting_human_track_links" if not track_reports else "evaluated",
        "eligibleFrames": eligible, "excludedIncompleteOrUncertainFrames": incomplete, "completeFramesWithoutExactRun": no_run,
        "linkedHumanTracks": len(track_reports), "detection": {"truePositive": true_positive, "falsePositive": predictions_total - true_positive,
            "falseNegative": truth_total - true_positive, "precision": true_positive / predictions_total if predictions_total else None,
            "recall": true_positive / truth_total if truth_total else None, "iouThreshold": .5},
        "pixelWidthBins": bins, "tracks": track_reports,
        "identitySwitches": sum(t["identitySwitches"] for t in track_reports if t["identitySwitches"] is not None) if any(t["identitySwitches"] is not None for t in track_reports) else None,
        "limitations": ["Earliest visibility is only among completely reviewed sampled frames, not the true continuous-video onset.",
            "Human links are ground truth assertions; automatic tracks are never used to manufacture labels.",
            "Pixel size does not establish physical distance. No species or flight performance is measured."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--grades-db", type=Path, default=Path("data/video-grades/video-grades.sqlite"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.packets.stat().st_size > 256 * 1024 ** 2:
        parser.error("Replay packet file exceeds 256 MiB")
    packets = []
    with args.packets.open(encoding="utf8") as stream:
        for line in stream:
            if len(line) > 16 * 1024 ** 2 or len(packets) >= 10000:
                raise ValueError("Replay packet count/size bound exceeded")
            packet = json.loads(line)
            packet.get("image", {}).pop("imageBase64", None)
            packets.append(packet)
    records, tracks = grade_records(args.grades_db)
    result = evaluate(packets, records, tracks)
    text = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        with args.output.open("x", encoding="utf8") as stream: stream.write(text + "\n")
    print(text)


if __name__ == "__main__":
    try: main()
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Track evaluation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
