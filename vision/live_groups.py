"""Optional bounded group guesses on native candidate crops, independent of tracking."""
from __future__ import annotations

import copy
import math
from pathlib import Path


def load_group_predictor(checkpoint=None):
    from vision.multiclass import Predictor
    return Predictor(Path(checkpoint) if checkpoint else Path(__file__).resolve().parent / "checkpoints/groups-v1.json")


def unknown(reason, native_size=None):
    return {"schema": "mycoscout.group-guess.v1", "guesses": [], "abstained": True,
            "abstentionReasons": [reason], "suggestedGroupId": "unknown",
            "humanVerificationRequired": True, "speciesConfirmed": False,
            "edibility": "not assessed", "salesApproval": False, "flightGateEnabled": False,
            "nativeCropSize": native_size, "classificationStatus": "not_classified"}


def classify_candidates(image, candidates, predictor, *, max_candidates=5, minimum_native_pixels=32):
    """Pass original-pixel crops to the predictor; resizing never counts as evidence.

    The existing classifier performs its own network preprocessing, but eligibility
    is based only on the detector box's width/height in the original color frame.
    Candidate order, confidence, boxes, tracks, and metric geometry are unchanged.
    """
    if type(max_candidates) is not int or not 1 <= max_candidates <= 5:
        raise ValueError("Group classification is limited to 1..5 candidates per frame")
    if type(minimum_native_pixels) is not int or minimum_native_pixels < 32:
        raise ValueError("At least 32 original pixels per crop dimension are required")
    if not isinstance(candidates, list) or len(candidates) > 100:
        raise ValueError("A frame may contain at most 100 candidates")
    result, eligible = copy.deepcopy(candidates), []
    for index, candidate in enumerate(result):
        box = candidate.get("bboxPixelsXYXY")
        if not isinstance(box, list) or len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
            raise ValueError("Classification needs finite native XYXY candidate boxes")
        x1, y1, x2, y2 = box
        if not 0 <= x1 < x2 <= image.width or not 0 <= y1 < y2 <= image.height:
            raise ValueError("Classification candidate lies outside native image")
        size = [x2-x1, y2-y1]
        if min(size) < minimum_native_pixels:
            candidate["classification"] = unknown("insufficient_native_pixels", size)
        else:
            candidate["classification"] = unknown("frame_classification_budget", size)
            eligible.append(index)
    eligible.sort(key=lambda index: result[index].get("confidence", 0), reverse=True)
    for index in eligible[:max_candidates]:
        candidate = result[index]
        x1, y1, x2, y2 = candidate["bboxPixelsXYXY"]
        crop = image.crop((math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)))
        prediction = copy.deepcopy(predictor.predict(crop))
        if not isinstance(prediction, dict) or type(prediction.get("abstained")) is not bool or not isinstance(prediction.get("guesses"), list):
            raise ValueError("Group predictor returned an invalid guess contract")
        prediction.update({"nativeCropSize": [x2-x1, y2-y1], "nativeCropPixels": list(crop.size),
                           "classificationStatus": "completed", "humanVerificationRequired": True,
                           "speciesConfirmed": False, "salesApproval": False, "flightGateEnabled": False})
        candidate["classification"] = prediction
    return result
