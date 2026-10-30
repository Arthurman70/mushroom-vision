"""Bounded temporal candidate tracking and calibrated RGBD landmark reconstruction.

This is not SLAM or a flight controller: camera poses must come from a synchronized
external odometry/SLAM source. RGB alone never acquires metric depth or GPS here.
Optical camera axes are x right, y down, z forward; map axes are east, north, up.
"""
from __future__ import annotations

import copy
import json
import math
import re
from collections import OrderedDict

import numpy as np

SCHEMA = "mycoscout.live-frame.v1"
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")


def number(value, name, low=-1e12, high=1e12):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be finite in {low}..{high}")
    return float(value)


def identifier(value, name):
    if not isinstance(value, str) or not ID.fullmatch(value):
        raise ValueError(f"Invalid {name}")
    return value


def iou(a, b):
    area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return area / max(1e-9, (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - area)


def validate_camera(camera, width, height, timestamp, max_skew=.05):
    if not isinstance(camera, dict):
        raise ValueError("Metric mapping requires calibrated camera metadata")
    c = copy.deepcopy(camera)
    identifier(c.get("calibrationId"), "calibrationId")
    if c.get("opticalFrame", "x-right-y-down-z-forward") != "x-right-y-down-z-forward":
        raise ValueError("Camera axes must use optical x-right/y-down/z-forward")
    k = c.get("intrinsics", {})
    for key in ("fx", "fy"):
        number(k.get(key), key, 1, 100000)
    number(k.get("cx"), "cx", 0, width)
    number(k.get("cy"), "cy", 0, height)
    if c.get("rectified") is not True:
        raise ValueError("Depth projection requires rectified, color-aligned imagery")
    if abs(number(c.get("poseTimestampSeconds"), "poseTimestampSeconds", 0) - timestamp) > max_skew:
        raise ValueError("Camera pose is not synchronized with image capture")
    transform = np.asarray(c.get("cameraToMap4x4"), dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError("Camera transform must be a finite rigid homogeneous matrix")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or not math.isclose(np.linalg.det(rotation), 1, abs_tol=1e-5):
        raise ValueError("Camera transform rotation must be right-handed and orthonormal")
    number(c.get("positionStdM"), "positionStdM", .001, 1000)
    number(c.get("orientationStdRad"), "orientationStdRad", .00001, math.pi)
    return c, transform


def validate_depth(depth, meta, width, height, timestamp, camera, max_skew=.05):
    if not isinstance(meta, dict) or meta.get("alignedTo") != "color" or meta.get("calibrationId") != camera["calibrationId"]:
        raise ValueError("Depth must be aligned to color with matching calibration")
    if abs(number(meta.get("timestampSeconds"), "depth timestamp", 0) - timestamp) > max_skew:
        raise ValueError("Depth is not synchronized with image capture")
    if meta.get("method") not in {"rgbd", "stereo", "surveyed"}:
        raise ValueError("Depth method is required")
    scale = number(meta.get("scaleToMeters"), "depth scale", .000001, 1000)
    minimum = number(meta.get("minRangeM"), "minimum depth", .01, 1000)
    maximum = number(meta.get("maxRangeM"), "maximum depth", minimum + .01, 1000)
    number(meta.get("stdM"), "depth uncertainty", .001, 1000)
    values = np.asarray(depth, dtype=float)
    if values.shape != (height, width):
        raise ValueError("Aligned depth shape must match the original color image")
    values = values * scale
    values[(~np.isfinite(values)) | (values < minimum) | (values > maximum)] = np.nan
    return values


def project_pixel(u, v, z, camera, transform):
    k = camera["intrinsics"]
    optical = np.array([(u - k["cx"]) * z / k["fx"], (v - k["cy"]) * z / k["fy"], z])
    return transform[:3, :3] @ optical + transform[:3, 3], optical


def locate_box(box, depth, meta, camera, transform):
    u, v = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    half_w, half_h = max(1, (box[2] - box[0]) * .15), max(1, (box[3] - box[1]) * .15)
    patch = depth[max(0, int(v - half_h)):min(depth.shape[0], int(v + half_h) + 1),
                  max(0, int(u - half_w)):min(depth.shape[1], int(u + half_w) + 1)]
    valid = patch[np.isfinite(patch)]
    if valid.size < 3 or valid.size < patch.size * .6:
        return None
    z = float(np.median(valid))
    spread = float(np.quantile(valid, .9) - np.quantile(valid, .1))
    if spread > max(.15, z * .15):
        return None  # Mixed foreground/background depth does not locate the mushroom.
    point, optical = project_pixel(u, v, z, camera, transform)
    k = camera["intrinsics"]
    jacobian = np.array([[z / k["fx"], 0, (u - k["cx"]) / k["fx"]],
                         [0, z / k["fy"], (v - k["cy"]) / k["fy"]], [0, 0, 1]])
    # Pixel-center and box/foreground ambiguity are included, not just sensor noise.
    pixel_std = max(1.5, min(box[2] - box[0], box[3] - box[1]) * .1)
    sigma_z = max(meta["stdM"], spread / 2)
    covariance = transform[:3, :3] @ jacobian @ np.diag([pixel_std ** 2, pixel_std ** 2, sigma_z ** 2]) @ jacobian.T @ transform[:3, :3].T
    sensor_range = float(np.linalg.norm(optical))
    floor_variance = camera["positionStdM"] ** 2 + (sensor_range * camera["orientationStdRad"]) ** 2
    covariance += np.eye(3) * floor_variance
    return {"positionEnuM": point.tolist(), "covarianceEnuM2": covariance.tolist(),
            "positionStdM": float(math.sqrt(np.linalg.eigvalsh(covariance).max())), "rangeM": sensor_range}


def validate_anchor(anchor, timestamp):
    if anchor is None:
        return None
    anchor = copy.deepcopy(anchor)
    if anchor.get("datum") != "WGS84" or anchor.get("mapAxes") != "true-enu":
        raise ValueError("GPS anchoring requires a WGS84 origin and true ENU map alignment")
    for key, lo, hi in (("latitude", -89, 89), ("longitude", -180, 180), ("altitudeM", -1000, 20000),
                        ("horizontalStdM", .001, 10000), ("verticalStdM", .001, 10000), ("headingStdRad", .00001, math.pi)):
        number(anchor.get(key), key, lo, hi)
    identifier(anchor.get("sourceId"), "anchor sourceId")
    number(anchor.get("timestampSeconds"), "anchor timestamp", 0, timestamp + .1)
    return anchor


def geographic(position, covariance, anchor):
    if anchor is None:
        return None
    east, north, up = position
    if math.hypot(east, north) > 10000:
        return None  # Local tangent approximation is deliberately range-bounded.
    latitude = math.radians(anchor["latitude"])
    a, e2 = 6378137.0, 6.69437999014e-3
    n = a / math.sqrt(1 - e2 * math.sin(latitude) ** 2)
    m = a * (1 - e2) / (1 - e2 * math.sin(latitude) ** 2) ** 1.5
    horizontal_variance = max(np.linalg.eigvalsh(np.asarray(covariance)[:2, :2])) + anchor["horizontalStdM"] ** 2 + (math.hypot(east, north) * anchor["headingStdRad"]) ** 2
    return {"latitude": anchor["latitude"] + math.degrees(north / (m + anchor["altitudeM"])),
            "longitude": ((anchor["longitude"] + math.degrees(east / ((n + anchor["altitudeM"]) * math.cos(latitude))) + 180) % 360) - 180,
            "altitudeM": anchor["altitudeM"] + up,
            "horizontalUncertaintyM": 2.45 * math.sqrt(horizontal_variance),
            "verticalUncertaintyM": 1.96 * math.sqrt(covariance[2][2] + anchor["verticalStdM"] ** 2),
            "uncertaintyMeaning": "Approximate 95% bound from supplied sensor/anchor errors; unmodeled bias remains"}


class LiveMapper:
    def __init__(self, session_id, *, max_tracks=128, max_landmarks=512, max_map_points=2048, track_gap_seconds=5, voxel_size_m=.1):
        self.session_id = identifier(session_id, "sessionId")
        for value, maximum, name in ((max_tracks, 512, "max_tracks"), (max_landmarks, 4096, "max_landmarks"), (max_map_points, 10000, "max_map_points")):
            if type(value) is not int or not 1 <= value <= maximum:
                raise ValueError(f"Invalid {name}")
        self.max_tracks, self.max_landmarks, self.max_map_points = max_tracks, max_landmarks, max_map_points
        self.track_gap_seconds = number(track_gap_seconds, "track gap", .01, 30)
        self.voxel_size_m = number(voxel_size_m, "voxel size", .01, 5)
        self.last_sequence, self.last_timestamp = -1, -1
        self.map_revision, self.simulation_only, self.image_geometry = None, None, None
        self.track_counter = 0
        self.seen_ids, self.tracks, self.landmarks, self.points = set(), {}, OrderedDict(), OrderedDict()

    def process(self, frame, *, depth=None, rgb=None):
        frame = copy.deepcopy(frame)
        frame_id = identifier(frame.get("frameId"), "frameId")
        seq = frame.get("sequence")
        if type(seq) is not int or seq <= self.last_sequence or seq > 100000000:
            raise ValueError("Frame sequences must increase strictly")
        stamp = number(frame.get("timestampSeconds"), "timestampSeconds", 0)
        if stamp <= self.last_timestamp or frame_id in self.seen_ids:
            raise ValueError("Duplicate or out-of-order frame timestamp/identity")
        if len(self.seen_ids) >= 100000:
            raise ValueError("Session reached 100000-frame bound; start a new session")
        if type(frame.get("simulationOnly")) is not bool:
            raise ValueError("simulationOnly must be explicit")
        revision = identifier(frame.get("mapRevision", "unlocated-v1"), "mapRevision")
        if self.map_revision is not None and revision != self.map_revision:
            raise ValueError("Map revision changed; start a new mapping session to avoid mixing coordinate frames")
        if self.simulation_only is not None and frame["simulationOnly"] != self.simulation_only:
            raise ValueError("Simulation and real observations cannot share a session")
        image = frame.get("image", {})
        width, height = image.get("width"), image.get("height")
        if type(width) is not int or type(height) is not int or min(width, height) < 1 or width * height > 8000000:
            raise ValueError("Image geometry must be positive and at most 8 megapixels")
        if self.image_geometry and self.image_geometry != (width, height):
            raise ValueError("Image geometry changed within session")
        if not isinstance(image.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", image["sha256"]):
            raise ValueError("Source image SHA256 is required")
        raw_detections = frame.get("detections", [])
        if not isinstance(raw_detections, list) or len(raw_detections) > 100:
            raise ValueError("At most 100 detections per frame")
        detections = []
        for index, item in enumerate(raw_detections):
            box = item.get("bboxPixelsXYXY")
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError("Detection box must be XYXY")
            box = [number(v, "box coordinate", 0, max(width, height)) for v in box]
            if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
                raise ValueError("Detection box lies outside original image")
            detections.append({"detectionId": f"{frame_id}:{index}", "bboxPixelsXYXY": box,
                "confidence": number(item.get("confidence"), "confidence", 0, 1), "label": "Unverified mushroom candidate",
                "positionEnuM": None, "covarianceEnuM2": None, "positionStdM": None, "rangeM": None,
                "locationStatus": "unlocated", "humanVerified": False})
            if "classification" in item:
                classification = copy.deepcopy(item["classification"])
                if not isinstance(classification, dict) or classification.get("schema") != "mycoscout.group-guess.v1" or type(classification.get("abstained")) is not bool or not isinstance(classification.get("guesses"), list) or len(classification["guesses"]) > 5:
                    raise ValueError("Invalid optional group classification")
                if len(json.dumps(classification, allow_nan=False)) > 16000:
                    raise ValueError("Group classification exceeds metadata limit")
                for guess in classification["guesses"]:
                    if not isinstance(guess, dict): raise ValueError("Invalid group guess")
                    number(guess.get("modelScore"), "group model score", 0, 1)
                classification.update(humanVerificationRequired=True, speciesConfirmed=False, salesApproval=False, flightGateEnabled=False)
                detections[-1]["classification"] = classification
        camera, transform, depth_values = None, None, None
        warnings = []
        anchor = validate_anchor(frame.get("georeference"), stamp)
        if depth is not None:
            camera, transform = validate_camera(frame.get("camera"), width, height, stamp)
            depth_values = validate_depth(depth, frame.get("depth"), width, height, stamp, camera)
            for detection in detections:
                located = locate_box(detection["bboxPixelsXYXY"], depth_values, frame["depth"], camera, transform)
                if located:
                    detection.update(located)
                    detection["locationStatus"] = "depth-projected-candidate"
                else:
                    warnings.append("A candidate has insufficient or ambiguous depth and remains unlocated")
        elif frame.get("depth") or frame.get("camera"):
            warnings.append("No aligned depth array supplied; positions remain unknown")
        descriptors = []
        if rgb is not None:
            pixels = np.asarray(rgb)
            if pixels.shape != (height, width, 3):
                raise ValueError("RGB appearance input must match image dimensions")
            for d in detections:
                x1, y1, x2, y2 = d["bboxPixelsXYXY"]
                region = pixels[int(y1):max(int(y1) + 1, int(y2)), int(x1):max(int(x1) + 1, int(x2))]
                descriptors.append(np.mean(region, axis=(0, 1)) / 255)
        else:
            descriptors = [None] * len(detections)
        # Only mutate state after all packet/geometry validation succeeds.
        self.last_sequence, self.last_timestamp = seq, stamp
        self.seen_ids.add(frame_id)
        self.map_revision, self.simulation_only, self.image_geometry = revision, frame["simulationOnly"], (width, height)
        active = {key: t for key, t in self.tracks.items() if stamp - t["lastSeenSeconds"] <= self.track_gap_seconds}
        pairs = []
        for index, d in enumerate(detections):
            for track_id, track in active.items():
                overlap = iou(d["bboxPixelsXYXY"], track["bboxPixelsXYXY"])
                distance = None
                if d["positionEnuM"] is not None and track.get("positionEnuM") is not None:
                    distance = float(np.linalg.norm(np.array(d["positionEnuM"]) - track["positionEnuM"]))
                    gate = min(1.5, max(.15, 3 * math.hypot(d["positionStdM"], track["positionStdM"])))
                    if distance > gate:
                        continue
                    score = 2 + 1 - distance / gate
                elif overlap >= .1:
                    score = overlap
                else:
                    continue
                if descriptors[index] is not None and track.get("appearance") is not None:
                    color_distance = float(np.linalg.norm(descriptors[index] - track["appearance"]))
                    if color_distance > .5:
                        continue
                    score += .2 * (1 - color_distance)
                pairs.append((score, index, track_id))
        assigned, used_tracks = {}, set()
        for score, index, track_id in sorted(pairs, reverse=True):
            if index not in assigned and track_id not in used_tracks:
                assigned[index] = track_id
                used_tracks.add(track_id)
        for index, detection in enumerate(detections):
            track_id = assigned.get(index)
            if track_id is None:
                self.track_counter += 1
                track_id = f"track-{self.track_counter:06d}"
            old = active.get(track_id, {})
            landmark_id = old.get("landmarkId", f"landmark-{track_id[6:]}")
            track = {"trackId": track_id, "landmarkId": landmark_id, "bboxPixelsXYXY": detection["bboxPixelsXYXY"],
                     "state": "observed", "observationCount": old.get("observationCount", 0) + 1,
                     "firstSeenSeconds": old.get("firstSeenSeconds", stamp), "lastSeenSeconds": stamp,
                     "positionEnuM": detection["positionEnuM"], "positionStdM": detection["positionStdM"],
                     "appearance": descriptors[index]}
            active[track_id] = track
            detection.update({"trackId": track_id, "landmarkId": landmark_id, "associationVerified": False})
            previous = self.landmarks.get(landmark_id, {})
            # Retain the best observed position. Repeated correlated observations must
            # not make uncertainty collapse as if each view were independent GNSS.
            best = detection if detection["positionEnuM"] is not None and (previous.get("positionStdM") is None or detection["positionStdM"] < previous["positionStdM"]) else previous
            position, covariance = best.get("positionEnuM"), best.get("covarianceEnuM2")
            landmark = {"landmarkId": landmark_id, "trackId": track_id, "positionEnuM": position,
                "covarianceEnuM2": covariance, "positionStdM": best.get("positionStdM"),
                "rangeM": detection["rangeM"], "gps": geographic(position, covariance, anchor) if position is not None else None,
                "inspectionPriority": "unlocated" if detection["rangeM"] is None else "habitat" if detection["rangeM"] >= 50 else "immediate",
                "humanVerified": False, "associationVerified": False, "observationCount": track["observationCount"],
                "confidence": detection["confidence"],
                "firstSeenSeconds": track["firstSeenSeconds"], "lastSeenSeconds": stamp,
                "bestFrameId": frame_id if best is detection else previous.get("bestFrameId"), "latestFrameId": frame_id,
                "locationStatus": "depth-projected-candidate" if position is not None else "unlocated"}
            if "classification" in detection:
                landmark["classification"] = copy.deepcopy(detection["classification"])
            self.landmarks[landmark_id] = landmark
            self.landmarks.move_to_end(landmark_id)
        for track_id, track in active.items():
            if track["lastSeenSeconds"] != stamp:
                track["state"] = "temporarily-lost"
        self.tracks = dict(sorted(active.items(), key=lambda pair: pair[1]["lastSeenSeconds"], reverse=True)[:self.max_tracks])
        while len(self.landmarks) > self.max_landmarks:
            self.landmarks.popitem(last=False)
        if depth_values is not None:
            step = max(1, int(math.ceil(math.sqrt(width * height / 256))))
            for v in range(step // 2, height, step):
                for u in range(step // 2, width, step):
                    if not math.isfinite(depth_values[v, u]):
                        continue
                    point, _ = project_pixel(u, v, depth_values[v, u], camera, transform)
                    key = tuple(np.floor(point / self.voxel_size_m).astype(int))
                    self.points[key] = {"positionEnuM": point.tolist(), "observedAtSeconds": stamp}
                    self.points.move_to_end(key)
            while len(self.points) > self.max_map_points:
                self.points.popitem(last=False)
        # The station retains prior landmarks. Only current observations are emitted
        # so an old range or old position cannot masquerade as fresh sensor evidence.
        emitted_landmarks = copy.deepcopy([landmark for landmark in self.landmarks.values() if landmark["lastSeenSeconds"] == stamp])
        if depth_values is None:
            # Do not present historical geometry as a fresh synchronized measurement.
            for landmark in emitted_landmarks:
                for field in ("positionEnuM", "covarianceEnuM2", "positionStdM", "gps", "rangeM"):
                    landmark[field] = None
                landmark["locationStatus"] = "unlocated"
        packet = {"schema": SCHEMA, "sessionId": self.session_id, "frameId": frame_id, "sequence": seq,
            "timestampSeconds": stamp, "simulationOnly": frame["simulationOnly"], "coordinateFrame": "map-enu",
            "mapRevision": revision, "image": image, "detections": detections,
            "tracks": [{k: v for k, v in t.items() if k != "appearance"} for t in self.tracks.values()],
            "landmarks": emitted_landmarks, "mapPoints": list(self.points.values()) if depth_values is not None else [],
            "mapMeaning": "Observed surface samples only; unknown/free space and flight clearance are not inferred",
            "mappingMode": "external-pose-rgbd" if depth_values is not None else "rgb-unlocated",
            "detectorStatus": frame.get("detectorStatus", "completed"), "actuationEnabled": False,
            "humanConfirmationRequired": True, "warnings": sorted(set(warnings))}
        for field in ("provenance", "detector", "flight"):
            if field in frame:
                packet[field] = frame[field]
        if depth_values is not None:
            packet["camera"] = camera
            packet["depth"] = {**frame["depth"], "valid": True}
        if anchor:
            packet["georeference"] = {"anchor": {k: anchor[k] for k in ("latitude", "longitude", "altitudeM")},
                **{k: anchor[k] for k in ("horizontalStdM", "verticalStdM", "headingStdRad", "sourceId", "timestampSeconds")},
                "datum": "WGS84", "mapAxes": "true-enu"}
        return packet
