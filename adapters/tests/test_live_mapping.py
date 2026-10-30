import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import shutil
import uuid
import unittest

import numpy as np
from PIL import Image

from adapters.live_replay import replay_collection, replay_fixture
from vision.live_mapping import LiveMapper, geographic

ROOT = Path(__file__).resolve().parents[2]


def frame(seq=0, boxes=None):
    return {"frameId": f"f-{seq}", "sequence": seq, "timestampSeconds": 1 + seq * .1, "simulationOnly": False,
            "mapRevision": "test-map", "image": {"width": 100, "height": 80, "sha256": "a" * 64},
            "detections": [{"bboxPixelsXYXY": b, "confidence": .7} for b in (boxes if boxes is not None else [[40, 30, 60, 50]])]}


def metric(seq=0, boxes=None):
    f = frame(seq, boxes)
    f["camera"] = {"calibrationId": "cal-1", "rectified": True, "intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 40},
        "cameraToMap4x4": [[1, 0, 0, 1], [0, 0, 1, 2], [0, -1, 0, 3], [0, 0, 0, 1]],
        "poseTimestampSeconds": f["timestampSeconds"], "positionStdM": .02, "orientationStdRad": .001}
    f["depth"] = {"alignedTo": "color", "calibrationId": "cal-1", "timestampSeconds": f["timestampSeconds"], "method": "rgbd",
                  "scaleToMeters": 1, "minRangeM": .1, "maxRangeM": 100, "stdM": .03}
    return f


class MappingTests(unittest.TestCase):
    def test_rgb_never_invents_location(self):
        result = LiveMapper("test").process(frame())
        self.assertIsNone(result["landmarks"][0]["positionEnuM"])
        self.assertIsNone(result["landmarks"][0]["gps"])
        self.assertEqual(result["mapPoints"], [])
        self.assertFalse(result["actuationEnabled"])

    def test_optical_forward_projects_north_not_up(self):
        result = LiveMapper("test").process(metric(), depth=np.full((80, 100), 4.))
        np.testing.assert_allclose(result["landmarks"][0]["positionEnuM"], [1, 6, 3])
        self.assertGreater(result["landmarks"][0]["positionStdM"], .03)
        self.assertIsNone(result["landmarks"][0]["gps"])
        self.assertTrue(result["depth"]["valid"])

    def test_pixels_right_and_down_project_east_and_down(self):
        result = LiveMapper("test").process(metric(boxes=[[60, 50, 80, 70]]), depth=np.full((80, 100), 4.))
        np.testing.assert_allclose(result["landmarks"][0]["positionEnuM"], [1.8, 6, 2.2])

    def test_stale_pose_and_depth_rejected_without_advancing(self):
        for field, key in (("camera", "poseTimestampSeconds"), ("depth", "timestampSeconds")):
            mapper = LiveMapper("test"); value = metric(); value[field][key] -= .2
            with self.assertRaises(ValueError):
                mapper.process(value, depth=np.full((80, 100), 4.))
            self.assertEqual(mapper.last_sequence, -1)

    def test_invalid_rotation_and_mismatched_calibration(self):
        f = metric(); f["camera"]["cameraToMap4x4"][0][0] = 2
        with self.assertRaises(ValueError):
            LiveMapper("test").process(f, depth=np.full((80, 100), 4.))
        f = metric(); f["depth"]["calibrationId"] = "wrong"
        with self.assertRaises(ValueError):
            LiveMapper("test").process(f, depth=np.full((80, 100), 4.))

    def test_invalid_or_background_mixed_depth_abstains(self):
        for depth in (np.zeros((80, 100)), np.full((80, 100), np.nan), np.tile([1., 10.], (80, 50))):
            result = LiveMapper("test").process(metric(), depth=depth)
            self.assertIsNone(result["landmarks"][0]["positionEnuM"])

    def test_track_assignment_one_to_one(self):
        mapper = LiveMapper("test")
        first = mapper.process(frame(boxes=[[10, 10, 30, 30], [45, 10, 65, 30]]))
        second = mapper.process(frame(1, [[12, 10, 32, 30], [44, 10, 64, 30]]))
        self.assertEqual([d["trackId"] for d in first["detections"]], [d["trackId"] for d in second["detections"]])
        self.assertEqual(len(set(d["trackId"] for d in second["detections"])), 2)
        self.assertTrue(all(t["observationCount"] == 2 for t in second["tracks"]))

    def test_two_boxes_cannot_both_take_one_track(self):
        mapper = LiveMapper("test"); mapper.process(frame())
        result = mapper.process(frame(1, [[40, 30, 60, 50], [41, 31, 61, 51]]))
        self.assertEqual(len(set(d["trackId"] for d in result["detections"])), 2)

    def test_metric_gate_overrules_overlapping_boxes(self):
        mapper = LiveMapper("test")
        a = mapper.process(metric(), depth=np.full((80, 100), 4.))
        b = mapper.process(metric(1), depth=np.full((80, 100), 8.))
        self.assertNotEqual(a["detections"][0]["trackId"], b["detections"][0]["trackId"])

    def test_duplicate_time_identity_and_map_revision_rejected(self):
        for mutate in (lambda f: f.update(sequence=0), lambda f: f.update(timestampSeconds=1),
                       lambda f: f.update(frameId="f-0"), lambda f: f.update(mapRevision="new"),
                       lambda f: f.update(simulationOnly=True)):
            mapper = LiveMapper("test"); mapper.process(frame()); next_frame = frame(1); mutate(next_frame)
            with self.assertRaises(ValueError): mapper.process(next_frame)

    def test_far_observation_is_habitat(self):
        result = LiveMapper("test").process(metric(), depth=np.full((80, 100), 60.))
        self.assertEqual(result["landmarks"][0]["inspectionPriority"], "habitat")

    def test_map_is_bounded_and_not_clearance(self):
        mapper = LiveMapper("test", max_map_points=10)
        result = mapper.process(metric(), depth=np.full((80, 100), 4.))
        self.assertEqual(len(result["mapPoints"]), 10)
        self.assertIn("flight clearance", result["mapMeaning"])
        self.assertNotIn("flight", result)

    def test_gps_requires_anchor_and_preserves_uncertainty(self):
        f = metric()
        f["georeference"] = {"latitude": 44, "longitude": -85, "altitudeM": 250, "datum": "WGS84", "mapAxes": "true-enu",
            "horizontalStdM": 3, "verticalStdM": 5, "headingStdRad": .02, "sourceId": "test-anchor", "timestampSeconds": 0}
        result = LiveMapper("test").process(f, depth=np.full((80, 100), 4.))
        gps = result["landmarks"][0]["gps"]
        self.assertGreater(gps["latitude"], 44)
        self.assertGreater(gps["longitude"], -85)
        self.assertGreater(gps["horizontalUncertaintyM"], 7.3)
        self.assertEqual(gps["altitudeM"], 253)

    def test_repeat_frames_do_not_collapse_covariance(self):
        mapper = LiveMapper("test")
        a = mapper.process(metric(), depth=np.full((80, 100), 4.))
        b = mapper.process(metric(1), depth=np.full((80, 100), 4.))
        self.assertEqual(a["landmarks"][0]["positionStdM"], b["landmarks"][0]["positionStdM"])

    def test_synthetic_fixture_tracks_camera_motion(self):
        packets = list(replay_fixture(ROOT / "examples/live-session-synthetic.json", "fixture", include_image=False))
        self.assertEqual(len(packets), 3)
        for packet in packets:
            self.assertTrue(packet["simulationOnly"])
            np.testing.assert_allclose(packet["landmarks"][0]["positionEnuM"], [0, 4, 1.1], atol=1e-8)
        self.assertEqual(packets[-1]["tracks"][0]["observationCount"], 3)

    def test_lost_landmark_not_reemitted_as_fresh_observation(self):
        mapper = LiveMapper("test")
        mapper.process(frame())
        packet = mapper.process(frame(1, []))
        self.assertEqual(packet["landmarks"], [])
        self.assertEqual(packet["tracks"][0]["state"], "temporarily-lost")

    def test_ros_quaternion_and_covariance_helpers(self):
        from adapters.live_ros2 import quaternion_matrix, covariance_uncertainties
        matrix = quaternion_matrix(0, 0, 0, 1, [1, 2, 3])
        np.testing.assert_allclose(np.asarray(matrix)[:3, :3], np.eye(3))
        np.testing.assert_allclose(np.asarray(matrix)[:3, 3], [1, 2, 3])
        pos, angle = covariance_uncertainties(np.diag([.04, .09, .01, .0025, .0025, .0025]).ravel())
        self.assertAlmostEqual(pos, .3); self.assertAlmostEqual(angle, .05)
        with self.assertRaises(ValueError): quaternion_matrix(0, 0, 0, 2, [0, 0, 0])
        with self.assertRaises(ValueError): covariance_uncertainties(np.zeros(36))
        with self.assertRaises(ValueError): covariance_uncertainties(np.diag([1, 1, 1, 1, 1, -1]).ravel())


class ReplayTests(unittest.TestCase):
    def test_unprocessed_frame_never_uses_similar_pixels_other_timestamp(self):
        root = ROOT / "data" / ("live-mapping-test-" + uuid.uuid4().hex)
        root.mkdir()
        try:
            image = root / "frame.jpg"; Image.new("RGB", (20, 20), "green").save(image)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            item = {"id": "frame-a", "sourceId": "video-a", "path": "frame.jpg", "sha256": digest, "timestampSeconds": 1, "width": 20, "height": 20}
            (root / "manifest.json").write_text(json.dumps({"schema": "mycoscout.video-collection.v1", "sources": [{"id": "video-a"}], "frames": [item]}))
            db = root / "runs.sqlite"; conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE runs(id TEXT,photo_hash TEXT,settings TEXT,result TEXT,video_provenance TEXT,status TEXT,created_at TEXT)")
            result = {"image": {"sha256": digest, "width": 20, "height": 20}, "methods": [{"id": "original", "candidates": [{"bboxPixelsXYXY": [1, 1, 5, 5], "confidence": .8}]}]}
            conn.execute("INSERT INTO runs VALUES(?,?,?,?,?,?,?)", ("run-a", digest, '{"scale":1}', json.dumps(result), json.dumps({**item, "timestampSeconds": 2}), "completed", "today"))
            conn.commit(); conn.close()
            packet = list(replay_collection(root, db, "replay", include_image=False))[0]
            self.assertEqual(packet["detectorStatus"], "unprocessed")
            self.assertEqual(packet["detections"], [])
        finally:
            if root.resolve().parent != (ROOT / "data").resolve() or not root.name.startswith("live-mapping-test-"):
                raise RuntimeError("Unexpected test cleanup path")
            shutil.rmtree(root)

    def test_collection_path_escape_rejected(self):
        from adapters.live_replay import contained
        with self.assertRaises(ValueError): contained(Path("examples"), "../../outside.jpg")


class EvaluationAndPhotoTests(unittest.TestCase):
    def test_capture_delivery_requires_exact_task_request_and_saved_photo(self):
        from adapters.live_photo import verify_capture_completion, CaptureCompletionConflict
        request = {"requestId":"request-a","taskId":"task-a","sessionId":"session-a","landmarkId":"landmark-a","mapRevision":"map-a"}
        digest = "a"*64
        response = {"find":{"id":"find-a","sessionId":"session-a","image":{"sha256":digest},"imageUrl":"/api/finds/find-a/image",
                    "metadata":{k:request[k] for k in ("requestId","taskId","landmarkId","mapRevision")}},
            "action":{"type":"completed","taskId":"task-a","capture":{"photoSha256":digest,"photoRef":"/api/finds/find-a/image"}},
            "state":{"sessionId":"session-a","tasks":[{"id":"task-a","status":"completed","captureRequest":request}]}}
        self.assertEqual(verify_capture_completion(response,request,digest),"find-a")
        mutations = [lambda r:r["action"].update(type="idle"), lambda r:r["action"].update(taskId="task-b"),
            lambda r:r["action"].update(requestId="request-b"), lambda r:r["find"]["metadata"].update(requestId="request-b"),
            lambda r:r["find"]["image"].update(sha256="b"*64), lambda r:r["state"].update(sessionId="session-b"),
            lambda r:r["state"]["tasks"][0].update(status="paused"),
            lambda r:r["state"]["tasks"][0]["captureRequest"].update(requestId="request-b")]
        for mutate in mutations:
            invalid = copy.deepcopy(response); mutate(invalid)
            with self.assertRaises(CaptureCompletionConflict): verify_capture_completion(invalid,request,digest)

    def navigation_input(self):
        return {"timestamp_seconds": 10., "vehicle_position": [1,2,3], "vehicle_orientation_xyzw": [0,0,0,1], "linear_velocity": [.03,.04,0],
            "pose_covariance": np.diag([.0004,.0004,.0004,.0001,.0001,.0001]).ravel().tolist(),
            "twist_covariance": np.diag([.0001,.0001,.0001,.001,.001,.001]).ravel().tolist(),
            "camera_to_map": [[1,0,0,1.2],[0,0,1,2],[0,-1,0,3.1],[0,0,0,1]],
            "battery": {"present": True, "percentage": .73, "timestampSeconds": 9.8},
            "camera_mount_mode": "rigid_calibrated"}

    def test_navigation_keeps_vehicle_origin_separate_from_camera_and_measures_aim(self):
        from adapters.live_ros2 import navigation_telemetry
        result = navigation_telemetry(**self.navigation_input())
        self.assertEqual(result["positionEnuM"], [1,2,3])
        self.assertEqual(result["cameraPositionEnuM"], [1.2,2,3.1])
        self.assertEqual(result["batteryPercent"], 73)
        self.assertAlmostEqual(result["speedMps"], .05)
        self.assertAlmostEqual(result["yawRad"], np.pi/2)
        self.assertAlmostEqual(result["pitchRad"], 0)
        self.assertAlmostEqual(result["vehicleYawRad"], 0)

    def test_navigation_never_invents_missing_or_stale_battery(self):
        from adapters.live_ros2 import navigation_telemetry
        for battery in (None, {"present": False}, {"present": True, "percentage": float("nan"), "timestampSeconds": 10},
                        {"present": True, "percentage": .7, "timestampSeconds": 7}, {"present": True, "percentage": .7, "timestampSeconds": 11}):
            with self.assertRaises(ValueError): navigation_telemetry(**{**self.navigation_input(), "battery": battery})

    def test_navigation_requires_known_camera_mount_and_actual_gimbal_feedback(self):
        from adapters.live_ros2 import navigation_telemetry
        base = self.navigation_input()
        with self.assertRaises(ValueError): navigation_telemetry(**{**base,"camera_mount_mode":"unknown"})
        base["camera_mount_mode"] = "measured_gimbal"
        for feedback in (None, {"timestampSeconds": 9, "positions": [0,0]}, {"timestampSeconds": 10, "positions": [float("nan"),0]}):
            with self.assertRaises(ValueError): navigation_telemetry(**{**base,"gimbal_feedback":feedback})
        result = navigation_telemetry(**{**base, "gimbal_feedback": {"timestampSeconds": 10, "positions": [0,0]}})
        self.assertEqual(result["cameraMountMode"], "measured_gimbal")

    def test_navigation_unknown_motion_or_loose_localization_blocks(self):
        from adapters.live_ros2 import navigation_telemetry
        for changes in ({"twist_covariance": [0]*36}, {"pose_covariance": [0]*36},
                        {"pose_covariance": np.eye(6).ravel().tolist()}, {"linear_velocity": [float("nan"),0,0]}):
            with self.assertRaises(ValueError): navigation_telemetry(**{**self.navigation_input(), **changes})

    def test_group_guesses_only_use_native_crops_and_preserve_abstention(self):
        from vision.live_groups import classify_candidates
        class Stub:
            def __init__(self): self.sizes = []
            def predict(self, image):
                self.sizes.append(image.size)
                return {"schema": "mycoscout.group-guess.v1", "abstained": True, "abstentionReasons": ["competing_guesses_too_close"],
                    "guesses": [{"groupId": "boletes", "label": "Boletes", "modelScore": .45}], "suggestedGroupId": "unknown"}
        predictor = Stub()
        candidates = [{"bboxPixelsXYXY": box, "confidence": .7} for box in ([0,0,31.9,40], [10,10,42,42], [20,20,80,70])]
        result = classify_candidates(Image.new("RGB", (100,100)), candidates, predictor)
        self.assertEqual(predictor.sizes, [(32,32), (60,50)])
        self.assertEqual(result[0]["classification"]["guesses"], [])
        self.assertEqual(result[0]["classification"]["abstentionReasons"], ["insufficient_native_pixels"])
        self.assertEqual(result[1]["classification"]["guesses"][0]["modelScore"], .45)
        self.assertTrue(result[1]["classification"]["abstained"])
        self.assertNotIn("classification", candidates[0])

    def test_group_classification_budget_does_not_change_tracking_or_confirmation(self):
        from vision.live_groups import classify_candidates
        class Stub:
            calls = 0
            def predict(self, image):
                self.calls += 1
                return {"schema": "mycoscout.group-guess.v1", "abstained": False, "guesses": [{"groupId": "morels", "modelScore": .9}],
                        "suggestedGroupId": "morels", "speciesConfirmed": True}
        predictor = Stub()
        candidates = [{"bboxPixelsXYXY": [5,5,40,40], "confidence": .6 + i/100} for i in range(7)]
        classified = classify_candidates(Image.new("RGB", (100,80)), candidates, predictor)
        self.assertEqual(predictor.calls, 5)
        self.assertEqual(classified[0]["classification"]["abstentionReasons"], ["frame_classification_budget"])
        baseline = LiveMapper("plain").process(frame(boxes=[[5,5,40,40]]))
        enriched = frame(boxes=[[5,5,40,40]]); enriched["detections"] = [classified[-1]]
        output = LiveMapper("enriched").process(enriched)
        self.assertEqual(output["detections"][0]["trackId"], baseline["detections"][0]["trackId"])
        self.assertIsNone(output["landmarks"][0]["positionEnuM"])
        self.assertFalse(output["landmarks"][0]["humanVerified"])
        self.assertFalse(output["landmarks"][0]["classification"]["speciesConfirmed"])
        self.assertEqual(output["landmarks"][0]["classification"]["guesses"][0]["groupId"], "morels")

    def test_partial_labels_do_not_become_tracking_accuracy(self):
        from adapters.evaluate_live_tracks import evaluate
        result = evaluate([], [{"context": {}, "review": {"complete": False, "humanVerified": True}}], {})
        self.assertEqual(result["status"], "awaiting_complete_labels")
        self.assertIsNone(result["detection"]["recall"])
        self.assertIsNone(result["identitySwitches"])

    def test_linked_tracks_measure_delay_switches_and_original_pixel_width(self):
        from adapters.evaluate_live_tracks import evaluate
        packets, records = [], []
        for index in range(3):
            p = frame(index, [] if index == 0 else [[40, 30, 60, 50]])
            p.update(detectorStatus="completed", provenance={"sourceId": "v1", "videoHash": "b" * 64}, detector={"model": "test-only"})
            if p["detections"]: p["detections"][0]["trackId"] = "machine-a" if index == 1 else "machine-b"
            packets.append(p)
            records.append({"context": {"frameId": p["frameId"], "sha256": "a" * 64, "sourceId": "v1", "videoHash": "b" * 64, "timestampSeconds": p["timestampSeconds"]},
                "review": {"complete": True, "humanVerified": True, "status": "visible", "boxes": [{"bboxPixelsXYXY": [40,30,60,50], "trackId": "human-a"}]}})
        result = evaluate(packets, records, {"human-a": {"id": "human-a", "sourceId": "v1", "videoHash": "b" * 64, "humanCreated": True}})
        self.assertAlmostEqual(result["detection"]["recall"], 2/3)
        self.assertAlmostEqual(result["tracks"][0]["delayWithinReviewedFramesSeconds"], .1)
        self.assertEqual(result["identitySwitches"], 1)
        self.assertEqual(result["pixelWidthBins"]["16to32"]["humanBoxes"], 3)

    def test_photo_ack_uses_saved_pixels_and_rejects_missing_motion_or_bad_pose(self):
        from adapters.live_photo import photo_payload
        root = ROOT / "data" / ("live-mapping-test-" + uuid.uuid4().hex)
        root.mkdir()
        try:
            path = root / "native.png"
            pixels = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (20, 20))
            Image.fromarray(pixels).save(path)
            request = {"requestId": "r", "taskId": "t", "sessionId": "s", "landmarkId": "m", "mapRevision": "map",
                "requestedMs": 200, "positionEnuM": [0,0,0], "aim": {"yawRad": 0, "pitchRad": 0}}
            evidence = {**request, "schema": "mycoscout.capture-evidence.v1", "synthetic": False, "nativeUncropped": True,
                "capturedMs": 1000, "poseTimestampMs": 1000, "photoSha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "yawRad": 0, "pitchRad": 0, "targetBboxPixelsXYXY": [0,0,40,40],
                "motionSamples": [{"timestampMs": t, "speedMps": .01} for t in range(200, 1001, 100)]}
            payload = photo_payload(request, path, evidence, min_sharpness=10, now_ms=1001)
            self.assertEqual(payload["ack"]["photoSha256"], evidence["photoSha256"])
            self.assertGreater(payload["ack"]["qualityEvidence"]["value"], 10)
            self.assertFalse(payload["ack"]["synthetic"])
            offset_request = {**request, "cameraPositionEnuM": [.1,0,0]}
            with self.assertRaises(ValueError): photo_payload(offset_request, path, evidence, min_sharpness=10, now_ms=1001)
            offset_payload = photo_payload(offset_request, path, {**evidence,"cameraPositionEnuM":[.1,0,0]}, min_sharpness=10, now_ms=1001)
            self.assertEqual(offset_payload["ack"]["cameraPositionEnuM"], [.1,0,0])
            for change in ({"positionEnuM": [1,0,0]}, {"motionSamples": []}, {"poseTimestampMs": 900}, {"photoSha256": "0"*64}):
                with self.assertRaises(ValueError): photo_payload(request, path, {**evidence, **change}, min_sharpness=10, now_ms=1001)
        finally:
            if root.resolve().parent != (ROOT / "data").resolve() or not root.name.startswith("live-mapping-test-"):
                raise RuntimeError("Unexpected cleanup path")
            shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
