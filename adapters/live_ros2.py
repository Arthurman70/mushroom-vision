"""Optional ROS 2 RGBD/RTAB-Map bridge; no flight or motor command publishers.

Uses rectified RGB, color-aligned depth, CameraInfo, odometry covariance and the
timestamped map-to-camera-optical TF. A one-frame queue drops stale pending work.
ROS/device integration must be validated on the companion computer; imports are
deferred so geometry helpers and CLI compilation work without ROS installed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
from pathlib import Path
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid

import numpy as np
from PIL import Image

from adapters.live_capture import ReviewedDetector
from adapters.live_replay import read_json
from adapters.review_stream import Station
from adapters.scout_frame import LocalDetector
from vision.live_mapping import LiveMapper


def quaternion_matrix(x, y, z, w, translation):
    q = np.asarray([x, y, z, w], dtype=float)
    if not np.isfinite(q).all() or abs(float(q @ q) - 1) > .001:
        raise ValueError("TF quaternion must be normalized")
    transform = np.eye(4)
    transform[:3, :3] = [[1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                          [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                          [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    transform[:3, 3] = translation
    if not np.isfinite(transform).all():
        raise ValueError("TF translation must be finite")
    return transform.tolist()


def covariance_uncertainties(flat):
    covariance = np.asarray(flat, dtype=float)
    if covariance.size != 36:
        raise ValueError("Odometry pose covariance needs 36 entries")
    covariance = covariance.reshape(6, 6)
    if not np.isfinite(covariance).all() or not np.allclose(covariance, covariance.T, atol=1e-8) or np.linalg.eigvalsh(covariance).min() < -1e-8:
        raise ValueError("Odometry covariance must be finite positive semidefinite")
    position = float(np.linalg.eigvalsh(covariance[:3, :3]).max())
    orientation = float(np.linalg.eigvalsh(covariance[3:, 3:]).max())
    if position <= 0 or orientation <= 0:
        raise ValueError("Unknown/all-zero odometry covariance is not a precision measurement")
    return math.sqrt(position), math.sqrt(orientation)


def navigation_telemetry(*, timestamp_seconds, vehicle_position, vehicle_orientation_xyzw, linear_velocity,
                         pose_covariance, twist_covariance, camera_to_map, battery,
                         camera_mount_mode, gimbal_feedback=None):
    """Use measured body pose/motion/power and measured optical direction only.

    No battery, gimbal angle or stationary-motion default is supplied. The camera
    translation is retained separately from the aircraft body origin.
    """
    stamp = float(timestamp_seconds)
    if not math.isfinite(stamp) or stamp < 0:
        raise ValueError("Navigation timestamp must be finite")
    position, velocity = np.asarray(vehicle_position, dtype=float), np.asarray(linear_velocity, dtype=float)
    if position.shape != (3,) or velocity.shape != (3,) or not np.isfinite(position).all() or not np.isfinite(velocity).all():
        raise ValueError("Measured vehicle position and velocity must be finite 3D vectors")
    if not isinstance(vehicle_orientation_xyzw, list) or len(vehicle_orientation_xyzw) != 4:
        raise ValueError("Measured vehicle orientation quaternion is required")
    body_transform = np.asarray(quaternion_matrix(*vehicle_orientation_xyzw, position.tolist()))
    position_std, orientation_std = covariance_uncertainties(pose_covariance)
    twist = np.asarray(twist_covariance, dtype=float)
    if twist.size != 36: raise ValueError("Measured motion needs odometry twist covariance")
    twist = twist.reshape(6, 6)
    if not np.isfinite(twist).all() or not np.allclose(twist, twist.T, atol=1e-8) or np.linalg.eigvalsh(twist).min() < -1e-8:
        raise ValueError("Motion covariance must be finite positive semidefinite")
    speed_variance = float(np.linalg.eigvalsh(twist[:3, :3]).max())
    if speed_variance <= 0 or speed_variance > .25 or position_std > .5 or orientation_std > .1:
        raise ValueError("Vehicle motion/localization uncertainty is unknown or too large for navigation proposals")
    if not isinstance(battery, dict) or battery.get("present") is not True:
        raise ValueError("Physical battery telemetry is required")
    battery_stamp, percentage = battery.get("timestampSeconds"), battery.get("percentage")
    if isinstance(percentage, bool) or not isinstance(percentage, (int, float)) or not math.isfinite(percentage) or not 0 <= percentage <= 1:
        raise ValueError("BatteryState percentage must be measured in 0..1")
    if not isinstance(battery_stamp, (int, float)) or not math.isfinite(battery_stamp) or not -.05 <= stamp - battery_stamp <= 1.5:
        raise ValueError("Physical battery telemetry is stale or from a different capture clock")
    if camera_mount_mode not in {"rigid_calibrated", "measured_gimbal"}:
        raise ValueError("Camera mount/aim must be calibrated or supported by measured gimbal feedback")
    if camera_mount_mode == "measured_gimbal":
        if not isinstance(gimbal_feedback, dict): raise ValueError("Actual gimbal joint feedback is required")
        joint_stamp, joints = gimbal_feedback.get("timestampSeconds"), gimbal_feedback.get("positions")
        if not isinstance(joint_stamp, (int, float)) or not math.isfinite(joint_stamp) or abs(joint_stamp - stamp) > .05:
            raise ValueError("Measured gimbal feedback is not synchronized")
        if not isinstance(joints, list) or len(joints) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in joints):
            raise ValueError("Measured gimbal yaw and pitch positions are required")
    transform = np.asarray(camera_to_map, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0,0,0,1], atol=1e-8):
        raise ValueError("Measured optical camera transform is invalid")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or abs(np.linalg.det(rotation)-1) > 1e-5:
        raise ValueError("Measured optical camera rotation must be rigid")
    forward = rotation[:, 2]
    return {"positionEnuM": position.tolist(), "timestampSeconds": stamp,
        "batteryPercent": float(percentage * 100), "batteryTimestampSeconds": float(battery_stamp),
        "speedMps": float(np.linalg.norm(velocity)), "speedStdMps": math.sqrt(speed_variance),
        "yawRad": math.atan2(forward[1], forward[0]), "pitchRad": math.atan2(forward[2], math.hypot(forward[0], forward[1])),
        "localized": True, "positionStdM": position_std, "orientationStdRad": orientation_std,
        "depthValid": True, "depthTimestampSeconds": stamp,
        "cameraPositionEnuM": transform[:3, 3].tolist(), "cameraMountMode": camera_mount_mode,
        "vehicleOrientationXyzw": vehicle_orientation_xyzw,
        "vehicleYawRad": math.atan2(body_transform[1,0], body_transform[0,0]),
        "vehiclePitchRad": math.asin(float(np.clip(-body_transform[2,0], -1, 1))),
        "measurementSource": "synchronized-vehicle-odometry-optical-TF-and-battery"}


class NavigationStation(Station):
    """Separate bounded network call; a slow semantic upload cannot queue navigation."""
    def publish(self, packet):
        headers = {"Content-Type": "application/json", "Accept": "application/json", "User-Agent": "MycoScout-navigation-bridge/0.8"}
        if self.token: headers["Authorization"] = "Bearer " + self.token
        body = json.dumps(packet, allow_nan=False, separators=(",", ":")).encode()
        if len(body) > 8 * 1024 ** 2: raise ValueError("Navigation payload exceeds bound")
        request = urllib.request.Request(self.origin + "/api/live/navigation", data=body, headers=headers)
        try:
            with self.opener.open(request, timeout=1) as response:
                # HTTP success acknowledges the server transaction; the navigation
                # producer does not need to download the complete dashboard state.
                response.read(1024)
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"Navigation update rejected ({error.code}): {error.read(512).decode(errors='replace')}") from None


def main():
    try:
        import cv2
        import rclpy
        from rclpy.node import Node
        from rclpy.time import Time
        from rclpy.duration import Duration
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image as RosImage, CameraInfo, BatteryState, JointState
        from nav_msgs.msg import Odometry
        import message_filters
        from cv_bridge import CvBridge
        from tf2_ros import Buffer, TransformListener
    except ImportError as error:
        raise RuntimeError("ROS 2, cv_bridge, message_filters, tf2_ros, sensor_msgs/nav_msgs and OpenCV are required on the companion computer") from error

    def seconds(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    class Bridge(Node):
        def __init__(self):
            super().__init__("mycoscout_live_bridge")
            defaults = {"station": "http://127.0.0.1:8787", "session_id": "ros-" + str(uuid.uuid4()),
                "rgb_topic": "/camera/color/image_rect", "depth_topic": "/camera/aligned_depth_to_color/image_raw",
                "info_topic": "/camera/color/camera_info", "odom_topic": "/odometry/filtered",
                "map_frame": "map", "map_revision": "map-v1", "map_is_true_enu": False,
                "vehicle_frame": "base_link", "battery_topic": "/mavros/battery",
                "camera_mount_mode": "unknown", "gimbal_feedback_topic": "/gimbal/joint_states",
                "gimbal_yaw_joint": "gimbal_yaw", "gimbal_pitch_joint": "gimbal_pitch",
                "navigation_enabled": True, "navigation_rate_hz": 5.0,
                "navigation_resolution_m": .5, "navigation_ttl_seconds": 1.0,
                "optical_frame": "camera_color_optical_frame", "rectified_color": False,
                "model": "models/grounding-dino-tiny", "active_model": "", "device": "cpu", "no_detector": False, "classify_groups": False,
                "depth_min_m": .2, "depth_max_m": -1.0, "depth_std_m": -1.0,
                "extrinsic_position_std_m": -1.0, "extrinsic_orientation_std_rad": -1.0,
                "anchor_json": "", "simulation_only": False, "max_frames": 1000}
            for name, value in defaults.items(): self.declare_parameter(name, value)
            self.config = {name: self.get_parameter(name).value for name in defaults}
            c = self.config
            if not c["rectified_color"] or not c["map_is_true_enu"]:
                raise ValueError("Confirm rectified_color and map_is_true_enu after configuring rectification and map alignment")
            if any(not math.isfinite(c[k]) or c[k] <= 0 for k in ("depth_max_m", "depth_std_m", "extrinsic_position_std_m", "extrinsic_orientation_std_rad")):
                raise ValueError("Set measured depth range/noise and calibrated extrinsic uncertainty parameters")
            if c["depth_max_m"] <= c["depth_min_m"] or not 1 <= c["max_frames"] <= 100000:
                raise ValueError("Invalid depth bounds or max_frames")
            if not .5 <= c["navigation_rate_hz"] <= 10:
                raise ValueError("Navigation publication rate must be .5..10 Hz")
            self.mapper = LiveMapper(c["session_id"])
            self.station = Station(c["station"])
            self.station.request("/api/live/session", {"sessionId": c["session_id"], "mode": "replay" if c["simulation_only"] else "live"})
            self.anchor = read_json(Path(c["anchor_json"])) if c["anchor_json"] else None
            self.detector = None if c["no_detector"] else ReviewedDetector(Path(c["active_model"]), c["device"]) if c["active_model"] else LocalDetector(Path(c["model"]), device=c["device"])
            self.group_predictor = None
            if c["classify_groups"]:
                from vision.live_groups import load_group_predictor
                self.group_predictor = load_group_predictor()
            self.cv = CvBridge()
            self.tf = Buffer(cache_time=Duration(seconds=10))
            self.listener = TransformListener(self.tf, self)
            self.pending, self.stopping = queue.Queue(maxsize=1), threading.Event()
            self.navigation_pending = queue.Queue(maxsize=1)
            self.battery, self.gimbal_feedback = None, None
            self.navigation_dropped, self.navigation_updates = 0, 0
            self.navigation_last_warning, self.navigation_last_enqueue = -math.inf, -math.inf
            self.navigation_station = NavigationStation(c["station"])
            self.occupancy_mapper = None
            if c["navigation_enabled"]:
                from vision.live_occupancy import OccupancyMapper
                self.occupancy_mapper = OccupancyMapper(c["session_id"], resolution_m=c["navigation_resolution_m"],
                    size=(32,32,16), ttl_seconds=c["navigation_ttl_seconds"])
            self.battery_subscriber = self.create_subscription(BatteryState, c["battery_topic"], self.receive_battery, qos_profile_sensor_data)
            self.gimbal_subscriber = self.create_subscription(JointState, c["gimbal_feedback_topic"], self.receive_gimbal, qos_profile_sensor_data) if c["camera_mount_mode"] == "measured_gimbal" else None
            self.sequence, self.dropped, self.completed = 0, 0, 0
            self.subscribers = [message_filters.Subscriber(self, msg, topic, qos_profile=qos_profile_sensor_data)
                for msg, topic in ((RosImage, c["rgb_topic"]), (RosImage, c["depth_topic"]),
                                   (CameraInfo, c["info_topic"]), (Odometry, c["odom_topic"]))]
            self.sync = message_filters.ApproximateTimeSynchronizer(self.subscribers, queue_size=5, slop=.04)
            self.sync.registerCallback(self.receive)
            self.worker = threading.Thread(target=self.process, daemon=True)
            self.worker.start()
            self.navigation_worker = threading.Thread(target=self.process_navigation, daemon=True)
            self.navigation_worker.start()

        def receive_battery(self, message):
            self.battery = {"present": bool(message.present), "percentage": float(message.percentage), "timestampSeconds": seconds(message.header.stamp)}

        def receive_gimbal(self, message):
            try:
                positions = dict(zip(message.name, message.position))
                self.gimbal_feedback = {"timestampSeconds": seconds(message.header.stamp),
                    "positions": [float(positions[self.config["gimbal_yaw_joint"]]), float(positions[self.config["gimbal_pitch_joint"]])]}
            except (KeyError, TypeError, ValueError):
                self.gimbal_feedback = None

        def receive(self, color, depth, info, odometry):
            if self.stopping.is_set(): return
            try:
                c = self.config
                stamp = seconds(color.header.stamp)
                if not c["simulation_only"] and (abs(time.time() - stamp) > 2):
                    raise ValueError("Capture clock is not synchronized UTC or frame was already stale")
                if color.header.frame_id != c["optical_frame"] or odometry.header.frame_id != c["map_frame"]:
                    raise ValueError("Color optical frame or odometry ENU frame differs from configured calibration")
                if any(abs(seconds(message.header.stamp) - stamp) > .04 for message in (depth, info, odometry)):
                    raise ValueError("Sensor timestamps exceed 40 ms synchronization tolerance")
                if info.width != color.width or info.height != color.height or depth.width != color.width or depth.height != color.height or color.width * color.height > 8000000:
                    raise ValueError("Aligned color/depth/CameraInfo dimensions disagree")
                if depth.encoding not in {"16UC1", "32FC1"}:
                    raise ValueError("Aligned depth encoding must be 16UC1 millimeters or 32FC1 meters")
                pose = self.tf.lookup_transform(c["map_frame"], c["optical_frame"], Time.from_msg(color.header.stamp))
                if abs(seconds(pose.header.stamp) - stamp) > .04:
                    raise ValueError("Optical camera TF is not synchronized")
                position_std, orientation_std = covariance_uncertainties(odometry.pose.covariance)
                t, q = pose.transform.translation, pose.transform.rotation
                transform = quaternion_matrix(q.x, q.y, q.z, q.w, [t.x, t.y, t.z])
                camera = {"calibrationId": "ros-" + hashlib.sha256(json.dumps(list(info.p)).encode()).hexdigest()[:24],
                    "rectified": True, "intrinsics": {"fx": info.p[0], "fy": info.p[5], "cx": info.p[2], "cy": info.p[6]},
                    "cameraToMap4x4": transform, "poseTimestampSeconds": stamp,
                    "positionStdM": math.hypot(position_std, c["extrinsic_position_std_m"]),
                    "orientationStdRad": math.hypot(orientation_std, c["extrinsic_orientation_std_rad"])}
                metadata = {"alignedTo": "color", "calibrationId": camera["calibrationId"], "timestampSeconds": seconds(depth.header.stamp),
                    "method": "rgbd", "scaleToMeters": .001 if depth.encoding == "16UC1" else 1,
                    "minRangeM": c["depth_min_m"], "maxRangeM": c["depth_max_m"], "stdM": c["depth_std_m"]}
                item = (self.sequence, stamp, color, depth, camera, metadata)
                self.sequence += 1
                if self.occupancy_mapper is not None and stamp - self.navigation_last_enqueue >= 1 / c["navigation_rate_hz"]:
                    self.navigation_last_enqueue = stamp
                    nav_item = (stamp, depth, camera, metadata, odometry, self.battery, self.gimbal_feedback)
                    try: self.navigation_pending.put_nowait(nav_item)
                    except queue.Full:
                        try: self.navigation_pending.get_nowait()
                        except queue.Empty: pass
                        self.navigation_dropped += 1
                        self.navigation_pending.put_nowait(nav_item)
                try: self.pending.put_nowait(item)
                except queue.Full:
                    try: self.pending.get_nowait()
                    except queue.Empty: pass
                    self.dropped += 1
                    self.pending.put_nowait(item)
            except Exception as error:
                self.get_logger().warning(f"Rejected sensor observation: {error}")

        def process_navigation(self):
            # This worker never invokes either image-recognition model. Its queue,
            # occupancy state and short-timeout HTTP client are independent.
            while not self.stopping.is_set():
                try: stamp, depth, camera, metadata, odometry, battery, gimbal = self.navigation_pending.get(timeout=.5)
                except queue.Empty: continue
                try:
                    c = self.config
                    if odometry.child_frame_id != c["vehicle_frame"]:
                        raise ValueError("Odometry child frame must be the calibrated VEHICLE body origin")
                    if not c["simulation_only"] and time.time() - stamp > .5:
                        raise ValueError("Navigation observation aged beyond 500 ms before processing")
                    body, orientation, velocity = odometry.pose.pose.position, odometry.pose.pose.orientation, odometry.twist.twist.linear
                    telemetry = navigation_telemetry(timestamp_seconds=stamp,
                        vehicle_position=[body.x,body.y,body.z], vehicle_orientation_xyzw=[orientation.x,orientation.y,orientation.z,orientation.w],
                        linear_velocity=[velocity.x,velocity.y,velocity.z],
                        pose_covariance=list(odometry.pose.covariance), twist_covariance=list(odometry.twist.covariance),
                        camera_to_map=camera["cameraToMap4x4"], battery=battery,
                        camera_mount_mode=c["camera_mount_mode"], gimbal_feedback=gimbal)
                    values = self.cv.imgmsg_to_cv2(depth, desired_encoding="passthrough")
                    voxel_map = self.occupancy_mapper.update(values, camera=camera, depth_meta=metadata,
                        timestamp_seconds=stamp, map_revision=c["map_revision"])
                    if not c["simulation_only"] and time.time() - stamp > .5:
                        raise ValueError("Navigation map aged beyond 500 ms during processing; publication skipped")
                    telemetry["depthTimestampSeconds"] = metadata["timestampSeconds"]
                    packet = {"schema": "mycoscout.live-navigation.v1", "simulationOnly": c["simulation_only"],
                        "sessionId": c["session_id"], "mapRevision": c["map_revision"], "timestampSeconds": stamp,
                        "telemetry": telemetry, "voxelMap": voxel_map,
                        "provenance": {"source": "independent-synchronized-depth-navigation", "simulationOnly": c["simulation_only"],
                            "droppedPendingFrames": self.navigation_dropped, "cameraCalibrationId": camera["calibrationId"]},
                        "actuationEnabled": False}
                    self.navigation_station.publish(packet)
                    self.navigation_updates += 1
                except Exception as error:
                    now = time.monotonic()
                    if now - self.navigation_last_warning >= 5:
                        self.get_logger().warning(f"Navigation evidence unavailable: {error}")
                        self.navigation_last_warning = now

        def process(self):
            while not self.stopping.is_set():
                try: seq, stamp, color, depth, camera, metadata = self.pending.get(timeout=.5)
                except queue.Empty: continue
                try:
                    c = self.config
                    bgr = self.cv.imgmsg_to_cv2(color, desired_encoding="bgr8")
                    values = self.cv.imgmsg_to_cv2(depth, desired_encoding="passthrough")
                    rgb = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    started = time.monotonic()
                    candidates = self.detector.predict(rgb)[:100] if self.detector else []
                    if self.group_predictor is not None:
                        from vision.live_groups import classify_candidates
                        candidates = classify_candidates(rgb, candidates, self.group_predictor)
                    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    if not ok or len(encoded) > 8 * 1024 ** 2: raise ValueError("Camera JPEG exceeds frame limit")
                    raw = encoded.tobytes()
                    frame = {"frameId": f"ros-{seq:09d}", "sequence": seq, "timestampSeconds": stamp,
                        "simulationOnly": c["simulation_only"], "mapRevision": c["map_revision"],
                        "image": {"width": rgb.width, "height": rgb.height, "mimeType": "image/jpeg", "sha256": hashlib.sha256(raw).hexdigest(), "imageBase64": base64.b64encode(raw).decode()},
                        "camera": camera, "depth": metadata, "detections": candidates,
                        "detectorStatus": "completed" if self.detector else "unprocessed",
                        "detector": {**self.detector.info, "inferenceSeconds": time.monotonic() - started, "classifyGroups": c["classify_groups"]} if self.detector else None,
                        "provenance": {"source": "ros2-synchronized-rgbd", "captureTimeKnown": True, "trainingAllowed": False,
                            "replay": c["simulation_only"], "droppedPendingFrames": self.dropped, "poseSource": "external-tf-and-odometry"}}
                    if self.anchor: frame["georeference"] = self.anchor
                    packet = self.mapper.process(frame, depth=values, rgb=rgb)
                    self.station.request("/api/live/frames", packet)
                    self.completed += 1
                    if self.completed >= c["max_frames"]: self.stopping.set()
                except Exception as error:
                    self.get_logger().error(f"Frame processing failed: {error}")

    rclpy.init()
    node = None
    try:
        node = Bridge()
        while rclpy.ok() and not node.stopping.is_set():
            rclpy.spin_once(node, timeout_sec=.2)
    finally:
        if node:
            node.stopping.set()
            node.worker.join(timeout=2)
            node.navigation_worker.join(timeout=2)
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
