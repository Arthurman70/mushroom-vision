"""Import one RTAB-Map OccupancyGrid snapshot for review, never control a drone.

Uses only the standard library for JSON conversion. ROS dependencies are imported
only for --ros-topic. A 2D occupancy grid does not prove safe walking terrain.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import math
import os
from pathlib import Path
import time
from urllib import error, parse, request


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def convert_grid(message, *, expected_frame="map", occupied_threshold=65):
    """Convert a ROS JSON shape. Only occupancy 0 becomes free; ambiguity blocks."""
    if expected_frame != "map":
        raise ValueError("The application requires frame map; transform the source map explicitly upstream")
    if type(occupied_threshold) is not int or not 1 <= occupied_threshold <= 100:
        raise ValueError("occupied_threshold must be between 1 and 100")
    header, info = message["header"], message["info"]
    frame_id = header["frame_id"]
    if frame_id != expected_frame:
        raise ValueError(f"Frame {frame_id!r} does not match {expected_frame!r}; no transform was supplied")
    width, height = info["width"], info["height"]
    for name, value in (("width", width), ("height", height)):
        if type(value) is not int or not 2 <= value <= 256:
            raise ValueError(f"{name} must be an integer from 2 to 256; crop upstream without changing metric scale")
    resolution = _finite(info["resolution"], "resolution")
    if not 0.1 <= resolution <= 100:
        raise ValueError("resolution must be from 0.1 to 100 m; resample conservatively upstream")
    position, rotation = info["origin"]["position"], info["origin"]["orientation"]
    x, y, z = (_finite(position[key], f"origin.{key}") for key in ("x", "y", "z"))
    if max(abs(x), abs(y)) > 1e6:
        raise ValueError("Map origin is outside the supported local frame")
    if abs(z) > 1e-6:
        raise ValueError("Nonzero origin z is unsupported by this 2D import; transform upstream")
    qx, qy, qz, qw = (_finite(rotation[key], f"orientation.{key}") for key in ("x", "y", "z", "w"))
    if max(abs(qx), abs(qy), abs(qz)) > 1e-6 or abs(abs(qw) - 1.0) > 1e-6:
        raise ValueError("Rotated or invalid grid origin rejected; publish an axis-aligned map first")
    values = message["data"]
    if not isinstance(values, list) or len(values) != width * height:
        raise ValueError("data length must equal width * height")
    cells = []
    for value in values:
        if type(value) is not int or not -1 <= value <= 100:
            raise ValueError("Occupancy values must be integers in [-1, 100]")
        cells.append(0 if value == 0 else 1 if value >= occupied_threshold else -1)
    stamp = header.get("stamp", {})
    return {
        "frame": {"id": frame_id, "unit": "m"},
        "grid": {"width": width, "height": height, "resolution": resolution,
                 "origin": {"x": x, "y": y}, "cells": cells},
        "groundTraversabilityVerified": False,
        "source": {
            "adapter": "rtabmap-occupancy-snapshot",
            "receivedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rosStamp": stamp,
            "occupiedThreshold": occupied_threshold,
            "cellOrder": "row-major; x changes fastest; y increases by row",
            "obstacleMeaning": "Cell 1 means an occupied obstacle; tree identity is not inferred.",
            "limitations": "No cliff, water, muck, overhead clearance or walking-surface verification.",
        },
    }


def ros_snapshot(topic, *, timeout, expected_frame, max_age):
    """Listen for one fresh sample. Does not publish or call any ROS service."""
    try:
        import rclpy
        from nav_msgs.msg import OccupancyGrid
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    except ImportError as exc:
        raise RuntimeError("ROS 2 dependencies missing. Run from a sourced ROS 2 Humble environment.") from exc
    rclpy.init(args=[])
    node = rclpy.create_node("mycoscout_readonly_map_snapshot")
    samples = []
    try:
        def receive(msg):
            if samples:
                return
            stamp = msg.header.stamp
            age = (node.get_clock().now().nanoseconds - (stamp.sec * 1_000_000_000 + stamp.nanosec)) / 1e9
            if msg.header.frame_id != expected_frame:
                return
            if not 0 <= age <= max_age:
                return
            p, q = msg.info.origin.position, msg.info.origin.orientation
            samples.append({
                "header": {"frame_id": msg.header.frame_id,
                           "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec}},
                "info": {"width": msg.info.width, "height": msg.info.height,
                         "resolution": msg.info.resolution,
                         "origin": {"position": {"x": p.x, "y": p.y, "z": p.z},
                                    "orientation": {"x": q.x, "y": q.y, "z": q.z, "w": q.w}}},
                "data": list(msg.data),
            })
        # Best-effort volatile accepts both reliable and best-effort publishers.
        # It intentionally waits for a new publication instead of importing stale latched data.
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        node.create_subscription(OccupancyGrid, topic, receive, qos)
        deadline = time.monotonic() + timeout
        while not samples and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=min(0.2, max(0.0, deadline - time.monotonic())))
        if not samples:
            raise RuntimeError("No fresh map received. Check topic, frame, ROS clock and publisher; no upload occurred.")
        return samples[0]
    finally:
        node.destroy_node()
        rclpy.shutdown()


def upload_snapshot(payload, server, token):
    parts = parse.urlsplit(server)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("Server must be an http(s) URL")
    if parts.username or parts.password or parts.query or parts.fragment or parts.path not in ("", "/"):
        raise ValueError("Server URL must contain only scheme, host and optional port")
    try:
        loopback = ipaddress.ip_address(parts.hostname).is_loopback
    except ValueError:
        loopback = parts.hostname.lower() == "localhost"
    if not loopback and not token:
        raise ValueError("MYCOSCOUT_TOKEN is required for a non-loopback server")
    if not loopback and parts.scheme != "https":
        raise ValueError("Use HTTPS for a non-loopback server so its access token is protected")
    target = server.rstrip("/") + "/api/map"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    body = json.dumps(payload, allow_nan=False).encode("utf-8")
    req = request.Request(target, data=body, headers=headers, method="POST")
    class NoRedirect(request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            raise RuntimeError("Redirect refused; use the final trusted server URL")
    opener = request.build_opener(NoRedirect())
    try:
        with opener.open(req, timeout=15) as response:
            if not 200 <= response.status < 300:
                raise RuntimeError(f"Import rejected with HTTP {response.status}")
            response.read(65536)
    except error.HTTPError as exc:
        raise RuntimeError(f"Import rejected with HTTP {exc.code}; survey was not confirmed") from None
    except error.URLError:
        raise RuntimeError("Server connection failed; survey import was not confirmed") from None


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    source = result.add_mutually_exclusive_group(required=True)
    source.add_argument("--json", type=Path, help="Saved OccupancyGrid JSON using standard ROS field names")
    source.add_argument("--ros-topic", help="Capture one new nav_msgs/OccupancyGrid message, e.g. /rtabmap/map")
    result.add_argument("--output", required=True, type=Path, help="New snapshot JSON output (will not overwrite)")
    result.add_argument("--frame", default="map", choices=("map",), help="Required application frame; transform upstream instead of renaming")
    result.add_argument("--occupied-threshold", type=int, default=65)
    result.add_argument("--timeout", type=float, default=20, help="Maximum ROS wait in seconds")
    result.add_argument("--max-age", type=float, default=5, help="Maximum ROS message age in seconds")
    result.add_argument("--upload", action="store_true", help="POST the snapshot after saving it")
    result.add_argument("--replace-survey", action="store_true",
                        help="Acknowledge upload replaces the survey map and clears its existing sightings")
    result.add_argument("--server", default="http://127.0.0.1:8787")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.upload and not args.replace_survey:
            raise ValueError("--upload also requires --replace-survey because import clears existing survey sightings")
        if not args.frame or not all(math.isfinite(v) and v > 0 for v in (args.timeout, args.max_age)):
            raise ValueError("Frame must be nonempty and timeout/max-age must be finite positive numbers")
        if args.output.exists():
            raise ValueError("Output exists; choose a new snapshot filename")
        if args.json:
            if args.json.stat().st_size > 8_000_000:
                raise ValueError("Input exceeds the 8 MB snapshot limit")
            message = json.loads(args.json.read_text(encoding="utf-8-sig"))
        else:
            message = ros_snapshot(args.ros_topic, timeout=args.timeout,
                                   expected_frame=args.frame, max_age=args.max_age)
        payload = convert_grid(message, expected_frame=args.frame, occupied_threshold=args.occupied_threshold)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(f"Saved occupancy snapshot: {args.output}. Walking traversability remains unverified.")
        if args.upload:
            upload_snapshot(payload, args.server, os.environ.get("MYCOSCOUT_TOKEN", ""))
            print("Survey map replaced. Existing survey sightings were cleared.")
        return 0
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        print(f"Map import failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

