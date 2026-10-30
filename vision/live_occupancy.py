"""Bounded, short-lived occupancy from calibrated depth and external camera pose.

No SLAM, motor control or inferred free space beyond observed depth. A single
thin ray is insufficient to clear a whole voxel: every sampled pixel in the
expanded voxel's projected footprint must see farther than its far face.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from vision.live_mapping import identifier, number, validate_camera, validate_depth


class OccupancyMapper:
    def __init__(self, session_id, *, resolution_m=.5, size=(32, 32, 16), ttl_seconds=1.0,
                 max_voxels=100000, max_endpoint_samples=4096):
        self.session_id = identifier(session_id, "sessionId")
        self.resolution = number(resolution_m, "resolution", .05, 2)
        if len(size) != 3 or any(type(n) is not int or not 2 <= n <= 128 for n in size):
            raise ValueError("Occupancy size requires three integer dimensions in 2..128")
        if type(max_voxels) is not int or not 1 <= max_voxels <= 100000 or math.prod(size) > max_voxels:
            raise ValueError("The bounded volume must contain no more than max_voxels (100000 maximum)")
        self.size = tuple(size)
        self.ttl = number(ttl_seconds, "voxel lifetime", .05, 1.5)
        if type(max_endpoint_samples) is not int or not 1 <= max_endpoint_samples <= 16384:
            raise ValueError("Invalid endpoint sample budget")
        self.max_endpoint_samples = max_endpoint_samples
        self.cells, self.revision, self.last_timestamp = {}, None, None
        self.indices = np.indices(self.size).reshape(3, -1).T
        self.signs = np.array(list(itertools.product((-1., 1.), repeat=3)))
        self.last_report = None

    def update(self, depth, *, camera, depth_meta, timestamp_seconds, map_revision):
        started = time.perf_counter()
        stamp = number(timestamp_seconds, "timestamp", 0)
        revision = identifier(map_revision, "mapRevision")
        if self.last_timestamp is not None and stamp <= self.last_timestamp:
            raise ValueError("Occupancy depth/pose timestamps must increase")
        shape = np.shape(depth)
        if len(shape) != 2 or shape[0] < 1 or shape[1] < 1 or shape[0] * shape[1] > 8000000:
            raise ValueError("Depth image must be bounded and two dimensional")
        height, width = shape
        camera, transform = validate_camera(camera, width, height, stamp)
        values = validate_depth(depth, depth_meta, width, height, stamp, camera)
        if self.revision != revision:
            self.cells = {}
        self.revision, self.last_timestamp = revision, stamp
        origin_index = np.floor(transform[:3, 3] / self.resolution).astype(int) - np.array(self.size) // 2
        origin = origin_index * self.resolution
        upper = origin_index + np.array(self.size)
        # World lattice keys allow a rolling volume without changing map axes.
        # Cells outside the bounded window and expired evidence are forgotten.
        self.cells = {key: cell for key, cell in self.cells.items()
                      if stamp - cell[1] <= self.ttl and all(origin_index[a] <= key[a] < upper[a] for a in range(3))}
        uncertain = camera["positionStdM"] > self.resolution / 2 or camera["orientationStdRad"] > .1 or depth_meta["stdM"] > self.resolution / 2
        if uncertain:
            # Unbounded endpoint uncertainty must not trigger huge dilation
            # work or allow an old free map to validate a poorly localized body.
            self.cells = {}
            values[:] = np.nan
        centers = origin + (self.indices + .5) * self.resolution
        rotation, translation = transform[:3, :3], transform[:3, 3]
        distances = np.linalg.norm(centers - translation, axis=1)
        # Expand all sides for pose translation and angular uncertainty before
        # projection, then retain a 3σ depth margin beyond the furthest corner.
        margins = 3 * (camera["positionStdM"] + camera["orientationStdRad"] * (distances + math.sqrt(3) * self.resolution))
        corners = centers[:, None, :] + (self.resolution / 2 + margins)[:, None, None] * self.signs[None, :, :]
        optical = (corners - translation) @ rotation
        z = optical[:, :, 2]
        k = camera["intrinsics"]
        valid_z = (z.min(axis=1) >= depth_meta["minRangeM"]) & (z.max(axis=1) + 3 * depth_meta["stdM"] <= depth_meta["maxRangeM"])
        safe_z = np.maximum(z, 1e-9)
        us = optical[:, :, 0] / safe_z * k["fx"] + k["cx"]
        vs = optical[:, :, 1] / safe_z * k["fy"] + k["cy"]
        bounds = np.stack((np.floor(us.min(axis=1)), np.floor(vs.min(axis=1)), np.ceil(us.max(axis=1)), np.ceil(vs.max(axis=1))), axis=1)
        candidates = np.flatnonzero(valid_z & (bounds[:, 0] >= 0) & (bounds[:, 1] >= 0) & (bounds[:, 2] < width) & (bounds[:, 3] < height))
        free = set()
        for index in candidates:
            x0, y0, x1, y1 = bounds[index].astype(int)
            patch = values[y0:y1 + 1, x0:x1 + 1]
            # np.min returns NaN if even one footprint sample is unknown.
            if patch.size and float(np.min(patch)) > float(z[index].max()) + 3 * depth_meta["stdM"]:
                free.add(tuple(int(v) for v in self.indices[index] + origin_index))

        # Endpoints are occupancy evidence. Sparse endpoint sampling reduces
        # cost but never creates free cells, because free checks inspect the full
        # projected depth footprint independently.
        stride = max(1, math.ceil(math.sqrt(values.size / self.max_endpoint_samples)))
        vv, uu = np.mgrid[0:height:stride, 0:width:stride]
        zz = values[::stride, ::stride]
        valid = np.isfinite(zz)
        optical_points = np.stack(((uu[valid] - k["cx"]) * zz[valid] / k["fx"],
                                   (vv[valid] - k["cy"]) * zz[valid] / k["fy"], zz[valid]), axis=1)
        points = optical_points @ rotation.T + translation
        occupied = set()
        if points.size:
            ranges = np.linalg.norm(optical_points, axis=1)
            radius = 3 * (camera["positionStdM"] + camera["orientationStdRad"] * ranges + depth_meta["stdM"])
            low = np.maximum(np.floor((points - radius[:, None]) / self.resolution).astype(int), origin_index)
            high = np.minimum(np.floor((points + radius[:, None]) / self.resolution).astype(int), upper - 1)
            retained = np.all(low <= high, axis=1)
            low, high = low[retained] - origin_index, high[retained] - origin_index + 1
            # Box dilation via a 3D difference array avoids expanding the same
            # uncertain box thousands of times. Cost is bounded by eight index
            # updates per endpoint plus one pass over the configured volume.
            differences = np.zeros(tuple(n + 1 for n in self.size), dtype=np.int32)
            for bits in itertools.product((0, 1), repeat=3):
                corner = [high[:, axis] if bits[axis] else low[:, axis] for axis in range(3)]
                np.add.at(differences, tuple(corner), -1 if sum(bits) % 2 else 1)
            counts = differences.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)
            occupied_indices = np.argwhere(counts[:-1, :-1, :-1] > 0) + origin_index
            occupied = {tuple(int(v) for v in index) for index in occupied_indices}
        # A recent occupied observation wins over later free rays until it ages
        # out. This avoids clearing a thin obstacle with an inconsistent frame.
        for key in free:
            if self.cells.get(key, (None,))[0] != "occupied":
                self.cells[key] = ("free", stamp)
        for key in occupied:
            self.cells[key] = ("occupied", stamp)
        voxels = {",".join(str(key[a] - int(origin_index[a])) for a in range(3)):
                  {"state": cell[0], "observedMs": cell[1] * 1000}
                  for key, cell in self.cells.items()}
        self.last_report = {"freeCells": sum(cell[0] == "free" for cell in self.cells.values()),
                            "occupiedCells": sum(cell[0] == "occupied" for cell in self.cells.values()),
                            "unknownCells": math.prod(self.size) - len(voxels),
                            "footprintChecks": len(candidates), "endpointSamples": len(points),
                            "uncertaintyBlocked": uncertain,
                            "processingSeconds": time.perf_counter() - started}
        return {"frame": "map_enu", "unit": "m", "mapRevision": revision,
                "origin": dict(zip(("x", "y", "z"), origin.tolist())), "resolutionM": self.resolution,
                "size": dict(zip(("x", "y", "z"), self.size)), "updatedMs": stamp * 1000, "voxels": voxels,
                "evidence": {"schema": "mycoscout.observed-depth-volume.v1", "freeSpaceMethod": "full_projected_voxel_footprint",
                             "sessionId": self.session_id, "calibrationId": camera["calibrationId"],
                             "depthTimestampSeconds": depth_meta["timestampSeconds"], "poseTimestampSeconds": camera["poseTimestampSeconds"],
                             "positionStdM": camera["positionStdM"], "orientationStdRad": camera["orientationStdRad"],
                             "depthStdM": depth_meta["stdM"], "ttlSeconds": self.ttl,
                             "unknownIsBlocked": True, "report": self.last_report}}
