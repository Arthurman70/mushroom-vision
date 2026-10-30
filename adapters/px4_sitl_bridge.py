"""Opt-in PX4 SITL position transport; never arms, takes off, or targets hardware.

Requires a separately running PX4 SITL process on this Linux host, a verified
local frame alignment, and fresh simulation observations/proposals from the
station. Mock-tested; actual PX4/Gazebo execution is not claimed.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler


class GuardError(ValueError):
    pass


def vector(value):
    return isinstance(value, list) and len(value) == 3 and all(type(n) in (int, float) and math.isfinite(n) for n in value)


def local_url(value):
    parsed = urlsplit(value)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path != "/api/live/state":
        raise GuardError("Only http://127.0.0.1:<port>/api/live/state is supported")
    return value


def verify_sitl_process(pid, proc_root=Path("/proc")):
    """An arbitrary MAVLink heartbeat is not proof of a simulator. Require the
    local PX4 SITL executable, simulator environment, and its own UDP socket."""
    if type(pid) is not int or pid <= 1:
        raise GuardError("An explicit local PX4 SITL PID is required")
    process = proc_root / str(pid)
    try:
        executable = (process / "exe").resolve(strict=True)
        parts = executable.parts
        if tuple(parts[-4:]) != ("build", "px4_sitl_default", "bin", "px4"):
            raise GuardError("PID is not the PX4 px4_sitl_default build")
        environment = (process / "environ").read_bytes().split(b"\0")
        if b"PX4_SIM_MODEL=gz_x500" not in environment:
            raise GuardError("PX4 process must explicitly run PX4_SIM_MODEL=gz_x500")
        socket_ids = set()
        for entry in (process / "fd").iterdir():
            try:
                target = os.readlink(entry)
                if target.startswith("socket:["):
                    socket_ids.add(target[8:-1])
            except OSError:
                continue
        # PX4 instance zero owns the SDK socket at 14580 and sends to 14540.
        owned = False
        for line in (proc_root / "net" / "udp").read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 9 and fields[9] in socket_ids:
                local, remote = fields[1], fields[2]
                if local in ("0100007F:38F4", "00000000:38F4") and remote in ("0100007F:38CC", "00000000:0000"):
                    owned = True
        if not owned:
            raise GuardError("SITL process does not own its expected local SDK UDP socket")
        return {"pid": pid, "executable": str(executable), "model": "gz_x500", "udpPort": 14580}
    except (OSError, RuntimeError) as error:
        raise GuardError("Cannot verify local Linux PX4 SITL process") from error


@dataclass(frozen=True)
class Alignment:
    session_id: str
    map_revision: str
    origin_enu: tuple

    @classmethod
    def parse(cls, value):
        if value.get("schema") != "mycoscout.sitl-alignment.v1" or value.get("validatedInSimulator") is not True or value.get("mapAxesAlignedWithEnu") is not True:
            raise GuardError("Explicit validated simulator ENU alignment is required")
        if not vector(value.get("px4LocalOriginInMapEnuM")) or not value.get("sessionId") or not value.get("mapRevision"):
            raise GuardError("Simulator alignment is incomplete")
        return cls(value["sessionId"], value["mapRevision"], tuple(value["px4LocalOriginInMapEnuM"]))

    def to_ned(self, position_enu, yaw_enu):
        if not vector(position_enu) or not math.isfinite(yaw_enu):
            raise GuardError("Finite position and yaw required")
        east, north, up = [position_enu[i] - self.origin_enu[i] for i in range(3)]
        return north, east, -up, (90 - math.degrees(yaw_enu)) % 360

    def from_ned(self, position_ned):
        north, east, down = position_ned
        return [east + self.origin_enu[0], north + self.origin_enu[1], -down + self.origin_enu[2]]


def eligible_proposal(state, alignment, now_ms):
    if state.get("sessionId") != alignment.session_id or state.get("mapRevision") != alignment.map_revision:
        raise GuardError("Session or map alignment changed")
    if state.get("mode") != "replay" or state.get("simulationDemo") is True or state.get("latestFrame", {}).get("simulationOnly") is not True:
        raise GuardError("Only a sensor-fed SITL session is accepted; UI teleport demos and live aircraft are excluded")
    action = state.get("lastAction") or {}
    age = now_ms - action.get("generatedAtMs", 0)
    if not 0 <= age <= 1000:
        raise GuardError("Flight proposal is stale")
    if action.get("actuationEnabled") is not False:
        raise GuardError("Input must be a non-actuating reference proposal")
    task = next((t for t in state.get("tasks", []) if t.get("id") == action.get("taskId")), None)
    if not task or task.get("status") != "running" or task.get("mapRevision") != alignment.map_revision:
        raise GuardError("Task is no longer running in this coordinate frame")
    if action.get("type") not in ("move_proposal", "aim_proposal", "settle", "capture_request", "awaiting_capture_ack"):
        raise GuardError("Proposal requires stopping offboard")
    return action


class ProposalFollower:
    """Bounded waypoint following, separate from image inference and HTTP I/O."""
    def __init__(self, alignment, max_speed=.5, interval=.1):
        self.alignment, self.max_step = alignment, max_speed * interval
        self.signature, self.index = None, 1

    def setpoint(self, state, current_ned, current_yaw_deg, now_ms):
        action = eligible_proposal(state, self.alignment, now_ms)
        current = self.alignment.from_ned(current_ned)
        yaw = math.radians(90 - current_yaw_deg)
        if action["type"] == "move_proposal":
            route = action.get("waypointsEnuM")
            if not isinstance(route, list) or not 2 <= len(route) <= 1000 or not all(vector(p) for p in route):
                raise GuardError("A bounded verified waypoint route is required")
            signature = (action["taskId"], json.dumps(route))
            if signature != self.signature:
                if math.dist(route[0], current) > .35:
                    raise GuardError("Simulator pose disagrees with the route start; verify frame alignment")
                self.signature, self.index = signature, 1
            while self.index < len(route) - 1 and math.dist(current, route[self.index]) < .1:
                self.index += 1
            target = route[self.index]
            d = math.dist(current, target)
            next_point = [current[i] + (target[i] - current[i]) * min(1, self.max_step / max(d, 1e-9)) for i in range(3)]
            if abs(target[2] - current[2]) > .25:
                raise GuardError("This transport does not authorize altitude changes")
            return self.alignment.to_ned(next_point, yaw)
        if action["type"] == "aim_proposal":
            yaw = action.get("aim", {}).get("yawRad")
            if type(yaw) not in (float, int) or not math.isfinite(yaw):
                raise GuardError("Aim yaw is missing")
        # Pitch remains a camera/gimbal adapter task; no false camera ACK is sent.
        return self.alignment.to_ned(current, yaw)


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise GuardError("Station redirects are not accepted")


def read_state(url, token):
    headers = {"Authorization": "Bearer " + token} if token else {}
    with build_opener(NoRedirect()).open(Request(local_url(url), headers=headers), timeout=.7) as response:
        data = response.read(16 * 1024 * 1024 + 1)
    if len(data) > 16 * 1024 * 1024:
        raise GuardError("Station state exceeds its size limit")
    return json.loads(data)


async def run(args, sdk=None):
    if not args.enable_sitl_offboard:
        raise GuardError("No connection made: --enable-sitl-offboard is required")
    verify_sitl_process(args.sitl_pid)
    alignment = Alignment.parse(json.loads(args.alignment.read_text()))
    local_url(args.state_url)
    if sdk is None:
        try:
            from mavsdk_grpc import System
            from mavsdk_grpc.offboard import PositionNedYaw
        except ImportError:
            from mavsdk import System
            from mavsdk.offboard import PositionNedYaw
        drone, position_type = System(), PositionNedYaw
    else:
        drone, position_type = sdk
    await drone.connect(system_address="udpin://127.0.0.1:14540")
    state, sensors = {}, {}
    follower = ProposalFollower(alignment)
    active = False
    async def collect(name, stream, transform=lambda value: value):
        async for value in stream:
            sensors[name] = (transform(value), time.monotonic())
    async def poll_station():
        while True:
            try:
                new = await asyncio.to_thread(read_state, args.state_url, os.environ.get(args.token_env, ""))
                state.clear();state.update(new)
            except Exception:
                state.clear()
            await asyncio.sleep(.2)
    tasks = [asyncio.create_task(poll_station()),
        asyncio.create_task(collect("connection", drone.core.connection_state(), lambda x: x.is_connected)),
        asyncio.create_task(collect("position", drone.telemetry.position_velocity_ned(), lambda x: (x.position.north_m, x.position.east_m, x.position.down_m))),
        asyncio.create_task(collect("attitude", drone.telemetry.attitude_euler(), lambda x: x.yaw_deg)),
        asyncio.create_task(collect("health", drone.telemetry.health(), lambda x: x.is_local_position_ok and x.is_global_position_ok and x.is_home_position_ok)),
        asyncio.create_task(collect("armed", drone.telemetry.armed())),
        asyncio.create_task(collect("in_air", drone.telemetry.in_air()))]
    last_verification = 0
    try:
        while True:
            started = time.monotonic()
            try:
                if started - last_verification > 1:
                    verify_sitl_process(args.sitl_pid);last_verification = started
                for key in ("position", "attitude"):
                    if key not in sensors or started - sensors[key][1] > .5:
                        raise GuardError("Fresh simulator telemetry required")
                for key in ("connection", "health", "armed", "in_air"):
                    if key not in sensors or sensors[key][0] is not True:
                        raise GuardError("Simulator must already be connected, localized, armed and airborne by its operator")
                setpoint = follower.setpoint(state, sensors["position"][0], sensors["attitude"][0], time.time() * 1000)
                await asyncio.wait_for(drone.offboard.set_position_ned(position_type(*setpoint)), timeout=.2)
                if not active:
                    # A timed-out RPC may still have reached PX4. Attempt stop on
                    # any uncertain start outcome instead of leaving it active.
                    active = True
                    await asyncio.wait_for(drone.offboard.start(), timeout=1)
            except Exception as error:
                if active:
                    # Exit offboard; never land, disarm, or arm. If HOLD is denied,
                    # report it and exit instead of pretending a stop succeeded.
                    await asyncio.wait_for(drone.offboard.stop(), timeout=1);active = False
                    print(json.dumps({"event": "sitl_offboard_stopped", "reason": str(error)}), flush=True)
            await asyncio.sleep(max(0, .1 - (time.monotonic() - started)))
    finally:
        if active:
            await asyncio.wait_for(drone.offboard.stop(), timeout=1)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enable-sitl-offboard", action="store_true")
    parser.add_argument("--sitl-pid", type=int, required=True)
    parser.add_argument("--alignment", type=Path, required=True)
    parser.add_argument("--state-url", default="http://127.0.0.1:8787/api/live/state")
    parser.add_argument("--token-env", default="MYCOSCOUT_TOKEN")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (GuardError, ImportError, OSError) as error:
        parser.exit(2, str(error) + "\n")
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
