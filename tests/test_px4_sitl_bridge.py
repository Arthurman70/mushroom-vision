import asyncio
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import AsyncMock, patch

from adapters.px4_sitl_bridge import Alignment, GuardError, ProposalFollower, eligible_proposal, local_url, run, verify_sitl_process


def alignment_data():
    return {"schema": "mycoscout.sitl-alignment.v1", "sessionId": "sim", "mapRevision": "m1", "validatedInSimulator": True,
            "mapAxesAlignedWithEnu": True, "px4LocalOriginInMapEnuM": [0, 0, 0]}


def station_state(now=100000):
    return {"sessionId": "sim", "mapRevision": "m1", "mode": "replay", "simulationDemo": False,
            "latestFrame": {"simulationOnly": True}, "tasks": [{"id": "t1", "status": "running", "mapRevision": "m1"}],
            "lastAction": {"type": "move_proposal", "taskId": "t1", "actuationEnabled": False, "generatedAtMs": now,
                           "waypointsEnuM": [[0, 0, 1], [3, 0, 1], [3, 3, 1]]}}


class SITLContractTests(unittest.TestCase):
    def setUp(self):
        self.alignment = Alignment.parse(alignment_data())

    def test_enu_ned_and_yaw_conversion(self):
        self.assertEqual(self.alignment.to_ned([2, 3, 4], 0), (3, 2, -4, 90))
        self.assertAlmostEqual(self.alignment.to_ned([2, 3, 4], math.pi/2)[3], 0)
        self.assertEqual(self.alignment.from_ned((3, 2, -4)), [2, 3, 4])
        data = alignment_data();data["px4LocalOriginInMapEnuM"] = [10, 20, 30]
        self.assertEqual(Alignment.parse(data).to_ned([12, 23, 34], 0), (3, 2, -4, 90))

    def test_alignment_requires_explicit_verified_origin(self):
        for field in ("validatedInSimulator", "mapAxesAlignedWithEnu", "px4LocalOriginInMapEnuM"):
            data = alignment_data();del data[field]
            with self.assertRaises(GuardError): Alignment.parse(data)

    def test_endpoint_is_fixed_to_loopback_station(self):
        self.assertEqual(local_url("http://127.0.0.1:8787/api/live/state"), "http://127.0.0.1:8787/api/live/state")
        for url in ("http://192.168.1.2:8787/api/live/state", "https://example.com/api/live/state", "http://user:pass@127.0.0.1/api/live/state", "http://127.0.0.1/api/live/state?target=x"):
            with self.assertRaises(GuardError): local_url(url)

    def test_live_demo_stale_cancelled_and_changed_frame_are_rejected(self):
        for edit in (lambda s:s.update(mode="live"), lambda s:s.update(simulationDemo=True), lambda s:s.update(mapRevision="m2"),
                     lambda s:s["lastAction"].update(generatedAtMs=90000), lambda s:s["tasks"][0].update(status="cancelled"),
                     lambda s:s["lastAction"].update(type="hold")):
            state = station_state();edit(state)
            with self.assertRaises(GuardError): eligible_proposal(state, self.alignment, 100000)

    def test_bounded_motion_preserves_route_and_refuses_bad_alignment(self):
        follower = ProposalFollower(self.alignment)
        point = follower.setpoint(station_state(), (0, 0, -1), 90, 100000)
        self.assertEqual(point, (0, .05, -1, 90))
        other = ProposalFollower(self.alignment)
        with self.assertRaisesRegex(GuardError, "route start"):
            other.setpoint(station_state(), (0, 5, -1), 90, 100000)
        state = station_state();state["lastAction"]["waypointsEnuM"][1][2] = 3
        with self.assertRaisesRegex(GuardError, "altitude"):
            ProposalFollower(self.alignment).setpoint(state, (0, 0, -1), 90, 100000)

    def test_camera_aim_changes_yaw_without_claiming_pitch_capture(self):
        state = station_state();state["lastAction"].update(type="aim_proposal", aim={"yawRad": math.pi/2, "pitchRad": -.5})
        self.assertEqual(ProposalFollower(self.alignment).setpoint(state, (0, 0, -1), 90, 100000), (0, 0, -1, 0))

    def test_simulator_process_must_own_expected_socket(self):
        udp = "header\n0: 0100007F:38F4 0100007F:38CC 01 00000000:00000000 00:00000000 00000000 1000 0 12345\n"
        with patch.object(Path, "resolve", return_value=Path("/opt/PX4/build/px4_sitl_default/bin/px4")), \
             patch.object(Path, "read_bytes", return_value=b"PX4_SIM_MODEL=gz_x500\0"), \
             patch.object(Path, "iterdir", return_value=iter([Path("fd1")])), \
             patch("os.readlink", return_value="socket:[12345]"), patch.object(Path, "read_text", return_value=udp):
            self.assertEqual(verify_sitl_process(123)["model"], "gz_x500")
        with self.assertRaises(GuardError): verify_sitl_process(0)

    def test_no_connection_without_explicit_opt_in(self):
        args = NS(enable_sitl_offboard=False)
        with self.assertRaisesRegex(GuardError, "No connection made"):
            asyncio.run(run(args, sdk=(object(), tuple)))


class SITLLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_streams_setpoints_and_stops_on_cancellation_without_arm_calls(self):
        async def stream(value):
            while True:
                yield value
                await asyncio.sleep(.03)
        telemetry = NS(
            position_velocity_ned=lambda:stream(NS(position=NS(north_m=0, east_m=0, down_m=-1))),
            attitude_euler=lambda:stream(NS(yaw_deg=90)),
            health=lambda:stream(NS(is_local_position_ok=True, is_global_position_ok=True, is_home_position_ok=True)),
            armed=lambda:stream(True), in_air=lambda:stream(True))
        offboard = NS(set_position_ned=AsyncMock(), start=AsyncMock(), stop=AsyncMock())
        drone = NS(connect=AsyncMock(), offboard=offboard, telemetry=telemetry, core=NS(connection_state=lambda:stream(NS(is_connected=True))))
        alignment = NS(read_text=lambda:json.dumps(alignment_data()))
        args = NS(enable_sitl_offboard=True, sitl_pid=123, alignment=alignment, state_url="http://127.0.0.1:8787/api/live/state", token_env="UNUSED_SITL_TEST_TOKEN")
        with patch("adapters.px4_sitl_bridge.verify_sitl_process", return_value={"pid":123}), \
             patch("adapters.px4_sitl_bridge.read_state", side_effect=lambda *_:station_state(time.time()*1000)):
            with self.assertRaises(asyncio.TimeoutError):
                await asyncio.wait_for(run(args, sdk=(drone, lambda *args:args)), timeout=.35)
        drone.connect.assert_awaited_once_with(system_address="udpin://127.0.0.1:14540")
        self.assertGreaterEqual(offboard.set_position_ned.await_count, 2)
        offboard.start.assert_awaited_once();offboard.stop.assert_awaited_once()
        self.assertFalse(hasattr(drone, "action"))


if __name__ == "__main__":
    unittest.main()
