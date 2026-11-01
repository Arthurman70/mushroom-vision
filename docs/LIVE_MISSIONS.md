# Shared mission queue and reference executor

The live station keeps tracked mushroom candidates, human reviews, frame metadata,
and an inspection queue in `data/live-mission/live-mission.sqlite` (the server may
choose a different directory). Phone and headset clients use the same service.
SQLite WAL transactions make a queue edit and its revision durable together.

The station produces **flight proposals only**. It has no MAVLink, serial,
arming, PX4 connection, or physical actuation transport. A separate opt-in
Linux PX4 SITL transport is provided below. The desktop executor is
a conservative reference for replay and simulation, not a real-time flight loop.
Sensor ingestion, task review and queue commands are usable independently of a
physical aircraft. A pending task is never evidence that a drone has flown it.

## Operator workflow

1. Start a replay or live session. Sessions keep independent maps and queues.
2. Stream canonical frame packets from the replay or camera adapter.
3. Review candidate identities and their location uncertainty. RGB-only tracks
   remain unlocated; they do not gain GPS pins or free-space maps.
4. Queue **inspect**, **photograph**, or **scan** for a candidate. Nearby located
   candidates automatically receive an inspection proposal. Candidates whose
   measured range plus three standard deviations reaches 50 m, or which have
   large position uncertainty, stay in the habitat lead list. Color/habitat-only
   leads remain habitat leads even if nearby.
5. Pause, cancel, resume/retry, or reorder tasks. A confirmed/rejected/uncertain
   human review is kept with the landmark. Rejection cancels its pending tasks.

An inspect or photograph task approaches an observed viewpoint, aims, settles,
requests a native photograph, and waits for a matching capture acknowledgment.
The current scan task executes one investigation view at its selected habitat
lead; it does not claim complete area coverage or autonomous exploration.

To run the deterministic clear-route and blocked-route demonstration:

```powershell
node flight/run-live-simulation.mjs
```

The report is `flight/reports/live-executor-latest.json`. All demonstration
positions, free-space voxels and photograph acknowledgments are explicitly
synthetic. `simulation://synthetic-photo-no-real-image` is not a captured image.
The station's explicit **simulation demo** control can step the same sequence.
Normal video replay never invents a camera acknowledgment.

## Service contract

`LiveMissionStore(directory, {now, executorOptions})` has asynchronous
`initialize()` and synchronous methods after initialization:

| Method | Result / purpose |
| --- | --- |
| `snapshot()` | Active state, session list, connection age and capability flags |
| `startSession({sessionId,mode,baseRevision?})` | Select/create `replay` or `live`; modes cannot be changed |
| `ingestFrame(packet)` | `{duplicate,state}`; exact duplicate is a no-op |
| `updateNavigation(packet)` | Independent fresh depth/pose/telemetry update and executor step; does not refresh old detection timestamps |
| `enqueue({type,landmarkId,idempotencyKey,baseRevision?})` | `{task,duplicate,state}` |
| `updateTask(id,{action,beforeTaskId?,baseRevision?})` | Updated state; actions pause/resume/cancel/retry/move |
| `review(id,{status,groupId?,note?,reviewer?,baseRevision?})` | Updated landmark review; not automatically training data |
| `tick({captureAck?,baseRevision?})` | `{action,state}`; proposes one executor step |
| `seedSimulation({blocked?,baseRevision?})` | Explicit new synthetic demo session |
| `stepSimulation({baseRevision?})` | Apply only the prior active synthetic proposal and advance one step |
| `close()` | Close the database |

Clients should supply `sessionId` and `baseControlRevision` for user mutations.
The separate control revision changes for operator edits, new automatic tasks,
and task lifecycle changes; ordinary high-rate sensor updates do not invalidate
an operator's pending edit. Legacy `baseRevision` is still supported. A conflict is
HTTP 409 through `ApiError`; re-read state instead of overwriting another client.
Retrying the same enqueue key returns the original task even after a newer
revision. Reusing that key for different work is an error. A session switch
pauses its running task; it does not move a command to the new session.

State includes `latestFrame`, bounded `frames`, `landmarks`, `tracks`, `tasks`,
`telemetry`, display-only surface `map.points`, optional `voxelMap`, `lastAction`,
and bounded `events`. `realFlightSupported`, `actuationEnabled`, and the physical
flight capability flags are false. Limits are 64 sessions, 100,000 deduplicated
frames per session, 120 recent frame summaries, 1,000 landmarks, 300 queued/history
tasks, 300 recent events, and 20,000 display surface points per packet. A full
queue removes old terminal task details first; an unresolved queue at capacity
rejects new work. The idempotency journal retains up to 10,000 accepted operator
command IDs per session, then requires a new session instead of forgetting keys.

## Frame evidence and clocks

The canonical schema is `mycoscout.live-frame.v1`; positions use metres in
`map-enu`. Packets include a session, frame ID, increasing sequence and capture
timestamp, explicit `simulationOnly`, map revision, image digest/dimensions,
detections, tracks, landmarks and map surface points. The HTTP image receiver
must validate the actual bytes and SHA-256 before setting a local `image.url`.
The mission store does not persist image base64. Original image filenames and
display URLs are not camera or depth calibration evidence.

For a metric landmark, supply calibrated intrinsics, a valid rigid 4×4
`cameraToMap4x4` transform, pose uncertainty, and depth evidence with matching
calibration and timestamps within 50 ms. The camera is optical x-right, y-down,
z-forward; the supplied transform explicitly maps it to east/north/up.
Measured range must agree with the camera and landmark positions. Without those
inputs, positions and map points must be absent. GPS additionally requires a
georeference anchor, source, heading/position uncertainty and nonzero uncertainty
on every geographic pin. A human classification does not improve GPS accuracy.

Replay uses its recorded relative capture clock. Live capture timestamps must
use synchronized UTC epoch seconds and cannot be ahead of reception by more
than 100 ms. Delayed detector outputs can be displayed for review, but the
executor compares sensor timestamps with actual wall time. A six-second-old
frame never becomes fresh flight evidence merely because it just arrived.
Hardware capture timestamps must come from calibrated camera exposure timing;
a USB frame-receipt timestamp alone does not establish exposure/IMU sync.

Surface points are **not** free space. Flight proposals additionally need:

```json
{
  "flight": {
    "telemetry": {
      "positionEnuM": [-4, 0, 1.5],
      "timestampSeconds": 100,
      "batteryPercent": 85,
      "speedMps": 0,
      "yawRad": 0,
      "pitchRad": 0,
      "localized": true,
      "depthValid": true,
      "depthTimestampSeconds": 100
    },
    "voxelMap": {
      "frame": "map_enu", "unit": "m", "mapRevision": "map-1",
      "origin": {"x": -6, "y": -4, "z": -1},
      "resolutionM": 0.5, "size": {"x": 24, "y": 16, "z": 12},
      "updatedMs": 100000,
      "voxels": {"0,0,0": {"state": "free", "observedMs": 100000}}
    }
  }
}
```

This example has only one observed free cell and cannot support an approach.
Absent cells remain unknown. Each corridor edge checks the entire aircraft
volume against occupied, unknown or expired cells. Defaults require navigation
and depth evidence no older than 1.5 seconds and a stationary target reacquired
within 10 seconds, at least 30% battery, a 0.65 m vehicle
clearance radius inflated by target uncertainty during approach, and a 2 m
target standoff. These are conservative reference settings, not certified X500
performance parameters. A map revision change blocks current tasks until
reacquisition and an explicit retry. The planner keeps the existing observed
flight altitude; it never descends blindly toward a ground mushroom.

## Independent observed-depth navigation

`vision/live_occupancy.py` implements `OccupancyMapper(session_id)` with
`update(depth, camera=..., depth_meta=..., timestamp_seconds=..., map_revision=...)`.
It uses the same synchronized, rectified, color-aligned depth/camera validation as
landmark mapping. It does not estimate pose or create SLAM measurements.

The default rolling volume is 32×32×16 voxels at 0.5 m resolution. World lattice
coordinates remain fixed as the bounded window follows the camera. A voxel is
free only if its entire projected footprint, expanded for pose uncertainty,
contains valid depth beyond its farthest corner plus a depth uncertainty margin.
A single ray crossing a voxel does not clear the unobserved remainder. Surface
endpoints and their uncertainty volume are occupied; occluded regions, pixels
with missing depth, the area behind the camera, and space beyond measured depth
stay unknown. Recent occupied evidence wins over conflicting free observations.
Cells expire after one second; a map revision discards the prior volume.

Endpoint dilation uses a bounded 3D difference array. A 640×480 synthetic plane
test took approximately 19 ms on this workstation with the default 16,384-cell
volume. This measures this geometry calculation only, not a guaranteed Jetson
frame rate, complete navigation latency, or obstacle-avoidance performance.
The hard volume limit is 100,000 cells. Excessive sensor uncertainty produces an
unknown-only volume instead of unsupported clearance.

The independent ROS navigation worker posts this envelope to
`POST /api/live/navigation`:

```json
{
  "schema": "mycoscout.live-navigation.v1",
  "sessionId": "the-active-session",
  "mapRevision": "the-active-map-revision",
  "timestampSeconds": 100,
  "simulationOnly": true,
  "telemetry": {},
  "voxelMap": {}
}
```

The empty objects above illustrate the envelope only. Supply the actual mapper
output and measured telemetry: vehicle body position, calibrated optical camera
position/aim and mount mode, speed, localization, depth timestamp and battery.
The service validates the occupancy provenance, clock, frame, bounds and mode.
Navigation may arrive before the first slow semantic frame. Older detections
can add mushroom evidence without overwriting newer vehicle pose or free space.
Each accepted navigation packet independently advances the proposal executor;
it never makes a stale mushroom observation look newly detected.

A forward-facing camera may leave the vehicle's swept clearance volume unknown,
especially behind or beside it. That legitimately blocks movement until other
observed views cover the volume. No automatically cleared bubble is placed
around the aircraft to force a route through unknown space. Camera aim is
computed from the optical camera origin, while vehicle clearance and movement
use the aircraft body origin. Capture evidence must preserve that distinction.

## Photograph acknowledgment

The request includes task/session/landmark/map IDs, request time, viewpoint,
camera yaw/pitch and a target native-frame width fraction of 0.7. The camera
worker must save its native image before returning the matching request IDs,
`capturedMs`, `photoRef`, actual `photoSha256`, native dimensions,
`nativeUncropped:true`, capture pose/aim, `settled:true`, and a measured quality
decision (`sharpnessAccepted`). An optional native-pixel target box determines
whether framing reached the requested fraction. Cropping does not count as
new image detail. Insufficient framing yields a saved-photo result that calls
for optical zoom or an operator close-up rather than moving inside standoff.

Live sessions reject synthetic acknowledgments. Replay acknowledges only
explicit synthetic capture evidence. Request identity, pose, aim, timestamp,
settling, quality and SHA syntax are checked by the executor. The transport's
trusted camera worker must independently verify that the referenced photo was
actually saved: the queue reports `independentlyVerified:false` because a JSON
assertion alone cannot establish this. No acknowledgment means no completed
photograph task; requests time out and block after eight seconds.

## Optional PX4 SITL transport

`adapters/px4_sitl_bridge.py` adds a real MAVSDK transport boundary for a local
PX4 software simulator. Its interfaces and control loop are mock-tested; PX4,
Gazebo and MAVSDK were not available in this Windows environment, so an actual
simulated flight is **not yet validated**. Physical flight is unsupported.

Run the station, this adapter and a separately installed PX4/Gazebo simulator in
the same Linux host/PID namespace. The adapter requires an already-running
`build/px4_sitl_default/bin/px4` process with `PX4_SIM_MODEL=gz_x500`, verifies its
owned SDK UDP socket through `/proc`, and binds MAVSDK only to
`udpin://127.0.0.1:14540`. There is no arbitrary vehicle URL, serial-device or
remote-host option. Install MAVSDK-Python in that Linux environment; the adapter
supports the current `mavsdk_grpc` and existing `mavsdk` import layouts.

First prepare and independently check an alignment file for the exact simulator
session. Do not copy these example origin values without measuring them:

```json
{
  "schema": "mycoscout.sitl-alignment.v1",
  "sessionId": "your-sitl-sensor-session",
  "mapRevision": "your-sitl-map-revision",
  "validatedInSimulator": true,
  "mapAxesAlignedWithEnu": true,
  "px4LocalOriginInMapEnuM": [0, 0, 0]
}
```

Then invoke it explicitly with the verified PX4 process ID:

```sh
python -m adapters.px4_sitl_bridge --enable-sitl-offboard \
  --sitl-pid 12345 --alignment simulator-alignment.json
```

The simulator operator must already have armed and taken off inside the
simulator. This adapter never arms, takes off, lands or disarms. It consumes only
a sensor-fed replay/SITL session, excludes live-aircraft sessions and the UI's
teleport demo, and requires current running-task proposals plus a matching map
revision. It does not create sensor frames, verified free space, camera pitch
movement, photographs or capture acknowledgments. Those need their own simulator
sensor/camera adapters; absent evidence leaves the queue blocked.

The 10 Hz setpoint loop runs separately from station HTTP polling and any image
inference, moves by bounded increments along verified waypoints, and converts
ENU position/yaw to PX4 NED using the explicit origin. Route-start disagreement
or unapproved altitude changes stop execution. Stale proposals, missing fresh
position/attitude, changed session/map, a paused task, or loss of localization
trigger an offboard stop. Stop failures are errors; the adapter does not claim
that HOLD succeeded if PX4 refuses it. Configure and validate PX4's own offboard
loss behavior before running this integration.

The APIs and simulator port were checked against the official
[MAVSDK position example](https://github.com/mavlink/MAVSDK-Python/blob/main/examples/offboard_position_ned.py)
and [PX4 simulation guide](https://docs.px4.io/main/en/simulation/).

## Remaining physical integration

First validate the complete sensor/map/task/camera loop against actual PX4 SITL,
including obstacle and link/localization-loss cases. Then validate synchronized
sensor capture and observed-free-volume production on the companion computer.
Its fast control and obstacle loops must run independently of semantic
inference, streaming and headset polling. A separately validated hardware
transport and tested aircraft/body/camera transforms are still required;
connecting this desktop reference directly to motors would bypass those checks.

## Combining full-frame and periodic detail scans

Use full-frame observations for broad search, with periodic or requested tiled
detail scans for inspection. Preserve full-frame coverage for fungi on logs,
stumps and tree bases outside the selected ground rectangle. The rectangle is
an image-space selection, not a measured ground surface. Color contrast and wood
context are habitat cues; they do not confirm a mushroom or identify a species.
Keep those cues separate from mushroom detections unless their cue type survives
the entire mapping and routing path. The current mapper labels its detection
inputs as mushroom candidates, so adding plain log/color boxes to that list
would misrepresent them.

Each result must retain the original sensor capture timestamp, synchronized pose
and depth. Record inference duration and completion time separately. A tiled
scan that takes 30 seconds is still a 30-second-old observation when it finishes;
changing its timestamp to completion would invalidate its geometry and freshness.
The live queue can retain that result for review, but the default target
reacquisition limit is 10 seconds, so it cannot dispatch an approach from that
stale target evidence. Obtain a fresh target observation and fresh navigation
before retrying a blocked inspection. Newly arriving navigation alone does not
refresh an old mushroom observation or automatically retry a blocked task.

Semantic packets are accepted in capture-time order. A sequential hybrid scanner
preserves this order. A future parallel scanner must coordinate result ordering
or use a separate delayed-review channel for an older tile result that finishes
after a newer full-frame result; it must not invent a newer capture time or pose.
Independent navigation packets continue to use their own capture clock, and an
older semantic frame cannot replace a newer body pose or observed free-space map.
The hybrid search does not change the obstacle-sensing backend or its freshness
requirements.

Allowing for processing time matters even at a seemingly low speed: **0.2 m/s
for 30 seconds covers 6 metres**. This is an illustration of observation delay,
not a recommended flight speed. Display actual processing and capture age when
considering a survey pace; speed alone cannot establish safe clearance or ensure
that a small target stays visible. The existing blur model additionally depends
on measured exposure, focal length, range and angular motion.

There is no above-ground-height measurement in the current mission contract.
Map-ENU altitude is not height above local terrain, and a lower image region is
not an altitude sensor. No fixed low-altitude command or automatic descent is
created by the hybrid search. The reference executor preserves its current
observed altitude and standoff; future terrain-following needs calibrated ground
geometry and independent clearance validation.

Routing remains unchanged: calibrated candidates whose range uncertainty reaches
50 metres or more stay habitat follow-ups; candidates entirely below that
boundary may receive an immediate investigation task. Unknown range remains
unlocated, regardless of box size, vivid color or detector confidence. A queued
task is a proposal awaiting fresh evidence, not a confirmed mushroom or a flight
command. These behaviors are checked in `tests/hybrid-mission.test.mjs` using
synthetic sensor fixtures against the real live mission store.
