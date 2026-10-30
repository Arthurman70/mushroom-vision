# Connected workflow — v0.8

Open `http://127.0.0.1:8787/live.html` for Mission control. The same page ships in
the Android companion and Quest web bundle. Phone and headset must connect to the
same station over trusted HTTPS; localhost on a phone means the phone itself.
Use **Connection** to enter the station address and its access token. The token
is held in memory, never exported or stored in browser storage.

## Try the connected workflow

1. Choose **Collected mushroom video**, then **Play recording**. Original frames,
   actual saved detector boxes and proposed persistent track IDs appear together.
   The twelve currently collected frames have no depth or georeferencing. Their
   candidates intentionally have no 3D/GPS positions. The existing group classifier
   adds tentative guesses from sufficiently large native crops. **Fresh detection +
   group guesses** reruns recognition from decoded video; the optional color contrast
   pass costs additional processing time. Fresh detection uses the explicitly
   activated experimental detector when one exists; stored replay preserves the
   detector that originally produced its observations.
2. Choose **Simulated depth-camera example** to see calibrated projection and
   repeated views of the same synthetic landmark, including position uncertainty.
   This is a geometry fixture, not a mushroom detection result.
3. Select **Try inspection simulation**, then **Step simulation** through approach,
   aiming, settling, requesting a photograph and acknowledging its simulated capture.
   The example contains explicit observed free-space evidence. Ordinary replay
   cannot invent that evidence. Queue, pause, reorder, cancel or retry tasks.
4. Select a candidate, enter a reviewer name, and save a classification. The record
   is shared with phone and headset. A human classification does not create depth,
   GPS, training permission or a complete detection annotation.
5. In **A measured learning loop**, prepare reviewed data. The page shows training,
   validation and test counts and explains why a dataset is too small. Training
   and choosing an experimental model are separate explicit actions.

The video grader offers a training-permission checkbox only for eligible footage.
Complete annotations, source permission and source-wide split rules are all
required. Current evaluation-only starter footage remains reserved for evaluation.
See [Training pipeline](../TRAINING_PIPELINE.md) for the export/training workflow.

## Sensor integration

[Live sensors](LIVE_SENSORS.md) documents executable adapters for:

- Local recordings and USB cameras, with actual GroundingDINO inference or the
  approved compact detector; optional experimental color enhancement.
- Synchronized RGBD recordings with original video hash, camera calibration,
  capture-time poses, explicit depth uncertainty and optional geographic anchor.
- ROS 2 RGB, aligned depth, CameraInfo, pose covariance and optical-frame TF from
  a separately installed RTAB-Map/odometry system.
- Human track evaluation and native-photo capture acknowledgment.

The station's original-frame view uses JPEG snapshots with detection metadata.
**Continuous video preview** accepts a separately configured WHEP/WebRTC endpoint.
Continuous preview is also rendered on the Quest board. Detection boxes remain on
their own timestamped snapshot, rather than being drawn on an unrelated video time.
The optional [local video preview service](LOCAL_VIDEO_PREVIEW.md) uses a verified
MediaMTX release and the actual collected recording. Its listeners are restricted
to this computer. Physical-camera and phone/headset media transport remain untested.

A separate ROS depth worker publishes conservative observed free volume and measured
vehicle, camera and battery state through `/api/live/navigation`. Its bounded queue
and network client run independently from the slower mushroom detector. Evidence
expires, and unobserved or uncertain cells remain blocked. This runs at a requested
5 Hz; hardware timing and flight behavior remain unvalidated.

The 3D view shows measured surface samples and local landmarks. Sparse points do
not establish free flight corridors or safe walking ground. GPS landmarks require
a true-ENU geographic anchor; estimated pins include uncertainty. The phone can
open such a coordinate in its map. The Quest view is a shared overview. Neither
screen currently aligns an outdoor AR marker to the exact physical mushroom.

## Tasks and photographs

Nearby located candidates automatically enter the inspection queue. A measured
range whose uncertainty reaches 50 m becomes a habitat lead. Unlocated candidates
can be queued, but movement proposals remain blocked until their locations and
observed free space are available. Human rejection cancels outstanding tasks.

Real capture upload uses `/api/live/captures`: image bytes must match their hash,
dimensions, pending request, session, landmark, map revision, timestamp, camera
pose and quality evidence. The native image is persisted in **Review finds** before
the executor receives its acknowledgment. If the task or session changes while
the image saves, the photo stays in the review inbox but a conflict response
prevents false task completion. The adapter counts only an exact matching task,
request and saved-photo completion. The capture adapter calculates sharpness;
operator/sensor claims about exposure pose and native framing still require device
validation. An undersized target remains a request for optical zoom or human closeup.

See [Live missions](LIVE_MISSIONS.md) for the task state machine and restricted
PX4 SITL bridge. The station has no physical-aircraft actuation endpoint. The SITL
bridge is a separate, explicit, localhost-only simulator transport, with process
identity checks, frame alignment and independent setpoint/watchdog loops. It never
arms or takes off a vehicle. Its transport tests use mocks; PX4/Gazebo is absent
from this Windows machine.

## What was and was not established

Actual saved detections were replayed over twelve real frames, and actual inference
was run on two frames decoded from the original video. Their observed processing
time is not fast enough for obstacle avoidance. No woodland detection percentages
or tracking accuracy are established by those runs. The synthetic sequence tests
geometry, and the flight fixture tests behavior; neither validates physical flight.

The current reviewed export contains one eligible image and five boxes. Its fixed
source partition is test, leaving train and validation empty. A real retraining
job is refused until enough independent reviewed data exists. The synthetic
gradient/checkpoint test validates the toolchain only. Model activation is an
experimental operator choice, not proof it beats the baseline.

Remaining device work is RGB/depth/IMU synchronization, camera extrinsics, map/GNSS
alignment, occupancy calibration, physical-camera streaming, Jetson speed
measurement, PX4/Gazebo integration, phone/Quest hardware testing and supervised
aircraft validation. The new ROS navigation worker is separately tested with
synthetic geometry; it is not a validated flight obstacle-avoidance system.
Stabilization and collision checks must remain onboard and independent of the UI.

No user grades were rewritten and no new detector was activated during this build.

## Primary integration references

- [RTAB-Map ROS 2](https://github.com/introlab/rtabmap_ros) and its
  [RGBD launch interface](https://github.com/introlab/rtabmap_ros/blob/ros2/rtabmap_launch/launch/rtabmap.launch.py).
- [PX4 offboard control](https://docs.px4.io/v1.17/en/flight_modes/offboard).
- [MediaMTX WebRTC features](https://mediamtx.org/docs/features/webrtc-specific-features).
