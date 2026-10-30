# Live and recorded sensor bridge

This release connects actual RGB detections to persistent candidate tracks, and
calibrated RGBD plus external camera poses to metric landmarks and a bounded 3D
surface map. It does not implement a new SLAM engine or claim aircraft flight.
RTAB-Map/another odometry system supplies the camera poses. The station's task
executor remains non-actuating and treats missing free-space evidence as blocked.

## Recorded images, with actual saved detector results

From the project directory, using the vision Python environment:

```powershell
.\runtime-vision\Scripts\python.exe -m adapters.live_replay --collection data/video-discovery-v1 --testbench-db data/testbench/testbench.sqlite --station http://127.0.0.1:8787 --session-id replay-first-pass --interval 0.5
```

The adapter creates a replay session before uploading. Use a new session ID for
each replay. `--emit` writes JSON packets to stdout for a parent process; `--output
FILE.jsonl` saves the same packet stream. Output files are never overwritten.
Optional `--classify-groups` adds the same bounded native-crop guesses described
below to real footage. It is off by default and prohibited for synthetic fixtures.

The twelve current frames have twelve completed original-RGB detector runs. The
replay selects a run only when frame ID, source ID, timestamp, image hash, image
dimensions and full-resolution settings match. Missing results are explicitly
`unprocessed`, not empty detector successes. Frame groups remain separate sessions
when a collection contains multiple videos. All original attribution is retained.

The observed twelve-frame check produced seven proposed track identities and no
metric locations. These are algorithmic associations, not human-verified tracks or
a detection-accuracy score. Sparse sampling, occlusion, similar mushrooms and rapid
camera motion can split identities or associate them incorrectly.

`simulationOnly:true` means replay execution cannot control a real aircraft; it does
not mean the original video images are generated. Real footage also carries
`provenance.synthetic:false` and `realRecording:true`.

## Actual video decoding or USB camera

OpenCV 4.10.0.84 is installed in the existing local runtime. The optional deployment
dependency is `opencv-python-headless==4.10.0.84`, alongside the existing vision
requirements. No camera is opened automatically by the station.

```powershell
.\runtime-vision\Scripts\python.exe -m adapters.live_capture --input data/archive-video-v1/originals/Freestonew-theHugeMushroom948.flv --max-frames 10 --sample-every 20 --classify-groups --station http://127.0.0.1:8787
.\runtime-vision\Scripts\python.exe -m adapters.live_capture --input 0 --max-frames 300 --classify-groups --station http://127.0.0.1:8787 --output data/my-camera-capture
```

`--output` creates a new directory containing native-size JPEG snapshots and packet
JSONL. Limits bound run duration and frame count. `--no-detector` sends a genuine
view/map feed without detections. `--color-sensitive` adds a second color-enhanced
detector pass and suppresses overlapping proposals; it is experimental and slower.
The original pixels are retained. `--active-model data/learning/active-model.json`
uses an explicitly activated reviewed detector after validating its checkpoint
hash; the baseline is used otherwise.

`--classify-groups` optionally loads the existing `groups-v1` classifier and sends
its broad-group guesses, scores and abstention reasons with each landmark to the
phone/VR view. It is off by default and adds computation. At most five candidates
per frame are classified, prioritizing detector score; each must span at least
32 original pixels in both dimensions. Smaller targets remain `unknown` with no
guesses. The native crop is passed to the existing model; its internal network
resizing never counts as additional source detail. A group guess cannot confirm
species, change target geometry, change tracking, authorize sales or enable flight.

Two frames decoded directly from the original FLV were run through the local
GroundingDINO detector in about 14.5 seconds total after model initialization.
This demonstrates actual decoding/inference, not real-time collision avoidance.
Camera previews are JPEG snapshots with metadata, not WebRTC video. USB timestamps
are host receipt times, not exposure synchronization; RGB-only USB feeds cannot
generate metric mushroom positions. Slow semantic detection must never drive the
aircraft stabilization or obstacle-avoidance loop.

## Synchronized recorded RGBD

`--sidecar sensor-session.json` is accepted for recorded files only. The sidecar
schema is `mycoscout.rgbd-sidecar.v1`, includes the exact video's `videoSha256`,
`coordinateFrame: "map-enu"`, a stable `mapRevision`, explicit `simulationOnly`,
and a `frames` list. Each item has:

- `sourceFrameIndex`, the decoded video index, and `timestampSeconds` from capture.
- `depthPath`, an aligned 2D numeric `.npy` (pickle disabled) or 16-bit PNG inside
  the sidecar directory.
- `camera`: `calibrationId`, `rectified:true`, `intrinsics:{fx,fy,cx,cy}`,
  `cameraToMap4x4`, `poseTimestampSeconds`, `positionStdM`, `orientationStdRad`.
- `depth`: `alignedTo:"color"`, matching `calibrationId`, `timestampSeconds`,
  `method:"rgbd"|"stereo"|"surveyed"`, `scaleToMeters`, calibrated `minRangeM`,
  `maxRangeM`, and measured `stdM`.
- Optional `georeference`: `datum:"WGS84"`, `mapAxes:"true-enu"`, origin
  `latitude`, `longitude`, `altitudeM`, `horizontalStdM`, `verticalStdM`,
  `headingStdRad`, `sourceId`, and capture-clock `timestampSeconds`.

The explicit camera transform maps optical coordinates (x right, y down, z forward)
to map ENU (east, north, up). It includes the calibrated camera-to-aircraft
extrinsics; a vehicle position alone cannot substitute for this transform. Images
must already be rectified and depth registered to their color pixels. Pose/depth
timestamps must match within 50 ms. Missing depth leaves a candidate unlocated;
invalid transforms, calibration, mismatched dimensions and stale timestamps fail.
Map-coordinate revision changes require a new session so old landmarks cannot be
silently mixed with a loop-closure-corrected coordinate system.

The center region of each box needs a majority of valid, consistent depths.
Ambiguous foreground/background mixtures abstain. Position covariance includes
pixel-center uncertainty, depth spread, camera position and angular uncertainty.
Repeated correlated frames do not falsely drive uncertainty toward zero. WGS84
conversion additionally includes anchor position and heading error and reports
an approximate 95% uncertainty bound, not exact GPS. Projections beyond the sensor's
declared useful range are discarded. Valid observations at 50 m or more become
habitat leads; closer ones receive immediate investigation priority.
If a small mushroom is unresolved by the depth sensor, a consistent patch can still
belong to the background. These are depth-projected candidates until a closer view
and human review confirm the target; uncertainty estimates cannot remove that
association error.

The map is a capped, voxel-deduplicated set of observed surfaces. It does not label
unseen space free, represent a certified collision map, or prove a route is safe.

## Synthetic geometry check

```powershell
.\runtime-vision\Scripts\python.exe -m adapters.live_replay --fixture examples/live-session-synthetic.json --station http://127.0.0.1:8787 --session-id geometry-check --interval 1
.\runtime-vision\Scripts\python.exe -m unittest adapters.tests.test_live_mapping -v
```

The fixture is visibly synthetic, with a constant depth plane and three known
camera positions. It reconstructs one stationary target at ENU `[0,4,1.1]` while
the camera moves east. This tests geometry and transport only, not mushroom
detection, SLAM accuracy, field GPS accuracy, or flight. No free corridor is added.

## ROS 2 / RTAB-Map companion bridge

`python -m adapters.live_ros2` is an executable optional ROS 2 node. It needs a
ROS 2 Linux environment with `rclpy`, `message_filters`, `tf2_ros`, `cv_bridge`,
`sensor_msgs`, `nav_msgs`, OpenCV, NumPy, Pillow and the detector dependencies.
RTAB-Map and the selected camera driver must already be running and calibrated.
This Windows machine has no ROS 2/camera/aircraft: the bridge's pure geometry and
covariance helpers are tested, but ROS transport and hardware are unverified.

Configure the actual camera topics; defaults are:

| Input | Default |
|---|---|
| Rectified color | `/camera/color/image_rect` |
| Depth aligned to color | `/camera/aligned_depth_to_color/image_raw` |
| Rectified camera calibration | `/camera/color/camera_info` |
| ENU-frame odometry and covariance | `/odometry/filtered` |
| Timestamped TF | `map` to `camera_color_optical_frame` |

Supply ROS parameters for `rectified_color:=true`, `map_is_true_enu:=true` only
after those properties are actually established; `depth_max_m`, `depth_std_m`,
`extrinsic_position_std_m`, and `extrinsic_orientation_std_rad` must come from the
camera specification/calibration and validation, not invented precision. Unknown
all-zero pose covariance is rejected. `anchor_json` is optional; without a measured
anchor there are no GPS coordinates. A new `session_id` and `map_revision` are
required after changing coordinates. Hardware capture clocks must agree with UTC.
Set the optional ROS parameter `classify_groups:=true` to send the same bounded
group guesses from synchronized RGBD captures; the default is false.

Color/depth/calibration/odometry are approximately synchronized within 40 ms, TF
is looked up at capture time, and the inference worker retains only the latest
pending frame. This prevents an unbounded backlog but cannot make a slow detector
real-time. The bridge publishes no motor commands and supplies no cleared-volume
claims from sparse surface points. Its separate navigation worker can supply the
conservative depth-derived occupancy evidence described below; headset commands
remain high-level queued tasks and aircraft actuation remains disabled.

### Independent navigation evidence

The ROS bridge has a second bounded, latest-frame worker for synchronized depth,
vehicle odometry, calibrated camera direction and measured battery state. It does
not call either recognition model. It publishes `/api/live/navigation` separately
from slow `/api/live/frames`, with an independent HTTP client and one-second network
timeout. The configured default is at most 5 updates/second, not a demonstrated
hardware rate. Frames older than 500 ms before or after map construction are
discarded in live mode. This is not a hard real-time flight-control guarantee.

The occupancy mapper checks complete projected voxel footprints against valid
aligned depth and uncertainty guards. Only observed space before a measured surface
can become free; missing depth, occlusions and unseen directions remain unknown.
Old evidence expires. Even a valid map can leave the aircraft's swept volume
unobserved, in which case the task remains blocked. No free corridor is invented to
make a demonstration fly.

Additional ROS inputs and parameters:

- `battery_topic` defaults to `/mavros/battery`, using actual
  `sensor_msgs/BatteryState` presence, fractional percentage and capture timestamp.
  Unknown percentage, absent or stale battery telemetry blocks navigation evidence.
- Odometry must name the calibrated `vehicle_frame` (default `base_link`) as its
  child frame. Body position comes from vehicle odometry; the camera's optical TF
  translation is stored separately. Measured linear velocity and nonzero finite
  pose/twist covariance are required; unknown all-zero covariance is rejected.
- `camera_mount_mode` defaults to `unknown`, so navigation proposals remain blocked
  until configured. Use `rigid_calibrated` only for a calibrated rigid mount. Use
  `measured_gimbal` only with actual synchronized `sensor_msgs/JointState` feedback
  at `gimbal_feedback_topic` (default `/gimbal/joint_states`), named
  `gimbal_yaw_joint` and `gimbal_pitch_joint`. Commanded gimbal angles do not supply
  measured orientation. Camera yaw/pitch is derived from the calibrated optical TF
  forward axis, with measured feedback required for a moving gimbal.
- `navigation_enabled:=false` disables this optional publisher. Other settings are
  `navigation_rate_hz` (0.5–10), `navigation_resolution_m` (default 0.5), and
  `navigation_ttl_seconds` (default 1). Fixed map dimensions are 32×32×16 voxels.

These safeguards and the pure telemetry conversion are tested locally. Actual ROS
topic connections, sensor calibration, map coverage, timing and woodland flight
still require hardware validation; the adapter publishes no PX4 actuation commands.

## Tracking evaluation against your grades

```powershell
.\runtime-vision\Scripts\python.exe -m adapters.evaluate_live_tracks --packets examples/live-session-recorded.jsonl --grades-db data/video-grades/video-grades.sqlite
```

This joins only exact video hashes, frame hashes, source IDs, timestamps and latest
human grades. Completely labeled frames provide one-to-one IoU detection metrics.
Human track links additionally provide earliest human-visible versus machine-matched
frame/pixel size, delay among reviewed samples, matched segments and identity
switches. Partial and uncertain grades are excluded; empty denominators stay null.
No human track links means `awaiting_human_track_links`; no complete reviews means
`awaiting_complete_labels`. Sampled-frame results cannot establish continuous-video
onset or an actual distance in metres.

## Real close-photo acknowledgment bridge

`adapters.live_photo` polls the authenticated live task queue and watches a bounded
folder for camera-written native images plus atomic `*.capture.json` evidence.
It does not take a fabricated pose from the requested viewpoint or mark an old
preview as a new photograph.

```powershell
.\runtime-vision\Scripts\python.exe -m adapters.live_photo --station http://127.0.0.1:8787 --folder data/camera-captures --min-sharpness 100 --duration-seconds 60
```

The example threshold is a starting setting, not a validated universal focus
criterion; calibrate it for the actual camera and target patch. Authentication uses
`MYCOSCOUT_TOKEN` like the other adapters. A sidecar must use
`schema:"mycoscout.capture-evidence.v1"`, `synthetic:false`, `nativeUncropped:true`,
the exact task `requestId`, `taskId`, `sessionId`, `landmarkId`, `mapRevision`, an
in-folder `imagePath`, the saved `photoSha256`, capture `capturedMs` and synchronized
`poseTimestampMs`, measured body `positionEnuM` and separate optical-camera
`cameraPositionEnuM` when required by the task, camera `yawRad` and `pitchRad`,
`targetBboxPixelsXYXY` in the native photo, and `motionSamples` containing measured
`timestampMs`/`speedMps` observations through exposure time. At least 800 ms of
settled motion must be present with no sampling gaps above 150 ms. This evidence
must come from the camera/vehicle integration; the bridge never supplies defaults.

The bridge computes target-patch luminance Laplacian variance directly from saved
pixels, checks checksum, pose, aim, timing and settling evidence, then uploads to
`/api/live/captures`. The station saves the photo before acknowledging completion.
Frame filling is evaluated from the measured target box; a photo that still needs
optical zoom/operator repositioning is not described as a close-up success. Real
camera capture transport and physical quality have not been tested on this machine.
