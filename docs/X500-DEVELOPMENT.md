# X500 mushroom survey development

This increment adds an executable **software simulation** for the Holybro X500 / OAK-D S2 / Jetson architecture. It does not connect to, arm or fly an aircraft. The companion software and physical sensor integration have not been deployed or tested on those devices.

## Run and inspect

From the `mycoscout` directory:

```powershell
node flight/run-simulation.mjs
node --test --test-isolation=none tests/flight.test.mjs
```

`flight/reports/latest.json` is the read-only UI interface. It contains `schemaVersion`, `simulationOnly`, `actuationEnabled`, `summary`, `cameraGeometry`, `exposureComparison`, and `scenarios`. Each scenario has its expected/final state, pass result, compact transition/capture events, and optional route. `flight/reports/proposed-setpoints.jsonl` retains every simulation step. `flight/reports/LATEST.md` is the human-readable result. These reports contain no real mushroom detector scores or flight measurements.

## Behavior implemented

`SurveyMission` in `flight/mission.mjs` progresses from DISARMED to PREFLIGHT and READY only after explicit simulation events and healthy input. A separate `begin_simulation` event starts the route. Its outputs always say `simulationOnly: true` and `actuationEnabled: false`.

At each survey station it stops, rotates to three camera headings and settles for an uninterrupted interval before requesting a synchronized image set. It turns to face the next segment while stationary, then translates at the lower of the configured speed and the image-motion limit. The stopping envelope includes an assumed reaction time and deceleration; these values must eventually be measured with the fully loaded aircraft.

A suspected mushroom causes verification at the current location. Three distinct, recent frames with accepted image quality and metric depth are a **development rule**, not a scientifically validated detection threshold. Detections remain unverified candidates. An explicitly requested verification viewpoint is allowed only through observed free 3D volume, with minimum standoff, limited distance and unchanged altitude. Seeing something on the ground never generates a blind descent command. Candidate evidence must stay fresh during an approach.

Each observation must carry its own original capture timestamp and associated capture pose; substituting the aircraft's later pose is rejected. The simple simulation pose is position plus yaw. A real camera projection adapter still needs calibrated full six-degree-of-freedom pose and extrinsics, synchronization and covariance propagation. The flight recorder already stores timestamp/position/yaw on each capture event. Unlocated Facebook images have neither metric depth nor drone capture pose, so they cannot become approach targets directly.

Stale depth, uncertain/stale localization, old map data, exhausted battery reserve, missing manual link, missing voxel evidence or a blocked corridor latch HOLD. Recovery needs an explicit `resume` event and fresh valid inputs. If localization is unreliable, no position-hold proposal is emitted; the result requests manual takeover. A HOLD is a software decision, not a promise that a real aircraft can physically hold position. ABORTED and MANUAL remain terminal. Abort does not invent a safe return-to-home path beneath branches.

## Full-volume planning

`flight/geometry.mjs` uses metre-based `map_enu` 3D voxels. Missing voxels mean unknown, and only explicit fresh `free` evidence allows traversal. Each segment is checked against voxel boxes inflated by the aircraft envelope, clearance and pose uncertainty. The complete segment is checked, including cells touched at boundaries. This blocks branches above the centerline and obstacles between endpoints. It is deliberately conservative and does not reuse the ground walking map.

The bounded six-neighbour A* planner supports 3D motion but keeps altitude fixed unless vertical movement is explicitly enabled within an observed altitude band. Its simplification checks every replacement segment. The trunk-detour scenario reserves body + uncertainty + clearance + stopping distance + arrival tolerance during planning; the mission rechecks motion against the newest map every step. Search exhaustion returns no route.

The voxel fixture's 0.5 m cells and refreshed ground-truth free space make the test reproducible. They do not demonstrate that a real camera sees thin twigs or clears a forest canopy. Production mapping needs probabilistic occupancy, ray-cast freespace, collision-relevant cell age, uncertainty-aware registration and a suitable resolution/performance budget. No-return depth is unknown, not free. A point-cloud surface alone cannot establish a free corridor.

## Near-to-far vision work and the flight connection

The camera model uses the pinhole relation `target_pixels = fx_pixels × target_diameter_m / camera_range_m`. `config/x500-simulation.json` deliberately marks its intrinsics **illustrative-unmeasured**. With the example focal length of 1,380 pixels, a 5 cm cap spans about 69 pixels at 1 m, 35 at 2 m, 17 at 4 m and 9 at 8 m, before occlusion or perspective losses. The 32-pixel design objective is not detector performance. `validatedDetectorMinPixels` stays null until held-out measurements exist.

Use existing close photographs to establish whether the model finds mushrooms at all. Split by original photograph/observation before creating derivatives so the same source does not appear in both training and testing. Test progressively smaller target-pixel bins and save negative scenes of leaves, bark, rocks and litter. Synthetic downscaling measures scale sensitivity; it cannot recreate actual distant depth, lighting, lens behavior, undergrowth or camera angle. It must not be reported as a measured flight distance.

Then collect measured-distance scenes with the exact OAK camera at 0.5, 1, 2, 4 and 8 m where visibility permits, with the aircraft stationary and unpowered. Record cap diameter, occlusion, camera angle, bounding boxes, illumination, focus, gain/exposure and hard negatives. Nearby handheld photographs are useful labels, but their detail does not prove the drone can see the same fungus above leaf litter. Fit the maximum working range only from held-out recall/false-positive curves by size and conditions. Update survey station spacing and revisit decisions from those results.

The motion approximation is `blur_pixels = fx × exposure_seconds × (transverse_speed / range + abs(angular_rate))`. With the example camera, 8 ms exposure and a 2 px budget, transverse speed is limited to 0.181 m/s at 1 m. At 40 ms it becomes 0.036 m/s. This is why the first behavior stops to look, and why dim woodland sharply reduces scanning efficiency. The equation excludes rolling-shutter distortion, focus changes and vibration; each needs measurement. The current simulation has one configured exposure value: a future hardware adapter must supply actual per-frame exposure and gate predictions against it.

The separate local vision experiment measured approximately **6.62 seconds per full image on the Windows CPU**; its first seven-view tiled scene took **40.7 seconds for inference**. These are desktop measurements, not Jetson benchmarks. Both exceed the simulation's 2,000 ms candidate-evidence age gate. The `delayed-semantic-result` scenario injects a 6,620 ms delay while depth remains current and confirms that approach is refused. The age gate is not relaxed to accommodate a slow model.

Keep semantic inference in an asynchronous queue keyed by capture ID and capture pose, separate from the fast depth/localization/obstacle loop. Old predictions may be reviewed as unlocated or stale candidate history; reacquire fresh synchronized evidence before any approach. Compare onboard acceleration and smaller/tiled models on actual hardware before choosing a latency budget. Tiling can add detections and false positives; a higher box count is not an accuracy improvement. No reliable distant-positive detector curve has been established, so `validatedDetectorMinPixels` remains null.

## Hardware and integration target

Use the actual X500 kit's Pixhawk for stabilization and a separately powered companion for vision and planning. PX4 documents the X500 v2/Pixhawk 6C development kit. Measure the complete build's mass, center of gravity, propeller envelope, power draw and flight margin with the selected battery; the bare kit's figures do not establish payload flight performance. [PX4 X500 guide](https://docs.px4.io/main/en/frames_multicopter/holybro_x500v2_pixhawk6c)

The OAK-D S2 supplies a high-resolution RGB camera and stereo depth over USB; Luxonis offers auto-focus and fixed-focus variants. A standard front-facing camera cannot provide simultaneous rear, lateral, upward and downward obstacle coverage. Those unknown sectors remain blocked in the planner until valid evidence exists. An additional downward range sensor and more directional depth coverage may be needed, but their integration is not implemented. [Luxonis OAK-D S2](https://shop.luxonis.com/products/oak-d-s2)

For the proposed **Jetson Orin Nano**, JetPack 6.2 provides a documented Ubuntu 22.04 baseline; this is distinct from the older Jetson Nano. This is a compatibility baseline, not a claim that 6.2 is the newest release or that inference throughput has been benchmarked. [NVIDIA JetPack 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62)

ROS 2 Humble targets Ubuntu 22.04 arm64, with scheduled support through May 2027. Plan a migration before that support window ends. The current PX4 guide offers Humble on 22.04, while recommending Jazzy on 24.04 for new general setups. Preserve the tested JetPack/OS/ROS combination rather than mixing package generations. [ROS target platforms](https://raw.githubusercontent.com/ros-infrastructure/rep/master/rep-2000.rst), [PX4 ROS 2 setup](https://docs.px4.io/main/en/ros2/user_guide)

The integration manifest in `config/x500-integration-plan.json` declares logical camera/depth/odometry streams and frame contracts. Its topic names are proposed remaps, not verified driver defaults. Luxonis maintains an official ROS driver and a Humble branch; resolve and record the exact tested source revision during bench setup. [Luxonis driver repository](https://github.com/luxonis/depthai-ros), [Humble branch](https://github.com/luxonis/depthai-ros/tree/humble)

PX4's uXRCE-DDS architecture is a suitable future telemetry bridge. Match `px4_msgs` definitions to the selected firmware. ENU/optical-frame versus NED/FRD transforms, time synchronization, odometry resets, QoS and offboard-loss handling need explicit integration tests. This increment intentionally has no control bridge. A future controller must meet PX4's continuous offboard proof-of-life and loss behavior, and still preserve independent pilot takeover. [PX4 ROS 2 architecture](https://docs.px4.io/main/en/ros2/user_guide), [PX4 offboard requirements](https://docs.px4.io/main/en/flight_modes/offboard)

## Remaining deployment work

1. Camera calibration, physical mounting, reliable power and thermal testing on the actual hardware.
2. Recorded walk-through sequences with synchronized RGB, depth, intrinsics, IMU, pose/covariance and transforms. Reject invalid depth instead of filling it as open space.
3. Detector evaluation on held-out woodland scenes and measured ranges; add model latency to the perception-age and stopping budgets.
4. A real 3D freespace mapper and localizer with reset/drift detection. Semantic cliffs, water and muck remain separate unvalidated ground-planning inputs.
5. PX4 SITL integration and failure injection, then props-off hardware tests. The current ideal-kinematics simulator does not replace these.
6. Supervised open-area flight characterization and conservative sensor coverage tests before any woodland maneuvering.

No existing phone/headset build becomes flight-ready by consuming this report. The report is useful for reviewing behavior and planning evidence collection now.
