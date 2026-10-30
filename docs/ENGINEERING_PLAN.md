# Remaining work to meet the original request
This repository is a deployable development preview. Completing the requested woodland robot is a robotics project requiring access to the aircraft and sensors.

## Recommended architecture
Synchronized RGB/depth/IMU → calibrated odometry → RTAB-Map 3D mapping → onboard 3D trajectory planning → current PX4 flight-control integration. A separate pedestrian terrain map feeds the MycoScout station. Detector boxes are projected through calibrated depth and camera pose into the map, then reviewed. The phone and Quest consume that shared map after their individual coordinate frames have been registered.

Flight must not depend on phone, headset, Wi-Fi or remote language-model latency.

## Required implementation and acceptance work
1. **Choose the airframe and sensors.** Verify payload, power, coverage, thermal behavior and loaded endurance. Inspect used hardware physically.
2. **Record synchronized sensor bags on the ground.** Calibrate camera intrinsics/extrinsics and time offsets; measure depth failures on thin branches, low light and motion.
3. **Validate localization.** Measure drift under foliage and relocalization after tracking resets. Keep map identities, revisions and uncertainty explicit.
4. **Implement image-to-map projection.** A box centre alone is insufficient. Use calibrated depth, foreground segmentation, camera pose and an uncertainty estimate. Reject missing/ambiguous depth.
5. **Evaluate fungi models.** Collect region/season-specific ground-truth images and close-ups. Measure detection precision/recall vs altitude, illumination and motion; distinguish candidate detection from species classification. No edibility decisions.
6. **Build pedestrian terrain semantics.** Geometry alone cannot reliably distinguish muck, water or hazardous ground. Add appropriate sensing and labels; block unknown/drop-off regions. Validate on foot.
7. **Integrate a maintained 3D flight planner.** EGO-Swarm ROS2/Aerostack2 may be research starting points; review GPL implications, versions and hardware interfaces. Do not use archived PX4-Avoidance as if it were supported.
8. **Implement onboard failure handling.** Independent RC takeover, localization-loss behavior, link-loss behavior, geofencing, battery limits and obstacle sensing health. Automatic vertical return-to-home can be dangerous under canopy.
9. **Register AR coordinate frames.** Start with a surveyed fiducial or known correspondences, not GPS/VPS assumptions. Record map-to-device transforms, time, revision and uncertainty; invalidate after tracking loss.
10. **Progress through controlled testing.** Software simulation, hardware-in-loop, propeller-off tests, guarded/netted space, supervised open space, then controlled sparse woodland with required visibility/permissions. Set measured acceptance criteria before advancing.

## Open-source projects selected, not falsely merged
RTAB-Map and GroundingDINO have actual adapters here. Three.js is bundled in the frontend. EGO-Swarm, PX4, Aerostack2, OpenCV, ONNX Runtime and native ARCore are potential next components; their names in this plan do not mean they are installed or integrated.

- [Current PX4 ROS2 interface](https://docs.px4.io/main/en/ros2/px4_ros2_control_interface)
- [EGO-Swarm ROS2 branch](https://github.com/ZJU-FAST-Lab/ego-planner-swarm/tree/ros2_version)
- [Aerostack2](https://github.com/aerostack2/aerostack2)
- [Archived PX4-Avoidance](https://github.com/PX4/PX4-Avoidance)
- [Deprecated MAVLink trajectory interface](https://mavlink.io/en/services/trajectory.html)

These sources describe capabilities and research examples, not proof of a turnkey woodland mushroom drone.
