# X500 flight behavior simulation report

Generated: 2026-10-26T14:49:36.883Z

**12/12 scenarios passed.** No drone was connected or flown. All images/candidates in this flight test are synthetic event records; vision tests are separate.

| Scenario | Result | Captures | Final state |
|---|---|---:|---|
| Two stations with three settled image captures each | PASS | 6 | COMPLETE |
| Observed 3D corridor detours around a trunk | PASS | 12 | COMPLETE |
| Branch intersects rotor envelope despite clear endpoints | PASS | 0 | HOLD |
| Missing depth behind foliage blocks the corridor | PASS | 0 | HOLD |
| Depth stream becomes stale | PASS | 0 | HOLD |
| Loss of localization requests takeover without a position proposal | PASS | 0 | HOLD |
| Fresh map header cannot hide expired voxel evidence | PASS | 0 | HOLD |
| Hover, gather distinct observations, approach through a known same-height corridor | PASS | 1 | CANDIDATE_VERIFY |
| Old semantic detections cannot authorize approach while live depth stays healthy | PASS | 0 | CANDIDATE_VERIFY |
| Ground mushroom never causes a descent command | PASS | 0 | CANDIDATE_VERIFY |
| Manual takeover remains latched | PASS | 0 | MANUAL |
| Abort does not invent a return path | PASS | 0 | ABORTED |

## Camera geometry (illustrative calibration)

5 cm fronto-parallel mushroom cap; focal length 1,380 pixels. Pixel width is a geometric estimate, not detector accuracy.

| Range | Cap pixels | 2px blur budget speed at 8ms exposure |
|---:|---:|---:|
| 0.5 m | 138.0 | 0.091 m/s |
| 1 m | 69.0 | 0.181 m/s |
| 2 m | 34.5 | 0.362 m/s |
| 4 m | 17.3 | 0.725 m/s |
| 8 m | 8.6 | 1.449 m/s |

At 1 m, increasing exposure from 8 ms to 40 ms reduces the transverse-motion blur budget from 0.181 m/s to 0.036 m/s. The controller caps survey speed at 0.3 m/s and also applies the blur limit. Captures require an additional stop and settle.

## Limits

- Synthetic free-space maps and synthetic candidate evidence, not observed woodland or detector measurements.
- Ideal kinematics only; PX4 SITL, ROS 2, OAK-D and Jetson integration have not been executed.
- No sockets, serial ports, arm commands or flight commands are sent.
- A forward camera alone does not establish coverage behind, above, below or beside the aircraft.
- Declared-free voxels in the fixture are ground truth supplied by the simulator. Real mapping must explicitly raycast freespace and preserve unknown cells.
