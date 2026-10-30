# Range-aware investigation and native close-up requests

This implementation changes simulation scheduling and inspection records. It sends no aircraft commands. Physical flight, actual camera capture, calibrated range accuracy and controller response remain untested.

## Route an observation

`flight/range-inspection.mjs` accepts synchronized calibrated metric position and capture pose, or an explicit measured range with value, uncertainty, measurement method, timestamp, source and calibration ID. Simulated metric fixtures are labeled as such. Range-only evidence never creates map coordinates. Archive pixels and assumed mushroom diameters cannot establish physical range. Inconsistent range and metric geometry are rejected as unknown.

| Evidence | Routing | Marking |
|---|---|---|
| Entire measured range interval below 50 m | `immediate_investigation` | Nearby visual candidate; human confirmation required |
| 50 m or farther, or uncertainty interval crosses 50 m | `habitat_followup` | Distant visual cue in habitat checker log |
| Color salience only, at any range | `habitat_followup` | Unconfirmed color contrast |
| Mushroom-like visual cue without range provenance | `unlocated_followup` | Request a fresh located observation |

`kind` and `originalKind` retain the incoming observation class. No branch sets a species, edible status or confirmed mushroom flag. A bounded `habitatLog` retains the distant/color observations even after the active queue expires; report writers save it to JSON. Duplicate source frames cannot refresh evidence. A new near observation of the same subject can promote its current route while retaining its earlier distant log entry.

## Inspect each target

Near candidates rank ahead of ordinary habitat context, regardless of their heuristic score. A nearby candidate pauses scouting at the next settled capture, not during an unchecked movement. The previous scout state is saved and resumes after the inspection or deferral.

The planner visits a viewpoint aimed at that individual target. It uses at most 2 m per approach segment, rechecks synchronized target evidence and the current 3D corridor, and preserves altitude. Full aircraft volume, braking clearance, map age, pose/depth age, battery, manual link, standoff, time and travel budgets remain mandatory. A shifted target causes checked retreat instead of silently redirecting an old route. Unknown occupancy never becomes free because of color or mushroom scores.

Every lead carries an `inspectionRequest`. The default budget permits three inspections and 12 m of reserved travel per mission. A budget limit retains deferred targets; it does not claim that every marker was visited. A blocked approach or missed photograph does not increment visits.

## Aim and capture

The planner computes yaw and pitch from the current camera viewpoint to the target. For the present fixed downward camera assumption it chooses a same-height viewpoint that can center the target. If the required pitch is unreachable, the request remains `manual_closeup_camera_pitch_unreachable`; a proposal is not a gimbal command.

At the final viewpoint it waits for low movement, low rotation, correct aim and the settle interval, then emits a unique target photo request. Intermediate route captures are explicitly `proposal_only`, with `photoCaptured:false`. They cannot mark a lead visited.

The acknowledgment must match request, lead and frame IDs, include a synchronized timestamp, native uncropped dimensions, photo reference/hash, capture pose and aim, and settled/sharpness evidence. The current simulation requires `synthetic:true`; missing or fabricated external acknowledgments cannot report a real photo. Acknowledgment metadata is not independently verified file content. The report separates simulated inspections from physical ones (zero).

The requested target width is 70% of the native sensor frame. A centered uncropped native bounding box is required to mark that framing achieved, and human identification remains pending. Digital crop and display enlargement do not add captured detail.

For the illustrative 1920 px camera, 1380 px focal length and 5 cm cap, 70% native width would need a range of approximately 0.051 m. This violates the 1 m minimum aircraft standoff. The result is `zoom_required_or_manual_closeup`, with a safe-viewpoint photo request and a pending close-up; it never drives the aircraft into the mushroom. At 1 m the cap spans approximately 69 native pixels (3.6% width), requiring about 19.5× optical magnification to reach the target width. These are geometry assumptions, not a camera recommendation or measured detector performance.

## Reproduce

Run `node --test --test-isolation=none tests/flight-range.test.mjs tests/flight-habitat.test.mjs tests/flight.test.mjs tests/flight-photo-replay.test.mjs`, then `node flight/run-range-simulation.mjs` and `node flight/run-habitat-simulation.mjs`.

`flight/reports/range-latest.json` contains concise checks, 50 m routing fixtures, framing geometry, bounded habitat log, per-target capture requests/acknowledgments and full simulation events. The 18 new tests exercise exact thresholds, uncertainty, invalid range, color-only cues, duplicate promotion, near priority, budget/geometry gates, native framing, pause/resume, no acknowledgment, fabricated acknowledgments and changed-target retreat.
