# Habitat and distant-scout simulation

**9/9 scenarios passed.** Synthetic scene evidence; no aircraft or live habitat classifier.

| Scenario | Result | Outcome |
|---|---|---|
| Low-confidence nearby visual cue receives segmented approach and acknowledged simulated photo | PASS | One synthetic photo acknowledgment; native close-up remains deferred; no real photograph |
| Nearby candidate takes priority over habitat context | PASS | Immediate nearby investigation precedes high-score habitat context |
| Unregistered photo stays a screen-only lead | PASS | No motion target from an unlocated photo |
| 6.62-second-old semantic results cannot start an inspection | PASS | Fresh evidence is requested while depth remains current |
| High habitat priority cannot clear unknown air | PASS | No route through a missing-depth wall |
| New obstacle stops detail inspection and triggers a checked retreat | PASS | Retreat uses newly validated 3D corridor; no blind return |
| Blocked retreat requires manual takeover | PASS | No automatic motion when return volume is unknown |
| Localization loss during habitat inspection suppresses position proposals | PASS | Manual takeover; no position hold with invalid localization |
| Pilot takeover preempts scouting and the inspection queue | PASS | Terminal manual control handoff |

## Separate archive inference

12 actual archive photos analyzed; 9 unlocated inspection leads, **zero metric map targets**. Mean offline inference: 0.326 seconds/image. Ecological accuracy is unmeasured. Model use: Non-commercial research/evaluation only.

## Limits

- All habitat labels and visual detections in these flight scenarios are synthetic inputs.
- Priorities are heuristic inspection preferences, not probability of mushrooms, species identity or safe ground.
- Generic scene segmentation, if provided separately, does not establish hardwood species, moisture or 3D freespace.
- No genuine distant-positive black-trumpet detector result or flight working range is established.
- The original 2-second action-evidence gate is preserved; measured desktop semantic latency currently exceeds it.
