# Black-trumpet scouting and habitat inspections

The new scheduler performs **distant scouting, then a closer inspection of a known safe viewpoint**. It can prioritize both weak mushroom-like visual cues and habitat areas where no mushroom has been detected. It remains a software simulation: the flight scenarios use supplied synthetic habitat labels and detections, not an onboard habitat classifier or a demonstrated far-distance black-trumpet detector.

## What the ecology supports

Michigan DNR's mushroom guide describes black trumpets in hardwood forests and gives a July–October season. That is a useful weak seasonal prior, not a claim that every Michigan site fruits throughout those months. Weather, region and local conditions vary. [Michigan DNR guide](https://www.michigan.gov/dnr/-/media/Project/Websites/dnr/Documents/Education/NatureHome/Guide_to_Mushrooms.pdf?hash=B088EA7276A5093ED0BDB992018CBA97&rev=5637039832ab41ddaae9aff4d22bf6ec)

Eastern North American black trumpets are commonly treated as **Craterellus fallax**, distinct from European **C. cornucopioides**. A molecular study supported that separation and found a broader host range than an oak-only rule would suggest, including associations with oak, hemlock and Virginia pine. The software therefore uses a black-trumpet-like **Craterellus complex** target and never excludes a site simply because conifers are present. An aerial silhouette cannot resolve those taxonomic distinctions. [Matheny et al., 2010, primary research](https://mathenylab.utk.edu/Site/Publications_files/Matheny_etal_Craterellus_fallax_M.2010.pdf)

A University of Michigan herbarium collection records C. fallax in Cheboygan County, Michigan, in mixed hardwoods on moss, with an October collection date. This is direct local occurrence evidence, not a habitat model trained across Michigan. [University of Michigan specimen record](https://quod.lib.umich.edu/h/herb00ic/x-339077/1)

Moss is helpful as a search cue but not a required association. Midwest American Mycological Information notes that dark trumpets can be easier to see on moss than on forest duff. This matters for dataset bias: photographs disproportionately taken on visible moss may teach a model the background rather than the fungus. [Midwest field guide](https://midwestmycology.org/craterellus-cornucopiodes/)

The scheduler also permits separately observed damp substrate, shade and leaf litter as **weak inspection heuristics**. Their weights are engineering choices, not validated black-trumpet occurrence probabilities. RGB darkness cannot prove moisture. Moss, wet soil, leaf litter and a favorable canopy cannot prove mushrooms are present, identify a species, or establish safe airspace or walkable ground.

## Implemented behavior

`flight/habitat-search.mjs` implements two components:

- `InspectionQueue` ranks, deduplicates, expires, reserves and revisits leads. Habitat and fungus leads keep separate types. A fresh nearby habitat lead can raise a fungus candidate's priority without changing its identification status. Duplicate frames cannot refresh age or inflate priority. Season alone is insufficient to create an actionable inspection.
- `HabitatScoutMission` first runs the existing survey's stationary far-view captures. It then chooses an eligible inspection lead, computes a closer same-altitude viewpoint, verifies every route segment in 3D and starts a second settled capture sequence. The destination is the camera viewpoint, not the mushroom itself.

The scheduler caps each viewpoint's straight-line advance at 2 m and reserves the planned route distance for both outward travel and a possible return; any detour increases that budget cost. This is a planning budget, not measured battery endurance. It never descends toward a dark patch on the ground. A desired 2 m viewing range is only a planning preference: aircraft altitude, clearance, foliage and available corridors may prevent reaching it. The route uses the actual aircraft envelope, uncertainty, stopping margin and arrival tolerance, through fresh explicitly observed free voxels. Missing depth remains unknown regardless of habitat priority.

If new obstacles or stale semantic evidence interrupt detail inspection, it can retreat only through a **newly checked** corridor to its previous viewpoint. If that corridor or localization is unavailable, it requests manual takeover and emits no movement proposal. Manual takeover and abort preempt the queue. A successful inspection records closer images and leaves species unverified; it does not turn an unsuccessful view into evidence of mushroom absence.

Queue configuration is in `config/x500-habitat-search.json`: 100 active lead slots, a 0.8 m same-kind deduplication radius, 120-second lead lifetime, 2-second maximum actionable observation age, two visits per lead, 30-second normal revisit cooldown, 60 seconds after an unsafe attempt, and per-session inspection/time/travel budgets. These are development values, not measured operational limits. Expired slots can be replaced; an already reserved lead cannot be dispatched twice. Evidence that is old but still within the queue lifetime may be shown as historical, requiring fresh acquisition before action.

Failed route checks defer a lead without counting a visit or spending an inspection reservation. Bounded replay caches, merged observation IDs, input batches and event histories prevent an unending camera stream from growing memory without limit. A full live replay cache rejects new evidence until entries expire rather than dropping still-valid duplicate protection. Reports expose any discarded history/input counts. Planning checks timestamp/provenance again even if its caller supplies an old cached eligibility flag; while waiting for leads, the scheduler still checks the aircraft's present 3D volume.

## Input contract and current segmentation limits

The machine-readable contract is `flight/habitat-observation.schema.json`. Submit observations to `InspectionQueue.ingest(observation, nowMs)` or include them as `observations[]` on a `HabitatScoutMission.step` input.

```json
{
  "id": "patch-17",
  "kind": "habitat",
  "frameId": "camera-frame-209",
  "capturedMs": 42000,
  "confidence": 0.7,
  "source": "reviewed_camera_observation",
  "month": 8,
  "signals": { "forest_context": 0.8, "vegetation_ground": 0.3 },
  "semanticFractions": { "tree": 0.4, "plant": 0.2, "earth": 0.2 },
  "capturePose": {
    "frame": "map_enu", "position": { "x": 0, "y": 0, "z": 2 },
    "yawRad": 0, "capturedMs": 42000
  },
  "metric": {
    "frame": "map_enu", "position": { "x": 5, "y": 0, "z": 0.3 },
    "depthKnown": true, "depthCapturedMs": 42000,
    "positionVarianceM2": 0.0025
  }
}
```

Confidence and signal strengths are not necessarily calibrated probabilities. Original capture time, captured pose and aligned depth time must agree for a metric map lead. A real projection adapter needs full 6-DoF transforms, camera calibration and covariance propagation; the simulation uses position plus yaw. Photos without depth or capture pose remain **screen-only leads**. If their original capture time is unknown, use `capturedMs: null`: receipt time manages the screen queue but is never substituted for the camera timestamp or made actionable.

Generic scene segmentation may supply image fractions for tree, plant, grass, earth, water, rock and path. Only weak generic `forest_context`, `vegetation_ground` and `exposed_ground` cues may be derived from those categories. They cannot populate `hardwood`, `moss`, `damp_ground` or `known_patch` without separate evidence. A predicted path is not a safe path; a predicted water region is not a metric obstacle map; neither changes voxels. Static photo segmentation experiments remain separate from a live onboard habitat classifier.

## Far-to-near limits for black trumpets

Small black trumpets blend into litter. With the existing **illustrative, unmeasured** focal length of 1,380 pixels, a front-facing 1 cm target is only 6.9 pixels wide at 2 m, 3.45 pixels at 4 m and 1.73 pixels at 8 m. A 3 cm target is about 20.7, 10.35 and 5.18 pixels at those ranges. These are geometry estimates, not detector recall; tilt, occlusion and blur make the usable evidence worse. `targetPixelEstimates` in the report exposes both sizes, and `validatedDetectorMinPixels` remains null.

The useful sequence is: record a distant field of view; nominate a weak visual cue or habitat area; obtain a metric registered observation; inspect from a closer already-clear viewpoint; then retain the close and far frames for evaluation. The far proposal need not be a confident species detection, but neither habitat nor a weak score can bypass a freshness, localization, depth or corridor gate. The original desktop model's roughly 6.62-second full-frame latency exceeds the current 2-second actionable evidence limit. The scheduler asks for reacquisition rather than pretending the old result describes the aircraft's present view.

Validate the approach with paired real near/far black-trumpet observations from the same scene, with measured camera range and object size; keep each source scene wholly within train, validation or test. Hold out dark leaf-litter negatives and other dark funnel-shaped objects. A smaller synthetic photo is a useful stress test, but not a genuine distant positive. Until those tests succeed, habitat-assisted inspection is a search strategy, not demonstrated autonomous mushroom finding.

## Reports and verification

```powershell
node flight/run-habitat-simulation.mjs
node --test --test-isolation=none tests/flight-habitat.test.mjs
```

`flight/reports/habitat-latest.json` contains `summary`, `mapTargets`, `screenLeads`, `schedule`, `targetPixelEstimates` and `scenarios`. Map targets include kind, local metric position, status, priority, reasons, capture provenance, age/expiry and eligibility. Screen leads have no metric position. Schedule entries separate scheduled, blocked, deferred and inspected outcomes with routes/viewpoints where applicable. All targets retain `identifiedSpecies: null`, `edible: null` and `groundSafety: "unknown"`.

`flight/reports/HABITAT-LATEST.md` records simulation outcomes. The scenarios cover weak distant detection followed by detail captures, habitat-first inspections, screen-only and stale-evidence refusal, unknown-volume blocking, revalidated retreat, blocked retreat, localization loss and pilot takeover. Passing these tests establishes those software decisions only. No aircraft was connected and no flight command transport exists.

When `experiments/results/habitat-v2/report.json` exists, the report generator also includes a separate `archiveInference` section. It reads actual offline segmentation measurements, strips original photo identifiers, ignores any claimed metric pose and replays generic contexts as screen-only leads with unknown capture times. That section reports actual image count, inference latency and proposed screen leads while keeping metric map target count at zero. Model identification and the recorded research-use restriction remain visible; model weights and photos are not included in the flight report. Offline inference success is not ecological accuracy or live sensor deployment.
