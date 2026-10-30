# Mushroom Vision

**Unfinished open-source research preview.** Mushroom Vision combines human
image review, reproducible vision experiments, optional shared-device compute,
and a simulated woodland field station. The internal name `MycoScout` is retained
in APIs and stored data for compatibility. Original code is MIT licensed.

[Community fieldwork](https://thegreatlakesgleaner.press/mushroom-lab/) ·
[Project introduction](https://thegreatlakesgleaner.press/articles/mushroom-vision.html) ·
[Contributing](CONTRIBUTING.md) · [Third-party licenses](THIRD_PARTY_NOTICES.md)

## Run the source preview

Install Node.js 24 or later, then run `npm start` and open
`http://127.0.0.1:8787/`. No npm dependencies are required for the station.
Open `/devices.html` to pair an optional helper, benchmark browser image-quality
analysis, or simulate connection delays. A fresh source checkout starts with
demo data; downloaded photos, model weights, private labels and installed
Android packages are intentionally absent.

Heavy Python vision components require Python 3.11–3.12 and dependencies from
`adapters/requirements-vision.txt` (heavy detection) or
`adapters/requirements-device.txt` (light device work, Python 3.10+). Pretrained weights have
separate licenses and are installed separately; see [vision setup](docs/VISION_RUNTIME.md).
See [shared devices](docs/DEVICE_COMPUTE.md), [data sources](docs/DATA_SOURCES.md),
and the [research vocabulary](config/research-taxa-v1.json).

Open `/ai-review.html` for the separate **AI review desk** and `/tracking-lab.html`
for **video tracking**. The desk preserves original-pixel boxes, evidence,
uncertainty and AI provenance without converting them into human confirmations.
An actual 21-frame visual pass and a six-frame, one-epoch inactive experiment
are described in [AI review run](docs/AI_REVIEW_RUN_V1.md). No validation accuracy
is claimed for that small experiment.

The tracking page supports a licensed recording with precomputed detections,
local videos with new sampled inference, and an explicitly requested live camera.
Tracks show observation age, prediction and loss; closer-photo proposals have no
actuator authority or invented GPS. See [tracking setup](docs/TRACKING_LAB.md).
The [distance retest](docs/VIDEO_DISTANCE_RETEST.md) reports fixed pixel-size
tests separately from any real-world range claim.

The new research vocabulary accepts 108 taxon labels, including 78 species and
30 genera, without changing Michigan certification scope. Accepting a label is
not evidence that the model can recognize it. Species candidates keep their
confidence, evidence source and uncertainty alongside the existing broad groups.

The latest bounded acquisition added **785 images**: 672 fungi and 113 wildlife
context frames. A newly trained experimental photo classifier supports **38 of
43 research categories**; it is separate from the default model. On 855 held-out
weakly labeled archive photos its top-category accuracy was 59.9%, or 82.1%
among the 53.6% of photos retained by its validation-selected confidence rule.
These are not verified species or drone-distance results. Read the
[expanded experiment and reproduction commands](docs/EXPANDED_VISION.md).

Phone, Quest, Raspberry Pi and other-device profiles describe implemented
protocols and compatibility targets. This release does not claim those physical
devices have been tested. Only optional image analysis leaves the field station;
stabilization, collision avoidance and failsafes must remain onboard an aircraft.
No physical flight control is enabled.

The long-term goals include better fungal catalogs and, after separate expert
evaluation, support for medical professionals investigating accidental ingestion.
The current software is not a species authority, edibility test or clinical tool.

## Current and earlier experiments

The records below describe local experiments and optional artifacts produced
during development. Their original datasets and weights are not included in
this public source checkout. Historical results do not establish the performance
of a newer candidate model or detection range in flight.

### MycoScout v0.8 connected workflow

Open [Mission control](http://127.0.0.1:8787/live.html) for recorded tracking, measured 3D landmarks, the shared task queue and reviewed-data training. Read [the connected workflow guide](docs/CONNECTED_WORKFLOW.md) for what works and the remaining hardware integration. Physical flight remains disabled.

The [Great Lakes Gleaner Mushroom Field Lab](https://thegreatlakesgleaner.press/mushroom-lab/) hosts 372 reusable video frames for volunteer annotation. The [local human grader](http://127.0.0.1:8787/video-grader.html) can show separate AI proposals, assistant visual corrections, and imported community reviews. Every copied box stays an editable draft until the curator explicitly saves a review. See [community deployment and the feedback workflow](deploy/community/README.md) for private export/import, contributor consent, frozen datasets, training, and activation.

The [research fieldbook](http://127.0.0.1:8787/lab.html) shows the 432-frame / 9-video AI pass and the saved training comparisons. The detector and crop-filter pilots use assistant-reviewed examples from the four training-eligible video groups; their measured recall was worse on the 19 complete human-reviewed comparison frames, so both remain inactive. Research-only footage and frozen evaluation sources are excluded from those training runs. AI review is not human confirmation, and this comparison provides no evidence below 64 target pixels or at measured flight distances.

The [video-frame pixel test](http://127.0.0.1:8787/lab.html#video-distance-heading) separately compares original RGB, fixed color enhancement, and overlapping tiles on three human-marked morel-like targets from one video, plus one inspected negative frame. The fixed conditions are native, 64, 32, 16, 6.9, 3.45, 1.38 and 0.69 requested pixels. Actual widths, matched targets, unmatched detections, and negative controls are reported separately. Whole-frame reduction on gray padding changes scene context and can create border detections; this is synthetic sensitivity evidence, not measured 10–100 m flight accuracy or species identification. Partial native labels from another video remain separate. Run `runtime-vision\Scripts\python.exe experiments\video_distance_benchmark.py --output experiments\results\video-distance-new` with the local server running to create a new frozen run; existing experiment output is never overwritten. This runs inference only and does not activate or train a model.

# MycoScout
A working development preview for a Nothing Phone (3a), Meta Quest 3, and a future drone field station.

**This is not a completed autonomous mushroom-hunting system.** The demo, data service, conservative walking planner, Android companion and Quest WebXR tabletop viewer are implemented. Version 0.8 adds live observation adapters, shared mission tasks and simulator-only transport. Physical flight control, validated species identification, automatic cliff/water/mud detection and outdoor AR registration remain unavailable.

## Start here
**Test your own photos:** open **http://127.0.0.1:8787/tester.html**. Try the licensed sample or upload a JPEG/PNG, compare original/color/tiled detection, draw the missed mushrooms, and save your labels. See [Vision workbench](docs/TESTBENCH.md).

1. Install **dist/mycoscout-phone-debug.apk** on the Android phone for the bundled offline demonstration. This is a signed testing APK, not a Play Store release.
2. On a computer with Node.js 24 or newer, run `npm start` in this folder. Open **http://127.0.0.1:8787**. No npm install is required for the service.
3. Select a candidate and choose **Plan walking route**. The synthetic world illustrates trees, water, cliffs, muck and unknown cells; it is not your location.
4. Quest 3: serve **dist/mycoscout-quest-web.zip** over trusted HTTPS, open the URL in Quest Browser, then use **Enter AR** or **Quest VR**. The map is a tabletop model, not a world-aligned navigation overlay. There is no native Quest APK.
5. For shared real survey data, deploy the station over HTTPS and enter the same station URL and access token on both devices. See [deployment](docs/DEPLOYMENT.md).
6. Real sensor inputs: see [integration setup](docs/INTEGRATIONS.md). The RTAB-Map adapter imports an occupancy snapshot. The GroundingDINO adapter generates image boxes for human review.
7. Open **X500 research lab** for measured photo experiments and simulated flight behavior. Run `npm run simulate:x500` to reproduce the flight scenarios. See [X500 development](docs/X500-DEVELOPMENT.md) and [vision pipeline](docs/VISION_PIPELINE.md).
8. **Version 0.3 adds black-trumpet research:** 1,174 licensed training/evaluation photos, a trained photo ranker, wide-view detection comparisons and habitat-assisted inspection simulation. Open the research lab for results, or see [vision V2](docs/VISION_V2.md), [dataset provenance](docs/DATASET_V2.md) and [habitat search](docs/HABITAT_SEARCH.md). Run `npm run simulate:habitat` to reproduce the search decisions.

Use **Preview survey JSON** for a local, validated preview of an exported real survey. It does not replace the shared station. Import detector JSON with **Import detection JSON**; these candidates remain unlocated until depth, camera pose and frame calibration are supplied.

## Version 0.7: video frames and human grading
Open **http://127.0.0.1:8787/video-grader.html** to grade a mixed feed of real video frames, mark visibility, and manually track one mushroom through earlier and later moments. Source videos, timestamps, licenses and recording-session groups stay attached. Complete human boxes can be compared with actual detector tests by pixel width, method and settings. No labels propagate automatically and no model is silently retrained. See [human grading](docs/VIDEO_GRADING.md) and [video extraction](docs/VIDEO_FRAMES.md).

## Version 0.6: interactive vision workbench
The **Test vision** app accepts original photos and compares whole-frame detection, local color enhancement, and four or nine overlapping tiles. Choose a threshold or reduce image scale to stress-test smaller targets. Every run retains its original image, settings, model version, progress and results. Color clues remain separate from mushroom detections.

Draw boxes, mark real candidates, request tentative group guesses, and save your own labels. Complete labels produce one-to-one detection scores; partial labels do not treat unlabeled regions as negatives. Reruns inherit labels from the exact same original, and each correction is versioned. Permission-controlled exports can be converted to COCO/YOLO data for a future detector. Nothing silently trains or deploys a new model. The installed station provides inference; the phone package shares this interface.

Actual smoke comparisons on one black-trumpet photo and its synthetic 6.9-pixel derivative did not demonstrate improved detection from either color or tiling. More proposals alone are not an improvement. The workbench makes that tradeoff measurable on new photos; it does not promise a real-distance success rate.

## Version 0.5: tiny clues and close investigations
The research fieldbook now compares original RGB with a fixed color-enhanced detector view at 6.9, 3.45, 1.38 and 0.69 pixels, with a 32-pixel control and background-only controls. A separate CIE Lab contrast proposer records unconfirmed color cues. All results remain synthetic pixel-scale evidence, never a measured 10/20/50/100 m detection rate. See [far-pixel test](docs/FAR_PIXEL_VISION.md).

The simulated inspection queue routes visual cues at or possibly beyond 50 m into a habitat follow-up log. Valid nearer candidates receive investigation priority; unknown range stays unresolved. Target-facing, settled native-photo requests track each lead. A visit or resized crop does not prove a close photo: the 70% frame-width goal remains pending when clearance, fixed camera pitch or optics prevent it. See [range and close-photo behavior](docs/RANGE_INSPECTION.md). The phone/Quest review inbox exposes the investigation categories and preserves human confirmation.

## Version 0.4: review and improve
Open **Review captures** for a shared phone/Quest inbox. Each frame can receive three ranked research-category guesses, a broader foraging group and an uncertain result. You confirm a type, correct an identification, record field evidence, or request another view. The Quest browser also includes a controller-operated VR review board. Actual hardware remains untested.

The collection now contains **3,250 licensed photos** (2,076 added). A newly trained 22-category model supports 21 categories; Michigan truffles have insufficient data. On 722 held-out images, first-category accuracy is **68.7%**, top-three accuracy **87.8%**. The validation-chosen confidence rule achieves **75.8%** accepted accuracy on the full test set, below its 80% validation target. These measure observation-labeled archive photos, not species certainty or drone performance.

Human reviews live in a SQLite database with photos, immutable history, model versions and explicit training rights. Portable exports can be merged into a new frozen dataset and used for an explicit retraining/evaluation run. Known evaluation photos remain held out; conflicting groups and labels are excluded. The Michigan catalog covers all 20 current MAMI study-guide entries and separately marks bolete research; it is not an exhaustive statutory whitelist.

See [human review and camera bridge](docs/HUMAN_REVIEW.md), [group vision and retraining](docs/GROUP_VISION.md), [database](docs/REVIEW_DATABASE.md), and [Michigan scope](docs/MICHIGAN_SCOPE.md). The capture bridge accepts sampled JPEG/PNG frames from a camera/detector folder; a physical OAK-D driver and continuous drone video are still integration work.

## What is implemented
| Function | Current behavior | Verification |
|---|---|---|
| 3D survey interface | Render local occupancy cells, illustrative obstacles, candidate markers and routes | Desktop and phone-size browser checks |
| Shared state | Authenticated API, local JSON persistence, cross-device polling | Backend tests |
| Ground routing | A* with clearance; unknown/hazard cells blocked; no diagonal corner-cutting | Planner tests |
| Android companion | Bundled offline assets, HTTPS station access, runtime camera and JSON file picker | Build, lint and APK signature checks; no device test |
| Quest AR/VR | Feature-detected WebXR tabletop view using Three.js | Code/browser checks; physical Quest untested |
| RTAB-Map bridge | Conservative single-snapshot ROS2 OccupancyGrid/JSON import | Conversion tests; live ROS untested |
| Fungi detector | Local GroundingDINO inference, optional overlapping tiles and duplicate suppression | Actual local Facebook-photo pilot; see research lab for measured limits |
| Candidate filter | Trainable mushroom/background crop verifier; not a species classifier | Group-separated training/validation/test with explicit image labels |
| Black-trumpet photo ranker | Frozen visual features plus a newly trained linear head; preserves detector candidates | 274 held-out observation-labeled photos: 77.7% precision, 94.9% recall; not localization or verified species ID |
| Habitat inspection | Generic scene cues, typed unlocated/metric leads and bounded inspection queue | Actual inference on 12 archive photos; 9 unlocated leads. Ecological accuracy unmeasured; flight behavior simulated |
| X500 behavior | Stop, settle, capture, verify and conservative 3D corridor proposals | Deterministic simulation only; no PX4 or physical flight validation |
| Real mushroom locations | Validated import of already-calibrated coordinates | Input validation; no automatic depth projection |
| Autonomous flight | No arming, takeoff, velocity or waypoint API | Explicitly rejected by server |
| Group guesses and review | Local 22-category model, confidence abstention, manual classifications, versioned review database and training export/import | 722-photo evaluation and API/browser tests; no automated species or edibility confirmation |
| Automatic hazards | No trained water, mud, cliff classifier | Manual/reviewed map semantics only |
| Outdoor AR alignment | No shared calibrated anchors or georeference | Not implemented |

An occupancy-free cell can be empty air above a drop-off. Survey walking routes require `groundTraversabilityVerified:true` from a human-reviewed terrain map and an explicit starting position in the map frame. This flag records an assertion, not an automatic safety guarantee.

## Hardware
No cheap used consumer drone was found that meets the whole request. For supervised imaging only, a used DJI Mini 4 Pro with RC-N2 is the most practical consumer fallback found. For autonomy development, start with a programmable PX4 platform with onboard depth sensing and compute. Estimated complete research hardware budget is **$1,800–2,500+**, with additional sensing potentially increasing it substantially. See [hardware research and current price sources](docs/HARDWARE.md) before buying.

The phone and Quest are clients; autonomous perception and flight failsafes must run onboard the drone. Wi-Fi loss must never remove collision avoidance or RC takeover. A one-camera system does not see behind itself or reliably detect fine branches.

## Verification
```powershell
npm test
python -m unittest discover -s adapters/tests -p "test_*.py"
```
Browser smoke tests require Playwright and Edge/Chromium; see [test report](docs/VALIDATION.md). No physical phone, headset or aircraft was connected. Optional model weights and the ROS2 flight stack are not bundled.

## Project layout
- `public/`: offline web/Android assets and shared browser planner.
- `core/`, `server.mjs`: survey storage, validation and routing API.
- `adapters/`: actual optional RTAB-Map and GroundingDINO bridges.
- `android/`: Java companion shell and Gradle project.
- `deploy/`: container and HTTPS reverse-proxy templates.
- `docs/`: hardware, integration, deployment and remaining engineering.
- `dist/`: deliverables and test evidence.

API coordinates are local metres, not latitude/longitude. JSON export is supported; GeoJSON export is refused until a valid georeferencing subsystem exists.

Open-source components retain their licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

