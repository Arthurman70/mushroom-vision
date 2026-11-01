# Tracking lab: a visible bridge from video to robotics

Open `/tracking-lab.html` on the field station. The default example is a real,
licensed 19.6-second video of approaching a mushroom, with 12 existing detector
samples and 19 unverified candidate boxes. Playback is continuous, but inference
on this example is **precomputed**, not live. Track identities are heuristic
image-plane associations; they have not been verified as specimen identities.

The recording is **“the huge mushroom,” freestonew**, under **CC BY-SA 2.0**:
<https://archive.org/details/Freestonew-theHugeMushroom948>.
The browser derivative removes audio and encodes H.264 while preserving video
presentation timestamps. Its license remains CC BY-SA 2.0. The original source
URL, attribution, license URL, original video hash, encoded video hash, original
frame hashes and detector versions are recorded in the replay manifest. These
media rights are separate from the software license.

`python scripts/build-tracking-demo.py` builds the replay from the already
acquired recording and the completed AI pass in this workspace. It refuses a
changed original recording or unmatched frame output. It does not download new
material or invent labels. The generated files live in `data/tracking-demo-v1`.
They are optional dataset assets, not required source-code files. If absent,
the browser presents a clearly labeled **synthetic illustration**, which is not
detector performance evidence. It demonstrates occlusion and lost tracks.

## New inference on a video or camera

Choose a local video, or explicitly select **Use camera** to request permission.
The complete local video stays in the browser. A selected sample is resized,
encoded as JPEG and sent to the connected owner station's
`POST /api/tracking-demo/detect`. The endpoint uses the installed GroundingDINO
worker and saves neither the sample nor a review.
It produces no training labels. Authentication and origin checks belong to the
normal station API, as with other local research tools.

The **Lower ground area** (`ground-tiles`) keeps up to 1920 × 1080
analysis pixels, without upscaling, and runs one full-frame context pass plus
four overlapping 2 × 2 detail tiles within normalized `[0, 0.35, 1, 1]`.
This lower 65% is a manually selected image rectangle, **not automatic ground
segmentation**. Context detections outside the tile rectangle remain available.
This selection never limits navigation or obstacle sensing.
The **Top of ground area** slider moves the rectangle's upper edge from 0% to
80% of image height; its default is 35%. Its live percentage and tile preview
show the selected area. It is enabled for hybrid and `ground-tiles`. Changing it
invalidates existing tracks and pending analysis just like changing the profile.

Choose **Whole frame** (`full-tiles`) for downward views or useful terrain higher
in the image. It uses `[0, 0, 1, 1]` with the same five-pass plan. **Fast full
frame** (`fast`) uses one original-RGB pass at up to 960 × 720. All three retain
the existing 0.3 detection threshold. Extra passes may improve recall or increase
false positives; tiling adds no optical detail and establishes no distance rate.
The preview outlines the full context and four detail tiles; the plan shows
capture dimensions and expected passes. Completed samples report actual model
passes, each pass's duration, total processing time and request latency.

**Analyze recording afresh** explicitly switches the licensed example into new
sampled inference. Previously saved detections are excluded from those tracks.
**Return to saved detections** restores the original replay. Merely selecting a
scan profile never changes old boxes or claims they were generated with new
settings. The source labels distinguish these modes.

The browser permits only one active request, samples at least two seconds apart
in source time and stops at 60 samples per source session. No pending frame
queue accumulates. Seeking, changing sources, changing the scan or changing station invalidates
outstanding results. Camera capture stops when the page leaves or source changes.
The model worker retains its own busy check and timeout.
Changing the scan also stops automatic sampling, clears tracks and closer-look
proposals, and preserves the 60-request session budget. An outstanding request
must finish before another can start, even though its result is discarded.
Canceled, failed or discarded analysis cannot resume paused playback. Automatic
resumption is limited to an accepted result from the unchanged tracking session.
Video resolution changes also clear earlier tracks and pending results.

CPU inference may take several seconds. **Pause the video while analyzing** is
enabled by default, allowing an observed result to be viewed on its actual frame.
Camera footage cannot be held this way; delayed results become stale/lost on the
current preview. The expandable **Last analyzed frame** shows the exact sampled
image, sample timestamp and measured request latency. Smooth video display is
not the same as real-time detection.

In the earlier one-pass browser verification, the real replay was also opened as a local
video and its 19.50-second frame was sent for fresh inference. The request took
18.80 seconds including worker startup and returned a roughly 281 by 232 source-
pixel mushroom candidate. The displayed box was aligned to the paused sampled
frame. This checks the new-inference path; it is not a sustained throughput
benchmark or a camera, phone, headset or aircraft hardware test.
That timing does not measure the new five-pass profiles.

## Track meaning

The pure browser tracker filters highly overlapping or closely nested duplicate
boxes, retaining the higher detector score. It then associates boxes by overlap, normalized center distance
and a bounded size-change gate, with a one-to-one greedy match. This can confuse
nearby similar mushrooms, camera cuts, occlusion or large camera movement. A new
track is a new candidate identity, not a discovery count.

- **Observed:** within 0.15 seconds of the latest sample in source time.
- **Predicted:** a short constant-velocity image-plane estimate after that sample.
- **Stale:** over 1.2 seconds since the last observation; motion extrapolation
  stops growing at this limit.
- **Lost:** over 4 seconds since the last observation; no box is drawn on the
  current video and no closer-look proposal is allowed.

The UI's age is **source/video time**, so it stops while a recording is paused.
It separately displays processing latency. Future physical use must enforce
wall-clock freshness against sensor timestamps, not treat a paused recording as
fresh physical evidence. Native pixel size is reconstructed from the analysis
resize and always refers to the last observed box. Detection scores are
uncalibrated, not species confidence.

## Closer-look proposals and robotics preparation

After two recent observations, a candidate can produce a downloadable proposal.
The proposal includes source ID, unique tracking-session ID, timestamp, observed
pixel box, pixel dimensions, desired frame fill and explicit prerequisites. No
proposal is posted to mission control or transmitted to an aircraft or robot.
RGB alone leaves `position`, `gps` and measured range null. If a downstream
caller supplies a measured range of 50 m or more, the proposal becomes a habitat
follow-up; this lab never infers that distance from pixels.

To connect a future robot, use the existing `adapters/live_capture.py` and
`vision/live_mapping.py` path for time-aligned calibrated RGB/depth/pose packets,
then the existing live-mission simulator. The next hardware integration needs:

1. Camera intrinsics and camera-to-body transforms, with recording timestamps.
2. Aligned depth, a metric pose with uncertainty and a local map that distinguishes
   observed free space from unknown space.
3. A trusted local-to-geographic transform before producing geographic landmarks.
4. Freshness, collision coverage and standoff checks on the onboard system.
5. Human confirmation and a closer-view capture acknowledgement before improving
   a training record or claiming a successful investigation.

Offboard vision may return delayed candidate evidence. It must not replace the
onboard flight controller, collision decisions or loss-of-link behavior. This
tracking lab sends **no actuator commands** and has no physical flight release.

## Server integration

`core/tracking-demo.mjs` exports:

- `readTrackingDemo(root)` for `GET /api/tracking-demo`.
- `readTrackingDemoVideo(root)` for the authenticated fixed replay video endpoint.
- `detectTrackingFrame(input, testbenchWorker)` for bounded, ephemeral inference.

Detection requests accept `profileId: 'ground-tiles' | 'full-tiles' | 'fast'`
(default `ground-tiles`) and optional `tileRegion: [left, top, right, bottom]`
in normalized 0–1 coordinates for tiled scanning. A custom region is accepted
under `ground-tiles`; `full-tiles` requires the whole rectangle and `fast`
rejects a tile region. Unknown profiles, empty/out-of-bounds regions and
nonfinite coordinates fail before inference. General testbench settings retain
legacy full-image behavior when `tileRegion` is absent.

The result includes the selected profile, exact settings, actual `scan.modelPasses`,
`scan.fullFrameContext`, `scan.tileRegion`, original-pixel
`scan.tileBoundsPixelsXYXY` and `scan.passTimings`. Worker metadata must agree
with the requested region, original-frame bounds, pass count and measured times.
Thresholds and scores remain uncalibrated; no model is trained or activated.

## Hybrid search, 30 October 2026

The tracking page now defaults to **Hybrid: fast + occasional detail**. It sends
the existing single-frame request types in a bounded sequence:

1. Start with fast, whole-frame inference.
2. Allow a detail sweep only after at least two accepted fast samples at distinct
   source times, and at least five source seconds since the first sample or the
   previous detail sweep. The browser interval is adjustable from 2 to 60 seconds.
3. Use lower-image tiles for two detail sweeps, then whole-image tiles for the
   third. Repeat, keeping full-frame context in every sweep.
4. Record actual capture time, effective scan type, passes and latency. A late
   scan never creates a queue of missed sweeps to catch up on.

This is serial inference with fewer expensive scans, not an asynchronous second
detector. The same approved model handles both kinds of sample; the experimental
weight candidates remain inactive. The browser threshold stays 0.3 and native
baseline capture stays 0.25. Neither threshold is a species probability. Tracking
between detector observations remains the existing short image-plane prediction,
with its stale/lost limits; a smooth preview does not make old evidence fresh.

Scanning the whole image periodically matters for fungi on trunks, stumps and
logs. Chicken of the woods can form conspicuous orange/yellow shelves on wood;
oysters form overlapping clusters and may be white, gray or tan. Large clusters
can occupy more pixels than an individual small mushroom, but their detection
range has not been measured here. No color rule assigns a species or suppresses
dark mushrooms. See the [US Forest Service field guide](https://www.fs.usda.gov/nrs/pubs/gtr/gtr_nrs79.pdf)
and [Missouri Department of Conservation oyster description](https://mdc.mo.gov/discover-nature/field-guide/oyster-mushroom).

The native `adapters.live_capture` CLI and fresh-inference mission replay also
default to `--vision-profile hybrid`. CLI controls are `--detail-interval 5`,
`--full-detail-every 3` and `--min-fast-frames 2`. Recorded videos use source
timestamps. USB capture uses a one-slot reader while inference is busy, retaining
the newest host-received frame instead of accumulating a camera-frame queue.
Host receipt is not a synchronized exposure timestamp; driver delay is unknown.
The packet reports skipped frames and receipt age rather than inventing sensor
freshness. Camera hardware and end-to-end flight have not been tested.

Slow motion alone cannot compensate for old inference. At an illustrative
0.2 m/s, a 30-second result is already six metres behind the camera. Delayed
tiles provide investigation evidence; a fresh observation, measured location
and current navigation evidence are needed before an approach. This change
sets no flight speed or altitude. The current system lacks measured ground
clearance for automatic low-altitude selection and still has no aircraft transport.

`public/hybrid-scan.mjs` and `vision/hybrid_scan.py` implement the cadence;
the per-frame HTTP endpoint continues to accept only `fast`, `ground-tiles`
and `full-tiles`. Thus saved replay data retains its original scan provenance.

The investigation log retains up to 20 captured frames and 100 unconfirmed
regions within the current source/scan session. Later empty scans do not erase
earlier finds. **Review captured frame** opens the saved annotated thumbnail
with its original timestamp; it cannot update a current track or authorize a
flight task. Export includes the retained image-coordinate cues and closer-look
proposals. Seeking or changing source/scan clears this local log explicitly.

### Measured hybrid smoke test

`experiments/results/hybrid-scan-v1/protocol.json` was frozen before running ten
native 640 × 480 frames at 0, 2, ..., 18 seconds of the licensed recording.
The actual capture adapter produced the expected sequence:
`fast, fast, fast, ground, fast, fast, ground, fast, fast, whole-frame`.
Every saved JPEG and source hash was checked, and RGB candidates remained
unlocated. Model weights and the native baseline threshold (0.25) were unchanged.

| Actual scan stage | Frames | Model passes | Mean scan time on this CPU |
| --- | ---: | ---: | ---: |
| Fast whole frame | 7 | 7 | 7.27 s |
| Ground detail | 2 | 10 | 36.11 s |
| Whole-frame detail | 1 | 5 | 35.72 s |

The run used 22 model forwards rather than the 50 an all-tiled scan of those ten
frames would require: **56% fewer passes**. This is an exact work-count
comparison, not a measured 56% wall-time speedup; the all-tiled sequence was not
rerun. Total scanning took 158.81 seconds, or 15.88 seconds per sampled frame.
The capture loop took 159.38 seconds after model initialization. Inference is
still too slow for continuous flight-speed detection on this CPU. A five-second
source-time interval does not promise a result every five wall-clock seconds.

No accuracy score was computed: the ten frames are correlated views from one
video without a new reference-label evaluation. This run tests cadence,
real inference, coordinate preservation, provenance and processing cost. It
does not establish long-range mushroom or wood-cluster recall.

Validation passed 407 Node checks and 72 focused Python checks. Six Node
integration checks initially hit the sandbox's child-process restriction; their
12-test training-pipeline suite then passed with local helper execution allowed.
Logs and per-frame results are saved beside the protocol in
`experiments/results/hybrid-scan-v1/`.

A browser check on 31 October used fresh inference on the licensed recording
at the unchanged browser threshold of 0.3. Three fast samples at approximately
0.32, 2.33 and 4.34 seconds were followed by a five-pass ground sweep at 6.35
seconds. The sweep took 36.98 seconds processing / 36.99 seconds request time.
All four sampled frames remained in the log, containing eight unconfirmed
regions. The historical frame dialog reopened the exact annotated evidence;
the downloaded JSON contained those eight timestamped cues with no species,
GPS, measured range or flight authority. Extra grass regions were visibly boxed.
Sampling was stopped, and the next pending fast result was discarded. Screenshots,
the verified export and the check record are saved as `hybrid-controls.jpg`,
`review-frame.jpg`, `ui-clue-export.json` and `ui-verification.json` in that folder.

The browser fetches the fixed video through the station's authenticated API and
plays a Blob URL, so tokens never appear in a media URL. Keep the detection route
inside the normal image body limits and request-rate protections. Unit tests
cover association, distinct IDs after reset, staleness, bounded work, proposal
gates, replay provenance and tamper rejection.

## Ground-area comparison, 30 October 2026

`experiments/results/ground-tiles-v1/report.json` records a fresh, frozen comparison
of eight existing native-resolution frames, with 88 actual model forwards. The
protocol, source hashes and per-method predictions are stored alongside it.
GroundingDINO weights were unchanged; all methods used box/text thresholds 0.25
and NMS IoU 0.5. The browser's existing 0.3 threshold remains unchanged and is not
the threshold of this comparison. No model training or activation occurred.

| Measurement | Single frame | Whole-view tiles | Lower-65% tiles |
| --- | ---: | ---: | ---: |
| Matched human mushroom regions, out of 3 | 2 | 2 | 3 |
| Unmatched alerts on the three positive frames | 5 | 18 | 23 |
| False alerts on one human-confirmed empty frame | 1 | 6 | 8 |
| Alerts on four separate AI-negative reference frames | 8 | 22 | 27 |
| Mean detector time per frame, this CPU | 8.30 s | 41.85 s | 28.59 s |

The ground-area scan recovered one additional labeled mushroom, cost 3.44 times
the single-pass processing time, and was 31.7% faster than whole-view tiles. It
also increased alert burden. Unmatched boxes on positive images are not all
established false mushrooms. AI absence references are not human ground truth
and are kept separate; none of the four explicitly marked hat/wood/bark/trunk
regions had a detection overlap of IoU 0.5 or higher, so the extra alerts cannot
all be attributed to those objects.

The human references come from one video. Two targets cross the ROI boundary;
none is wholly above it. Full-frame context retained a boundary target, but this
does not establish above-ROI recall. These previously inspected frames are a
small regression check, not an independent accuracy estimate or a measured
10/20/50/100-metre test. Ground-area tiling is a candidate-search default, not
evidence of confirmed species or safe flight.

Validation: 378 Node tests and 59 focused Python geometry, adapter and mapping
checks passed. The fixed evaluation harness also passed its four tests.

A fresh browser check used the licensed recording at 19.45 seconds, default
lower-65% scan and the unchanged 0.3 browser threshold. It completed five actual
passes in 33.09 seconds processing / 40.13 seconds request time including worker
startup. Six detector candidates became five displayed tracks after duplicate
filtering. The mushroom was boxed, but glove and grass regions also received
boxes. This verifies the UI/API path, not identification accuracy. The exact
display is saved in `experiments/results/ground-tiles-v1/tracking-result.png`;
`tracking-ground-controls.png` shows the adjustable default scan area.
