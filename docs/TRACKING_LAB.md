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
The complete local video stays in the browser. A selected sample is resized to
at most 960 × 720, encoded as JPEG and sent to the connected owner station's
`POST /api/tracking-demo/detect`. The endpoint uses the installed GroundingDINO
worker, only the original RGB method, and saves neither the sample nor a review.
It produces no training labels. Authentication and origin checks belong to the
normal station API, as with other local research tools.

The browser permits only one active request, samples at least two seconds apart
in source time and stops at 60 samples per source session. No pending frame
queue accumulates. Seeking, changing sources or changing station invalidates
outstanding results. Camera capture stops when the page leaves or source changes.
The model worker retains its own busy check and timeout.

CPU inference may take several seconds. **Pause my video while analyzing** is
enabled by default, allowing an observed result to be viewed on its actual frame.
Camera footage cannot be held this way; delayed results become stale/lost on the
current preview. The expandable **Last analyzed frame** shows the exact sampled
image, sample timestamp and measured request latency. Smooth video display is
not the same as real-time detection.

In the local browser verification, the real replay was also opened as a local
video and its 19.50-second frame was sent for fresh inference. The request took
18.80 seconds including worker startup and returned a roughly 281 by 232 source-
pixel mushroom candidate. The displayed box was aligned to the paused sampled
frame. This checks the new-inference path; it is not a sustained throughput
benchmark or a camera, phone, headset or aircraft hardware test.

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

The browser fetches the fixed video through the station's authenticated API and
plays a Blob URL, so tokens never appear in a media URL. Keep the detection route
inside the normal image body limits and request-rate protections. Unit tests
cover association, distinct IDs after reset, staleness, bounded work, proposal
gates, replay provenance and tamper rejection.
