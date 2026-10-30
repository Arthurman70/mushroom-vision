# Optional device compute

Mushroom Vision is unfinished research software. The shared-device utility is implemented for optional image analysis. It is not a distributed flight controller and does not make hardware autonomy claims.

Open `/devices.html` on the local station. It can normalize an uploaded image, benchmark image-quality analysis in the current browser, pair/revoke helpers, queue a photo, inspect results and simulate a connection. A file stays in the browser until **Queue this photo** is clicked. The existing mission dashboard remains available for human review.

## What runs where

| Location | Implemented work | Validation state |
| --- | --- | --- |
| This computer, Python | Quality/color metrics and existing local GroundingDINO candidate detector | Measured on three real saved video frames; report below |
| Modern browser | Canvas image decoding, downsampling and actual luminance, sharpness and color calculations | Portable implementation; use the page benchmark to measure the current browser |
| Nothing Phone (3a), Quest 3 | Same foreground browser worker and review pages | Hardware untested; no native mobile detector, neural accelerator or GPU delegate bundled |
| Raspberry Pi / other Python devices | Outbound worker with Pillow/NumPy quality analysis | Hardware untested; no ARM performance claim |
| Compatible Python machine with local detector weights | Optional candidate detection after the model loads | Announced only when explicitly enabled and successfully loaded |
| Drone | Stabilization, immediate depth/obstacle avoidance, failsafes | Always onboard; these are never queue jobs |

The quality worker computes real measurements; it does **not** identify a mushroom. `laplacianVariance` measures detail in a sample at up to 640 pixels per side. `colorOutlierFraction` measures pixels whose RGB distance from the sample mean exceeds 40. Lighting, foliage and shadows also change these scores. Browser/Pillow resizing can differ, so compare scores within a pipeline rather than assuming identical calibration.

Whole-image downsampling can destroy a distant mushroom's few pixels. The detector worker receives the original submitted raster. For future camera selection, prioritize useful full-resolution regions and only use a lower-resolution preview for scene overview. The current producer supports prepared source-pixel crops, but does not automatically choose a crop or restore a cropped box to a world location.

## Pair a browser

1. On the station page, enter the station owner token if one is configured. Choose a profile and create a named pairing.
2. For this browser, click **Use in this browser**. For another browser, open the station's page over intentionally configured trusted HTTPS and enter the displayed device token under **Connect an already paired browser**.
3. Keep the worker tab visible, then queue a photo from the station. The worker announces `image-quality-v1`, claims one matching frame, checks its SHA-256, analyzes pixels and returns a result.
4. Inspect the output or remove the pairing. Tokens remain in page memory. Device tokens and jobs disappear on station restart.

This feature never changes the listening address or firewall. The station still binds to loopback by default. HTTPS or a secure tunnel must be deliberately configured before a separate phone/headset can reach it. For development, a tunnel terminating on the device's loopback is also possible. Do not put tokens in URLs or public screenshots. Headset rendering has priority; mobile browsers can suspend background tabs, so unattended background processing is not promised.

## Run a Python helper

From the project root, create an environment with Python 3.10+ and install `adapters/requirements-device.txt`. The module does not install anything, download models, open a listening socket or accept arbitrary code jobs.

Set `MYCOSCOUT_DEVICE_TOKEN` to the device token using your shell's environment mechanism, then:

```text
python -m adapters.device_worker --station http://127.0.0.1:8787
```

`--once` claims at most one frame for a smoke test. HTTPS is required outside loopback. Redirects are refused. A failure leaves the lease to expire; it never substitutes a fake result.

On a compatible computer with the existing detector dependencies and local weights, explicitly enable the detector:

```text
python -m adapters.device_worker --station http://127.0.0.1:8787 --detector-model models/grounding-dino-tiny --threads 4 --device cpu
```

Use `--device cuda` only when the installed runtime actually supports CUDA. The detector loads local safetensors, disables remote model code/downloads, then advertises `fungi-candidates-v1`. Its boxes and scores remain unverified and never claim species, edibility or GPS position.

## Feed sampled camera frames

The producer consumes an image or a folder of completed JPEG/PNG files. A camera process should write elsewhere and atomically rename finished frames into the watched folder. This is an input contract, not a camera driver. Supply the station owner token in `MYCOSCOUT_TOKEN` when configured; this is distinct from a helper token.

```text
python -m adapters.device_enqueue path/to/completed-frames --session-id field-recording-001 --watch --kind fungi-candidates-v1 --interval-seconds 10 --ttl-ms 30000 --max-frames 60 --max-duration-seconds 600
```

Watch mode samples the newest new frame and deliberately skips older backlog. The adapter is bounded to 1–1000 frames, at most one selected frame per 0.5–60 seconds, and at most one hour. Frame identifiers include filename and image hash; the supplied session identity is preserved. Source raster orientation must already be normalized by the camera writer. No EXIF/GPS, camera pose, distance or flight clearance is inferred. This pathway does not write training labels or activate a model.

## Protocol and boundaries

`GET /api/device-compute/status` and `POST /api/device-compute/{pair,revoke,jobs,simulate}` use existing station-owner access. Only `POST /api/device-compute/worker` accepts a separate device bearer token, limited to `announce`, `claim`, and `result`.

Jobs accept `kind`, `sessionId`, `frameId`, `imageBase64`, `mimeType`, `ttlMs` and optional `targetDeviceId`. The station validates raster headers/dimensions, computes the content hash and issues a random lease. A worker result must return the exact job, lease, session, frame and image hash. Expired, duplicate, revoked or mismatched submissions are rejected. Results accept only the whitelisted schema and remain `unverified` with `humanReviewRequired: true` and `flightAuthority: false`.

Limits: 16 paired devices; one outstanding lease per device; 32 pending jobs; 32 MiB total pending compressed images; 4 MiB/8 megapixels per image; 1–120 seconds per job; 100 candidate boxes per detector result. The last 64 finished jobs are retained in memory. Expired/revoked image bytes are discarded. A disconnected helper cannot block a flight loop because no mission execution path depends on this queue. Workers are trusted collaborators: authentication/provenance does not make their ML output correct.

## Reproduce measurements and simulations

```text
python -m experiments.device_compute_benchmark --output experiments/results/device-compute-local.json
node experiments/device_compute_simulation.mjs experiments/results/device-compute-local.json experiments/results/device-compute-scenarios.json
```

The checked-in run is in `experiments/results/device-compute-v1/`. The measured PC has an AMD Ryzen 7 5700X3D, 16 logical processors and 32 GiB RAM. The installed PyTorch runtime is CPU-only; this does not establish whether a physical GPU is present or how it would perform with another runtime. Detector execution uses four CPU threads. Model loading, first-inference warmup, JPEG decoding and warm detector inference are reported separately.

Simulation is deterministic and uses the measured PC processing time plus explicit hypothetical bandwidth, latency, jitter, whole-transfer failures and deadlines. It models one worker and serialized image uploads, with no retransmission, encoding overhead, contention, power draw or thermal throttling. It is a scheduling experiment, not a phone/Quest/Pi benchmark and not a mushroom accuracy or distance test. No other physical device, radio or drone was tested.

Use `node --test --test-isolation=none tests/device-compute.test.mjs` and `python -m unittest adapters.tests.test_device_worker` for contract and actual pixel-processing tests. Real device validation should record sustained foreground performance, battery/heat, suspend behavior, image quality, connection loss and false detections before using any optional analysis operationally.
