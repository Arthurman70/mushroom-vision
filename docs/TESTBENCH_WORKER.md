# Local detection comparison worker

`vision.testbench_worker` is a persistent JSON-lines process used by the photo testing app. It reuses the existing local GroundingDINO model through `adapters.scout_frame.LocalDetector`. Model files must already exist in `models/grounding-dino-tiny`; the request cannot select a model, path, executable, device or download URL. CPU inference uses four threads. Loading is lazy so malformed requests do not load the model.

```powershell
runtime-vision\Scripts\python.exe -m vision.testbench_worker
```

Send one JSON object per line:

```json
{"id":"example-1","imageBase64":"...","mimeType":"image/jpeg","settings":{"methods":["original","color","tiles"],"tileGrid":2,"scale":1,"boxThreshold":0.25}}
```

The request accepts still JPEG/PNG only, up to 8 MiB and 20 megapixels, with at least 16 pixels per side. Format and MIME must agree. Animated PNG is rejected. EXIF orientation is applied before inference; the image checksum identifies the original uploaded bytes. Base64 uses the plain encoding, without a data-URL prefix.

Settings are validated independently for every request:

- `methods`: a nonempty, unique subset of `original`, `color`, `tiles`; default `original`. Processing and result order are canonical.
- `tileGrid`: 2 or 3; default 2.
- `scale`: 1, 0.5, 0.25 or 0.125; default 1.
- `boxThreshold`: finite 0.1–0.8; default 0.25. Text threshold remains 0.25.

Unknown setting keys are rejected. Threshold fields are reset on the warm detector before each comparison, so a previous request cannot silently carry its threshold forward.

## Comparison methods

**Original** runs the complete working RGB image. **Color** runs a separate image with the fixed local luminance/chroma enhancement from `vision.far_color`; original pixels are not modified. **Tiles** combines the original full-frame predictions with overlapping crops of the working image, then performs class-agnostic IoU 0.5 NMS. A grid spans the entire image, anchors the right and bottom edges, and has approximately 20% overlap. Integer rounding can slightly change the overlap. Very small scaled canvases deduplicate identical grid positions.

All selected methods use the same detector weights, phrase prompt and score threshold. They are compared independently; color predictions are never silently merged into the original or tiled comparison. When both `original` and `tiles` are selected, the full-frame pass runs once and is reused. Two-by-two tiles with all methods require six model forwards; three-by-three requires eleven for normal-size images.

Scale reduces the entire source image before all methods run. Coordinates are projected to the **original oriented image** using the actual rounded working dimensions, separately for X and Y. Tile offsets are applied before this projection. This avoids treating a nominal 0.5 scale as exactly one half when an odd-sized image rounds. The worker returns both original and working dimensions. The model processor may resize its input tensor; tiling provides larger views of existing working pixels, not additional optical detail.

Each method returns at most 100 validated boxes, ordered by descending uncalibrated score after NMS. `preNmsCandidateCount` reports valid above-threshold boxes before NMS and the display cap. Every candidate has a stable within-result method ID, original-coordinate box, score and source pass. No caller/model-provided species, range or flight instruction is propagated.

Color-only proposals are produced only when `color` is selected. They appear in a separate `colorCues` array, capped at 12, with `contrastScore`, `colorOnly: true` and `cueType: color_salience`. They have **no detector confidence or species label** and do not count as detected mushrooms. Their coordinates are also projected from the working image to the original image.

## Protocol and timing

Progress example:

```json
{"id":"example-1","status":"progress","progress":{"stage":"tile_2","completed":3,"total":6}}
```

Stages include `loading`, `original`, `color`, `tile_1` through `tile_9`, `color_cues` and `complete`. Progress is emitted around each model forward; completed counts are monotonic. A loading or color-cue event is not counted as a model forward. Standard output contains only JSON protocol; dependency diagnostics are redirected to standard error.

Success has `{id,status:"ok",result:{...}}` with schema `mycoscout.testbench-result.v1`, image/settings/model metadata, an array of method results, separate color cues, warnings, `actualDistanceM: null` and `actuationEnabled: false`. Errors have `{id,status:"error",error:"safe message"}`. Fixed validation messages are exposed; raw model exceptions, local paths and uploaded bytes are not.

`result.seconds` measures the comparison after image decode/model loading. Per-method `seconds` measures model forward plus candidate processing. Tiled seconds include the reused original pass for a standalone-method cost comparison; `additionalSeconds` records just the extra tile work, and `reusedPasses` identifies reuse when the original method is also displayed. `modelPasses` is the number actually executed. Do not add all displayed method timings to derive total wall time. Enhanced-view construction and color cue analysis are included in total comparison time, but not detector-forward timings.

A parent process should allow a bounded total runtime suitable for up to eleven CPU forwards, such as 240 seconds. Progress does not authorize an unlimited extension. The worker processes requests sequentially, recovers after invalid requests/model failures, and closes after an oversized input line instead of consuming an unbounded stream. Its maximum protocol input line is 12 MiB.

## Validation and interpretation

```powershell
runtime-vision\Scripts\python.exe -m unittest vision.tests.test_testbench -v
```

The 16 tests cover odd-sized tile coverage, small-grid deduplication, tile translation, fractional scale projection, invalid-box rejection, metadata stripping, NMS and output caps, independent methods and full-frame reuse, per-request threshold reset, original-pixel preservation, conditional color processing, separated color-cue coordinates, malformed settings, EXIF/MIME/animated-image validation, byte/pixel limits, protocol recovery, warm-model reuse, sanitized errors and request-line limits. Actual-model smoke results are recorded separately under `experiments/results/testbench-v1/`.

The app supports manually assessing missed objects and false positives. More detections, higher model scores or more aggressive contrast do not prove improvement. All comparisons require the same independently annotated ground truth, matching rule, scene grouping and negative controls. Scaling a photograph does not establish a real drone detection distance, and these outputs cannot command physical flight or confirm species, edibility or sale eligibility.
