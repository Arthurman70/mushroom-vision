# Local mushroom vision runtime

The Windows development computer has an isolated Python 3.12 CPU environment at
`runtime-vision/`. The official GroundingDINO tiny model is downloaded at
`models/grounding-dino-tiny/`. Both folders are local runtime assets, not phone or
Quest packages, and must be excluded from source archives.

## Reproducibility

- Model: [`IDEA-Research/grounding-dino-tiny`](https://huggingface.co/IDEA-Research/grounding-dino-tiny).
- Resolved revision: `a2bb814dd30d776dcf7e30523b00659f4f141c71`.
- Safetensors weight file: 689,359,096 bytes. Exact file checksums and model source
  are in `models/grounding-dino-tiny/mycoscout-model-provenance.json` and copied
  into each run's `summary.json`.
- Tested libraries: PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu, Transformers 4.51.3,
  NumPy 1.26.4, Pillow 11.2.1, safetensors 0.5.3.
- Full installed environment: `experiments/requirements-vision-windows-cpu.lock.txt`.
- Model code is provided by pinned Transformers; loading uses `trust_remote_code=False`,
  `use_safetensors=True`, and `local_files_only=True`. Photo inference has no need
  for network access. No photos are sent to an external model service.

The processor uses the versioned [GroundingDINO Transformers interface](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/grounding-dino).
Text prompts are lowercase phrases separated by periods.

## Create another isolated CPU environment

Use a supported CPython 3.12 installation. These commands install into the new
environment, without modifying global Python packages:

```powershell
python -m venv runtime-vision
& .\runtime-vision\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.6.0 torchvision==0.21.0
& .\runtime-vision\Scripts\python.exe -m pip install -r adapters/requirements-vision.txt
$env:HF_HUB_DISABLE_TELEMETRY='1'
$env:HF_HOME=(Join-Path (Get-Location) 'models\hf-cache')
& .\runtime-vision\Scripts\python.exe experiments/prepare_grounding_dino.py --revision a2bb814dd30d776dcf7e30523b00659f4f141c71
```

`prepare_grounding_dino.py` resolves the requested official model revision once
and downloads the allowed assets at that exact commit. The command above pins
the experiment's revision; omitting `--revision` resolves the current upstream
`main` branch, which can change.

## Run photographs locally

```powershell
$env:HF_HUB_OFFLINE='1'
& .\runtime-vision\Scripts\python.exe experiments/run_grounding_dino_batch.py `
  --input-dir data/facebook-michigan-fungus-hunters/raw `
  --output-dir experiments/results/new-review-run `
  --threads 4
```

The output folder must be new or empty. Each image gets its SHA256, oriented
resolution, original-pixel boxes, phrase-matching scores, and timing in a JSON
review file. The model is loaded once per run. `summary.json` records model
checksums, software versions, all per-image counts, and any errors. The default
exploratory box/text thresholds are both 0.25. The existing single-image adapter
has a more restrictive default box threshold of 0.35.

Scores and box counts do not establish accuracy. The detector can produce
overlapping boxes for the same mushroom under the two related prompt phrases.
Evaluation must consistently handle those duplicates, and use independently
reviewed boxes including photos with no mushrooms. A score threshold selected on
validation images must be frozen before assessing held-out test images.

### Fixed tiling and duplicate-suppression baseline

The batch runner accepts `--nms-iou 0.5` to enable explicit class-agnostic NMS.
`candidates` then contains the retained detections and `rawCandidates` retains
every unsuppressed result. This is a fixed initial baseline, not a threshold
tuned on the held-out photos. Duplicate detections with intersection-over-union
greater than 0.5 are suppressed in score order, even when their phrase labels
differ. Nearby individual mushrooms and tile-boundary fragments can still
produce ambiguous duplicates; NMS is not an instance-count guarantee.

`--tile-size 1024 --tile-overlap 0.25` adds edge-anchored crops **in addition to**
the full frame. Crops run through the same processor. Each output box is first
rescaled by the processor to its crop's pixels, then translated by the crop's
original-image top-left coordinate. The JSON records original crop bounds,
local and original boxes, source view index, tensor resolution, preprocessing
time, and inference time. Four automated geometry checks cover edge coverage,
narrow images, coordinate translation, and duplicate suppression.

```powershell
& .\runtime-vision\Scripts\python.exe experiments/run_grounding_dino_batch.py `
  --input-dir data/facebook-michigan-fungus-hunters/raw `
  --output-dir experiments/results/new-tiled-review `
  --include fb-1657901556042327.jpg fb-3422432984600787.jpg `
  --tile-size 1024 --tile-overlap 0.25 --nms-iou 0.5
```

Tiling is an experiment for retaining small targets during image resizing; it
adds latency proportional to the views. It does not infer actual camera distance
or recover occluded fungi. A reduction in source-photo resolution alone is a
detail-degradation test: the model resizes it again, so it is not a valid test of
smaller relative mushroom size. The separate fixed-canvas stress curriculum
tests target pixel size on an artificial background; neither synthetic protocol
validates real distant woodland imagery.

## Completed initial run, October 26, 2026

The first two actual group photos were processed locally without errors:

| Local image | Resolution | Candidate boxes at 0.25 | Largest score | CPU inference |
| --- | --- | --- | --- | --- |
| `fb-1678244137055026.jpg` | 2048 × 1459 | 7 | 0.801 | 7.04 s |
| `fb-4546032465542486.jpg` | 1536 × 2048 | 4 | 0.700 | 6.53 s |

Evidence is in `experiments/results/facebook-initial-2026-10-26/`. Model loading
took 1.38 seconds. Timings measure this Windows computer with four CPU threads;
they do not predict Jetson or drone hardware speed. These two images were a
runtime smoke check, not a representative accuracy benchmark. The grass photo
contains both large foreground and small background mushrooms; the detector
finds several but also emits duplicate boxes. Physical camera distances are
unknown.

The subsequent full-frame run processed all 17 acquired originals without
runtime errors. It emitted 64 raw phrase detections and 38 retained candidates
after the fixed cross-label NMS baseline. Mean CPU inference was 6.62 seconds per
image (median 6.46 seconds); all 17 photographs had at least one candidate. These
are execution and candidate-count statistics, not precision or recall. The
complete evidence is in
`experiments/results/facebook-fullframe-nms-2026-10-26/summary.json`.

Two selected contextual photos were also processed with the fixed tiling
baseline (one full frame plus six crops each):

| Local image | Full-frame candidates / CPU time | Tiled candidates / CPU time |
| --- | --- | --- |
| `fb-1657901556042327.jpg` | 3 / 6.56 s | 10 / 40.73 s |
| `fb-3422432984600787.jpg` | 3 / 7.81 s | 6 / 43.14 s |

Both columns use NMS IoU 0.5. Tiling increases the number of proposed boxes and
costs approximately six times the inference time here. It can split one fungal
colony into multiple regions and produce false positives, so the additional
boxes do not establish better accuracy. These are ordinary group photographs
with unknown physical camera distance, not validated far-distance drone frames.
Raw and retained boxes are both available under
`experiments/results/facebook-tiled-nms-2026-10-26/`.

Two separate synthetic tests also completed without runtime errors, each using
five scale levels (1, 1/2, 1/4, 1/8, 1/16) derived from the initial two photos:

- `experiments/results/facebook-stress-2026-10-26/`: reduced source resolution,
  blur, and JPEG degradation, with raw candidate output. The processor enlarges
  small source frames again, so this does not test smaller relative targets.
- `experiments/results/facebook-fixedcanvas-nms-2026-10-26/`: resized scenes
  centered on original-size neutral canvases, with fixed NMS IoU 0.5. This tests
  reduced target pixel size within a synthetic frame. It does not model woodland
  occlusion, outdoor depth accuracy, camera motion, or real camera distance.

These variants share two source photos and are not independent field samples.
Any labeled evaluation must preserve that dependence and report synthetic
results separately from original-photo results. No scores or thresholds were
optimized on these variants in the runtime experiments.

## Flight implications

At roughly seven seconds per frame on this computer, semantic detection belongs
in a stop-and-scan or asynchronously queued survey process. It must not own
time-critical obstacle avoidance. Depth and flight-control health must be
checked continuously by separate components. A delayed mushroom detection must
retain the capture timestamp and camera pose; applying it to the aircraft's
current pose would move the apparent mushroom location.

Model inference does not supply metric depth, free space, localization,
species certainty, or edibility. Candidate outputs remain `UNLOCATED` until
matched to synchronized calibrated RGB/depth/pose data. The results are
pretrained inference: no GroundingDINO weights have been retrained on group
images. The separate candidate-crop verifier was fitted from reviewed original
photo crops using source-group splits; that small linear model is distinct from
GroundingDINO. Its uncalibrated crop scores do not establish detector quality or
flight clearance. See `VISION_PIPELINE.md` and `VISION_RESULTS.md` for that
experiment's separate metrics and limitations.
