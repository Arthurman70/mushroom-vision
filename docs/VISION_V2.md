# V2: black-trumpet photo ranking and discovery experiments

V2 separates two questions: can a model rank a photo or crop as resembling the
black-trumpet-labeled training photos, and can a detector find the mushroom in a
larger woodland frame? A photo classifier cannot answer the second question by
itself. Neither task determines edibility or safe aircraft movement.

## Stronger transfer baseline

`vision/transfer.py` compares two feature families on the **same** original
photos and observer-disjoint split:

- Previous-style handcrafted RGB/HSV, spatial appearance and gradient features
  (130 values), followed by a newly fitted binary logistic head.
- Frozen official ImageNet ResNet18 features (512 values), followed by a newly
  fitted binary logistic head. The backbone weights remain unchanged.

The task is weak-label `black_trumpet` versus `other_fungi` photo classification.
Observation taxon labels are not independent image verification: photos can
contain context, multiple organisms, or identification errors. These results
must not be described as verified species identification, localization, or
forest-background rejection.

The official [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html)
weights are downloaded from `download.pytorch.org`, checked against the full
SHA256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`,
and loaded with `weights_only=True`. No remote Python or local photos are sent
to external inference services. Official preprocessing resizes to 256 pixels,
then centrally crops to 224 pixels and applies ImageNet normalization. That
crop may omit off-center fungi, a real limitation for contextual photographs.

Preparation and training:

```powershell
& .\runtime-vision\Scripts\python.exe experiments/prepare_resnet_v2.py
& .\runtime-vision\Scripts\python.exe -m vision.transfer `
  --manifest data/public-fungi-v2/manifest.json `
  --output-dir experiments/results/v2-transfer-01
```

The source manifest is required to be frozen before training. The preparation
script downloads only the official 46,830,571-byte weight file. Training uses
the existing isolated local Python environment.

## Leakage and measurement rules

- Observer, observer group, observation, photo ID, original SHA and file SHA
  cannot cross train/validation/test splits. Exact duplicate files are rejected.
- Synthetic images cannot enter this original-photo training/evaluation
  manifest. Files are checked against recorded hashes.
- Feature normalization is fitted on training rows only. Logistic heads use
  class-balanced loss and a frozen L2 grid of 0.001, 0.01 and 0.1.
- L2 strength and feature family are selected by validation average precision,
  then validation ROC AUC. Test outcomes do not choose the architecture.
- The review threshold maximizes validation recall subject to at least 0.80
  empirical validation precision and at least ten selected validation photos.
  If no threshold meets that target, the report explicitly falls back to
  validation F1 and states the precision target was not met.
- Held-out test metrics include precision, recall, F1, balanced accuracy,
  average precision and ROC AUC. Observer-group bootstrap intervals describe
  sampling variability within this weak-label dataset; they do not cover label
  error, unknown near-duplicates, camera distance or forest deployment shift.

Checkpoint files are ordinary JSON, feature caches are NPZ with pickle loading
disabled, and every result records the frozen manifest hash and official weight
hash. `rank_photos` can score local candidate crops for human review; checkpoint
validation refuses a purported calibrated probability or enabled flight gate.

## Runtime check already completed

The first 100 actual licensed public photos decoded and produced finite frozen
features. Four CPU threads with batches of 16 took **1.43 seconds in the neural
backbone and 3.37 seconds overall**. This is a partial-dataset throughput check,
not training, accuracy, or single-photo live latency. Evidence:
`experiments/results/v2-throughput/feature-runtime.json`.

Seven checks cover tied-score metrics, validation threshold fallbacks,
independent observer splits, training-only normalization, and the restriction
that rankings cannot be represented as confirmed species or flight clearance.

## Completed training and held-out photo results

The frozen licensed public dataset contains 1,174 original photos: 450 weak
black-trumpet positives and 724 other fungi. It has 693 training, 207 validation
and 274 test photos; test photos come from 131 observers. Its frozen SHA256 is
`55f39187f168f0c9283b8cdc076694ad345e90ee588387263fbdf1cfb1021755`.
All 1,174 photos passed file checks and feature extraction; extraction took
34.60 seconds including 16.31 seconds in the frozen neural backbone.

ResNet18 was selected using validation data. On the untouched 274-photo test
split, with each model's validation-selected threshold:

| Photo ranking model | TP / FP / FN / TN | Precision | Recall | F1 | Average precision |
| --- | --- | --- | --- | --- | --- |
| Same-data handcrafted features | 57 / 16 / 42 / 159 | 78.1% | 57.6% | 0.663 | 0.722 |
| Frozen ResNet18 + trained linear head | 94 / 27 / 5 / 148 | 77.7% | 94.9% | 0.855 | 0.910 |

These are weak-label **photo classification** measurements, not mushroom
discovery or verified species accuracy. The selected head uses L2 0.1 and a
review threshold of 0.38. It achieved 83.9% precision on validation photos, but
**test precision fell below the 80% target**. It incorrectly prioritized 17 of
72 related-fungi negatives and 10 of 103 broader-fungi negatives. The latter
cohort includes rusts and smuts; it is not a mushroom-free forest-background
benchmark. Observer bootstrap intervals for test precision and recall are
67.4–85.1% and 89.6–98.4%, respectively, excluding unknown label error.

Full results, model-selection trials, exact feature provenance and row scores:
`experiments/results/v2-transfer-01/study.json` and `photo-scores.json`.
The selected JSON head is packaged at
`vision/checkpoints/black-trumpet-ranker-v2.json`; the 45 MB pretrained backbone
is downloaded separately, rather than embedded in phone/headset packages.

## Run the selected local ranker

```powershell
# Whole-photo/crop review scores:
& .\runtime-vision\Scripts\python.exe experiments/rank_photos_v2.py `
  --input-dir data/black-trumpet-discovery-v2/images `
  --output-dir experiments/results/new-photo-ranking

# Add rankings to existing candidate JSON, preserving every candidate box:
& .\runtime-vision\Scripts\python.exe experiments/rank_photos_v2.py `
  --input-dir data/black-trumpet-discovery-v2/images `
  --detections experiments/results/v2-discovery-generic `
  --output-dir experiments/results/new-candidate-ranking
```

Both modes were executed locally on the 12-photo audit. Candidate mode preserved
all 26 generic-detector boxes and added `reviewRanking`; it did not filter,
relabel, move, or invent boxes. The classifier was trained on whole fungal
photos, so candidate crops and forest background are a different distribution.
Scores prioritize manual review only; no automatic acceptance or flight gate is
enabled. Output JSON records the checkpoint hash, training hash and threshold.
The 12-photo audit overlaps the classifier's train/validation/test split by
7/3/2 photos. Its candidate-ranking execution is a compatibility demonstration,
**not** an independent accuracy measurement for the learned ranker, and it was
not used to select or fit that ranker. The 274-photo classifier test metrics
above use the separate untouched test split.

## Connect saved vision results to the inspection queue

The following bridge was executed on the saved detector outputs and habitat
inference: **26 visual candidates plus 9 habitat contexts produced 35 screen-only
leads**, with no rejection, no metric target, no actionable lead and no flight
command. Rankings remain review metadata and cannot overwrite detector scores
or authorize movement. The exact replay is saved locally under
`experiments/results/v2-scout-session/queue-replay.json`.

```powershell
python -m vision.export_scout_leads `
  --manifest data/black-trumpet-discovery-v2/manifest.json `
  --detections experiments/results/v2-ranked-candidates `
  --habitat-report experiments/results/habitat-v2/report.json `
  --output experiments/results/new-scout-session/flight-observations.json
node flight/replay-photo-leads.mjs `
  --input experiments/results/new-scout-session/flight-observations.json `
  --output experiments/results/new-scout-session/queue-replay.json
```

The optional habitat experiment uses generic ADE20K tree/vegetation/earth
categories. Its broad-scene versus close-up gate uses **manually reviewed context
labels**, not an automatic close-up classifier. Nine broad scenes passed the
generic-cue gate and three labeled close-ups were withheld. Neither this gate
nor its inference runtime measures ecological suitability. Run the model only
for research/evaluation, consistent with its separate NVIDIA license:

```powershell
python experiments/prepare_habitat_v2.py --research-evaluation
python -m vision.habitat --research-evaluation `
  --manifest data/public-fungi-v2/habitat-selection-v2.json
```

## Discovery results

GroundingDINO discovery uses a fixed 0.25 box threshold, 0.25 text threshold and
class-agnostic NMS IoU 0.5. The generic prompt is `a mushroom. a fungus.`; the
predeclared black-aware prompt adds `a black trumpet mushroom. a black
funnel-shaped mushroom.`. Both should be compared against independently drawn
boxes on the same frozen benchmark. The V2 audit uses a full frame plus
640-pixel crops with 25% overlap on four contextual photos, preserving all crop
transforms and raw boxes. The smaller tiles are deliberate because these source
photos are at most 1024 pixels on their longest side. More proposals do not
establish better localization or fewer false positives.

The selected development localization audit contains 12 positive photos and 30
approximate visible-body boxes. Its frozen manifest hash is
`334b8c209db455e4b7d32dc5bd633b0a410638b8c05d10e8295722a1e02542a0`.
A single assistant annotated them before predictions; there is no independent
expert adjudication. The audit is not a randomly sampled operational test and
contains no background-only images, so it cannot estimate background
specificity. The four contextual photos also have a separate fixed-canvas
stress set at scales 1, 1/2, 1/4 and 1/8, with constant blur/JPEG processing and
an explicitly artificial neutral background.

The completed original-photo audit did **not** establish a better discovery
default:

| Fixed discovery configuration | Images / ground-truth boxes | TP / FP / FN | Precision | Recall |
| --- | --- | --- | --- | --- |
| Generic full frame | 12 / 30 | 20 / 6 / 10 | 76.9% | 66.7% |
| Black-aware full frame | 12 / 30 | 22 / 23 / 8 | 48.9% | 73.3% |
| Generic full frame, context subset | 4 / 15 | 10 / 1 / 5 | 90.9% | 66.7% |
| Black-aware full frame, same subset | 4 / 15 | 9 / 5 / 6 | 64.3% | 60.0% |
| Black-aware tiled, same subset | 4 / 15 | 12 / 17 / 3 | 41.4% | 80.0% |
| Generic tiled, same subset | 4 / 15 | 10 / 5 / 5 | 66.7% | 66.7% |

The additional generic-tiled comparison was exploratory, added after inspecting
the initial black-aware results. Generic tiles did not recover more labeled
bodies and introduced additional unmatched boxes. Black-aware tiles recovered
more bodies at a substantial false-positive cost. Tiling took approximately
23–26 CPU seconds per contextual image. The generic full-frame detector remains
the default. Evidence is in `experiments/results/v2-discovery-comparison.json`;
raw boxes, transformed crop boxes and timings remain in each run directory.

The generic detector also completed all 16 fixed-canvas stress images. Against
the same 15 transformed annotated bodies at each level, precision/recall were
75.0%/60.0% at scale 1, 90.0%/60.0% at 1/2, 50.0%/46.7% at 1/4, and
35.7%/33.3% at 1/8. At the smallest scale, none of eight annotated bodies with a
bounding-box geometric mean under 16 pixels matched a detection. This is a
small controlled failure observation, not a validated universal pixel cutoff.
The scale-1 control includes the same blur/JPEG processing as the smaller
variants, so its results differ slightly from the untouched original photos.
Full evidence: `experiments/results/v2-discovery-scale-stress.json`.

Synthetic reductions test pixel-scale sensitivity only. They cannot establish
that a black trumpet is detectable from a particular number of meters. Genuine
distant positives with known camera optics, range, pose and obscuration remain
required for that claim. All candidate rankings remain separate from the
continuous depth, localization and aircraft safety controls.
