# Corpus review training and distance retest

The earlier unbalanced research candidate did **not** improve the frozen distance test. It matched 0 of 3 human-marked regions at every tested scale, including the original video frames. Zero false alerts accompanied zero detections; this is low sensitivity, not evidence of a useful reduction in false alarms. The candidate remains inactive. The controlled weight-adjustment experiments below retain this result as a baseline.

## What entered training

The corpus contains 4,545 unique inventory images. The completed AI review pass covers all of them: 240 originals were individually inspected, and 4,305 received explicitly incomplete contact-sheet screening. These are AI opinions, not human verification. Thumbnail observations do not provide fit-ready detector boxes.

The final immutable export contains 94 complete original-image reviews. Of those, 82 frames from three training videos contain 32 mushroom boxes in 26 positive frames and 56 explicitly negative frames. Twelve frames from a separate validation video contain three boxes in one positive frame and eleven negative frames. That positive validation frame is an editorial chanterelle photograph inset, **not live woodland footage**. The sample is too narrow to establish field performance.

Twenty explicit nonfungal training crops supplement the full frames: 16 hats, two other objects (a collecting bag and gloves), one tree region and one wood region. Tags such as “tree” or “leaf” alone never create negatives. Any negative crop overlapping a marked fungus is rejected, including overlap introduced by rounding crop boundaries. Uncertain ground targets and partially hidden mushrooms remain incomplete instead of being erased as negatives. Four tree/wood regions are available in the validation frames.

The loader checks exact image bytes, decoded pixels, source groups, frozen splits, permissions, complete-original status, latest review revisions and exact agreement with the immutable ledger. The reserved test creator and linked duplicate content cannot enter fitting. The current manifest and ledger were copied into the run before optimization.

## Fitting protocol and result

- Architecture: Faster R-CNN MobileNetV3 320 FPN, initialized from the pinned official COCO weights with a new mushroom/background head. The backbone body is frozen; the FPN, RPN and head learn.
- Eight complete passes over 102 examples: **816 optimizer updates**, approximately 161 seconds on four CPU threads.
- Mild brightness, contrast, saturation and horizontal-flip augmentation; no invented mushrooms or sharpening-generated detail.
- Fixed score threshold 0.3 and box overlap threshold 0.5. The frozen rule selects the best AI-validation F1, then fewer negative-frame alerts, then more matched targets, with earlier ties.
- All eight passes missed the validation set's three targets. Passes six and seven also produced one alert on an explicitly negative validation frame. The predefined rule therefore selected **pass one**, a checkpoint with **102 updates**, from the eight-pass run. The selected model is not represented as having received 816 updates.
- Selected-checkpoint validation: 0/3 matched, three missed, zero false alerts on eleven negative frames. None landed on the four explicit validation confounder regions.
- A separate fit-only diagnostic on the 82 training originals matched 2/32 boxes and missed 30. No alerts appeared on the 56 training negatives or any of the twenty training confounder crops. These are training-set observations, not generalization accuracy.

The first attempt stopped after one update because a truly black opening frame produced no RPN proposals and empty-ROI classification loss became non-finite. The repaired fitting path supplies one whole-frame background proposal only for explicitly negative training frames with zero proposals. It does not alter inference. Thirty such fallbacks occurred during the successful run. The failed attempt and its explanation are retained. Separately, an importer omission that dropped the reviewers' explicit confounder arrays was repaired before the successful snapshot; corrected reviews are appended as new ledger revisions.

## Fixed distance regression

The candidate was evaluated once on the exact saved pixels and human presence boxes from `video-distance-v2`, at the predetermined thresholds. No score tuning or further fitting followed the results.

| View | Human regions matched | Missed | Unmatched alerts on positive frames | Explicit negative control |
|---|---:|---:|---:|---|
| Original video frames | 0/3 | 3 | 0 | 0 alerts / 1 frame |
| 64 px target width | 0/3 | 3 | 0 | Not measured |
| 32 px | 0/3 | 3 | 0 | Not measured |
| 16 px | 0/3 | 3 | 0 | Not measured |
| 6.9 px | 0/3 | 3 | 0 | 0 alerts / 1 frame |
| 3.45 px | 0/3 | 3 | 0 | Not measured |
| 1.38 px | 0/3 | 3 | 0 | Not measured |
| 0.69 px | 0/3 | 3 | 0 | Not measured |

The earlier six-frame AI candidate also matched zero targets at all eight scales. Both use the same architecture, threshold and frozen pixel views. GroundingDINO uses a different model and score threshold, so its counts remain separate. Only three correlated positive source frames and one negative original are represented. This already inspected benchmark is a regression diagnostic, not an untouched release test or a population detection rate. Pixel resizing does not measure 10/20/50/100-meter camera performance.

## Fresh check of the active detector

After fitting, the unchanged GroundingDINO raw method was run freshly on all 26 saved pixel views with the original prompt, pinned revision, box/text thresholds of 0.25 and overlap threshold of 0.5. No prediction cache was used. Every count matched its earlier v2 raw result.

| View | Active detector matched | Unmatched alerts on positive frames | Explicit negative control |
|---|---:|---:|---|
| Original frames | 2/3 | 5 | 1 alert / 1 frame |
| 64 px | 3/3 | 4 | Not measured |
| 32 px | 1/3 | 7 | Not measured |
| 16 px | 0/3 | 5 | Not measured |
| 6.9 px | 0/3 | 5 | 2 alerts / 1 frame |
| 3.45 px | 0/3 | 6 | Not measured |
| 1.38 px | 0/3 | 6 | Not measured |
| 0.69 px | 0/3 | 6 | Not measured |

This fresh check confirms the active detector still loses these small targets and can produce alerts on leaf litter. The original source images are identical across synthetic scales, so neither a confidence interval nor a per-meter success rate is justified. Enhanced-color and tiled methods were not rerun in this raw check; their separate v2 measurements remain preserved. The main lab table now prefers the v3 raw report and labels missing methods as not measured.

## Interpretation and next training priorities

The useful outcome is a functioning auditable review-to-training path and a clear failure measurement. Even on its fit images the selected candidate missed most targets, so lack of distant detail alone cannot explain its failure. The weak validation coverage cannot distinguish a useful field detector from an overly quiet one. Before another model-selection experiment, annotate more complete positive woodland scenes and difficult negative scenes from distinct sources, and obtain human-reviewed field validation with a useful target count. Retain hats, broken stumps, bark, leaves and mushrooms attached to wood in that collection. Do not use the repeatedly viewed distance frames to choose thresholds or future winners.

## Artifacts

- Successful fit: `experiments/results/ai-corpus-candidate-v2/report.json`
- Frozen input files: `experiments/results/ai-corpus-candidate-v2/inputs/`
- Research checkpoint: `experiments/results/ai-corpus-candidate-v2/candidate.pt`
- Fit-only diagnostic: `experiments/results/ai-corpus-candidate-v2/fit-only-diagnostic.json`
- Distance report and illustrated misses: `experiments/results/ai-corpus-distance-v1/report.json` and `failure-examples.jpg`
- Failed initial attempt: `experiments/results/ai-corpus-candidate-v1/failure.json`
- Fresh active-detector check: `experiments/results/video-distance-v3/report.json` and `failure-examples.jpg`

The report API exposes only bounded aggregate results. The Mushroom Lab's distance section displays the new candidate separately from the earlier candidate and active detector. No source images, file paths, identities, activation switch or aircraft command is exposed through this aggregate endpoint.

## Controlled weight adjustments, 30 October 2026

The earlier sampler interleaved classes but gave 76 of 102 single-image updates
to negative examples. Its long negative tail could encourage silence. The new
research driver, `experiments/train_weight_adjustment.py`, compares two fixed
recipes without changing the source images, annotations or operating threshold:

| Setting | Balanced 320 | Balanced 640 |
| --- | --- | --- |
| Internal shortest / maximum side | 320 / 640 px | 640 / 1280 px |
| Each optimizer batch | One positive + one negative image | Same |
| Each complete pass | 76 positive + 76 negative exposures | Same |
| Complete passes / optimizer updates | 6 / 456 | Same |
| Initial weights / head seed | Pinned COCO / 8251 | Same |
| Learning rate | 0.001 | Same |
| Backbone body | Frozen | Same |
| Detection threshold / matching IoU | 0.3 / 0.5 | Same |
| Checkpoint rule | Fixed final pass, declared before training | Same |

Each negative example is visited once per pass. All 26 positive originals are
visited, with deterministic source rotation and two or three exposures each.
The 20 explicit negative crops remain included. The count of positive images
equals the count of negative images; foreground/background anchors and regions
inside an image are not themselves forced to be equal. This is a recipe
comparison with the earlier run, not an isolated scalar loss-weight experiment:
batching, exposure and the checkpoint rule also changed. The two new recipes
differ only in their declared image-size configuration.

An 80-update training-only diagnostic first fits the largest fractional
single-target training original. It passed by detecting its own labeled target.
Those diagnostic weights are discarded; the main run starts again from the
seeded official initialization. This is a learnability check, not field accuracy.

The saved checkpoints contain explicit model configuration because resizing and
score filtering are not stored in weight tensors. Final diagnostics and
regression inference reload that configuration. The diagnostic output floor is
0.01 to expose weak predictions; decisions and reported operating metrics stay
at 0.3. Threshold curves are explanatory and do not pick a new threshold. RPN
proposal recall separates missing regions from weak classification scores.

The frozen regression protocol evaluates each final candidate once on 26 saved
raw pixel views and the four native originals with the default ground-tile
profile (20 additional forwards). Distinct source counts use original images,
not resized duplicates. The earlier DINO raw and tiled baselines retain their
separate 0.25 threshold and source hashes. AI-only training labels do not become
human verification, and these checkpoints cannot activate themselves.

Implementation follows the pretrained-detector/head replacement pattern in the
[TorchVision fine-tuning tutorial](https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

### Completed weight-adjustment results

Both recipes completed six passes and 456 optimizer updates. Their initial
classifier hashes match, their final weight hashes differ, and their maximum
classifier-weight changes were 0.04694 (320) and 0.06011 (640). These are actual
trained checkpoints, not changes to the reporting threshold. Reloading the
saved files reproduced the declared resize configuration for evaluation.

| Measurement at score 0.3 / IoU 0.5 | Earlier selected candidate | Balanced 320 | Balanced 640 |
| --- | ---: | ---: | ---: |
| Training targets matched | 2 / 32 | 16 / 32 | 22 / 32 |
| Separate validation targets matched | 0 / 3 | 0 / 3 | 0 / 3 |
| Native regression targets matched, raw | 0 / 3 | 2 / 3 | 2 / 3 |
| Targets matched at each reduced regression size, 64 px through 0.69 px | 0 / 3 | 0 / 3 | 0 / 3 |
| Native regression targets matched, ground tiles | Not measured | 2 / 3 | 2 / 3 |
| Unmatched alerts on native positive frames, ground tiles | Not measured | 3 | 1 |
| Alerts on the one native negative control, ground tiles | Not measured | 0 | 0 |

Training-image fit is not independent accuracy. The 640 recipe matched three of
five 32–64 px training targets, versus zero for the 320 recipe, but neither
matched the four 16–32 px training targets. The 640 model also produced three
alerts on the 56 negative training originals and three on the 11 negative
validation originals; the 320 model produced none on those negatives. More
training-image matches therefore do not establish better field performance.

Each candidate completed its single fixed evaluation: 26 raw forwards plus 20
forwards for four native ground-tiled frames. Both reports verified input
integrity and returned `insufficient_evidence`, with activation disabled. The
regression set contains only three correlated positive targets from one source
and one negative original, with no native targets under 64 px. Synthetic reduced
views cannot establish accuracy at a measured distance. The separate validation
positive is an editorial chanterelle inset, not woodland footage.

The saved active GroundingDINO ground-tile reference found all three native
targets, with 23 unmatched alerts on positive frames and eight alerts on the
negative control at its separate 0.25 threshold. The new models are quieter on
this tiny native set but lose one target and fail the reduced-target checks.
Neither replaces the active detector. No active-model pointer was created.

Mean CPU ground-tile processing time across the four native frames was 0.263 s
for 320 and 0.639 s for 640, including decoding, five passes and merging but
excluding model loading. These desktop timings are not drone throughput claims.
The runs used PyTorch 2.6.0+cpu, TorchVision 0.21.0+cpu and four CPU threads.
All 59 focused training, sampling, checkpoint, diagnostic and evaluation tests
passed.

The next useful weight experiment needs additional complete, human-reviewed
woodland positives and hard negatives from independent sources, including small
targets. Reusing this regression set to choose a lower threshold or a winning
checkpoint would not supply that evidence.

- Balanced 320 checkpoint and fit: `experiments/results/weight-balanced-320-v1/`
- Balanced 640 checkpoint and fit: `experiments/results/weight-balanced-640-v1/`
- Balanced 320 regression and per-frame predictions: `experiments/results/weight-balanced-320-eval-v1/`
- Balanced 640 regression and per-frame predictions: `experiments/results/weight-balanced-640-eval-v1/`
- Repeatable training driver: `experiments/train_weight_adjustment.py`
- Fixed evaluation driver: `experiments/evaluate_weight_adjustment.py`
