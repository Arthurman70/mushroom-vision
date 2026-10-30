# Corpus review training and distance retest

The new research candidate did **not** improve the frozen distance test. It matched 0 of 3 human-marked regions at every tested scale, including the original video frames. Zero false alerts accompanied zero detections; this is low sensitivity, not evidence of a useful reduction in false alarms. The candidate remains inactive.

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
