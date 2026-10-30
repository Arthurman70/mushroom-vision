# Independent AI visual review and one-pass experiment

This is an unfinished development experiment. Codex inspected 21 original video frames through image-viewing tools and submitted its observations through the dedicated AI-review API. No detector suggestions were viewed before labeling. The saved run is visible at `/ai-review.html`, with original-pixel boxes, evidence, reviewer identity, uncertainty and immutable revisions.

## What was actually reviewed

The fixed initial queue used frames 002, 012, 022, 032, 042 and 052 from each of three licensed video sources: `youtube-1Z9FgVpYris`, `youtube-ng7VOJHum7U` and `youtube-OhN1BQgK2L8`. This temporal selection was independent of detector scores. A separate validation queue inspected frames 002, 022 and 042 of `youtube-go_1s6wVScI`.

Across all 21 frames, the AI recorded five visible-candidate frames, three absent frames and thirteen uncertain frames. Two positive frames had incomplete instance coverage. Only six complete training frames entered the experimental export: three positive frames with five boxes, and three negative frames. Ambiguous leaf litter, occlusion, motion blur and prepared food were excluded rather than automatically treated as negatives.

The training queue spans three source groups. Existing held-out footage from both videos by Joshua McConkey and the archive lawn-mushroom clip remains reserved. The three inspected validation frames were all unsuitable or uncertain; there are **zero eligible validation frames** in this first small experiment. This pass cannot establish accuracy or improvement.

AI opinions remain `reviewerKind: ai`, `humanVerified: false` and `trainingAllowed: false`. Only this separate experiment interprets `experimentalTrainingAllowed: true`. Human grades and the ordinary training pipeline are unchanged.

## One actual optimizer pass

`experiments/train_ai_review_candidate.py` performed one epoch over the six eligible frames using the existing official COCO initialization for Faster R-CNN MobileNetV3 320 FPN. It performed six optimizer steps on CPU, with four threads and a fixed seed. It checked source partitions, licenses, image hashes and decoded-pixel uniqueness before fitting.

The fitted classifier weights changed and the saved checkpoint reloaded successfully. This demonstrates a working AI-review-to-experiment path. A lower training loss is not evidence that distant detection improved. No held-out footage was used to fit or select this candidate, and no active model was replaced.

The candidate uses `mushroom-vision.ai-candidate-checkpoint.v1`, a schema the operational model loader does not accept. The private review ledger, downloaded images, exports and model weights are excluded from GitHub. Generated local evidence is under `data/ai-visual-review-v1/` and `experiments/results/ai-review-candidate-v1/`.

## Separate held-out evaluation

After fitting, `experiments/evaluate_ai_candidate_distance.py` ran 26 fresh forwards against the frozen human presence boxes, with the score threshold fixed at 0.3 and matching IoU at 0.5. The candidate emitted no boxes: **0/3 targets matched at every one of the eight size conditions, including native resolution**. This is a failed candidate, not an improvement. No threshold tuning, additional fitting or activation followed this evaluation. The three positive frames come from one video, so the result is a small diagnostic rather than a population accuracy estimate.

The separate evaluation report is `experiments/results/ai-candidate-distance-v1/report.json`. The fitting report's `testEvaluationPerformed: false` describes fitting only; the later evaluation is deliberately recorded separately. The lab displays both results without confusing the candidate with the operational GroundingDINO detector.

To run another bounded experiment after inspecting new frames and exporting them:

```text
python -m experiments.train_ai_review_candidate --export path/to/train-export.json --export path/to/validation-export.json --output experiments/results/new-ai-candidate
```

Use a new output directory. Preserve source reservations, record uncertainty and compare future candidates against genuinely independent human annotations before considering deployment. The separate distance retest uses the old frozen human presence boxes and reports its limitations in `VIDEO_DISTANCE_RETEST.md`.
