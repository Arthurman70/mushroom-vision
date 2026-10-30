# Video-frame distance retest

This experiment separates **finding a mushroom** from **guessing the category of an already located crop**. The expanded category model does not supply new detection weights or restore detail lost at a distance.

Run the frozen retest with the optional local vision runtime and previously acquired assets:

```powershell
.\runtime-vision\Scripts\python.exe experiments/video_distance_retest.py
```

The script requires the original `video-distance-v1` report and its images, the video collection, both classifier checkpoints and their matching training manifests, and locally installed model weights. These private/local artifacts are deliberately absent from the source-only GitHub release. Choose a new `--output` for each repeat; existing runs are never overwritten.

## What is measured

- Three saved, human-boxed mushroom views and one human-labeled negative control from one video are reused byte for byte. All new detector inference is fresh. Frames and resized variants are correlated; they are not independent mushrooms.
- Original RGB and the existing fixed color enhancement run at native size and requested target widths of 64, 32, 16, 6.9, 3.45, 1.38 and 0.69 pixels. The exact realized widths are retained rather than rounded to whole pixels.
- The overlapping 2 × 2 tiled method runs at native, 64, 32 and 16 pixels. Smaller tiled conditions are **not remeasured**; the page leaves them unavailable instead of copying earlier results.
- Three additional native video frames with partial human annotations contain roughly 41–61 pixel lawn mushrooms. Only recall on the marked regions is measured; unmarked areas cannot supply false-positive labels.
- The detector, score threshold 0.25, overlap threshold 0.5 and image transforms are fixed. There are 107 new detector forwards, using four CPU threads.

The old and expanded category heads each see the same outward-rounded crop of the known human box. This deliberately gives the classifier a location that a real detector may fail to find. The coarse comparison category, “morel group,” is an assistant visual opinion about the native frames. The saved human reviews identify mushroom presence, with category unknown. Therefore this table reports **agreement with a provisional AI group label**, not human species accuracy.

Each head keeps its own previously selected validation policy: the earlier head uses score 0.45 / margin 0.10; the expanded head uses score 0.40 / margin 0.30. Retained-guess differences reflect both the head and its fixed abstention policy. None of these settings were selected from this retest.

Below 32 original pixels on either target dimension, classifier predictions are diagnostic only. A high score does not override the existing live-pipeline eligibility guard. Cropping and network resizing do not create original image detail.

## Evidence and integrity

The run writes `experiments/results/video-distance-v2/report.json`, fixed protocol and checkpoint hashes, per-frame predictions, category guesses, a separate small-target evidence file, and a failure contact sheet. The lab's video-distance panel prefers v2 when present, preserves v1 as a historical fallback, and refuses to display a corrupt v2 report as a successful old run.

Before inference, the script checks exact source-frame hashes, resized-image hashes, training-manifest/checkpoint consistency, and source-group or exact-image overlap with training and validation. It rechecks immutable inputs at completion. This is an exact-content and source-group audit, not a guarantee that no visually similar image exists elsewhere. No test frame enters training, and no review, split, threshold or deployed model is changed.

## Interpreting range

The 6.9, 3.45, 1.38 and 0.69 pixel conditions were originally motivated by an illustrative 5 cm mushroom projection at 10, 20, 50 and 100 meters. The videos have no calibrated camera-to-subject distance, lens model or measured cap sizes. These are **pixel-sensitivity tests**, not measured success rates at those distances. Whole-frame shrinkage on gray padding also differs from real scenery, motion blur, occlusion, atmosphere and sensor noise.

The practical investigation policy remains: collect a clearer view, retain uncertain candidates, and obtain human verification. A distant color anomaly is a reason to inspect a region, not a confirmed mushroom or species label.

## Measured detector results

The completed run made **107 fresh detector forwards in 913.1 seconds**. Every comparable match and unmatched-box count was identical to v1. The expanded category training did not change this detector.

| Requested target width | Original RGB matches | Color-enhanced matches | Tiled matches |
| --- | ---: | ---: | ---: |
| Native video views (66.9–259.9 px) | 2 / 3 | 2 / 3 | 2 / 3 |
| 64 px | 3 / 3 | 3 / 3 | 3 / 3 |
| 32 px | 1 / 3 | 1 / 3 | 3 / 3 |
| 16 px | 0 / 3 | 0 / 3 | 2 / 3 |
| 6.9 px | 0 / 3 | 0 / 3 | Not rerun |
| 3.45 px | 0 / 3 | 0 / 3 | Not rerun |
| 1.38 px | 0 / 3 | 0 / 3 | Not rerun |
| 0.69 px | 0 / 3 | 0 / 3 | Not rerun |

Tiling also produced more unmatched boxes: at 32 px it produced 11, versus 7 for original RGB and 4 for color-enhanced RGB. On the native negative frame, original/color each produced one false alert and tiling produced six. At 6.9 px scene scale, original and color each produced two false alerts on the one negative frame. These small controls are not population false-positive rates.

The separate partial-label native test matched **3 / 3** marked lawn-mushroom regions, with actual widths 41.2–60.6 px. This does not establish morel recall; repeated views and unknown unmarked regions prevent that interpretation.

The original inference report is retained alongside the final report. A summary derivation records the display-name mapping from “morels” to the checkpoint's canonical `true_morels` category ID; saved guesses, pixels, thresholds and detector counts were not changed.

## Crop-framing diagnostic

`experiments/video_distance_crop_context.py` compares three fixed ways of presenting the known target to both category heads: tight crop, the same crop centered on a neutral square, and a square crop with 25% surrounding context. Its protocol is written before inference. This exploratory diagnostic was motivated by the native tight-crop failures; it must not be treated as independent validation or used to tune a production policy on the test set.

For the expanded head, top-guess agreement with the provisional AI morel label was:

| Requested target width | Tight crop | Square padding | Square context |
| --- | ---: | ---: | ---: |
| Native video views | 0 / 3 | 2 / 3 | 2 / 3 |
| 64 px | 1 / 3 | 2 / 3 | 2 / 3 |
| 32 px | 1 / 3 | 2 / 3 | 2 / 3 |
| 16 px | 0 / 3 | 0 / 3 | 1 / 3 |
| 6.9, 3.45, 1.38 and 0.69 px | 0 / 3 at each width | 0 / 3 at each width | 0 / 3 at each width |

The square-context policy retained two agreeing guesses and abstained on the third native view. At 32 px it retained all three, including one wrong guess. Below 16 px several policies confidently retained wrong categories. The live native-pixel guard therefore remains essential; it was not relaxed or bypassed by this experiment.

Results are in `experiments/results/video-distance-context-v1/report.json`; the lab exposes only its bounded aggregate counts. A local contact sheet, `crop-framing-examples.jpg`, illustrates the original-view differences. No detector or live crop policy was changed. The next evaluation should use new, independently reviewed sources before adopting a framing policy.

## Separate AI-trained candidate

`experiments/evaluate_ai_candidate_distance.py` evaluated the inactive Faster R-CNN candidate once, after the main retest completed. It verified the checkpoint and AI-review export hashes and rejected overlapping fit/test source groups. This candidate had one training pass on **six AI-reviewed frames from three source groups, containing five provisional boxes**. It is a different architecture from GroundingDINO, evaluated at its fixed score threshold **0.30**, with the same overlap threshold **0.50**.

It emitted **no boxes above threshold** on any of the 26 frozen views: **0 / 3 targets at each of the eight conditions, including native**. Zero false alerts here reflect an inactive detector response, not successful recognition. The evaluation took 1.75 seconds after loading. No threshold was lowered, further training performed, or candidate activated in response to the result.

The report is `experiments/results/ai-candidate-distance-v1/report.json` and appears in a separately labeled lab section. It demonstrates why the one-pass AI-review exercise is a workflow prototype, not evidence of a better detector. More independently checked training examples and a separate development/validation evaluation are needed before another reserved test.
