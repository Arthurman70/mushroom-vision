# Michigan Fungus Hunters pilot — October 26, 2026

The first real-photo experiment ran on 17 locally imported group images from 14 conservatively separated uploader groups. The pretrained GroundingDINO detector ran successfully on all 17. A small candidate-crop verifier was actually trained on the reviewed photo crops. Its held-out results were weak, so it should remain an experiment and should not gate a flight or suppress operator-visible observations.

The labels are approximate visual annotations, with human confirmation still pending. No species labels, edibility labels, measured mushroom distances, authenticated capture metadata, or drone-camera recordings were collected.

## Detector result

Five images had complete scene annotations: four mushroom-positive images and one negative woodland landscape. They contained 14 annotated target boxes. Twelve dense-colony or otherwise partially annotated images were excluded from box-level accuracy metrics; their explicitly reviewed crops could still be used for verifier training.

With GroundingDINO Tiny revision `a2bb814dd30d776dcf7e30523b00659f4f141c71`, prompt `a mushroom. a fungus.`, box/text thresholds 0.25, and explicit class-agnostic NMS at IoU 0.5:

| Measure | Pilot result |
|---|---:|
| Matched target boxes | 8 of 14 |
| Missed target boxes | 6 |
| Unmatched detections | 2 |
| Precision | 80% |
| Recall | 57.1% |
| Negative landscape false alarms | 1 on 1 image |

These are descriptive pilot results on five labeled images, not a broad accuracy estimate. The five small annotated targets below 64 pixels square-root box area were in the same lawn image; all five targets below that size were missed. Eight of nine larger targets matched. This motivates better small-object data and controlled inspection views, but it does not establish a universal 64-pixel threshold.

## Actual training result

The split was selected once by deterministic connected source groups, before training. Photos, original hashes, duplicate files, and all variants from one source group remained together. The training data had 34 reviewed crops from eight groups; validation had 27 crops from three groups; test had nine crops from three groups.

The verifier learns RGB/HSV histograms, coarse spatial appearance, and gradients with class-balanced logistic regression. It is a crop classifier, not a fine-tuned GroundingDINO detector. It trained for 1,000 optimization steps. The threshold of 0.10 was selected on validation F1 and applied unchanged to test data.

| Held-out crop measure | Result |
|---|---:|
| Mushroom crops retained | 3 of 4 |
| Background crops incorrectly retained | 3 of 5 |
| Precision | 50% |
| Recall | 75% |
| Overall crop accuracy | 55.6% |

This baseline has poor background rejection on the tiny held-out sample. More representative hard negatives, additional independent encounters, camera-specific captures, and stronger learned features are needed. The held-out split contains only one completely annotated detector image, with one target, so the unchanged detector-plus-verifier result on that image is not evidence of a useful end-to-end improvement.

## Controlled scale experiment

Two annotated originals were transformed five ways each. The whole scene was reduced and centered on a fixed-size neutral gray canvas, then blurred at 0.6 pixels and JPEG-encoded at quality 80. Fixed canvas dimensions prevent model resizing from undoing relative object scale. The gray border is an artificial cue, and these images are not real distant observations.

| Scene scale within the original frame | Matched transformed boxes | Recall |
|---|---:|---:|
| 100% | 6 of 12 | 50% |
| 50% | 4 of 12 | 33.3% |
| 25% | 3 of 12 | 25% |
| 12.5% | 2 of 12 | 16.7% |
| 6.25% | 2 of 12 | 16.7% |

The 60 transformed boxes are repetitions of 12 annotations on only two source images, not 60 independent mushrooms. No transformed targets below 32 pixels square-root box area matched; 14 of 15 transformed targets at least 64 pixels matched. Different camera preprocessing, blur, caps, stems, and scene clutter can change these figures substantially.

The earlier downsample-only experiment is separately labeled a **resolution degradation** experiment. Because detector preprocessing resizes the input frame, its smaller image files do not establish smaller-relative-object performance.

## Artifacts and repeatability

`vision/runs/facebook-pilot-01/study.json` records the exact source-manifest SHA-256, split hash, detector configuration, input prediction-file hashes, validation results, and summary metrics. The frozen source-manifest hash is:

```text
a26c00004ff77ecf1f1a76a8f09b6d346df9f8191059050ac1bfc37dfa9ed23a
```

The same run folder contains the frozen dataset and group split, plain-JSON verifier checkpoint, training report, original detector evaluation, held-out evaluation, verifier candidate scores, fixed-canvas evaluation, and per-scale comparison. Source photographs and derived images remain local and must be excluded from public redistributable bundles.

Reproduce the frozen real-photo study with:

```text
python -m vision.run_pilot --manifest data/facebook-michigan-fungus-hunters/manifest.json --predictions experiments/results/facebook-fullframe-nms-2026-10-26 --output-dir vision/runs/facebook-pilot-01
```

The runner refuses changed source labels in the same output directory. Detector evaluation refuses missing successful-inference candidate arrays, mixed model configurations, and confidence thresholds below the detector's export threshold. Completely annotated negatives, missing inference, partial labels, and synthetic variants have separate handling.

The next evidence needed is real camera footage with measured distances and many empty woodland views. Flight development can use the observed small-target weakness to justify slow pause-and-inspect behavior and conservative standoff experiments. No detector output or verifier score establishes obstacle clearance, safe descent, or traversable ground.
