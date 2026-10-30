# Far-pixel color and discovery audit

This is a controlled **synthetic pixel-size development test**, not a measurement of how often a drone detects a mushroom at a real distance. The source images have unknown camera distances and no calibrated depth. Four previously used, licensed iNaturalist black-trumpet contextual photographs supply both the small scene islands and the background texture. Source identities, hashes, licenses and attribution are retained in the input manifest.

The frozen protocol is `experiments/results/far-pixel-v1/protocol.json`. `inputs.json` records generated image hashes, untouched source hashes, exact source dimensions, fractional geometry and attribution. `predictions/` contains every actual local model output and separate color-only cue output. `results.json` contains per-image matching; `report.json` is the aggregate suitable for the lab display.

## What the test does

- Four source scenes, four illustrative far-pixel levels (**6.9, 3.45, 1.38 and 0.69 pixels**), four additional **32-pixel controls**, and four clutter-only negative controls: **24 images, 48 actual model inferences**.
- Retains the source canvases (768 × 1024 or 1024 × 768). Each whole contextual scene is reduced onto a target-free background crop from that source. The first annotated body's width determines the requested target width; all annotated bodies remain available for localization matching.
- Uses 16× supersampled coverage and area integration. A requested 0.69-pixel body is not rounded up to a full opaque pixel. Both requested and realized floating-point target widths are saved; the small residual rounding comes from the supersampled scene grid.
- Compares the original RGB composite with a separate, fixed color-enhanced view through the **same cached GroundingDINO Tiny weights**, prompt, thresholds and full-frame preprocessing. Enhancement increases local chroma differences and luminance differences; original RGB pixels are preserved for evidence and human review.
- Separately measures local CIE Lab color-anomaly proposals with a fixed threshold of 8 CIE76 Delta E and at most 12 proposals per image. These proposals are investigation cues, never mushroom detections or species predictions.
- Requires IoU ≥ 0.5 for a detector match, with one-to-one matching. A box covering most of the frame cannot count as spotting a tiny mushroom. Primary-target recall and all unmatched detections are both recorded.
- Color-cue proximity uses the peak point inside the focal target box expanded by two native pixels. This deliberately permissive metric is reported separately from detector recall, alongside off-target cues. It must not be substituted for mushroom-detection accuracy.

The model's own processor resizes the complete retained canvas to its standard tensor size. The test does not crop around the known target or enlarge the target independently for inference. Its input tensor shape is recorded for every forward pass. No detector or color threshold is optimized on these results.

## Completed results

| Requested annotated-body width | Raw GroundingDINO focal hits | Color-enhanced focal hits | Color-only nearby cues |
| --- | ---: | ---: | ---: |
| 6.9 pixels | 0 / 4 | 0 / 4 | 1 / 4 |
| 3.45 pixels | 0 / 4 | 0 / 4 | 0 / 4 |
| 1.38 pixels | 0 / 4 | 0 / 4 | 0 / 4 |
| 0.69 pixels | 0 / 4 | 0 / 4 | 0 / 4 |
| 32-pixel positive control | 2 / 4 | 2 / 4 | 1 / 4 |

**The enhanced view did not improve mushroom detection in this pass.** It also increased false detections on the four negative controls from **7 to 8**; both methods produced at least one false detection on every negative frame. Across the 16 far-pixel positive composites, the raw detector produced 28 unmatched boxes and the enhanced detector 29. Neither method matched any of the 60 annotated bodies in those far composites, including the 16 focal targets. The 32-pixel controls establish that the pipeline can localize some larger targets, not that 32 pixels is a reliable operational cutoff.

Color salience produced the maximum budget of **12 cues on every image**. At 6.9 pixels, one of four focal targets had a nearby cue, but **46 of 48 cues were away from all annotated bodies**. The smaller far levels had 48 off-target cues each. The negative controls produced 48 cues in total. Therefore the one nearby color cue is not a validated improvement in mushroom detection: it came with substantial clutter and may reflect an artificial composite edge. Color cues remain low-trust investigation leads and are excluded from species-classifier evidence until the original image is reviewed.

All 48 planned model forwards completed without errors. The local model's nine recorded files were independently checked against their pinned SHA-256 manifest (`model-file-checks.json`). The final protocol hash is `d058a85f1a5e26955af0cfd0b2ada6405adeca80ca45842b3f460a47130de2f4`. Eight scientific invariant tests passed. No parameters were adjusted in response to these low results.

## Negative-control audit amendment

An initial pilot was stopped after seven first-scene forward passes. Enlarged visual inspection revealed an ambiguous tiny pale body in the third source's proposed negative crop. That crop was narrowed to clean moss **before any inference for that source**. Initial protocol, images and partial predictions are preserved in `invalidated-background-pilot/`. The corrected protocol declares this amendment; the final benchmark starts again from the first scene. Thresholds and all other choices remain unchanged. The negatives were visually reviewed by an assistant, not an expert, and are not a representative woodland-flight negative dataset.

## Reusable color analysis

```powershell
runtime-vision\Scripts\python.exe -m vision.far_color --image path\frame.jpg --output path\new-color-cues.json
```

Output schema: `mycoscout.color-proposals.v1`. The image block contains the original image SHA-256, EXIF-oriented dimensions and pixel convention. Each cue has `cueType: color_salience`, `colorOnly: true`, `status: UNLOCATED`, original-image coordinates, and no species, pose or depth. `contrastScore = tanh(maxDeltaE / 20)` is a bounded contrast severity score, **not** confidence that the object is a mushroom. Optional detector `colorEvidence` uses the same original-pixel measurements with `enhancedViewUsed` marking which model view produced the box.

The module exposes `enhance`, `contrast_map`, `color_evidence`, `proposals` and `analyze`. It is local-only, caps images at 20 megapixels and proposal budgets at 50, rejects non-finite parameters, and does not download models or write to the review database. The default budget is 12 proposals with 16-pixel peak suppression; clutter can easily fill that budget.

### Quarantined benchmark review copies

The initial frozen color prediction files omitted top-level `synthetic` and `evaluationOnly` markers, although their input manifest identified every frame as synthetic evaluation evidence. The frozen files remain unchanged. For review ingestion, use **`experiments/results/far-pixel-v1/review-ready-color/`** instead of the original color prediction files. All 24 derived copies explicitly set both flags to `true` and include `derivedFromPredictionSha256`. `review-ready-color/derivation-audit.json` records every original/derived hash and confirms that no inference was rerun. Candidate scores, boxes and image evidence are identical to the original reports.

The benchmark generator now writes both markers on all future color reports. `experiments/far_pixel_review_derivation.py` produces the immutable review copies from a prior run, validates the frozen protocol/input hashes, and refuses to overwrite an existing derivation. Two added regressions verify quarantine markers, unchanged source payloads, copied provenance hashes, and rejection of invalid schemas/digests; the current suite has **10 passing tests**. These flags prevent synthetic evaluation cues from being treated as new field-training records.

## Reproduce a fresh version

```powershell
runtime-vision\Scripts\python.exe -m unittest experiments.test_far_pixel -v
runtime-vision\Scripts\python.exe experiments/far_pixel_benchmark.py --output experiments\results\far-pixel-new-version
```

The runner refuses to overwrite a frozen protocol. It requires the existing frozen discovery manifest and cached `models/grounding-dino-tiny` safetensors, never downloads weights, and records the pinned model revision. Unit checks cover fractional area integration, unchanged original RGB, known Lab endpoints, flat-field rejection, bounded color-only cues, giant-box rejection, separate cue/detection metrics, empty-box validation and benchmark quarantine.

## Interpretation limits

The 10/20/50/100 m labels come only from the prior illustrative 5 cm cap/camera calculation. This test scales an approximate **whole-body annotation**, not a measured cap. It does not simulate optics, sensor noise, atmospheric loss, motion blur, occlusion or depth accuracy, and it cannot produce real distance success rates. Derivatives from a source share the same background and organism appearance. The scene-island boundary and stretched texture create synthetic artifacts that can trigger color or object proposals. The same source scenes were used in earlier development experiments; this is not a fresh independent test set.

Habitat routing and physical approach must use independently measured, quality-checked range and localization. Synthetic distance labels never become observed range, a map position, or flight authority. Even a real detector hit remains unconfirmed until a close original-RGB photo is reviewed by the human operator.
