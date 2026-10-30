# Mushroom vision experiments

The `vision` package adds a local, reproducible data and evaluation workflow. It can ingest authorized photos, retain photo/post provenance, validate labels, split by independent source groups, generate controlled scale/blur trials, score an existing detector, and train a small candidate-crop verifier. It does not identify edible species or establish autonomous flight readiness.

## Run the tools

Use Python 3.10 or newer with `vision/requirements.txt`. The desktop bundled Python already includes Pillow and NumPy on this development machine. Run these commands from the `mycoscout` directory, substituting actual local filenames:

```text
python -m vision.pipeline ingest --manifest vision/data/michigan/manifest.json --image PHOTO.jpg --id fb-photo-001 --source-url FACEBOOK_PHOTO_URL --group-id FACEBOOK_POST_ID --context closeup
python -m vision.pipeline validate --manifest vision/data/michigan/manifest.json
python -m vision.pipeline split --manifest vision/data/michigan/manifest.json --output vision/data/michigan/manifest-split.json
python -m vision.pipeline curriculum --manifest vision/data/michigan/manifest-split.json --output-dir vision/data/michigan/stress-01
python -m vision.pipeline evaluate --manifest vision/data/michigan/manifest-split.json --predictions vision/runs/run-01 --output vision/runs/run-01/evaluation.json --split test
python -m vision.pipeline train-verifier --manifest vision/data/michigan/manifest-split.json --checkpoint vision/models/verifier-01.json --report vision/runs/verifier-01.json
python -m unittest discover -s vision/tests -v
```

The source URL and group ID are required. For Facebook, record the canonical photo URL and the parent post ID, not a short-lived media CDN URL. All photos, crops, and resized variants from the same post must share a group ID. Multiple views or a burst from the same mushroom encounter should also share a group ID even when posted separately. Ingest copies one local file; this tool does not scrape Facebook, upload photos, publish group content, or grant redistribution rights. Keep group-derived research images out of public release bundles.

## Annotation format

The manifest schema is `mycoscout.vision-dataset.v1`, with an `images` array. Each record contains:

```json
{
  "id": "fb-photo-001",
  "path": "images/photo-001.jpg",
  "sha256": "64 lowercase hexadecimal characters computed from the exact file",
  "originalSha256": "same hash for an original; retained original hash for derived variants",
  "groupId": "same-facebook-post-or-capture-encounter",
  "sourceUrl": "canonical original photo URL",
  "sourceNote": "User-authorized local research; not licensed for redistribution",
  "width": 1280,
  "height": 960,
  "context": "closeup",
  "synthetic": false,
  "distanceM": null,
  "annotationStatus": "complete",
  "reviewer": "record the person or method that reviewed these boxes",
  "annotations": [
    { "label": "mushroom", "bboxPixelsXYXY": [40, 110, 300, 420] },
    { "label": "background", "bboxPixelsXYXY": [700, 200, 950, 400] }
  ]
}
```

Coordinates refer to the EXIF-oriented image, x right and y down, with an exclusive bottom/right boundary. Annotate a tight axis-aligned box around each visible mushroom. State the annotation method honestly: agent visual review, human review, or imported expert annotation are different levels of evidence. Detector predictions must never be silently relabeled as ground truth.

`complete` means the entire image has been checked and **all visible target mushrooms** have boxes. An image with zero mushroom boxes and complete review is a negative image. `unreviewed` means ground truth is unavailable; candidate counts may be shown but precision/recall remain unavailable. Explicit `background` boxes label mushroom-free candidate-like crops for verifier training, such as leaves, stones, bark, and pale wood. They do not create additional detector ground truth.

`partial` permits explicitly reviewed positive/background crops from an image whose entire fungal colony or scene cannot be exhaustively annotated. The verifier can use these reviewed crops. Detector evaluation excludes the whole image from precision/recall and reports it in `skipped.partial`; unboxed mushrooms in these frames are never counted as false alarms. A patch covering a dense fungal cluster is a valid positive classifier crop but is not automatically an individual-object detector label.

`context` describes the original capture: `closeup`, `context`, `far`, or `unknown`. Distance is null unless measured. A closeup shrunk to 100 pixels remains a synthetic transformation of a closeup, not a real distant scene. A small object in the frame is not proof of a particular physical camera distance.

Curriculum generation defaults to `--mode fixed-canvas`: it shrinks the entire scene, centers it on a neutral gray canvas with the original frame dimensions, applies blur/JPEG, and transforms boxes with both scale and offset. The detector therefore sees smaller objects relative to a fixed input frame. The artificial padding does not reproduce real habitat clutter. The separate `--mode downsample` reduces frame dimensions and tests lost image detail; a detector's normal resize may restore the original relative target size. Do not interpret downsample-only runs as a small-object scale test.

Validation checks image SHA-256, oriented dimensions, finite boxes, provenance, and split leakage. `split` links records sharing a post/group, original photo hash, or identical file hash into connected components, then deterministically assigns whole components to train/validation/test. At least three independent groups are needed. Allocation is by group, not individual crop. Inspect class balance after splitting; do not move difficult test examples into training to improve the reported result.

## Evaluate the detector

Run the existing `adapters/detect_fungi.py` against each image, or use a batch inference runner. Put outputs from one frozen model/prompt/threshold configuration into a separate run directory. The evaluator reads the adapter's `mycoscout.unlocated-candidates.v1` files and matches images by SHA-256. It also accepts a consolidated file:

```json
{
  "schema": "mycoscout.vision-predictions.v1",
  "detector": { "model": "record exact model", "revision": "record exact revision", "prompt": "a mushroom." },
  "images": [
    {
      "imageId": "fb-photo-001",
      "status": "ok",
      "candidates": [ { "confidence": 0.83, "bboxPixelsXYXY": [40, 110, 300, 420] } ]
    }
  ]
}
```

One-to-one confidence-ordered box matching uses IoU 0.5 by default. Duplicate detections count as false positives. Missed annotated mushrooms count as false negatives. Negative images measure false alarms. Missing or failed inference is reported separately and never treated as a successful empty result. Reports include image counts, actual and synthetic results separately, context breakdowns, and recall by the square root of ground-truth box area in pixels. Pixel buckets are under 16, 16–31, 32–63, and at least 64 pixels. The pixel-bucket summary combines selected records; use the real and synthetic manifests as separate evaluations for interpretable domain-specific size results.

GroundingDINO can return overlapping boxes for the words “mushroom” and “fungus.” Raw evaluation preserves them and penalizes duplicates. An explicitly separate run with `--nms-iou 0.5` applies class-agnostic non-maximum suppression before matching; the report records that choice and retains both raw and filtered candidate counts. NMS can also remove distinct overlapping mushrooms, so select its setting on validation examples and preserve the raw results.

Tune prompts, confidence thresholds, and tiling decisions on validation groups only. Freeze them before evaluating test groups. A report with a few social-media photos is a small pilot experiment, not a general accuracy estimate. Social-media positives systematically underrepresent empty forest floor, occlusion, motion blur, wet lenses, dappled sunlight, and the species or seasons absent from the sample.

## Train the candidate verifier

`train-verifier` learns a class-balanced logistic model on RGB/HSV histograms, coarse spatial appearance, and image gradients. It receives candidate crops; it does **not** discover boxes in an image. Its useful role is testing whether an inexpensive second stage can reject some detector false positives before an onboard inspection maneuver is proposed.

The tool requires at least six independent labeled original-photo groups in training and two each in validation and test, with mushroom and explicit background crops in every split. Training needs at least three of each crop class. These minimums permit an experiment; they do not certify useful generalization. The feature normalization is fitted on training data alone. The decision threshold is selected on validation F1 and then applied unchanged to test groups. Synthetic variants are excluded. The checkpoint is plain JSON, stores the exact manifest digest and optimizer settings, and uses no executable/pickle serialization.

The included full optimizer test trains only on generated geometric fixtures inside a temporary test directory. That demonstrates pipeline operation. It is not training on mushroom photographs, and its scores must not appear as mushroom recognition performance. Real-photo training reports should only be generated after sufficient actual labels exist.

## Close-to-far progression and flight implications

1. Review closeup detections and common false alarms. Establish labels for caps, clusters, shelf fungi, partial occlusions, and non-mushroom lookalikes.
2. Add real habitat/context frames with mushrooms still clearly visible to a reviewer. Include fully reviewed empty frames from the same woodland.
3. Run the separate synthetic scale/blur/JPEG set. Look for the pixel-size region where recall drops. It guides the next camera experiment, not a flight range guarantee.
4. Capture real stationary frames at measured ranges and viewing angles using the intended camera, lens, exposure, and resolution. Keep all frames from one encounter in one split.
5. Add synchronized slow-motion sequences, repeated views, illumination changes, and obstacle occlusion. Evaluate whether candidate observations persist across viewpoints, and measure false approaches per survey minute.
6. Feed the validated pixel requirement, camera calibration, measured depth uncertainty, and exposure budget into the inspection planner. If the expected mushroom size is too small in the frame, the planner may propose a closer view only when independent collision sensing certifies clearance. Missing detections never imply empty terrain or clearance.

Flight policy should prefer pause-and-observe passes at a conservative standoff, small viewpoint changes, and retreat on lost depth/pose confidence. A narrow depth camera does not validate rotor clearance in every direction. A detector score cannot establish a safe path through branches, determine ground traversability, or authorize descending toward a mushroom. Those remain independent planning and sensing requirements.
