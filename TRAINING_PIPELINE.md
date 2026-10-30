# Reviewed mushroom detector training

The grading feed now connects to an actual detector-training pipeline. Saving a
grade still does **not** automatically train or replace a model.

## Use the review workflow

1. In the human grading feed, draw a box around every mushroom you can actually
   see. Mark the image completely labeled only when all visible mushrooms have
   boxes. A confirmed empty frame is useful negative evidence. Uncertain or
   partially labeled frames are excluded.
2. On a source eligible for reuse, enable **Allow these reviewed labels to train
   future detectors** and save. Evaluation-only sources have that control
   disabled. Their complete reviews can evaluate a model but never train it.
3. In the live dashboard's learning panel, export reviewed data. Review the
   training, validation and test counts. The workbench's complete, permitted
   annotations join the same export.
4. Start a training job when all three partitions contain sufficient independent
   data. Review its heldout precision, recall, false positives, and original-pixel
   width results. Changing the score threshold changes the operating point;
   confidence is not a calibrated probability of correctness.
5. Explicitly activate a completed candidate for **experimental** live capture.
   This is an operator choice, not a claim that it beats the existing detector.
   Older completed candidates remain available for rollback through the same
   activation endpoint, subject to source-permission and checkpoint checks.

Current readiness audit (`experiments/results/reviewed-training-readiness.json`):
one eligible workbench image with five boxes, no completely labeled video
frames. The deterministic grouped split put that image in test. There are zero
training images and zero validation images, so a real training job was refused.
User labels and deployed models were not changed. The twelve source-video
frames remain incomplete for training/evaluation even when a quick visibility
grade has been saved.

## What is trained

`vision/training_pipeline.py` fine-tunes torchvision's Faster R-CNN MobileNetV3
Large 320 FPN detector, initialized from the fixed official COCO checkpoint. Its
classification head is replaced with background/mushroom classes; the backbone
is frozen while its feature pyramid, proposal network and prediction heads learn.
This is mushroom localization, not species identification. Existing group/species
guesses remain a separate classifier and human review step.

Official initialization is installed in `models/reviewed-detector/` and checked
against SHA-256
`907ea3f91ff92242bc1baea8049276a3e76bca48ce7560bd268cc029f37977b5`.
The 77,844,807-byte checkpoint successfully loaded into the local Torch 2.6.0 /
torchvision 0.21.0 runtime. No image data is uploaded for training.

The torchvision implementation uses the BSD-3-Clause code license. Pretrained
weights can have additional terms inherited from their source datasets; the
model card retains the official source and terms link instead of claiming that
the code license clears all training-image rights.

- [Official model description](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html)
- [Official pretrained model terms](https://docs.pytorch.org/vision/stable/models.html)
- [Exact official checkpoint](https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth)

Training runs use 1–30 epochs, a bounded learning rate and fixed score threshold,
SGD, mild brightness/contrast/saturation changes and horizontal flips. They do
not fabricate new mushrooms or relabel model predictions as human truth.
Validation chooses the checkpoint; test is evaluated only after that selection.
Each run saves a checkpoint, model card, validation history, and test predictions.
Checkpoint loading uses `weights_only=True`; arbitrary pickle models are not
accepted by the API.

## Dataset separation and integrity

Exports are immutable directories with a checksummed manifest and a final
`READY.json` marker. Partial exports cannot train. The source image bytes,
oriented decoded pixels, human review identities/revisions, source credits,
video hashes, timestamps and manual track IDs are retained. Images are decoded
to RGB PNG with metadata removed; label coordinates stay in original pixels.

Whole videos and manually linked sessions share a partition. Exact image hashes
and decoded-pixel duplicates are joined before splitting. A conservative image
hash/color heuristic also groups obvious near duplicates; it cannot prove that
different camera angles depict independent sites. Link related recordings and
photos with the same session name before export. The API accepts
`groupAssignments`, mapping record IDs from learning status to session names.

Conflicting annotations on identical decoded pixels are excluded. Evaluation
images are reserved even if their labels are incomplete; copying one into an
otherwise eligible workbench upload does not release it for training. Once a
source/session group has been used for validation or test, later exports cannot
move it back into training. A session joined to a test source becomes test-only.

Complete labels and current permissions are rechecked before export, before
training and before activation. Revocation or revision changes require a new
export. These are local, point-in-time rights assertions, not a remote rights
verification service.

Minimums are 20/5/5 complete train/validation/test images, including at least
10/2/2 positive images, with separate source groups. They only reject empty or
trivial runs; a sufficiently diverse woodland dataset will need much more.

## Evaluation and deployment

At the selected score threshold, a prediction matches a human box only when
intersection-over-union is at least 0.5. Matching is one-to-one with maximum
cardinality. Repeated predictions cannot all claim one mushroom. Reports include
precision, recall, F1, false positives per image, source-group counts, and recall
at widths `<4`, `4–7`, `8–15`, `16–31`, `32–63`, and `64+` original pixels.
An empty denominator is unknown (`null`), never perfect accuracy. Pixel width is
not a measured flight distance.

Minimal activation eligibility requires at least one real matched mushroom in
both validation and test. **This deliberately modest gate does not establish
superiority, reliable distant detection, or operational flight readiness.**
The dashboard requires explicit operator activation; the model card marks the
candidate experimental and operationally unvalidated. Compare its results with
the original detector on the same heldout frames before treating it as improved.

The station writes `data/learning/active-model.json`, containing the selected
checkpoint, checksum, model card, reviewer, activation time and previous model.
Live capture can load that selection. A running capture process must reload or
restart its selected detector to pick up a new activation; activation does not
alter a flight controller or arm a drone.

The reusable inference interface is:

```python
from vision.training_pipeline import load_detector, tensor_image, predictions_for
model, metadata = load_detector(checkpoint_path, device="cpu")
boxes = predictions_for(model, tensor_image(image_path), metadata["scoreThreshold"])
```

Each box has `bbox` in original XYXY pixels and `score`. The CLI equivalent is
`python -m vision.training_pipeline predict --checkpoint PATH --image PATH`.
Only trusted local operator tools use these paths; HTTP callers supply bounded
IDs and settings, never executable commands or arbitrary checkpoint paths.

## Station interface

- `GET /api/learning/status`: eligible counts, independent split readiness,
  stale-dataset notice, jobs and active selection.
- `POST /api/learning/export`: optional `groupAssignments`; returns an immutable
  `datasetId` and readiness report.
- `POST /api/learning/jobs`: `datasetId`, optional `epochs`, `learningRate`,
  `scoreThreshold`; starts one bounded asynchronous job.
- `GET /api/learning/jobs/:id`: current progress or completed model card.
- `POST /api/learning/jobs/:id/cancel`: stops the local child process.
- `POST /api/learning/activate`: completed `jobId` and `reviewer`.

The service uses fixed Python modules without a shell. One export or training
job runs at a time. Jobs have a time limit, output-size limits, cancellation,
persisted status and interrupted-job recovery. Official initialization downloads
only from its fixed PyTorch URL if it has not already been installed.

## Verification

Run `node --test tests/training-pipeline.test.mjs` and
`python -m unittest vision.tests.test_training_pipeline -v` with the local vision
environment. They cover permission changes, evaluation duplicate quarantine,
source grouping, immutable content checks, original-pixel metrics, concurrency,
cancellation and the real testbench store interface.

`experiments/results/reviewed-detector-toolchain-smoke-v2.json` records one
**synthetic software smoke test**: a finite gradient changed detector weights,
the checkpoint reloaded, and inference completed. Random pixels and a fabricated
box were used only to test the software path. They were not saved as a usable
model, added to training data, or reported as mushroom recognition accuracy.
