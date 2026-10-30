# Vision workbench

Open **http://127.0.0.1:8787/tester.html** on the running MycoScout station. The home, research and review pages also link to **Test vision**. The Android companion contains the same page; connect it to the computer's trusted HTTPS station for inference. A phone's localhost is not the computer's localhost.

## Test a photo

1. Choose an uncropped JPEG/PNG (up to8 MiB and20 megapixels), or select the licensed black-trumpet sample. Uploaded source bytes are preserved. Camera orientation is applied consistently to the view and box coordinates.
2. Choose original RGB, color enhancement, overlapping tiles, or a combination. Four tiles are the default; nine take longer. Tiles include the original full-frame pass. Color changes the inference view, while review crops retain original colors.
3. Set a detector threshold and image scale. A lower threshold permits more candidates and false alarms. Reducing scale shrinks the entire image. It does not simulate a calibrated physical distance.
4. Run the comparison. Processing happens on the station with locally installed GroundingDINO weights. The first test loads the model. One comparison runs at a time; progress and completed results are saved if you leave the page.
5. Select an overlay and inspect candidate crops. The zoom control enlarges the display, without adding source detail. A separate blue color-clue overlay represents contrast anomalies, not detections or species guesses.
6. Draw tight boxes around visible mushrooms, including misses. Select a broad group and optionally enter your field identification. You can also explicitly accept a model box after examining it. **Ask for group guesses** uses the existing local category classifier; its guesses do not become your annotations automatically.
7. Enter a reviewer name and save. Check **I marked every visible mushroom** only after labeling the full image. With zero boxes, that checkbox explicitly declares a mushroom-free image. Leave it unchecked for incomplete or uncertain labeling.

## Interpret the comparison

Matching is class-agnostic, one-to-one, with IoU≥0.5. The table shows candidate count, matched annotated mushrooms, missed annotated mushrooms, unmatched detections, recall and processing time. Complete annotations are required to display misses, false alarms and recall. No annotation means no measured accuracy; partial labels leave those totals blank.

Model confidence is not a calibrated probability of mushroom presence or species identity. A user-supplied distance is recorded as an unverified human measurement; pixel size is never converted into a real range. Group guesses concern the crop and may be wrong even when the detector localized a mushroom correctly.

Tiles can expose more image detail to the detector, but also create duplicate or partial candidates. Overlap and NMS reduce duplicates; the output is capped at100 candidates per method. Color cues are capped at12 and excluded from mushroom metrics. The settings/limitations disclosure retains these bounds, model provenance and timing. Tiled method time includes the reused original pass, so method times are not additive.

The development smoke tested one licensed black-trumpet scene and its existing synthetic6.9-pixel derivative. Original/color/tiles matched4/3/4 of5 annotated bodies in the source; all matched0/5 in the derivative. Tiles added unmatched proposals. This establishes that the new pipeline runs; it demonstrates no distant-detection improvement and is not independent field validation.

## Improve the evidence

Use the same visible mushroom at several measured distances, retain a close confirmation photo, and record lighting, motion blur, occlusion and camera settings. Include difficult negatives such as leaves, dark hollows and wet bark. Compare settings on development scenes, then evaluate the chosen approach on new scenes and photographers. Keep repeated or similar photos of the same scene in one data partition.

Reopen a run from the notebook to compare another setting. Its exact original image and saved annotations are inherited with provenance. Saving a correction creates an immutable annotation revision; stale concurrent saves are rejected. The station stores test photos and results separately from the field-find inbox in `data/testbench/`.

## Export for future detector training

The export button opens a versioned JSON manifest that you can download or copy containing eligible human annotations, image references, provenance, settings and model information. Only complete, explicitly permitted labels are eligible. Evaluation-only samples, conflicting annotations and revoked permission are excluded. Original hashes group exact duplicates and reruns; separate photos of the same scene still require additional grouping.

The image references require access to the same station. `adapters/export_testbench_dataset.py` can materialize an approved manifest into images, class-agnostic mushroom COCO/YOLO boxes and human label metadata. See its `--help` for the local station and output-directory options. It rechecks permission/revisions before marking the collection ready, assigns no automatic data split, and trains no model.

The Android shell does not support blob-download saving. Open the station's workbench in a regular browser to save exported JSON, or use the export adapter on the station. Physical phone/headset behavior remains untested.

No aircraft control, automatic model replacement, or physical camera-distance validation is added by this app.

## Extracted video frames

Open **Try real video frames** to browse the installed `data/video-discovery-v1/manifest.json` collection. Filter by source, follow the timeline, or place lower whole-frame sharpness scores first. Source credits and timestamps remain visible. A low sharpness score may also describe a smooth scene; it does not prove blur around a mushroom, fungal presence, or camera distance.

**Compare this frame** uses the current original/color/tile settings and retains the original extracted image, video hash, timestamp, sharpness measure and source group. Repeated comparisons preserve that provenance. All frames from the same source group receive one suggested data partition; the materialized COCO/YOLO export remains unsplit. Separate clips of the same original video must share a source group during extraction, and overlapping scenes across different videos still need human grouping before evaluation.

Downloaded starter frames remain evaluation-only. Your own eligible video frames still require explicit, complete human annotations and a checked training-permission box before export. Extraction and model predictions never establish labels or training rights. Use **Grade video frames** for the faster human review feed; its grades remain distinct from the Workbench's complete bounding-box annotation and export workflow.

Video origin is stored immutably per test run, so identical extracted pixels at different timestamps or in different videos retain their own observation identity. Database schema v2 preserves older images, reviews and quarantine flags. Earlier preview runs with only photo-level video metadata require a new comparison from the gallery before they can join grading metrics; the migration does not invent a timestamp. Exact pixels found in conflicting source groups quarantine those groups from training export until the grouping is resolved.
