# Real video frames for mushroom review

`adapters.video_frames` extracts unlabelled stills from explicitly supplied local
videos. It keeps a mix of relatively soft and sharp frames, so motion/focus blur
is represented instead of automatically discarded. It does not create synthetic
blur, identify a mushroom, estimate meters, train a model, or enable flight.

Requirements: the existing Python vision environment (Pillow and NumPy), plus
FFmpeg and ffprobe on PATH. OpenCV is not needed. Decoding permits local `file`
and `pipe` protocols only; this adapter does not download videos.

Create a sources manifest. Paths are relative to that JSON file, or explicit
absolute local paths. Every source needs its attribution and license page:

```json
{
  "schema": "mycoscout.video-sources.v1",
  "sources": [
    {
      "id": "woodland-video-01",
      "path": "originals/woodland-video-01.webm",
      "title": "Original source title",
      "sourceUrl": "https://example.org/source-page",
      "attribution": "Creator / source title / license",
      "license": "CC-BY-SA-4.0",
      "licenseUrl": "https://creativecommons.org/licenses/by-sa/4.0/",
      "licenseAllowsTraining": true,
      "evaluationOnly": false,
      "sourceGroupId": "original-woodland-recording-01"
    }
  ]
}
```

The URLs above illustrate the fields, not an actual downloadable source.
`licenseAllowsTraining` must reflect a checked source license. It defaults to
false; `evaluationOnly` defaults to true and is forced true without that explicit
license permission. Even with both fields set as above, **each extracted frame
has `trainingAllowed: false`** until separate human review and permission.

Run from the project directory:

```powershell
.\runtime-vision\Scripts\python.exe -m adapters.video_frames --sources data\video-sources-v1\sources.json --output-dir data\video-discovery-v1 --frames-per-video 12 --sample-interval 1 --max-seconds 180
```

The output directory may already hold research or attribution files. Existing
`frames`, `manifest.json`, `collection.json`, `contact-sheet.jpg`, or
`INCOMPLETE.json` cause a refusal; use a new directory for a fresh extraction.
`--ffmpeg` and `--ffprobe` can supply explicit executable paths.

Outputs:

- `manifest.json`: `mycoscout.video-collection.v1` with public source/license
  metadata, original video SHA-256, frame hashes, decoded presentation
  timestamps, dimensions, sharpness measurements and unverified annotation state.
- `frames/*.jpg`: full decoded dimensions, RGB JPEG quality 95, no enhancement,
  resizing or synthetic blur. JPEG output is not pixel-lossless; a separate
  decoded-pixel checksum records the pre-JPEG RGB pixels.
- `contact-sheet.jpg`: thumbnails and timestamp/sharpness captions for review.
- `INCOMPLETE.json` only on a failed extraction. There is then no success
  manifest; partial files must not be used as a completed collection.

Sampling uses video presentation timestamps, including variable frame rate
sources. At most the initial `--max-seconds` of each clip is sampled. Whole-frame
luminance Laplacian variance is divided into low/middle/high strata relative to
that same clip; it can reflect texture, focus, motion or compression and is not
a calibrated blur classifier. Selection spreads timestamps within each stratum.
Exact duplicates are removed globally within a video; near-duplicate suppression
stays within the same sharpness stratum so a softer version is not discarded
just because a sharper view of that scene exists. A nearly static clip can
therefore produce fewer than the requested maximum frames.

Default limits are 12 frames/video, one sampled candidate/second, and 180 seconds
per clip. Hard caps: 12 videos, 120 selected frames total, 60 selected frames/video,
480 candidate frames/video, 3,600 seconds/video, 20 megapixels, 2 GiB/video, 8 GiB
total source files, and 1 GiB temporary candidate/output image limits. Candidate
count is reduced before decoding for high resolution images; the effective
sampling interval is recorded. Each decoder process has a 240-second timeout.
For a longer field walk, `--max-seconds 1800` samples up to the first 30 minutes.
The candidate and output limits stay fixed: longer clips are sampled less
frequently across the requested duration, rather than producing more images.
The one-hour duration cap does not extend the decoder's wall-clock timeout.

`actualDistanceM` is always null: a small mushroom or a wide view is not a measured
range. Source titles are source metadata, not species labels. Empty `labels` and
`annotation.boxes` mean unreviewed, **not an empty/negative scene**. Mark visible
mushrooms with boxes and verify the scene before using it for accuracy scoring.

Every frame shares the source video's `sourceGroupId` and `partitionGroup`.
Without a supplied group, both are `video:` plus the video SHA-256. Supply the
same group for alternate encodings or excerpts of the same original recording.
Future dataset splits must keep entire groups together: neighboring video
frames in training and testing would overstate detection performance. No split,
human label propagation, or automatic model update is performed here.

Validation includes actual FFmpeg decoding of a short test video, timestamp and
RGB checks, hash/source-group provenance, sharp/soft stratification, duplicate
handling, safe defaults, input limits and incomplete-collection behavior:

```powershell
.\runtime-vision\Scripts\python.exe -m unittest adapters.tests.test_video_frames -v
```
