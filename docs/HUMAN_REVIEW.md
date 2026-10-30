# Camera-to-headset review

Version 0.4 adds a shared review inbox at `/review.html`. The field station runs the image model. The phone and Quest display the same captures and recorded decisions. This workflow uses sampled still frames, not a low-latency video stream; an OAK-D camera driver and a physical aircraft have not been connected to this build.

## Use it

1. Start the local station with Node.js 24+ and open **Review captures**. The trained model also needs the `runtime-vision` environment and the official ResNet18 backbone described in [group vision](GROUP_VISION.md).
2. Add a JPEG/PNG photo or use **Phone camera**. Uploaded images are resized for review, and embedded EXIF is not copied by the browser. Camera frames are dated when captured; archival uploads have an unknown capture time.
3. **Ask vision for guesses** returns up to three specific research categories, their broader groups and model scores. An uncertain result remains unresolved. A score compares trained classes and is not a species probability guarantee.
4. Choose a broad group and, optionally, a type within it. Enter a species only when making your own field identification. Record underside, substrate and other evidence; use a consistent specimen ID for multiple views of one find.
5. Enter a reviewer name. **Save my classification**, **Need another view** and **Reject this candidate** record distinct decisions. The image, predictions and earlier review events remain available in the database.
6. Opt into future training only when the photo's rights permit it. Supply the rights basis. The collection does not train on unreviewed model guesses.

The browser retains your reviewer name locally for convenience. It does not verify a professional certificate or digitally sign a commercial inspection. The optional certificate field records the reference as supplied. [Michigan scope](MICHIGAN_SCOPE.md) describes the documented curriculum and its limits.

## Quest

Serve the station over trusted HTTPS and enter the same station URL/token on both devices. Set your reviewer name, select a capture, then choose **Review in Quest VR**. Point a controller ray at a ranked guess; select **Save selected group** to record your decision, **Need another view**, or **Next capture**. **Exit / edit identification** returns to the regular browser for a different type, species name or field notes. The three guesses are clues even when the model abstains.

This is an implemented WebXR controller interface. Desktop rendering and the review API are tested; physical Quest controllers and the Nothing Phone 3a have not been tested here. New captures and predictions arrive through authenticated polling every four seconds. No API token is placed in image URLs.

## Connect a capture producer

The bridge accepts JPEG/PNG candidate frames from an existing camera or detector. A producer should write to a temporary filename and atomically rename the completed image into the watched folder. Keep one session ID throughout a survey so correlated frames stay together during dataset splitting.

```powershell
runtime-vision\Scripts\python.exe -m adapters.review_stream data\candidate-frames --watch --session-id survey-2026-10-26 --source drone_candidate --device-id x500-oak --max-frames 300
```

Use `--station https://YOUR-STATION` and the `MYCOSCOUT_TOKEN` environment variable for a remote station. Upload and classification run sequentially with bounded frame counts and at least one second between attempts. Exact duplicate bytes are skipped during a bridge run. A model failure leaves the capture available for manual review. Restarting the bridge starts a new in-memory duplicate list; the database still deduplicates stored image bytes.

For a saved research image, explicitly use `--source archive_replay --evaluation-only`. This reserves the find for evaluation rather than future training. The bridge never invents depth, geolocation, capture time or flight clearance. It does not itself operate an OAK-D device or an aircraft.

For existing GroundingDINO output, the [detector-crop bridge](DETECTION_REVIEW.md) sends each bounded candidate crop to this same inbox while retaining its parent frame and detector evidence.

## Build a portable reviewed dataset

**Export reviewed training data** downloads a versioned manifest. To include every eligible image, use:

```powershell
runtime-vision\Scripts\python.exe -m adapters.export_review_bundle --output-dir data\reviewed-releases
```

The result contains a manifest and checksum-verified photos under an immutable release directory. Attribution/rights declarations, reviewer, field evidence, source grouping and fixed partitions travel with each record. Unresolved, rejected, unlicensed and conflicting records stay out of training exports. A later correction or permission revocation changes the current export; rebuild future datasets from the current release.

The import and retraining instructions in [group vision](GROUP_VISION.md) merge this bundle into a new frozen dataset. Existing evaluation images must retain their partitions. Coarse-only human labels cannot silently become a specific class label. Keep older checkpoints, compare on held-out data, and deliberately select the model to deploy. No correction silently changes the running model.

See [review database](REVIEW_DATABASE.md) for API fields and backup instructions. The Docker station template provides the database and review service; local model inference additionally requires a compatible Python/PyTorch runtime, the trained head and its backbone. The default Windows runtime path and Linux runtime path are separate.
