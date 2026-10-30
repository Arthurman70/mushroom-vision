# Mushroom-group vision and review

The group model provides up to three photo/crop guesses for a person to inspect. It has 22 internal research classes, mapped to the ten broad categories in `config/michigan-mushrooms-v1.json`. Multiple guesses can share a broad category; their scores are never silently added. Classification does not discover fungi in a wide drone frame, establish a physical detection distance, certify a species, determine edibility, or authorize sale or flight.

## Measured first release

`groups-v1-fea75afafabb` was actually trained on 26 October 2026 from 3,250 licensed photos by 1,453 iNaturalist observers: 1,906 train, 622 validation and 722 test photos. The expansion added 2,076 images to 1,174 immutable V2 originals. Dataset SHA-256: `fea75afafabbb21f80075286769b84aafd5e556feb28abf60813de5075e69bcf`. The model emits 20 named research classes plus other/unsupported fungi. Michigan truffle is unsupported: the query yielded one licensed photo, assigned to test, and no training or validation examples.

| Frozen evaluation subset | Photos | Top guess correct | Correct class in top three | Non-abstained guesses | Correct among non-abstained |
|---|---:|---:|---:|---:|---:|
| All test photos | 722 | 68.7% | 87.8% | 512 (70.9%) | 75.8% (124 errors) |
| Newly acquired V3 test | 448 | 74.6% | 91.1% | 348 (77.7%) | 84.5% (54 errors) |
| Reused V2 test | 274 | 59.1% | 82.5% | 164 (59.9%) | 57.3% (70 errors) |

The validation-selected policy is score ≥0.45 and top-two margin ≥0.10, with an automatic abstention for the other-fungi top class. It achieved 80.0% on accepted validation examples but **failed to reach its 80% target on the overall test set**. The policy was not retuned on test. Other-fungi recall was only 38.4%, showing that the fallback misses many unsupported fungi; it is not an open-world safety detector. Morel, chanterelle, black-trumpet and bolete test precision/recall were respectively 87.5%/87.5%, 53.3%/64.9%, 79.1%/89.2%, and 70.7%/69.0%. These are weak-label photo-classification results, not distant drone detection or species guarantees.

The selected head regularization was 1.0 and temperature 0.6830. Ten-bin validation calibration error fell from 0.1568 to 0.0413; test calibration error was 0.0325. Calibration error is an aggregate score statistic and does not cancel the observed classification failures. Training plus feature extraction took 113.6 seconds locally. A real worker smoke run processed four validation images, rejected two malformed inputs and recovered, completing in 4.3 seconds including model startup. Warm classifier calls took about 20–27 ms on this CPU; image transfer/decode, drone networking and headset rendering are additional costs. Fifteen regression tests cover input validation, holdout preservation, duplicate reconciliation, parent-frame crop grouping and reviewed-data import.

## Data and reproducibility

`vision/acquire_groups.py` uses the official public iNaturalist API and its public open-data image host. It resolves scientific names with fungal kingdom and taxonomic-rank checks before selecting records, and fetches ancestry for all inherited V2 taxa before remapping classes. Each image retains a source link, observation/photo/observer IDs, image hashes, taxon evidence, and photo attribution/license. Only research-grade observations with CC0, CC BY, or CC BY-SA observation and photo licenses are accepted. Neither observation coordinates nor observation text are retained.

The separate `data/public-fungi-v3` directory contains copied V2 media and newly downloaded media. The frozen V2 source remains unchanged. New acquisitions use three media workers, API requests at least 1.1 seconds apart, at most five pages per taxon query, a per-observer cap, and a download budget below 2 GB including the small concurrency allowance. A single photo per observation is retained. Exact byte/normalized-pixel duplicates are excluded. Near-duplicate difference hashes connect photographer groups before splitting; conflicts with inherited partitions stop freezing or quarantine the new image. Fixed inherited photographer partitions are preserved. V2 test images are reused evaluation, explicitly reported separately from newly acquired V3 test images.

The dataset freezes `manifest.json` and its SHA-256 before feature extraction, fitting, or model selection. Original labels are weak observation taxonomy; they do not verify photo contents or growth form. Recent-observation, public-license and photographer selection introduce biases. Broader genera/families in the research classes must not be substituted for the narrower named species and exclusions in the Michigan curriculum catalog. In particular, an `aborted_entoloma` class label does **not** establish the aborted form, and broad Ganoderma, Pleurotus, Morchella and other genus coverage does not imply support for every named species. Boletes are a research category outside the confirmed 20-guide curriculum. No mushroom-free forest-background corpus is included.

Run acquisition in a new, unfrozen output directory:

```powershell
runtime-vision\Scripts\python.exe -m vision.acquire_groups --output-dir data/public-fungi-v3 --parent data/public-fungi-v2/manifest.json
```

## Actual model training

The local backbone is the official torchvision ResNet18 ImageNet model, with fixed SHA-256 `f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec`. The backbone is frozen. Input preprocessing is the official resize-to-256-shorter-side, 224-pixel central crop and ImageNet normalization; off-center evidence can be lost. The model reads local verified weights, using `torch.load(..., weights_only=True)`; inference does not download files or upload images.

Training fits a class-balanced linear softmax head on 512-dimensional features. Feature standardization uses training data only. The regularization choices 0.01, 0.1 and 1.0 are compared using validation macro-F1, with validation negative log likelihood breaking ties. A validation-only grid fits a temperature, then selects a score/margin policy with at least 20 accepted validation examples and a target 80% empirical accepted accuracy, maximizing accepted coverage. If no policy qualifies, it abstains on every image. A top guess of other/unsupported fungi always abstains. The same validation set supports selection, calibration and policy fitting; its results are optimistic estimates, and the held-out test report is the relevant check. Calibration is empirical, never a guarantee of correctness.

A class can be emitted only with at least 25 training photos from ten observers and five validation photos from three observers. Support decisions never consult test performance or test counts. Unsupported training labels map to the other-fungi class; evaluation retains their original labels so failures remain visible. Both class-level and coarse-category metrics are reported, including every class's support, precision/recall/F1, confusion matrix, top-one/top-three accuracy, calibration bins, review coverage and accepted errors. Counts and results for absent or underrepresented curriculum species belong in the separate coverage report.

```powershell
runtime-vision\Scripts\python.exe -m vision.multiclass train --output experiments/results/groups-reproduction-01 --checkpoint vision/checkpoints/groups-reproduction-01.json
runtime-vision\Scripts\python.exe -m unittest experiments.test_groups -v
```

Controlled research options are `--min-train-photos` (10–500, default 25), `--min-validation-photos` (3–200, default 5), and `--target-validation-accuracy` (0.5–1.0, default 0.8). The measured release used these defaults. They require a new training run and are recorded in the checkpoint; raising the target may reduce coverage or cause complete abstention. Existing checkpoints and completed experiment folders are protected from overwrite: choose new version paths. For reviewed local corpora, support counts separate public observers from local source/session groups. Local groups are not proof of independent photographers. Repeatedly selecting models from the same test results would weaken the benchmark; use a fresh independent field test for promotion decisions.

The fitted head is `vision/checkpoints/groups-v1.json`; its weights, feature means/scales, temperature, support metadata, selection policy and provenance travel together. Full experiment output is `experiments/results/groups-v1/study.json`, with per-image validation/test predictions in the same directory. Private images, full data directories, local dependency runtime and backbone weights stay outside the deployment source archive. The small head checkpoint is included. Install/download the pinned runtime and official backbone separately on the local machine.

## Local worker contract

```powershell
runtime-vision\Scripts\python.exe -m vision.multiclass worker --checkpoint vision/checkpoints/groups-v1.json
runtime-vision\Scripts\python.exe -m vision.multiclass predict path\to\photo.jpg
```

The worker accepts one JSON object per newline on standard input:

```json
{"id":"request-17","imageBase64":"...","mimeType":"image/jpeg","candidateId":"find-17","captureTimestamp":"2026-10-26T12:00:00Z"}
```

Only image bytes, an optional MIME assertion, and the request ID affect classification. Candidate IDs and timestamps are backend provenance, not image evidence. Images must be JPEG/PNG/WebP, at most 8 MiB and 20 megapixels; malformed base64, MIME mismatches, and oversized requests are rejected. Loading is local and startup readiness is logged to stderr, leaving stdout exclusively for response objects. No shell or path is accepted from a request. Each result also carries `reviewPolicy` with its score threshold, margin threshold and validation target status.

Successful response:

```json
{"id":"request-17","status":"ok","result":{"schema":"mycoscout.group-guess.v1","evidenceVersion":"groups-v1-...","guesses":[{"classId":"hericium","groupId":"tooth_fungi","label":"Hericium tooth-fungus group","modelScore":0.72}],"abstained":true,"abstentionReasons":["insufficient_model_score"],"suggestedGroupId":"unknown","scoreCalibration":{"method":"validation-temperature","empirical":true,"guaranteed":false},"humanVerificationRequired":true,"speciesConfirmed":false,"edibility":"not assessed","salesApproval":false,"flightGateEnabled":false}}
```

Scores express the fitted model's relative class evidence within this limited vocabulary. Even a high score can be wrong on a drone frame, a lookalike or an unfamiliar background. A non-abstained result is still a guess awaiting human review. Error responses have `{id,status:"error",error:{code,message}}`.

## Human labels and future fitting

AI guesses must never become labels automatically. A reviewed export may provide a specific class only when a reviewer explicitly supplied it; broad-category-only reviews cannot be silently mapped to one of 22 classes. Training permission and media rights are separate required attestations. `vision/import_reviews.py` reconciles byte hashes, normalized pixel hashes, perceptual near duplicates and source/session/specimen groups against the frozen corpus. Known photos and re-encodings are excluded as duplicate examples, and their frozen partition applies to other photos in the same source component. Candidate crops also carry `provenance.sourceMetadata.parentImageSha256`: a crop inherits its known parent's held-out partition even when its own image hash and appearance differ. Sibling crops with an unknown shared parent remain grouped together. Parent bytes are identified by SHA-256; pixel boxes remain provenance rather than labels. Conflicting prior partitions, conflicting near-duplicate labels, held-out downgrades or evaluation-only conflicts quarantine the component. No frozen labels are overwritten. Exact content and MIME are checked against the export, and photo paths cannot escape its local `photos` directory.

Use a new output version and rebuild from the original frozen corpus plus the latest complete review export each time. The importer refuses an already-imported base so revoked permission or an earlier incorrect review is not silently retained. Photo permissions and reviewer credentials are self-attested. Matching can miss altered copies or conservatively quarantine genuinely different but similar-looking images; a curator must review these cases. Only the `train` partition fits model weights; validation selects and calibrates, and test stays held out. The original V3 benchmark and deployed checkpoint remain unchanged by the importer.

```powershell
runtime-vision\Scripts\python.exe -m vision.import_reviews --base data/public-fungi-v3/manifest.json --export data/finds/exports/RELEASE.json --image-root data/finds --output data/public-fungi-v3-reviewed-01
runtime-vision\Scripts\python.exe -m vision.multiclass train --manifest data/public-fungi-v3-reviewed-01/manifest.json --output experiments/results/groups-reviewed-01 --checkpoint vision/checkpoints/groups-reviewed-01.json
```

For a downloaded portable review bundle, pass the bundle directory as `--image-root` and its manifest as `--export`. Inspect `review-import.json` and the new test metrics before explicitly selecting a new checkpoint. This creates a complete manual review/import/train/evaluate loop; the live application does not retrain or promote models automatically.

Primary model documentation: [torchvision ResNet18](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html). Curriculum context: [MAMI study guides](https://midwestmycology.org/mushroom-study-guides/). Photo provenance is retained per image in the frozen local manifest; showing or redistributing photos must retain the applicable attribution and license.
