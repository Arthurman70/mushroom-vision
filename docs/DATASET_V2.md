# Public mushroom corpus v2

The public expansion uses the official iNaturalist observation API to select a bounded research batch and downloads the corresponding licensed photographs from the iNaturalist AWS open-data bucket. It does not use Facebook cookies, browser-login scraping, or guessed image URLs. Each URL is derived from an actual API photo record using the documented image-size substitution.

The acquisition is capped at 1,200 accepted photos in this run, with at most one photo per observation and eight photos per observer. Photos and observation metadata must **both** have a CC0, CC BY, or CC BY-SA license at retrieval. Each image keeps its source observation/photo URLs, public IDs, exact attribution string, license code/link, taxon, downloaded dimensions, and file hash. No observation coordinates, place names, observer profile biographies, or observation text are retained.

These are **weak observation-level labels**. “Research grade” is a community taxonomic label for an observation; it does not prove that every photo contains a clearly visible mushroom, that the species is correct, or that all mushrooms in a frame are annotated. Every imported record starts with `annotationStatus: unreviewed` and an empty box list. Comparison fungi are not mushroom-free background images.

## Acquired and frozen on October 26, 2026

The completed run contains **1,174 photos from 1,174 observations and 566 observers**, occupying 660,990,107 bytes of accepted media. It includes **450 black-trumpet photos** and **724 comparison-fungi photos** (324 related fungi and 400 broad Basidiomycota examples). The photo licenses are 898 CC BY, 253 CC0, and 23 CC BY-SA. Thirteen observation API queries supplied the batch. Two duplicate-content records were excluded; only files referenced by the manifest belong to the corpus.

| Frozen split | Black trumpet | Other fungi | Total | Black-trumpet observers | Other-fungi observers |
|---|---:|---:|---:|---:|---:|
| Training | 274 | 419 | 693 | 163 | 216 |
| Validation | 77 | 130 | 207 | 44 | 64 |
| Test | 99 | 175 | 274 | 64 | 90 |

An observer may contribute to both classes within one split, so class observer counts must not be added to estimate unique observers. No cross-observer near-duplicate hash links were found in this run. All 1,174 image byte hashes, dimensions, licenses, and split-group checks pass `data/public-fungi-v2/validation.json`.

The frozen `manifest.json` SHA-256 is `55f39187f168f0c9283b8cdc076694ad345e90ee588387263fbdf1cfb1021755`. The schema is `mycoscout.weak-fungi-dataset.v2`; each `images` row supplies `path`, `sha256`, `weakLabel`, `groupId`, `observerId`, `observationId`, `photoId`, `splitGroupId`, and `split`, plus dimensions, taxon and licensing provenance. The exact class strings are `black_trumpet` and `other_fungi`.

A descriptive query name was corrected before releasing the final manifest for training: taxon 47169 is Basidiomycota, not Agaricomycetes. The acquired media, returned taxon names, labels, and split assignments did not change. `metadataCorrections` records this correction, and `manifest-acquired.json` retains the initial acquisition snapshot. Train only from the final frozen `manifest.json` whose hash is recorded above.

## Taxonomic target

For this experiment, `black_trumpet` is the explicit union of three iNaturalist taxa:

| Taxon | iNaturalist ID | Intended contribution |
|---|---:|---:|
| [Craterellus fallax](https://www.inaturalist.org/taxa/194231-Craterellus-fallax) | 194231 | 250 photos |
| [Craterellus cornucopioides](https://www.inaturalist.org/taxa/48607-Craterellus-cornucopioides) | 48607 | 150 photos |
| [Craterellus calicornucopioides](https://www.inaturalist.org/taxa/473935-Craterellus-calicornucopioides) | 473935 | 50 photos |

The scientific names remain separate in the manifest. This is a visual research target spanning those named taxa, not a claim that they are one species or that every mushroom called a black trumpet belongs to the target. The University of Washington's [Burke Herbarium account](https://www.burkeherbarium.org/imagecollection/taxon.php?Taxon=Craterellus+cornucopioides) distinguishes *C. fallax* from *C. cornucopioides* and describes relevant morphological differences. No edibility classifier is built from these labels.

Related comparison queries include *C. tubaeformis*, *C. ignicolor*, *C. cinereus*, *C. undulatus*, and *Helvella*. The broad comparison query selects Basidiomycota while excluding the queried genera; this phylum also includes rusts and smuts, so not every comparison photograph depicts a familiar cap-and-stem mushroom. This adds related shapes and other fungi without asserting that they are safe, unsafe, edible, or absent from mushroom habitat. Taxon IDs and returned source names, rather than captions or search-result snippets, determine the weak class.

## Access and licensing

iNaturalist separates observation licenses from photograph licenses; an open observation does not automatically license every attached photo. Its [licensing help](https://help.inaturalist.org/en/support/solutions/articles/151000175695) explains these independent settings. The downloader checks the individual selected photo and rejects noncommercial, no-derivatives, missing-license, flagged, hidden, and closed-host records.

The [API recommended practices](https://www.inaturalist.org/pages/api+recommended+practices) recommend approximately one API request per second, about 10,000 per day, and bounded batches. This tool spaces API calls by at least 1.1 seconds, requests up to 200 observations per page, identifies itself with a custom User-Agent, uses a maximum of three concurrent image downloads for this run, backs off on throttling, and caps downloaded media at approximately 1 GB. The API request budget is 200, far below the published daily guidance. It does not rotate IP addresses or evade access restrictions.

The [official API schema](https://github.com/inaturalist/iNaturalistAPI/blob/main/lib/views/swagger_v1.yml.ejs) documents the `large` image variant, up to 1,024 pixels in its longest dimension. The downloader uses that variant from `inaturalist-open-data.s3.amazonaws.com`; it does not request every original photograph. For significantly larger future corpora, iNaturalist's [research download guidance](https://help.inaturalist.org/en/support/solutions/articles/151000223300-how-i-download-inaturalist-photos-to-use-for-research-) recommends its [Licensed Observation Images dataset](https://registry.opendata.aws/inaturalist-open-data/) and [monthly metadata exports](https://github.com/inaturalist/inaturalist-open-data).

License attribution travels with each photo in the manifest. Preserve it when displaying or redistributing the photo, link the recorded license, and indicate any later modifications. CC BY-SA adaptations carry their applicable license conditions. The corpus has no single blanket photo license; each photograph retains its own. License codes link to the current Creative Commons 4.0 pages, or CC0 1.0, with that mapping stated explicitly in each record. Source attribution strings are retained verbatim.

## Split and duplicate handling

The downloader drops exact file duplicates and normalized-pixel duplicates from the manifest. It then creates a fixed split from observer identity, with a predetermined hash seed and intended proportions of 60% training, 20% validation, and 20% test. There is no test-performance-based choice of seed. All observations from one observer remain together.

Before assigning the split, a conservative 64-bit image difference hash connects photos from different observers when their Hamming distance is at most two; those observer groups then stay in the same split. This can over-group similar images, and it cannot catch every duplicate or common location. The exact links are retained in `splitPolicy.nearDuplicateLinks`. Split class/observer counts must be checked before training. No synthetic variants are created before this freeze.

`manifest.json` is the immutable training input. `manifest-sha256.json` records its hash. `manifest-progress.json` is only an intermediate download inventory and must never be used to select model hyperparameters or report final model metrics. A feature-throughput benchmark on an intermediate inventory is permitted because it does not fit a classifier or inspect held-out performance.

## Run and validate

From the MycoScout project directory:

```text
python -m vision.acquire_public --output-dir data/public-fungi-v2 --max-images 1200 --workers 3 --per-observer 8
python -m vision.acquisition.validate --manifest data/public-fungi-v2/manifest.json --output data/public-fungi-v2/validation.json
python -m unittest vision.acquisition.test_acquire -v
```

The downloader refuses to replace an existing frozen manifest. Use a new destination for a later expansion. A failed partial acquisition can reuse already downloaded files, but its final manifest is not released to training until the complete grouping pass finishes.

To request up to 1,200 **additional** photos while retaining this experiment unchanged:

```text
python -m vision.acquire_public --resume-from data/public-fungi-v2/manifest.json --more 1200 --output-dir data/public-fungi-v3 --workers 3 --per-observer 8
python -m vision.acquisition.validate --manifest data/public-fungi-v3/manifest.json --output data/public-fungi-v3/validation.json
```

This expansion command is implemented and tested with local fixtures; a second network acquisition has not been run. It validates the parent freeze hash and image bytes, copies inherited images into a separate portable directory, skips already-seen observation/photo/content IDs, and requests a bounded additional batch. The old manifest and old images are never rewritten or hard-linked. Allow space for another copy of the existing 661 MB plus the new photos. The one-photo-per-observation and eight-photos-per-observer caps apply across the combined corpus. Scarce licenses, duplicate content, observer caps, and ten pages per query can leave the acquired count below `--more`.

Existing observers retain their original split. New observers use the original deterministic seed; any new near-duplicate component connected to an existing observer inherits that observer's split. If new links bridge two previously frozen splits, the tool refuses to freeze the expanded corpus and leaves it for review. It does not move an old test observer into training. The new manifest records its parent's SHA-256 and the inherited count. An inherited test set has already been used in the earlier experiment and must not be described as a newly untouched benchmark.

`--resume-from` means expand a **completed frozen corpus** into a new directory. To retry an interrupted, not-yet-frozen expansion, rerun the same command and destination; compatible downloaded files are reused, the API inventory is queried again, and no partial inventory is treated as a finished split. Frozen output directories always require a new destination.

The validator checks image bytes and dimensions, independent photo/observation licenses, attribution, empty detection labels, the explicit class map, and leakage across observer, observation, photo, exact-file, normalized-pixel, and connected split groups. It does not certify taxonomic identity or terrain safety.

## What the corpus can establish

It can support a stronger **whole-image or candidate-crop target classifier** and a measured comparison against the earlier handmade-feature baseline. It cannot by itself establish far-away discovery accuracy, a physical camera range, fungus bounding-box recall, fungus-free terrain, or safe flight clearance. Those require independently annotated context frames and genuine empty woodland examples. The visually selected `annotation-selection-20.json` provides 20 in-situ candidates for independent box review, not ready-made ground truth.

The separate 12-photo, 30-box localization audit in `data/black-trumpet-discovery-v2` is a selected **development audit of fixed pretrained detectors**, not the held-out classifier test set. Its audit manifest uses `split: test` for the chosen evaluation subset, while the source corpus assignments remain seven training photos, three validation photos, and two test photos. `audit-provenance.json` records the original split for every photo and hashes both manifests without rewriting the frozen audit. Its all-positive composition cannot measure mushroom-free false-alarm rates.

Recent-observation ordering, open-license filtering, global geography, seasonal sampling, and photographer habits create selection bias. Some images are harvested specimens or close details. The observation taxon can be correct while an individual photograph mainly shows habitat. A good result on this corpus therefore does not justify autonomous movement toward a dark patch or confirm a mushroom species.
