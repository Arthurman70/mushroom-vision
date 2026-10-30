# Mushroom Vision data sources

Mushroom Vision is an unfinished open-source research project. Its broader recognition catalog and training code are experimental. Data acquisition adds examples and possible labels; it does not verify those labels, demonstrate a detection range, or establish a medical or food-safety application.

The machine-readable registry is `config/dataset-sources.json`. Local media and contributor records stay under the ignored `data/` tree; publishing the software does not republish or relicense those images.

## Downloaded collections

The new mushroom collection is `data/biodiversity-fungi-v1/manifest.json`. It extends the 3,250-image V3 corpus with **672 newly downloaded photographs (398,664,149 bytes)** and 21 research genus groups, including Amanita, Russula, Galerina, Inocybe, Gyromitra and Verpa. There are now **3,922 images and 43 research classes**, with 501 distinct source taxon IDs represented. Those taxon IDs are source labels, not 501 trained or verified species classifiers. The new photos represent 199 source taxa and 431 photographer accounts, with 464 CC BY, 183 CC0 and 25 CC BY-SA photo licenses.

The combined frozen split is 2,319 train / 748 validation / 855 test images. Every stored image passed its checksum and partition checks; all inherited partitions remain intact. Some new groups still lack enough independent validation examples: Galerina has two and Inocybe one. An expanded catalog does not make these groups validated model outputs. Final counts and checksums are recorded in the collection's `summary.json` and `validation.json`; `progress.json` is explicitly unfinished and must not be trained from.

The separate wildlife sample, `data/wildlife-context-v1/manifest.json`, contains **113 images, 45,751,020 image bytes**, across 21 source categories and 56 camera locations. Its original camera split supplies 78 training and 35 validation images. The metadata archives add 9,017,301 bytes. These are useful for reviewing terrain/context distractions, animals and possible mushroom false positives. An animal dataset's “empty” label means **no labeled animal**, not “no mushroom.” Every frame starts with unknown mushroom visibility and is excluded from mushroom-negative detector training. Human review must decide that label.

Together these runs downloaded **785 new images / 444,415,169 image bytes**, plus the wildlife metadata. Previously downloaded V3 images were copied locally into the new frozen mushroom collection, not downloaded again. `scripts/report_biodiversity.py` generates the compact aggregate report at `experiments/results/biodiversity-v1/acquisition-report.json`. The fungal manifest SHA-256 is `b23fd8f2f57afbadebb68e496b8e3f147a8f2e89a4e6d97d636482034735b86c`.

Reproduce into fresh output directories:

```powershell
runtime-vision\Scripts\python.exe -m vision.acquire_biodiversity --parent data/public-fungi-v3/manifest.json --output data/biodiversity-fungi-v1 --per-genus 32
runtime-vision\Scripts\python.exe adapters/acquire_wildlife_context.py --output data/wildlife-context-v1 --per-category 6
```

The parent V3 corpus is produced by the existing `vision.acquire_public` and `vision.acquire_groups` workflows documented in `docs/DATASET_V2.md` and `docs/GROUP_VISION.md`. A new acquisition samples current public observations, so future runs can differ; each saved manifest is the authoritative record for that run. Use a fresh output path for every completed version.

The mushroom importer preserves frozen photographer partitions, exact-image identities and conservative near-duplicate connections. Newly downloaded photos remain `annotationStatus: unreviewed` with no boxes. It checks each photograph's reuse license and retains attribution. Existing broad `other_fungi` examples are reassigned to newly supported genera when recorded ancestry supports that mapping. Otherwise those examples would contradict the new genus labels. Old test images remain reused evaluation, not a fresh independent test set.

## Sources and access routes

| Source | Value for Mushroom Vision | Access and reuse |
| --- | --- | --- |
| [iNaturalist Licensed Observation Images](https://registry.opendata.aws/inaturalist-open-data/) | Broad fungal and wildlife diversity | Public image bucket, with per-photo licenses. This pilot accepts CC0, CC BY and CC BY-SA photos and separately checks observation rights. For substantial scaling, use the [monthly metadata exports](https://github.com/inaturalist/inaturalist-open-data) recommended by [iNaturalist's research guidance](https://help.inaturalist.org/en/support/solutions/articles/151000223300-how-i-download-inaturalist-photos-to-use-for-research-). |
| [Caltech Camera Traps on LILA](https://lila.science/datasets/caltech-camera-traps) | Wildlife/context scenes and source camera separation | Public per-image files; CDLA-Permissive-1.0. The bounded sample above avoids the full 105 GB archive. Retain the source's citation and rights-holder fields. |
| [GBIF](https://techdocs.gbif.org/en/openapi/) | Find images and occurrences from many institutions | Search API and registered-account bulk exports with download DOIs. Media licenses can differ from occurrence licenses. Deduplicate records already obtained through iNaturalist or Danish sources before merging. |
| [Mushroom Observer](https://github.com/MushroomObserver/mushroom-observer/blob/main/README_API.md) | Fungal observations and associated image metadata | Maintainers explicitly direct AI/bulk work to nightly CSV dumps instead of the API. The homepage links an [Images for Machine Learning route](https://mushroomobserver.org/articles/20). That article returned 403 during this check, so the request route is documented rather than bypassed. |
| [Danish Fungi 2024](https://github.com/BohemianVRA/DanishFungiDataset) | Fine-grained classification with observation-separated splits | Download links are public, but the repository explicitly restricts data, metadata and models to noncommercial research. Keep any future research-only experiment separate; request written terms before broader use or model redistribution. |
| [FungiTastic](https://bohemianvra.github.io/FungiTastic/) | Species recognition, segmentation and open-set tests | Its [download guide](https://github.com/BohemianVRA/FungiTastic/blob/main/dataset/README.md) supports subsets and sizes; Kaggle is an alternative. Preserve official temporal/observation and DNA-backed test separation. Clarify data and model redistribution rights before importing into a reusable release. |
| [FungiCLEF 2025](https://www.imageclef.org/FungiCLEF2025) | A rare-species benchmark and expert research contacts | The 2025 competition has ended. Use the organizer's published contact for current access and terms; don't represent a closed registration as a new application. |

## Prepared requests for additional access

These drafts are saved for the project owner to send. **No email, form or GitHub issue has been sent**, and no new account has been created. They do not claim university affiliation, nonprofit status, clinical validation, or permission to relicense contributors' photographs.

### Danish Fungi / FungiTastic / FungiCLEF maintainers

Published contact: **lukaspicek@gmail.com**, listed on the [Danish Fungi repository](https://github.com/BohemianVRA/DanishFungiDataset) and [FungiCLEF organizer page](https://www.imageclef.org/FungiCLEF2025).

Subject: Mushroom Vision — dataset access and release terms for an unfinished open-source project

Hello,

We are developing Mushroom Vision, an unfinished open-source project for human-reviewed mushroom recognition and field observation. Our public project introduction is https://thegreatlakesgleaner.press/articles/mushroom-vision.html and the volunteer review lab is https://thegreatlakesgleaner.press/mushroom-lab/.

We would like to evaluate fungal classification, segmentation, low-pixel detection and uncertainty using an appropriately sized subset of your dataset. We would retain attribution and original observation/temporal splits, keep DNA-backed test material out of training, and report failures as well as successes. The software is experimental and is not a medical diagnosis, edibility decision or validated autonomous drone system.

Could you confirm the applicable terms for research training, public demonstration, publishing model weights, and possible future commercial use? If those permissions differ, can you identify a subset suitable for an openly reusable model and a separate research-only subset? Please also advise whether bounding boxes, segmentation masks, genuine distant/blurred images or video sequences are available, and the appropriate download route and citation.

We will not treat the code repository license as permission to relicense the image data. Thank you for considering the project.

### Mushroom Observer

Route: [webmaster contact form](https://mushroomobserver.org/admin/emails/webmaster_questions/new), reached from the [official homepage](https://mushroomobserver.org/). A verified account may be required for website use; do not invent credentials or scrape around that requirement.

Subject: Mushroom Vision — recommended bulk image access for open-source fungal vision

Hello,

We are developing Mushroom Vision, an unfinished open-source project with a public human-review lab at https://thegreatlakesgleaner.press/mushroom-lab/. We understand that Mushroom Observer's API is not intended for AI/bulk scraping, and would like to use your recommended CSV and Images for Machine Learning routes instead.

Could you point us to the current bulk image download procedure and per-image license/attribution fields? Our initial interest is a bounded, diverse set of fungal observations, especially distant, blurred and environmental views, with observation IDs preserved for leakage-resistant evaluation. We would distinguish community taxon labels from verified image annotations, retain source credit, and keep noncommercial-only material out of broader-use releases.

Please let us know whether any additional permission is needed for training, sharing resulting weights, or a public volunteer labeling interface. The project is not a validated clinical or food-safety tool. Thank you.

## Larger acquisitions

The initial new-media budget is two gigabytes. Expand by explicit versioned batches after checking class balance, source independence, review workload and available storage. A large archive without usable annotations, license evidence and a protected test set is less useful than a smaller well-documented collection. The two implemented downloaders are bounded and refuse to overwrite frozen manifests.
