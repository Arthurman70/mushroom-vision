# Expanded Mushroom Vision experiment

This is an unfinished, exploratory classifier for whole photographs or selected
crops. It does not localize mushrooms, determine range, verify species, or replace
the default reviewed detector. The `/species-lab.html` page exposes it separately.

## Acquisition and identity

The expanded frozen fungal corpus has 3,922 photos: 3,250 inherited and 672 new.
It covers 43 research categories and 501 source taxon IDs, with 2,319 training,
748 validation and 855 test photos. Source taxonomy is weak evidence, not an
independent human confirmation of what is visible in every image. The parent
photographer and duplicate groups retain their previous partitions. Inherited
test examples are reused evaluation, not a newly untouched test set.

Manifest SHA-256:
`b23fd8f2f57afbadebb68e496b8e3f147a8f2e89a4e6d97d636482034735b86c`

New fungal image bytes total 398,664,149, licensed per image as 464 CC BY, 183 CC0
and 25 CC BY-SA. An independent 113-image LILA wildlife subset adds 45,751,020
bytes. Wildlife categories, including animal-empty frames, do not establish
mushroom absence and are excluded from mushroom-negative training pending review.
See [sources and access routes](DATA_SOURCES.md).

## Actual training and evaluation

We extracted frozen official ResNet18 features for all 3,922 photos, fitted
three regularized linear heads, selected by validation results, then calibrated
temperature and the abstention rule using validation only. Test results were
read after those choices. The experimental support floor is 10 training photos
from at least 10 groups and three validation photos from at least three groups.
These small support thresholds are exploratory, not evidence of robust coverage.

38 categories meet the support floor. Michigan truffle, Galerina, Inocybe,
Psilocybe and Lepiota remain unsupported; their labels are still available for
human research annotation. The 108-name review vocabulary is finer than these
model categories and must not be presented as 108 trained species.

| Held-out archive-photo measure | Result |
|---|---:|
| Test images | 855 |
| Top category accuracy | 59.88% |
| Correct category among top three | 78.83% |
| Broad visual group accuracy | 67.37% |
| Guesses passing score ≥0.40 and margin ≥0.30 | 458 / 855 (53.57%) |
| Accuracy within those retained guesses | 82.10% |
| Incorrect retained guesses | 82 |
| Abstained images | 397 |

The class set differs from earlier 22-category experiments, so these figures
are not a like-for-like improvement claim. Rare groups, unfamiliar backgrounds,
occlusion, tiny targets and field cameras still need dedicated evaluation.
Confidence scores are empirical model outputs, not probabilities of a correct
species determination. No benchmark establishes clinical usefulness or detection
success at 10, 20, 50 or 100 metres.

The new head remains an opt-in research endpoint; existing models and approved
training workflows are unchanged. No downloaded source label becomes a trusted
human grade automatically.

## Reproduce on a fresh checkout

Install the optional vision dependencies and official ResNet18 weights according
to [vision setup](VISION_RUNTIME.md) and [group vision](GROUP_VISION.md). Download
versioned collections, preserving their full licenses and original splits:

```sh
python -m vision.acquire_public --output-dir data/public-fungi-v2 --max-images 1200
python -m vision.acquire_groups --parent data/public-fungi-v2/manifest.json --output-dir data/public-fungi-v3
python -m vision.acquire_biodiversity --parent data/public-fungi-v3/manifest.json --output data/biodiversity-fungi-v1 --per-genus 32
python -m vision.multiclass train --manifest data/biodiversity-fungi-v1/manifest.json --output experiments/results/groups-expanded-v1 --checkpoint vision/checkpoints/groups-expanded-v1.json --min-train-photos 10 --min-validation-photos 3
```

Live source collections change. A fresh download will have a different manifest
and results; exact reproduction needs the original frozen, licensed manifest
and original image bytes. Public source releases exclude images, private reviews
and model weights. Never overwrite a frozen dataset or experiment to reuse its
name. The head is loaded only from the fixed local checkpoint path shown above.

## Compute measurements

On the development Ryzen 7 5700X3D PC with 32 GiB RAM and CPU-only PyTorch,
three saved video frames gave a median 38.8 ms for Python quality analysis and
7.59 s for GroundingDINO detection (four CPU threads). Model loading and warm-up
were measured separately. A browser run on one morel frame measured 10.8 ms
median for quality analysis, including image decoding; its canvas pipeline is
different and not a detector benchmark. No phone, Quest or Pi hardware was tested.

With the measured detector time and an assumed 30 Mbit/s link, the simulator
completed 100/100 jobs at one frame per ten seconds under a 15-second deadline;
one frame per second overloaded the worker (15/100 in time). These are modeled
connections, not measured radio performance. Prefer occasional investigation
frames and locally selected quality crops; preserve original pixels and location
evidence. See [device protocol, workers and measurements](DEVICE_COMPUTE.md).
