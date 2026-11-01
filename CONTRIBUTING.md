# Contributing to Mushroom Vision

Mushroom Vision is unfinished, open-source research software. Its internal
MycoScout names remain for compatibility. We welcome fixes, documented device
measurements, licensed data integrations, and careful image annotations.

## Start locally

Use Node.js 24 or later and run `npm start`. The station binds only to localhost
by default. Optional Python vision tools need their separately listed
dependencies and model files. Do not commit runtime folders, downloaded data,
tokens, camera recordings, locations, or contributor reviews.

Run `npm test` for the station and community regression checks. New device,
taxonomy and downloader tests should accompany changes to those contracts.
Tests must use isolated temporary directories, never a production database.

## Evidence and data

Keep the original image license, photographer credit, source URL, content hash
and observation/video grouping. A taxonomy label from a source is weak evidence
until image content is reviewed. Never promote generated guesses or votes to
expert verification automatically. Keep all frames from one recording session
together when splitting training and evaluation data.

Report actual measured latency separately from simulated bandwidth, packet loss
and device assumptions. A desktop benchmark is not a Quest, phone or Raspberry
Pi benchmark. Improving a photo classifier does not establish drone detection
range or species certainty.

## Contribute AI annotations

Use the [public AI guide](https://thegreatlakesgleaner.press/mushroom-lab/ai-contribute.html)
and [OpenAPI contract](https://thegreatlakesgleaner.press/mushroom-lab/ai-openapi.json).
Register your tool's provider, model and version, lease an image, retrieve its
original pixels, then submit your observations with the lease and an idempotency
key. Keep the temporary credential private. Inspect the image itself; do not
copy a source taxon label into a claim that a mushroom is visible.

Return original-pixel boxes, visibility, uncertainty and short visual evidence.
An uncertain judgment is useful when the image does not support an identification.
Mark a frame complete only when you have inspected it for every visible mushroom.
AI model identity and actual inspection are self-attested, and all contributions
remain AI opinions in quarantine. They are visible to reviewers but cannot
overwrite human annotations. An operator imports this queue separately; curated
training eligibility and held-out source boundaries still apply. See
[the review workflow](docs/AI_VISUAL_REVIEW.md#public-ai-contribution-inbox).

## Scope

No flight arming or physical motion control is exposed. Any future aircraft
integration needs its own engineering and field validation. Species guesses
are research assistance, not edibility, clinical diagnosis, or Michigan sales
approval. The medical and biodiversity goals require independent expert work.

By contributing code, you agree to make it available under the project's MIT
license. Data and model contributions require a separate, compatible license;
the code license does not grant rights to someone else's images.
