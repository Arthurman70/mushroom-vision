# Research identification vocabulary

Mushroom Vision accepts 108 research names: 78 species names and 30 genera, across
14 broad visual groups. This includes morel lookalikes, other gilled mushrooms,
jelly fungi, coral fungi and stinkhorns. These are annotation choices, **not a
claim that 108 taxa can be identified by the current model**.

`config/research-taxa-v1.json` is separate from the unchanged Michigan study
catalog. It does not expand certification or sales eligibility. The catalog
preserves names and typed aliases from the cited study guides without treating
historical names as automatically equivalent modern taxa. Such entries say
`source_name_unresolved`. Twenty-one additional genus names have exact-name
iNaturalist resolutions frozen in `config/research-taxa-sources-v1.json`.
Each entry links to its source records. A genus identification does not identify
a species, and photo resemblance does not establish edibility.

## Reviewing an image

Mark the visible mushroom and choose a broad group. Open the optional
identification section to record a research name, low/moderate/high subjective
confidence, evidence type and a note. Leave the name unresolved when the detail
is insufficient. A claim based on microscopy, DNA, field notes, another view or
an expert reference requires a note identifying that evidence. The system
records the claim; it does not validate those external records.

Public fieldwork, local video grading and the phone/Quest review page all use the
same catalog and editor. Changing a broad group clears an incompatible draft
name. Old group labels and old reviews remain valid. No historical permission
or human approval is inferred from the new vocabulary.

The saved `taxonCandidate` records `taxonId`, scientific name, rank, broad group,
confidence, confidence meaning, evidence source/note and catalog version.
Validation always sets `status: tentative`, `speciesVerified: false`,
`trainingApproved: false`, and `modelSupport: catalog_only`. A client cannot
override those fields by submitting a forged confirmation. Candidate metadata
survives community export/import, local review history and training exports.
It does not become a species training label automatically. Existing class-label
and curator-approval rules still govern training.

## Updating the catalog

Run `python scripts/build-research-taxonomy.py` to rebuild from the source guide
catalog and frozen resolution snapshot. It creates identical catalog modules in
`community/public` and `public`, and copies the shared editor for offline phone
packaging. `tests/research-taxonomy.test.mjs` checks source links, identifier
uniqueness, metadata validation, legacy compatibility, durable round trips and
package parity. New model support requires a separately trained and evaluated
model; changing this vocabulary never enables it.
