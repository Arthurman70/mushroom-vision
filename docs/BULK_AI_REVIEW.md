# All-source AI image review

Open `/bulk-ai-review.html` on the local station. The image archive shows coverage, source partitions, visibility, evidence size, original images, optional AI mushroom/nonfungal boxes, and every saved revision. The page is read-only: opening it does not create an observation. Human confirmation remains in the Gleaner community lab.

The archive API requires the existing owner token when configured and is restricted to the loopback station. The full inventory includes private/research-only material. Public community packaging separately applies each image's redistribution rights. Never copy the full local inventory or private originals into the public site.

## Frozen originals

`scripts/inventory_bulk_ai_review.py` built the frozen `data/bulk-ai-review-v1/manifest.json` and its SHA-256 pin. This pass inventories 4,545 unique originals from 9,719 source references: 492 extracted video frames, 3,939 fungi photos, 113 wildlife photos, and one submitted photo. Exact-byte duplicates keep all provenance aliases. The inventory excludes synthetic fixtures, QA images, temporary images and derived detection/find crops. Do not regenerate it during a review run.

The manifest includes dimensions, exact bytes hash, decoded pixel hash, source identity, rights, aliases and frozen partition. Pixel hashing is SHA-256 of compact JSON `[oriented_width,oriented_height]` followed by the raw bytes of Pillow `ImageOps.exif_transpose(image).convert('RGB')`. Strict source/image/pixel holdouts are propagated before freezing. The current inventory has 2,555 training, 833 validation, 1,097 test and 60 unassigned images; 4,467 have public display permission and 78 remain local only.

Sheets contain up to sixteen images at a generated maximum side of 384 pixels. The viewing tool can further resize the sheet. A review therefore records generated tile size separately from an actual reported display size. If tool scaling is unknown, the displayed dimensions are null. Preview-only judgments are always incomplete and never become fitting labels, including previews judged absent.

## Durable observations

`core/bulk-ai-review.mjs` maintains the append-only SQLite ledger. AI identity, original hash, run, revision, visibility, completeness, evidence, viewed scope, boxes and source split remain explicit. It rejects stale revisions, changed bytes, changed sheet bytes, out-of-bounds boxes, unsupported authority, and complete preview/crop labels. Previous AI original reviews remain separate historical evidence. They are not human confirmations.

Original-image evidence takes precedence over preview evidence. Among original-image revisions, the latest judgment controls export; an uncertain correction retires an older complete judgment. A whole frame can be a negative only after a complete original-image review explicitly judges it absent. Context tags such as tree, leaf, stump or hat do not make negative examples. A nonfungal box requires explicit evidence and cannot overlap a positive mushroom box.

Owner-local routes:

- `GET /api/bulk-ai-review/status`
- `GET /api/bulk-ai-review/images?offset=0&limit=24&status=all&mode=all&split=all`
- `GET /api/bulk-ai-review/images/{imageId}` — source metadata and immutable review history
- `GET /api/bulk-ai-review/images/{imageId}/original` — verified image bytes
- `POST /api/bulk-ai-review/runs` — title and AI reviewer model identity
- `POST /api/bulk-ai-review/runs/{runId}/reviews` — `{ "reviews": [...] }`, 1–256 actually observed images
- `POST /api/bulk-ai-review/export` — snapshots only; no training or activation

Review JSON uses `imageId`, `sha256`, `baseRevision`, `status` (`visible`, `absent`, `uncertain`), `complete`, `boxes`, `confidence`, `evidence`, `modelSuggestionsViewed`, and `observation`. Original observations use `{mode:"original-image",viewed:true,displayedWidth,displayedHeight}`. Preview observations use `{mode:"contact-sheet",viewed:true,sheetId}` plus optional actual `renderedSheetWidth`/`renderedSheetHeight`. Positive boxes use original-pixel `bboxPixelsXYXY`, `groupId`, `confidence`, and `evidence`. Confounders additionally require `kind` and `explicitNonFungal:true`. JSON submission itself does not prove pixels were viewed; the assistant must actually inspect them.

## Assistant helpers and exports

After inspecting originals, `scripts/import-observed-ai-notes.mjs RUN NOTES...` imports notes with frame IDs, retaining the note-file hash and source-frame match. `scripts/submit-ai-sheet-notes.mjs RUN NOTES` accepts compact `{reviewerKind:"ai",humanVerified:false,viewedSheets:["sheet-0095"],records:[{index:1520,status:"uncertain",evidence:"Visible observation..."}]}`. It fills the pinned source hash, records preview-only scope and skips images already reviewed, so a weak preview cannot overwrite an original observation. No source-class default labels are accepted or generated.

`scripts/bulk-ai-review.mjs export` writes `ledger.json` and `training-export.json`. The ledger contains every immutable revision and a SHA-256 of recursively key-sorted JSON for each record. The training snapshot records the exact inventory hash, exact ledger-file hash, and selected ledger record hash. Its schema is `mushroom-vision.ai-corpus-training.v1`, with `reviewerKind:"ai"`, `humanVerified:false`, and `activationEligible:false`.

Only complete original-image AI annotations with permitted training rights and training/validation splits are exported. Tests, unassigned sources, uncertain/incomplete observations, and preview-only labels stay outside fitting. Validation remains validation. The experimental trainer independently verifies paths, bytes, decoded pixels, provenance, ledger correspondence, and holdouts before fitting. Its checkpoint cannot replace the operational model.

Checks: `node --test --test-isolation=none tests/bulk-ai-review.test.mjs tests/server.test.mjs`. The focused tests cover immutable authority, revision conflicts, image/sheet integrity, preview resolution and exclusion, explicit nonfungal evidence, holdouts, original-over-preview selection, gallery bounds, and owner/worker access separation.
