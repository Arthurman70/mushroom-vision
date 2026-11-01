# AI visual review desk

Open `/ai-review.html` on the local field station. This is an unfinished research tool for an assistant that can **actually inspect images**. Its judgments remain AI annotations even when the assistant performs the same visual task as a person.

The desk shows original frames, accepts boxes in original pixels, enlarges selected crops, records broad-group confidence and evidence, and exports a separate candidate dataset. It does not convert AI opinions into human confirmations, activate a detector, or send robot commands.

## Review procedure

1. Create a finite run with the actual reviewing model identity. If the exact model version is unavailable, say so rather than inventing it. A default run spreads timestamps across eligible sources without ranking by model scores.
2. Inspect the original image. Use an original-pixel crop where needed. A crop enlarges existing information; it cannot prove something absent from the pixels.
3. Mark **visible**, **absent**, or **uncertain**. Use uncertainty for unresolved tiny shapes, blur, occlusion, or a scene that cannot be searched adequately. A title saying “morels” is not evidence that a particular frame contains one.
4. For visible mushrooms, draw boxes tightly around the visible fruiting bodies and enter a broad group, confidence, and the visual reason. Do not infer a species from a source title. Confidence is the reviewer's judgment, not a calibrated probability.
5. Mark a review complete only after inspecting the whole frame and all its discernible mushrooms. Complete absent frames may be experimental negatives; uncertain frames are excluded from fitting.
6. Save the AI review. Corrections append another revision and require the current revision number. Earlier records cannot be edited or deleted through this interface.

Existing model suggestions are hidden by default. Revealing them records `modelSuggestionsViewed:true`, so a later analysis can distinguish blind visual review from assisted review. It does not copy boxes automatically. Evidence from another video frame must be stored separately in `temporalEvidence`; it never creates visibility or boxes in this frame.

Humans should use the separate human grader. If a person later adopts an AI candidate, they must inspect the original pixels and preserve the assistance provenance in their own review.

## Partition protection

`config/ai-review-partitions-v1.json` fixes whole source groups for this independent AI experiment:

| Partition | Source groups |
| --- | --- |
| Train | `youtube:1Z9FgVpYris`, `youtube:ng7VOJHum7U`, `youtube:OhN1BQgK2L8` |
| Validation | `youtube:go_1s6wVScI` |
| Test | `youtube:creator:JoshuaMcConkeyFishing`, `archive:Freestonew-theHugeMushroom948` |

The server also reads the current `TrainingPipeline.sourceSnapshot()`. Existing frozen validation/test partitions, evaluation hashes, evaluation-only sources, and shared video identities override a less restrictive assignment. Duplicate image hashes connect whole source groups, preventing another frame from a duplicated held-out clip entering training. These checks run both when creating the queue and when exporting it. Unknown groups remain unassigned until an explicit source policy exists.

Test review runs may be inspected and saved, but their records never appear in training exports. Validation records can be exported for comparison, with `experimentalTrainingAllowed:false`; only a separate experimental trainer may consume AI training records. The ordinary human-reviewed pipeline never reads this ledger.

## Local assistant API

All routes are behind the existing station owner authentication. The browser sends the access token only in an authorization header and retains it only in memory. Files, URLs, executable commands, and checkpoints cannot be nominated through this API.

| Method | Route | Use |
| --- | --- | --- |
| GET | `/api/ai-review/status` | Source counts, partitions and recent runs |
| GET | `/api/ai-review/experiment` | Read-only completed experimental pass summary, or explicit not-run state |
| POST | `/api/ai-review/runs` | Freeze a finite queue |
| GET | `/api/ai-review/runs/{runId}` | Original frame identities, image URLs and latest reviews |
| POST | `/api/ai-review/runs/{runId}/reviews/{frameId}` | Append an AI review |
| GET | `/api/ai-review/runs/{runId}/reviews/{frameId}` | Immutable review history |
| GET | `/api/ai-review/runs/{runId}/export` | Current eligible AI candidates and exclusion reasons |

Run request example:

```json
{
  "title": "Blind original-frame review",
  "reviewer": { "name": "Codex visual reviewer", "model": "Exact model if known; otherwise explicitly unreported" },
  "split": "train",
  "limit": 18
}
```

An optional `frameIds` array selects 1–100 specific frames in the chosen partition. The server verifies the original file hashes and dimensions before freezing the queue. Each selected frame includes its source/video hashes, source group, split and source credits.

After reading the actual image, submit a review using hashes returned by the run:

```json
{
  "baseRevision": 0,
  "sha256": "<exact original image SHA-256>",
  "videoHash": "<exact source video SHA-256>",
  "viewedOriginal": true,
  "modelSuggestionsViewed": false,
  "status": "visible",
  "complete": true,
  "boxes": [
    {
      "bboxPixelsXYXY": [100, 100, 130, 145],
      "groupId": "morels",
      "label": "Morel candidate",
      "confidence": 0.75,
      "evidence": "Illustrative coordinates only. Replace with evidence from the actual frame."
    }
  ],
  "evidence": "Describe the original frame, unresolved regions and reasons for the chosen status.",
  "viewedCrops": [],
  "temporalEvidence": []
}
```

The example is a schema illustration, not a real annotation. A JSON claim that an image was viewed is not proof of viewing. The assistant must use its image-viewing tool or the browser first. `visibility: "yes" | "no" | "uncertain"` is accepted as an alternative to `status`.

Optional temporal evidence has `{frameId, sha256, note}` entries. The server verifies that each reference belongs to the same exact source video. It remains a separate contextual assertion with `inferredVisibility:false`.

## Export contract

`mycoscout.ai-visual-review-export.v1` contains `records` and `excluded`. A record preserves `runId`, `frameId`, `sourceId`, `sourceGroupId`, `sha256`, `videoHash`, `width`, `height`, `split`, `status`, `complete`, `boxes`, reviewer model identity, evidence and revision. Boxes use `bboxPixelsXYXY` in original pixels.

Every record retains:

```json
{
  "reviewerKind": "ai",
  "humanVerified": false,
  "labelAuthority": "ai-visual-review",
  "trainingAllowed": false,
  "experimentalTrainingAllowed": true,
  "automaticActivation": false
}
```

`experimentalTrainingAllowed` is true only for eligible training records. Validation records retain false. The export's `productionTrainingAllowed:false` is deliberate. An isolated trainer must explicitly support this AI schema, keep it out of production model activation, recheck source/hash holdouts, and report results separately from human-ground-truth evaluations. A small AI-labeled run does not demonstrate improved field detection.

Durable records are stored in `ai-visual-review.sqlite` in the server's AI review directory. This private ledger and images remain outside the source-only GitHub package.

## Server integration

Initialize the store after the ordinary learning service:

```js
const aiReview = await new AiReviewStore(aiReviewDirectory, {
  videos,
  collectionId: videoCollectionId,
  partitionsFile: resolve(ROOT, 'config/ai-review-partitions-v1.json'),
  holdoutSnapshot: () => learning.sourceSnapshot()
}).initialize();
```

Under the existing owner authentication, call `aiReview.dispatch(method, pathname, parsedInput)` for `/api/ai-review/` routes and return its `{status,data}`. Do not expose it through device-worker limited authorization. Call `aiReview.close()` on shutdown. No cloud AI service or API key is required for an assistant already able to inspect the images.

Focused checks:

```text
node --test tests/ai-review.test.mjs
```

The checks cover authority, immutable history, strict holdouts, duplicate-source leakage, source mutation, bounds, temporal evidence, and original-pixel UI geometry.

## First assistant pass

The first saved pass inspected 21 original video frames: 18 from training sources and three from the reserved validation source. It recorded five visible frames, three absent frames and 13 uncertain frames. Two of the visible frames were deliberately marked incomplete. Six complete training frames with five boxes across three source groups qualified for the separate one-pass experiment. None of the validation views qualified for evaluation, so the report has no validation accuracy or improvement claim.

The run report is `experiments/results/ai-review-candidate-v1/report.json` on the station. Its checkpoint uses `mushroom-vision.ai-candidate-checkpoint.v1`, which the operational detector loader rejects. The focused Python contract suite tests that rejection without loading a neural network, and also tests authority, frozen validation/test precedence, original and decoded pixel holdouts, invalid geometry, image mutation, and path containment:

```text
python -m unittest experiments.test_ai_review_candidate -v
```

The trainer's stricter input checks were rerun against the saved export after review; the same six frames and five boxes remain eligible. The training computation itself did not need repeating.

The AI review desk displays this completed experiment above the frame queue: six optimizer steps, six frames, five boxes and three source groups, with zero eligible validation frames. Its summary endpoint reads one fixed report and returns only aggregate counts, mean training loss and inactive-model status. It never exposes frame IDs, source export IDs, local paths, weights or free-form report contents. Missing reports show “not run”; unreadable or inconsistent reports show “summary unavailable.” No activation control is attached to the card.

## Public AI contribution inbox

The Gleaner exposes a separate contribution workflow at `/mushroom-lab/ai-contribute.html`, with its machine-readable contract at `/mushroom-lab/ai-openapi.json`. Agents register a self-reported model identity, retrieve a leased original, inspect it and submit pixel-coordinate boxes or an explicit uncertain judgment. Image delivery is recorded, while actual visual inspection remains self-attested. Each opinion stays `reviewerKind: "ai"`, `reviewStatus: "quarantined"`, `humanVerified: false` and `trainingApproved: false`. Human reviewers can inspect these opinions in the lab. Downloading an image does not establish that an AI inspected it, and confidence values are not calibrated probabilities.

An authorized operator exports this inbox separately from community human reviews:

```text
python deploy/community/deploy.py export-ai-contributions --output data/community-ai-export-20261031.json
node community/import-ai-contributions.mjs data/community-ai-export-20261031.json data/community-ai-inbox-v1 data/community-publish-all-v1
```

Use the exact current public collection directory as the final argument. The importer verifies its original image bytes, dimensions, source credits, license, source group and evaluation split. Changed duplicate IDs, forged human authority, stale image hashes and conflicting group splits fail closed. It preserves the original annotation permission receipt and appends previously unseen opinions to a separate quarantine file; repeating the same export does not duplicate records. Neither command writes human-review data, starts training, downloads remote URLs, executes submitted code or activates weights.

Curators should inspect the original image and each proposed box, especially hats, bark, leaves and stumps. An independent human correction belongs in the human review workflow with AI assistance acknowledged. Only after the existing curator approval process should eligible training-source records enter the approved training pool. Evaluation-only groups must remain evaluation-only, even when a human agrees with an AI. The quarantined AI inbox must never be renamed to a human export or passed directly to the training worker; it is evidence for review, not ground truth.

Focused import checks: `node --test --test-isolation=none tests/community-ai-import.test.mjs`. Public API checks: `node --test --test-isolation=none tests/community-ai-contributions.test.mjs`.
