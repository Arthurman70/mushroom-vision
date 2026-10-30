# Local finds and human review database

The V0.4 backend accepts camera photos, stores local classifier suggestions, and sends updated find records to phone/Quest clients through an authenticated polling feed. It does **not** connect an actual camera, operate a drone, identify edible mushrooms, verify credentials, or authorize sales. Uploaded photographs demonstrate the ingestion path; they are not evidence of a working live drone driver.

Node 24 is required for its built-in `node:sqlite` module. The default database is `data/finds/finds.sqlite`, with content-addressed images in `data/finds/photos/` and immutable dataset manifests in `data/finds/exports/`. `MYCOSCOUT_FINDS_DIR` can select another trusted local directory. This setting is server configuration, never an API parameter. Back up the entire directory after stopping the server so the database and photo files remain together.

SQLite uses WAL, foreign keys, and full synchronous writes. Find creation, prediction events, and human review events have independent identities. Triggers prohibit updating/deleting existing finds and events. Later reviews append corrections; they never rewrite the original reviewer decision. Current views are derived from that event history. Photos are named using their SHA256, and stored bytes are rechecked before serving. No cloud service receives these photos.

## API

Every route below inherits the server's existing Bearer-token and exact-origin checks. LAN binding requires a token of at least 24 characters. Use the existing trusted HTTPS proxy or VPN for remote clients. An image element cannot supply a Bearer header; authenticated clients should fetch the image as a blob and display an object URL.

| Route | Behavior |
| --- | --- |
| `POST /api/finds` | Save a JPEG/PNG image and provenance; return `{find}`. |
| `GET /api/finds?since=0&limit=100` | Return `{finds,cursor,hasMore}` for finds changed after an event cursor. |
| `GET /api/finds/:id` | Return the current `{find}` including separate prediction and latest human review. |
| `GET /api/finds/:id/image` | Return original image bytes with `Cache-Control: no-store`. |
| `POST /api/finds/:id/reviews` | Append a human decision; return `{find,review}`. |
| `GET /api/events?since=0&limit=100` | Return append-only event records with IDs and prediction scores. |
| `POST /api/vision/classify` | Run the fixed local Python worker for `{findId}` or a direct image. |
| `GET /api/vision/status` | Report checkpoint presence, class count, worker start/load/busy status; no live camera connection is implied. |
| `GET /api/catalog` | Return the local Michigan catalog and its source/scope notes. |
| `GET /api/training/export` | Save and return a versioned manifest containing eligible human-reviewed, opted-in records. |

Polling clients must retain the returned cursor, immediately request more pages while `hasMore` is true, and then poll again using that cursor. A find may recur after new events; replace its local displayed value by ID. The cursor counts events, not photos. `limit` is 1–200. One prediction event includes the top three internal class guesses, coarse group IDs, and model scores. Scores are not verified identification probabilities.

Upload example (image bytes omitted):

```json
{
  "imageBase64": "<base64 JPEG or PNG>",
  "mimeType": "image/jpeg",
  "source": "phone_upload",
  "captureTimestamp": null,
  "deviceId": "phone-1",
  "sessionId": "survey-2026-10-26-A",
  "groupId": "specimen-A-004",
  "specimen": {"viewType": "cap", "specimenId": "specimen-A-004"},
  "metadata": {"actualLiveHardware": false},
  "rights": {"trainingOptIn": false},
  "evaluationOnly": false
}
```

`captureTimestamp` accepts an ISO date or Unix milliseconds; leave it null when unknown. The server records receipt time separately and never substitutes it for capture time. `groupId` on an **upload** is a specimen/source grouping identifier, not a predicted mushroom category. Use project-unique identifiers: sessions and specimen IDs join across phone/drone sources and devices. `camera`, `depth`, and `metadata` accept bounded JSON provenance. They do not create metric map locations or clear unknown airspace.

Review example:

```json
{
  "action": "confirm_group",
  "expectedReviewId": null,
  "groupId": "chanterelles_trumpets",
  "classId": "black_trumpets",
  "reviewerName": "Field reviewer",
  "credentialReference": null,
  "specimen": {
    "specimenId": "specimen-A-004",
    "viewType": "underside",
    "cap": "Funnel shaped",
    "underside": "Observed wrinkles",
    "stem": "Recorded field observation",
    "substrate": "Leaf litter",
    "odor": "Not assessed",
    "sporePrint": "Not taken",
    "size": {"capMm": 20},
    "environment": "Recorded field context",
    "features": ["Field note requiring review"]
  },
  "notes": "Broad visual label only; species remains unverified",
  "rights": {
    "trainingOptIn": true,
    "license": "Photographer permits local research training",
    "attestedBy": "Photographer"
  }
}
```

Actions are `confirm_group`, `correct_species`, `unresolved`, and `reject`. Here `groupId` must be one of the catalog's coarse groups. `correct_species` requires an explicit `species` text value. Optional `classId` must match a catalog entry and its group; a class is never copied from the AI prediction. A species entered by a person remains `speciesReported`, not professional verification. A reviewer name/credential reference is self-asserted. Every prediction/find/review keeps sales and flight approval false. `humanReviewed` and the reported label are separate from `professionallyVerified:false` and `speciesConfirmed:false`.

Supported specimen fields are cap, underside, stem, substrate, odor, sporePrint, size, environment, bruising, texture, habitat, locationNotes, other, viewType, features, and specimenId. Later reviews preserve earlier character fields when omitted. A new specimen ID adds permanent grouping evidence; removing or changing its displayed value cannot split previously linked records into separate datasets.

Send `expectedReviewId` with the review ID that was visible when the form opened, or explicit null for a never-reviewed find. The server checks this inside the append transaction. A newer review causes `409 REVIEW_CONFLICT`; reload before saving, including when another device revoked training permission. Prediction updates do not change this review version. Omitting the field retains older-client compatibility but forfeits conflict detection; current clients should always supply it.

## Local classifier

The server lazily launches only the fixed `vision.multiclass worker --checkpoint vision/checkpoints/groups-v1.json` module. On Windows it uses `runtime-vision/Scripts/python.exe`; on Linux it uses `runtime-vision/bin/python`. A trusted server operator may set `MYCOSCOUT_PYTHON` to the intended Python executable. API callers cannot supply executable paths, commands, checkpoints, or input file paths.

The worker uses newline-delimited JSON on stdin/stdout and logs on stderr. Each response must match its request ID. Only one image is in flight; another returns `429 CLASSIFIER_BUSY`. Startup and inference together have a 90-second timeout. Invalid or oversized replies kill the worker; the next request can start a fresh instance. The worker receives offline model settings, so it must have its environment, weights and checkpoint prepared locally. Missing runtime/model/checkpoint returns a visible error, never fabricated guesses. Actual compatibility and inference speed on Jetson still need device testing. `workerLoaded` becomes true only after a successful worker response; checkpoint presence alone is not proof of model readiness.

## Versioned training releases

Only the latest explicit `confirm_group` or `correct_species` decision plus training permission can enter an export. Permission needs `trainingOptIn:true`, a license/permission statement, and an attesting person. AI guesses, unresolved finds, rejected findings, and the unknown group do not supply labels. Contradictory human labels for identical image bytes are excluded until resolved. Permissions/credentials are declarations, not automated verification.

The manifest schema is `mycoscout.reviewed-training-export.v1`. Each record has `findId`, `image` (SHA256, size, MIME, relative path and authenticated URL), `sourceGroupId`, `partition`, `labels`, `review`, `rights`, `specimen`, and `provenance`. `provenance.sourceMetadata` preserves bounded upload metadata, including original photo license, photographer attribution, source URL and observation references when supplied. It is provenance, never a label source. `labels.classId` is null unless a human explicitly chose that finer class. The 22-class importer must skip broad-only records instead of inferring a fine class from the model. `image.relativePath` is relative to the finds database directory, not the `exports` directory.

Partition assignment uses a deterministic initial 80/10/10 hash distribution; small samples need not have those proportions. Exact image hashes, shared session IDs, initial source group IDs, and all specimen IDs supplied during reviews form connected groups. All records in a group share a partition. Existing assignments never silently move across training/test boundaries: conflicts quarantine the whole connected group. `evaluationOnly:true` forces a new group to test; connecting it to an established non-test group causes quarantine. Even non-exported finds retain grouping links.

An importer must deduplicate identical SHA256 images and reconcile them against the existing acquisition corpus. Inherited validation/test assignments take precedence; a conflict must not become training input. Use train records for fitting, validation for model selection, and test only for evaluation. Near-duplicate images without declared shared provenance cannot be reliably identified here: group sequences/specimens before export and retain the original corpus's observer/session grouping.

The release ID hashes its records, exclusions and catalog version; unchanged exports keep the same ID. Manifests are written once. Revocation/correction affects the next release but cannot retract files already copied or models already trained. Rebuild from the current release when permissions or labels change. Exporting does not trigger automatic training, cloud upload, publication, or deployment.

## Bounds and validation

Uploads accept only canonical base64 (or an explicit JPEG/PNG data URL), verify MIME against magic bytes and structural headers, limit decoded images to 8 MiB/20 megapixels/20000 pixels per side, and reject animated PNG. Structural validation is not full image decoding; the Python worker also must decode and validate the image before inference. The JSON body limit is 12 MiB for image endpoints; other existing endpoints retain their 1 MiB limit. Metadata/character JSON is bounded in depth, field counts and lengths.

Storage limits are 10000 finds, 2 GiB of unique photo bytes, 100000 total events, 200 reviews and 500 predictions per find, and eight queued database writes. Export/archive the local dataset before those limits. No public static directory contains the private image store. Model output is limited to 256 KiB; reviews validate categories against the catalog. Existing server tests still cover authentication, CORS, path traversal, walking-route gates, and absent flight control routes.
