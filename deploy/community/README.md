# Public mushroom contributor service

This deployment publishes only the standalone community app and the explicitly exported public collection at `https://thegreatlakesgleaner.press/mushroom-lab/`. It contains no station control, camera stream, flight API, private photos, or research-only media. Model suggestions remain unverified until a person reviews them; contributor reviews are not automatically approved training labels.

Run from the project directory with its Python runtime. Set `MYCOSCOUT_DEPLOY_TARGET` to your authorized SSH `user@host`, or keep an `sshTarget` value in ignored `data/deployment-target.json`. No passwords or tokens belong in these files. The supplied deployment recipe describes Gleaner's existing container layout; adapt its domain, paths and proxy network to your own host before deployment.

```powershell
.\runtime-vision\Scripts\python.exe deploy/community/package.py --collection data/community-publish-v1 --output dist/community-20261029-v1.tar.gz
.\runtime-vision\Scripts\python.exe deploy/community/deploy.py upload --bundle dist/community-20261029-v1.tar.gz --release community-20261029-v1
.\runtime-vision\Scripts\python.exe deploy/community/deploy.py publish --release community-20261029-v1
.\runtime-vision\Scripts\python.exe deploy/community/deploy.py status
```

Uploading only stages a release. Publication validates the latest live Caddy configuration, builds the small app on the existing Node 24 image, backs up Caddy, preserves the contributor database, starts the isolated service, checks its health, adds only the contributor route, and verifies the public URL. Other virtual hosts remain intact. A failure attempts rollback. A concurrent Caddy edit stops the operation instead of overwriting that edit.

If the frontend changes after upload but before publication, `deploy.py refresh-staged-app --release RELEASE-ID` refreshes only the app and deployment files. It refuses to mutate the currently published release. New app versions need a new release; suggestion updates have the separate command below.

For app-only updates with the same images, `deploy.py clone-current --release NEW-RELEASE-ID` stages a new release on the server and hard-links its immutable images. Then run `refresh-staged-app`, optionally `update-ai --file PATH --release NEW-RELEASE-ID`, and `publish` for that new release. This transfers no image bytes and keeps the previous app and suggestions intact for rollback.

The Docker service has a read-only filesystem and collection, a persistent `/data` database, no host port, dropped capabilities, and bounded resources. Only the existing Caddy container reaches it on the shared proxy network. The proxy overwrites the client address header used by the app's rate limits. Requests to the www contributor URL redirect to the canonical origin.

## Refresh progressive AI suggestions

```powershell
.\runtime-vision\Scripts\python.exe deploy/community/deploy.py update-ai --file data/community-publish-v1/ai-suggestions.json
```

This validates frame IDs, source IDs, exact image hashes and unverified flags, saves the preceding AI file, and atomically replaces the suggestions file. Images are not copied and the service does not restart. It never promotes AI results to human training labels.

## Retrieve contributor reviews privately

```powershell
.\runtime-vision\Scripts\python.exe deploy/community/deploy.py export-reviews --output data/community-contributions-20261029.json
```

The export runs inside the container over SSH and writes a new local JSON file. The public service has no export endpoint. Existing local files are not overwritten. The export retains reviewer pseudonyms, consent, revisions, AI assistance, and `trainingApproved: false` for later local expert review.

## Turn contributions into reviewed training examples

Run these commands from `mycoscout`. The dated filename avoids overwriting a prior export:

```powershell
$reviewExport = 'data/community-contributions-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json'
.\runtime-vision\Scripts\python.exe deploy/community/deploy.py export-reviews --output $reviewExport
& 'C:\Program Files\nodejs\node.exe' community/import-reviews.mjs $reviewExport
```

Import verifies every frame against the local main collection and merges into `data/community-inbox-v1/reviews.json`. It retains immutable imported revisions in `history`, rejects conflicting revisions or changed pixels, and leaves every contribution `community-unverified`. Import itself does not write human grades or start training. These commands are the manual recovery path; the optional local worker below synchronizes fresh exports automatically.

New public submissions acknowledge the displayed participation terms, permitting training and evaluation use of their annotations, including commercial software. Image licenses remain separate. The service records the terms version, notice hash and acceptance time for that revision. Historical withheld consent is preserved; it is not changed merely by updating the app. Contributors can withdraw future-use permission from the same anonymous browser session. Withdrawal cannot undo completed training, and participation permission does not make a contribution a trusted label.

Open the local curator at `http://127.0.0.1:8787/video-grader.html`. Choose a video and frame, then **Show suggestions** (or **Refresh suggestions**). Raw AI proposals, **Assistant visual audit · still needs your confirmation**, and volunteer reviews appear separately. Assistant audits are bound to the original frame hash and source; their accepted or redrawn boxes remain unverified. Research-only audit rows are not offered here. **Copy as draft** adds an editable, unlinked box, preserves existing boxes, and clears the completeness checkbox. It never saves a grade or assumes that the same box in another video frame is the same mushroom.

Inspect the original image, correct/add/remove boxes, choose broad groups where justified, and link individual tracks only when independently established. For positive examples, check **I boxed every visible mushroom** only after checking the whole image. For negatives, explicitly choose **None visible**; leave uncertain images uncertain. Enter the curator name and explicitly save. Enable **Allow these reviewed labels to train future detectors** only where the source and your reviewed labels permit training. These local saves, stored in `data/video-grades/video-grades.sqlite`, are the human authority; a volunteer's pending review or an assistant's completeness claim is not.

Assistance is recorded even when suggestions are only viewed. The current UI conservatively cites all displayed volunteer review IDs, rather than distinguishing viewed from copied boxes. Dataset release excludes the entire cited frame unless **both the originally cited revision and the latest imported revision for each contributor** grant training consent and match the same frame, source, and SHA-256. Missing evidence, conflicting history, or a later revocation also excludes it. Frozen dataset records retain original review IDs/revisions, contributor pseudonyms, source/hash lineage, and the current consent revision. Rechecking before training or activation invalidates an older release after such changes. This does not retroactively erase an already trained checkpoint; import current permissions before preparing a new release.

For a manually requested training job, open `http://127.0.0.1:8787/live.html` and use **Prepare reviewed dataset**, inspect its readiness and manifest, then **Train a candidate model**. An equivalent local API sequence is:

```powershell
$stationUrl = 'http://127.0.0.1:8787'
$stationHeaders = @{}
if ($env:MYCOSCOUT_TOKEN) { $stationHeaders.Authorization = 'Bearer ' + $env:MYCOSCOUT_TOKEN }
$dataset = Invoke-RestMethod -Method Post -Uri "$stationUrl/api/learning/export" -Headers $stationHeaders -ContentType 'application/json' -Body '{}'
$dataset
# Start only after reviewing the release and its readiness; inadequate datasets are rejected.
$trainingRequest = @{ datasetId = $dataset.datasetId; epochs = 3; learningRate = 0.005; scoreThreshold = 0.3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri "$stationUrl/api/learning/jobs" -Headers $stationHeaders -ContentType 'application/json' -Body $trainingRequest
```

The strict release uses complete, explicitly approved local human grades and approved testbench examples, checks source permissions, freezes pixels and provenance, and keeps video/source groups and near duplicates together. Research-only media are excluded. Main-collection evaluation-only frames may remain in held-out tests but cannot enter training or validation. A small set of examples may fail readiness; the release report explains missing independent examples. Training changes model weights; this pipeline trains its candidate detector, not the raw proposal model. Collecting unreviewed contributions or exporting a dataset alone starts no training. An enabled automatic worker can start a candidate job after the approved-data thresholds and release readiness checks pass.

Review the completed candidate's held-out metrics before explicitly choosing **Use experimental model** with a reviewer identity. Activation rechecks the frozen release and checkpoint digest, updates `data/learning/active-model.json`, and takes effect for a newly started camera session. No AI proposal, volunteer vote, assistant audit, or completed training job activates itself. A passed test is not a field-performance or flight-safety certification.

## Automatic candidate training on the local vision PC

The private station console is `http://127.0.0.1:8787/learning-loop.html`. It shows approved-data counts, readiness or rejection reasons, synchronization state, the current job and candidate comparisons. Its controls use the station's existing token and origin checks. The public Gleaner service has no training, export, release-test or activation endpoint.

Enable community synchronization and automatic training in the console when the local PC is ready to run the worker. While the station runs, synchronization checks every five minutes and repeats immediately before dataset export, job start and comparison. A failed required sync leaves the cycle waiting for current permissions. The bridge imports private SSH exports and publishes only a sanitized progress summary about once a minute. The public page treats a summary older than fifteen minutes as stale; it does not infer that training continues while the PC is offline.

The default trigger is **50 newly approved unique frames from at least 3 independent source sessions**, with **at most 1 training attempt per rolling 24 hours**. Duplicate image hashes count once; many frames from one video do not become independent sessions. Existing eligible, unconsumed labels count at bootstrap. Previously reserved validation/test groups and research-only sources do not count toward new training evidence. Passing this trigger still requires a sufficiently independent train/validation/test release; a rejected release retains its reasons rather than repeatedly exporting unchanged evidence.

Cycles persist configuration, consumed evidence, stage, job identity, daily reservations and bounded retry history under `data/learning/automatic`. Restart recovery reuses completed exports and accepted jobs by idempotency key. Only one learning operation can run at a time. **Pause** prevents the next stage but does not kill an already running job; **Cancel** cancels the current cycle. Failed or cancelled evidence is not relaunched indefinitely without a meaningful evidence or recipe change.

Automatic jobs use `validationOnly: true`: they train a candidate and compare it with the active reviewed detector, or the installed proposal baseline when there is no active detector. Daily cycles do not inspect the final test partition. A candidate needs a passing validation comparison against the current baseline before **Run final release test** becomes useful. This explicit action checks the frozen test partition. After reviewing a passing release result, a named reviewer may select **Use reviewed candidate**. Activation verifies current permissions, the checkpoint, and the saved release proof while holding the worker lock; it does not rewrite the immutable model card. Training and comparison never promote a model automatically.

The provided Windows scripts support a current-user logon worker. Installation is a separate action; the presence of these scripts does not mean a task is installed:

```powershell
# Register and start the current-user logon task.
powershell.exe -NoProfile -File scripts/install-learning-worker.ps1
# Inspect the installed task, if present.
Get-ScheduledTask -TaskName 'MycoScout-Gleaner-LearningWorker'
# Alternatively run the supervisor directly in a dedicated shell.
powershell.exe -NoProfile -File scripts/start-learning-worker.ps1
```

The supervisor uses the existing local Node/Python installation, detects an existing station, and restarts a station it launches after exit. It keeps logs under `data/learning/worker-stdout.log` and `worker-stderr.log`; it does not train on the public VPS. The user must be signed in and the PC awake, with network access for synchronization. Keep the station's private credential configuration intact. Pause the loop in the console to stop new learning work. The `data/learning/worker-stop` sentinel stops future supervisor launches, but does not kill an already running station.

## Keep inference, audit, and learning distinct

After manual activation, the background worker refreshes public AI suggestions with the selected reviewed detector. It uses the packaged public collection only, checks current permissions and frozen model/source hashes, caches exact image/model results, and publishes only a complete pass through the existing SSH publisher. The local console reports its progress. A crash resumes the saved request; a failed pass keeps the previous published suggestions. The refresh never turns proposals into approved labels. No active reviewed model means the original suggestions remain in place.

The private loop status is `GET /api/learning/loop`; public aggregate status is `GET /mushroom-lab/api/training-status`. Public progress contains no reviewer identities, review content, local paths, model file paths, or administrative controls. Disabled community synchronization blocks reuse of imported review permissions until a fresh sync succeeds.

- **Inference:** `vision.video_ai_pass` runs the existing model and saves unverified boxes. It does not change model weights. Its default inputs include a separate research collection; those results stay local and must not be added to the public package or strict training release.
- **Visual audit:** `data/ai-video-review-v1/assistant-audit.json` records assistant acceptance/rejection/redrawing against exact frame hashes. The curator can inspect these as suggestions; `humanVerified` stays false.
- **Evaluation:** the commands below freeze already saved human labels and compare proposals against them. They neither change those labels nor train a model. Output files must be new filenames.
- **Training:** manual jobs or the enabled automatic coordinator change candidate weights using strict, permitted human labels. Automatic jobs evaluate validation data only; the release test remains an explicit action. Experimental assistant-label pilot jobs remain `activationEligible: false`; they are not a substitute for curated labels.
- **Activation:** a separate explicit reviewer action changes the station's selected model only after release checks. A comparison result or completed training job cannot activate itself.

```powershell
$auditStamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$auditSnapshot = 'data/human-video-evaluation-' + $auditStamp + '.json'
$auditReport = 'data/human-video-evaluation-' + $auditStamp + '-report.json'
.\runtime-vision\Scripts\python.exe -m vision.evaluate_video_ai snapshot --station http://127.0.0.1:8787 --output $auditSnapshot
.\runtime-vision\Scripts\python.exe -m vision.evaluate_video_ai evaluate --snapshot $auditSnapshot --proposals data/ai-video-review-v1 --output $auditReport
```

The evaluation snapshot helper currently reads an unprotected loopback station; use the local UI/export workflow if the station requires a bearer token. Never remove station authentication to make a helper work.

## Rollback

Use the exact backup path reported by publication or `status`:

```powershell
.\runtime-vision\Scripts\python.exe deploy/community/deploy.py rollback --release community-20261029-v1 --backup /root/gleaner-mushroom-lab/backups/REPORTED-BACKUP
```

Rollback restores the preceding service and Caddy configuration only if no unrelated proxy edit has intervened. It preserves the live contributor database; the database snapshot exists for manual recovery, not automatic replacement of newer reviews. On a first deployment, rollback removes only this managed service and route. Original releases and backups are retained.

The homepage's shared navigation link has a separate narrowly scoped helper and backup: `site-link.py add` inserts the existing nav-array item, and `site-link.py rollback` restores its saved file only if no later navigation edit has occurred. Remove that link before rolling back the first service deployment to avoid a dead link.

Source-specific attribution, source URLs, license URLs and extraction changes must remain visible. CC BY-SA source frames retain that license. Private and noncommercial research collections must not be added to the public collection.
