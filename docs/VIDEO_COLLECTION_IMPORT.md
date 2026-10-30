# Append video footage to the human grading feed

Extract each new acquisition into a fresh collection using `adapters.video_frames`.
Then validate the append before publishing it:

```powershell
.\runtime-vision\Scripts\python.exe -m adapters.merge_video_collections --destination data/video-discovery-v1 --incoming data/morels-new-collection
.\runtime-vision\Scripts\python.exe -m adapters.merge_video_collections --destination data/video-discovery-v1 --incoming data/morels-new-collection --apply
```

Repeat `--incoming` to append several extraction batches in one operation. Without
`--apply`, no files change. After applying, click **Refresh collection** in the
video grader. The station reads the manifest on each request, so a restart is not
needed. Keep the destination collection directory and collection ID unchanged:
existing human grades and track links depend on those identities.

The importer preserves all existing source/frame records, timestamps, hashes,
grouping, rights flags, settings and image bytes. It never opens or modifies the
grade database. It copies original extracted images without resizing or
enhancement, saves a byte-for-byte backup of the previous manifest, and replaces
the manifest atomically only after validation and complete image copies.
Source videos and incoming collections remain in place. The old contact sheet
remains an archive of its original batch; use the grader for the combined feed.

An exact retry is a no-op. If publication is interrupted, the old manifest remains
usable; already copied images can be reused on retry after their hashes match.
A surviving `.video-merge.lock` requires checking that its recorded process has
stopped before removing that exact lock. The importer does not remove an active
lock or delete earlier media.

Conflicts stop the import: changed source or frame metadata under an existing ID,
the same video under another identity, conflicting permissions within a source
group, or identical image bytes/pixels assigned to different groups. Related
clips can intentionally share a group when their permission/quarantine settings
agree. Identical images at different times within one group are reported, and
their distinct timestamps are retained. Training still requires complete human
annotations and permission; importing footage does not grant either. No source
title or filename becomes a mushroom ground-truth label.

Current capacity: **200 sources, 2,000 frames, 8 MiB manifest, 8 MiB per image and
20 megapixels per image**. The gallery returns at most 24 images per request, while
the grading page receives the full bounded collection. The extractor creates
smaller batches: at most 12 source videos and 120 output frames total, with at most
60 frames per video. More footage can be retained outside the active collection
for later reviewed imports. Raising capacity requires coordinated changes to
the collection reader, grading snapshot, API and UI, not just the importer.

Validation:

```powershell
.\runtime-vision\Scripts\python.exe -m unittest adapters.tests.test_merge_video_collections -v
```
