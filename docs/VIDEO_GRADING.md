# Human video grading feed

Open **http://127.0.0.1:8787/video-grader.html** on the station. The Vision Workbench links to it as **Grade video frames**.

## Review a collection

1. Enter your reviewer name. Choose mixed videos, video/time order, or lower whole-frame sharpness first.
2. Rate each frame **Mushrooms visible**, **None visible**, or **Uncertain**. Each grade saves before the feed advances. Turn off automatic advancement to remain on a frame.
3. A quick visibility grade does not require boxes. To measure detection, draw a tight box around every visible mushroom and check **I boxed every visible mushroom**. **None visible** records an explicitly reviewed empty frame. **Uncertain** never becomes a negative example.
4. For the same physical mushroom across time, create a named track and select it before drawing. Each saved box records the original pixel coordinates, video identity, timestamp, and track. A track cannot be attached to another video, even if both videos belong to one recording session.
5. Use **Earlier frame**, **Later frame**, or the timeline to work backward from a clear view. The earlier frame's status and box remain unconfirmed until you review it. Boxes and identities are never copied automatically.
6. **Test detector on this frame** opens a real original/color/tile comparison in the workbench. Return to the grading feed and refresh after the comparison completes.

The pixel-width table joins current, complete human annotations with completed tests of that exact frame. Each model, method and settings combination is separate. The latest repeated test per frame/configuration is counted once. One-to-one matches require IoU of at least 0.5; recall is matched human boxes / human boxes. No denominator means no percentage. Reports include source-group counts and original versus working-image pixel widths. These correlated video observations do not establish independent field accuracy or physical camera distance.

## Provenance and storage

The extractor installs `data/video-discovery-v1/manifest.json` with decoded frames, original-video hashes, timestamps, authors, licenses and recording-session groups. The gallery and grader serve only verified images within that collection. Changing a previously graded frame's identity or geometry is rejected.

Grades and tracks are stored separately in `data/video-grades/video-grades.sqlite`. Reviews have immutable history and optimistic revision checks. Shuffling changes display order, never frame IDs, timestamps or grouping. Human grades are separate from workbench annotations; no background model training or automatic annotation transfer occurs. The starter research collection is evaluation-only.

**View grading export** shows a copyable JSON snapshot of ratings, manually linked boxes, source-video context and detector comparisons. It is an audit/review export, not permission to train. The workbench's existing approved-data COCO/YOLO export remains a separate explicit workflow.

## Collect more footage

See [Video extraction](VIDEO_FRAMES.md) for local videos. `vision/acquire_videos.py` accepts a verified Commons metadata manifest and selected page IDs, caps downloads, checks published size/SHA1, and retains attribution. It stops on HTTP 429/503 and records the host's retry time. It does not bypass login, download controls, or rate limits.

Keep clips from the same outing/scene in a shared source group, and preserve that group across future training, validation and testing partitions. The extractor samples sharpness strata rather than discarding blurry frames; whole-frame sharpness is only a review ordering cue. Leaf texture, lighting and compression affect it. Review the mushroom itself before deciding whether it is blurred, small or absent.

Use actual moving-camera footage and retain originals. Timelapses, slideshows, synthetic degradation and source-title species hints must not be presented as measured flight-distance evidence. No verified black-trumpet video was found in the initial bounded source search.
