# Full pixel regression, V5

The [published comparison](https://thegreatlakesgleaner.press/mushroom-lab/testing-pixels.html)
contains 101 cases and 505 unique saved model passes. It compares unchanged
GroundingDINO weights on native images and seven target-width conditions. This
is a development regression using previously inspected sources, not independent
validation, a training run or a measured drone-distance trial.

## Results

A match requires intersection-over-union of at least 0.5 with a frozen selected
reference region. Each region matches at most one prediction. The score threshold
is 0.25. The three human targets come from one video; the nine AI photo references
are approximate, incomplete and include both individual mushrooms and clusters.
Keep these authorities separate.

| Requested target width | Illustrative range for a 5 cm target | Human video, whole frame → tiled | AI photo references, whole frame → tiled |
| --- | ---: | ---: | ---: |
| Native | Not measured | 2/3 → 2/3 | 9/9 → 9/9 |
| 64 px | — | 3/3 → 3/3 | 9/9 → 9/9 |
| 32 px | — | 2/3 → 3/3 | 5/9 → 9/9 |
| 16 px | — | 0/3 → 1/3 | 0/9 → 3/9 |
| 6.9 px | 10 m | 0/3 → 0/3 | 0/9 → 0/9 |
| 3.45 px | 20 m | 0/3 → 0/3 | 0/9 → 0/9 |
| 1.38 px | 50 m | 0/3 → 0/3 | 0/9 → 0/9 |
| 0.69 px | 100 m | 0/3 → 0/3 | 0/9 → 0/9 |

The range examples use `width_pixels = 1380 × 0.05 / range_metres` from the
illustrative, uncalibrated camera configuration. Physical organism widths, lens
calibration and camera-to-target ranges were not measured. A larger cluster
would span more pixels at the same range; these sources do not establish how
often the system would find it in flight.

Tiling also increased alerts on native negative controls. The one human-labelled
negative produced one whole-frame alert and six tiled alerts. Four separate
AI absence references produced eight and 22 alerts respectively. The controls
were tested only at native scale. No distance-dependent false-positive rate was
measured. Additional boxes on incompletely annotated positive photographs are
unmatched alerts, not established false positives.

## What the model received

Each of the 12 positive sources has a native case and seven synthetic cases.
Five native-only negatives bring the total to 101. The 84 saved synthetic PNGs
shrink the whole source scene onto a 1920 × 1080 RGB127 gray canvas. They never
enlarge the target back to its original footprint. Prespecified positions are
derived from hashes; realized width, height, resampling scales and tile coverage
are recorded. Sixteen-times fractional coverage is integrated into the final
saved pixels; it creates no new visual detail.

The tiled method adds four overlapping tiles to the same whole-frame context
pass. Sharing that context gives five unique model passes per case, rather than
six. The model processor resizes inputs toward an 800-pixel short edge with a
1,333-pixel long-edge limit; the report records the resulting target footprints.
A target below one pixel in width can still exceed one pixel in height.

Gray padding, scene boundaries, synthetic scaling and changed scene context do
not model real woodland clutter. An independent inspection found alerts on some
tiles containing only uniform gray padding. Those alerts remain in the results;
none were removed or rescored after inspection. Native and synthetic results are
therefore not expected to form a smooth monotonic accuracy curve.

The earlier [round-four experiment](https://thegreatlakesgleaner.press/mushroom-lab/testing-round4.html)
reduced detail and then enlarged the whole image again. Its “32px” and “16px”
labels describe retained detail, not the small footprint tested here. Do not
combine the two curves.

## Saved evidence and recovery

The original run stopped after saving 92 cases because of a Windows file lock.
The failed report and saved predictions were preserved. The explicit recovery
copied verified evidence into `experiments/results/full-pixel-v5/recovery-01`
and ran only the nine missing frozen cases. The completed report is
`experiments/results/full-pixel-v5/recovery-01/report.json`.

There are **505 unique persisted planned model passes**. Total attempted passes
are unknown because the interrupted executor may have completed unrecorded work.
Reported wall time includes the interruption and recovery. Two CPU workers used
four Torch threads each; measured latency includes contention and is not a phone,
Quest, Raspberry Pi or onboard drone benchmark. Tiled timing includes its shared
whole-frame pass and must not be added to whole-frame timing.

The independent verifier recounts matches, extra alerts, geometry, native-only
controls and all prediction hashes before export. Its receipt establishes saved
evidence consistency, not independent human truth. Protocol SHA-256:
`fa4e092b6118fc3888c6cae89e4e5ff539ae390f6321fe0529427562fe08d3ef`.
Completed report SHA-256:
`03748dcbd6a960024c395a3ba94aad25ea10e214efa496fcbe8a5d98f3cf7e04`.

The public [result JSON](https://thegreatlakesgleaner.press/mushroom-lab/testing-pixels.json)
contains the comparison, credited source metadata, boxes and hashes. The inspector
verifies saved image bytes and dimensions before drawing overlays, supports pixel
zoom and links back to original-image grading. Synthetic test views never enter
the human grading or training collection.

## Source and reproduction limits

The source package includes the evaluator, verifier, interruption recovery,
public exporter and viewer. Downloaded images, model weights, private review
ledgers and experiment output directories are excluded. A fresh source checkout
does not contain the inputs needed to reproduce this historical run, and its
community viewer cannot show the saved evidence without the separately licensed
images. Use the published inspector to review this run.

With the exact local artifacts and dependencies available, the completed recovery
can be independently recounted into a new receipt without running inference:

```powershell
python experiments/verify_full_pixel_v5.py --output experiments/results/full-pixel-v5/recovery-01 --receipt experiments/results/full-pixel-v5/recovery-01/recheck.json
```

The receipt path must be new. The recovery wrapper is deliberately specific to
this interrupted run, not a general silent retry mechanism. Focused contract
tests exercise scoring, evidence verification, recovery and publication with
fixtures; they do not load a detector or substitute for the recorded experiment:

```powershell
python -m unittest experiments.test_full_pixel_v5 experiments.test_verify_full_pixel_v5 experiments.test_recover_full_pixel_v5 experiments.test_publish_full_pixel_v5
node --test --test-isolation=none tests/community-testing-pixels.test.mjs
```

No weights, source labels, active model selection or aircraft commands changed.
