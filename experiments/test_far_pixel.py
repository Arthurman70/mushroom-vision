"""Scientific invariants for fractional-pixel probes and bounded color cues."""
import unittest

import numpy as np
from PIL import Image, ImageDraw
from vision.far_color import enhance, proposals, lab, color_evidence, contrast_map
from experiments.far_pixel_benchmark import fractional_scene, score, cue_score, quarantine_color_report


class FarPixelTests(unittest.TestCase):
    def test_fractional_target_retains_subpixel_area_and_dimensions(self):
        source = Image.new("RGB", (200, 200), "black")
        background = Image.new("RGB", (100, 100), "white")
        annotations = [{"bboxPixelsXYXY": [0, 0, 200, 200]}]
        result, boxes, transform = fractional_scene(background, source, annotations, .69, (.505, .505))
        self.assertEqual(result.size, background.size)
        self.assertLess(boxes[0][2] - boxes[0][0], 1)
        self.assertAlmostEqual(transform["realizedTargetWidthPixels"], .6875)
        a = np.asarray(result)
        self.assertGreater(a.min(), 0)
        self.assertLess(a.min(), 255)
        self.assertAlmostEqual(float((255 - a[..., 0]).sum()) / 255, .6875 ** 2, places=2)

    def test_original_rgb_unchanged(self):
        source = Image.new("RGB", (64, 64), (100, 100, 100))
        ImageDraw.Draw(source).rectangle((25, 25, 40, 40), fill=(102, 112, 100))
        before = source.tobytes()
        adjusted = enhance(source)
        self.assertEqual(source.tobytes(), before)
        self.assertNotEqual(adjusted.tobytes(), before)

    def test_lab_known_neutral_endpoints(self):
        neutral = lab(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8))
        self.assertAlmostEqual(float(neutral[0, 0, 0]), 0, places=4)
        self.assertAlmostEqual(float(neutral[0, 1, 0]), 100, places=3)
        self.assertLess(abs(float(neutral[0, 1, 1])), .01)

    def test_flat_field_has_no_color_cues(self):
        self.assertEqual(proposals(Image.new("RGB", (64, 64), (90, 80, 70))), [])

    def test_color_cues_are_bounded_and_never_species(self):
        rng = np.random.default_rng(7)
        im = Image.fromarray(rng.integers(0, 256, (128, 128, 3), dtype=np.uint8))
        cues = proposals(im, max_proposals=3)
        self.assertEqual(len(cues), 3)
        for cue in cues:
            self.assertIsNone(cue["species"])
            self.assertFalse(cue["poseKnown"])
            self.assertTrue(cue["colorOnly"])
            self.assertTrue(0 <= cue["contrastScore"] <= 1)
        with self.assertRaises(ValueError):
            proposals(im, 100)
        with self.assertRaises(ValueError):
            proposals(im, min_delta_e=float("nan"))

    def test_tiny_target_not_counted_from_whole_frame_box(self):
        metric = score([{"bboxPixelsXYXY": [0, 0, 100, 100]}], [[50, 50, 50.69, 50.69]])
        self.assertFalse(metric["focalHit"])
        self.assertEqual(metric["falsePositive"], 1)

    def test_color_nearby_cue_metric_is_separate_from_detection(self):
        cues = [{"pointPixelsXY": [51, 51], "bboxPixelsXYXY": [47, 47, 56, 56]}]
        box = [[50, 50, 50.69, 50.69]]
        self.assertTrue(cue_score(cues, box)["focalCueHit"])
        self.assertFalse(score(cues, box)["focalHit"])

    def test_color_evidence_clips_but_rejects_empty_boxes(self):
        delta = contrast_map(Image.new("RGB", (10, 10), "black"))
        self.assertEqual(color_evidence(delta, [-1, -1, 2, 2])["contrastScore"], 0)
        with self.assertRaises(ValueError):
            color_evidence(delta, [15, 15, 20, 20])

    def test_benchmark_color_reports_are_quarantined_without_mutating_source(self):
        payload = {"schema": "mycoscout.color-proposals.v1", "candidates": [{"contrastScore": .8}],
                   "image": {"sha256": "a" * 64}}
        wrapped = quarantine_color_report(payload, "b" * 64)
        self.assertIs(wrapped["synthetic"], True)
        self.assertIs(wrapped["evaluationOnly"], True)
        self.assertEqual(wrapped["derivedFromPredictionSha256"], "b" * 64)
        self.assertNotIn("synthetic", payload)
        wrapped["candidates"][0]["contrastScore"] = 0
        self.assertEqual(payload["candidates"][0]["contrastScore"], .8)

    def test_benchmark_color_quarantine_rejects_wrong_schema_or_digest(self):
        with self.assertRaises(ValueError):
            quarantine_color_report({"schema": "mycoscout.unlocated-candidates.v1"})
        with self.assertRaises(ValueError):
            quarantine_color_report({"schema": "mycoscout.color-proposals.v1"}, "not-a-digest")


if __name__ == "__main__":
    unittest.main()
