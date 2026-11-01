import copy
import unittest
from PIL import Image
from experiments.evaluate_full_pixel_v5 import (
    CANVAS, SS, LEVELS, SETTINGS, aggregate, fractional_canvas,
    pass_geometry, position_for, score, validate_references,
)
from vision.testbench import pass_count, settings_for


class FullPixelTests(unittest.TestCase):
    def test_fractional_region_is_not_rounded_up_to_one_pixel(self):
        source = Image.new("RGB", (200, 100), "red")
        result, box, geometry = fractional_canvas(source, [50, 30, 150, 70], .69, [.25, .75])
        self.assertEqual(result.size, CANVAS)
        self.assertEqual(geometry["supersampling"], 16)
        self.assertAlmostEqual(box[2] - box[0], .6875)
        self.assertGreater(box[3] - box[1], 0)
        self.assertLess(box[3] - box[1], 1)
        self.assertFalse(geometry["sceneReEnlarged"])
        self.assertIsNone(geometry["actualDistanceM"])

    def test_source_scaling_maps_full_scene_and_selected_region(self):
        source = Image.new("RGB", (200, 100), "red")
        _, box, geometry = fractional_canvas(source, [50, 10, 150, 90], 32, [.75, .25])
        sx, sy = geometry["sceneScaleXY"]
        bounds = geometry["sceneBoundsPixelsXYXY"]
        self.assertAlmostEqual(bounds[2] - bounds[0], 200 * sx)
        self.assertAlmostEqual(bounds[3] - bounds[1], 100 * sy)
        self.assertAlmostEqual(box[0], bounds[0] + 50 * sx)
        self.assertAlmostEqual(box[1], bounds[1] + 10 * sy)
        self.assertAlmostEqual(box[2] - box[0], 32.)

    def test_gray_padding_remains_exact_outside_whole_scene(self):
        source = Image.new("RGB", (200, 100), "red")
        result, _, geometry = fractional_canvas(source, [50, 10, 150, 90], 16, [.25, .25])
        self.assertEqual(result.getpixel((0, 0)), (127, 127, 127))
        self.assertEqual(result.getpixel((1919, 1079)), (127, 127, 127))
        bounds = geometry["sceneBoundsPixelsXYXY"]
        center = (round((bounds[0] + bounds[2]) / 2), round((bounds[1] + bounds[3]) / 2))
        self.assertEqual(result.getpixel(center), (255, 0, 0))

    def test_whole_patch_is_clamped_in_bounds_without_cropping(self):
        source = Image.new("RGB", (1200, 600), "green")
        _, _, geometry = fractional_canvas(source, [1000, 200, 1128, 400], 64, [.25, .25])
        left, top, right, bottom = geometry["sceneBoundsPixelsXYXY"]
        self.assertTrue(geometry["focalPositionClamped"])
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, CANVAS[0])
        self.assertLessEqual(bottom, CANVAS[1])
        self.assertAlmostEqual(right - left, 600.)

    def test_hash_position_is_fixed_and_on_predeclared_grid(self):
        first = position_for("inat-photo-753536713")
        self.assertEqual(first, position_for("inat-photo-753536713"))
        self.assertTrue(all(v in (.25, .5, .75) for v in first))

    def test_single_selected_target_matches_only_one_overlapping_prediction(self):
        target = [0., 0., 100., 100.]
        boxes = [{"bbox": target, "score": .8}, {"bbox": [0., 0., 90., 90.], "score": .9},
                 {"bbox": [150., 0., 200., 100.], "score": .7}]
        result = score(target, boxes, False, "ai")
        self.assertEqual(result["hits"], 1)
        self.assertEqual(result["matchedPredictionIndex"], 1)
        self.assertEqual(result["extraOverlappingAlerts"], 1)
        self.assertEqual(result["unmatchedPositiveBoxes"], 2)
        self.assertIsNone(result["humanCompletePositiveFalseAlerts"])

    def test_iou_point_five_is_applied_to_fractional_boxes(self):
        target = [10., 20., 10.5, 21.]
        hit = score(target, [{"bbox": [10., 20., 10.25, 21.], "score": .9}], False, "human")
        miss = score(target, [{"bbox": [10., 20., 10.24, 21.], "score": .9}], False, "human")
        self.assertEqual(hit["hits"], 1)
        self.assertEqual(miss["hits"], 0)

    def test_negative_authorities_stay_separate(self):
        boxes = [{"bbox": [0., 0., 10., 10.], "score": .7}]
        human = score(None, boxes, True, "human")
        ai = score(None, boxes, True, "ai")
        self.assertEqual(human["humanNegativeFalseAlerts"], 1)
        self.assertIsNone(ai["humanNegativeFalseAlerts"])
        self.assertEqual(ai["aiNegativeOpinionAlerts"], 1)
        self.assertIsNone(human["aiNegativeOpinionAlerts"])

    def test_raw_and_tiles_share_one_context_pass(self):
        settings = settings_for(SETTINGS)
        self.assertEqual(pass_count(CANVAS, settings), 5)
        self.assertEqual((12 * len(LEVELS) + 5) * 5, 505)

    def test_tile_coverage_distinguishes_center_from_corner_and_reports_processor_size(self):
        corner = pass_geometry(CANVAS, [470., 260., 490., 280.])
        center = pass_geometry(CANVAS, [950., 530., 970., 550.])
        self.assertEqual(corner["fullyCoveringTileCount"], 1)
        self.assertEqual(center["fullyCoveringTileCount"], 4)
        self.assertEqual((corner["passes"][0]["inputWidth"], corner["passes"][0]["inputHeight"]), (1333, 750))
        for item in corner["passes"]:
            if item["selectedTargetCoverageFraction"] == 0:
                self.assertEqual(item["selectedTargetInputWidthPixels"], 0.)
                self.assertEqual(item["selectedTargetInputHeightPixels"], 0.)

    def test_reference_authority_completeness_counts_and_identity_are_enforced(self):
        refs = []
        for authority, negative, count in (("human", False, 3), ("ai", False, 9), ("human", True, 1), ("ai", True, 4)):
            for _ in range(count):
                refs.append({"frameId": str(len(refs)), "referenceAuthority": authority,
                    "isNegativeControl": negative, "referenceComplete": authority == "human" or negative,
                    "originalTargetPixelsXYXY": None if negative else [0., 0., 100., 100.]})
        validate_references(refs)
        for index, key, value in ((0, "referenceComplete", False), (3, "referenceComplete", True),
                                  (12, "referenceComplete", False), (13, "referenceComplete", False),
                                  (0, "frameId", "1"), (0, "referenceAuthority", "ai"),
                                  (12, "originalTargetPixelsXYXY", [0., 0., 2., 2.])):
            modified = copy.deepcopy(refs); modified[index][key] = value
            with self.assertRaises(ValueError): validate_references(modified)

    def test_untested_negative_scales_are_null(self):
        metric = score([0., 0., 20., 20.], [], False, "ai")
        rows = [{"cohort": "ai-photo", "conditionId": "32px", "isNegativeControl": False,
                 "realizedTargetWidthPixels": 32., "realizedTargetHeightPixels": 20.,
                 "methods": {"raw": {"metrics": metric, "seconds": 1.}}}]
        item = aggregate(rows)[1]["conditions"][0]
        self.assertIsNone(item["negativeAlertCount"])
        self.assertIsNone(item["negativeFrameCount"])
        self.assertEqual(item["targetCount"], 1)

    def test_invalid_or_upscaling_synthetic_geometry_rejected(self):
        source = Image.new("RGB", (200, 100))
        for width in (0, -1, float("nan"), 101):
            with self.assertRaises(ValueError):
                fractional_canvas(source, [50, 10, 150, 90], width, [.25, .25])
        with self.assertRaises(ValueError):
            fractional_canvas(source, [50, 10, 150, 90], 32, [.2, .3])


if __name__ == "__main__": unittest.main()
