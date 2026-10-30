"""Geometry checks for original-pixel view transforms and duplicate suppression."""
import unittest
from run_grounding_dino_batch import scan_views, to_original, class_agnostic_nms


class ViewTests(unittest.TestCase):
    def test_crops_cover_edges_without_escaping_image(self):
        views = scan_views(2048, 1459, 1024, 0.25)
        self.assertEqual(views[0], (0, 0, 2048, 1459))
        self.assertEqual(len(views), 7)
        self.assertIn((1024, 435, 2048, 1459), views)
        for x1, y1, x2, y2 in views:
            self.assertTrue(0 <= x1 < x2 <= 2048)
            self.assertTrue(0 <= y1 < y2 <= 1459)

    def test_single_narrow_dimension_and_small_image(self):
        self.assertEqual(scan_views(512, 400, 1024), [(0, 0, 512, 400)])
        views = scan_views(512, 2048, 1024)
        self.assertEqual(views[-1], (0, 1024, 512, 2048))

    def test_original_coordinates_are_translation_not_rescaling(self):
        candidate = {"bboxPixelsXYXY": [10, 20, 90, 120], "confidence": 0.8}
        result = to_original(candidate, (768, 435, 1792, 1459), 4)
        self.assertEqual(result["bboxPixelsXYXY"], [778, 455, 858, 555])
        self.assertEqual(result["sourceBboxPixelsXYXY"], [10, 20, 90, 120])
        self.assertEqual(candidate["bboxPixelsXYXY"], [10, 20, 90, 120])

    def test_nms_suppresses_across_phrases_but_preserves_separate_objects(self):
        boxes = [
            {"bboxPixelsXYXY": [0, 0, 100, 100], "confidence": 0.6, "modelLabel": "fungus"},
            {"bboxPixelsXYXY": [1, 1, 101, 101], "confidence": 0.9, "modelLabel": "mushroom"},
            {"bboxPixelsXYXY": [101, 0, 120, 20], "confidence": 0.4, "modelLabel": "mushroom"},
        ]
        self.assertEqual([item["confidence"] for item in class_agnostic_nms(boxes, 0.5)], [0.9, 0.4])
        self.assertEqual(len(boxes), 3)


if __name__ == "__main__":
    unittest.main()
