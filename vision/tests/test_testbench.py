import base64
import io
import json
import unittest
from unittest.mock import patch

from PIL import Image
from vision.testbench import (RequestError, MAX_CANDIDATES, bounded_nms, compare, decode_image,
                              pass_count, projected_candidates, settings_for, tile_boxes,
                              region_tile_boxes, tile_region_bounds)
from vision.testbench_worker import serve


SHA = "a" * 64


def encoded_photo(size=(100, 80), format="PNG", exif=None):
    image = Image.new("RGB", size, (40, 70, 90))
    stream = io.BytesIO()
    image.save(stream, format=format, **({"exif": exif} if exif is not None else {}))
    return base64.b64encode(stream.getvalue()).decode("ascii")


class FakeDetector:
    info = {"model": "software fixture", "resolvedRevision": "test", "device": "cpu", "threads": 1,
            "prompt": "a mushroom. a fungus.", "boxThreshold": .999}

    def __init__(self, responses=None):
        self.responses = responses
        self.calls = []
        self.box_threshold = .999
        self.text_threshold = .999

    def predict(self, image):
        self.calls.append({"size": image.size, "threshold": self.box_threshold,
                           "textThreshold": self.text_threshold, "pixels": image.tobytes()})
        if self.responses is not None:
            return self.responses[len(self.calls) - 1]
        return [{"bboxPixelsXYXY": [0, 0, image.width, image.height], "confidence": .5,
                 "species": "must be stripped", "measuredRangeM": 10}]


class TestbenchTests(unittest.TestCase):
    def test_tile_grid_covers_odd_sized_edges_with_overlap(self):
        for grid in (2, 3):
            boxes = tile_boxes(101, 79, grid)
            self.assertEqual(len(boxes), grid * grid)
            self.assertEqual(min(b[0] for b in boxes), 0)
            self.assertEqual(max(b[2] for b in boxes), 101)
            self.assertEqual(max(b[3] for b in boxes), 79)
            for y in range(79):
                for x in range(101):
                    self.assertTrue(any(b[0] <= x < b[2] and b[1] <= y < b[3] for b in boxes))
            top_row = [b for b in boxes if b[1] == 0]
            self.assertGreater(top_row[0][2], top_row[1][0])

    def test_tiny_grids_deduplicate_and_preserve_coverage(self):
        boxes = tile_boxes(2, 2, 3)
        self.assertEqual(len(boxes), 4)
        self.assertEqual(len(set(boxes)), 4)

    def test_optional_region_preserves_legacy_settings_and_full_grid(self):
        settings = settings_for({"methods": ["tiles"]})
        self.assertNotIn("tileRegion", settings)
        self.assertEqual(region_tile_boxes(101, 79, 2), tile_boxes(101, 79, 2))
        self.assertEqual(region_tile_boxes(101, 79, 2, [0, 0, 1, 1]), tile_boxes(101, 79, 2))

    def test_region_rounding_and_odd_grid_coverage(self):
        region = [.1, .35, .9, .95]
        self.assertEqual(tile_region_bounds(101, 79, region), (10, 27, 91, 76))
        boxes = region_tile_boxes(101, 79, 2, region)
        self.assertEqual(len(boxes), 4)
        for y in range(27, 76):
            for x in range(10, 91):
                self.assertTrue(any(x1 <= x < x2 and y1 <= y < y2 for x1, y1, x2, y2 in boxes))
        self.assertTrue(all(10 <= x1 < x2 <= 91 and 27 <= y1 < y2 <= 76 for x1, y1, x2, y2 in boxes))

    def test_region_keeps_context_and_projects_boundary_duplicates_once(self):
        image = Image.new("RGB", (100, 80))
        region = [0, .35, 1, 1]
        boxes = region_tile_boxes(100, 80, 2, region)
        truth = [45, 48, 52, 55]  # Visible in all four overlapping tiles.
        outside = {"bboxPixelsXYXY": [4, 4, 12, 12], "confidence": .7}
        responses = [[outside, {"bboxPixelsXYXY": truth, "confidence": .6}]]
        for left, top, _, _ in boxes:
            responses.append([{"bboxPixelsXYXY": [truth[0]-left, truth[1]-top, truth[2]-left, truth[3]-top], "confidence": .8}])
        detector = FakeDetector(responses)
        progress = []
        result = compare(image, SHA, {"methods": ["tiles"], "tileRegion": region}, detector, progress.append)
        method = result["methods"][0]
        self.assertEqual(result["modelPasses"], 5)
        self.assertEqual(detector.calls[0]["size"], image.size)
        self.assertEqual([call["size"] for call in detector.calls[1:]], [(b[2]-b[0], b[3]-b[1]) for b in boxes])
        self.assertEqual(sorted(row["bboxPixelsXYXY"] for row in method["candidates"]), sorted([truth, outside["bboxPixelsXYXY"]]))
        self.assertEqual(method["preNmsCandidateCount"], 6)
        self.assertTrue(method["fullFrameContext"])
        self.assertEqual(method["tileRegion"], region)
        self.assertEqual(method["tileBoundsPixelsXYXY"], [list(b) for b in boxes])
        self.assertEqual([row["kind"] for row in method["passTimings"]], ["context", "tile", "tile", "tile", "tile"])
        self.assertAlmostEqual(sum(row["seconds"] for row in method["passTimings"]), method["seconds"])
        self.assertEqual(progress[-1], {"stage": "complete", "completed": 5, "total": 5})

    def test_scaled_region_projects_to_original_and_does_not_crop_context(self):
        detector = FakeDetector(responses=[[], [{"bboxPixelsXYXY": [0, 0, 2, 3], "confidence": .7}], [], [], []])
        result = compare(Image.new("RGB", (101, 79)), SHA,
                         {"methods": ["tiles"], "tileRegion": [0, .35, 1, 1], "scale": .5}, detector)
        method = result["methods"][0]
        self.assertEqual(detector.calls[0]["size"], (50, 40))
        box = method["candidates"][0]["bboxPixelsXYXY"]
        for actual, expected in zip(box, [0, 14*79/40, 2*101/50, 17*79/40]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(method["passTimings"][0]["bboxPixelsXYXY"], [0, 0, 101, 79])
        self.assertAlmostEqual(method["tileBoundsPixelsXYXY"][0][1], 14*79/40)

    def test_tile_translation_and_actual_scaled_dimensions(self):
        rows = projected_candidates([{"bboxPixelsXYXY": [1, 2, 10, 15], "confidence": .7}],
                                    (20, 10, 40, 30), (101, 79), (50, 40), .25, "tile_2")
        expected = [21 * 101 / 50, 12 * 79 / 40, 30 * 101 / 50, 25 * 79 / 40]
        for actual, wanted in zip(rows[0]["bboxPixelsXYXY"], expected):
            self.assertAlmostEqual(actual, wanted)

    def test_projection_discards_invalid_boxes_and_invented_metadata(self):
        rows = projected_candidates([
            {"bboxPixelsXYXY": [-5, -5, 120, 120], "confidence": .9, "species": "invented", "range": 3},
            {"bboxPixelsXYXY": [0, 0, 10, 10], "confidence": .1},
            {"bboxPixelsXYXY": [float("nan"), 0, 10, 10], "confidence": .9},
            {"bboxPixelsXYXY": [10, 10, 0, 0], "confidence": .9},
            {"bboxPixelsXYXY": [0, 0, 10, 10], "confidence": True}],
            (0, 0, 100, 80), (100, 80), (100, 80), .25, "original")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bboxPixelsXYXY"], [0, 0, 100, 80])
        self.assertEqual(set(rows[0]), {"bboxPixelsXYXY", "confidence", "sourcePass"})

    def test_nms_prefers_best_duplicate_and_caps_output(self):
        rows = [{"bboxPixelsXYXY": [x, y, x + 2, y + 2], "confidence": .4}
                for x in range(0, 60, 3) for y in range(0, 60, 3)]
        rows.append({"bboxPixelsXYXY": [0, 0, 2, 2], "confidence": .9})
        result = bounded_nms(rows, "tiles")
        self.assertEqual(len(result), MAX_CANDIDATES)
        self.assertEqual(result[0]["confidence"], .9)
        self.assertEqual(sum(r["bboxPixelsXYXY"] == [0, 0, 2, 2] for r in result), 1)

    def test_methods_remain_independent_and_full_frame_is_reused(self):
        detector = FakeDetector()
        progress = []
        result = compare(Image.new("RGB", (101, 79)), SHA,
                         {"methods": ["original", "color", "tiles"], "tileGrid": 2}, detector, progress.append)
        self.assertEqual(len(detector.calls), 6)
        self.assertEqual(result["modelPasses"], 6)
        methods = {m["id"]: m for m in result["methods"]}
        self.assertEqual(methods["original"]["passes"], 1)
        self.assertEqual(methods["color"]["passes"], 1)
        self.assertEqual(methods["tiles"]["passes"], 5)
        self.assertEqual(methods["tiles"]["reusedPasses"], 1)
        self.assertEqual(progress[-1], {"stage": "complete", "completed": 6, "total": 6})
        self.assertEqual(sorted(p["completed"] for p in progress), [p["completed"] for p in progress])
        self.assertIsNone(result["actualDistanceM"])
        self.assertFalse(result["actuationEnabled"])

    def test_request_threshold_resets_and_does_not_modify_original_pixels(self):
        detector = FakeDetector()
        image = Image.new("RGB", (101, 79), (60, 20, 70))
        original = image.tobytes()
        first = compare(image, SHA, {"methods": ["original"], "boxThreshold": .8, "scale": .5}, detector)
        second = compare(image, SHA, {"methods": ["original"]}, detector)
        self.assertEqual([c["threshold"] for c in detector.calls], [.8, .25])
        self.assertTrue(all(c["textThreshold"] == .25 for c in detector.calls))
        self.assertEqual(first["methods"][0]["candidates"], [])
        self.assertEqual(second["model"]["boxThreshold"], .25)
        self.assertEqual(image.tobytes(), original)
        self.assertEqual(first["image"]["workingWidth"], 50)
        self.assertEqual(first["image"]["workingHeight"], 40)

    def test_scaled_full_frame_box_projects_back_to_original(self):
        result = compare(Image.new("RGB", (101, 79)), SHA, {"methods": ["original"], "scale": .125}, FakeDetector())
        self.assertEqual(result["methods"][0]["candidates"][0]["bboxPixelsXYXY"], [0, 0, 101, 79])

    def test_only_requested_modes_and_color_cues_are_run(self):
        with patch("vision.testbench.proposals", return_value=[]) as cues:
            result = compare(Image.new("RGB", (100, 80)), SHA, {"methods": ["tiles"]}, FakeDetector())
            cues.assert_not_called()
            self.assertEqual([m["id"] for m in result["methods"]], ["tiles"])
            self.assertEqual(result["methods"][0]["reusedPasses"], 0)
            self.assertEqual(result["colorCues"], [])
        with patch("vision.testbench.proposals", return_value=[]) as cues:
            result = compare(Image.new("RGB", (100, 80)), SHA, {"methods": ["color"]}, FakeDetector())
            cues.assert_called_once()
            self.assertEqual(result["modelPasses"], 1)

    def test_color_cues_are_separate_and_project_to_original(self):
        cue = {"bboxPixelsXYXY": [1, 2, 4, 5], "pointPixelsXY": [2, 3], "contrastScore": .7,
               "colorEvidence": {"scoreMeaning": "not probability"}}
        with patch("vision.testbench.proposals", return_value=[cue]):
            result = compare(Image.new("RGB", (100, 80)), SHA, {"methods": ["color"], "scale": .5}, FakeDetector())
        self.assertEqual(result["colorCues"][0]["bboxPixelsXYXY"], [2, 4, 8, 10])
        self.assertTrue(result["colorCues"][0]["colorOnly"])
        self.assertNotIn("confidence", result["colorCues"][0])
        self.assertIsNone(result["colorCues"][0]["species"])

    def test_bad_settings_rejected(self):
        for value in ([], {"methods": []}, {"methods": ["original", "original"]}, {"methods": ["unknown"]},
                      {"tileGrid": True}, {"tileGrid": 4}, {"scale": True}, {"scale": .2},
                      {"boxThreshold": float("nan")}, {"boxThreshold": .9}, {"model": "remote"}):
            with self.subTest(value=value), self.assertRaises(RequestError):
                settings_for(value)

    def test_bad_regions_rejected(self):
        for region in (None, "0,0,1,1", [], [0, 0, 1], [0, 0, 1, 1, 1],
                       [-.01, 0, 1, 1], [0, 0, 1.01, 1], [1, 0, 0, 1], [0, .5, 1, .5],
                       [0, 0, True, 1], [0, float("nan"), 1, 1], [0, 0, float("inf"), 1]):
            with self.subTest(region=region), self.assertRaises(RequestError):
                settings_for({"methods": ["tiles"], "tileRegion": region})
        with self.assertRaises(RequestError):
            settings_for({"methods": ["original"], "tileRegion": [0, 0, 1, 1]})

    def test_decode_applies_exif_and_checks_still_mime(self):
        exif = Image.Exif(); exif[274] = 6
        image, checksum = decode_image(encoded_photo((64, 32), "JPEG", exif), "image/jpeg")
        self.assertEqual(image.size, (32, 64))
        self.assertEqual(len(checksum), 64)
        with self.assertRaises(RequestError):
            decode_image(encoded_photo(), "image/jpeg")
        stream = io.BytesIO()
        Image.new("RGB", (32, 32)).save(stream, format="PNG", save_all=True,
                                       append_images=[Image.new("RGB", (32, 32), "white")], duration=20)
        with self.assertRaises(RequestError):
            decode_image(base64.b64encode(stream.getvalue()).decode(), "image/png")

    def test_decode_rejects_bytes_size_and_dimension_limits(self):
        for data in ("!not-base64", "", "A" * (12 * 1024 * 1024)):
            with self.assertRaises(RequestError):
                decode_image(data, "image/png")
        with self.assertRaises(RequestError):
            decode_image(encoded_photo((10, 20)), "image/png")
        with patch("vision.testbench.MAX_IMAGE_PIXELS", 100):
            with self.assertRaises(RequestError):
                decode_image(encoded_photo((20, 20)), "image/png")

    def test_too_many_predictor_candidates_fails(self):
        detector = FakeDetector(responses=[[{}] * 1001])
        with self.assertRaises(RuntimeError):
            compare(Image.new("RGB", (100, 80)), SHA, {}, detector)

    def test_worker_recovers_errors_and_reuses_one_warm_model(self):
        good = {"id": "one", "imageBase64": encoded_photo(), "mimeType": "image/png", "settings": {"boxThreshold": .8}}
        second = {**good, "id": "two", "settings": {}}
        lines = b"not json\n" + json.dumps(good).encode() + b"\n" + json.dumps(second).encode() + b"\n"
        output = io.StringIO()
        detector = FakeDetector()
        with patch("sys.stderr", new=io.StringIO()), patch("builtins.print"):
            factory_calls = []
            serve(io.BytesIO(lines), output, lambda: (factory_calls.append(1) or detector))
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0]["status"], "error")
        self.assertEqual([m["id"] for m in messages if m["status"] == "ok"], ["one", "two"])
        self.assertEqual(len(factory_calls), 1)
        self.assertEqual([c["threshold"] for c in detector.calls], [.8, .25])

    def test_worker_sanitizes_model_errors_and_bounds_input_lines(self):
        good = {"id": "safe", "imageBase64": encoded_photo(), "mimeType": "image/png"}
        output = io.StringIO()
        def failed():
            raise OSError("secret filesystem path and token")
        serve(io.BytesIO(json.dumps(good).encode() + b"\n"), output, failed)
        self.assertNotIn("secret", output.getvalue())
        self.assertEqual(json.loads(output.getvalue().splitlines()[-1])["status"], "error")
        output = io.StringIO()
        with patch("vision.testbench_worker.MAX_REQUEST_LINE", 32):
            serve(io.BytesIO(b"x" * 100 + b"\n"), output, failed)
        self.assertEqual(len(output.getvalue().splitlines()), 1)
        self.assertEqual(json.loads(output.getvalue())["status"], "error")


if __name__ == "__main__":
    unittest.main()
