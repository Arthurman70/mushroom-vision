import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from adapters.live_capture import argument_parser, capture_scan_settings, main, scan_frame
from vision.testbench import region_tile_boxes


class Detector:
    info = {"model": "fixture", "resolvedRevision": "fixture", "boxThreshold": .25}

    def __init__(self, threshold=.25, responses=None):
        self.box_threshold = threshold
        self.text_threshold = .25
        self.responses = responses
        self.calls = []

    def predict(self, image):
        self.calls.append((image.size, self.box_threshold, self.text_threshold))
        return self.responses[len(self.calls)-1] if self.responses is not None else []


class LiveCaptureTests(unittest.TestCase):
    def test_cli_default_and_explicit_profile_contract(self):
        args = argument_parser().parse_args(["--input", "0", "--emit"])
        self.assertEqual(args.vision_profile, "hybrid")
        self.assertIsNone(args.tile_region)
        self.assertEqual(capture_scan_settings()["tileRegion"], [0, .35, 1, 1])
        self.assertEqual(capture_scan_settings("full-tiles")["tileRegion"], [0, 0, 1, 1])
        self.assertEqual(capture_scan_settings("fast")["methods"], ["original"])
        self.assertNotIn("tileRegion", capture_scan_settings("fast"))
        with self.assertRaises(ValueError):
            capture_scan_settings("fast", [0, 0, 1, 1])
        with self.assertRaises(ValueError):
            capture_scan_settings("auto-ground")

    def test_roi_override_validates_before_camera_or_model_is_opened(self):
        for region in ([0, .5, 1, .5], [-.1, 0, 1, 1], [0, 0, 1, float("nan")]):
            with self.subTest(region=region), self.assertRaises(ValueError):
                capture_scan_settings("ground-tiles", region)
        with patch("adapters.live_capture.LocalDetector") as factory, patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--input", "0", "--emit", "--vision-profile", "fast", "--tile-region", "0", "0", "1", "1"])
            factory.assert_not_called()

    def test_five_pass_scan_projects_and_deduplicates_at_roi_boundaries(self):
        truth, context_only = [45, 48, 52, 55], [5, 5, 12, 12]
        boxes = region_tile_boxes(100, 80, 2, [0, .35, 1, 1])
        responses = [[{"bboxPixelsXYXY": context_only, "confidence": .7}]]
        for left, top, _, _ in boxes:
            responses.append([{"bboxPixelsXYXY": [truth[0]-left, truth[1]-top, truth[2]-left, truth[3]-top], "confidence": .8}])
        detector = Detector(responses=responses)
        detections, scan = scan_frame(Image.new("RGB", (100, 80)), detector)
        self.assertEqual(len(detector.calls), 5)
        self.assertEqual(sorted(row["bboxPixelsXYXY"] for row in detections), sorted([truth, context_only]))
        self.assertEqual(scan["modelPasses"], 5)
        self.assertTrue(scan["fullFrameContext"])
        self.assertEqual(scan["tileBoundsPixelsXYXY"], [list(b) for b in boxes])
        self.assertEqual([p["kind"] for p in scan["passTimings"]], ["context", "tile", "tile", "tile", "tile"])
        self.assertEqual(scan["methods"][0]["passes"], 5)
        self.assertEqual(scan["preNmsCandidateCount"], 5)
        self.assertEqual(scan["candidateCount"], 2)
        self.assertIn("not semantic ground", scan["regionMeaning"])

    def test_full_and_fast_profiles_use_expected_views(self):
        image = Image.new("RGB", (100, 80))
        for profile, count, region in (("full-tiles", 5, [0, 0, 1, 1]), ("fast", 1, None)):
            detector = Detector()
            _, scan = scan_frame(image, detector, profile=profile)
            self.assertEqual(len(detector.calls), count)
            self.assertEqual(scan["tileRegion"], region)
            self.assertEqual(detector.calls[0][0], image.size)
        _, scan = scan_frame(image, Detector(), tile_region=[.1, .2, .9, .8])
        self.assertEqual(scan["tileRegion"], [.1, .2, .9, .8])

    def test_saved_threshold_is_not_overwritten_or_raised(self):
        for threshold in (.05, .3, .95):
            responses = [[{"bboxPixelsXYXY": [1, 1, 8, 8], "confidence": threshold}], [], [], [], []]
            detector = Detector(threshold=threshold, responses=responses)
            detections, scan = scan_frame(Image.new("RGB", (100, 80)), detector)
            self.assertEqual(len(detections), 1)
            self.assertEqual(scan["settings"]["boxThreshold"], threshold)
            self.assertEqual(detector.box_threshold, threshold)
            self.assertEqual([call[1] for call in detector.calls], [threshold]*5)
        detector = Detector(responses=[[{"bboxPixelsXYXY": [1, 1, 8, 8], "confidence": .2}], [], [], [], []])
        del detector.box_threshold
        detector.metadata = {"scoreThreshold": .1}
        detector.predict = lambda image: [{"bboxPixelsXYXY": [1, 1, 8, 8], "confidence": .2}]
        _, scan = scan_frame(Image.new("RGB", (100, 80)), detector)
        self.assertEqual(scan["settings"]["boxThreshold"], .1)
        self.assertFalse(hasattr(detector, "box_threshold"))

    def test_color_remains_an_optional_full_frame_pass(self):
        image = Image.new("RGB", (100, 80))
        detector = Detector()
        with patch("vision.far_color.enhance", return_value=image) as enhance:
            _, scan = scan_frame(image, detector, color_sensitive=True)
        enhance.assert_called_once_with(image)
        self.assertEqual(scan["modelPasses"], 6)
        self.assertEqual(scan["passTimings"][-1]["kind"], "color")
        self.assertEqual(scan["settings"]["methods"], ["color", "tiles"])
        self.assertEqual(detector.calls[-1][0], image.size)

    def test_capture_emits_scan_metadata_and_no_detector_stays_unprocessed(self):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        stream = io.BytesIO(); Image.fromarray(image).save(stream, format="JPEG")
        encoded = np.frombuffer(stream.getvalue(), dtype=np.uint8)

        class Capture:
            def __init__(self):
                self.done = False
                self.released = False
            def isOpened(self): return True
            def set(self, *args): return True
            def get(self, key): return 30.
            def read(self):
                if self.done: return False, None
                self.done = True; return True, image
            def release(self): self.released = True

        for no_detector in (False, True):
            capture = Capture()
            cv2 = SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_BUFFERSIZE=1, CAP_PROP_FPS=2,
                                  COLOR_BGR2RGB=3, IMWRITE_JPEG_QUALITY=4, cvtColor=lambda data, mode: data,
                                  imencode=lambda *args: (True, encoded))
            detector, output = Detector(), io.StringIO()
            args = ["--input", "0", "--emit", "--max-frames", "1"] + (["--no-detector"] if no_detector else [])
            with patch.dict("sys.modules", {"cv2": cv2}), patch("adapters.live_capture.LocalDetector", return_value=detector), \
                    patch("sys.stdout", output), patch("sys.stderr", new=io.StringIO()):
                main(args)
            packet = json.loads(output.getvalue())
            self.assertTrue(capture.released)
            self.assertFalse(packet["actuationEnabled"])
            self.assertEqual(packet["mappingMode"], "rgb-unlocated")
            if no_detector:
                self.assertEqual(packet["detectorStatus"], "unprocessed")
                self.assertIsNone(packet["detector"])
                self.assertEqual(detector.calls, [])
            else:
                self.assertEqual(packet["detector"]["scan"]["profileId"], "hybrid")
                self.assertEqual(packet["detector"]["scan"]["effectiveProfileId"], "fast")
                self.assertEqual(packet["detector"]["scan"]["modelPasses"], 1)
                self.assertIsNone(packet["detector"]["scan"]["tileRegion"])
                self.assertEqual(packet["provenance"]["cameraDelivery"]["bufferSlots"], 1)
                self.assertFalse(packet["provenance"]["cameraDelivery"]["sensorExposureTimeKnown"])


if __name__ == "__main__":
    unittest.main()
