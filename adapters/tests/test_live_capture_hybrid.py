import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from adapters.live_capture import LatestCameraFrames, main, scan_frame
from adapters.tests.test_live_capture import Detector
from vision.hybrid_scan import HybridScanScheduler


class HybridCaptureTests(unittest.TestCase):
    def test_replay_cadence_uses_video_time_and_preserves_native_frames(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        stream = io.BytesIO(); Image.fromarray(frame).save(stream, format="JPEG")
        encoded = np.frombuffer(stream.getvalue(), dtype=np.uint8)

        class Capture:
            index = -1
            released = False
            def isOpened(self): return True
            def set(self, *args): return True
            def get(self, key): return 1.
            def read(self):
                self.index += 1
                return (True, frame) if self.index < 7 else (False, None)
            def release(self): self.released = True

        capture, detector, output = Capture(), Detector(), io.StringIO()
        cv2 = SimpleNamespace(VideoCapture=lambda source: capture, CAP_PROP_BUFFERSIZE=1, CAP_PROP_FPS=2,
                              COLOR_BGR2RGB=3, IMWRITE_JPEG_QUALITY=4, cvtColor=lambda data, mode: data,
                              imencode=lambda *args: (True, encoded))
        # The injected decoder supplies every pixel; use an existing path only
        # to satisfy the CLI's file-presence guard without temporary-file ACLs.
        with patch.dict("sys.modules", {"cv2": cv2}), patch("adapters.live_capture.LocalDetector", return_value=detector), \
                patch("sys.stdout", output), patch("sys.stderr", new=io.StringIO()):
            main(["--input", str(Path(__file__)), "--emit", "--sample-every", "2", "--max-frames", "4"])
        packets = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual([p["timestampSeconds"] for p in packets], [0, 2, 4, 6])
        self.assertEqual([p["detector"]["scan"]["effectiveProfileId"] for p in packets],
                         ["fast", "fast", "fast", "ground-tiles"])
        self.assertEqual(len(detector.calls), 8)
        self.assertTrue(capture.released)
        self.assertTrue(all(p["image"]["width"] == 100 and p["image"]["height"] == 80 for p in packets))
        self.assertTrue(all(p["provenance"]["cameraDelivery"] is None for p in packets))
        self.assertTrue(all(p["mappingMode"] == "rgb-unlocated" for p in packets))

    def test_scan_uses_actual_profiles_full_coordinates_and_approved_threshold(self):
        image = Image.new("RGB", (100, 80))
        scheduler = HybridScanScheduler()
        scans = []
        for timestamp in (0, 2, 6, 8, 10, 12, 14, 16, 18):
            detector = Detector(threshold=.3)
            _, scan = scan_frame(image, detector, profile="hybrid", scheduler=scheduler,
                timestamp_seconds=timestamp, tile_region=[.1, .5, .9, 1])
            scans.append(scan)
            self.assertEqual(scan["profileId"], "hybrid")
            self.assertEqual(detector.calls[0][0], image.size)
            self.assertTrue(all(row[1] == .3 for row in detector.calls))
        self.assertEqual([scan["modelPasses"] for scan in scans], [1, 1, 5, 1, 1, 5, 1, 1, 5])
        self.assertEqual(scans[2]["tileRegion"], [.1, .5, .9, 1])
        self.assertEqual(scans[-1]["tileRegion"], [0, 0, 1, 1])
        self.assertEqual(scans[-1]["effectiveProfileId"], "full-tiles")
        self.assertEqual(scans[-1]["cadence"]["scope"], "whole-frame-detail")
        self.assertFalse(scans[-1]["cadence"]["actuationEnabled"])

    def test_failed_detector_does_not_advance_cadence(self):
        image, scheduler = Image.new("RGB", (100, 80)), HybridScanScheduler()
        for timestamp in (0, 2):
            scan_frame(image, Detector(), profile="hybrid", scheduler=scheduler, timestamp_seconds=timestamp)
        detector = Detector()
        detector.predict = lambda image: (_ for _ in ()).throw(RuntimeError("fixture failure"))
        with self.assertRaises(RuntimeError):
            scan_frame(image, detector, profile="hybrid", scheduler=scheduler, timestamp_seconds=6)
        _, scan = scan_frame(image, Detector(), profile="hybrid", scheduler=scheduler, timestamp_seconds=6)
        self.assertEqual(scan["effectiveProfileId"], "ground-tiles")
        self.assertEqual(scan["cadence"]["detailSweepIndex"], 1)

    def test_hybrid_requires_explicit_state_and_rejects_invalid_crop_before_inference(self):
        image, detector = Image.new("RGB", (100, 80)), Detector()
        with self.assertRaises(ValueError):
            scan_frame(image, detector, profile="hybrid", timestamp_seconds=0)
        with self.assertRaises(ValueError):
            scan_frame(image, detector, profile="fast", scheduler=HybridScanScheduler())
        with self.assertRaises(ValueError):
            scan_frame(image, detector, profile="hybrid", scheduler=HybridScanScheduler(),
                       timestamp_seconds=0, tile_region=[0, .5, 1, .5])
        self.assertEqual(detector.calls, [])

    def test_color_remains_explicit_extra_pass_and_does_not_become_a_species_claim(self):
        image = Image.new("RGB", (100, 80))
        with patch("vision.far_color.enhance", return_value=image):
            _, scan = scan_frame(image, Detector(), profile="hybrid", scheduler=HybridScanScheduler(),
                                timestamp_seconds=0, color_sensitive=True)
        self.assertEqual(scan["modelPasses"], 2)
        self.assertEqual(scan["effectiveProfileId"], "fast")
        self.assertNotIn("species", scan)


class LatestFrameTests(unittest.TestCase):
    def test_multiple_received_frames_replace_one_slot_and_eof_delivers_last(self):
        frame = np.zeros((20, 30, 3), dtype=np.uint8)
        done = threading.Event()

        class Capture:
            index = 0
            released = False
            def read(self):
                if self.index == 8:
                    done.set()
                    return False, None
                self.index += 1
                return True, frame + self.index
            def release(self): self.released = True

        capture = Capture()
        reader = LatestCameraFrames(capture)
        self.assertTrue(done.wait(1.))
        latest = reader.next_frame(-1)
        self.assertEqual(latest[0], 7)
        self.assertEqual(int(latest[1][0, 0, 0]), 8)
        self.assertGreater(latest[2], 0)
        self.assertGreater(latest[3], 0)
        self.assertIsNone(reader.next_frame(7))
        self.assertTrue(reader.close())
        self.assertTrue(capture.released)

    def test_reader_failure_propagates_instead_of_reusing_old_frame(self):
        class Capture:
            def read(self): raise OSError("fixture disconnection")
            def release(self): pass
        reader = LatestCameraFrames(Capture())
        try:
            with self.assertRaisesRegex(RuntimeError, "Camera reader failed"):
                reader.next_frame(-1)
        finally:
            self.assertTrue(reader.close())

    def test_slow_camera_wait_is_bounded_and_release_unblocks_shutdown(self):
        stopped = threading.Event()
        class Capture:
            def read(self):
                stopped.wait(2.)
                return False, None
            def release(self): stopped.set()
        reader = LatestCameraFrames(Capture())
        self.assertIsNone(reader.next_frame(-1, timeout=.01))
        self.assertTrue(reader.close())

    def test_invalid_camera_geometry_fails(self):
        class Capture:
            def read(self): return True, np.zeros((20, 30), dtype=np.uint8)
            def release(self): pass
        reader = LatestCameraFrames(Capture())
        try:
            with self.assertRaises(RuntimeError):
                reader.next_frame(-1)
        finally:
            reader.close()


if __name__ == "__main__":
    unittest.main()
