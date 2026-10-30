"""Standard-library contract tests. Does not require ROS, torch or model weights."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent.parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mapping = load("rtabmap_snapshot")
vision = load("detect_fungi")


def sample():
    return {
        "header": {"frame_id": "map", "stamp": {"sec": 1700000000, "nanosec": 0}},
        "info": {"width": 3, "height": 2, "resolution": 0.5,
                 "origin": {"position": {"x": -2, "y": 3, "z": 0},
                            "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}}},
        "data": [-1, 0, 1, 64, 65, 100],
    }


class GridTests(unittest.TestCase):
    def test_only_exactly_zero_is_free(self):
        converted = mapping.convert_grid(sample())
        self.assertEqual(converted["grid"]["cells"], [-1, 0, -1, -1, 1, 1])
        self.assertFalse(converted["groundTraversabilityVerified"])
        self.assertEqual(converted["frame"], {"id": "map", "unit": "m"})

    def test_origin_and_row_order_retained(self):
        converted = mapping.convert_grid(sample())
        self.assertEqual(converted["grid"]["origin"], {"x": -2.0, "y": 3.0})
        self.assertEqual(converted["grid"]["resolution"], 0.5)

    def test_different_frame_rejected(self):
        with self.assertRaises(ValueError):
            mapping.convert_grid(sample(), expected_frame="odom")

    def test_rotation_rejected(self):
        data = sample()
        data["info"]["origin"]["orientation"] = {"x": 0, "y": 0, "z": 1, "w": 0}
        with self.assertRaises(ValueError):
            mapping.convert_grid(data)

    def test_negative_identity_quaternion_allowed(self):
        data = sample()
        data["info"]["origin"]["orientation"]["w"] = -1
        self.assertEqual(mapping.convert_grid(data)["grid"]["cells"], [-1, 0, -1, -1, 1, 1])

    def test_zero_quaternion_rejected(self):
        data = sample()
        data["info"]["origin"]["orientation"]["w"] = 0
        with self.assertRaises(ValueError):
            mapping.convert_grid(data)

    def test_off_plane_rejected(self):
        data = sample()
        data["info"]["origin"]["position"]["z"] = 1
        with self.assertRaises(ValueError):
            mapping.convert_grid(data)

    def test_nonfinite_origin_rejected(self):
        for value in (float("nan"), float("inf"), True):
            data = sample()
            data["info"]["origin"]["position"]["x"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                mapping.convert_grid(data)

    def test_invalid_sizes_rejected(self):
        for value in (0, 1, 257, 2.5, True):
            data = sample()
            data["info"]["width"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                mapping.convert_grid(data)

    def test_resolution_rejected(self):
        for value in (0.05, 101, -1, float("nan"), True):
            data = sample()
            data["info"]["resolution"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                mapping.convert_grid(data)

    def test_length_mismatch_rejected(self):
        data = sample()
        data["data"].pop()
        with self.assertRaises(ValueError):
            mapping.convert_grid(data)

    def test_invalid_occupancy_rejected(self):
        for value in (-2, 101, 0.5, True, "0"):
            data = sample()
            data["data"][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                mapping.convert_grid(data)

    def test_unknown_stays_unknown_at_low_threshold(self):
        self.assertEqual(mapping.convert_grid(sample(), occupied_threshold=1)["grid"]["cells"][0], -1)

    def test_invalid_threshold_rejected(self):
        with self.assertRaises(ValueError):
            mapping.convert_grid(sample(), occupied_threshold=0)


class UploadTests(unittest.TestCase):
    def test_remote_needs_token(self):
        with self.assertRaisesRegex(ValueError, "TOKEN"):
            mapping.upload_snapshot({}, "https://192.168.1.10:8787", "")

    def test_remote_needs_tls(self):
        with self.assertRaisesRegex(ValueError, "HTTPS"):
            mapping.upload_snapshot({}, "http://192.168.1.10:8787", "test-token")

    def test_url_credentials_rejected(self):
        with self.assertRaises(ValueError):
            mapping.upload_snapshot({}, "https://name:password@example.com", "test-token")

    def test_query_rejected(self):
        with self.assertRaises(ValueError):
            mapping.upload_snapshot({}, "https://example.com?key=unused", "test-token")

    def test_exact_request_and_auth(self):
        fake_response = mock.MagicMock()
        fake_response.__enter__.return_value.status = 200
        opener = mock.MagicMock()
        opener.open.return_value = fake_response
        with mock.patch.object(mapping.request, "build_opener", return_value=opener):
            mapping.upload_snapshot({"groundTraversabilityVerified": False},
                                    "http://127.0.0.1:8787", "test-token")
        req = opener.open.call_args.args[0]
        self.assertEqual(req.full_url, "http://127.0.0.1:8787/api/map")
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-token")
        self.assertFalse(json.loads(req.data)["groundTraversabilityVerified"])


class VisionTests(unittest.TestCase):
    def test_no_world_coordinates_or_species_are_created(self):
        result = vision.candidate_records([[1, 2, 8, 9]], [0.8], ["mushroom"], 10, 10)[0]
        self.assertEqual(result["status"], "UNLOCATED")
        self.assertFalse(result["poseKnown"])
        self.assertFalse(result["depthKnown"])
        self.assertIsNone(result["species"])
        self.assertTrue(all(k not in result for k in ("x", "y", "z", "frameId")))

    def test_boxes_are_clipped_to_image(self):
        result = vision.candidate_records([[-3, -2, 20, 30]], [0.7], ["fungus"], 10, 10)
        self.assertEqual(result[0]["bboxPixelsXYXY"], [0, 0, 10, 10])

    def test_invalid_boxes_and_scores_are_removed(self):
        result = vision.candidate_records(
            [[1, 1, 0, 5], [1, 1, 5, 5], [1, 1, 5, 5], [float("nan"), 0, 5, 5]],
            [0.8, float("nan"), 1.5, 0.5], ["a"] * 4, 10, 10)
        self.assertEqual(result, [])

    def test_scores_sorted_descending(self):
        result = vision.candidate_records([[1, 1, 2, 2]] * 2, [0.2, 0.9], ["a", "b"], 10, 10)
        self.assertEqual([entry["confidence"] for entry in result], [0.9, 0.2])

    def test_mismatched_outputs_rejected(self):
        with self.assertRaises(ValueError):
            vision.candidate_records([], [0.5], [], 10, 10)


class CliTests(unittest.TestCase):
    def test_help_works_without_site_packages(self):
        for name in ("rtabmap_snapshot", "detect_fungi"):
            result = subprocess.run([sys.executable, "-S", str(HERE / (name + ".py")), "--help"],
                                    capture_output=True, text=True, timeout=10)
            with self.subTest(name=name):
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

    def test_json_snapshot_smoke_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            source, output = path / "input.json", path / "snapshot.json"
            source.write_text(json.dumps(sample()), encoding="utf-8-sig")
            args = ["--json", str(source), "--output", str(output)]
            self.assertEqual(mapping.main(args), 0)
            self.assertFalse(json.loads(output.read_text())["groundTraversabilityVerified"])
            before = output.read_bytes()
            self.assertEqual(mapping.main(args), 2)
            self.assertEqual(output.read_bytes(), before)

    def test_upload_requires_explicit_survey_replacement(self):
        with tempfile.TemporaryDirectory() as temp:
            with mock.patch.object(mapping, "upload_snapshot") as upload:
                result = mapping.main(["--json", "unused.json", "--output",
                                       str(Path(temp) / "out.json"), "--upload"])
                self.assertEqual(result, 2)
                upload.assert_not_called()


if __name__ == "__main__":
    unittest.main()

