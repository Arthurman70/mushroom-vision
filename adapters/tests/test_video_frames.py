from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageFilter

from adapters.video_frames import (Candidate, bounded_number, extract_collection, frame_metrics,
                                   read_sources, select_frames, sha256_file, MAX_CANDIDATES_PER_VIDEO,
                                   MAX_OUTPUT_BYTES)


def source_record(path, **extra):
    return {"id": "test-video", "path": str(path), "title": "Fixture video", "attribution": "Test fixture author",
            "sourceUrl": "https://example.org/fixture", "license": "CC0-1.0",
            "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0/", **extra}


class VideoSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "fixture.mp4"
        self.video.write_bytes(b"explicit source, decoding mocked in validation tests")
        self.manifest = self.root / "sources.json"

    def write(self, records):
        self.manifest.write_text(json.dumps({"schema": "mycoscout.video-sources.v1", "sources": records}), encoding="utf-8")

    def test_defaults_do_not_grant_training_rights(self):
        self.write([source_record("fixture.mp4")])
        actual = read_sources(self.manifest)[0]
        self.assertFalse(actual["licenseAllowsTraining"])
        self.assertTrue(actual["evaluationOnly"])
        self.assertEqual(actual["localPath"], self.video.resolve())

    def test_training_eligibility_needs_explicit_license_permission(self):
        self.write([source_record(self.video, evaluationOnly=False)])
        self.assertTrue(read_sources(self.manifest)[0]["evaluationOnly"])
        self.write([source_record(self.video, evaluationOnly=False, licenseAllowsTraining=True)])
        self.assertFalse(read_sources(self.manifest)[0]["evaluationOnly"])

    def test_archived_flv_local_source_is_supported(self):
        flv = self.root / "archived.flv"
        flv.write_bytes(b"local original FLV fixture")
        self.write([source_record(flv)])
        self.assertEqual(read_sources(self.manifest)[0]["localPath"].suffix, ".flv")

    def test_urls_network_and_missing_attribution_rejected(self):
        for bad in ("https://example.org/video.mp4", "//server/share/video.mp4", "\\\\server\\share\\video.mp4"):
            self.write([source_record(bad)])
            with self.assertRaises(ValueError):
                read_sources(self.manifest)
        row = source_record(self.video)
        row["attribution"] = ""
        self.write([row])
        with self.assertRaises(ValueError):
            read_sources(self.manifest)

    def test_invalid_ids_groups_and_duplicate_ids_rejected(self):
        for extra in ({"id": "../escape"}, {"sourceGroupId": "../escape"}, {"evaluationOnly": "false"}):
            self.write([source_record(self.video, **extra)])
            with self.assertRaises(ValueError):
                read_sources(self.manifest)
        self.write([source_record(self.video), source_record(self.video)])
        with self.assertRaises(ValueError):
            read_sources(self.manifest)

    def test_numeric_caps_reject_nonfinite_and_bool(self):
        for value in (float("inf"), float("nan"), True, 0, 61):
            with self.assertRaises(ValueError):
                bounded_number(value, "count", 1, 60, integer=True)

    def test_collection_cap_checked_before_tools_or_outputs(self):
        self.write([source_record(self.video, id=f"clip-{i}") for i in range(3)])
        with self.assertRaisesRegex(ValueError, "120-image"):
            extract_collection(self.manifest, self.root / "output", frames_per_video=60)
        self.assertFalse((self.root / "output").exists())

    def test_long_clip_sampling_spans_later_footage_without_raising_candidate_budget(self):
        self.write([source_record(self.video)])
        observed = []

        def decoded(source, directory, ffmpeg, interval, max_seconds, limit):
            clip_seconds = min(2400, max_seconds)
            observed.append((interval, max_seconds, limit))
            self.assertLessEqual(limit, MAX_CANDIDATES_PER_VIDEO)
            self.assertLessEqual(limit * (1280 * 720 * 4 + 65536), MAX_OUTPUT_BYTES)
            self.assertGreaterEqual(interval, clip_seconds / (limit - 1))
            # Return real image files for three sampled positions, including a
            # later section that the former ten-minute duration cap omitted.
            result = []
            for index, timestamp in enumerate((0, clip_seconds / 2, clip_seconds - interval)):
                path = directory / f"{index}.png"
                image = Image.new("RGB", (1280, 720), (index * 100, 30, 40))
                image.save(path)
                blur, digest, dhash, thumb = frame_metrics(image)
                result.append(Candidate(path, timestamp, blur, digest, dhash, thumb, 1280, 720))
            return result

        for requested in (1800, 3600):
            with self.subTest(max_seconds=requested):
                output = self.root / f"long-{requested}"
                with patch("adapters.video_frames.executable", return_value="mock-local-decoder"), \
                        patch("adapters.video_frames.probe_video", return_value={
                            "width": 1280, "height": 720, "durationSeconds": 2400, "codec": "h264"}), \
                        patch("adapters.video_frames.decode_candidates", side_effect=decoded), \
                        patch("builtins.print"):
                    result = extract_collection(self.manifest, output, frames_per_video=3,
                                                sample_interval=1, max_seconds=requested)
                self.assertEqual(len(result["frames"]), 3)
                self.assertGreater(max(frame["timestampSeconds"] for frame in result["frames"]), 1194)
                self.assertEqual(result["settings"]["maxSecondsPerVideo"], requested)
                self.assertEqual(result["settings"]["maxCandidateFramesPerVideo"], 480)
                self.assertEqual(result["sources"][0]["effectiveSampleIntervalSeconds"], observed[-1][0])

    def test_one_hour_duration_cap_rejects_before_decoding_or_output(self):
        self.write([source_record(self.video)])
        for invalid in (0, 3600.01, float("inf"), True):
            with self.subTest(max_seconds=invalid):
                output = self.root / "invalid-duration"
                with patch("adapters.video_frames.executable", side_effect=AssertionError("Must validate first")):
                    with self.assertRaises(ValueError):
                        extract_collection(self.manifest, output, max_seconds=invalid)
                self.assertFalse(output.exists())


class SamplingTests(unittest.TestCase):
    def candidate(self, timestamp, blur, digest, thumb_value=10, dhash=0):
        return Candidate(Path("unused.png"), timestamp, blur, digest, dhash,
                         np.full((32, 32, 3), thumb_value, dtype=np.int16), 64, 48)

    def test_dedup_keeps_blurry_variant_of_same_scene(self):
        rows = [self.candidate(0, 1, "blurry"), self.candidate(1, 4, "medium"), self.candidate(2, 100, "sharp")]
        actual = select_frames(rows, 3)
        self.assertEqual({row.digest for row in actual}, {"blurry", "medium", "sharp"})
        self.assertEqual({row.stratum for row in actual}, {"low", "middle", "high"})

    def test_exact_and_near_duplicates_are_suppressed(self):
        rows = [self.candidate(0, 4, "same"), self.candidate(1, 4, "same"), self.candidate(2, 4, "similar")]
        self.assertEqual(len(select_frames(rows, 3)), 1)

    def test_stratification_includes_low_and_high_sharpness(self):
        rows = [self.candidate(i, i + 1, str(i), thumb_value=i * 20) for i in range(12)]
        selected = select_frames(rows, 6)
        self.assertEqual(len(selected), 6)
        self.assertEqual({row.stratum for row in selected}, {"low", "middle", "high"})
        self.assertEqual([row.timestamp for row in selected], sorted(row.timestamp for row in selected))

    def test_sharpness_statistic_and_color_metrics_do_not_modify_image(self):
        checker = np.indices((64, 64)).sum(axis=0) % 2 * 255
        rgb = np.stack((checker, np.zeros_like(checker), 255 - checker), axis=2).astype(np.uint8)
        image = Image.fromarray(rgb)
        before = image.tobytes()
        sharp = frame_metrics(image)[0]
        blurred = frame_metrics(image.filter(ImageFilter.GaussianBlur(2)))[0]
        self.assertGreater(sharp, blurred)
        self.assertEqual(before, image.tobytes())


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools unavailable")
class ActualDecodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "fixture.mkv"
        frames = []
        for index in range(16):
            yy, xx = np.indices((48, 64))
            pixels = np.stack(((xx * (index + 1) * 3) % 256, np.full_like(xx, index * 15),
                               (yy * (index + 2) * 4) % 256), axis=2).astype(np.uint8)
            frames.append(pixels.tobytes())
        subprocess.run([shutil.which("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                        "-s", "64x48", "-r", "4", "-i", "pipe:0", "-an", "-c:v", "ffv1", str(self.video)],
                       input=b"".join(frames), capture_output=True, timeout=30, check=True)
        self.manifest = self.root / "sources.json"
        self.manifest.write_text(json.dumps({"schema": "mycoscout.video-sources.v1", "sources": [
            source_record(self.video, licenseAllowsTraining=True, evaluationOnly=False, sourceGroupId="original-shared-clip")]}), encoding="utf-8")

    def test_actual_rgb_decode_timestamps_provenance_and_contact_sheet(self):
        output = self.root / "output"
        output.mkdir()
        (output / "research.json").write_text("{}")
        result = extract_collection(self.manifest, output, frames_per_video=6, sample_interval=0.25, max_seconds=4)
        self.assertEqual(result["schema"], "mycoscout.video-collection.v1")
        self.assertEqual(len(result["frames"]), 6)
        self.assertEqual(result["sources"][0]["videoHash"], sha256_file(self.video))
        self.assertNotIn("localPath", result["sources"][0])
        self.assertEqual(result["sources"][0]["candidateFrameCount"], 16)
        for frame in result["frames"]:
            self.assertEqual((frame["width"], frame["height"]), (64, 48))
            self.assertEqual(frame["partitionGroup"], "original-shared-clip")
            self.assertEqual(frame["sourceGroupId"], "original-shared-clip")
            self.assertFalse(frame["trainingAllowed"])
            self.assertFalse(frame["evaluationOnly"])
            self.assertEqual(frame["labels"], [])
            self.assertFalse(frame["annotation"]["complete"])
            self.assertIsNone(frame["actualDistanceM"])
            self.assertEqual(frame["sha256"], sha256_file(output / frame["path"]))
            self.assertAlmostEqual(frame["timestampSeconds"] * 4, round(frame["timestampSeconds"] * 4))
            with Image.open(output / frame["path"]) as image:
                mean_green = np.asarray(image)[:, :, 1].mean()
                self.assertAlmostEqual(mean_green, round(frame["timestampSeconds"] * 4) * 15, delta=2)
        self.assertTrue((output / "contact-sheet.jpg").is_file())
        self.assertTrue((output / "research.json").is_file())
        self.assertTrue((output / "manifest.json").is_file())
        self.assertFalse((output / "INCOMPLETE.json").exists())
        with self.assertRaisesRegex(ValueError, "already contains"):
            extract_collection(self.manifest, output)

    def test_failure_has_no_success_manifest(self):
        output = self.root / "output"
        with patch("adapters.video_frames.decode_candidates", side_effect=ValueError("fixture decode failure")):
            with self.assertRaisesRegex(ValueError, "fixture decode failure"):
                extract_collection(self.manifest, output)
        self.assertTrue((output / "INCOMPLETE.json").is_file())
        self.assertFalse((output / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
