"""Offline acquisition checks: no platform calls, downloads or user data writes."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import acquire_morel_videos as acquisition


VIDEO_ID = acquisition.VIDEO_IDS[0]
LICENSE = "Creative Commons Attribution license (reuse allowed)"


def metadata(**overrides):
    return {"id": VIDEO_ID, "availability": "public", "license": LICENSE,
            "title": "Morel hunting fixture", "uploader": "Fixture creator", "channel_id": "fixture-channel",
            "channel_url": "https://www.youtube.com/channel/fixture-channel", "upload_date": "20200102",
            "description": "Fixture description", "duration": 20.5, "format_id": "136",
            "width": 1280, "height": 720, "fps": 30, "vcodec": "avc1.4d401f", **overrides}


def rendition(**overrides):
    return {"format_id": "fixture", "ext": "mp4", "vcodec": "avc1.4d401f", "acodec": "none",
            "height": 720, "fps": 30, "protocol": "https", "filesize": 1000, "tbr": 100,
            **overrides}


class LicenseAndFormatTests(unittest.TestCase):
    def test_filter_accepts_only_reviewed_public_exact_cc_license(self):
        self.assertIsNone(acquisition.licensed(metadata()))
        for changes in ({"id": "unreviewed"}, {"availability": "unlisted"},
                        {"availability": "private"}, {"availability": None},
                        {"license": "Standard YouTube License"}, {"license": None},
                        {"license": "Creative Commons Attribution license (reuse allowed) in title only"}):
            with self.subTest(changes=changes):
                self.assertIsInstance(acquisition.licensed(metadata(**changes)), str)

    def test_incomplete_metadata_defers_filter_until_full_information(self):
        self.assertIsNone(acquisition.licensed({}, incomplete=True))
        self.assertIsNotNone(acquisition.licensed({}))

    def test_format_prefers_highest_bounded_h264_then_video_only(self):
        options = [rendition(format_id="480", height=480, fps=60),
                   rendition(format_id="720-audio", acodec="mp4a.40.2", fps=60),
                   rendition(format_id="720-video", fps=30)]
        self.assertEqual(list(acquisition.select_format({"formats": options}))[0]["format_id"], "720-video")

    def test_oversize_unknown_size_non_https_non_h264_high_resolution_rejected(self):
        for changes in ({"filesize": acquisition.MAX_FILE + 1}, {"filesize": None},
                        {"filesize": 0}, {"filesize": float("nan")}, {"filesize": float("inf")},
                        {"filesize": "1000"}, {"height": 1080}, {"height": 0},
                        {"protocol": "m3u8_native"}, {"protocol": "http"},
                        {"vcodec": "vp9"}, {"vcodec": "none"}, {"ext": "webm"}):
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    list(acquisition.select_format({"formats": [rendition(**changes)]}))

    def test_exact_file_cap_is_permitted(self):
        selected = list(acquisition.select_format({"formats": [rendition(filesize=acquisition.MAX_FILE)]}))
        self.assertEqual(selected[0]["filesize"], acquisition.MAX_FILE)


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="morel-acquisition-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "acquisition"
        self.options, self.calls = [], []

    def fake_downloader(self, *, info=None, content=b"Local fixture media bytes", failure=None,
                        progress_bytes=None):
        owner = self
        info = deepcopy(metadata() if info is None else info)

        class FakeDownloader:
            def __init__(self, options):
                self.options = options
                owner.options.append(options)

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def extract_info(self, url, download):
                owner.calls.append((url, download))
                if failure:
                    raise failure
                if self.options["match_filter"](info):
                    return None
                for hook in self.options["progress_hooks"]:
                    hook({"status": "finished", "downloaded_bytes": progress_bytes or len(content),
                          "total_bytes": len(content)})
                destination = Path(self.options["outtmpl"].replace("%(id)s", info["id"]).replace("%(ext)s", "mp4"))
                if not destination.exists():
                    destination.write_bytes(content)
                return info

        return FakeDownloader

    def run_acquisition(self, **kwargs):
        with patch.object(acquisition, "YoutubeDL", self.fake_downloader(**kwargs)), patch("builtins.print"):
            return acquisition.acquire(self.output, [VIDEO_ID])

    def test_success_records_rendition_license_creator_hash_and_unverified_labels(self):
        record = self.run_acquisition()[0]
        self.assertEqual(record["status"], "downloaded")
        self.assertEqual(record["id"], VIDEO_ID)
        self.assertEqual(record["sourceUrl"], "https://www.youtube.com/watch?v=" + VIDEO_ID)
        self.assertEqual(record["licenseEvidence"], LICENSE)
        self.assertEqual(record["creator"], "Fixture creator")
        self.assertEqual(record["channelId"], "fixture-channel")
        self.assertEqual(record["formatId"], "136")
        self.assertEqual(record["path"], "videos/" + VIDEO_ID + ".mp4")
        self.assertEqual(record["sha256"], acquisition.sha256(self.output / record["path"]))
        self.assertFalse(record["labelsVerified"])
        self.assertEqual(record["rendition"], "Platform rendition; not original camera footage")
        self.assertEqual(self.calls, [(record["sourceUrl"], True)])
        self.assertEqual(self.options[0]["retries"], 0)
        self.assertEqual(self.options[0]["max_filesize"], acquisition.MAX_FILE)
        self.assertNotIn("cookiefile", self.options[0])
        self.assertNotIn("cookiesfrombrowser", self.options[0])

    def test_verified_retry_checks_original_bytes_and_skips_downloader(self):
        first = self.run_acquisition()[0]
        with patch.object(acquisition, "YoutubeDL", side_effect=AssertionError("Must not redownload")), patch("builtins.print"):
            second = acquisition.acquire(self.output, [VIDEO_ID])[0]
        self.assertEqual(first, second)

    def test_corrupt_or_missing_existing_media_is_rejected(self):
        record = self.run_acquisition()[0]
        path = self.output / record["path"]
        path.write_bytes(b"Replaced media")
        with patch.object(acquisition, "YoutubeDL", side_effect=AssertionError("Must not redownload")), patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "checksum"):
                acquisition.acquire(self.output, [VIDEO_ID])
            path.unlink()
            with self.assertRaisesRegex(ValueError, "checksum"):
                acquisition.acquire(self.output, [VIDEO_ID])

    def test_existing_record_identity_license_size_and_checksum_cannot_change(self):
        valid = self.run_acquisition()[0]
        record_path = self.output / ("download-" + VIDEO_ID + ".json")
        invalid_values = {"id": acquisition.VIDEO_IDS[1], "sourceUrl": "https://example.org/unrelated",
                          "path": "videos/other.mp4", "licenseEvidence": "Standard YouTube License",
                          "bytes": valid["bytes"] + 1, "sha256": "0" * 64}
        for key, value in invalid_values.items():
            with self.subTest(field=key):
                changed = {**valid, key: value}
                acquisition.save_json(record_path, changed)
                with patch.object(acquisition, "YoutubeDL", side_effect=AssertionError("Must not download")), \
                        patch("builtins.print"):
                    with self.assertRaises(ValueError):
                        acquisition.acquire(self.output, [VIDEO_ID])
        acquisition.save_json(record_path, valid)
        acquisition.validate_existing(valid, VIDEO_ID, self.output / valid["path"])

    def test_unverified_final_and_partial_files_refused_before_network_and_preserved(self):
        for previous_status in (None, "failed"):
            for suffix in (".mp4", ".mp4.part"):
                with self.subTest(previous_status=previous_status, suffix=suffix):
                    output = self.root / (str(previous_status) + suffix)
                    media = output / "videos"
                    media.mkdir(parents=True)
                    orphan = media / (VIDEO_ID + suffix)
                    original = b"Keep unverified media for inspection"
                    orphan.write_bytes(original)
                    if previous_status:
                        acquisition.save_json(output / ("download-" + VIDEO_ID + ".json"),
                                              {"id": VIDEO_ID, "status": previous_status, "error": "Interrupted fixture"})
                    with patch.object(acquisition, "YoutubeDL", side_effect=AssertionError("Must not download")):
                        with self.assertRaisesRegex(ValueError, "Unverified prior media"):
                            acquisition.acquire(output, [VIDEO_ID])
                    self.assertEqual(orphan.read_bytes(), original)

    def test_unlicensed_result_cannot_create_successful_record(self):
        record = self.run_acquisition(info=metadata(license="Standard YouTube License"))[0]
        self.assertEqual(record["status"], "failed")
        self.assertFalse((self.output / "videos" / (VIDEO_ID + ".mp4")).exists())

    def test_errors_omit_expiring_media_urls_and_failed_attempt_can_retry(self):
        record = self.run_acquisition(failure=ValueError("Cannot read https://video.example/stream?secret=abc now"))[0]
        self.assertEqual(record["status"], "failed")
        self.assertNotIn("secret=abc", record["error"])
        self.assertIn("[URL omitted]", record["error"])
        self.assertEqual(self.run_acquisition()[0]["status"], "downloaded")

    def test_total_budget_blocks_before_invoking_downloader(self):
        with patch.object(acquisition, "MAX_FILE", 20), patch.object(acquisition, "MAX_TOTAL", 19), \
                patch.object(acquisition, "YoutubeDL", side_effect=AssertionError("Must not download")):
            with self.assertRaisesRegex(ValueError, "budget"):
                acquisition.acquire(self.output, [VIDEO_ID])

    def test_progress_overrun_and_final_file_size_cannot_be_successful(self):
        with patch.object(acquisition, "MAX_FILE", 10), patch.object(acquisition, "MAX_TOTAL", 100):
            record = self.run_acquisition(content=b"short", progress_bytes=11)[0]
            self.assertEqual(record["status"], "failed")
            self.assertIn("per-video limit", record["error"])
            record = self.run_acquisition(content=b"x" * 11, progress_bytes=5)[0]
            self.assertEqual(record["status"], "failed")
            self.assertIn("size limit", record["error"])


if __name__ == "__main__":
    unittest.main()
