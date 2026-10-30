from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy
from datetime import datetime, timezone
from email.message import Message
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from vision.acquire_videos import NoRedirect, acquire, main, retry_deadline, verified_source


DATA = b"licensed local fixture video content"


def row(identifier="123", data=DATA):
    return {"id": identifier, "title": "File:Fixture.webm", "author": "Fixture author",
            "downloadUrl": f"https://upload.wikimedia.org/wikipedia/commons/a/ab/Fixture-{identifier}.webm",
            "sourceUrl": "https://commons.wikimedia.org/wiki/File:Fixture.webm",
            "license": "CC BY 4.0", "licenseUrl": "https://creativecommons.org/licenses/by/4.0/",
            "licenseEvidence": "Verified fixture evidence", "sourceGroupId": "creator-session-1",
            "sha1": hashlib.sha1(data).hexdigest(), "bytes": len(data)}


def manifest(rows=None):
    return {"schema": "mycoscout.video-sources-research.v1", "sources": rows or [row()]}


class FakeResponse(io.BytesIO):
    def __init__(self, source, content=DATA, *, status=200, headers=None, url=None):
        super().__init__(content)
        self.status, self.url = status, url or source["downloadUrl"]
        self.headers = headers if headers is not None else {"Content-Length": str(len(content))}

    def getcode(self):
        return self.status

    def geturl(self):
        return self.url


class FakeOpener:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        value = self.results.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class VerifiedVideoMetadataTests(unittest.TestCase):
    def test_exact_host_original_path_and_no_credentials_query_or_redirect(self):
        self.assertEqual(verified_source(row()).hostname, "upload.wikimedia.org")
        invalid_urls = [
            "http://upload.wikimedia.org/wikipedia/commons/a/ab/Fixture.webm",
            "https://upload.wikimedia.org.evil.invalid/wikipedia/commons/a/ab/Fixture.webm",
            "https://user:secret@upload.wikimedia.org/wikipedia/commons/a/ab/Fixture.webm",
            "https://upload.wikimedia.org:443/wikipedia/commons/a/ab/Fixture.webm",
            "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/ab/Fixture.webm/1080p.webm",
            "https://upload.wikimedia.org/wikipedia/commons/a/ab/..%2Fsecret.webm",
            "https://upload.wikimedia.org/wikipedia/commons/a/ab/Fixture.webm?token=x",
        ]
        for url in invalid_urls:
            with self.subTest(url=url), self.assertRaises(ValueError):
                verified_source({**row(), "downloadUrl": url})
        with self.assertRaisesRegex(ValueError, "redirects"):
            NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.org/file.webm")

    def test_open_license_metadata_must_agree_and_have_evidence(self):
        for updates in ({"license": "CC BY-NC 4.0"}, {"licenseUrl": "https://creativecommons.org/licenses/by-nc/4.0"},
                        {"licenseEvidence": ""}, {"author": ""}, {"licenseAllowsTraining": False}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                verified_source({**row(), **updates})
        verified_source({**row(), "license": "CC0", "licenseUrl": "https://creativecommons.org/publicdomain/zero/1.0"})

    def test_file_size_hash_and_group_limits(self):
        for updates in ({"bytes": True}, {"bytes": 120_000_001}, {"bytes": 0}, {"sha1": "abcd"}, {"sourceGroupId": "../escape"}):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                verified_source({**row(), **updates})

    def test_backoff_seconds_and_dates_are_recorded_without_retry(self):
        now = datetime(2026, 10, 28, 22, 28, tzinfo=timezone.utc)
        self.assertEqual(retry_deadline("600", now), "2026-10-28T22:38:00+00:00")
        self.assertEqual(retry_deadline("Wed, 28 Oct 2026 22:38:00 GMT", now), "2026-10-28T22:38:00+00:00")
        self.assertEqual(retry_deadline("Wed, 28 Oct 2026 22:00:00 GMT", now), now.isoformat())
        self.assertIsNone(retry_deadline("invalid", now))
        self.assertIsNone(retry_deadline(None, now))


class VideoDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "collection"

    def download(self, source_manifest, opener, ids=None, **kwargs):
        with patch("vision.acquire_videos.urllib.request.build_opener", return_value=opener), redirect_stdout(io.StringIO()):
            return acquire(source_manifest, self.output, ids or ["123"], **kwargs)

    def test_success_preserves_hash_license_and_session_group_without_training(self):
        source = row()
        opener = FakeOpener([FakeResponse(source)])
        result = self.download(manifest(), opener)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["downloadedBytes"], len(DATA))
        actual = result["sources"][0]
        self.assertEqual(actual["sha256"], hashlib.sha256(DATA).hexdigest())
        self.assertEqual(actual["sourceGroupId"], "creator-session-1")
        self.assertEqual(actual["sourceMetadata"]["licenseEvidence"], source["licenseEvidence"])
        self.assertTrue(actual["evaluationOnly"])
        self.assertEqual(actual["annotations"], [])
        self.assertIsNone(actual["distanceM"])
        self.assertFalse(result["automaticTraining"])
        self.assertEqual((self.output / actual["path"]).read_bytes(), DATA)
        self.assertEqual(json.loads((self.output / "sources.json").read_text())["status"], "complete")
        self.assertEqual(opener.calls[0][1], 45)

    def test_hash_mismatch_is_not_promoted_to_successful_video(self):
        source = row()
        opener = FakeOpener([FakeResponse(source, b"x" * len(DATA))])
        result = self.download(manifest(), opener)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["sources"], [])
        self.assertFalse((self.output / "downloads/commons-123.webm").exists())
        self.assertIn("SHA1", result["failures"][0]["error"])

    def test_size_overrun_stops_before_oversized_write(self):
        source = row()
        result = self.download(manifest(), FakeOpener([FakeResponse(source, DATA + b"extra", headers={})]))
        self.assertEqual(result["sources"], [])
        self.assertIn("exceeds", result["failures"][0]["error"])
        self.assertEqual((self.output / "downloads/commons-123.webm.partial").stat().st_size, 0)

    def test_header_size_and_unexpected_response_url_rejected(self):
        source = row()
        result = self.download(manifest(), FakeOpener([FakeResponse(source, headers={"Content-Length": "999"})]))
        self.assertIn("size differs", result["failures"][0]["error"])
        self.output = self.root / "other"
        result = self.download(manifest(), FakeOpener([FakeResponse(source, url="https://example.org/file.webm")]))
        self.assertIn("selected original", result["failures"][0]["error"])

    def test_429_stops_all_following_sources_and_preserves_backoff(self):
        headers = Message()
        headers["Retry-After"] = "600"
        error = HTTPError(row()["downloadUrl"], 429, "Back off", headers, None)
        opener = FakeOpener([error])
        result = self.download(manifest([row(), row("456")]), opener, ["123", "456"])
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(result["unattemptedIds"], ["456"])
        self.assertEqual(result["failures"][0]["httpStatus"], 429)
        self.assertEqual(result["failures"][0]["retryAfter"], "600")
        self.assertIsNotNone(result["failures"][0]["retryNotBefore"])

    def test_503_stops_remaining_requests_after_partial_success(self):
        error = HTTPError(row()["downloadUrl"], 503, "Unavailable", {}, None)
        opener = FakeOpener([FakeResponse(row()), error])
        result = self.download(manifest([row(), row("456"), row("789")]), opener, ["123", "456", "789"])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(len(opener.calls), 2)
        self.assertEqual(result["unattemptedIds"], ["789"])
        self.assertEqual(len(result["sources"]), 1)

    def test_budget_and_duplicate_id_validation_happen_before_network_or_output(self):
        with patch("vision.acquire_videos.urllib.request.build_opener") as build:
            for source_manifest, ids, max_bytes in ((manifest(), ["123"], len(DATA) - 1),
                                                   (manifest(), ["123", "123"], 400_000_000),
                                                   (manifest([row(), row()]), ["123"], 400_000_000),
                                                   (manifest(), ["999"], 400_000_000)):
                with self.assertRaises(ValueError):
                    acquire(source_manifest, self.output, ids, max_bytes)
            build.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_total_time_limit_stops_download(self):
        opener = FakeOpener([FakeResponse(row())])
        with patch("vision.acquire_videos.time.monotonic", side_effect=[0, 301]):
            result = self.download(manifest(), opener)
        self.assertEqual(result["sources"], [])
        self.assertIn("five-minute", result["failures"][0]["error"])

    def test_zero_success_cli_returns_failure(self):
        path = self.root / "research.json"
        path.write_text(json.dumps(manifest()), encoding="utf-8")
        result = {"sources": [], "failures": [{"id": "123"}], "downloadedBytes": 0}
        with patch("sys.argv", ["acquire_videos", "--manifest", str(path), "--output-dir", str(self.output), "--ids", "123"]), \
                patch("vision.acquire_videos.acquire", return_value=result), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                main()
        self.assertEqual(raised.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
