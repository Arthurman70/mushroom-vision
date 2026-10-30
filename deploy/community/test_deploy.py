"""Guard the shared proxy boundary and immutable release contents."""
from pathlib import Path
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import importlib.util
import json
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("community_deploy_server", HERE / "deploy-server.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)
transport_spec = importlib.util.spec_from_file_location("community_deploy_transport", HERE / "deploy.py")
transport = importlib.util.module_from_spec(transport_spec)
transport_spec.loader.exec_module(transport)


class ProxyPatchTests(unittest.TestCase):
    def setUp(self):
        self.fragment = (HERE / "caddy-route.txt").read_text()
        self.original = b"other.example {\n respond ok\n}\nthegreatlakesgleaner.press, www.thegreatlakesgleaner.press {\n import existing\n handle {\n  file_server\n }\n}\nlast.example {\n respond ok\n}\n"

    def test_only_adds_expected_route_preserving_other_sites(self):
        patched = deploy.patched_config(self.original, self.fragment)
        self.assertEqual(patched.replace(("\n" + self.fragment.rstrip()).encode(), b"", 1), self.original)
        self.assertIn(b"header_up X-Forwarded-For {remote_host}", patched)
        self.assertIn(b"https://thegreatlakesgleaner.press{uri}", patched)

    def test_repeated_patch_is_identical(self):
        first = deploy.patched_config(self.original, self.fragment)
        self.assertEqual(deploy.patched_config(first, self.fragment), first)

    def test_unknown_or_ambiguous_route_is_rejected(self):
        for original in [self.original + b"\n# /mushroom-lab", self.original + deploy.BEGIN.encode(), self.original.replace(b"www.thegreatlakesgleaner.press", b"different.example")]:
            with self.subTest(original=original), self.assertRaises(ValueError):
                deploy.patched_config(original, self.fragment)

    def test_config_write_refuses_concurrent_changes(self):
        current = Mock()
        current.read_bytes.return_value = b"newer site edit"
        with patch.object(deploy, "CADDY", current):
            with self.assertRaises(RuntimeError):
                deploy.write_config(b"replacement", deploy.digest(self.original))
        current.open.assert_not_called()


class CollectionRefreshTests(unittest.TestCase):
    def test_refresh_only_changes_rights_limited_ai_coverage(self):
        namespace={};exec(transport.COLLECTION_REFRESH_VALIDATOR,namespace);validate=namespace['validate_collection_refresh']
        original={'schema':'mycoscout.community-collection.v1','frames':[{'id':'original','sha256':'same'}],'sources':[{'id':'source'}],'restrictedRecords':[{'id':'image-private','reason':'publication-rights-unresolved','aiReviewed':False}]}
        updated=deepcopy(original);updated['restrictedRecords'][0]['aiReviewed']=True
        self.assertEqual(validate(original,updated),updated)
        for mutate in [lambda v:v['frames'][0].update(sha256='replacement'),lambda v:v['sources'].clear(),lambda v:v['restrictedRecords'][0].update(id='different'),lambda v:v['restrictedRecords'][0].update(path='private-file'),lambda v:v['restrictedRecords'][0].update(aiReviewed='yes')]:
            bad=deepcopy(updated);mutate(bad)
            with self.assertRaises(ValueError):validate(original,bad)


class TrainingStatusTests(unittest.TestCase):
    def setUp(self):
        namespace = {}
        exec(transport.STATUS_VALIDATOR, namespace)
        self.clean = namespace["clean_training_status"]
        self.sample = {
            "schema": "mycoscout.community-learning-status.v1",
            "updatedAt": "2026-01-01T12:00:00.000Z",
            "worker": {"state": "waiting_sync", "enabled": True, "paused": False},
            "approvedFrames": 12, "newFrames": 5, "newSessions": 2,
            "thresholds": {"newFrames": 50, "newSessions": 3, "maxRunsPerDay": 1},
            "latestCandidate": {
                "status": "candidate_better", "completedAt": "2026-01-01T11:00:00.000Z",
                "comparison": {"baselineRecall": .5, "candidateRecall": .7,
                    "baselineFalsePositivesPerFrame": 1.2, "candidateFalsePositivesPerFrame": .8}},
            "automaticActivation": False,
        }

    def test_only_public_allowlisted_aggregates_cross_the_boundary(self):
        payload = deepcopy(self.sample)
        for obj in (payload, payload["worker"], payload["thresholds"], payload["latestCandidate"], payload["latestCandidate"]["comparison"]):
            obj.update({"reviewerId": "PRIVATE_REVIEWER", "path": "PRIVATE_PATH", "error": "PRIVATE_SECRET"})
        payload["message"] = "PRIVATE_SECRET"
        cleaned = self.clean(payload)
        self.assertNotIn("PRIVATE_", json.dumps(cleaned))
        self.assertEqual(cleaned["worker"], self.sample["worker"])
        self.assertEqual(cleaned["latestCandidate"], self.sample["latestCandidate"])
        self.assertEqual(cleaned["message"], "Waiting for a fresh community permissions check.")
        self.assertIs(cleaned["automaticActivation"], False)

    def test_timestamp_offsets_are_normalized_for_the_public_reader(self):
        for given in ("2026-01-01T12:00:00Z", "2026-01-01T12:00:00+00:00", "2026-01-01T07:00:00-05:00", "2026-01-01T12:00:00.000999Z"):
            payload = deepcopy(self.sample)
            payload["updatedAt"] = payload["latestCandidate"]["completedAt"] = given
            with self.subTest(given=given):
                cleaned = self.clean(payload)
                self.assertEqual(cleaned["updatedAt"], "2026-01-01T12:00:00.000Z")
                self.assertEqual(cleaned["latestCandidate"]["completedAt"], cleaned["updatedAt"])

    def test_bad_or_unzoned_and_future_timestamps_are_rejected(self):
        future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        for given in (None, True, 123, "", "invalid", "2026-01-01T12:00:00", "x" * 65, future):
            for candidate_time in (False, True):
                payload = deepcopy(self.sample)
                target = payload["latestCandidate"] if candidate_time else payload
                target["completedAt" if candidate_time else "updatedAt"] = given
                with self.subTest(given=given, candidate=candidate_time), self.assertRaises(ValueError):
                    self.clean(payload)

    def test_unknown_counts_stay_null_but_invalid_counts_are_rejected(self):
        for key in ("approvedFrames", "newFrames", "newSessions"):
            payload = deepcopy(self.sample)
            payload[key] = None
            self.assertIsNone(self.clean(payload)[key])
        for given in (-1, True, 1.0, "1", 10000001, [], {}):
            for thresholds in (False, True):
                payload = deepcopy(self.sample)
                (payload["thresholds"] if thresholds else payload)["newFrames"] = given
                with self.subTest(given=given, thresholds=thresholds), self.assertRaises(ValueError):
                    self.clean(payload)
        payload = deepcopy(self.sample)
        payload["thresholds"]["newFrames"] = None
        with self.assertRaises(ValueError):
            self.clean(payload)

    def test_nested_object_and_enum_types_fail_closed(self):
        for key in ("worker", "thresholds", "latestCandidate"):
            for given in ([], "private", 3, True):
                payload = deepcopy(self.sample)
                payload[key] = given
                with self.subTest(key=key, given=given), self.assertRaises(ValueError):
                    self.clean(payload)
        for given in (None, [], "private", 3, True):
            payload = deepcopy(self.sample)
            payload["latestCandidate"]["comparison"] = given
            with self.subTest(comparison=given), self.assertRaises(ValueError):
                self.clean(payload)
        for given in ([], {}, "unknown"):
            payload = deepcopy(self.sample)
            payload["worker"]["state"] = given
            with self.subTest(workerState=given), self.assertRaises(ValueError):
                self.clean(payload)
            payload = deepcopy(self.sample)
            payload["latestCandidate"]["status"] = given
            with self.subTest(candidateStatus=given), self.assertRaises(ValueError):
                self.clean(payload)

    def test_candidate_metrics_reject_invalid_types_and_nonfinite_values(self):
        for key in self.sample["latestCandidate"]["comparison"]:
            for given in (True, "0.5", [], {}, -1, float("nan"), float("inf"), 100001):
                payload = deepcopy(self.sample)
                payload["latestCandidate"]["comparison"][key] = given
                with self.subTest(key=key, given=given), self.assertRaises(ValueError):
                    self.clean(payload)
            payload = deepcopy(self.sample)
            payload["latestCandidate"]["comparison"][key] = None
            self.assertIsNone(self.clean(payload)["latestCandidate"]["comparison"][key])
        payload = deepcopy(self.sample)
        payload["latestCandidate"]["comparison"]["candidateRecall"] = 1.01
        with self.assertRaises(ValueError):
            self.clean(payload)

    def test_invalid_schema_and_activation_or_boolean_flags_are_rejected(self):
        for key, given in (("schema", "other"), ("automaticActivation", True), ("automaticActivation", 0)):
            payload = deepcopy(self.sample)
            payload[key] = given
            with self.subTest(key=key, given=given), self.assertRaises(ValueError):
                self.clean(payload)
        for key in ("enabled", "paused"):
            payload = deepcopy(self.sample)
            payload["worker"][key] = 1
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.clean(payload)

    def test_update_sends_only_sanitized_json_and_uses_the_same_remote_validator(self):
        payload = deepcopy(self.sample)
        payload["private"] = "PRIVATE_SECRET"
        with patch.object(transport.Path, "read_bytes", return_value=json.dumps(payload).encode()), patch.object(transport, "ssh_command") as ssh:
            transport.update_training_status("local-status.json")
        ssh.assert_called_once()
        self.assertEqual(ssh.call_args.args[0], ["python3", "-c", transport.UPDATE_TRAINING_STATUS])
        sent = ssh.call_args.kwargs["input"]
        self.assertNotIn(b"PRIVATE_SECRET", sent)
        self.assertEqual(json.loads(sent), self.clean(payload))
        self.assertTrue(transport.UPDATE_TRAINING_STATUS.startswith(transport.STATUS_VALIDATOR))
        compile(transport.UPDATE_TRAINING_STATUS, "remote-status-updater", "exec")

    def test_invalid_or_oversized_local_status_never_reaches_ssh(self):
        for raw in (b" " * 65537, b"{bad json", b"{}"):
            with self.subTest(size=len(raw)), patch.object(transport.Path, "read_bytes", return_value=raw), patch.object(transport, "ssh_command") as ssh:
                with self.assertRaises(ValueError):
                    transport.update_training_status("local-status.json")
                ssh.assert_not_called()


if __name__ == "__main__":
    unittest.main()
