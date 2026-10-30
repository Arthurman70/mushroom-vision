import json
from pathlib import Path
import tempfile
import unittest
from scripts.prepare_github_source import inspect, selected


class SourcePackageTests(unittest.TestCase):
    def write(self, root, relative, value):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf8')

    def test_deployment_operations_and_recordings_stay_out(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ['deploy/community/discovery.json', 'deploy/community/another-export.json',
                         'deploy/community/runtime-notes.txt', 'data/private.json',
                         'examples/live-session-recorded.jsonl', 'examples/live-session-synthetic.jsonl']:
                self.write(root, name, '{}')
            for name in ['deploy/community/deploy.py', 'deploy/community/README.md',
                         'deploy/community/compose.yaml', 'examples/live-session-navigation-synthetic.json']:
                self.write(root, name, '{}')
            actual = {p.relative_to(root).as_posix() for p in selected(root)}
            self.assertEqual(actual, {'deploy/community/deploy.py', 'deploy/community/README.md',
                                     'deploy/community/compose.yaml', 'examples/live-session-navigation-synthetic.json'})

    def test_private_json_is_rejected_even_outside_known_private_directories(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write(root, 'public/innocent-name.json', json.dumps({'schema': 'mycoscout.community-reviews.v1', 'reviews': []}))
            self.write(root, 'config/other.json', json.dumps({'nested': [{'reviewerId': 'synthetic-private-marker'}]}))
            files, findings = inspect(root)
            self.assertEqual(len(files), 2)
            self.assertEqual(len(findings), 2)
            self.assertTrue(all(x['reason'] == 'private-or-embedded-image-json' for x in findings))

    def test_secrets_machine_paths_and_literal_ssh_targets_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            samples = ['ghp_' + 'x' * 40, 'C:' + '/Users/' + 'someone/private',
                       '/' + 'home/' + 'someone/private', 'operator@' + '203.0.113.7']
            for i, value in enumerate(samples):
                self.write(root, f'docs/example-{i}.md', value)
            _, findings = inspect(root)
            self.assertEqual(len(findings), len(samples))

    def test_safe_aggregate_json_is_hashed_without_local_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.write(root, 'public/report.json', json.dumps({'schema': 'summary.v1', 'images': 25, 'recall': .5}))
            files, findings = inspect(root)
            self.assertEqual(findings, [])
            self.assertEqual(files[0]['path'], 'public/report.json')
            self.assertEqual(len(files[0]['sha256']), 64)
            self.assertNotIn(str(root), json.dumps(files))


if __name__ == '__main__':
    unittest.main()
