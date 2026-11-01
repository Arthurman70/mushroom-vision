import base64
import importlib.util
import json
from pathlib import Path
import shutil
import unittest
import uuid

SPEC = importlib.util.spec_from_file_location('community_deploy', Path(__file__).with_name('deploy.py'))
DEPLOY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPLOY)
ROOT = Path(__file__).resolve().parents[2]


class AppRefreshBatchesTest(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / 'test-runtime' / ('refresh-' + uuid.uuid4().hex)
        self.root.mkdir(parents=True)

    def tearDown(self):
        if self.root.resolve().parent != (ROOT / 'test-runtime').resolve():
            raise ValueError('Unexpected test directory')
        shutil.rmtree(self.root)

    def asset(self, name, content):
        path = self.root / name
        path.write_bytes(content)
        return path, 'community/public/' + name

    def test_batches_preserve_every_byte_and_limit(self):
        paths = [self.asset(f'pixel-view-fixture-{i}--64px.png', bytes([i])*80) for i in range(6)]
        batches = DEPLOY.app_refresh_batches(paths, maximum_bytes=400)
        self.assertGreater(len(batches), 1)
        merged = {}
        for batch in batches:
            self.assertLessEqual(len(batch), 400)
            for key, value in json.loads(batch).items():
                self.assertNotIn(key, merged)
                merged[key] = base64.b64decode(value, validate=True)
        self.assertEqual(merged, {name: path.read_bytes() for path, name in paths})

    def test_large_single_asset_fails_before_any_batch_is_returned(self):
        paths = [self.asset('small.js', b'a'), self.asset('large.png', b'b'*500)]
        with self.assertRaisesRegex(ValueError, 'One app asset'):
            DEPLOY.app_refresh_batches(paths, maximum_bytes=200)

    def test_duplicate_remote_name_rejected(self):
        asset = self.asset('same.js', b'hello')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            DEPLOY.app_refresh_batches([asset, asset])

    def test_empty_assets_need_no_mutation(self):
        self.assertEqual(DEPLOY.app_refresh_batches([]), [])


if __name__ == '__main__':
    unittest.main()
