import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from adapters.review_stream import Station, station_origin, send_frame
from adapters.export_review_bundle import export_bundle


class FakeStation:
    def __init__(self, manifest=None, photo=b'example'):
        self.manifest, self.photo, self.calls = manifest, photo, []
    def request(self, path, data=None, binary=False):
        self.calls.append((path, data))
        if path == '/api/training/export':
            return json.loads(json.dumps(self.manifest))
        if path.endswith('/image'):
            return self.photo
        if path == '/api/finds':
            return {'find': {'id': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}}
        raise RuntimeError('Classifier busy')


class ReviewBridgeTests(unittest.TestCase):
    def test_origin_rejects_cleartext_remote_and_embedded_secrets(self):
        for url in ['http://192.168.1.8:8787', 'https://user:secret@example.org', 'https://example.org/?token=secret', 'file:///tmp/foo', 'https://example.org/api']:
            with self.subTest(url=url), self.assertRaises(ValueError):
                station_origin(url)
        self.assertEqual(station_origin('https://example.org/'), 'https://example.org')
        self.assertEqual(station_origin('http://127.0.0.1:8787'), 'http://127.0.0.1:8787')

    def test_image_survives_classifier_failure_and_replay_is_never_capture_time(self):
        with tempfile.TemporaryDirectory() as folder:
            photo = Path(folder)/'frame.jpg'
            photo.write_bytes(b'\xff\xd8\xffsynthetic-test-fixture')
            station = FakeStation()
            result = send_frame(station, photo, session_id='test', evaluation_only=True)
            self.assertFalse(result['classified'])
            self.assertIn('classificationError', result)
            payload = station.calls[0][1]
            self.assertTrue(payload['evaluationOnly'])
            self.assertIsNone(payload['captureTimestamp'])
            self.assertFalse(payload['rights']['trainingOptIn'])
            self.assertEqual(payload['sessionId'], 'test')

    def test_export_checks_image_hash_and_is_repeatable(self):
        raw=b'fixture image bytes';digest=hashlib.sha256(raw).hexdigest()
        manifest={'schema':'mycoscout.reviewed-training-export.v1','releaseId':'reviewed-'+'a'*64,
            'records':[{'findId':'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa','image':{'sha256':digest,'mimeType':'image/jpeg',
                'url':'/api/finds/aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa/image','bytes':len(raw),'relativePath':'untrusted'}}]}
        with tempfile.TemporaryDirectory() as folder:
            result=export_bundle(FakeStation(manifest,raw),Path(folder))
            export_bundle(FakeStation(manifest,raw),Path(folder))
            output=Path(result['manifest'])
            saved=json.loads(output.read_text())
            self.assertEqual(saved['records'][0]['image']['relativePath'],'photos/'+digest+'.jpg')
            self.assertEqual((output.parent/'photos'/f'{digest}.jpg').read_bytes(),raw)
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            export_bundle(FakeStation(manifest,b'corrupt'),Path(folder))

    def test_export_rejects_cross_origin_photo_urls(self):
        manifest={'schema':'mycoscout.reviewed-training-export.v1','releaseId':'reviewed-'+'a'*64,
            'records':[{'findId':'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa','image':{'sha256':'a'*64,'mimeType':'image/jpeg',
                'url':'https://other.example/steal','bytes':5}}]}
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            export_bundle(FakeStation(manifest),Path(folder))

    def test_request_blocks_arbitrary_api_urls(self):
        station=Station('http://127.0.0.1:8787',token='test-token')
        for path in ['https://other.example', '/api/finds?token=x', '/something']:
            with self.assertRaises(ValueError):
                station.request(path)


if __name__=='__main__':
    unittest.main()
