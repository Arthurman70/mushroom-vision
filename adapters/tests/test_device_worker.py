import base64
import hashlib
import io
import unittest
from PIL import Image
from adapters.device_worker import decode_job, execute_job, quality_metrics


class DeviceWorkerTests(unittest.TestCase):
    def job(self):
        stream = io.BytesIO()
        Image.new('RGB', (12, 8), (100, 100, 100)).save(stream, format='PNG')
        raw = stream.getvalue()
        return {'kind': 'image-quality-v1', 'image': {'imageBase64': base64.b64encode(raw).decode()},
                'imageSha256': hashlib.sha256(raw).hexdigest(), 'width': 12, 'height': 8}

    def test_checks_image_identity_before_processing(self):
        job = self.job()
        self.assertEqual(decode_job(job).size, (12, 8))
        with self.assertRaises(ValueError):
            decode_job({**job, 'imageSha256': '0' * 64})
        with self.assertRaises(ValueError):
            decode_job({**job, 'width': 13})
        with self.assertRaises(ValueError):
            decode_job({**job, 'kind': 'flight-command'})

    def test_real_quality_computation(self):
        result = execute_job(self.job())
        self.assertAlmostEqual(result['meanLuminance'], 100)
        self.assertEqual(result['laplacianVariance'], 0)
        self.assertEqual(result['colorOutlierFraction'], 0)
        self.assertGreaterEqual(result['processingMs'], 0)

    def test_large_image_is_sampled_without_inventing_resolution(self):
        result = quality_metrics(Image.new('RGB', (1280, 720), (255, 0, 0)))
        self.assertEqual((result['width'], result['height']), (1280, 720))
        self.assertEqual((result['sampleWidth'], result['sampleHeight']), (640, 360))
        self.assertEqual(result['meanChroma'], 255)


if __name__ == '__main__':
    unittest.main()
