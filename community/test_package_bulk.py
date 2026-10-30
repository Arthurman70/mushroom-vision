import hashlib
import importlib.util
import io
import unittest
from pathlib import Path
from PIL import Image
spec=importlib.util.spec_from_file_location('bulk_package',Path(__file__).with_name('package-bulk.py'))
bulk=importlib.util.module_from_spec(spec);spec.loader.exec_module(bulk)

class BulkPublicationTests(unittest.TestCase):
    def test_removes_metadata_without_reencoding_image_pixels(self):
        image=Image.new('RGB',(47,31),(90,120,30));exif=Image.Exif();exif[270]='PRIVATE DESCRIPTION';exif[315]='PRIVATE AUTHOR'
        stream=io.BytesIO();image.save(stream,'JPEG',exif=exif);raw=stream.getvalue();clean=bulk.strip_jpeg_metadata(raw)
        self.assertNotIn(b'PRIVATE',clean)
        with Image.open(io.BytesIO(raw)) as a,Image.open(io.BytesIO(clean)) as b:
            self.assertEqual(a.tobytes(),b.tobytes());self.assertEqual(a.size,b.size);self.assertFalse(b.getexif())
        self.assertEqual(clean,bulk.strip_jpeg_metadata(clean))
    def test_license_gate_requires_explicit_display_rights_and_attribution(self):
        c={'license':'CC BY-SA 2.0','sourceUrl':'https://example.test/photo','licenseUrl':'https://example.test/license','attribution':'Creator','publicDisplayAllowed':True}
        self.assertTrue(bulk.allowed_credit(c));self.assertTrue(bulk.allowed_credit({**c,'license':'CDLA-Permissive-1.0'}))
        for patch in ({'license':'Unknown'},{'license':'CC BY-NC 4.0'},{'publicDisplayAllowed':False},{'attribution':''}):self.assertFalse(bulk.allowed_credit({**c,**patch}))
    def test_malformed_jpeg_does_not_pass(self):
        for raw in (b'private text',b'\xff\xd8\xff\xe1\x00\x20short'):
            with self.assertRaises(ValueError):bulk.strip_jpeg_metadata(raw)

if __name__=='__main__':unittest.main()
