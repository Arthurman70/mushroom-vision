"""Exercise the remote resume programs locally without SSH or a live release."""
from pathlib import Path
from contextlib import contextmanager, redirect_stdout
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import uuid4

HERE = Path(__file__).resolve().parent
def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value
transport = module('resume_test_transport', 'deploy.py')
with patch.dict(sys.modules, {'deploy': transport}):
    resume = module('resume_test_helper', 'resume_upload.py')

@contextmanager
def workspace():
    root = HERE / ('tmp-resume-test-' + uuid4().hex)
    root.mkdir(); (root/'releases').mkdir()
    current = root/'releases/live-release'; current.mkdir()
    staged = root/'releases/staged-release'; staged.mkdir()
    (root/'current.json').write_text(json.dumps({'release':str(current)}))
    (staged/'UPLOAD-INCOMPLETE').write_text('incomplete')
    try: yield root, current, staged
    finally:
        if root.resolve().parent != HERE or root.is_symlink(): raise ValueError('Unexpected fixture directory')
        shutil.rmtree(root)

def execute(program, root, raw=b'', release='staged-release'):
    program = program.replace("base=pathlib.Path('/root/gleaner-mushroom-lab')", 'base=pathlib.Path('+repr(str(root))+')')
    output = io.StringIO()
    with patch.object(sys, 'argv', ['remote', release]), patch.object(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(raw))), redirect_stdout(output):
        exec(program, {})
    return json.loads(output.getvalue())

def archive(name, data):
    output=io.BytesIO()
    with tarfile.open(fileobj=output,mode='w') as tar:
        member=tarfile.TarInfo(name); member.size=len(data); tar.addfile(member,io.BytesIO(data))
    return output.getvalue()

class ResumeUploadTests(unittest.TestCase):
    def test_current_and_unmarked_releases_cannot_be_modified(self):
        with workspace() as (root,current,staged):
            with self.assertRaisesRegex(ValueError,'current release'):execute(resume.SCAN,root,b'{}','live-release')
            (staged/'UPLOAD-INCOMPLETE').unlink()
            with self.assertRaisesRegex(ValueError,'incomplete'):execute(resume.SCAN,root,b'{}')

    def test_release_and_control_symlinks_are_rejected_before_resolution_or_writes(self):
        with workspace() as (root,current,staged):
            actual = Path.is_symlink
            for linked in [staged,staged/'release-manifest.json',staged/'UPLOAD-INCOMPLETE']:
                with patch.object(Path,'is_symlink',lambda path: path==linked or actual(path)):
                    with self.assertRaises(ValueError):execute(resume.SCAN,root,b'{}')
            self.assertFalse((staged/'release-manifest.json').exists())

    def test_traversal_and_reserved_control_members_fail_without_writing(self):
        with workspace() as (root,current,staged):
            for name in ['../outside','/absolute','release-manifest.json','UPLOAD-INCOMPLETE','nested//alias']:
                manifest={'schema':'mycoscout.community-release.v1','files':{name:{'bytes':1,'sha256':hashlib.sha256(b'x').hexdigest()}},'frames':0}
                with self.assertRaises(ValueError):execute(resume.SCAN,root,json.dumps(manifest).encode())
            self.assertFalse((staged/'release-manifest.json').exists())

    def test_bad_hash_keeps_marker_and_valid_copy_only_marks_staged_release_ready(self):
        with workspace() as (root,current,staged):
            name='collection/image.jpg';data=b'fixture-image'
            manifest={'schema':'mycoscout.community-release.v1','files':{name:{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}},'frames':1}
            self.assertEqual(execute(resume.SCAN,root,json.dumps(manifest).encode())['missing'],[name])
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):execute(resume.COPY,root,archive(name,b'x'*len(data)))
            self.assertTrue((staged/'UPLOAD-INCOMPLETE').exists())
            self.assertFalse((staged/name).exists())
            self.assertEqual(execute(resume.COPY,root,archive(name,data))['copied'],1)
            result=execute(resume.VERIFY,root)
            self.assertTrue(result['ready']);self.assertFalse(result['published'])
            self.assertEqual(json.loads((root/'current.json').read_text())['release'],str(current))
            self.assertFalse((staged/'UPLOAD-INCOMPLETE').exists())

if __name__=='__main__':unittest.main()
