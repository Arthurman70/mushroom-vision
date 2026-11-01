"""Exercise appended publication boundaries without contacting the server."""
from copy import deepcopy
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch
import hashlib
import importlib.util
import json
import unittest
import uuid
import shutil

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('append_images',HERE/'extend-collection.py')
extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(extension)

@contextmanager
def test_directory():
    root=(HERE/'test-runtime').resolve();root.mkdir(exist_ok=True)
    path=root/uuid.uuid4().hex;path.mkdir()
    try:yield path
    finally:
        if path.resolve().parent!=root:raise ValueError('Unexpected test cleanup path')
        shutil.rmtree(path)

def fixture():
    old={'schema':'mycoscout.community-collection.v1','frames':[{'id':'old','sourceId':'old-source','sha256':'a'*64,'sourceGroupId':'old-group','evaluationOnly':True}],
         'sources':[{'id':'old-source','attribution':'Old immutable credit'}],'privacy':'unchanged'}
    old_ai={'schema':'mycoscout.public-ai-suggestions.v1','frames':[{'frameId':'old','audit':'unchanged'}],'run':{'status':'completed','totalFrames':1,'humanVerified':False,'reviewedFrames':1}}
    raw=b'synthetic-image';sha=hashlib.sha256(raw).hexdigest();source_id='photo-'+sha[:24]
    frame={'id':'new-image','sourceId':source_id,'kind':'photo','width':20,'height':10,'timestampSeconds':0,'sha256':sha,'originalSha256':sha,'path':'images/'+sha+'.jpg','sourceGroupId':'new-group','videoHash':None,'evaluationOnly':False,'sourceLicenseAllowsTraining':True,'actualDistanceM':None}
    source={'id':source_id,'title':'Licensed research photo','attribution':'Synthetic author','license':'CC-BY','licenseUrl':'https://creativecommons.org/licenses/by/4.0/','sourceUrl':'https://www.inaturalist.org/observations/123','changes':'Metadata removed.'}
    new=deepcopy(old);new['frames'].append(frame);new['sources'].append(source)
    ai=deepcopy(old_ai);ai['frames'].append({'frameId':frame['id'],'sourceId':source_id,'imageSha256':sha,'status':'pending','model':'Not yet reviewed','boxes':[],'audit':None});ai['run'].update(status='in-progress',totalFrames=2,humanVerified=False)
    files={'collection/manifest.json':extension.encoded(new),'collection/ai-suggestions.json':extension.encoded(ai),'collection/'+frame['path']:raw}
    return old,old_ai,new,ai,files,[frame['id']]

class AppendValidationTests(unittest.TestCase):
    def test_valid_append_keeps_prior_evidence_and_pending_new_opinions(self):
        data=fixture();self.assertEqual(len(extension.validate_append(*data)),1)
        self.assertTrue(extension.REMOTE.startswith(extension.APPEND_VALIDATOR))

    def test_source_overrides_bad_paths_unknown_rights_and_invalid_dimensions_fail(self):
        mutations=[lambda n,a:n['sources'].append(deepcopy(n['sources'][0])),lambda n,a:n['sources'][-1].update(license='CC-BY-NC'),
                   lambda n,a:n['sources'][-1].update(licenseUrl='https://creativecommons.org/licenses/by-sa/4.0/'),lambda n,a:n['sources'][-1].update(sourceUrl='file:///private'),
                   lambda n,a:n['frames'][-1].update(id='../outside'),lambda n,a:n['frames'][-1].update(width=True),lambda n,a:n['frames'][-1].update(width=20000001),
                   lambda n,a:n['frames'][-1].update(path='../outside.jpg'),lambda n,a:n['frames'][-1].update(evaluationOnly='false'),
                   lambda n,a:n['frames'][-1].update(sourceGroupId='old-group'),lambda n,a:n['frames'][-1].update(trainingApproved=True)]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                old,old_ai,new,ai,files,ids=fixture();mutate(new,ai)
                with self.assertRaises(ValueError):extension.validate_append(old,old_ai,new,ai,files,ids)

    def test_appended_ai_cannot_claim_confirmation_or_modify_earlier_opinions(self):
        mutations=[lambda n,a:a['frames'][-1].update(humanVerified=True),lambda n,a:a['frames'][-1].update(status='completed'),
                   lambda n,a:a['frames'][-1].update(imageSha256='f'*64),lambda n,a:a['frames'].append({'frameId':'extra'}),
                   lambda n,a:a['frames'][0].update(audit='changed'),lambda n,a:a['run'].update(humanVerified=True),lambda n,a:a['run'].update(reviewedFrames=999),
                   lambda n,a:n['sources'][0].update(attribution='changed'),lambda n,a:n.update(privacy='changed')]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                old,old_ai,new,ai,files,ids=fixture();mutate(new,ai)
                with self.assertRaises(ValueError):extension.validate_append(old,old_ai,new,ai,files,ids)

    def test_hash_mismatch_and_unreferenced_archive_members_fail(self):
        old,old_ai,new,ai,files,ids=fixture();files['collection/'+new['frames'][-1]['path']]=b'changed'
        with self.assertRaisesRegex(ValueError,'hash'):extension.validate_append(old,old_ai,new,ai,files,ids)
        old,old_ai,new,ai,files,ids=fixture();files['collection/images/'+'f'*64+'.jpg']=b'unreferenced'
        with self.assertRaisesRegex(ValueError,'Unreferenced'):extension.validate_append(old,old_ai,new,ai,files,ids)

    def test_local_builder_rejects_unknown_split_or_id_before_touching_output(self):
        with test_directory() as temp:
            root=Path(temp);old,old_ai,*_=fixture();snapshot=root/'snapshot.json';snapshot.write_text(json.dumps({'manifest':old,'suggestions':old_ai}))
            source=root/'source.json';output=root/'append.tar.gz'
            for row in [{'id':'safe-id','split':'anything'},{'id':'../bad','split':'train'}]:
                source.write_text(json.dumps({'images':[row]}))
                with self.assertRaisesRegex(ValueError,'ID or training split'):extension.build(snapshot,source,output)
                self.assertFalse(output.exists())

    def test_local_builder_packages_a_verified_jpeg_without_changing_prior_entries(self):
        from PIL import Image
        import tarfile
        with test_directory() as temp:
            root=Path(temp);old,old_ai,*_=fixture();snapshot=root/'snapshot.json';snapshot.write_text(json.dumps({'manifest':old,'suggestions':old_ai}))
            image=root/'image.jpg';Image.new('RGB',(20,10),(70,100,40)).save(image);raw=image.read_bytes()
            row={'id':'new-real-fixture','split':'train','sourceGroupId':'new-group','sourceUrl':'https://www.inaturalist.org/observations/123','path':'image.jpg',
                 'sha256':hashlib.sha256(raw).hexdigest(),'width':20,'height':10,'photoLicense':{'code':'cc-by','url':'https://creativecommons.org/licenses/by/4.0/','attribution':'Synthetic fixture'}}
            source=root/'source.json';source.write_text(json.dumps({'images':[row]}));output=root/'append.tar.gz'
            result=extension.build(snapshot,source,output);self.assertEqual(result['added'],1)
            with tarfile.open(output) as archive:manifest=json.load(archive.extractfile('collection/manifest.json'))
            self.assertEqual(manifest['frames'][0],old['frames'][0]);self.assertEqual(manifest['sources'][0],old['sources'][0])

class AIExportTransportTests(unittest.TestCase):
    def test_ai_export_uses_separate_script_and_refuses_human_schema(self):
        transport=extension.deploy
        with test_directory() as temp:
            path=Path(temp)/'ai.json'
            good={'schema':'mushroom-vision.ai-contribution-export.v1','reviewerKind':'ai','trainingApproved':False,'humanVerified':False,'contributions':[]}
            with patch.object(transport,'ssh_command',side_effect=[b'{}',json.dumps(good).encode(),b'']) as remote:
                transport.export_ai_contributions(path)
                self.assertIn('/app/export-ai-contributions.mjs',remote.call_args_list[0].args[0]);self.assertEqual(json.loads(path.read_text()),good)
            bad=Path(temp)/'wrong.json'
            with patch.object(transport,'ssh_command',side_effect=[b'{}',b'{"schema":"mycoscout.community-reviews.v1","trainingApproved":false}',b'']):
                with self.assertRaises(ValueError):transport.export_ai_contributions(bad)
            self.assertFalse(bad.exists())

if __name__=='__main__':unittest.main()
