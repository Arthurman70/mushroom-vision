import importlib.util
import json
from pathlib import Path
import sys
import shutil
import uuid
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
SPEC=importlib.util.spec_from_file_location('ai_contribution_site',HERE/'ai-contribution-site.py')
site=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(site)


class AiContributionPublicationTests(unittest.TestCase):
    def setUp(self):
        self.root=HERE/'test-workspaces'/uuid.uuid4().hex
        self.root.mkdir(parents=True)
        originals={
            'index.html':'<main><a class="btn btn-light" href="articles/mushroom-vision.html">Read about Mushroom Vision</a></main>',
            'articles/mushroom-vision.html':'<article><p>Preserve original prose.</p><section class="article-toc" aria-labelledby="sources-title">Sources</section></article>',
            'sitemap.xml':'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><url><loc>https://example.test/old</loc></url></urlset>',
        }
        self.snapshot=self.root/'original.json'
        self.originals=originals
        self.snapshot.write_text(json.dumps({name:{'text':text,'sha256':site.digest(text.encode())} for name,text in originals.items()}),encoding='utf8')
        self.staged=self.root/'staged'

    def tearDown(self):
        resolved=self.root.resolve()
        if resolved.parent != (HERE/'test-workspaces').resolve():raise ValueError('Unexpected cleanup target')
        shutil.rmtree(resolved)

    def test_remote_program_compiles(self):
        compile(site.REMOTE,'remote','exec')
        self.assertEqual(site.NAMES,('index.html','articles/mushroom-vision.html','sitemap.xml'))

    def test_stage_preserves_prose_and_existing_urls(self):
        site.stage(self.snapshot,self.staged)
        article=(self.staged/'articles/mushroom-vision.html').read_text(encoding='utf8')
        self.assertIn('<p>Preserve original prose.</p>',article)
        self.assertIn('https://example.test/old',(self.staged/'sitemap.xml').read_text())
        request=site.prepare(self.staged,self.snapshot)
        self.assertEqual(len(request['files']),3)
        self.assertTrue(site.ID.fullmatch(request['id']))
        self.assertTrue(all(row['beforeSha256'] for row in request['files']))

    def test_changed_snapshot_hash_rejected(self):
        data=json.loads(self.snapshot.read_text());data['index.html']['text']='tampered';self.snapshot.write_text(json.dumps(data))
        with self.assertRaises(ValueError):site.stage(self.snapshot,self.staged)

    def test_unknown_staged_file_rejected(self):
        site.stage(self.snapshot,self.staged)
        (self.staged/'robots.txt').write_text('not allowed')
        with self.assertRaises(ValueError):site.prepare(self.staged,self.snapshot)

    def test_duplicate_contribution_links_rejected(self):
        data=json.loads(self.snapshot.read_text());text=data['index.html']['text']+'ai-contribute.html';data['index.html']={'text':text,'sha256':site.digest(text.encode())};self.snapshot.write_text(json.dumps(data))
        with self.assertRaises(ValueError):site.stage(self.snapshot,self.staged)

    def test_changed_anchor_rejected(self):
        data=json.loads(self.snapshot.read_text());text='<main>Changed layout</main>';data['index.html']={'text':text,'sha256':site.digest(text.encode())};self.snapshot.write_text(json.dumps(data))
        with self.assertRaises(ValueError):site.stage(self.snapshot,self.staged)


if __name__=='__main__':unittest.main()
