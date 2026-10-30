from pathlib import Path
import hashlib, json, zipfile
ROOT=Path(__file__).resolve().parents[1]
DIST=ROOT/'dist'
DIST.mkdir(exist_ok=True)
def write_zip(name, pairs):
    with zipfile.ZipFile(DIST/name,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for source,target in sorted(pairs,key=lambda p:p[1]):
            archive.write(source,target)
    with zipfile.ZipFile(DIST/name) as archive:
        assert archive.testzip() is None
def descendants(folder):
    for file in (ROOT/folder).rglob('*'):
        if not file.is_file(): continue
        rel=file.relative_to(ROOT)
        if any(p in {'.gradle','build','__pycache__','node_modules','.venv','runs','results'} for p in rel.parts): continue
        if file.name in {'local.properties','.env'} or file.suffix in {'.keystore','.jks','.pyc'}: continue
        if file.name.startswith('live-session-') and file.suffix=='.jsonl': continue
        if 'src/main/assets' in rel.as_posix(): continue
        yield file,rel.as_posix()
write_zip('mycoscout-quest-web.zip',[(p,p.relative_to(ROOT/'public').as_posix()) for p in (ROOT/'public').rglob('*') if p.is_file()])
pairs=[]
for folder in ['public','core','tests','adapters','android','docs','deploy','scripts','flight','config','vision','experiments','examples']:
    pairs.extend(descendants(folder))
for name in ['README.md','TRAINING_PIPELINE.md','THIRD_PARTY_NOTICES.md','package.json','server.mjs','START-MYCOSCOUT.cmd','.gitignore','.dockerignore']:
    pairs.append((ROOT/name,name))
for name in ['mycoscout-phone-debug.apk','mycoscout-quest-web.zip','ui-test-results.json','research-ui-checks.json','research-ui-checks-v2.json','release-checks.json','desktop-preview.png','phone-preview.png','scout-lab-desktop.png','scout-lab-phone.png','review-desktop.png','review-phone.png','review-ui-checks-v4.json','review-api-checks-v4.json','node-checks-v4.txt','far-lab-desktop.png','far-lab-phone.png','far-ui-checks-v5.json','node-checks-v5.txt','node-checks-v6.txt','testbench-normalizer-checks-v6.json','testbench-ui-checks-v6.json','testbench-desktop.png','testbench-phone.png','video-grader-desktop.png','video-grader-phone.png','video-grader-ui-checks-v7.json','node-checks-v7.txt','node-checks-v8.txt','python-live-checks-v8.txt','live-ui-checks-v8.json','live-desktop.jpg','live-phone.jpg','capture-api-checks-v8.txt','capture-race-checks-v8.txt']:
    if (DIST/name).exists():pairs.append((DIST/name,'dist/'+name))
write_zip('mycoscout-development-bundle.zip',pairs)
items=[]
for name in ['mycoscout-phone-debug.apk','mycoscout-quest-web.zip','mycoscout-development-bundle.zip']:
    p=DIST/name
    items.append({'file':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
(DIST/'SHA256SUMS.txt').write_text(''.join(i['sha256']+'  '+i['file']+'\n' for i in items),encoding='utf8')
(DIST/'artifacts.json').write_text(json.dumps({'version':'0.8.0','status':'development-preview','hardwareTested':False,'privatePhotosIncluded':False,'pretrainedModelWeightsIncluded':False,'sourceBundleIncludesPhotoRankerHead':True,'phoneAndQuestRunModels':False,'sourceBundleIncludesGroupClassifierHead':True,'artifacts':items},indent=2),encoding='utf8')
print(json.dumps(items,indent=2))
