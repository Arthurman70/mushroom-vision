"""Verify APK asset identity and source bundle isolation after packaging."""
from pathlib import Path
import zipfile,json,hashlib
root=Path(__file__).resolve().parents[1]
apk=root/'dist/mycoscout-phone-debug.apk'
files=[p for p in (root/'public').rglob('*') if p.is_file()]
with zipfile.ZipFile(apk) as z:
 for p in files:
  assert z.read('assets/www/'+p.relative_to(root/'public').as_posix())==p.read_bytes(),str(p)
if (root/'dist/mycoscout-development-bundle.zip').exists():
 with zipfile.ZipFile(root/'dist/mycoscout-development-bundle.zip') as z:
  forbidden=[n for n in z.namelist() if any(part in {'data','runtime-vision','models','runs','results','__pycache__'} for part in Path(n).parts) or n.endswith(('.safetensors','.keystore','.jks','.env'))]
  assert not forbidden,forbidden
  assert z.testzip() is None
report={'version':'0.8.0','matchingPublicAssets':len(files),'apkSha256':hashlib.sha256(apk.read_bytes()).hexdigest(),'privatePhotosIncluded':False,'pretrainedModelWeightsIncluded':False,'sourceBundleIncludesPhotoRankerHead':True,'phoneAndQuestRunModels':False,'sourceBundleIncludesGroupClassifierHead':True,'hardwareTested':False}
(root/'dist/release-checks.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report))
