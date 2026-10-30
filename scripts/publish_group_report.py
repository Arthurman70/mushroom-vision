"""Publish only measured aggregate results; no photos, personal data or model weights."""
import json
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
study=json.loads((ROOT/'experiments/results/groups-v1/study.json').read_text(encoding='utf8'))
manifest_path=ROOT/'data/public-fungi-v3/manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf8'))
assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()==study['manifestSha256']
count=len(manifest['images'])
classes=study['classes']
test=study['metrics']['test']
unsupported=[c['label'] for c in classes if not c['supported']]
summary=(f"{count:,} licensed photos; {sum(c['supported'] for c in classes)} supported research categories. "
         f"The first category guess matched {test['top1Accuracy']:.1%} of held-out observation labels; "
         f"the correct category appeared in the top three {test['top3Accuracy']:.1%} of the time. "
         "These are archive-photo results, not verified species or drone field accuracy.")
if unsupported:summary+=' Insufficient evidence: '+', '.join(unsupported)+'.'
report={'schema':'mycoscout.group-report.v1','status':'measured','evidenceVersion':study['evidenceVersion'],
    'photoCount':count,'newPhotoCount':sum(not r.get('inherited',False) for r in manifest['images']),
    'manifestSha256':study['manifestSha256'],'classes':classes,'metrics':study['metrics'],
    'scoreCalibration':study['scoreCalibration'],'abstention':study['abstention'],'summary':summary,
    'limitations':study.get('limitations',[]),'humanVerificationRequired':True,'hardwareTested':False}
(ROOT/'public/group-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps({'photoCount':count,'testTop1':test['top1Accuracy'],'testTop3':test['top3Accuracy'],'unsupported':unsupported}))
