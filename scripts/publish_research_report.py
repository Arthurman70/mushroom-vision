"""Publish aggregate experiment evidence only; never copies private source photos."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
def read(p): return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
study=read('vision/runs/facebook-pilot-01/study.json')
scale=read('vision/runs/facebook-pilot-01/fixedcanvas-scale-results.json')
full=read('experiments/results/facebook-fullframe-nms-2026-10-26/summary.json')
verifier=study['verifierCropTraining']
report={
 'schema':'mycoscout.public-research-report.v1','generatedAt':study['createdAt'],
 'summary':{'imageCount':17,'sourceGroupCount':14,'annotatedCount':17,'fullyAnnotatedCount':5,'trainCount':10,'validationCount':4,'testCount':3,'status':'Pilot measured','verifierStatus':'The crop filter was trained, but remains experimental and is not enabled for flight or automatic rejection.'},
 'runs':[
 {'model':'GroundingDINO Tiny','stage':'Original photos','measured':True,'synthetic':False,'metrics':study['detectorAllComplete'],'notes':['5 fully annotated photos: 8 of 14 labeled mushrooms found; 2 false-positive boxes. Approximate visual labels, pending human review.','12 partially annotated colony photos are excluded from whole-image detection metrics.','All 17 photos were processed locally. Mean inference: '+str(round(full['meanInferenceSeconds'],2))+' seconds on this Windows CPU. Jetson speed is unmeasured.','The background-only landscape produced a false positive. No real distant mushroom-positive calibration was collected.']},
 {'model':'Trained crop filter','stage':'Held-out crops','measured':True,'synthetic':False,'metrics':verifier['cropMetrics']['test'],'notes':['Actual training: 1,000 iterations on 34 labeled crops from 8 uploader groups. Validation: 27 crops / 3 groups. Test: 9 crops / 3 groups.','Held-out crop accuracy: 55.6% (5/9). This is weak evidence and is not an end-to-end detector score.','GroundingDINO weights were not fine-tuned. The fitted model is a small mushroom/background candidate filter; it cannot identify species.']},
 {'model':'GroundingDINO Tiny','stage':'Synthetic size progression','measured':True,'synthetic':True,'sizeBins':[{'label':str(float(k)*100).rstrip('0').rstrip('.')+'% scene','recall':v['recall'],'groundTruth':v['groundTruth']} for k,v in scale['byScale'].items()], 'notes':['The same 2 original photos were scaled into a fixed-size canvas at 100%, 50%, 25%, 12.5% and 6.25%. Blur, JPEG and gray padding are artificial.','Recall fell from 6/12 to 2/12 labeled targets. Repeated variants are not independent mushrooms or measured camera distances.']}
 ],
 'notes':['Precision is the share of proposed boxes that matched a labeled mushroom. Recall is the share of labeled mushrooms found.','Sources: Michigan Fungus Hunters Facebook group, 17 photos from 14 uploaders. Uploader groups stay in one split. Source authenticity and capture locations are unverified.','Tiled inspection took about 41–43 seconds per photo on this computer. More boxes on dense colonies do not prove better accuracy.','No species, edibility, metric mushroom coordinates, real flight tests or calibrated detection range are established. Private photos and trained research checkpoint remain local.'],
 'model':study['detector'],'sourceManifestSha256':study['sourceManifestSha256'],
 'scaleEvidence':scale['byScale']
}
(ROOT/'public/research-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print('Wrote public aggregate research report; no private photos or uploader identifiers.')
