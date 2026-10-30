"""Fresh raw active-detector pass on exactly the saved human-box pixel views."""
from pathlib import Path
from datetime import datetime, timezone
import copy
import json
import shutil
import sys
import time

from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.video_distance_retest import read, digest, write_examples
from experiments.video_distance_benchmark import metric_counts
from vision.video_ai_pass import atomic_json
from vision.testbench import compare
from adapters.scout_frame import LocalDetector


def run(output=ROOT/'experiments/results/video-distance-v3', previous=ROOT/'experiments/results/video-distance-v2'):
    if output.exists():
        raise ValueError('Frozen active-detector rerun already exists')
    old=read(previous/'report.json')
    if old.get('status')!='completed' or old['sample']['labelAuthority']!='human':
        raise ValueError('Complete frozen human-box reference required')
    hashes={previous/'report.json':digest(previous/'report.json')}
    for row in old['records']:
        path=(previous/row['imagePath']).resolve()
        if not path.is_relative_to(previous.resolve()) or digest(path)!=row['derivedImageSha256']:
            raise ValueError('Frozen pixel views changed')
        hashes[path]=row['derivedImageSha256']
    detector=LocalDetector(ROOT/'models/grounding-dino-tiny',device='cpu',threads=4,box_threshold=.25,text_threshold=.25)
    old_model=read(previous/'predictions'/(old['records'][0]['id']+'.json'))['model']
    for key in ('model','resolvedRevision','prompt'):
        if detector.info[key]!=old_model[key]:
            raise ValueError('Active detector differs from frozen reference: '+key)
    output.mkdir(parents=True);(output/'images').mkdir();(output/'predictions').mkdir()
    protocol={**copy.deepcopy(old['protocol']),'schema':'mycoscout.active-video-distance-protocol.v1',
        'frozenPreviousReportSha256':digest(previous/'report.json'),'methods':['raw'],
        'boxThreshold':.25,'textThreshold':.25,'scoreThreshold':.25,'iouThreshold':.5,
        'reuseInference':False,'cpuThreads':4,'plannedModelForwards':len(old['records']),
        'plannedNewForwards':len(old['records']),'plannedReusedNativeForwards':0,'tiledConditions':[],
        'model':detector.info,'trainingPerformed':False,'modelActivation':False,'actualDistanceM':None,
        'purpose':'Post-corpus-review fresh check of unchanged active detector; independent of inactive candidate'}
    protocol.pop('classifierProvenance',None);protocol.pop('classification',None)
    atomic_json(output/'protocol.json',protocol)
    report={'schema':'mycoscout.video-distance-report.v1','runVersion':'video-distance-v3','status':'running',
        'generatedAt':datetime.now(timezone.utc).isoformat(),'sample':copy.deepcopy(old['sample']),
        'progress':{'completed':0,'total':len(old['records']),'unit':'model-forwards'},
        'protocol':protocol,'rows':[{**{k:r[k] for k in ('id','label','native','targetPixels','realizedTargetWidths','realizedTargetWidthPixels')},
            'raw':None,'color':None,'tiles':None} for r in old['rows']],
        'records':[],'classificationRows':[],'nativeBaseline':None,'nativeSmallPositiveOnly':None,'previousComparison':[],
        'actualDistanceM':None,'actuationEnabled':False,
        'limitations':['This is a fresh raw-only pass with unchanged GroundingDINO weights and fixed thresholds.',
            'Color enhancement and tiles were not rerun here; their earlier v2 experiment is preserved on disk.',
            'The AI-trained candidate is inactive and reported separately. Its training did not change this detector.',
            'Only three correlated positive frames and one negative original are represented; this is not woodland population accuracy.',
            'The frozen test was already inspected. Repeating it provides a regression check, not independent validation.',
            'Scaled widths simulate pixel footprints, not measured meters, camera approach, blur or occlusion.',
            'Unmatched alerts on positive images and false alerts on explicit negative controls are reported separately.']}
    atomic_json(output/'report.json',report)
    started=time.perf_counter()
    for source in old['records']:
        row={k:copy.deepcopy(v) for k,v in source.items() if k not in {'predictions','classification','crop'}}
        path=output/row['imagePath'];shutil.copyfile(previous/row['imagePath'],path)
        with Image.open(path) as opened:image=opened.convert('RGB')
        result=compare(image,row['derivedImageSha256'],{'methods':['original'],'tileGrid':2,'scale':1,'boxThreshold':.25},detector)
        atomic_json(output/'predictions'/(row['id']+'.json'),result)
        row['predictions']={'raw':[{'id':b['id'],'bbox':b['bboxPixelsXYXY'],'score':b['confidence']} for b in result['methods'][0]['candidates']]}
        row['reusedInference']=False;report['records'].append(row)
        report['progress']['completed']+=result['modelPasses']
        report['elapsedSeconds']=time.perf_counter()-started
        selected=[r for r in report['records'] if r['condition']==row['condition']]
        if len(selected)==sum(r['condition']==row['condition'] for r in old['records']):
            summary=next(r for r in report['rows'] if r['id']==row['condition'])
            summary['raw']=metric_counts(selected,'raw');summary['complete']=True
            earlier=next(r for r in old['rows'] if r['id']==row['condition'])['raw']
            report['previousComparison'].append({'id':row['condition'],'matchedDelta':summary['raw']['matched']-earlier['matched'],
                'falsePositiveDelta':summary['raw']['falsePositives']-earlier['falsePositives'],
                'negativeFalsePositiveDelta':summary['raw']['negativeFalsePositives']-earlier['negativeFalsePositives']})
            print(json.dumps({'condition':row['condition'],'raw':summary['raw']}),flush=True)
        atomic_json(output/'report.json',report)
    if any(digest(path)!=expected for path,expected in hashes.items()):
        raise ValueError('Frozen reference changed during evaluation')
    write_examples(output,report['records'])
    report['status']='completed';report['generatedAt']=datetime.now(timezone.utc).isoformat()
    atomic_json(output/'report.json',report)
    return report


if __name__=='__main__':run()
