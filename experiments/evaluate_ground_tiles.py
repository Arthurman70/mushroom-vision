"""Frozen native-image regression: whole image, whole-image tiles, manual lower ROI.

Never trains, alters human/AI reviews, promotes a model, or commands hardware.
AI negative references remain a separate diagnostic stratum from human labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vision.metrics import iou
from vision.video_ai_pass import atomic_json

DEFAULT_OUTPUT = ROOT/'experiments/results/ground-tiles-v1'
HUMAN_IDS = ['youtube-m9aGEh33w0I-043', 'youtube-m9aGEh33w0I-005',
             'youtube-m9aGEh33w0I-022', 'youtube-m9aGEh33w0I-040']
AI_IDS = ['youtube-OhN1BQgK2L8-008', 'youtube-OhN1BQgK2L8-048',
          'youtube-go_1s6wVScI-027', 'youtube-go_1s6wVScI-034']
REGION = [0, .35, 1, 1]
METHODS = [
    {'id':'original','label':'Whole original image','settings':{'methods':['original'],'tileGrid':2,'scale':1,'boxThreshold':.25},'passes':1},
    {'id':'full_tiles','label':'Whole-image 2×2 tiles + full context','settings':{'methods':['tiles'],'tileGrid':2,'scale':1,'boxThreshold':.25},'passes':5},
    {'id':'ground_tiles','label':'Manual lower-65% 2×2 tiles + full context','settings':{'methods':['tiles'],'tileGrid':2,'scale':1,'boxThreshold':.25,'tileRegion':REGION},'passes':5},
]
LIMITATIONS = [
    'Eight previously inspected native frames form a small correlated regression, not an independent woodland accuracy estimate.',
    'The three positive human regions and one human negative come from one source video. Unmatched positive-image alerts are not automatically false mushrooms.',
    'Four additional absence and object-region references are earlier full-original AI opinions, not human ground truth; their results are reported separately.',
    'AI-reference sources include train and validation partitions of an inactive candidate. This evaluates unchanged GroundingDINO, not that candidate or a generalization claim.',
    'The lower 65% is a manually chosen image-space region. No ground segmentation, depth, camera pose, safe corridor or mushroom habitat is inferred.',
    'Full-frame context is retained in both tiled methods. It can still miss mushrooms above the ROI; tile focus adds no detail there.',
    'Original pixel widths are measured annotation dimensions. No physical distance, flight visibility or optical resolution is inferred.',
    'Thresholds, weights, labels, frame selection and ROI are frozen before inference and are not tuned to these results.',
]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def checked_path(value):
    path = (ROOT/value).resolve(strict=True)
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError('Input must be a file inside the project')
    return path


def roi_relationship(box, bounds):
    x1,y1,x2,y2 = box; left,top,right,bottom = bounds
    if x2 <= left or x1 >= right or y2 <= top or y1 >= bottom:
        return 'outside'
    return 'inside' if left <= x1 and top <= y1 and x2 <= right and y2 <= bottom else 'straddles'


def match_predictions(truth, candidates):
    """Score-sorted, one-to-one matching at fixed IoU .5. Never duplicate credit."""
    ordered = sorted(enumerate(candidates), key=lambda pair: (-pair[1]['confidence'], pair[0]))
    available = set(range(len(truth))); pairs=[]; unmatched=[]
    for index,prediction in ordered:
        eligible = [(iou(prediction['bboxPixelsXYXY'], truth[t]['bbox']),t) for t in sorted(available)]
        overlap,target = max(eligible, default=(0,None), key=lambda pair: (pair[0],-pair[1]))
        if target is not None and overlap >= .5:
            available.remove(target);pairs.append({'predictionIndex':index,'truthIndex':target,'iou':overlap,'sourcePass':prediction.get('sourcePass')})
        else:
            unmatched.append(index)
    return {'truePositives':len(pairs),'falseNegatives':len(available),'matches':pairs,
            'unmatchedPredictionIndexes':unmatched,'missedTruthIndexes':sorted(available)}


def score_record(record, candidates):
    matched = match_predictions(record['truth'], candidates)
    positive = record['referenceStatus']=='visible'
    # Whole-frame absence can be scored only for an explicitly complete reference.
    negative = record['referenceStatus']=='absent' and record['referenceComplete'] is True
    matched.update({'predictions':len(candidates),'unmatchedPositiveAlerts':len(matched['unmatchedPredictionIndexes']) if positive else None,
                    'negativeFalseAlerts':len(candidates) if negative and record['referenceAuthority']=='human' else None,
                    'aiReferenceNegativeAlerts':len(candidates) if negative and record['referenceAuthority']=='ai' else None,
                    'confounderHits':[],'targetROI':[]})
    for region in record['confounders']:
        hits=[i for i,p in enumerate(candidates) if iou(p['bboxPixelsXYXY'],region['bboxPixelsXYXY'])>=.5]
        matched['confounderHits'].append({'kind':region['kind'],'bboxPixelsXYXY':region['bboxPixelsXYXY'],
            'overlappingAlertIndexes':hits,'criterion':'IoU >= 0.5 with an existing AI non-mushroom region','humanVerified':False})
    found={pair['truthIndex']:pair for pair in matched['matches']}
    for index,truth in enumerate(record['truth']):
        matched['targetROI'].append({'truthIndex':index,'sourceWidthPixels':truth['bbox'][2]-truth['bbox'][0],
            'sourceHeightPixels':truth['bbox'][3]-truth['bbox'][1],
            'relation':roi_relationship(truth['bbox'],record['roiBoundsPixelsXYXY']),
            'matched':index in found,'matchedBy':found.get(index,{}).get('sourcePass')})
    return matched


def summarize(records):
    rows=[]
    for method in METHODS:
        completed=[r for r in records if method['id'] in r.get('methods',{})]
        human_positive=[r for r in completed if r['referenceAuthority']=='human' and r['referenceStatus']=='visible']
        human_negative=[r for r in completed if r['referenceAuthority']=='human' and r['referenceStatus']=='absent']
        ai_negative=[r for r in completed if r['referenceAuthority']=='ai' and r['referenceStatus']=='absent']
        evidence=lambda r:r['methods'][method['id']]
        count=lambda items,key:sum(evidence(r)['metrics'].get(key) or 0 for r in items)
        rows.append({'method':method['id'],'framesCompleted':len(completed),
            'modelForwards':sum(evidence(r)['modelPasses'] for r in completed),
            'modelSeconds':sum(evidence(r)['seconds'] for r in completed),
            'meanSecondsPerFrame':sum(evidence(r)['seconds'] for r in completed)/len(completed) if completed else None,
            'humanPositiveFrames':len(human_positive),'truePositives':count(human_positive,'truePositives'),
            'falseNegatives':count(human_positive,'falseNegatives'),'unmatchedPositiveAlerts':count(human_positive,'unmatchedPositiveAlerts'),
            'humanNegativeFrames':len(human_negative),'negativeFalseAlerts':count(human_negative,'negativeFalseAlerts'),
            'aiNegativeReferenceFrames':len(ai_negative),'aiReferenceNegativeAlerts':count(ai_negative,'aiReferenceNegativeAlerts'),
            'aiConfounderRegionsWithHits':sum(bool(c['overlappingAlertIndexes']) for r in completed for c in evidence(r)['metrics']['confounderHits']),
            'outsideROIHumanTargets':sum(t['relation']=='outside' for r in human_positive for t in evidence(r)['metrics']['targetROI']),
            'straddlingROIHumanTargets':sum(t['relation']=='straddles' for r in human_positive for t in evidence(r)['metrics']['targetROI'])})
    return rows


def prepare(output):
    from vision.testbench import tile_region_bounds,region_tile_boxes,settings_for,pass_count
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('Protocol output already exists; choose a new directory')
    old_path=ROOT/'experiments/results/video-distance-v2/report.json'
    old=read(old_path)
    if old.get('status')!='completed' or old['sample']['labelAuthority']!='human':raise ValueError('Frozen human reference missing')
    collection_path=ROOT/'data/video-discovery-v1/manifest.json';collection=read(collection_path)
    source_frames={r['id']:r for r in collection['frames']}
    inventory_path=ROOT/'data/bulk-ai-review-v1/manifest.json';inventory=read(inventory_path)
    ledger_path=ROOT/'data/bulk-ai-review-v1/ledger.json';ledger=read(ledger_path)
    latest={}
    for row in sorted(ledger['records'],key=lambda r:r['createdAt']):
        if row['observation']['mode']=='original-image':latest[row['imageId']]=row
    inputs={relative(p):digest(p) for p in [old_path,collection_path,inventory_path,ledger_path,
        ROOT/'vision/testbench.py',ROOT/'adapters/scout_frame.py',Path(__file__)]}
    for path in (ROOT/'models/grounding-dino-tiny').iterdir():
        if path.is_file():inputs[relative(path)]=digest(path)
    references=[]
    for frame_id in HUMAN_IDS:
        native=next(r for r in old['records'] if r['frameId']==frame_id and r['condition']=='native')
        frame=source_frames[frame_id]
        if native['imageSha256']!=frame['sha256']:raise ValueError('Human image identity changed')
        references.append({'frameId':frame_id,'path':relative(ROOT/'data/video-discovery-v1'/frame['path']),
            'sha256':frame['sha256'],'sourceId':frame['sourceId'],'sourceGroupId':native['sourceGroupId'],
            'referenceAuthority':'human','referenceStatus':'visible' if native['kind']=='positive' else 'absent',
            'referenceComplete':True,'truth':native['truth'],'confounders':[],
            'referenceEvidence':{'reportSha256':digest(old_path),'recordId':native['id']},
            'datasetRole':'previously inspected human regression reference'})
    for frame_id in AI_IDS:
        source=next(r for r in inventory['records'] if any(a['id']==frame_id for a in r['aliases']))
        audit=latest[source['id']]
        if audit['sha256']!=source['sha256'] or audit['status']!='absent' or audit['complete'] is not True or not audit['confounders']:
            raise ValueError('A selected AI reference is not a complete prior original absence review')
        references.append({'frameId':frame_id,'path':source['path'],'sha256':source['sha256'],'sourceId':source['sourceId'],
            'sourceGroupId':source['sourceGroupId'],'referenceAuthority':'ai','referenceStatus':'absent','referenceComplete':True,
            'truth':[],'confounders':audit['confounders'],'referenceEvidence':{'reviewId':audit['id'],'revision':audit['revision'],
                'ledgerSha256':digest(ledger_path),'evidence':audit['evidence'],'humanVerified':False},
            'datasetRole':'previous AI reference; inactive candidate source partition '+source['split']})
    for record in references:
        path=checked_path(record['path'])
        if digest(path)!=record['sha256']:raise ValueError('Original image bytes changed')
        inputs[record['path']]=record['sha256']
        with Image.open(path) as opened:width,height=ImageOps.exif_transpose(opened).size
        record.update(width=width,height=height,roiBoundsPixelsXYXY=list(tile_region_bounds(width,height,REGION)),
            fullTileBoundsPixelsXYXY=[list(b) for b in region_tile_boxes(width,height,2)],
            groundTileBoundsPixelsXYXY=[list(b) for b in region_tile_boxes(width,height,2,REGION)])
        record['sourceTargetWidthsPixels']=[r['bbox'][2]-r['bbox'][0] for r in record['truth']]
        if sum(pass_count((width,height),settings_for(m['settings'])) for m in METHODS)!=11:raise ValueError('Unexpected inference budget')
    native_result=read(ROOT/'experiments/results/video-distance-v2/predictions'/(old['records'][0]['id']+'.json'))
    protocol={'schema':'mycoscout.ground-tiles-protocol.v1','createdAt':datetime.now(timezone.utc).isoformat(),
        'selection':'Fixed existing native human benchmark plus the first chosen complete AI hat, wood, bark and trunk references; no new labels or tuning',
        'methods':METHODS,'frameOrder':[r['frameId'] for r in references],'methodOrder':[m['id'] for m in METHODS],
        'references':references,'inputSha256':inputs,'expectedModel':{k:native_result['model'][k] for k in ['model','resolvedRevision','prompt']},
        'cpuThreads':4,'device':'cpu','scoreThreshold':.25,'textThreshold':.25,'iouThreshold':.5,'nmsIou':.5,'overlapFraction':.2,
        'roi':REGION,'roiRounding':'floor left/top; ceil right/bottom','fullFrameContext':True,
        'plannedModelForwards':88,'reuseInference':False,'trainingPerformed':False,'modelActivation':False,
        'actualDistanceM':None,'actuationEnabled':False,'limitations':LIMITATIONS,
        'previousV2Comparison':[{k:r[k] for k in ['id','raw','color','tiles']} for r in old['rows']]}
    output.mkdir(parents=True);(output/'predictions').mkdir()
    with (output/'protocol.json').open('x',encoding='utf-8') as stream:json.dump(protocol,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'prepared':relative(output),'protocolSha256':digest(output/'protocol.json'),'frames':len(references),'plannedModelForwards':88}),flush=True)
    return protocol


def verify_inputs(protocol):
    for path,wanted in protocol['inputSha256'].items():
        if digest(checked_path(path))!=wanted:raise ValueError('Frozen input changed: '+path)


def run(output):
    from vision.testbench import compare
    from adapters.scout_frame import LocalDetector
    output=Path(output).resolve();protocol=read(output/'protocol.json');protocol_hash=digest(output/'protocol.json')
    if (output/'report.json').exists():raise FileExistsError('An evaluation has already started here; no overwrite or silent reuse')
    if protocol.get('schema')!='mycoscout.ground-tiles-protocol.v1' or protocol['plannedModelForwards']!=88:raise ValueError('Unknown protocol')
    verify_inputs(protocol)
    started=time.perf_counter()
    detector=LocalDetector(ROOT/'models/grounding-dino-tiny',device='cpu',threads=4,box_threshold=.25,text_threshold=.25)
    for key,value in protocol['expectedModel'].items():
        if detector.info[key]!=value:raise ValueError('Frozen detector changed: '+key)
    report={'schema':'mycoscout.ground-tiles-report.v1','status':'running','protocolSha256':protocol_hash,'protocolPath':relative(output/'protocol.json'),
        'model':detector.info,'modelLoadSeconds':time.perf_counter()-started,'progress':{'completed':0,'total':88,'unit':'model-forwards'},
        'records':[],'summary':[],'actualDistanceM':None,'trainingPerformed':False,'actuationEnabled':False,'limitations':LIMITATIONS}
    atomic_json(output/'report.json',report)
    for reference in protocol['references']:
        record={**reference,'methods':{}};report['records'].append(record)
        with Image.open(checked_path(reference['path'])) as source:image=ImageOps.exif_transpose(source).convert('RGB')
        for method in protocol['methods']:
            result=compare(image,reference['sha256'],method['settings'],detector)
            if result['modelPasses']!=method['passes']:raise ValueError('Unexpected model forwards')
            result_path=output/'predictions'/(reference['frameId']+'--'+method['id']+'.json')
            with result_path.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
            observed=result['methods'][0]
            record['methods'][method['id']]={'metrics':score_record(reference,observed['candidates']),
                'seconds':observed['seconds'],'wallSeconds':result['seconds'],'modelPasses':result['modelPasses'],
                'predictionPath':relative(result_path),'predictionSha256':digest(result_path),
                'passTimings':observed.get('passTimings',[]),'tileBoundsPixelsXYXY':observed.get('tileBoundsPixelsXYXY',[])}
            report['progress']['completed']+=result['modelPasses'];report['elapsedSeconds']=time.perf_counter()-started
            report['summary']=summarize(report['records']);atomic_json(output/'report.json',report)
            metrics=record['methods'][method['id']]['metrics']
            print(json.dumps({'frame':reference['frameId'],'method':method['id'],'progress':report['progress']['completed'],
                'total':88,'seconds':round(observed['seconds'],2),'elapsedSeconds':round(report['elapsedSeconds'],2),
                'matched':metrics['truePositives'],'missed':metrics['falseNegatives'],'alerts':metrics['predictions']}),flush=True)
    verify_inputs(protocol)
    if digest(output/'protocol.json')!=protocol_hash:raise ValueError('Protocol changed during evaluation')
    report.update(status='completed',completedAt=datetime.now(timezone.utc).isoformat(),inputIntegrityVerified=True)
    atomic_json(output/'report.json',report)
    print(json.dumps({'completed':True,'report':relative(output/'report.json'),'summary':report['summary']}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args();(prepare if args.action=='prepare' else run)(args.output)
