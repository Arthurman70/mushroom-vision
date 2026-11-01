"""Independent, read-only audit of completed V5 predictions and localization counts.

No detector, evaluator or training module is imported. Only a new verification
receipt may be written, after every completed-output check has passed.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments/results/full-pixel-v5'
LEVELS=[('native',None),('64px',64.),('32px',32.),('16px',16.),('6p9px',6.9),('3p45px',3.45),('1p38px',1.38),('0p69px',.69)]
SETTINGS={'methods':['original','tiles'],'tileGrid':2,'scale':1,'boxThreshold':.25}
COHORTS={'human-video':'Saved human video references','ai-photo':'Incomplete AI photo references'}

def require(condition,message):
    if not condition:raise ValueError(message)

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def load(path):
    with Path(path).open(encoding='utf8') as stream:return json.load(stream)

def checked(path,root=ROOT):
    value=(root/path).resolve(strict=True)
    require(value.is_relative_to(root.resolve()) and value.is_file(),'Input outside project or not a regular file')
    return value

def read_hashed_json(path,expected):
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected,'Saved prediction hash mismatch')
    return json.loads(raw)

def number(value):return type(value) in (float,int) and math.isfinite(value)

def close(a,b):return number(a) and number(b) and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10)

def same(a,b):
    if isinstance(a,dict) and isinstance(b,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
    if isinstance(a,list) and isinstance(b,list):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    if number(a) and number(b):return close(a,b)
    return type(a) is type(b) and a==b

def overlap(a,b):
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    total=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/total if total else 0.

def recompute_metrics(box,candidates,authority):
    if box is None:
        return {'hits':None,'targetCount':0,'unmatchedPositiveBoxes':None,'extraOverlappingAlerts':None,
                'negativeAlerts':len(candidates),'negativeHasAlerts':bool(candidates),
                'humanNegativeFalseAlerts':len(candidates) if authority=='human' else None,
                'aiNegativeOpinionAlerts':len(candidates) if authority=='ai' else None}
    overlaps=[overlap(box,p['bbox']) for p in candidates]
    eligible=[i for i,value in enumerate(overlaps) if value>=.5]
    winner=max(eligible,key=lambda i:(candidates[i]['score'],overlaps[i],-i)) if eligible else None
    hit=int(winner is not None)
    return {'hits':hit,'targetCount':1,'matchedPredictionIndex':winner,'bestIoU':max(overlaps,default=0.),
            'unmatchedPositiveBoxes':len(candidates)-hit,'extraOverlappingAlerts':max(0,len(eligible)-1),
            'humanCompletePositiveFalseAlerts':len(candidates)-hit if authority=='human' else None,
            'negativeAlerts':None,'negativeHasAlerts':None}

def expected_tiles(width,height):
    def intervals(length):
        size=min(length,math.ceil(length/1.8));return [(start,start+size) for start in sorted({0,length-size})]
    return [[x1,y1,x2,y2] for y1,y2 in intervals(height) for x1,x2 in intervals(width)]

def validate_prediction_record(job,record,prediction,expected_model):
    for key,value in job.items():
        if key!='methods':require(key in record and same(value,record[key]),'Report input differs from frozen job: '+key)
    require(record.get('uniqueModelForwards')==5,'Wrong scene forward count')
    require(set(record.get('methods',{}))=={'raw','tiles'},'Missing paired methods')
    require(prediction.get('schema')=='mycoscout.testbench-result.v1' and prediction.get('modelPasses')==5,'Wrong prediction schema or forward count')
    require(same(prediction.get('settings'),SETTINGS),'Prediction settings differ from fixed experiment')
    image=prediction.get('image',{})
    require(image.get('sha256')==job['derivedImageSha256'] and image.get('width')==job['width'] and image.get('height')==job['height'] and
            image.get('workingWidth')==job['width'] and image.get('workingHeight')==job['height'],'Prediction image identity or scale mismatch')
    for key,value in expected_model.items():require(prediction.get('model',{}).get(key)==value,'Prediction model identity mismatch')
    for key,value in {'boxThreshold':.25,'textThreshold':.25,'nmsIou':.5,'downloadsEnabled':False,'device':'cpu','threads':4}.items():
        require(prediction['model'].get(key)==value,'Prediction model settings mismatch: '+key)
    require([m.get('id') for m in prediction.get('methods',[])]==['original','tiles'],'Unexpected prediction methods')
    tiles=expected_tiles(job['width'],job['height']);full=[0,0,job['width'],job['height']]
    bounds={'original':full,**{f'tile_{i+1}':box for i,box in enumerate(tiles)}}
    recalculated={}
    for method,name,passes,reused in zip(prediction['methods'],['raw','tiles'],[1,5],[0,1]):
        require(method.get('passes')==passes and method.get('reusedPasses')==reused,'Context reuse or tile count changed')
        require(number(method.get('seconds')) and method['seconds']>=0,'Invalid inference time')
        rows=method.get('candidates');require(isinstance(rows,list) and len(rows)<=100,'Invalid candidate list')
        candidates=[]
        for row in rows:
            box=row.get('bboxPixelsXYXY');score=row.get('confidence');origin=row.get('sourcePass')
            require(isinstance(box,list) and len(box)==4 and all(number(n) for n in box) and number(score) and .25<=score<=1,'Invalid candidate geometry or score')
            require(0<=box[0]<box[2]<=job['width'] and 0<=box[1]<box[3]<=job['height'],'Candidate outside full image')
            require(origin in bounds and (name=='tiles' or origin=='original'),'Candidate came from an unknown pass')
            b=bounds[origin]
            require(b[0]-1e-8<=box[0] and b[1]-1e-8<=box[1] and box[2]<=b[2]+1e-8 and box[3]<=b[3]+1e-8,'Candidate projection escaped its tile')
            candidates.append({'bbox':box,'score':score,'sourcePass':origin})
        require(all(candidates[i]['score']>=candidates[i+1]['score'] for i in range(len(candidates)-1)),'Saved NMS candidates are not score ordered')
        require(all(overlap(a['bbox'],b['bbox'])<=.5+1e-12 for i,a in enumerate(candidates) for b in candidates[i+1:]),'Saved NMS candidates violate frozen overlap limit')
        if name=='tiles':
            require(method.get('fullFrameContext') is True and method.get('tileRegion')==[0.,0.,1.,1.],'Unexpected tile region')
            require(method.get('tileBoundsWorkingPixelsXYXY')==tiles and same(method.get('tileBoundsPixelsXYXY'),tiles),'Wrong tile coverage')
            timing=method.get('passTimings',[])
            require(len(timing)==5 and [p.get('kind') for p in timing]==['context','tile','tile','tile','tile'],'Missing raw or tile forwards')
            require(all(same(item.get('bboxPixelsXYXY'),box) and number(item.get('seconds')) and item['seconds']>=0 for item,box in zip(timing,[full,*tiles])),'Invalid pass timing geometry')
            require(close(timing[0]['seconds'],prediction['methods'][0]['seconds']),'Tiled context does not share raw timing')
            require(close(sum(p['seconds'] for p in timing),method['seconds']) and close(sum(p['seconds'] for p in timing[1:]),method.get('additionalSeconds')),'Tiled elapsed time is inconsistent')
        metrics=recompute_metrics(job['targetPixelsXYXY'],candidates,job['referenceAuthority'])
        expected={'candidates':candidates,'metrics':metrics,'seconds':method['seconds'],'modelPasses':passes,'sharedContextPasses':reused,'passTimings':method.get('passTimings',[])}
        require(same(record['methods'][name],expected),'Report candidates, scores or metrics disagree with saved prediction: '+name)
        recalculated[name]=expected
    require(number(prediction.get('seconds')) and prediction['seconds']>=0 and close(record.get('sceneWallSeconds'),prediction['seconds']),'Scene wall time differs from prediction')
    return {**job,'methods':recalculated,'sceneWallSeconds':prediction['seconds']}

def aggregate_independent(records):
    cohorts=[]
    for cohort,label in COHORTS.items():
        conditions=[]
        for level,pixels in LEVELS:
            rows=[r for r in records if r['cohort']==cohort and r['conditionId']==level]
            for method in ['raw','tiles']:
                if not rows:continue
                positive=[r for r in rows if not r['isNegativeControl']];negative=[r for r in rows if r['isNegativeControl']]
                metric=lambda row:row['methods'][method]['metrics']
                conditions.append({'id':level,'mode':method,'targetPixels':pixels,'hits':sum(metric(r)['hits'] for r in positive),'targetCount':len(positive),
                    'unmatchedPositiveBoxes':sum(metric(r)['unmatchedPositiveBoxes'] for r in positive),'extraOverlappingAlerts':sum(metric(r)['extraOverlappingAlerts'] for r in positive),
                    'negativeAlertCount':sum(metric(r)['negativeAlerts'] for r in negative) if negative else None,
                    'negativeFramesWithAlerts':sum(metric(r)['negativeHasAlerts'] for r in negative) if negative else None,'negativeFrameCount':len(negative) if negative else None,
                    'meanSecondsPerFrame':sum(r['methods'][method]['seconds'] for r in rows)/len(rows),'measurementFrames':len(rows),
                    'realizedTargetWidths':[r['realizedTargetWidthPixels'] for r in positive],'realizedTargetHeights':[r['realizedTargetHeightPixels'] for r in positive]})
        cohorts.append({'id':cohort,'label':label,'referenceAuthority':'human' if cohort=='human-video' else 'ai-incomplete',
            'targetCount':3 if cohort=='human-video' else 9,'negativeFrameCount':1 if cohort=='human-video' else 4,'conditions':conditions})
    return cohorts

def verify_report_data(protocol,report,prediction_loader):
    require(protocol.get('schema')=='mushroom-vision.full-pixel-v5-protocol.v1' and protocol.get('status')=='frozen','Invalid frozen protocol')
    require(report.get('schema')=='mushroom-vision.full-pixel-v5-report.v1' and report.get('status')=='completed' and report.get('inputIntegrityVerified') is True,'Only completed runs can be verified')
    require(report.get('trainingPerformed') is False and report.get('weightsChanged') is False and report.get('actualDistanceM') is None,'Unexpected training, weight or range claims')
    require(report.get('progress')=={'completed':505,'total':505,'scenesCompleted':101,'scenesTotal':101},'Incomplete inference budget')
    require(protocol.get('plannedModelForwards')==505 and protocol.get('settings')==SETTINGS and protocol.get('levels')==[{'id':a,'targetPixels':b} for a,b in LEVELS],'Protocol comparison differs')
    refs=protocol['references'];require(len(refs)==17 and len({r['frameId'] for r in refs})==17,'Duplicate or missing source references')
    require(Counter((r['referenceAuthority'],r['isNegativeControl']) for r in refs)=={('human',False):3,('ai',False):9,('human',True):1,('ai',True):4},'Reference authority counts changed')
    by_id={r['frameId']:r for r in refs}
    for r in refs:
        require(r['referenceComplete'] is (r['referenceAuthority']=='human' or r['isNegativeControl']),'Reference completeness changed')
        require((r['originalTargetPixelsXYXY'] is None)==r['isNegativeControl'],'Reference target/absence conflict')
        require(r['cohort']==('human-video' if r['referenceAuthority']=='human' else 'ai-photo'),'Cohort authority mismatch')
    require(report.get('frames')==refs,'Report references differ from frozen protocol')
    expected_keys={(r['frameId'],level) for r in refs for level,pixels in LEVELS if pixels is None or not r['isNegativeControl']}
    jobs=protocol['jobs'];records=report['records'];require(len(jobs)==len(records)==101,'Expected exactly 101 scene jobs and records')
    job_map={(j['frameId'],j['conditionId']):j for j in jobs};record_map={(r['frameId'],r['conditionId']):r for r in records}
    require(len(job_map)==len(record_map)==101 and set(job_map)==set(record_map)==expected_keys,'Missing, duplicated or unexpected scene')
    audited=[]
    for index,job in enumerate(jobs):
        require(job['order']==index,'Frozen job order changed');ref=by_id[job['frameId']]
        for key in ('referenceAuthority','referenceComplete','cohort','isNegativeControl','originalTargetPixelsXYXY'):require(job[key]==ref[key],'Job reference binding changed')
        require(job['sourceImageSha256']==ref['imageSha256'],'Job source hash differs from reference')
        box=job['targetPixelsXYXY'];geometry=job['geometry'];native=job['conditionId']=='native'
        if native:
            require(box==ref['originalTargetPixelsXYXY'] and job['derivedImageSha256']==ref['imageSha256'] and (job['width'],job['height'])==(ref['width'],ref['height']),'Native original geometry changed')
        else:
            require((job['width'],job['height'])==(1920,1080) and geometry['supersampling']==16 and geometry['sceneReEnlarged'] is False,'Wrong synthetic canvas or restored geometry')
            original=ref['originalTargetPixelsXYXY'];sx,sy=geometry['sceneScaleXY'];left,top,_,_=geometry['sceneBoundsPixelsXYXY']
            projected=[original[0]*sx+left,original[1]*sy+top,original[2]*sx+left,original[3]*sy+top]
            require(same(box,projected),'Synthetic target does not follow frozen scene transform')
        if box is not None:require(close(job['realizedTargetWidthPixels'],box[2]-box[0]) and close(job['realizedTargetHeightPixels'],box[3]-box[1]),'Realized target size mismatch')
        record=record_map[(job['frameId'],job['conditionId'])];prediction=prediction_loader(record)
        audited.append(validate_prediction_record(job,record,prediction,protocol['expectedModel']))
    cohorts=aggregate_independent(audited)
    require(same(report.get('cohorts'),cohorts),'Report aggregate counts or timings do not match saved predictions')
    require(close(report.get('summedSceneInferenceSeconds'),sum(r['sceneWallSeconds'] for r in audited)),'Report cumulative inference time mismatch')
    for key,value in protocol['expectedModel'].items():require(report.get('model',{}).get(key)==value,'Report model identity changed')
    return cohorts

def validate_recovery_data(protocol,report,plan,receipt,failed,hashes,prediction_hashes):
    """Audit the preserved failure and exact reused/new partition, not attempts."""
    require(plan.get('schema')=='mushroom-vision.full-pixel-recovery-plan.v1','Invalid recovery plan')
    require(receipt.get('schema')=='mushroom-vision.full-pixel-recovery-receipt.v1' and receipt.get('status')=='completed','Incomplete recovery receipt')
    recovery=report.get('recovery');require(isinstance(recovery,dict),'Missing recovery disclosure')
    for value in (plan,receipt,recovery):
        require(value.get('reusedScenes')==92 and value.get('newScenes')==9,'Recovery reused/new count mismatch')
        require('totalAttemptedModelForwards' in value and value['totalAttemptedModelForwards'] is None,'Interrupted attempted work must remain unknown')
        require(isinstance(value.get('disclosure'),str) and 'unknown' in value['disclosure'],'Missing unknown-attempt disclosure')
        require(value['disclosure']==plan['disclosure'],'Recovery disclosure changed')
    require(plan.get('uniquePlannedModelForwards')==505 and plan.get('newRecoveryModelForwards')==45 and
            plan.get('workers')==2 and plan.get('torchThreadsPerWorker')==4,'Recovery plan execution settings changed')
    require(receipt.get('uniquePersistedModelForwards')==505 and receipt.get('newRecoveryModelForwards')==45 and
            recovery.get('uniquePersistedModelForwards')==505,'Recovery persisted count changed')
    require(receipt.get('trainingPerformed') is False and receipt.get('weightsChanged') is False and
            receipt.get('elapsedWallSecondsIncludesInterruption') is True,'Recovery work or timing disclosure changed')
    for value in (plan,receipt):
        require(value.get('protocolSha256')==hashes['protocolSha256'] and value.get('failedReportSha256')==hashes['failedReportSha256'] and
                value.get('recoveryWrapperSha256')==hashes['recoveryWrapperSha256'],'Recovery provenance hashes disagree')
        require(same(value.get('settings'),protocol['settings']) and value.get('softwareVersions')==protocol['softwareVersions'],'Recovery settings or versions changed')
    require(plan.get('expectedModel')==protocol['expectedModel'],'Recovery expected model changed')
    require(receipt.get('planSha256')==recovery.get('planSha256')==hashes['planSha256'] and
            recovery.get('failedReportSha256')==hashes['failedReportSha256'] and
            recovery.get('receiptSha256')==hashes['receiptSha256'],'Recovery report does not bind plan and receipt')
    require(failed.get('status')=='failed' and failed.get('error',{}).get('type')=='PermissionError' and
            failed.get('protocolSha256')==hashes['protocolSha256'],'Archived failure does not match interrupted run')
    require(failed.get('progress')=={'completed':460,'total':505,'scenesCompleted':92,'scenesTotal':101},'Archived saved progress changed')
    jobs={(j['frameId'],j['conditionId']):j for j in protocol['jobs']}
    old={(r['frameId'],r['conditionId']):r for r in failed['records']}
    current={(r['frameId'],r['conditionId']):r for r in report['records']}
    ledger={(r['frameId'],r['conditionId']):r for r in plan['sourcePredictions']}
    missing={(r['frameId'],r['conditionId']):r for r in plan['missingJobs']}
    require(len(old)==len(failed['records'])==len(ledger)==len(plan['sourcePredictions'])==92 and
            len(missing)==len(plan['missingJobs'])==9,'Recovery evidence has missing or duplicate identities')
    require(set(old)==set(ledger) and set(old).isdisjoint(missing) and set(old)|set(missing)==set(jobs)==set(current),'Recovery is not the exact frozen saved/missing partition')
    for identity,item in ledger.items():
        original=old[identity];final=current[identity]
        require(item.get('recordOrder')==jobs[identity]['order'] and item.get('sourcePath')==original['predictionPath'] and
                item.get('sha256')==original['predictionSha256']==final['predictionSha256'],'Reused prediction identity or bytes changed')
        require(same({k:v for k,v in original.items() if k!='predictionPath'},
                     {k:v for k,v in final.items() if k!='predictionPath'}),'Reused record was changed during recovery')
    for identity,item in missing.items():require(item.get('order')==jobs[identity]['order'],'Recovered missing job order changed')
    require(len(prediction_hashes)==101 and receipt.get('predictionHashes')==prediction_hashes,'Recovery receipt does not bind every persisted prediction')
    require(close(receipt.get('originalReportedWallSeconds'),failed.get('elapsedWallSeconds')) and
            number(receipt.get('recoveryWallSeconds')) and receipt['recoveryWallSeconds']>=0,'Recovery phase timings changed')
    return {'status':'verified','planSha256':hashes['planSha256'],'receiptSha256':hashes['receiptSha256'],
            'failedReportSha256':hashes['failedReportSha256'],'recoveryWrapperSha256':hashes['recoveryWrapperSha256'],
            'reusedScenes':92,'newScenes':9,'uniquePersistedModelForwards':505,'newRecoveryModelForwards':45,
            'totalAttemptedModelForwards':None,'elapsedWallSecondsIncludesInterruption':True,'disclosure':plan['disclosure']}

def verify_recovery_evidence(output,root,protocol,report,protocol_hash,prediction_hashes):
    paths={name:output/file for name,file in [('plan','recovery-plan.json'),('receipt','recovery-receipt.json'),('failed','failed-report.json')]}
    if 'recovery' not in report:
        require(not any(p.exists() for p in paths.values()),'Recovery evidence exists without report disclosure')
        return None
    wrapper=checked('experiments/recover_full_pixel_v5.py',root)
    paths={k:checked(v,root) for k,v in paths.items()}
    evidence={k:load(v) for k,v in paths.items()}
    hashes={'protocolSha256':protocol_hash,'planSha256':digest(paths['plan']),'receiptSha256':digest(paths['receipt']),
            'failedReportSha256':digest(paths['failed']),'recoveryWrapperSha256':digest(wrapper)}
    result=validate_recovery_data(protocol,report,evidence['plan'],evidence['receipt'],evidence['failed'],hashes,
                                 {p.name:sha for p,sha in prediction_hashes.items()})
    require(digest(checked(evidence['plan']['originalFailedReportPath'],root))==hashes['failedReportSha256'],'Original failed report is not preserved')
    for item in evidence['plan']['sourcePredictions']:
        require(digest(checked(item['sourcePath'],root))==item['sha256'],'Original reused prediction changed')
    for key,path in paths.items():
        name={'plan':'planSha256','receipt':'receiptSha256','failed':'failedReportSha256'}[key]
        require(digest(path)==hashes[name],'Recovery evidence changed during verification')
    return result

def verify_run(output=OUT,receipt=None,root=ROOT):
    from PIL import Image,ImageOps
    output=Path(output).resolve();root=Path(root).resolve()
    protocol_path=checked(output/'protocol.json',root);report_path=checked(output/'report.json',root)
    protocol_hash=digest(protocol_path);report_hash=digest(report_path);protocol=load(protocol_path);report=load(report_path)
    require(report.get('status')=='completed','Only completed runs can be verified; no receipt written')
    require(load(output/'protocol-sha256.json').get('sha256')==protocol_hash==report.get('protocolSha256'),'Frozen protocol hash mismatch')
    require({name:importlib.metadata.version(name) for name in protocol['softwareVersions']}==protocol['softwareVersions'],'Frozen software versions changed')
    for path,expected in protocol['inputSha256'].items():require(digest(checked(path,root))==expected,'Frozen input hash changed: '+path)
    for job in protocol['jobs']:
        path=checked(job['imagePath'],root);require(digest(path)==job['derivedImageSha256'],'Derived image hash mismatch')
        with Image.open(path) as opened:require(ImageOps.exif_transpose(opened).size==(job['width'],job['height']),'Derived image dimensions mismatch')
    prediction_hashes={}
    def predictions(record):
        path=checked(record['predictionPath'],root)
        require(path.parent==output/'predictions' and path.name==f"{record['frameId']}--{record['conditionId']}.json",'Prediction file identity/path mismatch')
        prediction_hashes[path]=record['predictionSha256'];return read_hashed_json(path,record['predictionSha256'])
    cohorts=verify_report_data(protocol,report,predictions)
    require(len(prediction_hashes)==101,'Expected 101 unique prediction artifacts')
    recovery=verify_recovery_evidence(output,root,protocol,report,protocol_hash,prediction_hashes)
    require(digest(protocol_path)==protocol_hash and digest(report_path)==report_hash,'Evidence changed during verification')
    for path,expected in prediction_hashes.items():require(digest(path)==expected,'Prediction changed during verification')
    result={'schema':'mushroom-vision.full-pixel-independent-verification.v1','status':'verified','verifiedAt':datetime.now(timezone.utc).isoformat(),
        'verifierSha256':digest(Path(__file__)),'protocolSha256':protocol_hash,'reportSha256':report_hash,'sourceReferences':17,'positiveReferenceRegions':12,
        'completeHumanPositiveRegions':3,'incompleteAIPositiveRegions':9,'nativeNegativeControls':5,'scenes':101,'uniquePredictionFiles':101,'uniqueModelForwards':505,
        'uniquePersistedModelForwards':505,'totalAttemptedModelForwards':None if report.get('recovery') else 505,
        'softwareVersions':protocol['softwareVersions'],'recovery':recovery,
        'syntheticImages':84,'cohorts':cohorts,'trainingPerformed':False,'weightsChanged':False,'actualDistanceM':None,
        'scope':'Independent consistency and localization recount from saved predictions and frozen pixels; not independent human confirmation or proof of field accuracy.'}
    destination=Path(receipt).resolve() if receipt else output/'independent-verification.json'
    require(destination.is_relative_to(root) and destination not in (protocol_path,report_path),'Receipt path must be a new project artifact')
    with destination.open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--receipt',type=Path)
    args=parser.parse_args();result=verify_run(args.output,args.receipt)
    print(json.dumps({k:v for k,v in result.items() if k!='cohorts'}))
