"""Small synthetic evidence fixtures; no detector, source data or model loading."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
import uuid

from experiments.verify_full_pixel_v5 import (
    LEVELS, SETTINGS, COHORTS, aggregate_independent, expected_tiles,
    read_hashed_json, recompute_metrics, validate_prediction_record,
    verify_report_data, verify_run, validate_recovery_data,
)


def fixtures():
    model={'model':'fixture detector','resolvedRevision':'sha256:'+'a'*64,'prompt':'a mushroom. a fungus.'}
    refs=[]
    for authority,negative,count in [('human',False,3),('ai',False,9),('human',True,1),('ai',True,4)]:
        for i in range(count):
            identity=f'{authority}-{negative}-{i}'
            refs.append({'frameId':identity,'referenceAuthority':authority,'isNegativeControl':negative,
                'referenceComplete':authority=='human' or negative,'cohort':'human-video' if authority=='human' else 'ai-photo',
                'originalTargetPixelsXYXY':None if negative else [100.,100.,200.,200.],
                'imageSha256':hashlib.sha256(identity.encode()).hexdigest(),'width':1280,'height':720})
    jobs=[];records=[];predictions={}
    for condition,pixels in LEVELS:
        for ref in refs:
            if pixels is not None and ref['isNegativeControl']:continue
            original=ref['originalTargetPixelsXYXY'];scale=1 if pixels is None else pixels/100
            box=None if original is None else [v*scale for v in original]
            width,height=(1280,720) if pixels is None else (1920,1080)
            image_hash=ref['imageSha256'] if pixels is None else hashlib.sha256((ref['frameId']+condition).encode()).hexdigest()
            job={k:copy.deepcopy(ref[k]) for k in ['frameId','referenceAuthority','referenceComplete','isNegativeControl','cohort','originalTargetPixelsXYXY']}
            job.update(order=len(jobs),conditionId=condition,targetPixels=pixels,width=width,height=height,
                sourceImageSha256=ref['imageSha256'],derivedImageSha256=image_hash,
                imagePath=f"fixture/{ref['frameId']}--{condition}.png",targetPixelsXYXY=box,
                realizedTargetWidthPixels=None if box is None else box[2]-box[0],
                realizedTargetHeightPixels=None if box is None else box[3]-box[1],
                geometry={'supersampling':None if pixels is None else 16,'sceneReEnlarged':False,
                    'sceneScaleXY':[scale,scale],'sceneBoundsPixelsXYXY':[0,0,1280*scale,720*scale]},methods={})
            candidate_rows=[] if box is None else [{'bboxPixelsXYXY':box,'confidence':.8,'sourcePass':'original'}]
            candidate_rows.append({'bboxPixelsXYXY':[1000.,600.,1010.,610.],'confidence':.4,'sourcePass':'original'})
            tiles=expected_tiles(width,height)
            timing=[{'kind':'context' if i==0 else 'tile','bboxPixelsXYXY':b,'seconds':1.} for i,b in enumerate([[0,0,width,height],*tiles])]
            methods=[{'id':'original','candidates':copy.deepcopy(candidate_rows),'seconds':1.,'passes':1,'reusedPasses':0},
                {'id':'tiles','candidates':copy.deepcopy(candidate_rows),'seconds':5.,'additionalSeconds':4.,'passes':5,'reusedPasses':1,
                 'fullFrameContext':True,'tileRegion':[0.,0.,1.,1.],'tileBoundsWorkingPixelsXYXY':tiles,
                 'tileBoundsPixelsXYXY':tiles,'passTimings':timing}]
            prediction={'schema':'mycoscout.testbench-result.v1','modelPasses':5,'settings':copy.deepcopy(SETTINGS),
                'image':{'sha256':image_hash,'width':width,'height':height,'workingWidth':width,'workingHeight':height},
                'model':{**model,'boxThreshold':.25,'textThreshold':.25,'nmsIou':.5,'downloadsEnabled':False,'device':'cpu','threads':4},
                'methods':methods,'seconds':6.}
            record={**copy.deepcopy(job),'uniqueModelForwards':5,'sceneWallSeconds':6.}
            for name,method in zip(['raw','tiles'],methods):
                candidates=[{'bbox':c['bboxPixelsXYXY'],'score':c['confidence'],'sourcePass':c['sourcePass']} for c in candidate_rows]
                if box is None:
                    metrics={'hits':None,'targetCount':0,'unmatchedPositiveBoxes':None,'extraOverlappingAlerts':None,
                        'negativeAlerts':1,'negativeHasAlerts':True,'humanNegativeFalseAlerts':1 if ref['referenceAuthority']=='human' else None,
                        'aiNegativeOpinionAlerts':1 if ref['referenceAuthority']=='ai' else None}
                else:
                    metrics={'hits':1,'targetCount':1,'matchedPredictionIndex':0,'bestIoU':1.,'unmatchedPositiveBoxes':1,
                        'extraOverlappingAlerts':0,'humanCompletePositiveFalseAlerts':1 if ref['referenceAuthority']=='human' else None,
                        'negativeAlerts':None,'negativeHasAlerts':None}
                record['methods'][name]={'candidates':copy.deepcopy(candidates),'metrics':metrics,'seconds':method['seconds'],
                    'modelPasses':method['passes'],'sharedContextPasses':method['reusedPasses'],'passTimings':method.get('passTimings',[])}
            jobs.append(job);records.append(record);predictions[(ref['frameId'],condition)]=prediction
    cohorts=[]
    for cohort,label in COHORTS.items():
        count,negative=(3,1) if cohort=='human-video' else (9,4)
        conditions=[]
        for condition,pixels in LEVELS:
            for mode,seconds in [('raw',1.),('tiles',5.)]:
                conditions.append({'id':condition,'mode':mode,'targetPixels':pixels,'hits':count,'targetCount':count,
                    'unmatchedPositiveBoxes':count,'extraOverlappingAlerts':0,'negativeAlertCount':negative if pixels is None else None,
                    'negativeFramesWithAlerts':negative if pixels is None else None,'negativeFrameCount':negative if pixels is None else None,
                    'meanSecondsPerFrame':seconds,'measurementFrames':count+(negative if pixels is None else 0),
                    'realizedTargetWidths':[100. if pixels is None else pixels]*count,'realizedTargetHeights':[100. if pixels is None else pixels]*count})
        cohorts.append({'id':cohort,'label':label,'referenceAuthority':'human' if cohort=='human-video' else 'ai-incomplete',
            'targetCount':count,'negativeFrameCount':negative,'conditions':conditions})
    protocol={'schema':'mushroom-vision.full-pixel-v5-protocol.v1','status':'frozen','plannedModelForwards':505,
        'settings':copy.deepcopy(SETTINGS),'levels':[{'id':level,'targetPixels':pixels} for level,pixels in LEVELS],
        'references':refs,'jobs':jobs,'expectedModel':model}
    report={'schema':'mushroom-vision.full-pixel-v5-report.v1','status':'completed','inputIntegrityVerified':True,
        'trainingPerformed':False,'weightsChanged':False,'actualDistanceM':None,
        'progress':{'completed':505,'total':505,'scenesCompleted':101,'scenesTotal':101},'frames':copy.deepcopy(refs),
        'records':records,'cohorts':cohorts,'summedSceneInferenceSeconds':606.,'model':copy.deepcopy(model)}
    return protocol,report,predictions

def recovery_fixtures():
    protocol,report,_=fixtures();protocol['softwareVersions']={'test-library':'1.0'}
    hashes={name:str(i)*64 for i,name in enumerate(['protocolSha256','failedReportSha256','recoveryWrapperSha256','planSha256','receiptSha256'],1)}
    predictions={}
    for record in report['records']:
        name=f"{record['frameId']}--{record['conditionId']}.json"
        record['predictionPath']='evidence/predictions/'+name
        record['predictionSha256']=hashlib.sha256(name.encode()).hexdigest();predictions[name]=record['predictionSha256']
    failed={'status':'failed','error':{'type':'PermissionError'},'protocolSha256':hashes['protocolSha256'],
        'progress':{'completed':460,'total':505,'scenesCompleted':92,'scenesTotal':101},
        'elapsedWallSeconds':1000.,'records':copy.deepcopy(report['records'][:92])}
    for record in report['records']:record['predictionPath']=record['predictionPath'].replace('evidence/','evidence/recovery/')
    common={'reusedScenes':92,'newScenes':9,'totalAttemptedModelForwards':None,
        'disclosure':'Saved work is preserved; total attempted forwards are unknown after interruption.'}
    provenance={k:hashes[k] for k in ('protocolSha256','failedReportSha256','recoveryWrapperSha256')}
    plan={**common,**provenance,'schema':'mushroom-vision.full-pixel-recovery-plan.v1',
        'uniquePlannedModelForwards':505,'newRecoveryModelForwards':45,'workers':2,'torchThreadsPerWorker':4,
        'settings':copy.deepcopy(protocol['settings']),'softwareVersions':copy.deepcopy(protocol['softwareVersions']),
        'expectedModel':copy.deepcopy(protocol['expectedModel']),
        'sourcePredictions':[{'frameId':r['frameId'],'conditionId':r['conditionId'],'recordOrder':r['order'],
            'sourcePath':r['predictionPath'],'sha256':r['predictionSha256']} for r in failed['records']],
        'missingJobs':[{'frameId':r['frameId'],'conditionId':r['conditionId'],'order':r['order']} for r in report['records'][92:]]}
    receipt={**common,**provenance,'schema':'mushroom-vision.full-pixel-recovery-receipt.v1','status':'completed',
        'uniquePersistedModelForwards':505,'newRecoveryModelForwards':45,'planSha256':hashes['planSha256'],
        'settings':copy.deepcopy(protocol['settings']),'softwareVersions':copy.deepcopy(protocol['softwareVersions']),
        'trainingPerformed':False,'weightsChanged':False,'elapsedWallSecondsIncludesInterruption':True,
        'predictionHashes':copy.deepcopy(predictions),'originalReportedWallSeconds':1000.,'recoveryWallSeconds':100.}
    report['recovery']={**common,'uniquePersistedModelForwards':505,
        **{k:hashes[k] for k in ('planSha256','failedReportSha256','receiptSha256')}}
    return protocol,report,plan,receipt,failed,hashes,predictions


class RecoveryAuditTests(unittest.TestCase):
    def test_exact_92_reused_and_9_new_partition_passes(self):
        result=validate_recovery_data(*recovery_fixtures())
        self.assertEqual(result['uniquePersistedModelForwards'],505)
        self.assertIsNone(result['totalAttemptedModelForwards'])

    def test_attempted_total_cannot_be_falsely_declared_known_or_omitted(self):
        for target in (1,2,3):
            for missing in (False,True):
                args=list(recovery_fixtures());value=args[target]['recovery'] if target==1 else args[target]
                if missing:del value['totalAttemptedModelForwards']
                else:value['totalAttemptedModelForwards']=505
                with self.subTest(target=target,missing=missing),self.assertRaisesRegex(ValueError,'remain unknown'):validate_recovery_data(*args)

    def test_saved_record_or_prediction_hash_cannot_change_on_recovery(self):
        for target in ('record','ledger','receipt'):
            args=list(recovery_fixtures())
            if target=='record':args[1]['records'][0]['methods']['raw']['candidates'][0]['score']=.9
            if target=='ledger':args[2]['sourcePredictions'][0]['sha256']='0'*64
            if target=='receipt':args[3]['predictionHashes'][next(iter(args[3]['predictionHashes']))]='0'*64
            with self.subTest(target=target),self.assertRaises(ValueError):validate_recovery_data(*args)

    def test_partition_cannot_duplicate_or_reclassify_saved_work(self):
        for target in ('duplicate','saved'):
            args=list(recovery_fixtures());missing=args[2]['missingJobs']
            if target=='duplicate':missing[-1]=copy.deepcopy(missing[0])
            else:missing[-1]={k:args[4]['records'][0][k] for k in ('frameId','conditionId','order')}
            with self.subTest(target=target),self.assertRaises(ValueError):validate_recovery_data(*args)

    def test_receipt_hash_and_software_changes_fail(self):
        for target in ('hash','version','timing'):
            args=list(recovery_fixtures())
            if target=='hash':args[1]['recovery']['receiptSha256']='0'*64
            if target=='version':args[3]['softwareVersions']['test-library']='2.0'
            if target=='timing':args[3]['elapsedWallSecondsIncludesInterruption']=False
            with self.subTest(target=target),self.assertRaises(ValueError):validate_recovery_data(*args)


class IndependentPixelAuditTests(unittest.TestCase):
    def setUp(self):self.protocol,self.report,self.predictions=fixtures()

    def verify(self):
        return verify_report_data(self.protocol,self.report,lambda row:self.predictions[(row['frameId'],row['conditionId'])])

    def test_all_101_jobs_and_505_forwards_recount_to_known_totals(self):
        cohorts=self.verify()
        self.assertEqual(len(self.report['records']),101)
        self.assertEqual(sum(c['hits'] for c in cohorts[0]['conditions']),48)
        self.assertEqual(sum(c['hits'] for c in cohorts[1]['conditions']),144)
        self.assertEqual(cohorts[0]['conditions'][0]['negativeAlertCount'],1)
        self.assertEqual(cohorts[1]['conditions'][0]['negativeAlertCount'],4)

    def test_tampered_report_score_candidate_or_metric_is_rejected(self):
        for alteration in ('score','candidate','metric'):
            self.protocol,self.report,self.predictions=fixtures()
            method=self.report['records'][0]['methods']['raw']
            if alteration=='score':method['candidates'][0]['score']=.9
            if alteration=='candidate':method['candidates'][0]['bbox'][0]+=1
            if alteration=='metric':method['metrics']['hits']=0
            with self.subTest(alteration=alteration),self.assertRaisesRegex(ValueError,'candidates, scores or metrics'):self.verify()

    def test_changed_aggregate_detected_even_when_record_metrics_are_unchanged(self):
        self.report['cohorts'][0]['conditions'][0]['hits']+=1
        with self.assertRaisesRegex(ValueError,'aggregate'):self.verify()

    def test_duplicate_job_or_missing_record_is_rejected(self):
        self.report['records'][-1]=copy.deepcopy(self.report['records'][0])
        with self.assertRaisesRegex(ValueError,'Missing, duplicated'):self.verify()
        self.report['records'].pop()
        with self.assertRaisesRegex(ValueError,'101'):self.verify()

    def test_ai_cannot_be_promoted_to_human_authority(self):
        ref=next(r for r in self.protocol['references'] if r['referenceAuthority']=='ai')
        ref['referenceAuthority']='human';ref['referenceComplete']=True
        with self.assertRaisesRegex(ValueError,'authority counts'):self.verify()

    def test_incomplete_ai_annotation_cannot_be_called_complete(self):
        next(r for r in self.protocol['references'] if r['referenceAuthority']=='ai' and not r['isNegativeControl'])['referenceComplete']=True
        with self.assertRaisesRegex(ValueError,'completeness'):self.verify()

    def test_prediction_image_or_model_identity_is_bound(self):
        key=next(iter(self.predictions));self.predictions[key]['image']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'image identity'):self.verify()
        self.protocol,self.report,self.predictions=fixtures()
        self.predictions[key]['model']['resolvedRevision']='sha256:'+'0'*64
        with self.assertRaisesRegex(ValueError,'model identity'):self.verify()

    def test_missing_tile_shared_context_or_incorrect_pass_timing_is_rejected(self):
        for alteration in ('tile','reuse','time','bounds'):
            self.protocol,self.report,self.predictions=fixtures()
            method=next(iter(self.predictions.values()))['methods'][1]
            if alteration=='tile':method['passTimings'].pop()
            if alteration=='reuse':method['reusedPasses']=0
            if alteration=='time':method['passTimings'][0]['seconds']=2.
            if alteration=='bounds':method['tileBoundsPixelsXYXY'][0][2]-=1
            with self.subTest(alteration=alteration),self.assertRaises(ValueError):self.verify()

    def test_invalid_boxes_scores_or_tile_projection_are_rejected(self):
        for alteration in ('nan','threshold','outside','tile'):
            self.protocol,self.report,self.predictions=fixtures()
            row=next(iter(self.predictions.values()))['methods'][1]['candidates'][0]
            if alteration=='nan':row['confidence']=float('nan')
            if alteration=='threshold':row['confidence']=.2
            if alteration=='outside':row['bboxPixelsXYXY'][0]=-1
            if alteration=='tile':row['sourcePass']='tile_4'
            with self.subTest(alteration=alteration),self.assertRaises(ValueError):self.verify()

    def test_extra_overlaps_count_once_and_partial_labels_do_not_create_false_positives(self):
        target=[0,0,10,10]
        rows=[{'bbox':[0,0,7,10],'score':.6},{'bbox':[3,0,10,10],'score':.8},{'bbox':[20,20,30,30],'score':.4}]
        human=recompute_metrics(target,rows,'human');ai=recompute_metrics(target,rows,'ai')
        self.assertEqual((human['hits'],human['matchedPredictionIndex'],human['extraOverlappingAlerts'],human['unmatchedPositiveBoxes']),(1,1,1,2))
        self.assertEqual(human['humanCompletePositiveFalseAlerts'],2)
        self.assertIsNone(ai['humanCompletePositiveFalseAlerts'])
        self.assertIsNone(recompute_metrics(None,rows,'ai')['humanNegativeFalseAlerts'])
        self.assertEqual(recompute_metrics(None,rows,'ai')['aiNegativeOpinionAlerts'],3)

    def test_hash_tampering_rejected_and_partial_run_never_writes_a_receipt(self):
        folder=Path(__file__).resolve().parent/('tmp-pixel-verification-test-'+uuid.uuid4().hex);folder.mkdir()
        try:
            prediction=folder/'prediction.json';prediction.write_text('{"value":1}',encoding='utf8')
            expected=hashlib.sha256(prediction.read_bytes()).hexdigest()
            self.assertEqual(read_hashed_json(prediction,expected),{'value':1})
            prediction.write_text('{"value":2}',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):read_hashed_json(prediction,expected)
            (folder/'protocol.json').write_text('{}',encoding='utf8')
            (folder/'report.json').write_text('{"status":"running"}',encoding='utf8')
            with self.assertRaisesRegex(ValueError,'Only completed'):verify_run(folder)
            self.assertFalse((folder/'independent-verification.json').exists())
        finally:
            for name in ('prediction.json','protocol.json','report.json'):
                path=folder/name
                if path.exists():path.unlink()
            folder.rmdir()


if __name__=='__main__':unittest.main()
