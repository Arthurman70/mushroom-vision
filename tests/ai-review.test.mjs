import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,rm,mkdir,writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,resolve,sep } from 'node:path';
import { AiReviewStore,readAiCandidateSummary } from '../core/ai-review.mjs';
import { pixelBox,reviewContract } from '../public/ai-review.js';

const source=(id,group=id)=>({id,title:id,sourceGroupId:group,videoHash:(id==='train'?'a':id==='validation'?'b':'c').repeat(64),licenseAllowsTraining:true,evaluationOnly:false});
const frame=(id,src,n=0)=>({id,sourceId:src.id,sourceGroupId:src.sourceGroupId,source:src,sha256:String(n+1).repeat(64),width:100,height:50,timestampSeconds:n,blurScore:3,evaluationOnly:false,imageUrl:'/api/testbench/videos/frames/'+id+'/image'});
async function fixture(t){
  const directory=await mkdtemp(join(tmpdir(),'mycoscout-ai-review-')),sources=[source('train'),source('validation'),source('test')],frames=[frame('a',sources[0]),frame('b',sources[0],1),frame('v',sources[1],2),frame('t',sources[2],3)];let snapshot={};
  const videos={read:async()=>({available:true,sources,frames}),frame:async id=>{const metadata=frames.find(f=>f.id===id);if(!metadata)throw new Error('missing');return {metadata};}};
  const options={videos,collectionId:'fixture',sourceSplits:{train:'train',validation:'validation',test:'test'},holdoutSnapshot:async()=>snapshot},store=await new AiReviewStore(directory,options).initialize();
  t.after(async()=>{store.close();assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});});return {directory,store,frames,options,holdout:value=>{snapshot=value;}};
}
const runInput=(extra={})=>({title:'Visual fixture',reviewer:{name:'Fixture assistant',model:'fixture-only'},frameIds:['a','b'],split:'train',...extra});
const review=(frame,extra={})=>({baseRevision:0,sha256:frame.sha256,videoHash:frame.videoHash,viewedOriginal:true,modelSuggestionsViewed:false,status:'visible',complete:true,boxes:[{bboxPixelsXYXY:[10,10,20,30],groupId:'morels',confidence:.7,evidence:'Pitted conical cap visible.'}],evidence:'Original fixture inspected.',...extra});

test('AI reviews remain append-only candidates through export and restart',async t=>{
  const {store}=await fixture(t),run=await store.createRun(runInput());assert.equal(run.frames.length,2);assert.equal(run.reviewerKind,'ai');
  const first=await store.saveReview(run.id,'a',review(run.frames[0]));assert.equal(first.humanVerified,false);assert.equal(first.trainingAllowed,false);
  await store.saveReview(run.id,'a',review(run.frames[0],{baseRevision:1,status:'uncertain',complete:false,boxes:[],evidence:'Correction: not enough detail.'}));
  await assert.rejects(()=>store.saveReview(run.id,'a',review(run.frames[0])),{code:'AI_REVIEW_CONFLICT'});
  assert.throws(()=>store.db.prepare('UPDATE ai_reviews SET revision=9').run(),/immutable/);assert.throws(()=>store.db.prepare('DELETE FROM ai_runs').run(),/immutable/);
  store.close();await store.initialize();assert.equal(store.history(run.id,'a').length,2);assert.equal(store.getRun(run.id).summary.reviewedFrames,1);
  const exported=await store.exportRun(run.id);assert.equal(exported.records.length,0);assert.deepEqual(exported.excluded.map(e=>e.reason),['incomplete_or_uncertain','unreviewed']);assert.equal(exported.productionTrainingAllowed,false);
});

test('train and validation export keeps distinct authority; test is never exported',async t=>{
  const {store}=await fixture(t);
  for(const [split,id]of [['train','a'],['validation','v'],['test','t']]){const run=await store.createRun(runInput({split,frameIds:[id]}));await store.saveReview(run.id,id,review(run.frames[0]));const out=await store.exportRun(run.id);
    if(split==='test'){assert.equal(out.records.length,0);assert.equal(out.excluded[0].reason,'held_out_test');}else{assert.equal(out.records[0].split,split);assert.equal(out.records[0].experimentalTrainingAllowed,split==='train');assert.equal(out.records[0].reviewerKind,'ai');assert.equal(out.records[0].humanVerified,false);}}
});

test('existing held-out groups and hashes override proposed train partitions',async t=>{
  const {store,holdout,frames}=await fixture(t);holdout({frozenPartitions:{'source:train':'test'}});await assert.rejects(()=>store.createRun(runInput()),{code:'AI_REVIEW_PARTITION'});
  holdout({evaluationHashes:[frames[0].sha256]});await assert.rejects(()=>store.createRun(runInput()),{code:'AI_REVIEW_PARTITION'});const status=await store.status();assert.equal(status.counts.test,3);
  holdout({frozenPartitions:{['video:'+frames[0].source.videoHash]:'validation'}});await assert.rejects(()=>store.createRun(runInput()),{code:'AI_REVIEW_PARTITION'});
});

test('export rereads holdouts and excludes newly reserved whole groups',async t=>{
  const {store,holdout}=await fixture(t),run=await store.createRun(runInput());await store.saveReview(run.id,'a',review(run.frames[0]));assert.equal((await store.exportRun(run.id)).records.length,1);
  holdout({evaluationSources:[{groupKeys:['source:train']}]});const out=await store.exportRun(run.id);assert.equal(out.records.length,0);assert.ok(out.excluded.every(e=>e.reason==='held_out_test'));
});

test('duplicate pixels join source groups and prevent related clip leakage',async t=>{
  const {store,frames}=await fixture(t);frames[3].sha256=frames[0].sha256;const status=await store.status();assert.equal(status.counts.test,3);await assert.rejects(()=>store.createRun(runInput()),{code:'AI_REVIEW_PARTITION'});
});

test('invalid authority, boxes, completeness, hash and missing observation are rejected',async t=>{
  const {store}=await fixture(t),run=await store.createRun(runInput()),f=run.frames[0];
  for(const change of [{humanVerified:true},{reviewerKind:'human'},{viewedOriginal:false},{modelSuggestionsViewed:undefined},{sha256:'f'.repeat(64)},{videoHash:'e'.repeat(64)},{status:'uncertain'},{status:'absent'},{boxes:[]},{boxes:[{bboxPixelsXYXY:[-1,0,2,3],groupId:'morels',confidence:.5,evidence:'Invalid outside frame.'}]},{boxes:[{bboxPixelsXYXY:[1,1,3,3],groupId:'fake',confidence:.5,evidence:'Invalid group.'}]},{boxes:[{bboxPixelsXYXY:[1,1,3,3],groupId:'morels',confidence:2,evidence:'Invalid confidence.'}]}])await assert.rejects(()=>store.saveReview(run.id,'a',review(f,change)));
  assert.equal(store.getRun(run.id).summary.reviewedFrames,0);
});

test('temporal evidence is independent and cannot cross source videos',async t=>{
  const {store}=await fixture(t),run=await store.createRun(runInput());await store.saveReview(run.id,'a',review(run.frames[0],{status:'uncertain',complete:false,boxes:[],temporalEvidence:[{frameId:'b',sha256:run.frames[1].sha256,note:'Later view; does not prove this frame.'}]}));
  assert.equal(store.getRun(run.id).reviews.a.status,'uncertain');assert.equal(store.getRun(run.id).reviews.a.temporalEvidence[0].inferredVisibility,false);assert.equal((await store.exportRun(run.id)).records.length,0);
  await assert.rejects(()=>store.saveReview(run.id,'a',review(run.frames[0],{baseRevision:1,temporalEvidence:[{frameId:'v',sha256:'3'.repeat(64),note:'Foreign video.'}]})),{code:'AI_REVIEW_TEMPORAL'});
});

test('default queue is finite, source spread, and dispatch does not handle arbitrary routes',async t=>{
  const {store}=await fixture(t),run=await store.createRun(runInput({frameIds:undefined,limit:1}));assert.equal(run.frames.length,1);assert.equal((await store.dispatch('GET','/api/ai-review/runs/'+run.id)).data.id,run.id);assert.equal(await store.dispatch('GET','/api/ai-review/runs/../../secret'),null);await assert.rejects(()=>store.createRun(runInput({limit:0})));await assert.rejects(()=>store.createRun(runInput({frameIds:['a','a']})));
});

test('source mutation prevents exporting or saving old frame labels',async t=>{
  const {store,frames}=await fixture(t),run=await store.createRun(runInput());await store.saveReview(run.id,'a',review(run.frames[0]));frames[0].width=101;await assert.rejects(()=>store.saveReview(run.id,'a',review(run.frames[0],{baseRevision:1})),{code:'AI_REVIEW_INTEGRITY'});assert.equal((await store.exportRun(run.id)).excluded[0].reason,'source_changed');
});

test('UI coordinates preserve original pixels and contracts state observation authority',()=>{
  assert.deepEqual(pixelBox([90.2,40.9],[-3,2],100,50),[0,2,90,41]);assert.equal(pixelBox([1,1],[1,1],100,50),null);
  const body=reviewContract({sha256:'a',videoHash:'b'});assert.equal(body.viewedOriginal,true);assert.equal(body.modelSuggestionsViewed,false);assert.equal(body.status,'uncertain');assert.equal(body.complete,false);assert.equal(body.humanVerified,undefined);
});

test('saved experiment summary is read-only, sanitized and explicit when missing or invalid',async t=>{
  const {directory}=await fixture(t);let summary=await readAiCandidateSummary(directory);assert.equal(summary.available,false);assert.equal(summary.status,'not-run');
  const folder=join(directory,'experiments/results/ai-review-candidate-v1');await mkdir(folder,{recursive:true});const file=join(folder,'report.json');
  const report={schema:'mushroom-vision.ai-candidate-run.v1',completed:true,epochs:1,labelAuthority:'ai-visual-review',humanVerified:false,activationEligible:false,productionModelChanged:false,testEvaluationPerformed:false,checkpointReloadVerified:true,
    optimizerSteps:6,trainFrames:6,trainSources:3,trainPositiveFrames:3,trainBoxes:5,validationFrames:0,elapsedSeconds:2.75,
    losses:Array.from({length:6},(_,i)=>({frameId:'private-frame-'+i,loss:.5+i/10})),sourceExports:[{runId:'private-run'}],checkpoint:'/private/weights/model.pt',unexpectedToken:'private-token',limitations:['private/free-form content']};
  await writeFile(file,JSON.stringify(report));summary=await readAiCandidateSummary(directory);assert.equal(summary.available,true);assert.equal(summary.optimizerSteps,6);assert.equal(summary.trainBoxes,5);assert.equal(summary.validationAvailable,false);assert.equal(summary.meanLoss,.75);assert.equal(summary.inactive,true);assert.equal(summary.activationEligible,false);assert.equal(summary.humanVerified,false);assert.match(summary.notice,/No eligible validation/);assert.doesNotMatch(JSON.stringify(summary),/private|sourceExports|checkpoint.*\.pt|unexpectedToken/);
  for(const change of [{humanVerified:true},{activationEligible:true},{productionModelChanged:true},{optimizerSteps:7},{validationFrames:-1},{losses:[{loss:1}]},{trainSources:7}]){await writeFile(file,JSON.stringify({...report,...change}));summary=await readAiCandidateSummary(directory);assert.equal(summary.available,false);assert.equal(summary.status,'unavailable');assert.equal(summary.trainFrames,undefined);}
  await writeFile(file,'{ broken json');assert.equal((await readAiCandidateSummary(directory)).status,'unavailable');
});
