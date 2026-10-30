import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdir, rm, readFile, writeFile } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { TrainingPipeline } from '../core/training-pipeline.mjs';
import { TestbenchStore } from '../core/testbench-store.mjs';

const root=resolve('.'),fixtureRoot=resolve(root,'experiments','tmp-tests');
const bytes=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAANklEQVR4nO3NMQEAMAjEwKdCqq4SEVgJYWHLCUjq3c6ms1qPgwEHyAFygBwgB8gBcoAcIAchHz0UAXbiXQJAAAAAAElFTkSuQmCC','base64');
const hash=createHash('sha256').update(bytes).digest('hex');
async function fixture(t,options={}){
  const directory=resolve(fixtureRoot,'learning-'+randomUUID());await mkdir(directory,{recursive:true});
  const record={runId:'11111111-1111-4111-8111-111111111111',image:{sha256:hash,width:32,height:32},sceneId:'scene-'+hash,
    sourceCredits:{attribution:'Software test fixture'},annotation:{id:'22222222-2222-4222-8222-222222222222',revision:1,reviewer:'Test fixture',rightsAttestation:'Software test only',
      boxes:[{id:'a',bboxPixelsXYXY:[1,1,10,10]}]}};
  const testbench={exportDataset:async()=>({records:[record],summary:{excluded:{}}}),list:()=>({runs:[],total:0}),image:async()=>({bytes,sha256:hash})};
  const videos={read:async()=>({frames:[]})},videoGrades={};
  const service=await new TrainingPipeline(root,directory,{testbench,videos,videoGrades,...options}).initialize();
  t.after(async()=>{await service.stop();assert.ok(directory.startsWith(fixtureRoot+sep));await rm(directory,{recursive:true,force:true});});
  return {service,record,testbench,videos,videoGrades,directory};
}

test('learning exports real bytes and refuses insufficient data without starting training',async t=>{
  const {service}=await fixture(t);const before=await service.status();assert.equal(before.eligibleImages,1);assert.equal(before.active,null);
  const result=await service.export();assert.equal(result.images,1);assert.equal(result.readiness.ready,false);
  const dataset=service.datasets.get(result.datasetId);const manifest=JSON.parse(await readFile(resolve(dataset.directory,'manifest.json'),'utf8'));
  assert.equal(manifest.records[0].boxes.length,1);assert.equal(manifest.records[0].originalSha256,hash);
  await assert.rejects(service.startJob({datasetId:result.datasetId}),error=>error.code==='LEARNING_INSUFFICIENT_DATA');assert.equal(service.jobs.size,0);
});

test('training source revision and permission changes invalidate earlier exports',async t=>{
  const {service,record,testbench}=await fixture(t);const exported=await service.export(),dataset=service.datasets.get(exported.datasetId);
  await service.recheckDataset(dataset);record.annotation.revision=2;
  await assert.rejects(service.recheckDataset(dataset),error=>error.code==='LEARNING_SOURCE_CHANGED');
  testbench.exportDataset=async()=>({records:[],summary:{excluded:{rightsNotGranted:1}}});
  await assert.rejects(service.recheckDataset(dataset),error=>error.code==='LEARNING_SOURCE_CHANGED');
});

test('strict video releases retain assistance lineage and reject missing, changed or revoked community consent',async t=>{
  const {service,videos,videoGrades,testbench,directory}=await fixture(t);
  testbench.exportDataset=async()=>({records:[],summary:{excluded:{}}});
  const frame={id:'community-frame',sha256:hash,width:32,height:32,timestampSeconds:1,sourceId:'community-video',sourceGroupId:'community-outing',evaluationOnly:false,
    source:{id:'community-video',videoHash:'a'.repeat(64),sourceGroupId:'community-outing',licenseAllowsTraining:true}};
  const review={id:'local-human-review',reviewer:'Curator',status:'visible',complete:true,trainingAllowed:true,
    boxes:[{bboxPixelsXYXY:[1,1,10,10]}],assistance:{aiAssisted:true,communityReviewIds:[7]}};
  videos.read=async()=>({frames:[frame]});videos.frame=async()=>({image:{bytes,sha256:hash},metadata:frame});videoGrades.get=()=>({revision:1,review});
  service.communityInboxFile=resolve(directory,'community-inbox.json');
  let snapshot=await service.sourceSnapshot();assert.equal(snapshot.records.length,0);assert.equal(snapshot.excluded.communityPermission,1);
  const original={id:7,revision:1,reviewerId:'b'.repeat(64),frameId:frame.id,sourceId:frame.sourceId,sha256:hash,createdAt:'2026-10-29T00:00:00Z',trainingConsent:true,labelAuthority:'community-unverified'};
  const put=async(reviews,history=reviews)=>writeFile(service.communityInboxFile,JSON.stringify({schema:'mycoscout.community-inbox.v1',reviews,history}));
  await put([original]);snapshot=await service.sourceSnapshot();
  assert.deepEqual(snapshot.records[0].assistance,review.assistance);assert.equal(snapshot.records[0].communityEvidence[0].revision,1);
  assert.equal(snapshot.records[0].communityEvidence[0].labelAuthority,'community-unverified');
  const release=await service.export(),dataset=service.datasets.get(release.datasetId);
  assert.deepEqual(dataset.records[0].assistance,review.assistance);assert.equal(dataset.records[0].communityEvidence[0].sha256,hash);
  await service.recheckDataset(dataset);
  const revoked={...original,id:8,revision:2,trainingConsent:false};await put([revoked],[original,revoked]);
  snapshot=await service.sourceSnapshot();assert.equal(snapshot.records.length,0);assert.equal(snapshot.excluded.communityPermission,1);
  await assert.rejects(service.recheckDataset(dataset),error=>error.code==='LEARNING_SOURCE_CHANGED');
  const restored={...original,id:9,revision:3};await put([restored],[original,revoked,restored]);
  snapshot=await service.sourceSnapshot();assert.equal(snapshot.records[0].communityEvidence[0].id,7);assert.equal(snapshot.records[0].communityEvidence[0].currentConsentReviewId,9);
  for(const invalid of [{...original,trainingConsent:false},{...original,sha256:'f'.repeat(64)},{...original,sourceId:'another-video'},{...original,frameId:'another-frame'}]){
    await put([invalid]);assert.equal((await service.sourceSnapshot()).records.length,0);
  }
  await put([restored],[original,{...restored,trainingConsent:false}]);assert.equal((await service.sourceSnapshot()).records.length,0);
  await put([restored],[original,revoked,{...revoked,id:10,trainingConsent:true},restored]);assert.equal((await service.sourceSnapshot()).records.length,0);
  await put([{...revoked,sha256:'invalid'}],[original]);await assert.rejects(service.sourceSnapshot(),error=>error.code==='LEARNING_COMMUNITY_EVIDENCE');
  review.assistance={aiAssisted:true,communityReviewIds:[]};snapshot=await service.sourceSnapshot();assert.equal(snapshot.records.length,1);assert.equal(snapshot.records[0].assistance.aiAssisted,true);
});

test('known unreviewed evaluation duplicate quarantines otherwise eligible upload',async t=>{
  const {service,testbench}=await fixture(t);testbench.list=()=>({runs:[{id:'eval-id',evaluationOnly:true,image:{sha256:hash}}],total:1});
  const exported=await service.export(),data=service.datasets.get(exported.datasetId);
  assert.equal(data.records[0].split,'test');assert.equal(data.records[0].evaluationQuarantine,true);
});

test('session grouping rejects unknown records and unsafe job parameters',async t=>{
  const {service}=await fixture(t);
  await assert.rejects(service.export({groupAssignments:{missing:'site'}}),/missing or ineligible/);
  await assert.rejects(service.startJob({datasetId:'../../other'}),error=>error.code==='DATASET_NOT_FOUND');
  const datasetId='dataset-'+'a'.repeat(24);service.datasets.set(datasetId,{datasetId,readiness:{ready:true},records:[]});
  await assert.rejects(service.startJob({datasetId,epochs:Infinity}),/1–30/);
  await assert.rejects(service.startJob({datasetId,learningRate:NaN}),/1–30/);
});

test('one bounded fixed-module job at a time; cancellation stops its process',async t=>{
  const calls=[];const spawnWorker=(executable,args,options)=>{const child=new EventEmitter();child.stdout=new PassThrough();child.stderr=new PassThrough();
    child.kill=()=>setImmediate(()=>child.emit('exit',1));calls.push({executable,args,options});return child;};
  const {service}=await fixture(t,{spawnWorker});
  const datasetId='dataset-'+'a'.repeat(24);service.datasets.set(datasetId,{datasetId,readiness:{ready:true},records:[],directory:resolve(fixtureRoot,'unused')});
  const job=await service.startJob({datasetId});await new Promise(done=>setTimeout(done,30));
  assert.deepEqual(calls[0].args,['-m','vision.training_pipeline','download-pretrained']);assert.equal(calls[0].options.shell,undefined);
  await assert.rejects(service.startJob({datasetId}),error=>error.code==='LEARNING_BUSY');
  assert.equal(service.cancel(job.id).status,'cancelled');await service.running;assert.equal(service.job(job.id).status,'cancelled');assert.equal(service.busy,false);
});

test('activation rejects nonexistent and unevaluated candidates',async t=>{
  const {service}=await fixture(t);
  await assert.rejects(service.activate({jobId:'job-none',reviewer:'Tester'}),error=>error.code==='LEARNING_NOT_ACTIVATABLE');
  assert.equal(service.active,null);
});

test('restart marks an unfinished persisted training job interrupted',async t=>{
  const {directory,service,testbench,videos,videoGrades}=await fixture(t),id='job-'+randomUUID();
  await writeFile(resolve(directory,'jobs',id+'.json'),JSON.stringify({id,status:'running'}));
  const restarted=await new TrainingPipeline(root,directory,{testbench,videos,videoGrades}).initialize();
  assert.equal(restarted.job(id).status,'interrupted');await restarted.stop();
});

test('learning status and export use the actual testbench pagination contract',async t=>{
  const {directory,service}=await fixture(t);
  const actual=await new TestbenchStore(resolve(directory,'actual-testbench'),root,{stop(){}}).initialize();
  service.testbench=actual;
  try{const status=await service.status();assert.equal(status.eligibleImages,0);const result=await service.export();assert.equal(result.images,0);assert.equal(result.readiness.ready,false);}
  finally{await actual.stop();}
});

test('idempotent exports recover a ready release and reject changed labels without exporting twice',async t=>{
  const {service,record,directory,testbench,videos,videoGrades}=await fixture(t);
  const first=await service.export({idempotencyKey:'loop-export-test'}),second=await service.export({idempotencyKey:'loop-export-test'});
  assert.equal(first.datasetId,second.datasetId);assert.equal(service.datasets.size,1);
  const restarted=await new TrainingPipeline(root,directory,{testbench,videos,videoGrades}).initialize();
  try{assert.equal((await restarted.export({idempotencyKey:'loop-export-test'})).datasetId,first.datasetId);
    record.annotation.revision=2;await assert.rejects(restarted.export({idempotencyKey:'loop-export-test'}),error=>error.code==='LEARNING_SOURCE_CHANGED');
    await assert.rejects(restarted.export({idempotencyKey:'loop-export-test',groupAssignments:{other:'group'}}),error=>error.code==='LEARNING_IDEMPOTENCY_CONFLICT');
  }finally{await restarted.stop();}
});

test('idempotent automatic job starts recover the same persisted job and preserve validation-only intent',async t=>{
  const {service,directory,testbench,videos,videoGrades}=await fixture(t),datasetId='dataset-'+'e'.repeat(24);
  service.datasets.set(datasetId,{datasetId,readiness:{ready:true},records:[],directory});
  let executions=0;service.execute=async job=>{executions++;job.status='running';await service.persistJob(job);};
  const input={datasetId,idempotencyKey:'loop-job-test',validationOnly:true},first=await service.startJob(input),again=await service.startJob(input);
  assert.equal(first.id,again.id);assert.equal(executions,1);assert.equal(again.validationOnly,true);
  await assert.rejects(service.startJob({...input,epochs:7}),error=>error.code==='LEARNING_IDEMPOTENCY_CONFLICT');
  const restarted=await new TrainingPipeline(root,directory,{testbench,videos,videoGrades}).initialize();
  try{const recovered=await restarted.startJob(input);assert.equal(recovered.id,first.id);assert.equal(recovered.status,'interrupted');}finally{await restarted.stop();}
});

test('validation-only training forwards its fixed CLI flag and requires an explicit release hook to activate',async t=>{
  const {service,directory}=await fixture(t),datasetId='dataset-'+'d'.repeat(24),id='job-'+randomUUID(),output=resolve(directory,'candidate-output');
  await mkdir(output);const checkpoint=Buffer.from('fixture model bytes'),checkpointSha256=createHash('sha256').update(checkpoint).digest('hex');
  const card={activationEligible:false,validationOnly:true,checkpointSha256};
  const dataset={datasetId,readiness:{ready:true},records:[],directory};service.datasets.set(datasetId,dataset);
  const calls=[];service.runProcess=async args=>{calls.push(args);if(args[2]==='train')await writeFile(resolve(output,'model-card.json'),JSON.stringify(card));};
  const job={id,datasetId,outputDirectory:output,epochs:3,learningRate:.005,scoreThreshold:.3,validationOnly:true,status:'queued'};service.jobs.set(id,job);
  await service.execute(job,dataset);assert.ok(calls[1].includes('--validation-only'));assert.equal(job.status,'completed');
  await writeFile(resolve(output,'candidate.pt'),checkpoint);
  await assert.rejects(service.activate({jobId:id,reviewer:'Fixture reviewer'}),error=>error.code==='LEARNING_RELEASE_CHECK_REQUIRED');
  assert.equal(service.busy,false);
  service.releaseCheck=async()=>({approved:false});await assert.rejects(service.activate({jobId:id,reviewer:'Fixture reviewer'}),error=>error.code==='LEARNING_RELEASE_CHECK_FAILED');
  assert.equal(service.busy,false);
  let release,checking;const entered=new Promise(done=>{checking=done;});
  service.releaseCheck=async()=>{checking();return new Promise(done=>{release=done;});};
  const activation=service.activate({jobId:id,reviewer:'Fixture reviewer'});await entered;assert.equal(service.busy,true);
  await assert.rejects(service.startJob({datasetId}),error=>error.code==='LEARNING_BUSY');
  await assert.rejects(service.export(),error=>error.code==='LEARNING_BUSY');
  await assert.rejects(service.activate({jobId:id,reviewer:'Another reviewer'}),error=>error.code==='LEARNING_BUSY');
  release({approved:true,decision:'candidate_better',comparisonSha256:'a'.repeat(64)});
  const active=await activation;assert.equal(active.releaseComparison.approved,true);assert.equal(service.busy,false);
  assert.equal(active.modelCard.activationEligible,false);assert.equal(active.experimental,true);
});
