import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,readFile,writeFile,rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { LearningLoop,learningLoopEvidence } from '../core/learning-loop.mjs';

const sha=value=>createHash('sha256').update(String(value)).digest('hex');
const DAY=86400000;
function snapshot(start=0,size=50,groupPrefix='outing',groups=3){
  const records=Array.from({length:size},(_,index)=>({id:'frame-'+(start+index),originalSha256:sha('pixels-'+(start+index)),complete:true,humanVerified:true,
    trainingAllowed:true,evaluationOnly:false,status:'visible',boxes:[{bboxPixelsXYXY:[1,1,20,20]}],groupKeys:['source:'+groupPrefix+'-'+index%groups,'video:'+groupPrefix+'-'+index%groups]}));
  return {records,sourceRevisionSignatures:Object.fromEntries(records.map(row=>[row.id,sha(row.id)])),evaluationHashes:[],evaluationSources:[],frozenPartitions:{}};
}
const merge=(a,b)=>({...a,records:[...a.records,...b.records],sourceRevisionSignatures:{...a.sourceRevisionSignatures,...b.sourceRevisionSignatures}});
const comparison=()=>({schema:'mycoscout.candidate-baseline-comparison.v1',status:'completed',decision:'candidate_not_better',activationAllowed:false,
  baseline:{recall:.6,falsePositivesPerFrame:.1},candidate:{recall:.5,falsePositivesPerFrame:.2},reasons:['PRIVATE reason'],path:'PRIVATE path'});
async function fixture(t){
  const directory=await mkdtemp(join(tmpdir(),'learning-loop-'));let time=Date.parse('2026-10-29T12:00:00Z'),current=snapshot(),jobCounter=0;
  const calls={exports:[],starts:[],comparisons:[],preflight:[],created:0},requests=new Map(),jobs=new Map();
  const learning={busy:false,sourceSnapshot:async()=>structuredClone(current),
    export:async input=>{calls.exports.push(input);return {datasetId:'dataset-'+sha(JSON.stringify(current.sourceRevisionSignatures)).slice(0,24),readiness:{ready:true,reasons:[]},images:current.records.length};},
    startJob:async input=>{calls.starts.push(input);if(requests.has(input.idempotencyKey))return structuredClone(requests.get(input.idempotencyKey));
      const job={id:'job-'+(++jobCounter),status:'running',datasetId:input.datasetId,validationOnly:input.validationOnly};requests.set(input.idempotencyKey,job);jobs.set(job.id,job);calls.created++;return structuredClone(job);},
    job:async id=>{if(!jobs.has(id))throw Error('missing job');return structuredClone(jobs.get(id));},
    cancel:async id=>{jobs.get(id).status='cancelled';}};
  const options={learning,now:()=>time,beforeWork:async stage=>{calls.preflight.push(stage);},compareCandidate:async job=>{calls.comparisons.push(job.id);return comparison();}};
  let loop=await new LearningLoop(directory,options).initialize();
  t.after(async()=>{await loop.close();await rm(directory,{recursive:true,force:true});});
  return {directory,learning,calls,jobs,options,get loop(){return loop;},get current(){return current;},set current(value){current=value;},
    advance:ms=>{time+=ms;},restart:async()=>{await loop.close();loop=await new LearningLoop(directory,options).initialize();return loop;}};
}
async function enable(f){await f.loop.configure({enabled:true});}
async function complete(f){await f.loop.tick();await f.loop.tick();const id=f.loop.snapshot().active.jobId;f.jobs.get(id).status='completed';await f.loop.tick();await f.loop.tick();return id;}

test('disabled bootstrap honestly counts current approved evidence without starting work',async t=>{
  const f=await fixture(t);const unknown=f.loop.publicSummary();assert.equal(unknown.approvedFrames,null);
  const state=await f.loop.tick();assert.equal(state.workerState,'disabled');assert.deepEqual(state.counts,{approvedFrames:50,newFrames:50,newSessions:3});
  assert.equal(f.calls.exports.length,0);assert.equal(f.calls.starts.length,0);assert.equal(f.loop.publicSummary().automaticActivation,false);
});

test('duplicate pixels, incomplete labels, evaluation quarantine and ungrouped photos do not manufacture readiness',async t=>{
  const f=await fixture(t);const records=f.current.records;
  for(const row of records)row.originalSha256=records[0].originalSha256;
  let evidence=learningLoopEvidence(f.current,f.loop.snapshot(),f.loop.snapshot().config.recipe);assert.equal(evidence.approvedFrames,1);assert.equal(evidence.newSessions,1);
  f.current=snapshot();f.current.records[0].complete=false;f.current.records[1].trainingAllowed=false;f.current.records[2].humanVerified=false;
  f.current.evaluationHashes=[f.current.records[3].originalSha256];evidence=learningLoopEvidence(f.current,f.loop.snapshot(),f.loop.snapshot().config.recipe);
  assert.equal(evidence.newSessions,2);assert.ok(evidence.approvedFrames<47);
  f.current=snapshot();for(const row of f.current.records)row.groupKeys=['scene:'+row.originalSha256];
  await enable(f);const state=await f.loop.tick();assert.equal(state.counts.newSessions,0);assert.equal(state.workerState,'waiting_data');assert.equal(f.calls.exports.length,0);
});

test('one complete cycle exports, trains validation-only, compares and never activates or duplicates consumed data',async t=>{
  const f=await fixture(t);await enable(f);const id=await complete(f);
  let state=f.loop.snapshot();assert.equal(state.workerState,'candidate_ready');assert.equal(state.active,null);assert.equal(f.calls.created,1);
  assert.equal(f.calls.starts[0].validationOnly,true);assert.deepEqual(f.calls.preflight,['export','train','compare']);assert.deepEqual(f.calls.comparisons,[id]);
  assert.equal(state.latestCandidate.status,'candidate_not_better');assert.equal(state.consumedFrames.length,50);
  state=await f.loop.tick();assert.equal(state.counts.newFrames,0);assert.equal(state.counts.newSessions,0);assert.equal(f.calls.created,1);
  const publicData=f.loop.publicSummary();assert.equal(JSON.stringify(publicData).includes('PRIVATE'),false);assert.equal(JSON.stringify(publicData).includes(id),false);
  assert.equal(publicData.latestCandidate.comparison.candidateRecall,.5);assert.equal(publicData.automaticActivation,false);
});

test('parallel ticks coalesce and the rolling daily limit survives restart',async t=>{
  const f=await fixture(t);await enable(f);const first=f.loop.tick(),second=f.loop.tick();assert.equal(first,second);await first;assert.equal(f.calls.exports.length,1);
  await f.loop.tick();const id=f.loop.snapshot().active.jobId;f.jobs.get(id).status='completed';await f.loop.tick();await f.loop.tick();
  f.current=merge(f.current,snapshot(50,50,'new-outing'));await f.loop.tick();await f.restart();let state=await f.loop.tick();
  assert.equal(state.workerState,'waiting_daily_limit');assert.equal(f.calls.created,1);
  f.advance(DAY+1);state=await f.loop.tick();assert.equal(f.calls.created,2);assert.equal(state.workerState,'training');
});

test('permission-sync failures block export and fresh changes after sync invalidate the captured cycle',async t=>{
  const f=await fixture(t);await enable(f);f.loop.beforeWork=async()=>{throw Error('PRIVATE SSH failure');};
  let state=await f.loop.tick();assert.equal(state.workerState,'waiting_sync');assert.equal(f.calls.exports.length,0);
  assert.equal(f.loop.publicSummary().message,'Waiting for current review permissions.');assert.equal(JSON.stringify(f.loop.publicSummary()).includes('PRIVATE'),false);
  f.advance(300001);f.loop.beforeWork=async()=>{f.current.records[0].trainingAllowed=false;f.current.sourceRevisionSignatures[f.current.records[0].id]=sha('revoked');};
  state=await f.loop.tick();assert.equal(state.active,null);assert.equal(state.history.at(-1).outcome,'rejected');assert.equal(f.calls.exports.length,0);
});

test('pause during export and cancellation preserve operator intent without starting the next job',async t=>{
  const f=await fixture(t);await enable(f);let release,started;const entered=new Promise(resolve=>{started=resolve;});
  const original=f.learning.export;f.learning.export=async input=>{started();await new Promise(resolve=>{release=resolve;});return original(input);};
  const running=f.loop.tick();await entered;await f.loop.pause();release();await running;
  assert.equal(f.loop.snapshot().paused,true);await f.loop.tick();assert.equal(f.calls.starts.length,0);
  await f.loop.resume();await f.loop.tick();assert.equal(f.calls.created,1);const id=f.loop.snapshot().active.jobId;
  await f.loop.cancel();assert.equal(f.jobs.get(id).status,'cancelled');assert.equal(f.loop.snapshot().active,null);
  await f.loop.tick();assert.equal(f.calls.created,1);
});

test('a lost job-start response is recovered by the same durable idempotency key after restart',async t=>{
  const f=await fixture(t);await enable(f);await f.loop.tick();const original=f.learning.startJob;let lost=true;
  f.learning.startJob=async input=>{const result=await original(input);if(lost){lost=false;throw Error('Connection lost after acceptance');}return result;};
  await f.loop.tick();assert.equal(f.calls.created,1);assert.equal(f.loop.snapshot().workerState,'retry_wait');
  await f.restart();f.advance(300001);await f.loop.tick();assert.equal(f.calls.created,1);assert.equal(f.calls.starts.length,2);
  assert.equal(f.calls.starts[0].idempotencyKey,f.calls.starts[1].idempotencyKey);assert.equal(f.loop.snapshot().active.stage,'monitor');
});

test('interrupted jobs retry at most the configured limit and each retry respects daily quota',async t=>{
  const f=await fixture(t);await enable(f);await f.loop.tick();await f.loop.tick();let id=f.loop.snapshot().active.jobId;
  f.jobs.get(id).status='interrupted';await f.restart();await f.loop.tick();f.advance(300001);
  assert.equal((await f.loop.tick()).workerState,'waiting_daily_limit');f.advance(DAY);await f.loop.tick();assert.equal(f.calls.created,2);
  id=f.loop.snapshot().active.jobId;f.jobs.get(id).status='failed';f.jobs.get(id).error='PRIVATE device failure';
  let state=await f.loop.tick();assert.equal(state.active,null);assert.equal(state.workerState,'failed');f.advance(DAY);await f.loop.tick();assert.equal(f.calls.created,2);
});

test('unready releases retain their rejection details and do not repeatedly export the same evidence',async t=>{
  const f=await fixture(t);await enable(f);f.learning.export=async input=>{f.calls.exports.push(input);return {datasetId:'dataset-not-ready',readiness:{ready:false,reasons:['Validation has no independent source.']}};};
  let state=await f.loop.tick();assert.equal(state.active,null);assert.deepEqual(state.lastReadiness.reasons,['Validation has no independent source.']);assert.equal(f.calls.created,0);
  await f.loop.tick();assert.equal(f.calls.exports.length,1);assert.match(f.loop.snapshot().history[0].message,/Validation/);
});

test('one directory has one coordinator owner and corrupt state is preserved rather than reset',async t=>{
  const f=await fixture(t);await assert.rejects(new LearningLoop(f.directory,f.options).initialize(),error=>error.code==='LOOP_LOCKED');
  await f.loop.close();const file=join(f.directory,'learning-loop.json');await writeFile(file,'{broken');
  await assert.rejects(new LearningLoop(f.directory,f.options).initialize());assert.equal(await readFile(file,'utf8'),'{broken');
});

test('automatic recipes cannot disable validation-only and failed comparisons never promote candidates',async t=>{
  const f=await fixture(t);await assert.rejects(f.loop.configure({recipe:{validationOnly:false}}),/validation only/);
  await enable(f);await f.loop.tick();await f.loop.tick();const id=f.loop.snapshot().active.jobId;f.jobs.get(id).status='completed';await f.loop.tick();
  f.loop.compareCandidate=async()=>({...comparison(),activationAllowed:true});await f.loop.tick();f.advance(300001);await f.loop.tick();
  assert.equal(f.loop.snapshot().workerState,'failed');assert.equal(f.loop.snapshot().active,null);assert.equal(f.loop.publicSummary().automaticActivation,false);
});
