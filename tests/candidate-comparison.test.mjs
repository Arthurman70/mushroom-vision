import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve, sep } from 'node:path';
import { CandidateComparisons } from '../core/candidate-comparison.mjs';
import { ApiError } from '../core/validation.mjs';

async function fixture(t){
  const prefix=resolve(tmpdir(),'mycoscout-comparison-'),directory=await mkdtemp(prefix);
  t.after(async()=>{assert.ok(directory.startsWith(prefix)&&directory.startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});});
  const dataset={datasetId:'dataset-'+'a'.repeat(24),directory:resolve(directory,'dataset')};
  const job={id:'job-'+'1'.repeat(36),status:'completed',datasetId:dataset.datasetId,outputDirectory:resolve(directory,'candidate'),validationOnly:true,modelCard:{checkpointSha256:'b'.repeat(64),activationEligible:false}};
  const jobs=new Map([[job.id,job]]),calls=[],controls={decision:'candidate_better',syncs:0,rechecks:0,activations:0};
  const learning={busy:false,active:null,timeoutMs:1000,datasets:new Map([[dataset.datasetId,dataset]]),
    checkIdle(){if(this.busy)throw new ApiError('Busy',409,'LEARNING_BUSY');},
    job(id){if(!jobs.has(id))throw new Error('Missing job');return structuredClone(jobs.get(id));},
    async recheckDataset(){controls.rechecks++;if(controls.recheck)await controls.recheck();},
    async activate(){controls.activations++;throw new Error('Comparison must never activate a model.');},
    async runProcess(args){
      calls.push(args);if(controls.run)await controls.run(args);
      if(args.includes('--validate-only'))return;
      const output=args[args.indexOf('--output')+1],split=args[args.indexOf('--split')+1];
      const report={schema:'mycoscout.candidate-baseline-comparison.v1',status:'completed',split,datasetId:dataset.datasetId,
        candidateCheckpointSha256:job.modelCard.checkpointSha256,datasetManifestSha256:'c'.repeat(64),decision:controls.decision,
        activationAllowed:false,automaticActivation:false,testEvaluationPerformed:split==='test',
        models:{baseline:learning.active?{kind:'reviewed_detector',checkpointSha256:learning.active.checkpointSha256}:{kind:'installed_grounding_dino'}},
        baseline:{recall:.5,falsePositivesPerFrame:.1},candidate:{recall:.7,falsePositivesPerFrame:.05},...controls.report};
      await mkdir(output,{recursive:true});await writeFile(resolve(output,'comparison.json'),JSON.stringify(report));await writeFile(resolve(output,'COMPLETE.json'),'{}');
    }};
  const beforeWork=async()=>{controls.syncs++;if(controls.sync)await controls.sync();};
  const service=await new CandidateComparisons(directory,{learning,beforeWork}).initialize();
  return {service,learning,job,jobs,dataset,calls,controls,directory,beforeWork};
}
const inferenceCalls=calls=>calls.filter(args=>!args.includes('--validate-only'));

test('release testing requires a current passing validation comparison and never activates automatically',async t=>{
  const {service,learning,job,calls,controls}=await fixture(t);
  await assert.rejects(service.compare(job,{split:'test'}),error=>error.code==='VALIDATION_COMPARISON_REQUIRED');assert.equal(calls.length,0);
  controls.decision='candidate_not_better';await service.compare(job);
  await assert.rejects(service.compare(job,{split:'test'}),error=>error.code==='VALIDATION_COMPARISON_REQUIRED');
  assert.equal(inferenceCalls(calls).length,1);assert.equal(learning.busy,false);assert.equal(controls.activations,0);assert.equal(learning.active,null);
});

test('passing validation permits an explicit test comparison and separate release proof without model promotion',async t=>{
  const {service,learning,job,calls,controls}=await fixture(t);
  const validation=await service.compare(job);assert.equal(validation.split,'validation');
  await assert.rejects(service.releaseCheck(job),error=>error.code==='RELEASE_COMPARISON_REQUIRED');
  const release=await service.compare(job,{split:'test'});assert.equal(release.testEvaluationPerformed,true);
  const proof=await service.releaseCheck(job);assert.equal(proof.approved,true);assert.equal(proof.comparisonId,release.comparisonId);
  assert.equal(proof.candidateCheckpointSha256,job.modelCard.checkpointSha256);assert.match(proof.reportSha256,/^[a-f0-9]{64}$/);
  assert.deepEqual(inferenceCalls(calls).map(args=>args[args.indexOf('--split')+1]),['validation','test']);
  assert.equal(controls.activations,0);assert.equal(learning.active,null);assert.equal(job.modelCard.activationEligible,false);
});

test('baseline change invalidates validation and release caches instead of reusing an old passing result',async t=>{
  const {service,learning,job,jobs,dataset,calls}=await fixture(t);
  const previous=await service.compare(job);await service.compare(job,{split:'test'});
  const baseline={...job,id:'job-'+'2'.repeat(36),outputDirectory:resolve(dataset.directory,'baseline'),modelCard:{checkpointSha256:'d'.repeat(64)}};
  jobs.set(baseline.id,baseline);learning.active={jobId:baseline.id,checkpointSha256:baseline.modelCard.checkpointSha256};
  await assert.rejects(service.compare(job,{split:'test'}),error=>error.code==='VALIDATION_COMPARISON_REQUIRED');
  await assert.rejects(service.releaseCheck(job),error=>error.code==='RELEASE_COMPARISON_REQUIRED');
  const next=await service.compare(job);assert.notEqual(next.comparisonId,previous.comparisonId);assert.equal(inferenceCalls(calls).length,3);
  assert.ok(inferenceCalls(calls).at(-1).includes('--baseline-dataset'));
  learning.active.checkpointSha256='e'.repeat(64);
  await assert.rejects(service.compare(job),error=>error.code==='BASELINE_CHANGED');assert.equal(learning.busy,false);
});

test('worker reservation blocks a second comparison during asynchronous synchronization and clears after failure',async t=>{
  const {service,learning,job,controls}=await fixture(t);let release,entered;
  const ready=new Promise(done=>{entered=done;});controls.sync=()=>{entered();return new Promise(done=>{release=done;});};
  const first=service.compare(job);await ready;assert.equal(learning.busy,true);
  await assert.rejects(service.compare(job),error=>error.code==='LEARNING_BUSY');
  controls.sync=null;release();await first;assert.equal(learning.busy,false);assert.equal(service.pending,null);
  controls.sync=async()=>{throw new Error('Fixture permission failure');};
  await assert.rejects(service.compare(job),/Fixture permission failure/);assert.equal(learning.busy,false);assert.equal(service.pending,null);
});

test('completed comparison recovers after restart by verifying frozen evidence without another inference pass',async t=>{
  const {service,learning,job,calls,directory,beforeWork}=await fixture(t);
  const first=await service.compare(job),restarted=await new CandidateComparisons(directory,{learning,beforeWork}).initialize();
  const again=await restarted.compare(job);assert.equal(first.comparisonId,again.comparisonId);assert.equal(inferenceCalls(calls).length,1);
  assert.equal(calls.filter(args=>args.includes('--validate-only')).length,2);assert.equal(restarted.status()[0].status,'completed');
});

test('changed report identity, activation claims, and missing test evidence fail closed',async t=>{
  const {service,learning,job,directory}=await fixture(t);const validation=await service.compare(job),path=resolve(directory,validation.comparisonId,'comparison.json');
  const original=JSON.parse(await readFile(path,'utf8'));
  for(const change of [{candidateCheckpointSha256:'f'.repeat(64)},{datasetId:'wrong'},{activationAllowed:true},{automaticActivation:true}]){
    await writeFile(path,JSON.stringify({...original,...change}));await assert.rejects(service.compare(job),error=>error.code==='COMPARISON_CHANGED');assert.equal(learning.busy,false);
  }
  await writeFile(path,JSON.stringify(original));const release=await service.compare(job,{split:'test'}),releasePath=resolve(directory,release.comparisonId,'comparison.json');
  const report=JSON.parse(await readFile(releasePath,'utf8'));report.testEvaluationPerformed=false;await writeFile(releasePath,JSON.stringify(report));
  await assert.rejects(service.releaseCheck(job),error=>error.code==='RELEASE_COMPARISON_FAILED');
});

test('permission changes after inference prevent completed comparison publication',async t=>{
  const {service,learning,job,controls}=await fixture(t);
  controls.sync=async()=>{if(controls.syncs===2)throw new Error('Permissions withdrawn during comparison');};
  await assert.rejects(service.compare(job),/Permissions withdrawn/);assert.equal(service.status()[0].status,'failed');assert.equal(learning.busy,false);
});

test('cached comparisons and release proofs recheck permissions after validating frozen evidence',async t=>{
  const {service,learning,job,calls,controls}=await fixture(t);await service.compare(job);let withdrawn=false;
  controls.run=async args=>{if(args.includes('--validate-only'))withdrawn=true;};
  controls.recheck=async()=>{if(withdrawn)throw new ApiError('Fixture consent withdrawn',409,'LEARNING_SOURCE_CHANGED');};
  await assert.rejects(service.compare(job),error=>error.code==='LEARNING_SOURCE_CHANGED');
  assert.equal(inferenceCalls(calls).length,1);assert.equal(learning.busy,false);
  withdrawn=false;controls.run=null;await service.compare(job,{split:'test'});
  controls.run=async args=>{if(args.includes('--validate-only'))withdrawn=true;};
  await assert.rejects(service.releaseCheck(job),error=>error.code==='LEARNING_SOURCE_CHANGED');
  assert.equal(inferenceCalls(calls).length,2);assert.equal(learning.active,null);
});
