import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, readFile, writeFile, rm, access } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { CommunitySuggestionsRefresh } from '../core/community-suggestions-refresh.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const save=(path,value)=>writeFile(path,JSON.stringify(value));
async function fixture(t){
  const root=await mkdtemp(resolve(tmpdir(),'mycoscout-suggestion-refresh-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const directory=resolve(root,'refresh'),collectionDir=resolve(root,'public'),jobDir=resolve(root,'job'),datasetDir=resolve(root,'dataset');
  for(const path of [collectionDir,jobDir,datasetDir])await mkdir(path,{recursive:true});
  await writeFile(resolve(jobDir,'candidate.pt'),'fake checkpoint; never loaded');
  const checkpointSha256=sha(await readFile(resolve(jobDir,'candidate.pt')));
  await save(resolve(datasetDir,'manifest.json'),{datasetId:'dataset-fixture',records:[]});
  const card={schema:'mycoscout.detector-model-card.v1',architecture:'fixture-architecture',datasetId:'dataset-fixture',
    datasetManifestSha256:sha(await readFile(resolve(datasetDir,'manifest.json'))),checkpoint:'candidate.pt',checkpointSha256,
    validationOnly:true,activationEligible:false,scoreThreshold:.3};
  await save(resolve(jobDir,'model-card.json'),card);
  const cardSha256=sha(await readFile(resolve(jobDir,'model-card.json')));
  await save(resolve(jobDir,'COMPLETE.json'),{checkpointSha256,datasetId:card.datasetId,modelCardSha256:cardSha256});
  const completeSha256=sha(await readFile(resolve(jobDir,'COMPLETE.json')));
  const frames=[];
  for(let i=0;i<2;i++){const bytes=Buffer.from('fixture image '+i);await writeFile(resolve(collectionDir,'frame-'+i+'.png'),bytes);frames.push({id:'frame-'+i,sourceId:'source',sha256:sha(bytes),width:80,height:60,path:'frame-'+i+'.png'});}
  const manifest={schema:'mycoscout.community-collection.v1',frames,sources:[]};
  await save(resolve(collectionDir,'manifest.json'),manifest);
  const job={id:'job-fixture',status:'completed',datasetId:card.datasetId,outputDirectory:jobDir,modelCard:card,validationOnly:true};
  const dataset={datasetId:card.datasetId,directory:datasetDir};
  const calls={process:[],publish:[],checks:0,before:0};let now=100000;
  const learning={active:{schema:'mycoscout.active-detector.v1',jobId:job.id,datasetId:job.datasetId,
    checkpoint:resolve(jobDir,'candidate.pt'),checkpointSha256,modelCard:card,
    releaseComparison:{approved:true,candidateCheckpointSha256:checkpointSha256},reviewer:'Private reviewer'},
    busy:false,closed:false,child:null,job:()=>structuredClone(job),datasets:new Map([[dataset.datasetId,dataset]]),
    recheckDataset:async()=>{calls.checks++;},
    runProcess:async args=>{calls.process.push(args);assert.equal(learning.busy,true);await output(args);}};
  const bridge={state:{enabled:true},beforeWork:async()=>{calls.before++;},run:async args=>{assert.equal(learning.busy,true);calls.publish.push(args);}};
  async function output(args,mutate=()=>{}){
    const model={name:'Fixture',model:card.architecture,checkpointSha256,version:'reviewed-'+checkpointSha256.slice(0,16),resolvedRevision:'sha256:'+checkpointSha256,
      modelCardSha256:cardSha256,completeSha256,boxThreshold:.3,cropPolicy:'full_frame',nmsIou:.5,maxBoxes:64,humanVerified:false,trainingAllowed:false};
    const data={schema:'mycoscout.public-ai-suggestions.v1',run:{status:'completed',reviewedFrames:2,totalFrames:2,
      manifestSha256:sha(await readFile(resolve(collectionDir,'manifest.json'))),checkpointSha256,modelVersion:model.version,humanVerified:false,trainingAllowed:false},
      frames:frames.map(frame=>({frameId:frame.id,imageSha256:frame.sha256,sourceId:frame.sourceId,status:'completed',model,
        boxes:[{bbox:[1,2,20,30],score:.8,label:'Unverified mushroom candidate',humanVerified:false,trainingAllowed:false}]}))};
    mutate(data);await save(args[args.indexOf('--output')+1],data);
  }
  const options={learning,bridge,collectionDir,now:()=>now};
  const worker=await new CommunitySuggestionsRefresh(root,directory,options).initialize();
  t.after(()=>worker.close());
  return {worker,root,directory,collectionDir,jobDir,datasetDir,learning,bridge,options,calls,output,advance:ms=>{now+=ms;}};
}

test('disabled, absent active and busy workers do no inference or publication',async t=>{
  const f=await fixture(t);f.bridge.state.enabled=false;assert.equal((await f.worker.tick()).reason,'disabled');
  f.bridge.state.enabled=true;const active=f.learning.active;f.learning.active=null;assert.equal((await f.worker.tick()).reason,'no_active_model');
  f.learning.active=active;f.learning.busy=true;assert.equal((await f.worker.tick()).reason,'busy');
  assert.deepEqual(f.calls,{process:[],publish:[],checks:0,before:0});
});

test('verified refresh uses fixed CLI, permission checks and persists restart dedup without private snapshot fields',async t=>{
  const f=await fixture(t);assert.deepEqual(await f.worker.tick(),{completed:true,frames:2});
  assert.equal(f.calls.before,3);assert.equal(f.calls.checks,3);assert.equal(f.calls.publish.length,1);
  assert.deepEqual(f.calls.process[0].slice(0,2),['-m','vision.refresh_community_suggestions']);
  assert.deepEqual(f.calls.publish[0].slice(0,2),['update-ai','--file']);assert.equal(f.learning.busy,false);
  const snapshot=f.worker.snapshot();assert.equal(snapshot.completedRefreshes,1);assert.equal(snapshot.lastFrameCount,2);
  assert.ok(!JSON.stringify(snapshot).includes('Private reviewer'));assert.ok(!JSON.stringify(snapshot).includes(f.root));
  await f.worker.tick();const resumed=await new CommunitySuggestionsRefresh(f.root,f.directory,f.options).initialize();
  assert.equal((await resumed.tick()).reason,'current');assert.equal(f.calls.process.length,1);assert.equal(f.calls.publish.length,1);await resumed.close();
});

test('concurrent ticks reserve learning before asynchronous permission checks',async t=>{
  const f=await fixture(t);let resume;const wait=new Promise(r=>{resume=r;});
  f.bridge.beforeWork=async()=>{f.calls.before++;await wait;};
  const first=f.worker.tick();assert.equal(f.learning.busy,true);assert.equal((await f.worker.tick()).reason,'busy');
  resume();await first;assert.equal(f.calls.process.length,1);assert.equal(f.calls.publish.length,1);
});

test('pointer changes and permission revocation after inference prevent publication',async t=>{
  for(const mutation of ['pointer','permission'])await t.test(mutation,async st=>{
    const f=await fixture(st);f.learning.runProcess=async args=>{await f.output(args);
      if(mutation==='pointer')f.learning.active={...f.learning.active,activatedAt:'changed'};
      else f.learning.recheckDataset=async()=>{throw new Error('private revoked review information');};};
    assert.equal((await f.worker.tick()).failed,true);assert.equal(f.calls.publish.length,0);assert.equal(f.learning.busy,false);
    assert.ok(!JSON.stringify(f.worker.snapshot()).includes('private revoked'));
  });
});

test('partial output, rights escalation, source bytes and checkpoint tampering fail closed',async t=>{
  for(const mutation of ['incomplete','human','source','checkpoint','threshold'])await t.test(mutation,async st=>{
    const f=await fixture(st);f.learning.runProcess=async args=>{
      await f.output(args,data=>{if(mutation==='incomplete')data.frames.pop();if(mutation==='human')data.frames[0].boxes[0].humanVerified=true;if(mutation==='threshold')data.frames[0].model.boxThreshold=.1;});
      if(mutation==='source')await writeFile(resolve(f.collectionDir,'frame-0.png'),'changed');
      if(mutation==='checkpoint')await writeFile(resolve(f.jobDir,'candidate.pt'),'changed');
    };
    assert.equal((await f.worker.tick()).failed,true);assert.equal(f.calls.publish.length,0);
  });
});

test('failed upload retains the same UUID output for restart retry and respects backoff',async t=>{
  const f=await fixture(t);let uploads=0;
  f.bridge.run=async()=>{uploads++;if(uploads===1)throw new Error('unknown upload outcome');};
  assert.equal((await f.worker.tick()).failed,true);assert.equal((await f.worker.tick()).reason,'retry_wait');
  const first=f.calls.process[0][f.calls.process[0].indexOf('--output')+1];
  f.advance(60001);const resumed=await new CommunitySuggestionsRefresh(f.root,f.directory,f.options).initialize();
  assert.equal((await resumed.tick()).completed,true);const second=f.calls.process[1][f.calls.process[1].indexOf('--output')+1];
  assert.equal(first,second);assert.equal(uploads,2);assert.equal(resumed.snapshot().completedRefreshes,1);await resumed.close();
});

test('close cancels in-flight refresh without publishing',async t=>{
  const f=await fixture(t);let entered,finish;const started=new Promise(r=>{entered=r;}),gate=new Promise(r=>{finish=r;});
  f.learning.runProcess=async args=>{entered(args);await gate;await f.output(args);};
  const run=f.worker.tick(),args=await started,closing=f.worker.close();finish();await closing;await run;
  assert.equal(f.calls.publish.length,0);assert.equal(f.learning.busy,false);assert.equal(f.worker.snapshot().status,'stopped');
  await access(args[args.indexOf('--cancel-file')+1]);
});

test('active release approval and COMPLETE card binding are mandatory',async t=>{
  for(const mutation of ['release','card'])await t.test(mutation,async st=>{
    const f=await fixture(st);if(mutation==='release')f.learning.active.releaseComparison=null;
    else await writeFile(resolve(f.jobDir,'model-card.json'),'{}');
    assert.equal((await f.worker.tick()).failed,true);assert.equal(f.calls.process.length,0);assert.equal(f.calls.publish.length,0);
  });
});
