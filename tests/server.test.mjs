import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { createApp } from '../server.mjs';
import { StateStore } from '../core/state.mjs';

async function fixture(t,options={}) {
  const prefix=join(tmpdir(),'mycoscout-test-');
  const dir=await mkdtemp(prefix), publicDir=join(dir,'public');
  await mkdir(publicDir); await writeFile(join(publicDir,'index.html'),'<html>fixture</html>');
  await writeFile(join(dir,'secret.txt'),'private');
  const app=await createApp({host:'127.0.0.1',token:'',stateFile:join(dir,'state.json'),communityInboxDir:join(dir,'community-inbox'),publicDir,backgroundLearning:false,...options});
  const address=await app.listen(0), base='http://127.0.0.1:'+address.port;
  t.after(async()=>{
    app.server.closeAllConnections();
    await new Promise((resolveClose,reject)=>app.server.close(error=>error?reject(error):resolveClose()));
    await app.closed;
    assert.ok(resolve(dir).startsWith(resolve(prefix)));
    assert.ok(resolve(dir).startsWith(resolve(tmpdir())+sep));
    await rm(dir,{recursive:true,force:true});
  });
  const request=async(path,body,headers={})=>{
    const response=await fetch(base+path,{method:body===undefined?'GET':'POST',
      headers:body===undefined?headers:{'Content-Type':'application/json',...headers},
      body:body===undefined?undefined:JSON.stringify(body)});
    return {status:response.status,headers:response.headers,data:await response.json()};
  };
  return {app,dir,base,request};
}
const surveyMap=(verified=false)=>({frame:{id:'map',unit:'m'},groundTraversabilityVerified:verified,
  grid:{width:8,height:8,resolution:1,origin:{x:0,y:0},cells:Array(64).fill(0)}});
const point={x:5.5,y:5.5,z:0,label:'Unknown fungus',poseKnown:true,depthKnown:true,frameId:'map',confidence:0};

test('bulk review lazily opens a pinned owner-local inventory and rejects LAN access',async t=>{
  const token='bulk-review-owner-token-at-least-24-chars',headers={Authorization:'Bearer '+token};
  const dir=await mkdtemp(join(tmpdir(),'bulk-server-fixture-'));
  const bytes=JSON.stringify({schema:'mycoscout.bulk-ai-inventory.v1',records:[]});await writeFile(join(dir,'manifest.json'),bytes);await writeFile(join(dir,'manifest-sha256.json'),JSON.stringify({sha256:createHash('sha256').update(bytes).digest('hex')}));
  const local=await fixture(t,{token,bulkAiReviewDir:dir});
  t.after(async()=>{assert.ok(resolve(dir).startsWith(resolve(tmpdir())+sep));await rm(dir,{recursive:true,force:true});});
  assert.equal((await local.request('/api/bulk-ai-review/status')).status,401);
  const status=await local.request('/api/bulk-ai-review/status',undefined,headers);assert.equal(status.status,200);assert.equal(status.data.totalImages,0);
  assert.equal((await local.request('/api/bulk-ai-review/images?limit=999',undefined,headers)).status,400);
  const gallery=await local.request('/api/bulk-ai-review/images',undefined,headers);assert.equal(gallery.status,200);assert.deepEqual(gallery.data.records,[]);
  const run=await local.request('/api/bulk-ai-review/runs',{title:'Test',reviewer:{model:'test-model'}},headers);assert.equal(run.status,201);assert.equal(run.data.humanVerified,false);
  const lan=await fixture(t,{host:'0.0.0.0',token,bulkAiReviewDir:dir});const denied=await lan.request('/api/bulk-ai-review/status',undefined,headers);assert.equal(denied.status,403);assert.equal(denied.data.code,'LOCAL_REVIEW_ONLY');
});

test('paired compute helpers cannot acquire station-owner permissions',async t=>{
  const owner='station-owner-test-token-at-least-24-chars';
  const {request}=await fixture(t,{token:owner});
  const ownerHeaders={Authorization:'Bearer '+owner};
  assert.equal((await request('/api/device-compute/pair',{name:'Test helper',profileId:'other-browser'})).status,401);
  const paired=await request('/api/device-compute/pair',{name:'Test helper',profileId:'other-browser'},ownerHeaders);
  assert.equal(paired.status,200);
  const workerHeaders={Authorization:'Bearer '+paired.data.deviceToken};
  const announce={op:'announce',capabilities:['image-quality-v1'],runtime:'unit-test'};
  assert.equal((await request('/api/device-compute/worker',announce,workerHeaders)).status,200);
  assert.equal((await request('/api/device-compute/status',undefined,workerHeaders)).status,401);
  assert.equal((await request('/api/learning/status',undefined,workerHeaders)).status,401);
  assert.equal((await request('/api/ai-review/status',undefined,workerHeaders)).status,401);
  assert.equal((await request('/api/ai-review/runs',{title:'Unauthorized'},workerHeaders)).status,401);
  assert.equal((await request('/api/bulk-ai-review/status',undefined,workerHeaders)).status,401);
  assert.equal((await request('/api/bulk-ai-review/runs',{title:'Unauthorized'},workerHeaders)).status,401);
  assert.equal((await request('/api/tracking-demo',undefined,workerHeaders)).status,401);
  assert.equal((await request('/api/tracking-demo/video',undefined,workerHeaders)).status,401);
  assert.equal((await request('/api/tracking-demo/detect',{},workerHeaders)).status,401);
  assert.equal((await request('/api/device-compute/pair',{name:'Unauthorized',profileId:'other-browser'},workerHeaders)).status,401);
  assert.equal((await request('/api/device-compute/worker',announce,{...workerHeaders,Origin:'https://untrusted.example'})).status,403);
  assert.equal((await request('/api/device-compute/revoke',{deviceId:paired.data.deviceId},ownerHeaders)).status,200);
  assert.equal((await request('/api/device-compute/worker',announce,workerHeaders)).status,401);
  const research=await request('/api/research-catalog',undefined,ownerHeaders);
  assert.equal(research.status,200);
  assert.ok(research.data.taxa.length>=60);
  const aiStatus=await request('/api/ai-review/status',undefined,ownerHeaders);
  assert.equal(aiStatus.status,200);assert.equal(aiStatus.data.humanVerified,false);
  assert.equal((await request('/api/ai-review/runs',{reviewerKind:'human'},ownerHeaders)).status,400);
  assert.equal((await request('/api/tracking-demo',undefined,ownerHeaders)).status,200);
  assert.equal((await request('/api/tracking-demo/detect',{},ownerHeaders)).status,400);
});

test('API lifecycle persists real inputs, clears demo, and enforces reviewed walking ground',async t=>{
  const {request,app,dir}=await fixture(t);
  const initial=await request('/api/state');
  assert.equal(initial.data.mode,'simulation');
  assert.equal(initial.data.capabilities.autonomousFlight,false);
  assert.equal((await request('/api/route',{goalId:'demo-1',mode:'walk'})).data.route.status,'ready');
  const switched=await request('/api/mode',{mode:'survey'});
  assert.equal(switched.data.observations.length,0); assert.equal(switched.data.grid,null);
  assert.equal((await request('/api/observations',point)).status,400);
  await request('/api/map',surveyMap());
  const added=await request('/api/observations',point);
  assert.equal(added.status,201);
  const requestRoute={goalId:added.data.observation.id,mode:'walk',start:{x:2.5,y:2.5}};
  assert.equal((await request('/api/route',requestRoute)).data.code,'GROUND_NOT_VERIFIED');
  await request('/api/map',surveyMap(true));
  assert.equal((await request('/api/state')).data.observations.length,0);
  const reviewedPoint=await request('/api/observations',point);
  const missingStart=await request('/api/route',{mode:'walk',goalId:reviewedPoint.data.observation.id});
  assert.equal(missingStart.status,400); assert.match(missingStart.data.error,/explicit/);
  const routed=await request('/api/route',{...requestRoute,goalId:reviewedPoint.data.observation.id});
  assert.equal(routed.data.route.status,'ready');
  await app.store.pending;
  const saved=JSON.parse(await readFile(join(dir,'state.json'),'utf8'));
  assert.equal(saved.mode,'survey'); assert.equal(saved.observations[0].label,'Unknown fungus');
  assert.equal(saved.groundTraversabilityVerified,true);
  const reloaded=await new StateStore(join(dir,'state.json')).initialize();
  assert.equal(reloaded.snapshot().observations.length,1); assert.equal(reloaded.snapshot().route,null);
  assert.equal(reloaded.snapshot().telemetry.position,null);
});

test('flight route and unsupported drone commands cannot actuate hardware',async t=>{
  const {request}=await fixture(t);
  const flight=await request('/api/route',{mode:'flight',goalId:'demo-1'});
  assert.equal(flight.status,409); assert.equal(flight.data.code,'FLIGHT_NOT_IMPLEMENTED');
  assert.equal((await request('/api/arm',{})).status,404);
  assert.equal((await request('/api/takeoff',{})).status,404);
  assert.equal((await request('/api/health')).data.actuation,false);
});

test('private photo review is exact-path, uncached, protected and unavailable over LAN',async t=>{
  const token='example-private-review-token-12345';
  const local=await fixture(t,{token,photoReviewFile:join(tmpdir(),'missing-mycoscout-review.html')});
  assert.equal((await local.request('/api/research/photo-review')).status,401);
  const headers={Authorization:'Bearer '+token};
  assert.equal((await local.request('/api/research/photo-review',undefined,headers)).data.code,'REVIEW_NOT_GENERATED');
  const review=join(local.dir,'review.html'); await writeFile(review,'<h1>Private fixture</h1>');
  const served=await fixture(t,{photoReviewFile:review});
  const response=await fetch(served.base+'/api/research/photo-review');
  assert.equal(response.status,200); assert.match(await response.text(),/Private fixture/);
  assert.equal(response.headers.get('cache-control'),'no-store');
  assert.match(response.headers.get('content-security-policy'),/default-src 'none'/);
  assert.equal((await served.request('/api/research/photo-review/extra')).status,404);
  const trumpet=await fixture(t,{trumpetReviewFile:review});
  assert.equal((await fetch(trumpet.base+'/api/research/trumpet-review')).status,200);
  assert.equal((await trumpet.request('/api/research/trumpet-review/extra')).status,404);
  const lan=await fixture(t,{host:'0.0.0.0',token,photoReviewFile:review});
  assert.equal((await lan.request('/api/research/photo-review',undefined,headers)).data.code,'LOCAL_REVIEW_ONLY');
  assert.equal((await lan.request('/api/research/trumpet-review',undefined,headers)).data.code,'LOCAL_REVIEW_ONLY');
  assert.equal((await local.request('/api/research/trumpet-review')).status,401);
});

test('local maps export as JSON and cannot masquerade as geographic coordinates',async t=>{
  const {request}=await fixture(t);
  const exported=await request('/api/export?format=json');
  assert.equal(exported.status,200); assert.equal(exported.data.frame.unit,'m');
  assert.equal((await request('/api/export?format=geojson')).data.code,'GEOREFERENCE_REQUIRED');
  assert.equal((await request('/api/export?format=unknown')).status,400);
});

test('concurrent observations are serialized without losing data',async t=>{
  const {request}=await fixture(t);
  const results=await Promise.all(Array.from({length:20},(_,i)=>request('/api/observations',{...point,label:'sample-'+i})));
  assert.ok(results.every(r=>r.status===201));
  const state=(await request('/api/state')).data;
  assert.equal(state.observations.length,24);
  assert.equal(new Set(state.observations.map(o=>o.id)).size,24);
});

test('authentication and exact-origin CORS protect the shared API',async t=>{
  const token='example-test-token-with-enough-length';
  const {request,base}=await fixture(t,{token});
  assert.equal((await request('/api/state')).status,401);
  assert.equal((await request('/api/research/video-learning')).status,401);
  assert.equal((await request('/api/research/video-distance')).status,401);
  assert.equal((await request('/api/review-assistance/status')).status,401);
  assert.equal((await request('/api/state',undefined,{Authorization:'Bearer wrong'})).status,401);
  assert.equal((await request('/api/state',undefined,{Authorization:'Bearer '+token})).status,200);
  const learningReport=await request('/api/research/video-learning',undefined,{Authorization:'Bearer '+token});
  assert.equal(learningReport.status,200);
  assert.equal(learningReport.data.schema,'mycoscout.video-learning-summary.v1');
  assert.equal(learningReport.headers.get('cache-control'),'no-store');
  const distanceReport=await request('/api/research/video-distance',undefined,{Authorization:'Bearer '+token});
  assert.equal(distanceReport.status,200);
  assert.equal(distanceReport.headers.get('cache-control'),'no-store');
  const allowed=await request('/api/state',undefined,{Authorization:'Bearer '+token,Origin:'https://appassets.androidplatform.net'});
  assert.equal(allowed.status,200); assert.equal(allowed.headers.get('access-control-allow-origin'),'https://appassets.androidplatform.net');
  const disallowed=await request('/api/state',undefined,{Authorization:'Bearer '+token,Origin:'https://attacker.example'});
  assert.equal(disallowed.status,403);
  const preflight=await fetch(base+'/api/state',{method:'OPTIONS',headers:{Origin:'https://appassets.androidplatform.net','Access-Control-Request-Method':'GET','Access-Control-Request-Headers':'Authorization'}});
  assert.equal(preflight.status,204);
  assert.match(preflight.headers.get('access-control-allow-headers'),/Authorization/);
  await assert.rejects(createApp({host:'0.0.0.0',token:''}),/requires/);
});

test('bad JSON, excess bodies and path traversal are rejected',async t=>{
  const {base,request}=await fixture(t);
  let response=await fetch(base+'/api/map',{method:'POST',headers:{'Content-Type':'application/json'},body:'{'});
  assert.equal(response.status,400);
  response=await fetch(base+'/api/reset',{method:'POST',body:'{}'});
  assert.equal(response.status,415);
  response=await fetch(base+'/api/map',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({padding:'x'.repeat(1024*1024)})});
  assert.equal(response.status,413);
  response=await fetch(base+'/%2e%2e%2fsecret.txt');
  assert.equal(response.status,403);
  response=await fetch(base+'/core/state.mjs');
  assert.equal(response.status,404);
  assert.equal((await request('/api/observations',{...point,x:999})).status,400);
  const { get } = await import('node:http');
  response=await new Promise((resolveResponse,reject)=>{
    get(base+'/',{headers:{Host:'malicious.example'}},r=>{r.resume();resolveResponse({status:r.statusCode});}).on('error',reject);
  });
  assert.equal(response.status,403);
});

test('invalid saved state is preserved instead of silently replaced',async t=>{
  const {dir}=await fixture(t);
  const path=join(dir,'corrupt.json'); await writeFile(path,'not-json');
  await assert.rejects(new StateStore(path).initialize(),SyntaxError);
  assert.equal(await readFile(path,'utf8'),'not-json');
});

test('failed mutations leave state unchanged and do not poison later writes',async t=>{
  const {app}=await fixture(t);
  await assert.rejects(app.store.mutate(state=>{state.mode='survey';throw new Error('test');}),/test/);
  assert.equal(app.store.snapshot().mode,'simulation');
  await app.store.mutate(state=>{state.observations=[];});
  assert.equal(app.store.snapshot().observations.length,0);
});

test('automatic learning control routes require the station token and an allowed origin',async t=>{
  const token='learning-control-fixture-token-12345',{request,app}=await fixture(t,{token});
  assert.equal((await request('/api/learning/loop')).status,401);
  for(const path of ['config','pause','resume','cancel','tick']){
    assert.equal((await request('/api/learning/loop/'+path,{})).status,401);
    assert.equal((await request('/api/learning/loop/'+path,{},{Authorization:'Bearer '+token,Origin:'https://untrusted.example'})).status,403);
  }
  assert.equal((await request('/api/learning/release-check',{})).status,401);
  assert.equal((await request('/api/learning/community-sync',{})).status,401);
  const response=await request('/api/learning/loop',undefined,{Authorization:'Bearer '+token});
  assert.equal(response.status,200);assert.equal(response.headers.get('cache-control'),'no-store');
  assert.equal(response.data.loop.config.enabled,false);assert.equal(app.learning.jobs.size,0);assert.equal(app.learning.active,null);
});

test('automatic learning API persists bounded settings, pause and resume, and remains idle without approved data',async t=>{
  const {request,app,dir}=await fixture(t);
  app.learning.sourceSnapshot=async()=>({records:[],sourceRevisionSignatures:{},evaluationHashes:[],evaluationSources:[],frozenPartitions:{}});
  let response=await request('/api/learning/loop/config',{enabled:true,minimumNewFrames:50,minimumNewSessions:3});
  assert.equal(response.status,200);assert.equal(response.data.config.recipe.validationOnly,true);
  assert.equal((await request('/api/learning/loop/config',{recipe:{validationOnly:false}})).status,400);
  response=await request('/api/learning/loop/pause',{});assert.equal(response.data.paused,true);assert.equal(response.data.workerState,'paused');
  response=await request('/api/learning/loop/tick',{});assert.equal(response.status,202);await app.learningLoop.tick();
  assert.equal(app.learningLoop.snapshot().workerState,'paused');
  response=await request('/api/learning/loop/resume',{});assert.equal(response.data.paused,false);
  await request('/api/learning/loop/tick',{});await app.learningLoop.tick();
  response=await request('/api/learning/loop');assert.equal(response.data.loop.workerState,'waiting_data');
  assert.deepEqual(response.data.loop.counts,{approvedFrames:0,newFrames:0,newSessions:0});assert.equal(response.data.loop.automaticActivation,false);
  assert.equal((await request('/api/learning/loop/cancel',{})).status,200);
  const persisted=JSON.parse(await readFile(join(dir,'learning','automatic','learning-loop.json'),'utf8'));
  assert.equal(persisted.config.enabled,true);assert.equal(persisted.config.minimumNewFrames,50);assert.equal(app.learning.jobs.size,0);assert.equal(app.learning.active,null);
});

test('explicit release route reports missing validation without launching test inference or activating',async t=>{
  const {request,app,dir}=await fixture(t),jobId='job-'+'1'.repeat(36),datasetId='dataset-'+'a'.repeat(24);let inference=0;
  app.learning.datasets.set(datasetId,{datasetId,directory:join(dir,'fixture-dataset')});
  app.learning.jobs.set(jobId,{id:jobId,status:'completed',datasetId,outputDirectory:join(dir,'fixture-job'),validationOnly:true,modelCard:{checkpointSha256:'b'.repeat(64),activationEligible:false}});
  app.learning.recheckDataset=async()=>{};app.learning.runProcess=async()=>{inference++;throw new Error('Release inference must not run without validation.');};
  assert.equal((await request('/api/learning/release-check',{jobId})).status,202);
  let state;
  for(let i=0;i<20;i++){state=(await request('/api/learning/loop')).data;if(state.releaseRequest?.status!=='running')break;await new Promise(done=>setImmediate(done));}
  assert.equal(state.releaseRequest.status,'failed');assert.match(state.releaseRequest.error,/passing validation/);
  assert.equal(inference,0);assert.equal(app.learning.active,null);assert.equal(app.learning.busy,false);
});
