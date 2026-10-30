import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { createHash } from 'node:crypto';
import { createApp } from '../server.mjs';

const imageBase64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2ZQkAAAAASUVORK5CYII=';
const image={width:1,height:1,mimeType:'image/png',sha256:createHash('sha256').update(Buffer.from(imageBase64,'base64')).digest('hex'),imageBase64};
const packet=()=>({schema:'mycoscout.live-frame.v1',sessionId:'test-recording',frameId:'frame-0',sequence:0,timestampSeconds:1,simulationOnly:true,coordinateFrame:'map-enu',mapRevision:'map-1',actuationEnabled:false,
 image:{...image},detections:[{detectionId:'d1',bboxPixelsXYXY:[0,0,1,1],confidence:.5,trackId:'t1',landmarkId:'m1'}],tracks:[],landmarks:[{landmarkId:'m1',trackId:'t1',positionEnuM:null,gps:null,rangeM:null}],mapPoints:[]});
async function fixture(t,token=''){
 const dir=await mkdtemp(join(tmpdir(),'mycoscout-live-api-')),app=await createApp({host:'127.0.0.1',token,stateFile:join(dir,'state.json')});const address=await app.listen(0),base=`http://127.0.0.1:${address.port}`;
 const req=async(path,body,authorized=true)=>{const response=await fetch(base+path,{method:body===undefined?'GET':'POST',headers:{...(body===undefined?{}:{'Content-Type':'application/json'}),...(token&&authorized?{Authorization:'Bearer '+token}:{})},...(body===undefined?{}:{body:JSON.stringify(body)})});return {status:response.status,data:await response.json()};};
 t.after(async()=>{app.server.closeAllConnections();await new Promise(done=>app.server.close(done));await app.closed;await app.live.close();await app.learning.stop();assert.ok(resolve(dir).startsWith(resolve(tmpdir())+sep));await rm(dir,{recursive:true,force:true});});return {app,base,req};
}
test('camera byte verification, persistent track review and task queue work across shared APIs',async t=>{
 const {req,base}=await fixture(t);assert.equal((await req('/api/live/session',{sessionId:'test-recording',mode:'replay'})).status,201);
 const bad=packet();bad.image.sha256='a'.repeat(64);assert.equal((await req('/api/live/frames',bad)).status,400);assert.equal((await req('/api/live/state')).data.landmarks.length,0);
 const accepted=await req('/api/live/frames',packet());assert.equal(accepted.status,201);const state=(await req('/api/live/state')).data;assert.equal(state.landmarks.length,1);assert.equal(state.landmarks[0].gps,null);
 const preview=await fetch(base+state.latestFrame.image.url);assert.deepEqual(Buffer.from(await preview.arrayBuffer()),Buffer.from(imageBase64,'base64'));
 assert.equal((await req('/api/live/frames',packet())).data.duplicate,true);
 const task=await req('/api/live/tasks',{type:'photograph',landmarkId:'m1',idempotencyKey:'one-task',baseRevision:state.revision});assert.equal(task.status,201);
 assert.equal((await req('/api/live/tasks',{type:'photograph',landmarkId:'m1',idempotencyKey:'one-task'})).data.duplicate,true);
 const tick=await req('/api/live/tick',{});assert.equal(tick.data.action.type,'hold');assert.equal(tick.data.action.reason,'target_position_unverified');
 assert.equal((await req('/api/live/tick',{captureAck:{}})).status,409);
 const reviewed=await req('/api/live/landmarks/m1/review',{status:'confirmed',reviewer:'Test fixture',groupId:'boletes'});assert.equal(reviewed.data.landmarks[0].humanVerified,true);assert.equal(reviewed.data.landmarks[0].review.trainingAllowed,false);
 assert.equal((await req('/api/learning/status')).status,200);
});
test('mission images and mutations inherit authentication, origin and session boundaries',async t=>{
 const {req}=await fixture(t,'live-test-token-at-least-24-characters');assert.equal((await req('/api/live/state',undefined,false)).status,401);
 await req('/api/live/session',{sessionId:'live-test',mode:'live'});const mismatch=await req('/api/live/frames',packet());assert.equal(mismatch.status,409);
 assert.equal((await req('/api/live/replay/start',{source:'../../secret'})).status,400);
 assert.equal((await req('/api/live/captures',{imageBase64,ack:{synthetic:true}})).status,409);
});
test('browser simulation finishes only after modeled capture acknowledgment and keeps real actuation off',async t=>{
 const {req}=await fixture(t);let state=(await req('/api/live/simulation/start',{blocked:false})).data;assert.equal(state.simulationDemo,true);
 for(let i=0;i<8&&!state.tasks.some(t=>t.status==='completed');i++){const result=await req('/api/live/simulation/step',{baseRevision:state.revision});assert.equal(result.status,200);state=result.data.state;}
 assert.equal(state.tasks[0].status,'completed');assert.equal(state.actuationEnabled,false);assert.equal(state.tasks[0].capture.synthetic,true);
 const old=state.revision;await req('/api/live/landmarks/sim-mushroom-1/review',{status:'uncertain',reviewer:'Test fixture'});assert.equal((await req('/api/live/tick',{baseRevision:old})).status,409);
});

test('independent depth navigation reaches the shared mission before semantic recognition',async t=>{
 const {req}=await fixture(t,'navigation-token-at-least-24-characters');
 const nav=JSON.parse(await readFile(new URL('../examples/live-session-navigation-synthetic.json',import.meta.url),'utf8'));
 await req('/api/live/session',{sessionId:nav.sessionId,mode:'replay'});
 assert.equal((await req('/api/live/navigation',nav,false)).status,401);
 const result=await req('/api/live/navigation',nav);assert.equal(result.status,201);assert.equal(result.data.action.reason,'awaiting_semantic_observation');
 const state=(await req('/api/live/state')).data;assert.equal(state.latestFrame,null);assert.equal(state.telemetry.batteryPercent,nav.telemetry.batteryPercent);
 assert.ok(Object.values(state.voxelMap.voxels).some(cell=>cell.state==='free'));
 assert.equal((await req('/api/live/navigation',nav)).data.duplicate,true);
 assert.equal((await req('/api/live/tick',{sessionId:'another-session',baseControlRevision:state.controlRevision})).status,409);
});

test('a task cancelled while its original photo is saving cannot acknowledge a different task',async t=>{
 const {app,req,base}=await fixture(t);app.mission.seedSimulation();for(let i=0;i<4;i++)app.mission.stepSimulation();
 const state=app.mission.requireActive(),task=state.tasks[0],capture=task.captureRequest;assert.ok(capture);
 state.mode='live';capture.requestedMs=Date.now()-10;app.mission.save(state);
 const ack={...capture,capturedMs:Date.now(),synthetic:false,photoSha256:image.sha256,photoRef:'camera-original',nativeUncropped:true,width:1,height:1,positionEnuM:capture.positionEnuM,yawRad:capture.aim.yawRad,pitchRad:capture.aim.pitchRad,settled:true,sharpnessAccepted:true,targetBboxPixelsXYXY:[0,0,1,1]};
 const originalCreate=app.finds.create.bind(app.finds);let saved;
 app.finds.create=async input=>{saved=await originalCreate(input);app.mission.updateTask(task.id,{action:'cancel'});return saved;};
 const result=await req('/api/live/captures',{imageBase64,mimeType:'image/png',ack});
 assert.equal(result.status,409);assert.equal(result.data.code,'LIVE_CAPTURE_TASK_CHANGED');assert.equal(result.data.savedFindId,saved.id);
 assert.equal(app.mission.snapshot().tasks[0].status,'cancelled');
 const photo=await fetch(base+'/api/finds/'+saved.id+'/image');assert.equal(photo.status,200);assert.deepEqual(Buffer.from(await photo.arrayBuffer()),Buffer.from(imageBase64,'base64'));
});

test('saved camera bytes complete only a still-valid capture and otherwise remain in review',async t=>{
 for(const interrupt of [false,true])await t.test(interrupt?'new battery hold during save':'unchanged request',async t=>{
  const {app,req}=await fixture(t);app.mission.seedSimulation();for(let i=0;i<4;i++)app.mission.stepSimulation();
  const state=app.mission.requireActive(),task=state.tasks[0],capture=task.captureRequest,now=Date.now();
  state.mode='live';state.lastCaptureSeconds=now/1000;state.receivedAtMs=now;state.landmarks[0].lastSeenMs=now;
  state.telemetry.timestampSeconds=now/1000;state.telemetry.depthTimestampSeconds=now/1000;state.voxelMap.updatedMs=now;
  for(const cell of Object.values(state.voxelMap.voxels))cell.observedMs=now;
  task.settledSinceMs=now-1200;capture.requestedMs=now-20;app.mission.save(state);
  const ack={...capture,capturedMs:now,synthetic:false,photoSha256:image.sha256,photoRef:'camera-original',nativeUncropped:true,width:1,height:1,positionEnuM:capture.positionEnuM,yawRad:capture.aim.yawRad,pitchRad:capture.aim.pitchRad,settled:true,sharpnessAccepted:true,targetBboxPixelsXYXY:[0,0,1,1]};
  if(interrupt){const original=app.finds.create.bind(app.finds);app.finds.create=async input=>{const find=await original(input),current=app.mission.requireActive();current.telemetry.batteryPercent=10;app.mission.save(current);return find;};}
  const result=await req('/api/live/captures',{imageBase64,mimeType:'image/png',ack});
  if(interrupt){assert.equal(result.status,409);assert.equal(result.data.code,'CAPTURE_NOT_ACCEPTED');assert.ok(result.data.savedFindId);assert.equal(app.mission.snapshot().tasks[0].blockedReason,'battery_reserve_required');}
  else{assert.equal(result.status,201,JSON.stringify(result.data));assert.equal(result.data.action.type,'completed');assert.equal(result.data.action.taskId,task.id);assert.equal(result.data.find.image.sha256,image.sha256);assert.equal(result.data.find.metadata.requestId,capture.requestId);assert.equal(result.data.action.capture.photoRef,result.data.find.imageUrl);assert.equal(result.data.state.landmarks[0].capture.photoRef,result.data.find.imageUrl);}
 });
});
