import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { LiveMissionStore, validateLivePacket } from '../core/live-mission.mjs';
import { createSimulationPacket, createSyntheticCaptureAck, validateCaptureAck, stepLiveExecutor } from '../flight/live-executor.mjs';

async function fixture(t,{nowMs=1800000000000}={}) {
  const directory=await mkdtemp(join(tmpdir(),'mycoscout-mission-'));let clock=nowMs;
  const store=await new LiveMissionStore(directory,{now:()=>clock}).initialize();
  t.after(async()=>{if(store.db)store.close();await rm(directory,{recursive:true,force:true});});
  store.startSession({sessionId:'test-session',mode:'replay'});
  const packet=createSimulationPacket({sessionId:'test-session'});
  return {store,directory,packet,advance:ms=>clock+=ms,now:()=>clock};
}

function navigation(packet){
  const data=structuredClone(packet),telemetry=data.flight.telemetry,voxelMap=data.flight.voxelMap;
  telemetry.cameraPositionEnuM=[...telemetry.positionEnuM];telemetry.cameraMountMode='rigid_calibrated';
  voxelMap.evidence={schema:'mycoscout.observed-depth-volume.v1',sessionId:data.sessionId,freeSpaceMethod:'full_projected_voxel_footprint',calibrationId:data.camera.calibrationId,depthTimestampSeconds:data.timestampSeconds,poseTimestampSeconds:data.timestampSeconds,positionStdM:.02,orientationStdRad:.01,depthStdM:.02};
  return {schema:'mycoscout.live-navigation.v1',sessionId:data.sessionId,mapRevision:data.mapRevision,timestampSeconds:data.timestampSeconds,simulationOnly:data.simulationOnly,telemetry,voxelMap};
}

test('frame deduplication survives restart and rejects edited evidence',async t=>{
  const {store,directory,packet,now}=await fixture(t);
  const first=store.ingestFrame(packet);assert.equal(first.state.landmarks.length,1);assert.equal(first.state.tasks.length,1);
  assert.equal(store.ingestFrame(packet).duplicate,true);
  const changed=structuredClone(packet);changed.detections[0].confidence=.5;
  assert.throws(()=>store.ingestFrame(changed),{code:'LIVE_FRAME_CONFLICT'});
  store.close();const reopened=await new LiveMissionStore(directory,{now}).initialize();
  try{assert.equal(reopened.ingestFrame(packet).duplicate,true);assert.equal(reopened.snapshot().landmarks[0].observationCount,1);}finally{reopened.close();}
});

test('ordered synchronized captures prevent backwards poses and frame order',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);
  const earlier=createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:99});
  assert.throws(()=>store.ingestFrame(earlier),{code:'LIVE_OUT_OF_ORDER'});
  const skewed=createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:101});skewed.camera.poseTimestampSeconds=100;
  assert.throws(()=>store.ingestFrame(skewed),/synchronized/);
  assert.equal(store.snapshot().lastSequence,0);
});

test('RGB-only evidence is reviewable but never gains metric points or GPS',async t=>{
  const {store,packet}=await fixture(t);
  delete packet.camera;delete packet.depth;delete packet.flight;packet.mapPoints=[];
  Object.assign(packet.landmarks[0],{positionEnuM:null,positionStdM:null,rangeM:null,gps:null});
  store.ingestFrame(packet);assert.equal(store.snapshot().landmarks[0].inspectionPriority,'unlocated');assert.equal(store.snapshot().tasks.length,0);
  store.enqueue({type:'inspect',landmarkId:'sim-mushroom-1',idempotencyKey:'rgb-task'});
  assert.equal(store.tick().action.reason,'target_position_unverified');
  packet.landmarks[0].positionEnuM=[1,2,3];assert.throws(()=>validateLivePacket(packet,'replay'),/Metric landmarks/);
});

test('unanchored GPS and uncalibrated metric maps are rejected atomically',async t=>{
  const {store,packet}=await fixture(t);packet.landmarks[0].gps={latitude:43,longitude:-85,altitudeM:230,horizontalUncertaintyM:3};
  assert.throws(()=>store.ingestFrame(packet),/georeferencing evidence/);
  assert.equal(store.snapshot().frames.length,0);
  packet.landmarks[0].gps=null;packet.camera.cameraToMap4x4[0][0]=2;
  assert.throws(()=>store.ingestFrame(packet),/Metric landmarks/);
});

test('50m uncertainty boundary sends observations to habitat log',async t=>{
  const {store,packet}=await fixture(t);packet.landmarks[0].rangeM=49.95;packet.landmarks[0].positionStdM=.02;packet.landmarks[0].positionEnuM=[45.95,0,1.5];
  store.ingestFrame(packet);assert.equal(store.snapshot().landmarks[0].inspectionPriority,'habitat');assert.equal(store.snapshot().tasks.length,0);
  const closer=createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:101});closer.landmarks[0].rangeM=49;closer.landmarks[0].positionEnuM=[45,0,1.5];
  store.ingestFrame(closer);assert.equal(store.snapshot().landmarks[0].inspectionPriority,'immediate');assert.equal(store.snapshot().tasks.length,1);
});

test('queue idempotency, revision conflicts, task controls and review persist',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);
  const request={type:'photograph',landmarkId:'sim-mushroom-1',idempotencyKey:'one-photo',baseRevision:store.snapshot().revision};
  const added=store.enqueue(request);assert.equal(added.duplicate,false);assert.equal(store.enqueue(request).duplicate,true);
  assert.throws(()=>store.enqueue({...request,type:'scan'}),{code:'LIVE_IDEMPOTENCY_CONFLICT'});
  assert.throws(()=>store.updateTask(added.task.id,{action:'pause',baseRevision:1}),{code:'LIVE_REVISION_CONFLICT'});
  store.updateTask(added.task.id,{action:'pause'});assert.equal(store.snapshot().tasks[1].status,'paused');
  store.updateTask(added.task.id,{action:'resume'});store.updateTask(added.task.id,{action:'move',beforeTaskId:store.snapshot().tasks[0].id});
  assert.equal(store.snapshot().tasks[0].id,added.task.id);
  store.review('sim-mushroom-1',{status:'rejected',reviewer:'field observer'});
  assert.ok(store.snapshot().tasks.every(t=>t.status==='cancelled'));assert.equal(store.snapshot().landmarks[0].review.trainingAllowed,false);
});

test('session switching isolates maps, modes and queues and pauses an active mission',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);store.tick();
  store.startSession({sessionId:'other',mode:'live'});assert.equal(store.snapshot().landmarks.length,0);
  assert.throws(()=>store.ingestFrame(packet),{code:'LIVE_MODE_MISMATCH'});
  store.startSession({sessionId:'test-session',mode:'replay'});assert.equal(store.snapshot().tasks[0].status,'paused');
  assert.throws(()=>store.startSession({sessionId:'test-session',mode:'live'}),{code:'LIVE_MODE_MISMATCH'});
});

test('map revision change requires explicit retry and target reacquisition',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);store.tick();
  const changed=createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:101});changed.mapRevision='relocalized-map';changed.flight.voxelMap.mapRevision='relocalized-map';store.ingestFrame(changed);
  assert.equal(store.snapshot().tasks[0].blockedReason,'map_revision_changed_reacquire_target');
  assert.equal(store.tick().action.type,'idle');
  store.updateTask(store.snapshot().tasks[0].id,{action:'retry'});assert.equal(store.tick().action.type,'move_proposal');
});

test('old landmarks are not drawn in a changed map before reacquisition',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);
  const changed=createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:101});changed.mapRevision='relocalized-map';changed.flight.voxelMap.mapRevision='relocalized-map';changed.landmarks=[];changed.tracks=[];changed.detections=[];
  store.ingestFrame(changed);const landmark=store.snapshot().landmarks[0];assert.equal(landmark.positionEnuM,null);assert.equal(landmark.gps,null);assert.deepEqual(landmark.previousLocation.positionEnuM,[1,0,0]);
});

test('stale, unknown or occupied space blocks moves',async t=>{
  const {store,packet,advance}=await fixture(t);store.ingestFrame(packet);advance(11000);
  assert.equal(store.tick().action.reason,'target_observation_stale');
  let demo=store.seedSimulation({blocked:true});const result=store.stepSimulation();assert.equal(result.action.type,'hold');assert.match(result.action.reason,/corridor|occupied/);
  demo=store.seedSimulation();const state=store.requireActive();state.voxelMap.voxels={};store.save(state);
  assert.equal(store.tick().action.reason,'unknown_voxel');
});

test('low battery, missing localization and missing depth block movement',async t=>{
  const {store}=await fixture(t);
  for(const [field,value,reason] of [['batteryPercent',20,'battery_reserve_required'],['localized',false,'fresh_localization_required'],['depthValid',false,'fresh_depth_required']]){
    store.seedSimulation();const state=store.requireActive();state.telemetry[field]=value;store.save(state);assert.equal(store.tick().action.reason,reason);
  }
});

test('explicit simulation demonstrates approach, aim, settle, saved capture ACK and review',async t=>{
  const {store}=await fixture(t);store.seedSimulation();const actions=[];
  for(let i=0;i<5;i++)actions.push(store.stepSimulation().action.type);
  assert.deepEqual(actions,['move_proposal','aim_proposal','settle','capture_request','completed']);
  const state=store.snapshot(),task=state.tasks[0];assert.equal(task.capture.synthetic,true);assert.equal(task.capture.framingSatisfied,true);
  assert.equal(state.landmarks[0].capture.taskId,task.id);assert.equal(state.landmarks[0].capture.photoRef,task.capture.photoRef);
  assert.match(task.capture.photoRef,/^simulation:/);assert.equal(state.actuationEnabled,false);assert.equal(state.capabilities.realFlight,false);
});

test('normal replay never fabricates a capture ACK and rejects unproven camera evidence',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);
  assert.throws(()=>store.stepSimulation(),{code:'LIVE_SIMULATION_REQUIRED'});
  const state=store.requireActive(),task=state.tasks[0];task.status='running';task.phase='capture';task.captureRequest={requestId:'capture-test',taskId:task.id,sessionId:state.sessionId,landmarkId:task.landmarkId,mapRevision:state.mapRevision,requestedMs:100000,positionEnuM:[-4,0,1.5],aim:{yawRad:0,pitchRad:0}};
  assert.equal(validateCaptureAck({synthetic:true},task,state,100000).accepted,false);
  const live={...state,mode:'live'};assert.equal(validateCaptureAck({...task.captureRequest,synthetic:true},task,live,100000).reason,'synthetic_capture_forbidden_in_live_session');
});

test('capture requires a matching acknowledgment and expires while fresh telemetry continues',async t=>{
  const {store}=await fixture(t);store.seedSimulation();for(let i=0;i<4;i++)store.stepSimulation();
  const before=store.requireActive(),task=before.tasks[0];assert.equal(before.lastAction.type,'capture_request');
  assert.throws(()=>store.tick({captureAck:{requestId:'wrong'}}),{code:'LIVE_CAPTURE_TASK_CHANGED'});assert.equal(store.snapshot().tasks[0].status,'running');
  const nowMs=task.captureRequest.requestedMs+9000;
  before.landmarks[0].lastSeenMs=nowMs;before.telemetry.timestampSeconds=nowMs/1000;before.telemetry.depthTimestampSeconds=nowMs/1000;before.voxelMap.updatedMs=nowMs;
  for(const cell of Object.values(before.voxelMap.voxels))cell.observedMs=nowMs;
  const result=stepLiveExecutor(before,{nowMs});assert.equal(result.reason,'capture_ack_timeout');assert.equal(task.status,'blocked');
});

test('late capture acknowledgments never advance another task after pause, cancel or block',async t=>{
  for(const transition of ['pause','cancel','block'])await t.test(transition,async st=>{
    const {store}=await fixture(st);store.seedSimulation();for(let i=0;i<4;i++)store.stepSimulation();
    const captured=store.snapshot(),task=captured.tasks[0],ack=createSyntheticCaptureAck(captured,task,task.captureRequest.requestedMs);
    const next=store.enqueue({type:'photograph',landmarkId:task.landmarkId,idempotencyKey:'next-photo'}).task;
    if(transition==='block'){
      const current=store.requireActive();current.telemetry.batteryPercent=20;store.save(current);
      assert.equal(store.tick().action.type,'hold');
    }else store.updateTask(task.id,{action:transition});
    const before=store.snapshot();
    assert.throws(()=>store.tick({captureAck:ack}),{status:409,code:'LIVE_CAPTURE_TASK_CHANGED'});
    const after=store.snapshot();assert.equal(after.revision,before.revision);assert.deepEqual(after.tasks,before.tasks);assert.deepEqual(after.lastAction,before.lastAction);
    assert.equal(after.tasks.find(item=>item.id===next.id).status,'queued');
  });
});

test('capture acknowledgments cannot bind to a replacement request or a switched session',async t=>{
  const {store}=await fixture(t);store.seedSimulation();for(let i=0;i<4;i++)store.stepSimulation();
  const state=store.snapshot(),task=state.tasks[0],ack=createSyntheticCaptureAck(state,task,task.captureRequest.requestedMs);
  const replaced=store.requireActive();replaced.tasks[0].captureRequest.requestId='new-camera-request';store.save(replaced);
  let before=store.snapshot();assert.throws(()=>store.tick({captureAck:ack}),{code:'LIVE_CAPTURE_TASK_CHANGED'});assert.equal(store.snapshot().revision,before.revision);
  store.startSession({sessionId:'switched-empty-session',mode:'live'});before=store.snapshot();
  assert.throws(()=>store.tick({captureAck:ack}),{code:'LIVE_CAPTURE_TASK_CHANGED'});assert.equal(store.snapshot().revision,before.revision);
  store.seedSimulation();for(let i=0;i<4;i++)store.stepSimulation();before=store.snapshot();
  assert.throws(()=>store.tick({captureAck:ack}),{code:'LIVE_CAPTURE_TASK_CHANGED'});assert.deepEqual(store.snapshot().tasks,before.tasks);
});

test('a paused simulation cannot apply an old movement proposal',async t=>{
  const {store}=await fixture(t);store.seedSimulation();store.stepSimulation();const before=store.snapshot();
  store.updateTask(before.tasks[0].id,{action:'pause'});store.stepSimulation();assert.deepEqual(store.snapshot().telemetry.positionEnuM,before.telemetry.positionEnuM);
});

test('slow live detector remains visible but cannot refresh old sensor evidence',async t=>{
  const {store,now}=await fixture(t);store.startSession({sessionId:'live-session',mode:'live'});
  const packet=createSimulationPacket({sessionId:'live-session',timeSeconds:now()/1000-6});packet.simulationOnly=false;
  store.ingestFrame(packet);assert.equal(store.snapshot().connection.captureAgeMs,6000);assert.equal(store.tick().action.reason,'fresh_localization_required');
  const future=createSimulationPacket({sessionId:'live-session',sequence:1,timeSeconds:now()/1000+1});future.simulationOnly=false;
  assert.throws(()=>store.ingestFrame(future),{code:'LIVE_CLOCK_INVALID'});
});

test('sensor frame updates do not invalidate operator control intent; another operator does',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);const initial=store.snapshot();
  store.ingestFrame(createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:101}));
  assert.equal(store.snapshot().controlRevision,initial.controlRevision);assert.notEqual(store.snapshot().revision,initial.revision);
  store.enqueue({sessionId:'test-session',baseControlRevision:initial.controlRevision,type:'photograph',landmarkId:'sim-mushroom-1',idempotencyKey:'control-test'});
  assert.throws(()=>store.review('sim-mushroom-1',{sessionId:'test-session',baseControlRevision:initial.controlRevision,status:'confirmed'}),{code:'LIVE_REVISION_CONFLICT'});
});

test('navigation can precede slow semantics and old frames cannot overwrite it',async t=>{
  const {store,packet}=await fixture(t);
  const nav=navigation(createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:101,positionEnuM:[-3,0,1.5]}));
  assert.equal(store.updateNavigation(nav).action.reason,'awaiting_semantic_observation');
  const control=store.snapshot().controlRevision;assert.equal(store.updateNavigation(nav).duplicate,true);
  store.ingestFrame(packet);assert.deepEqual(store.snapshot().telemetry.positionEnuM,[-3,0,1.5]);assert.equal(store.snapshot().lastNavigationSeconds,101);
  assert.equal(control,0);assert.equal(store.tick().action.type,'move_proposal');
});

test('navigation rejects stale, reordered or unproven free volume atomically',async t=>{
  const {store,packet}=await fixture(t);store.ingestFrame(packet);
  const nav=navigation(createSimulationPacket({sessionId:'test-session',sequence:1,timeSeconds:101}));
  const bad=structuredClone(nav);delete bad.voxelMap.evidence;assert.throws(()=>store.updateNavigation(bad),/depth-volume evidence/);
  assert.equal(store.updateNavigation(nav).action.type,'move_proposal');
  const changed=structuredClone(nav);changed.telemetry.batteryPercent=50;assert.throws(()=>store.updateNavigation(changed),{code:'LIVE_OUT_OF_ORDER'});
  const other=structuredClone(nav);other.mapRevision='new-map';assert.throws(()=>store.updateNavigation(other),{code:'LIVE_MAP_MISMATCH'});
  assert.equal(store.snapshot().telemetry.batteryPercent,85);
});

test('fast live navigation does not refresh a stationary landmark after its reacquisition deadline',async t=>{
  const {store,now,advance}=await fixture(t);store.startSession({sessionId:'live-session',mode:'live'});
  const slow=createSimulationPacket({sessionId:'live-session',timeSeconds:now()/1000-6});slow.simulationOnly=false;delete slow.flight;store.ingestFrame(slow);
  const current=createSimulationPacket({sessionId:'live-session',sequence:1,timeSeconds:now()/1000});current.simulationOnly=false;
  assert.equal(store.updateNavigation(navigation(current)).action.type,'move_proposal');
  advance(5000);const newer=createSimulationPacket({sessionId:'live-session',sequence:2,timeSeconds:now()/1000});newer.simulationOnly=false;
  assert.equal(store.updateNavigation(navigation(newer)).action.reason,'target_observation_stale');
  assert.equal(store.snapshot().landmarks[0].lastSeenMs,slow.timestampSeconds*1000);
});

test('camera aim uses calibrated optical origin instead of vehicle body origin',async t=>{
  const {store}=await fixture(t);store.seedSimulation();store.stepSimulation();store.stepSimulation();
  const state=store.requireActive();state.telemetry.cameraPositionEnuM=[state.telemetry.positionEnuM[0],1,1.5];state.telemetry.yawRad=0;state.telemetry.pitchRad=0;store.save(state);
  const action=store.tick().action;assert.equal(action.type,'aim_proposal');assert.ok(action.aim.yawRad<-.5);
});
