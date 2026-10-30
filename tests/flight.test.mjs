import test from 'node:test';
import assert from 'node:assert/strict';
import { SurveyMission } from '../flight/mission.mjs';
import { projection,blurPixels,blurLimitedSpeed,sweptClearance,planCorridor3D } from '../flight/geometry.mjs';
import { configFixture,sceneFixture,inputFixture,candidateFixture,box,refresh } from '../flight/fixtures.mjs';
import { runScenarios } from '../flight/run-simulation.mjs';

const radiusM=0.87;
const clear=(map,a,b,nowMs=0)=>sweptClearance(map,a,b,{radiusM,nowMs});
function ready(waypoints=[{x:3,y:0,z:2}]) {
  const mission=new SurveyMission(configFixture(),waypoints);
  mission.step({...inputFixture(0),event:'preflight'});
  assert.equal(mission.step(inputFixture(100)).state,'READY');
  return mission;
}
function running() { const mission=ready(); mission.step({...inputFixture(200),event:'begin_simulation'}); return mission; }

test('camera projection halves pixel width when range doubles',()=>{
  const c=configFixture().camera;
  assert.equal(projection(c,1),69); assert.equal(projection(c,2),34.5);
  assert.throws(()=>projection(c,0),/rangeM/);
});
test('image blur derives speed limit from focal length, exposure and rotation',()=>{
  const c=configFixture().camera,v=blurLimitedSpeed(c,1);
  assert.ok(Math.abs(blurPixels(c,{speedMps:v,rangeM:1})-2)<1e-9);
  assert.ok(blurLimitedSpeed({...c,exposureS:c.exposureS*2},1)<v);
  assert.equal(blurLimitedSpeed(c,1,1),0);
});
test('3D segment detects a branch even when both endpoints are clear',()=>{
  const map=box(sceneFixture(),{x:1,y:-0.5,z:2.5},{x:1.5,y:0.5,z:3});
  assert.equal(clear(map,{x:0,y:0,z:2},{x:0,y:0,z:2}).clear,true);
  assert.equal(clear(map,{x:3,y:0,z:2},{x:3,y:0,z:2}).clear,true);
  assert.equal(clear(map,{x:0,y:0,z:2},{x:3,y:0,z:2}).reason,'occupied_voxel');
});
test('unknown cells are not treated as open air',()=>{
  const map=box(sceneFixture(),{x:1,y:-0.5,z:1.5},{x:1.5,y:0.5,z:2.5},'unknown');
  assert.equal(clear(map,{x:0,y:0,z:2},{x:3,y:0,z:2}).reason,'unknown_voxel');
});
test('voxel freshness is checked individually even when map timestamp is fresh',()=>{
  const map=sceneFixture(0); map.updatedMs=2000;
  assert.equal(clear(map,{x:0,y:0,z:2},{x:1,y:0,z:2},2000).reason,'stale_voxel');
});
test('map exterior and floor require full rotor envelope clearance',()=>{
  assert.equal(clear(sceneFixture(),{x:0,y:0,z:0.6},{x:1,y:0,z:0.6}).reason,'occupied_voxel');
  assert.equal(clear(sceneFixture(),{x:-1.5,y:0,z:2},{x:0,y:0,z:2}).reason,'outside_known_volume');
});
test('2D walking map and malformed geometry are rejected',()=>{
  assert.equal(clear({frame:'map',width:20,height:20,cells:[]},{x:0,y:0,z:2},{x:1,y:0,z:2}).reason,'invalid_geometry');
  assert.equal(clear(sceneFixture(),{x:NaN,y:0,z:2},{x:1,y:0,z:2}).clear,false);
});
test('future or missing free-cell timestamps block geometry',()=>{
  const map=sceneFixture(); map.voxels['4,6,6'].observedMs=10000;
  assert.equal(clear(map,{x:0,y:0,z:2},{x:1,y:0,z:2}).reason,'stale_voxel');
});
test('3D planner routes around a trunk and every simplified edge clears the body',()=>{
  const map=box(sceneFixture(),{x:1.5,y:-0.5,z:0},{x:2,y:0.5,z:5}),start={x:0,y:0,z:2},goal={x:4,y:0,z:2};
  const plan=planCorridor3D(map,start,goal,{radiusM,nowMs:0});
  assert.equal(plan.status,'ready'); assert.ok(plan.points.some(p=>Math.abs(p.y)>1));
  for(let i=1;i<plan.points.length;i++) assert.equal(clear(map,plan.points[i-1],plan.points[i]).clear,true);
  assert.ok(plan.points.every(p=>p.z===2));
});
test('3D planning does not authorize descent by default',()=>{
  const route=planCorridor3D(sceneFixture(),{x:0,y:0,z:2},{x:3,y:0,z:3},{radiusM,nowMs:0});
  assert.equal(route.reason,'altitude_change_not_authorized');
});
test('3D planning supports explicitly bounded known vertical routes',()=>{
  const route=planCorridor3D(sceneFixture(),{x:0,y:0,z:2},{x:3,y:0,z:3},{radiusM,nowMs:0,allowVertical:true,minZM:1.5,maxZM:3.5});
  assert.equal(route.status,'ready');
});
test('preflight and begin are distinct simulation events with no arm command',()=>{
  const m=ready(); assert.equal(m.state,'READY'); assert.equal(m.step(inputFixture(200)).proposal,null);
  const result=m.step({...inputFixture(300),event:'begin_simulation'});
  assert.equal(result.state,'SURVEY_TRANSIT'); assert.equal(result.actuationEnabled,false); assert.equal(result.simulationOnly,true);
  assert.equal(JSON.stringify(m.history).includes('arm_command'),false);
});
test('invalid preflight remains unable to start',()=>{
  const m=new SurveyMission(configFixture(),[{x:3,y:0,z:2}]); m.step({...inputFixture(0),event:'preflight'});
  const i=inputFixture(100); i.telemetry.depth.valid=false;
  assert.equal(m.step({...i,event:'begin_simulation'}).state,'PREFLIGHT');
});
test('non-simulation config and understated pose envelope are rejected',()=>{
  const c=configFixture(); c.simulationOnly=false; assert.throws(()=>new SurveyMission(c,[{x:0,y:0,z:2}]),/simulation/);
  const d=configFixture(); d.vehicle.maxPoseSigmaM=0.01; assert.throws(()=>new SurveyMission(d,[{x:0,y:0,z:2}]),/uncertainty/);
});
for(const reason of ['localization_unreliable','depth_unavailable','map_stale','battery_reserve','manual_link_lost','invalid_motion']) {
  test(`active mission latches hold on ${reason}`,()=>{
    const m=running(),i=inputFixture(300);
    if(reason==='localization_unreliable') i.telemetry.localization.valid=false;
    if(reason==='depth_unavailable') i.telemetry.depth.updatedMs=-1000;
    if(reason==='map_stale') i.map.updatedMs=-1000;
    if(reason==='battery_reserve') i.telemetry.batteryFraction=0.1;
    if(reason==='manual_link_lost') i.telemetry.manualLinkAvailable=false;
    if(reason==='invalid_motion') i.telemetry.speedMps=NaN;
    const result=m.step(i); assert.equal(result.state,'HOLD'); assert.equal(result.reason,reason); assert.equal(result.manualTakeoverRequired,true);
    if(reason==='localization_unreliable') assert.equal(result.proposal,null);
    assert.equal(m.step(inputFixture(400)).reason,'explicit_resume_required');
  });
}
test('stale localization cannot produce a position-hold setpoint',()=>{
  const m=running(),i=inputFixture(300); i.telemetry.localization.updatedMs=-100;
  const out=m.step(i); assert.equal(out.proposal,null); assert.equal(out.manualTakeoverRequired,true);
});
test('long loop gap and backward clocks fail hold',()=>{
  assert.equal(running().step(inputFixture(2000)).reason,'control_loop_gap');
  assert.equal(running().step(inputFixture(100)).reason,'control_loop_gap');
});
test('explicit resume revalidates the new corridor',()=>{
  const m=running(),i=inputFixture(300); i.telemetry.depth.valid=false; m.step(i);
  const resumed=m.step({...inputFixture(400),event:'resume'}); assert.equal(resumed.state,'SURVEY_TRANSIT');
});
test('manual takeover and abort are latched terminal states',()=>{
  for(const event of ['manual','abort']) { const m=running(),out=m.step({...inputFixture(300),event}); assert.equal(out.state,event==='manual'?'MANUAL':'ABORTED'); assert.equal(m.step({...inputFixture(400),event:'resume'}).state,out.state); }
});
test('stop-and-look refuses captures in motion, then waits a full settle interval',()=>{
  const m=ready([{x:0,y:0,z:2}]); let now=200;
  let i=inputFixture(now); i.telemetry.speedMps=0.2; i.event='begin_simulation'; assert.equal(m.step(i).captureRequested,false);
  const yaw=configFixture().survey.captureYawOffsetsRad[0]; let result;
  for(let n=0;n<6;n++) { now+=200; i=inputFixture(now); i.telemetry.yawRad=yaw; result=m.step(i); if(n<5) assert.equal(result.captureRequested,false); }
  assert.equal(result.captureRequested,true); assert.equal(result.capture.synthetic,true);
});
test('drift from the camera station cancels settling and repositions',()=>{
  const m=ready([{x:0,y:0,z:2}]); m.step({...inputFixture(200),event:'begin_simulation'});
  const out=m.step(inputFixture(300,{x:0.3,y:0,z:2})); assert.equal(out.reason,'drifted_from_capture_station');
});
test('candidate begins verification hover and does not descend on detection',()=>{
  const m=running(),i={...inputFixture(300),event:'candidate',candidate:candidateFixture(300)};
  const out=m.step(i); assert.equal(out.state,'CANDIDATE_VERIFY'); assert.equal(out.proposal.type,'position_hold'); assert.equal(out.proposal.position.z,2);
});
test('repeated same-frame detections do not satisfy verification evidence',()=>{
  const m=running(),c=candidateFixture(300); c.observations.forEach(o=>o.frameId='same');
  m.step({...inputFixture(300),event:'candidate',candidate:c});
  assert.equal(m.step({...inputFixture(400),event:'approach'}).reason,'candidate_needs_more_frames');
});
test('candidate approach rejects unknown corridor and any descent',()=>{
  const m=running(),c=candidateFixture(300,{x:3,y:0,z:1});
  m.step({...inputFixture(300),event:'candidate',candidate:c});
  assert.equal(m.step({...inputFixture(400),event:'approach'}).reason,'approach_altitude_change_blocked');
  const n=running(),candidate=candidateFixture(300,{x:1.8,y:0,z:2}); n.step({...inputFixture(300),event:'candidate',candidate});
  const map=box(sceneFixture(400),{x:1,y:-0.5,z:1.5},{x:1.5,y:0.5,z:2.5},'unknown');
  assert.equal(n.step({...inputFixture(400,{x:0,y:0,z:2},map),event:'approach'}).reason,'approach_corridor_blocked');
});
test('candidate approach follows a verified same-height corridor and expires evidence',()=>{
  const m=running(),c=candidateFixture(300); m.step({...inputFixture(300),event:'candidate',candidate:c});
  assert.equal(m.step({...inputFixture(400),event:'approach'}).state,'APPROACH_TRANSIT');
  for(let now=600;now<=2400;now+=200) m.step(inputFixture(now));
  assert.equal(m.state,'HOLD'); assert.equal(m.history.at(-1).reason,'candidate_evidence_expired');
});
test('braking reserves volume in all directions, including a lateral branch',()=>{
  const m=running(),i=inputFixture(300); i.telemetry.speedMps=1;
  // The ordinary body corridor is free; a high-speed stopping envelope reaches the floor.
  assert.equal(m.step(i).reason,'braking_corridor_blocked');
});
test('malformed observations cannot bypass verification or crash the mission',()=>{
  const m=running(),candidate=candidateFixture(300); candidate.observations={bad:true};
  m.step({...inputFixture(300),event:'candidate',candidate});
  assert.equal(m.step({...inputFixture(400),event:'approach'}).reason,'candidate_needs_more_frames');
});
test('a swept body touching an occupied voxel face is blocked',()=>{
  const map=box(sceneFixture(),{x:1,y:-0.5,z:1.5},{x:1.5,y:0,z:2.5});
  assert.equal(sweptClearance(map,{x:1.25,y:0.5,z:2},{x:2,y:0.5,z:2},{radiusM:0.5,nowMs:0}).clear,false);
});
test('survey reorients at rest before translation after yaw scanning',()=>{
  const m=running(),i=inputFixture(300); i.telemetry.yawRad=0.4;
  const out=m.step(i); assert.equal(out.reason,'orient_before_translation'); assert.equal(out.proposal.type,'position_hold'); assert.equal(out.proposal.speedMps,0);
});
test('all integrated deterministic scenarios reach their required outcomes',()=>{
  const {runs}=runScenarios(); assert.equal(runs.length,12);
  assert.deepEqual(runs.filter(r=>!r.report.passed).map(r=>r.report.id),[]);
  assert.equal(runs.find(r=>r.report.id==='stop-and-look').mission.captureCount,6);
  assert.equal(runs.find(r=>r.report.id==='localization-loss').mission.history.at(-1).proposal,null);
});
test('delayed semantic inference is rejected even when the depth stream is current',()=>{
  const m=running(),candidate=candidateFixture(300-6620);
  m.step({...inputFixture(300),event:'candidate',candidate});
  const out=m.step({...inputFixture(400),event:'approach'});
  assert.equal(out.state,'CANDIDATE_VERIFY'); assert.equal(out.reason,'candidate_needs_more_frames'); assert.equal(out.proposal.type,'position_hold'); assert.equal(out.evidence.distinctFrames,0);
});
test('candidate projection requires the capture pose and timestamp, not a later aircraft pose',()=>{
  const m=running(),candidate=candidateFixture(300);
  for(const observation of candidate.observations) observation.capturePose.capturedMs=400;
  m.step({...inputFixture(300),event:'candidate',candidate});
  assert.equal(m.step({...inputFixture(400),event:'approach'}).reason,'candidate_needs_more_frames');
});
