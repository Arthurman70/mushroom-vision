import test from 'node:test';
import assert from 'node:assert/strict';
import {InspectionQueue,HabitatScoutMission,inspectionPlan} from '../flight/habitat-search.mjs';
import {observationRange,routeObservation,cameraAim,closePhotoPlan,validatePhotoAcknowledgment} from '../flight/range-inspection.mjs';
import {searchConfigFixture,leadFixture,emulateHabitat} from '../flight/run-habitat-simulation.mjs';
import {configFixture,sceneFixture,inputFixture,box} from '../flight/fixtures.mjs';

function measured(valueM,uncertaintyM=0){return {id:`range-${valueM}`,frameId:`range-frame-${valueM}`,kind:'fungus',confidence:.8,capturedMs:0,measuredRange:{valueM,uncertaintyM,method:'surveyed',capturedMs:0,sourceId:'test-distance',calibrationId:'fixture-survey'},synthetic:true};}
test('10m and 20m are immediate; exact 50m and 100m remain habitat follow-ups',()=>{
  const q=new InspectionQueue(searchConfigFixture());
  for(const m of [10,20,50,100])q.ingest(measured(m),0);
  for(const l of q.snapshot(0)){assert.equal(l.routingCategory,l.rangeEvidence.valueM<50?'immediate_investigation':'habitat_followup');assert.equal(l.eligible,false);assert.equal(l.confirmedMushroom,false);assert.equal(l.identifiedSpecies,null);}
  assert.equal(q.habitatLog.length,2);assert.ok(q.habitatLog.every(e=>e.originalKind==='fungus'));
});
test('range interval crossing 50m stays habitat while wholly near interval is immediate',()=>{
  for(const [value,uncertainty,expected]of [[49.9,.2,'habitat_followup'],[49,.9,'immediate_investigation'],[50,0,'habitat_followup']]){const o=measured(value,uncertainty);assert.equal(routeObservation(o,observationRange(o)).routingCategory,expected);}
});
test('invalid and unproven range values cannot manufacture immediate routing',()=>{
  for(const change of [{valueM:0},{valueM:-1},{valueM:NaN},{valueM:Infinity},{uncertaintyM:-1},{valueM:Number.MAX_VALUE,uncertaintyM:Number.MAX_VALUE},{capturedMs:1},{method:'pixel_size'},{sourceId:''},{calibrationId:''}]){const o=measured(10);Object.assign(o.measuredRange,change);const r=observationRange(o);assert.equal(r.status,'unknown');assert.equal(routeObservation(o,r).routingCategory,'unlocated_followup');}
  const o=measured(10);delete o.measuredRange;o.bbox={width:6.9};o.syntheticDistanceM=10;assert.equal(observationRange(o).status,'unknown');
});
test('calibrated synchronized metric projection supports range and conflicts remain unknown',()=>{
  const o=leadFixture(0,'fungus',{x:10,y:0,z:2});o.synthetic=false;assert.equal(observationRange(o,true).status,'unknown');o.capturePose.calibrationId='camera-1';o.metric.calibrationId='camera-1';assert.equal(observationRange(o,true).valueM,10);
  o.measuredRange=measured(40).measuredRange;assert.equal(observationRange(o,true).reason,'range_and_metric_geometry_disagree');
});
test('color contrast is always an unconfirmed habitat cue even at close measured range',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=measured(2);o.cueType='color_salience';q.ingest(o,0);const lead=q.snapshot(0)[0];assert.equal(lead.routingCategory,'habitat_followup');assert.equal(lead.confirmedMushroom,false);assert.equal(lead.kind,'fungus');assert.match(lead.label,/color contrast/);assert.equal(q.habitatLog.length,1);
});
test('located color cue can request a photo without inventing habitat signals or mushroom identity',()=>{
  const r=emulateHabitat('color-only','color-only',{kinds:['habitat'],mutate:({input})=>{input.observations[0].cueType='color_salience';input.observations[0].signals={};}});
  assert.equal(r.report.completedSimulationInspections,1);assert.equal(r.report.mapTargets[0].routingCategory,'habitat_followup');assert.equal(r.report.mapTargets[0].confirmedMushroom,false);assert.ok(r.report.photoHistory.some(p=>p.type==='request'));
});
test('metric coordinates without calibrated provenance remain screen-only outside explicit simulation',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0,'fungus');o.synthetic=false;q.ingest(o,0);assert.equal(q.snapshot(0)[0].locationStatus,'screen_only');assert.equal(q.snapshot(0)[0].eligible,false);
  o.frameId='calibrated-frame';o.metric.calibrationId='calibration-1';o.capturePose.calibrationId='calibration-1';q.ingest(o,0);assert.equal(q.snapshot(0)[0].locationStatus,'metric');assert.equal(q.snapshot(0)[0].eligible,true);
});
test('same subject range updates deduplicate and promote only fresh near evidence',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=measured(50);q.ingest(o,0);assert.equal(q.ingest(o,1).accepted,false);
  const n=measured(20);n.id=o.id;n.frameId='new-frame';n.capturedMs=100;n.measuredRange.capturedMs=100;q.ingest(n,100);assert.equal(q.leads.size,1);assert.equal(q.snapshot(100)[0].routingCategory,'immediate_investigation');assert.equal(q.habitatLog.length,1);assert.equal(q.habitatLog[0].rangeEvidence.valueM,50);
});
test('nearby low score ranks before high-score habitat without overriding safety or budget',()=>{
  const c=searchConfigFixture(),q=new InspectionQueue(c);q.ingest(leadFixture(0),0);q.ingest(leadFixture(0,'fungus'),0);assert.equal(q.snapshot(0)[0].kind,'fungus');
  const lead=q.snapshot(0)[0],blocked=box(sceneFixture(),{x:1.5,y:-3,z:-1},{x:2,y:7,z:6},'unknown');assert.equal(inspectionPlan(configFixture(),c,blocked,{x:0,y:0,z:2},lead,0).status,'blocked');assert.equal(q.reserve(lead.id,0,c.budget.maxPlannedTravelM+1),false);
});
test('bounded habitat log retains audit entries after queue expiry without false mushroom IDs',()=>{
  const c=searchConfigFixture();c.limits.maxHabitatLog=2;const q=new InspectionQueue(c);for(const r of [50,60,70])q.ingest(measured(r),0);q.snapshot(130000);assert.equal(q.habitatLog.length,2);assert.equal(q.droppedHabitatLog,1);assert.ok(q.habitatLog.every(e=>e.confirmedMushroom===false));
});
test('habitat log preserves known target coordinates and unknown range never invents coordinates',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0,'fungus',{x:60,y:0,z:2});q.ingest(o,0);q.snapshot(130000);assert.deepEqual(q.habitatLog[0].position,{x:60,y:0,z:2});assert.equal(q.habitatLog[0].coordinateFrame,'map_enu');
  q.ingest({...measured(100),capturedMs:130000,measuredRange:{...measured(100).measuredRange,capturedMs:130000}},130000);assert.equal(q.habitatLog[1].position,null);
});
test('safe native photo geometry exposes physically unreachable fill and digital crop limitation',()=>{
  const camera=configFixture().camera,c=searchConfigFixture().inspection,lead={position:{x:0,y:0,z:0}};const p=closePhotoPlan(camera,c,lead,{x:1,y:0,z:0});assert.equal(p.targetFillFraction,.7);assert.ok(Math.abs(p.requiredRangeM-.05133928571428571)<1e-9);assert.ok(p.requiredRangeM<c.minStandoffM);assert.equal(p.closeupStatus,'zoom_required_or_manual_closeup');assert.equal(p.digitalCropAddsDetail,false);assert.equal(p.framingVerified,false);assert.equal(p.targetDiameterMeasured,false);
});
test('target aim geometry controls yaw and pitch without aircraft descent',()=>{
  const aim=cameraAim({x:0,y:0,z:2},{x:0,y:3,z:.3});assert.equal(aim.yawRad,Math.PI/2);assert.ok(aim.pitchRad<0);
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0,'fungus'),0);const plan=inspectionPlan(configFixture(),searchConfigFixture(),sceneFixture(),{x:0,y:0,z:2},q.snapshot(0)[0],0);assert.equal(plan.viewpoint.z,2);assert.equal(plan.photoPlan.aim.yawRad,0);assert.equal(plan.moreApproachRequired,true);
});
test('segmented approach pauses then resumes scouting and requests one final target photo',()=>{
  const r=emulateHabitat('range-visit','range-visit');const report=r.report;
  assert.ok(report.events.some(e=>e.reason==='near_candidate_pauses_scout_at_settled_capture'));assert.ok(report.events.some(e=>e.reason==='inspection_budget_exhausted_resume_scout'));assert.ok(report.events.some(e=>e.reason==='fresh_target_segment_replanned'));
  assert.equal(report.photoHistory.filter(p=>p.type==='request').length,1);assert.equal(report.photoHistory.filter(p=>p.type==='ack').length,1);assert.equal(report.completedInspections,0);assert.equal(report.completedSimulationInspections,1);const lead=report.mapTargets[0];assert.equal(lead.visits,1);assert.equal(lead.photoStatus,'simulated_safe_photo_manual_closeup_required');assert.equal(lead.inspectionRequest.closeupPending,true);assert.equal(lead.confirmedMushroom,false);
});
test('route captures without final photo acknowledgment do not count as visits or real photographs',()=>{
  const r=emulateHabitat('no-ack','no-ack',{acknowledgePhoto:false});assert.ok(r.report.events.some(e=>e.capture));assert.equal(r.report.completedInspections,0);assert.equal(r.report.completedSimulationInspections,0);assert.equal(r.report.mapTargets[0].visits,0);assert.equal(r.report.mapTargets[0].photoStatus,'capture_ack_missing_manual_closeup_required');
});
test('omitted or fabricated external acknowledgment cannot claim real inspection',()=>{
  const r=emulateHabitat('external-ack','external-ack',{mutate:({input})=>{if(input.photoAck)delete input.photoAck.synthetic;}});assert.equal(r.report.completedInspections,0);assert.equal(r.report.completedSimulationInspections,0);assert.ok(r.report.events.some(e=>e.reason==='capture_ack_rejected'));assert.equal(r.report.mapTargets[0].visits,0);
});
test('frame identity quality native pixels and timestamps are required for capture acknowledgments',()=>{
  const c=configFixture(),s=searchConfigFixture(),request={requestId:'r1',frameId:'f1',leadId:'l1',requestedMs:100,position:{x:0,y:0,z:2},aim:{yawRad:0,pitchRad:-.61}},input=inputFixture(200),good={requestId:'r1',frameId:'f1',leadId:'l1',capturedMs:150,position:{x:0,y:0,z:2},yawRad:0,pitchRad:-.61,photoSha256:'0'.repeat(64),photoRef:'simulation://fixture',nativeUncropped:true,widthPx:1920,heightPx:1080,settled:true,sharpnessAccepted:true,synthetic:true,targetBbox:[250,300,1670,780]};
  assert.equal(validatePhotoAcknowledgment(good,request,input,c,s).framingSatisfied,true);
  for(const change of [{frameId:'other'},{leadId:'other'},{requestId:'other'},{capturedMs:0},{photoSha256:'none'},{nativeUncropped:false},{widthPx:300},{settled:false},{sharpnessAccepted:false},{yawRad:1},{pitchRad:1},{synthetic:false}])assert.equal(validatePhotoAcknowledgment({...good,...change},request,input,c,s).accepted,false);
  assert.equal(validatePhotoAcknowledgment({...good,targetBbox:[0,0,1400,400]},request,input,c,s).framingSatisfied,false);
});
test('stale or shifted target during segment cannot borrow the old route',()=>{
  const m=new HabitatScoutMission(configFixture(),searchConfigFixture(),[{x:0,y:0,z:2}]),id=m.queue.ingest(leadFixture(0,'fungus'),0).leadId;m.selected=m.queue.snapshot(0)[0];m.queue.reserve(id,0,4);m.anchor={x:0,y:0,z:2};m.stage='DETAIL';m.lastNowMs=0;m.child.lastNowMs=0;
  const input=inputFixture(100,{x:.3,y:0,z:2});input.observations=[leadFixture(100,'fungus',{x:5.4,y:0,z:.3})];const out=m.step(input);assert.equal(out.stage,'RETREAT');assert.equal(out.trigger,'target_registration_changed');assert.equal(m.queue.leads.get(id).visits,0);
});
