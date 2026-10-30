import test from 'node:test';
import assert from 'node:assert/strict';
import {InspectionQueue,HabitatScoutMission,inspectionPlan,summarizeArchiveInference} from '../flight/habitat-search.mjs';
import {searchConfigFixture,leadFixture,runHabitatScenarios} from '../flight/run-habitat-simulation.mjs';
import {configFixture,sceneFixture,inputFixture,box} from '../flight/fixtures.mjs';

test('unregistered photos stay screen-only and have no map position',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0);delete o.metric;
  assert.equal(q.ingest(o,0).locationStatus,'screen_only');const lead=q.snapshot(0)[0];assert.equal(lead.eligible,false);assert.equal(lead.position,null);
});
test('unknown photo capture time stays null rather than being replaced with the inference time',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0);o.capturedMs=null;delete o.capturePose;delete o.metric;
  assert.equal(q.ingest(o,10000).accepted,true);const lead=q.snapshot(10000)[0];assert.equal(lead.originalCapturedMs,null);assert.equal(lead.captureTimeKnown,false);assert.equal(lead.receivedMs,10000);assert.equal(lead.eligible,false);assert.equal(lead.evidence[0].capturedMs,null);
});
test('metric leads need exact capture/depth timestamps and a capture pose',()=>{
  for(const mutate of [o=>delete o.capturePose,o=>o.capturePose.capturedMs=1,o=>o.metric.depthCapturedMs=1,o=>o.metric.depthKnown=false,o=>o.metric.positionVarianceM2=1,o=>o.capturePose.frame='gps']){
    const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0);mutate(o);assert.equal(q.ingest(o,0).locationStatus,'screen_only');
  }
});
test('weak distant mushroom visual evidence can queue an inspection without claiming species',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0,'fungus'),0);const lead=q.snapshot(0)[0];
  assert.equal(lead.eligible,true);assert.equal(lead.confidence,0.18);assert.equal(lead.identifiedSpecies,null);assert.equal(lead.edible,null);assert.equal(lead.groundSafety,'unknown');assert.equal(lead.priorityIsProbability,false);
});
test('generic segmentation cannot manufacture oak, moss or moisture evidence',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0);o.signals={forest_context:0.8,vegetation_ground:0.4,exposed_ground:0.2};o.semanticFractions={tree:0.5,plant:0.2,water:0.1,path:0.1};q.ingest(o,0);const lead=q.snapshot(0)[0];
  assert.equal(lead.signals.hardwood,undefined);assert.equal(lead.signals.moss,undefined);assert.equal(lead.signals.damp_ground,undefined);assert.equal(lead.groundSafety,'unknown');assert.ok(lead.priority<0.2);
});
test('season alone and zero-confidence evidence cannot schedule a viewpoint',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0);o.signals={};q.ingest(o,0);assert.equal(q.snapshot(0)[0].eligible,false);
  const n=new InspectionQueue(searchConfigFixture()),p=leadFixture(0,'fungus');p.confidence=0;n.ingest(p,0);assert.equal(n.snapshot(0)[0].eligible,false);
});
test('repeated frames neither inflate priority nor refresh stale evidence',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0);q.ingest(o,0);const initial=q.snapshot(0)[0].priority;
  assert.equal(q.ingest(o,1800).reason,'duplicate_frame_does_not_refresh_evidence');assert.equal(q.snapshot(2100)[0].eligible,false);assert.equal(q.snapshot(2100)[0].priority,initial);
});
test('nearby same-kind metric observations merge while habitat and fungi stay distinct',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0),0);const o=leadFixture(100,'habitat',{x:5.2,y:0,z:0.3});o.id='new-patch-id';q.ingest(o,100);q.ingest(leadFixture(100,'fungus'),100);
  assert.equal(q.leads.size,2);assert.equal(q.snapshot(100).find(l=>l.kind==='habitat').observationIds.length,2);
});
test('fresh nearby habitat increases visual inspection priority without changing identification',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0,'fungus'),0);const before=q.snapshot(0)[0].priority;q.ingest(leadFixture(0),0);const after=q.snapshot(0).find(l=>l.kind==='fungus');assert.ok(after.priority>before);assert.equal(after.identifiedSpecies,null);
});
test('old and future observations are rejected rather than becoming active leads',()=>{
  const q=new InspectionQueue(searchConfigFixture());assert.equal(q.ingest(leadFixture(-130000),0).accepted,false);assert.equal(q.ingest(leadFixture(100),0).accepted,false);
});
test('6.62-second inference may remain historical but is not action-eligible',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0,'fungus'),6620);const lead=q.snapshot(6620)[0];assert.equal(lead.locationStatus,'metric');assert.equal(lead.eligible,false);assert.ok(lead.reasons.some(r=>r.includes('Fresh')));
});
test('queue expiry makes room for new observations at capacity',()=>{
  const c=searchConfigFixture();c.queue.maxLeads=1;const q=new InspectionQueue(c);q.ingest(leadFixture(0),0);const o=leadFixture(130000,'fungus');assert.equal(q.ingest(o,130000).accepted,true);assert.equal(q.leads.size,1);assert.equal(q.snapshot(130000)[0].kind,'fungus');
});
test('reservation prevents duplicate dispatch and respects count and distance budgets',()=>{
  const c=searchConfigFixture();c.budget.maxInspections=1;c.budget.maxPlannedTravelM=4;const q=new InspectionQueue(c);const id=q.ingest(leadFixture(0),0).leadId;
  assert.equal(q.reserve(id,0,5),false);assert.equal(q.reserve(id,0,4),true);assert.equal(q.reserve(id,0,0),false);assert.equal(q.withinBudget(0),false);
});
test('revisit needs fresh evidence, cooldown and remaining visit allowance',()=>{
  const q=new InspectionQueue(searchConfigFixture()),id=q.ingest(leadFixture(0),0).leadId;q.reserve(id,0,1);q.finish(id,500,'no_species_confirmation');q.ingest(leadFixture(1000),1000);assert.equal(q.snapshot(1000)[0].eligible,false);
  q.ingest(leadFixture(31000),31000);assert.equal(q.snapshot(31000)[0].eligible,true);q.reserve(id,31000,1);q.finish(id,31500,'no_species_confirmation');q.ingest(leadFixture(62000),62000);assert.equal(q.snapshot(62000)[0].eligible,false);
});
test('outside-season observations are not excluded from inspection',()=>{
  const q=new InspectionQueue(searchConfigFixture()),o=leadFixture(0);o.month=2;q.ingest(o,0);assert.equal(q.snapshot(0)[0].eligible,true);
});
test('same-height viewpoint gets closer without targeting the ground mushroom position',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0,'fungus'),0);const lead=q.snapshot(0)[0],from={x:0,y:0,z:2};const plan=inspectionPlan(configFixture(),searchConfigFixture(),sceneFixture(),from,lead,0);
  assert.equal(plan.status,'ready');assert.equal(plan.viewpoint.z,2);assert.ok(plan.proposedRangeM<plan.initialRangeM);assert.ok(plan.routeDistanceM<=2.001);assert.ok(plan.reservedTravelM>=plan.routeDistanceM*2);assert.notDeepEqual(plan.viewpoint,lead.position);
});
test('ideal habitat never marks unknown voxels free',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0),0);const map=box(sceneFixture(),{x:1.5,y:-3,z:-1},{x:2,y:7,z:6},'unknown'),before=JSON.stringify(map);
  const plan=inspectionPlan(configFixture(),searchConfigFixture(),map,{x:0,y:0,z:2},q.snapshot(0)[0],0);assert.equal(plan.status,'blocked');assert.equal(JSON.stringify(map),before);
});
test('manual and abort events preempt queued work and remain terminal',()=>{
  for(const event of ['manual','abort']){const mission=new HabitatScoutMission(configFixture(),searchConfigFixture(),[{x:0,y:0,z:2}]);const out=mission.step({...inputFixture(0),event,observations:[leadFixture(0)]});assert.equal(out.stage,event==='manual'?'MANUAL':'ABORTED');assert.equal(out.proposal,null);assert.equal(mission.step({...inputFixture(100),event:'begin_simulation'}).stage,out.stage);}
});
test('all distant-scout, habitat, stale evidence and retreat scenarios meet their outcomes',()=>{
  const runs=runHabitatScenarios();assert.equal(runs.length,9);assert.deepEqual(runs.filter(r=>!r.report.passed).map(r=>r.id),[]);
  const visual=runs.find(r=>r.id==='far-visual-detail').report;assert.ok(visual.events.some(e=>e.reason==='distant_scout_capture'));assert.ok(visual.events.some(e=>e.reason==='closer_inspection_scheduled'));assert.ok(visual.events.some(e=>e.reason==='target_photo_acknowledged'));assert.equal(visual.completedInspections,0);assert.equal(visual.completedSimulationInspections,1);
});
test('archive inference summary strips photo identifiers and cannot create metric targets',()=>{
  const observation=leadFixture(0);observation.id='private-photo-123';observation.frameId='private-frame-456';observation.context={semanticFractions:{trees:0.5,vegetation:0.4,ground:0.1},captureTimeKnown:false};observation.signals={forest_context:0.8,hardwood:1,damp_ground:1};
  const summary=summarizeArchiveInference({measuredInference:true,images:[{inferenceSeconds:0.8}],observations:[observation]},searchConfigFixture());
  assert.equal(summary.measuredInference,true);assert.equal(summary.screenLeadCount,1);assert.equal(summary.mapTargetCount,0);assert.equal(summary.screenLeads[0].captureTimeKnown,false);assert.equal(summary.screenLeads[0].signals.hardwood,undefined);assert.equal(summary.screenLeads[0].eligible,false);assert.equal(JSON.stringify(summary).includes('private-photo'),false);assert.equal(JSON.stringify(summary).includes('private-frame'),false);assert.equal(summary.ecologicalAccuracyMeasured,false);
});
test('replay cache and rejected history are bounded without evicting fresh duplicate protection',()=>{
  const c=searchConfigFixture();c.limits.maxSeenFrames=4;c.limits.maxRejectedHistory=3;const q=new InspectionQueue(c);
  for(let i=0;i<4;i++)assert.equal(q.ingest(leadFixture(i),i).accepted,true);
  for(let i=4;i<20;i++)assert.equal(q.ingest(leadFixture(i),i).reason,'evidence_cache_capacity');
  assert.equal(q.seenFrames.size,4);assert.equal(q.rejected.length,3);assert.equal(q.rejectedCount,16);assert.equal(q.ingest(leadFixture(0),20).reason,'duplicate_frame_does_not_refresh_evidence');
  assert.equal(q.ingest(leadFixture(130000),130000).accepted,true);assert.equal(q.seenFrames.size,1);
});
test('merged observation IDs, source strings and retained metadata have bounded shapes',()=>{
  const c=searchConfigFixture();c.limits.maxObservationIdsPerLead=2;c.limits.maxStringLength=32;const q=new InspectionQueue(c);
  for(let i=0;i<8;i++){const o=leadFixture(i);o.id=`source-${i}`;o.source='x'.repeat(100);o.bbox={x:1,y:2,width:3,height:4,unexpected:'x'.repeat(10000)};o.metric.position.extra='not retained';assert.equal(q.ingest(o,i).accepted,true);}
  const lead=q.snapshot(8)[0];assert.equal(q.leads.size,1);assert.equal(lead.observationIds.length,2);assert.equal(lead.source.length,32);assert.equal(lead.bbox.unexpected,undefined);assert.equal(lead.position.extra,undefined);assert.equal(lead.evidence.length,8);
});
test('a stale cached eligibility flag cannot authorize route planning',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0),0);const old=q.snapshot(0)[0];assert.equal(old.eligible,true);
  const plan=inspectionPlan(configFixture(),searchConfigFixture(),sceneFixture(3000),{x:0,y:0,z:2},old,3000);assert.equal(plan.reason,'fresh_metric_evidence_required');assert.deepEqual(plan.points,[]);
});
test('a modified target position cannot borrow another metric observation provenance',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0),0);const lead=q.snapshot(0)[0];lead.position.x=8;
  assert.equal(inspectionPlan(configFixture(),searchConfigFixture(),sceneFixture(),{x:0,y:0,z:2},lead,0).reason,'fresh_metric_evidence_required');
});
test('fresh metric data can replace archive receipt ordering, but stale metric data stays ineligible',()=>{
  for(const [captured,now,eligible]of [[950,1100,true],[0,3000,false]]){const q=new InspectionQueue(searchConfigFixture()),archive=leadFixture(0);archive.capturedMs=null;archive.frameId='archive';delete archive.capturePose;delete archive.metric;q.ingest(archive,1000);q.ingest(leadFixture(captured),now);const lead=q.snapshot(now)[0];assert.equal(q.leads.size,1);assert.equal(lead.captureTimeKnown,true);assert.equal(lead.eligible,eligible);assert.equal(lead.originalCapturedMs,captured);}
});
test('a newer screen-only observation cannot retain old metric coordinates',()=>{
  const q=new InspectionQueue(searchConfigFixture());q.ingest(leadFixture(0),0);const o=leadFixture(100);delete o.metric;q.ingest(o,100);const lead=q.snapshot(100)[0];assert.equal(lead.locationStatus,'screen_only');assert.equal(lead.position,null);assert.equal(lead.eligible,false);
});
test('waiting for new leads still checks the full current aircraft volume',()=>{
  const m=new HabitatScoutMission(configFixture(),searchConfigFixture(),[{x:0,y:0,z:2}]);m.stage='AWAITING_EVIDENCE';m.lastNowMs=0;m.child.lastNowMs=0;
  const map=box(sceneFixture(100),{x:0,y:-0.5,z:1.5},{x:0.5,y:0.5,z:2.5},'unknown'),out=m.step(inputFixture(100,{x:0,y:0,z:2},map));
  assert.equal(out.stage,'MANUAL_REQUIRED');assert.equal(out.reason,'current_aircraft_volume_unavailable');assert.equal(out.proposal,null);
});
test('mission, child and schedule logs and per-step observation ingestion are bounded',()=>{
  const c=searchConfigFixture();c.limits.maxMissionHistory=3;c.limits.maxScheduleHistory=2;c.limits.maxInputObservations=2;
  const m=new HabitatScoutMission(configFixture(),c,[{x:0,y:0,z:2}]);
  for(let i=0;i<6;i++)m.step(inputFixture(i*100));
  for(let i=0;i<5;i++)m.recordSchedule({leadId:`lead-${i}`,status:'deferred',reason:'exploration_budget',nowMs:0});
  m.step({...inputFixture(600),observations:Array.from({length:5},(_,i)=>({...leadFixture(600),id:`obs-${i}`,frameId:`frame-${i}`}))});
  assert.equal(m.history.length,3);assert.equal(m.child.history.length,3);assert.equal(m.schedule.length,2);assert.equal(m.queue.seenFrames.size,2);assert.equal(m.report().retention.droppedInputObservations,3);assert.ok(m.report().retention.droppedHistoryEvents>0);assert.equal(m.report().retention.droppedScheduleEvents,3);
});
test('repeated budget deferrals do not flood the schedule with identical entries',()=>{
  const m=new HabitatScoutMission(configFixture(),searchConfigFixture(),[{x:0,y:0,z:2}]);for(let nowMs=0;nowMs<5000;nowMs+=200)m.recordSchedule({leadId:'one',status:'deferred',reason:'exploration_budget',nowMs});assert.equal(m.schedule.length,1);
});
test('expired active target triggers checked retreat rather than continued forward motion',()=>{
  const m=new HabitatScoutMission(configFixture(),searchConfigFixture(),[{x:0,y:0,z:2}]);const id=m.queue.ingest(leadFixture(0,'fungus'),0).leadId;m.selected=m.queue.snapshot(0)[0];m.queue.reserve(id,0,4);m.anchor={x:0,y:0,z:2};m.stage='DETAIL';m.lastNowMs=2400;m.child.lastNowMs=2400;
  const out=m.step(inputFixture(2500,{x:0.3,y:0,z:2}));assert.equal(out.stage,'RETREAT');assert.equal(out.trigger,'inspection_evidence_expired');assert.equal(out.proposal.type,'position_hold');assert.ok(out.route.at(-1).x===0);
});
test('a blocked route check defers the lead without pretending an inspection visit occurred',()=>{
  const q=new InspectionQueue(searchConfigFixture()),id=q.ingest(leadFixture(0),0).leadId;q.deferUnsafe(id,0,'unknown_voxel');q.ingest(leadFixture(60000),60000);q.deferUnsafe(id,60000,'unknown_voxel');q.ingest(leadFixture(120000),120000);
  const lead=q.snapshot(120000)[0];assert.equal(lead.visits,0);assert.equal(lead.pathAttempts,2);assert.equal(lead.eligible,true);assert.equal(q.budget.inspections,0);assert.equal(q.reserve(id,120000,2),true);
});
