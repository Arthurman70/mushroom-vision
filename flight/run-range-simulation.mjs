import {writeFile,mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {InspectionQueue,inspectionPlan} from './habitat-search.mjs';
import {closePhotoPlan} from './range-inspection.mjs';
import {searchConfigFixture,leadFixture,emulateHabitat} from './run-habitat-simulation.mjs';
import {configFixture,sceneFixture} from './fixtures.mjs';

export function runRangeScenarios(){
  const config=searchConfigFixture(),flight=configFixture(),checks=[];
  const add=(title,passed,result)=>checks.push({title,passed,result});
  const queue=new InspectionQueue(config);
  for(const rangeM of [10,20,50,100])queue.ingest({id:`measured-${rangeM}`,frameId:`fixture-${rangeM}`,kind:'fungus',cueType:'detector',confidence:.2,capturedMs:0,synthetic:true,measuredRange:{valueM:rangeM,uncertaintyM:0,method:'surveyed',sourceId:'simulation-range-fixture',calibrationId:'simulation-fixture',capturedMs:0}},0);
  const routing=queue.snapshot(0).map(l=>({rangeM:l.rangeEvidence.valueM,routingCategory:l.routingCategory,eligible:l.eligible,confirmedMushroom:l.confirmedMushroom}));
  add('10 m and 20 m prioritize confirmation; 50 m and 100 m enter habitat log',routing.every(r=>r.routingCategory===(r.rangeM<50?'immediate_investigation':'habitat_followup'))&&queue.habitatLog.length===2,'Measured-range simulation fixtures only; no physical-distance detection accuracy is implied.');
  const uncertain=new InspectionQueue(config);uncertain.ingest({id:'uncertain',frameId:'uncertain',kind:'fungus',confidence:.2,capturedMs:0,synthetic:true,measuredRange:{valueM:49.9,uncertaintyM:.2,method:'surveyed',sourceId:'fixture',calibrationId:'fixture',capturedMs:0}},0);
  add('Uncertainty crossing 50 m remains habitat',uncertain.snapshot(0)[0].routingCategory==='habitat_followup','49.9 ±0.2 m is retained as an unconfirmed habitat follow-up.');
  const color=new InspectionQueue(config);color.ingest({id:'color',frameId:'color',kind:'fungus',cueType:'color_salience',confidence:.99,capturedMs:null},0);
  add('Unknown-range color cue cannot become a mushroom marker',color.snapshot(0)[0].routingCategory==='habitat_followup'&&!color.snapshot(0)[0].eligible,'Color contrast creates an unlocated habitat cue, never a species guess or flight target.');
  const visit=emulateHabitat('segmented-visit','Segmented close inspection'),missing=emulateHabitat('missing-ack','Missing photo acknowledgment',{acknowledgePhoto:false});
  const external=emulateHabitat('fabricated-external','Fabricated external photo acknowledgment',{mutate:({input})=>{if(input.photoAck)input.photoAck.synthetic=false;}});
  add('Nearby target pauses then resumes the scout route',visit.report.events.some(e=>e.reason==='near_candidate_pauses_scout_at_settled_capture')&&visit.report.events.some(e=>e.reason==='inspection_budget_exhausted_resume_scout'),'Inspection begins at a settled scout capture; remaining scout work resumes with current clearance checks.');
  add('Approach segments replan fresh geometry before next movement',visit.report.events.some(e=>e.reason==='fresh_target_segment_replanned'),'2 m maximum approach segments preserve altitude, fresh target evidence, map clearance and travel/time budgets.');
  add('One final target photo requires explicit synthetic acknowledgment',visit.report.photoHistory.filter(p=>p.type==='ack').length===1&&visit.report.completedInspections===0,'Intermediate route captures are proposals only; real photographs captured: zero.');
  add('No photo acknowledgment means no completed visit',missing.report.mapTargets[0].visits===0&&missing.report.completedSimulationInspections===0,'The close-up request remains pending/manual after timeout.');
  add('Fabricated external acknowledgment cannot claim a physical photo',external.report.completedInspections===0&&external.report.completedSimulationInspections===0,'Simulation accepts only explicitly synthetic capture acknowledgments.');
  const framing=closePhotoPlan(flight.camera,config.inspection,{position:{x:0,y:0,z:0}},{x:1,y:0,z:0});
  add('Close-photo framing geometry reports a zoom/manual closeup requirement',framing.requiredRangeM<config.inspection.minStandoffM&&framing.closeupStatus==='zoom_required_or_manual_closeup','Illustrative 5 cm cap needs approximately 5.1 cm range to fill 70% native width; the 1 m minimum standoff is preserved.');
  const q=new InspectionQueue(config);q.ingest(leadFixture(0,'fungus'),0);const stale=inspectionPlan(flight,config,sceneFixture(3000),{x:0,y:0,z:2},q.snapshot(0)[0],3000);
  add('Stale evidence cannot continue an approach',stale.reason==='fresh_metric_evidence_required','Range thresholds and priority never bypass synchronized fresh depth/pose and corridor gates.');
  return {schema:'mycoscout.range-inspection-report.v1',generatedAt:new Date().toISOString(),simulationOnly:true,actuationEnabled:false,physicalFlights:0,physicalPhotos:0,summary:{checks:checks.length,passed:checks.filter(c=>c.passed).length,failed:checks.filter(c=>!c.passed).length},checks,routing,framing,habitatLog:queue.habitatLog,photoRequests:visit.report.photoHistory,scenarios:[visit.report,missing.report,external.report],notes:['All ranges in this report are measured-distance simulation fixtures, not estimates from synthetic image sizes.','Unknown range stays unknown; native color is a search cue, never confirmation.','A 70% native-frame-width close-up remains pending when the aircraft cannot safely obtain it; digital crop is not additional captured detail.','This is behavior simulation without camera transport, gimbal control, MAVLink or physical flight.']};
}
export async function writeRangeReport(){const report=runRangeScenarios(),directory=new URL('./reports/',import.meta.url);await mkdir(directory,{recursive:true});await writeFile(new URL('range-latest.json',directory),JSON.stringify(report,null,2)+'\n');await writeFile(new URL('RANGE-LATEST.md',directory),['# Range-aware inspection simulation','',`${report.summary.passed}/${report.summary.checks} checks passed. Physical flights: 0. Physical photographs: 0.`,'','| Check | Result | Evidence |','|---|---|---|',...report.checks.map(c=>`| ${c.title} | ${c.passed?'PASS':'FAIL'} | ${c.result} |`),'',...report.notes.map(n=>`- ${n}`),''].join('\n'));return report;}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){const report=await writeRangeReport();console.log(JSON.stringify(report.summary));if(report.summary.failed)process.exitCode=1;}
