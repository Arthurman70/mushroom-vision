import {readFileSync} from 'node:fs';
import {mkdir,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {HabitatScoutMission,summarizeArchiveInference} from './habitat-search.mjs';
import {distance,projection} from './geometry.mjs';
import {configFixture,sceneFixture,inputFixture,refresh,box} from './fixtures.mjs';

export function searchConfigFixture(){return JSON.parse(readFileSync(new URL('../config/x500-habitat-search.json',import.meta.url),'utf8'));}
export function leadFixture(nowMs=0,kind='habitat',position={x:5,y:0,z:0.3},capturePosition={x:0,y:0,z:2}){
  return {id:`synthetic-${kind}`,kind,frameId:`synthetic-${kind}-${nowMs}`,capturedMs:nowMs,confidence:kind==='fungus'?0.18:0.8,source:'simulated_semantics',month:8,signals:kind==='habitat'?{hardwood:0.9,moss:0.7,leaf_litter:0.8,damp_ground:0.4,shaded_ground:0.8}:{},capturePose:{frame:'map_enu',position:{...capturePosition},yawRad:0,capturedMs:nowMs},metric:{frame:'map_enu',position:{...position},depthKnown:true,depthCapturedMs:nowMs,positionVarianceM2:0.0025},synthetic:true};
}
function compact(events){return events.filter((e,i)=>i===0||e.stage!==events[i-1].stage||e.reason!==events[i-1].reason||e.capture);}
export function emulateHabitat(id,title,{kinds=['fungus'],mutate=()=>{},stop=null,config=searchConfigFixture(),maxMs=100000,acknowledgePhoto=true}={}){
  config=structuredClone(config);config.budget.maxInspections=1;
  const mission=new HabitatScoutMission(configFixture(),config,[{x:0,y:0,z:2}]),map=sceneFixture(),position={x:0,y:0,z:2};
  let yawRad=0,yawRateRadS=0,speedMps=0,triggered=false;
  for(let nowMs=0;nowMs<=maxMs;nowMs+=200){
    const input=inputFixture(nowMs,position,refresh(map,nowMs));Object.assign(input.telemetry,{yawRad,yawRateRadS,speedMps});
    if(nowMs===0)input.event='preflight';if(nowMs===400)input.event='begin_simulation';
    input.observations=kinds.map((kind,i)=>leadFixture(nowMs,kind,{x:5,y:i*0.5,z:0.3},position));
    if(acknowledgePhoto&&mission.pendingPhoto){const request=mission.pendingPhoto,width=Math.max(1,Math.min(1078,request.photoPlan.expectedNativeWidthPx));input.photoAck={requestId:request.requestId,frameId:request.frameId,leadId:request.leadId,capturedMs:nowMs,position:{...position},yawRad,pitchRad:configFixture().camera.fixedMountPitchRad,photoSha256:'0'.repeat(64),photoRef:'simulation://no-real-photo',nativeUncropped:true,widthPx:1920,heightPx:1080,settled:true,sharpnessAccepted:true,targetBbox:[960-width/2,540-width/2,960+width/2,540+width/2],synthetic:true};}
    const result=mutate({input,mission,position,map,triggered});if(result===true)triggered=true;
    const out=mission.step(input);
    if(stop?.({out,mission,triggered})||['COMPLETE','MANUAL','ABORTED','MANUAL_REQUIRED'].includes(mission.stage))break;
    speedMps=0;yawRateRadS=0;
    if(out.proposal?.type==='position'){
      const d=distance(position,out.proposal.position),travel=Math.min(d,out.proposal.speedMps*0.2);if(d>0)for(const a of ['x','y','z'])position[a]+=(out.proposal.position[a]-position[a])/d*travel;speedMps=travel/0.2;
    }
    if(Number.isFinite(out.proposal?.yawRad)){const err=Math.atan2(Math.sin(out.proposal.yawRad-yawRad),Math.cos(out.proposal.yawRad-yawRad)),delta=Math.max(-0.08,Math.min(0.08,err));yawRad+=delta;yawRateRadS=delta/0.2;}
  }
  const report=mission.report();report.events=compact(report.events);return {id,title,mission,report,triggered};
}
export function runHabitatScenarios(){
  const runs=[];
  const emulate=emulateHabitat;
  const add=(run,passed,expected)=>{run.report={id:run.id,title:run.title,passed,expected,...run.report};runs.push(run);};
  let r=emulate('far-visual-detail','Low-confidence nearby visual cue receives segmented approach and acknowledged simulated photo');add(r,r.mission.completedSimulationInspections===1&&r.mission.completedInspections===0&&r.mission.stage==='COMPLETE','One synthetic photo acknowledgment; native close-up remains deferred; no real photograph');
  r=emulate('habitat-priority-detail','Nearby candidate takes priority over habitat context',{kinds:['habitat','fungus']});add(r,r.mission.completedSimulationInspections===1&&r.mission.schedule[0]?.kind==='fungus','Immediate nearby investigation precedes high-score habitat context');
  r=emulate('screen-only','Unregistered photo stays a screen-only lead',{kinds:['habitat'],mutate:({input})=>{delete input.observations[0].metric;delete input.observations[0].capturePose;},stop:({out})=>out.stage==='AWAITING_EVIDENCE'});add(r,r.mission.queue.budget.inspections===0&&r.report.screenLeads.length===1&&r.report.mapTargets.length===0,'No motion target from an unlocated photo');
  r=emulate('stale-semantic','6.62-second-old semantic results cannot start an inspection',{kinds:['fungus'],mutate:({input})=>{input.observations=[leadFixture(input.nowMs-6620,'fungus')];},stop:({out})=>out.stage==='AWAITING_EVIDENCE'});add(r,r.mission.queue.budget.inspections===0,'Fresh evidence is requested while depth remains current');
  r=emulate('habitat-does-not-clear-unknown','High habitat priority cannot clear unknown air',{kinds:['habitat'],mutate:({map})=>box(map,{x:1.5,y:-3,z:-1},{x:2,y:7,z:6},'unknown'),stop:({out})=>out.stage==='AWAITING_EVIDENCE'});add(r,r.mission.queue.budget.inspections===0&&r.mission.schedule.some(s=>s.status==='blocked'),'No route through a missing-depth wall');
  r=emulate('revalidated-retreat','New obstacle stops detail inspection and triggers a checked retreat',{mutate:({input,mission,position,map,triggered})=>{if(triggered||mission.stage==='DETAIL'&&position.x>0.2){box(map,{x:1.5,y:-3,z:-1},{x:2,y:7,z:6},'unknown');return true;}},stop:({out,triggered})=>triggered&&out.stage==='SELECT_INSPECTION'});add(r,r.report.events.some(e=>e.reason==='retreat_through_revalidated_corridor')&&r.mission.stage==='SELECT_INSPECTION','Retreat uses newly validated 3D corridor; no blind return');
  r=emulate('retreat-unavailable','Blocked retreat requires manual takeover',{mutate:({mission,position,map,triggered})=>{if(triggered||mission.stage==='DETAIL'&&position.x>0.2){box(map,{x:0,y:-0.5,z:1.5},{x:0.5,y:0.5,z:2.5},'unknown');return true;}}});add(r,r.mission.stage==='MANUAL_REQUIRED'&&r.report.events.at(-1).reason==='retreat_corridor_unavailable','No automatic motion when return volume is unknown');
  r=emulate('localization-loss','Localization loss during habitat inspection suppresses position proposals',{kinds:['habitat'],mutate:({input,mission,position})=>{if(mission.stage==='DETAIL'&&position.x>0.2)input.telemetry.localization.valid=false;}});add(r,r.mission.stage==='MANUAL_REQUIRED'&&r.report.events.at(-1).proposal===null,'Manual takeover; no position hold with invalid localization');
  r=emulate('manual-takeover','Pilot takeover preempts scouting and the inspection queue',{mutate:({input,mission})=>{if(mission.stage==='DETAIL')input.event='manual';}});add(r,r.mission.stage==='MANUAL','Terminal manual control handoff');
  return runs;
}
export async function writeHabitatReport(){
  const runs=runHabitatScenarios(),sample=runs.find(r=>r.id==='habitat-priority-detail').report,report={schemaVersion:1,simulationOnly:true,actuationEnabled:false,generatedAt:new Date().toISOString(),target:'Black-trumpet-like fungi; no species confirmation',summary:{scenarios:runs.length,passed:runs.filter(r=>r.report.passed).length,failed:runs.filter(r=>!r.report.passed).length,physicalFlights:0,liveHabitatClassifications:0},mapTargets:sample.mapTargets,screenLeads:runs.find(r=>r.id==='screen-only').report.screenLeads,schedule:sample.schedule,scenarios:runs.map(r=>r.report),notes:['All habitat labels and visual detections in these flight scenarios are synthetic inputs.','Priorities are heuristic inspection preferences, not probability of mushrooms, species identity or safe ground.','Generic scene segmentation, if provided separately, does not establish hardwood species, moisture or 3D freespace.','No genuine distant-positive black-trumpet detector result or flight working range is established.','The original 2-second action-evidence gate is preserved; measured desktop semantic latency currently exceeds it.']};
  report.targetPixelEstimates={calibrationStatus:configFixture().camera.calibrationStatus,detectorPerformanceMeasured:false,validatedDetectorMinPixels:null,rows:[1,2,4,8].map(rangeM=>({rangeM,smallCapDiameterM:0.01,smallCapPixels:projection(configFixture().camera,rangeM,0.01),exampleCapDiameterM:0.03,exampleCapPixels:projection(configFixture().camera,rangeM,0.03)}))};
  try{const archive=JSON.parse(readFileSync(new URL('../experiments/results/habitat-v2/report.json',import.meta.url),'utf8'));report.archiveInference=summarizeArchiveInference(archive,searchConfigFixture());}catch(error){report.archiveInference={status:error.code==='ENOENT'?'not_available':'unreadable',measuredInference:false,mapTargetCount:0};}
  const directory=new URL('./reports/',import.meta.url);await mkdir(directory,{recursive:true});await writeFile(new URL('habitat-latest.json',directory),JSON.stringify(report,null,2)+'\n');
  await writeFile(new URL('HABITAT-LATEST.md',directory),['# Habitat and distant-scout simulation','',`**${report.summary.passed}/${report.summary.scenarios} scenarios passed.** Synthetic scene evidence; no aircraft or live habitat classifier.`, '', '| Scenario | Result | Outcome |','|---|---|---|',...report.scenarios.map(s=>`| ${s.title} | ${s.passed?'PASS':'FAIL'} | ${s.expected} |`),'','## Separate archive inference','',report.archiveInference.status==='available'?`${report.archiveInference.imageCount} actual archive photos analyzed; ${report.archiveInference.screenLeadCount} unlocated inspection leads, **zero metric map targets**. Mean offline inference: ${report.archiveInference.meanInferenceSeconds?.toFixed(3)} seconds/image. Ecological accuracy is unmeasured. Model use: ${report.archiveInference.useRestriction}.`:'No readable archive inference report was available.','','## Limits','',...report.notes.map(n=>`- ${n}`),''].join('\n'));
  return report;
}
if(process.argv[1]&&path.resolve(process.argv[1])===fileURLToPath(import.meta.url)){const report=await writeHabitatReport();console.log(JSON.stringify({simulationOnly:true,...report.summary,report:'flight/reports/habitat-latest.json'},null,2));if(report.summary.failed)process.exitCode=1;}
