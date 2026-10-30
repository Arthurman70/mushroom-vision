import { mkdir,writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { SurveyMission } from './mission.mjs';
import { geometryReport,planCorridor3D,distance,blurLimitedSpeed } from './geometry.mjs';
import { configFixture,sceneFixture,inputFixture,candidateFixture,box,refresh } from './fixtures.mjs';

function compactEvents(history) {
  return history.filter((row,i)=>i===0 || row.state!==history[i-1].state || row.reason!==history[i-1].reason || row.captureRequested)
    .map(({nowMs,state,reason,proposal,capture,evidence,clearance})=>({nowMs,state,reason,...(proposal?{proposal}:{}),...(capture?{capture}:{}),...(evidence?{evidence}:{}),...(clearance&&!clearance.clear?{clearance}:{})}));
}
function result(id,title,mission,expected,details={}) {
  return {id,title,simulationOnly:true,passed:mission.state===expected,expectedFinalState:expected,finalState:mission.state,durationMs:mission.history.at(-1)?.nowMs??0,captures:mission.captureCount,events:compactEvents(mission.history),...details};
}
function readyMission(waypoints=[{x:3,y:0,z:2}],map=sceneFixture(),config=configFixture()) {
  const m=new SurveyMission(config,waypoints);
  m.step({...inputFixture(0,{x:0,y:0,z:2},refresh(map,0)),event:'preflight'});
  m.step(inputFixture(100,{x:0,y:0,z:2},refresh(map,100)));
  return m;
}
function emulateSurvey(id,title,map,waypoints,config=configFixture()) {
  const mission=readyMission(waypoints,map,config),position={x:0,y:0,z:2};
  let yawRad=0,speedMps=0,yawRateRadS=0;
  for(let nowMs=200;nowMs<180000;nowMs+=200) {
    const input=inputFixture(nowMs,position,refresh(map,nowMs)); Object.assign(input.telemetry,{yawRad,speedMps,yawRateRadS});
    if(nowMs===200) input.event='begin_simulation';
    const output=mission.step(input);
    if(['COMPLETE','HOLD','ABORTED','MANUAL'].includes(output.state)) break;
    const proposal=output.proposal; speedMps=0; yawRateRadS=0;
    if(proposal?.type==='position') {
      const d=distance(position,proposal.position),travel=Math.min(d,proposal.speedMps*0.2);
      if(d>0) for(const axis of ['x','y','z']) position[axis]+=(proposal.position[axis]-position[axis])/d*travel;
      speedMps=travel/0.2;
    }
    if(Number.isFinite(proposal?.yawRad)) {
      const error=Math.atan2(Math.sin(proposal.yawRad-yawRad),Math.cos(proposal.yawRad-yawRad));
      const delta=Math.max(-0.4*0.2,Math.min(0.4*0.2,error)); yawRad+=delta; yawRateRadS=delta/0.2;
    }
  }
  return {mission,report:result(id,title,mission,'COMPLETE',{route:waypoints,emulator:'Ideal kinematics with bounded translation and yaw; no aerodynamic, wind, motor, estimator or PX4 simulation.'})};
}

export function runScenarios() {
  const config=configFixture(),runs=[];
  runs.push(emulateSurvey('stop-and-look','Two stations with three settled image captures each',sceneFixture(),[{x:0,y:0,z:2},{x:3,y:0,z:2}]));
  const trunkMap=box(sceneFixture(),{x:1.5,y:-0.5,z:0},{x:2,y:0.5,z:5});
  const plannedSpeed=Math.min(config.survey.maxSpeedMps,blurLimitedSpeed(config.camera,1));
  const stoppingMargin=plannedSpeed*config.survey.reactionTimeS+plannedSpeed**2/(2*config.survey.maxDecelerationMps2);
  const planningRadiusM=config.vehicle.radiusM+config.vehicle.clearanceM+config.vehicle.maxPoseSigmaM+stoppingMargin+config.survey.arrivalToleranceM;
  const planned=planCorridor3D(trunkMap,{x:0,y:0,z:2},{x:4,y:0,z:2},{radiusM:planningRadiusM,nowMs:0});
  const detour=emulateSurvey('trunk-detour','Observed 3D corridor detours around a trunk',trunkMap,planned.points);
  detour.report.planner={...planned,planningRadiusM}; runs.push(detour);
  for(const [id,title,mutate,expectedReason] of [
    ['overhead-branch','Branch intersects rotor envelope despite clear endpoints',i=>box(i.map,{x:1,y:-0.5,z:2.5},{x:1.5,y:0.5,z:3}),'occupied_voxel'],
    ['unknown-space','Missing depth behind foliage blocks the corridor',i=>box(i.map,{x:1,y:-0.5,z:1.5},{x:1.5,y:0.5,z:2.5},'unknown'),'unknown_voxel'],
    ['depth-dropout','Depth stream becomes stale',i=>{i.telemetry.depth.updatedMs=-1000;},'depth_unavailable'],
    ['localization-loss','Loss of localization requests takeover without a position proposal',i=>{i.telemetry.localization.valid=false;},'localization_unreliable'],
    ['stale-free-space','Fresh map header cannot hide expired voxel evidence',i=>{for(const cell of Object.values(i.map.voxels)) cell.observedMs=-2000;},'stale_voxel']
  ]) {
    const mission=readyMission(); mission.step({...inputFixture(200),event:'begin_simulation'});
    const input=inputFixture(300); mutate(input); const output=mission.step(input);
    const report=result(id,title,mission,'HOLD',{expectedReason}); report.passed&&=output.reason===expectedReason;
    if(id==='localization-loss') report.passed&&=output.proposal===null;
    runs.push({mission,report});
  }
  {
    const mission=readyMission(); mission.step({...inputFixture(200),event:'begin_simulation'});
    mission.step({...inputFixture(300),event:'candidate',candidate:candidateFixture(300)});
    const position={x:0,y:0,z:2}; let speedMps=0,approachStarted=false,reached=false;
    for(let nowMs=500;nowMs<15000;nowMs+=200) {
      const input={...inputFixture(nowMs,position),candidate:candidateFixture(nowMs)}; input.telemetry.speedMps=speedMps;
      if(mission.captureCount>0 && !approachStarted) { input.event='approach'; approachStarted=true; }
      const output=mission.step(input); speedMps=0;
      if(output.proposal?.type==='position') {
        const d=distance(position,output.proposal.position),travel=Math.min(d,output.proposal.speedMps*0.2);
        if(d>0) for(const axis of ['x','y','z']) position[axis]+=(output.proposal.position[axis]-position[axis])/d*travel;
        speedMps=travel/0.2;
      }
      if(output.reason==='verification_viewpoint_reached') { reached=true; break; }
    }
    const report=result('candidate-verification','Hover, gather distinct observations, approach through a known same-height corridor',mission,'CANDIDATE_VERIFY',{candidate: candidateFixture(0),viewpointReached:reached}); report.passed&&=reached;
    runs.push({mission,report});
  }
  {
    const mission=readyMission(); mission.step({...inputFixture(200),event:'begin_simulation'});
    const candidate=candidateFixture(300-6620);
    mission.step({...inputFixture(300),event:'candidate',candidate});
    const output=mission.step({...inputFixture(400),event:'approach'});
    const report=result('delayed-semantic-result','Old semantic detections cannot authorize approach while live depth stays healthy',mission,'CANDIDATE_VERIFY',{injectedSemanticLatencyMs:6620,observationAgeLimitMs:config.candidate.maxObservationAgeMs,depthCurrent:true});
    report.passed&&=output.reason==='candidate_needs_more_frames'&&output.proposal?.type==='position_hold'; runs.push({mission,report});
  }
  {
    const mission=readyMission(); mission.step({...inputFixture(200),event:'begin_simulation'});
    mission.step({...inputFixture(300),event:'candidate',candidate:candidateFixture(300,{x:1,y:0,z:0.5})});
    const output=mission.step({...inputFixture(400),event:'approach'}),report=result('no-blind-descent','Ground mushroom never causes a descent command',mission,'CANDIDATE_VERIFY'); report.passed&&=output.reason==='approach_altitude_change_blocked'; runs.push({mission,report});
  }
  for(const event of ['manual','abort']) {
    const mission=readyMission(); mission.step({...inputFixture(200),event:'begin_simulation'}); mission.step({...inputFixture(300),event}); mission.step({...inputFixture(400),event:'resume'});
    runs.push({mission,report:result(event,event==='manual'?'Manual takeover remains latched':'Abort does not invent a return path',mission,event==='manual'?'MANUAL':'ABORTED')});
  }
  const dimConfig=structuredClone(config); dimConfig.camera.exposureS=0.04;
  return {runs,cameraGeometry:geometryReport(config.camera),exposureComparison:[{exposureMs:config.camera.exposureS*1000,blurLimitedSpeedMps:blurLimitedSpeed(config.camera,1)},{exposureMs:40,blurLimitedSpeedMps:blurLimitedSpeed(dimConfig.camera,1)}]};
}

export async function writeReports(outputDir=new URL('./reports/',import.meta.url)) {
  const {runs,cameraGeometry,exposureComparison}=runScenarios(),directory=outputDir instanceof URL?fileURLToPath(outputDir):path.resolve(outputDir);
  await mkdir(directory,{recursive:true});
  const summary={scenarios:runs.length,passed:runs.filter(r=>r.report.passed).length,failed:runs.filter(r=>!r.report.passed).length,captures:runs.reduce((n,r)=>n+r.mission.captureCount,0),physicalFlights:0,realDetectorInvocations:0,transport:'none'};
  const report={schemaVersion:1,simulationOnly:true,actuationEnabled:false,generatedAt:new Date().toISOString(),hardwareTarget:configFixture().hardwareTarget,summary,cameraGeometry,exposureComparison,scenarios:runs.map(r=>r.report),limitations:['Synthetic free-space maps and synthetic candidate evidence, not observed woodland or detector measurements.','Ideal kinematics only; PX4 SITL, ROS 2, OAK-D and Jetson integration have not been executed.','No sockets, serial ports, arm commands or flight commands are sent.','A forward camera alone does not establish coverage behind, above, below or beside the aircraft.','Declared-free voxels in the fixture are ground truth supplied by the simulator. Real mapping must explicitly raycast freespace and preserve unknown cells.']};
  await writeFile(path.join(directory,'latest.json'),JSON.stringify(report,null,2)+'\n');
  const trace=runs.flatMap(r=>r.mission.history.map(step=>JSON.stringify({scenarioId:r.report.id,...step}))).join('\n')+'\n';
  await writeFile(path.join(directory,'proposed-setpoints.jsonl'),trace);
  const lines=['# X500 flight behavior simulation report','',`Generated: ${report.generatedAt}`,'',`**${summary.passed}/${summary.scenarios} scenarios passed.** No drone was connected or flown. All images/candidates in this flight test are synthetic event records; vision tests are separate.`,'','| Scenario | Result | Captures | Final state |','|---|---|---:|---|',...report.scenarios.map(r=>`| ${r.title} | ${r.passed?'PASS':'FAIL'} | ${r.captures} | ${r.finalState} |`),'','## Camera geometry (illustrative calibration)','','5 cm fronto-parallel mushroom cap; focal length 1,380 pixels. Pixel width is a geometric estimate, not detector accuracy.','','| Range | Cap pixels | 2px blur budget speed at 8ms exposure |','|---:|---:|---:|',...cameraGeometry.rows.map(r=>`| ${r.rangeM} m | ${r.targetPixels.toFixed(1)} | ${r.blurLimitedSpeedMps.toFixed(3)} m/s |`),'',`At 1 m, increasing exposure from 8 ms to 40 ms reduces the transverse-motion blur budget from ${exposureComparison[0].blurLimitedSpeedMps.toFixed(3)} m/s to ${exposureComparison[1].blurLimitedSpeedMps.toFixed(3)} m/s. The controller caps survey speed at 0.3 m/s and also applies the blur limit. Captures require an additional stop and settle.`,'','## Limits','',...report.limitations.map(s=>`- ${s}`),''];
  await writeFile(path.join(directory,'LATEST.md'),lines.join('\n'));
  return report;
}

if(process.argv[1] && path.resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const report=await writeReports(); console.log(JSON.stringify({simulationOnly:true,...report.summary,report:'flight/reports/latest.json'},null,2));
  if(report.summary.failed) process.exitCode=1;
}
