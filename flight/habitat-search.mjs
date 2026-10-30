import { SurveyMission, validateConfig } from './mission.mjs';
import { point,distance,positive,planCorridor3D,blurLimitedSpeed } from './geometry.mjs';
import {observationRange,routeObservation,cameraAim,closePhotoPlan,validatePhotoAcknowledgment,angleDifference} from './range-inspection.mjs';

const clamp=n=>Math.max(0,Math.min(1,n));
const signalNames=['hardwood','moss','leaf_litter','damp_ground','shaded_ground','known_patch','forest_context','vegetation_ground','exposed_ground'];
function validConfidence(n){return Number.isFinite(n)&&n>=0&&n<=1;}
function validCapturePose(o){try{point(o.capturePose?.position);return Number.isFinite(o.capturedMs)&&o.capturePose.frame==='map_enu'&&o.capturePose.capturedMs===o.capturedMs&&Number.isFinite(o.capturePose.yawRad);}catch{return false;}}
function clone(v){return structuredClone(v);}
const xyz=p=>({x:p.x,y:p.y,z:p.z});
function safeBbox(box){if(!box||typeof box!=='object')return null;const out={};for(const k of ['x','y','width','height','x1','y1','x2','y2'])if(Number.isFinite(box[k]))out[k]=box[k];if(['pixel','normalized'].includes(box.units))out.units=box.units;return Object.keys(out).length?out:null;}
function planningRadius(config,currentSpeedMps=0){const speed=Math.max(currentSpeedMps,Math.min(config.survey.maxSpeedMps,blurLimitedSpeed(config.camera,config.camera.nearestSurveyRangeM)));return config.vehicle.radiusM+config.vehicle.clearanceM+config.vehicle.maxPoseSigmaM+speed*config.survey.reactionTimeS+speed**2/(2*config.survey.maxDecelerationMps2)+config.survey.arrivalToleranceM;}

export function validateSearchConfig(config){
  if(config?.simulationOnly!==true || config.weightsAreHeuristic!==true || config.liveHabitatClassifierImplemented!==false)throw new Error('Only explicitly heuristic simulation habitat configurations are supported');
  for(const [section,fields]of Object.entries({queue:['maxLeads','dedupeRadiusM','habitatContextRadiusM','leadExpiryMs','maxActionEvidenceAgeMs','maxFutureSkewMs','maxPositionVarianceM2','maxVisitsPerLead','revisitMs','unsafeRevisitMs'],budget:['maxInspections','maxPlannedTravelM','maxInspectionTimeMs'],inspection:['desiredRangeM','minStandoffM','maxSingleApproachM','minimumRangeImprovementM']}))for(const field of fields)positive(config[section]?.[field],`${section}.${field}`);
  if(config.inspection.allowAltitudeChange!==false)throw new Error('Habitat inspection altitude changes are disabled');
  if(config.queue.maxActionEvidenceAgeMs>config.queue.leadExpiryMs)throw new Error('Action evidence lifetime must not exceed lead expiry');
  for(const key of ['maxSeenFrames','maxObservationIdsPerLead','maxRejectedHistory','maxMissionHistory','maxScheduleHistory','maxInputObservations','maxStringLength','maxHabitatLog','maxPhotoHistory'])if(!Number.isInteger(config.limits?.[key])||config.limits[key]<1||config.limits[key]>100000)throw new Error(`limits.${key} must be an integer in 1..100000`);
  for(const key of ['distantHabitatThresholdM','targetFrameFillFraction','pitchToleranceRad','maxApproachSegments','captureAckTimeoutMs'])positive(config.inspection[key],`inspection.${key}`);
  if(config.inspection.distantHabitatThresholdM!==50||config.inspection.targetFrameFillFraction>1||!Number.isInteger(config.inspection.maxApproachSegments))throw new Error('Invalid range or frame-fill inspection policy');
  for(const [section,key]of [['queue','maxLeads'],['queue','maxVisitsPerLead'],['budget','maxInspections']])if(!Number.isInteger(config[section][key]))throw new Error(`${section}.${key} must be an integer`);
  for(const name of [...signalNames,'season','visual','nearbyHabitat'])if(!validConfidence(config.weights?.[name]))throw new Error(`Invalid weight ${name}`);
  return config;
}

/** Prioritizes areas to inspect. It never writes occupancy or declares a species. */
export class InspectionQueue {
  constructor(config){this.config=clone(validateSearchConfig(config));this.leads=new Map();this.seenFrames=new Map();this.sequence=0;this.rejected=[];this.rejectedCount=0;this.habitatLog=[];this.droppedHabitatLog=0;this.budget={inspections:0,plannedTravelM:0,startedMs:null};}
  ingest(observation,nowMs){
    const o=observation,c=this.config.queue,limits=this.config.limits;
    const reject=reason=>{const result={accepted:false,reason};this.rejectedCount++;this.rejected.push(result);if(this.rejected.length>limits.maxRejectedHistory)this.rejected.shift();return result;};
    if(!o||!['habitat','fungus'].includes(o.kind)||typeof o.id!=='string'||!o.id||typeof o.frameId!=='string'||!o.frameId||!validConfidence(o.confidence)||(!Number.isFinite(o.capturedMs)&&o.capturedMs!=null)||!Number.isFinite(nowMs))return reject('malformed_observation');
    const captureTimeKnown=Number.isFinite(o.capturedMs),evidenceMs=captureTimeKnown?o.capturedMs:nowMs;
    if(o.id.length>limits.maxStringLength||o.frameId.length>limits.maxStringLength)return reject('identifier_too_long');
    if(captureTimeKnown&&(nowMs-o.capturedMs>c.leadExpiryMs||o.capturedMs-nowMs>c.maxFutureSkewMs))return reject('observation_expired_or_future');
    const identity=`${o.kind}:${o.id}:${o.frameId}`;
    for(const [cacheKey,receivedMs]of this.seenFrames)if(nowMs-receivedMs>c.leadExpiryMs)this.seenFrames.delete(cacheKey);
    if(this.seenFrames.has(identity))return reject('duplicate_frame_does_not_refresh_evidence');
    // Reject new frames at capacity rather than evicting still-live replay protection.
    if(this.seenFrames.size>=limits.maxSeenFrames)return reject('evidence_cache_capacity');
    // A map marker requires historical calibrated depth+pose provenance. Motion separately requires recent evidence.
    let located=false;
    try{point(o.metric?.position);const calibrated=o.synthetic===true||typeof o.metric.calibrationId==='string'&&o.metric.calibrationId.length>0&&o.metric.calibrationId.length<=limits.maxStringLength&&o.metric.calibrationId===o.capturePose?.calibrationId;located=calibrated&&validCapturePose(o)&&o.metric.frame==='map_enu'&&o.metric.depthKnown===true&&o.metric.depthCapturedMs===o.capturedMs&&Number.isFinite(o.metric.positionVarianceM2)&&o.metric.positionVarianceM2>=0&&o.metric.positionVarianceM2<=c.maxPositionVarianceM2;}catch{}
    let lead=[...this.leads.values()].find(l=>l.kind===o.kind&&l.observationIds.includes(o.id));
    if(!lead&&located)lead=[...this.leads.values()].find(l=>l.kind===o.kind&&l.locationStatus==='metric'&&distance(l.position,o.metric.position)<=c.dedupeRadiusM);
    if(!lead&&this.leads.size>=c.maxLeads){const expired=[...this.leads.values()].filter(l=>nowMs-l.lastObservedMs>c.leadExpiryMs&&l.status!=='inspecting').sort((a,b)=>a.lastObservedMs-b.lastObservedMs)[0];if(expired)this.leads.delete(expired.id);else return reject('queue_capacity');}
    if(!lead){lead={id:`inspection-${++this.sequence}`,kind:o.kind,label:o.kind==='habitat'?'Possible black-trumpet habitat':'Black-trumpet-like visual candidate',observationIds:[],createdMs:evidenceMs,lastObservedMs:evidenceMs,locationStatus:'screen_only',signals:{},confidence:0,visits:0,nextEligibleMs:0,status:'queued',evidence:[],identifiedSpecies:null,edible:null,groundSafety:'unknown',liveClassifier:false};this.leads.set(lead.id,lead);}
    this.seenFrames.set(identity,nowMs);
    if(!lead.observationIds.includes(o.id)){lead.observationIds.push(o.id);if(lead.observationIds.length>limits.maxObservationIdsPerLead)lead.observationIds.shift();}
    // A real capture can supersede an unlocated archive receipt even when it was captured slightly earlier.
    const newer=evidenceMs>=lead.lastObservedMs||(located&&lead.captureTimeKnown===false);
    if(newer){
      lead.lastObservedMs=evidenceMs;lead.originalCapturedMs=captureTimeKnown?o.capturedMs:null;lead.captureTimeKnown=captureTimeKnown;lead.receivedMs=nowMs;lead.confidence=o.confidence;lead.capturePose=validCapturePose(o)?{frame:'map_enu',position:xyz(o.capturePose.position),yawRad:o.capturePose.yawRad,capturedMs:o.capturePose.capturedMs}:null;lead.frameId=o.frameId;
      lead.originalKind=o.kind;lead.cueType=o.cueType==='color_salience'?'color_salience':o.kind==='habitat'?'habitat':'detector';lead.rangeEvidence=observationRange(o,located);const routing=routeObservation({...o,cueType:lead.cueType},lead.rangeEvidence,this.config.inspection.distantHabitatThresholdM);Object.assign(lead,routing);lead.humanConfirmationRequired=true;lead.confirmedMushroom=false;lead.photoStatus??='closer_photo_requested';
      lead.inspectionRequest={...(lead.inspectionRequest??{}),directTargetViewRequired:true,nativePhotoRequired:true,targetNativeWidthFraction:this.config.inspection.targetFrameFillFraction,humanConfirmationRequired:true,actuationEnabled:false,status:lead.photoStatus};
      if(o.targetExtent?.measured===true&&Number.isFinite(o.targetExtent.diameterM)&&o.targetExtent.diameterM>0&&o.targetExtent.capturedMs===o.capturedMs&&typeof o.targetExtent.sourceId==='string')lead.targetExtent={measured:true,diameterM:o.targetExtent.diameterM,capturedMs:o.targetExtent.capturedMs,sourceId:o.targetExtent.sourceId.slice(0,limits.maxStringLength)};else lead.targetExtent=null;
      if(lead.routingCategory==='habitat_followup'){
        const entry={leadId:lead.id,originalKind:lead.originalKind,cueType:lead.cueType,label:lead.label,routingCategory:lead.routingCategory,routingReason:lead.routingReason,rangeEvidence:clone(lead.rangeEvidence),locationStatus:located?'metric':'screen_only',position:located?xyz(o.metric.position):null,coordinateFrame:located?'map_enu':null,capturedMs:captureTimeKnown?o.capturedMs:null,firstLoggedMs:nowMs,lastLoggedMs:nowMs,frameId:o.frameId,confirmedMushroom:false,humanConfirmationRequired:true};
        const existing=this.habitatLog.findIndex(row=>row.leadId===lead.id);if(existing>=0)this.habitatLog[existing]={...entry,firstLoggedMs:this.habitatLog[existing].firstLoggedMs};else this.habitatLog.push(entry);
        if(this.habitatLog.length>limits.maxHabitatLog){this.habitatLog.shift();this.droppedHabitatLog++;}
      }
      lead.locationStatus=located?'metric':'screen_only';lead.position=located?xyz(o.metric.position):null;lead.metric=located?{frame:'map_enu',position:xyz(o.metric.position),depthKnown:true,depthCapturedMs:o.metric.depthCapturedMs,positionVarianceM2:o.metric.positionVarianceM2}:null;
      lead.bbox=safeBbox(o.bbox);lead.signals={};
      lead.semanticFractions={};for(const key of ['tree','plant','grass','earth','water','rock','path'])if(validConfidence(o.semanticFractions?.[key]))lead.semanticFractions[key]=o.semanticFractions[key];
      for(const key of signalNames)if(validConfidence(o.signals?.[key]))lead.signals[key]=o.signals[key];
      lead.month=Number.isInteger(o.month)&&o.month>=1&&o.month<=12?o.month:null;
      lead.source=typeof o.source==='string'?o.source.slice(0,limits.maxStringLength):'unspecified';
      lead.synthetic=o.synthetic===true||o.source==='simulated_semantics';lead.evidenceMode=lead.synthetic?'simulation_fixture':o.archivalPhoto===true?'archival_inference':'unspecified';
      if(lead.status==='expired')lead.status='queued';
    }
    lead.evidence.push({frameId:o.frameId,capturedMs:captureTimeKnown?o.capturedMs:null,receivedMs:nowMs,source:typeof o.source==='string'?o.source.slice(0,limits.maxStringLength):'unspecified',confidence:o.confidence,located});lead.evidence=lead.evidence.slice(-8);
    return {accepted:true,leadId:lead.id,locationStatus:lead.locationStatus,reason:located?'metric_inspection_lead':'screen_only_needs_depth_and_capture_pose'};
  }
  habitatScore(lead){const w=this.config.weights;let score=0;const reasons=[];
    for(const key of signalNames){const strength=lead.signals[key]??0;if(strength>0){score+=w[key]*strength*lead.confidence;reasons.push(`${key.replaceAll('_',' ')} cue (${Math.round(strength*100)}% input strength; heuristic)`);}}
    if(lead.month>=7&&lead.month<=10){score+=w.season;reasons.push('Michigan July–October season cue; no fruiting guarantee');}
    return {score:clamp(score),reasons};
  }
  snapshot(nowMs){
    const result=[];
    for(const lead of this.leads.values()){
      const ageMs=nowMs-lead.lastObservedMs;if(ageMs>this.config.queue.leadExpiryMs)lead.status='expired';
      let {score,reasons}=this.habitatScore(lead);
      if(lead.kind==='fungus'){
        let context=0;
        if(lead.locationStatus==='metric')for(const other of this.leads.values())if(other.kind==='habitat'&&other.locationStatus==='metric'&&nowMs-other.lastObservedMs<=this.config.queue.leadExpiryMs&&distance(lead.position,other.position)<=this.config.queue.habitatContextRadiusM)context=Math.max(context,this.habitatScore(other).score);
        score=clamp(this.config.weights.visual*lead.confidence+this.config.weights.nearbyHabitat*context);reasons=[`Unverified visual cue (${Math.round(lead.confidence*100)}% detector score)`,lead.routingReason,...(context>0?['Nearby independently observed habitat raises inspection priority']:[])];
      }
      if(lead.locationStatus==='screen_only')reasons.push('Screen-only lead: no metric navigation target');
      if(!lead.captureTimeKnown)reasons.push('Original capture time unknown; receipt time only manages the screen queue');
      if(ageMs>this.config.queue.maxActionEvidenceAgeMs)reasons.push('Fresh capture/depth/pose required before motion');
      if(lead.nextEligibleMs>nowMs)reasons.push('Revisit cooldown');
      const hasCue=lead.confidence>0&&(lead.kind==='fungus'||lead.cueType==='color_salience'||Object.values(lead.signals).some(v=>v>0));
      if(!hasCue)reasons.push('No positive observed cue; season alone cannot create an inspection');
      const eligible=hasCue&&lead.captureTimeKnown&&lead.locationStatus==='metric'&&ageMs<=this.config.queue.maxActionEvidenceAgeMs&&ageMs>=-this.config.queue.maxFutureSkewMs&&!['expired','inspecting'].includes(lead.status)&&lead.visits<this.config.queue.maxVisitsPerLead&&lead.nextEligibleMs<=nowMs;
      result.push({...clone(lead),priority:score,priorityIsProbability:false,reasons,ageMs,eligible,expiresMs:lead.lastObservedMs+this.config.queue.leadExpiryMs});
    }
    return result.sort((a,b)=>Number(b.routingCategory==='immediate_investigation')-Number(a.routingCategory==='immediate_investigation')||b.priority-a.priority||a.createdMs-b.createdMs||a.id.localeCompare(b.id));
  }
  withinBudget(nowMs,travelM=0){const c=this.config.budget;return Number.isFinite(nowMs)&&Number.isFinite(travelM)&&travelM>=0&&this.budget.inspections<c.maxInspections&&this.budget.plannedTravelM+travelM<=c.maxPlannedTravelM&&(this.budget.startedMs===null||nowMs>=this.budget.startedMs&&nowMs-this.budget.startedMs<c.maxInspectionTimeMs);}
  reserve(leadId,nowMs,travelM){positive(travelM,'travelM',true);if(!this.withinBudget(nowMs,travelM))return false;const lead=this.leads.get(leadId);if(!lead||!this.snapshot(nowMs).some(l=>l.id===leadId&&l.eligible))return false;this.budget.inspections++;this.budget.plannedTravelM+=travelM;this.budget.startedMs??=nowMs;lead.status='inspecting';return true;}
  deferUnsafe(leadId,nowMs,reason){const lead=this.leads.get(leadId);if(!lead)return;lead.status='deferred';lead.lastOutcome=reason;lead.pathAttempts=(lead.pathAttempts??0)+1;lead.nextEligibleMs=nowMs+this.config.queue.unsafeRevisitMs;}
  reserveSegment(travelM,nowMs){const b=this.budget,c=this.config.budget;if(!Number.isFinite(travelM)||travelM<0||b.plannedTravelM+travelM>c.maxPlannedTravelM||b.startedMs!==null&&nowMs-b.startedMs>=c.maxInspectionTimeMs)return false;b.plannedTravelM+=travelM;return true;}
  finish(leadId,nowMs,outcome){const lead=this.leads.get(leadId);if(!lead)return;lead.visits++;lead.status='inspected';lead.lastOutcome=outcome;lead.nextEligibleMs=nowMs+(outcome==='unsafe'?this.config.queue.unsafeRevisitMs:this.config.queue.revisitMs);}
}

/** Sanitized archive replay for research UI; no image IDs, photos or metric poses. */
export function summarizeArchiveInference(report,config){
  if(!report||!Array.isArray(report.observations)||!Array.isArray(report.images))throw new Error('Archive report requires observations and images');
  const queue=new InspectionQueue(config);
  for(const [index,raw]of report.observations.entries()){
    const grouped=raw.context?.semanticFractions??{},signals={};
    for(const key of ['forest_context','vegetation_ground','exposed_ground'])if(validConfidence(raw.signals?.[key]))signals[key]=raw.signals[key];
    queue.ingest({id:`archive-context-${index+1}`,frameId:`archive-frame-${index+1}`,kind:'habitat',capturedMs:null,confidence:validConfidence(raw.confidence)?raw.confidence:0,source:'SegFormer offline research inference',archivalPhoto:true,signals,semanticFractions:{tree:grouped.trees??0,plant:grouped.vegetation??0,earth:grouped.ground??0,water:grouped.water??0,rock:grouped.rock??0,path:grouped.path??0}},0);
  }
  const leads=queue.snapshot(0),durations=report.images.map(i=>i.inferenceSeconds).filter(n=>Number.isFinite(n)&&n>=0);
  return {status:'available',measuredInference:report.measuredInference===true,ecologicalAccuracyMeasured:false,imageCount:report.images.length,screenLeadCount:leads.length,mapTargetCount:0,rejectedCount:queue.rejectedCount,meanInferenceSeconds:durations.length?durations.reduce((a,b)=>a+b,0)/durations.length:null,modelId:report.model?.modelId??null,modelRevision:report.model?.resolvedRevision??null,useRestriction:report.model?.useRestriction??'Consult model license',generatedAt:report.generatedAt??null,mapTargets:[],screenLeads:leads,notes:['Actual offline image inference; habitat suitability accuracy has not been measured.','Original image capture times are unknown in this replay.','All photo contexts remain unlocated; no metric targets were created.','No source photographs or original photo identifiers are included in this summary.','Generic scene categories do not establish moss, moisture, hardwood species, fungal presence or flight clearance.']};
}

export function inspectionPlan(flightConfig,searchConfig,map,from,lead,nowMs){
  try{point(from);point(lead.position);}catch{return {status:'blocked',reason:'metric_target_required',points:[]};}
  const evidenceAge=nowMs-lead.originalCapturedMs;
  let metricMatches=false;try{point(lead.metric?.position);metricMatches=distance(lead.position,lead.metric.position)<1e-9;}catch{}
  const validProvenance=metricMatches&&validCapturePose({capturedMs:lead.originalCapturedMs,capturePose:lead.capturePose})&&lead.metric?.depthKnown===true&&lead.metric.frame==='map_enu'&&lead.metric.depthCapturedMs===lead.originalCapturedMs&&Number.isFinite(lead.metric.positionVarianceM2)&&lead.metric.positionVarianceM2>=0&&lead.metric.positionVarianceM2<=searchConfig.queue.maxPositionVarianceM2;
  if(lead.locationStatus!=='metric'||!lead.eligible||lead.captureTimeKnown!==true||!Number.isFinite(evidenceAge)||evidenceAge>searchConfig.queue.maxActionEvidenceAgeMs||evidenceAge < -searchConfig.queue.maxFutureSkewMs||!validProvenance)return {status:'blocked',reason:'fresh_metric_evidence_required',points:[]};
  const c=searchConfig.inspection,currentRange=distance(from,lead.position),horizontal=Math.hypot(lead.position.x-from.x,lead.position.y-from.y),vertical=Math.abs(from.z-lead.position.z);
  if(currentRange<=c.minStandoffM)return {status:'blocked',reason:'already_inside_minimum_standoff',points:[]};
  // A target below the vehicle cannot make it descend. Inspect from the same altitude.
  const desiredHorizontalByRange=Math.sqrt(Math.max(0,c.desiredRangeM**2-vertical**2));
  const verticalOffset=lead.position.z-from.z,pitch=flightConfig.camera.fixedMountPitchRad;
  const alignedHorizontal=vertical>1e-6&&Math.sign(verticalOffset)===Math.sign(pitch)&&Math.abs(pitch)>1e-6?vertical/Math.tan(Math.abs(pitch)):desiredHorizontalByRange;
  const desiredHorizontal=Math.max(desiredHorizontalByRange,alignedHorizontal);
  const travel=Math.min(c.maxSingleApproachM,Math.max(0,horizontal-desiredHorizontal));
  const viewpoint=travel<c.minimumRangeImprovementM?xyz(from):{x:from.x+(lead.position.x-from.x)/horizontal*travel,y:from.y+(lead.position.y-from.y)/horizontal*travel,z:from.z};
  if(distance(viewpoint,lead.position)<c.minStandoffM)return {status:'blocked',reason:'minimum_standoff',points:[]};
  const radiusM=planningRadius(flightConfig);
  const route=planCorridor3D(map,from,viewpoint,{radiusM,nowMs,maxVoxelAgeMs:flightConfig.safety.maxVoxelAgeMs,maxClockSkewMs:flightConfig.safety.maxClockSkewMs});
  if(route.status!=='ready')return {...route,reason:route.reason??'no_verified_corridor'};
  const routeDistanceM=route.points.slice(1).reduce((n,p,i)=>n+distance(route.points[i],p),0);
  const photoPlan=closePhotoPlan(flightConfig.camera,c,lead,viewpoint),remainingHorizontal=Math.max(0,horizontal-travel-desiredHorizontal);
  return {...route,leadId:lead.id,kind:lead.kind,viewpoint,routeDistanceM,reservedTravelM:routeDistanceM*2,initialRangeM:currentRange,proposedRangeM:distance(viewpoint,lead.position),reason:'inspect_viewpoint_only',identifiedSpecies:null,photoPlan,moreApproachRequired:remainingHorizontal>=c.minimumRangeImprovementM};
}

/** Two-stage scout/detail scheduling around the existing safety-gated simulator. */
export class HabitatScoutMission {
  constructor(flightConfig,searchConfig,scoutWaypoints){this.flightConfig=clone(validateConfig(flightConfig));this.searchConfig=clone(validateSearchConfig(searchConfig));this.queue=new InspectionQueue(searchConfig);this.child=new SurveyMission(flightConfig,scoutWaypoints);this.stage='SCOUT';this.history=[];this.lastNowMs=null;this.selected=null;this.anchor=null;this.autoStart=0;this.completedInspections=0;this.completedSimulationInspections=0;this.captureSequence=0;this.photoHistory=[];this.droppedPhotoHistory=0;this.schedule=[];this.droppedHistoryEvents=0;this.droppedScheduleEvents=0;this.droppedInputObservations=0;}
  output(input,reason,childOutput=null,extra={}){const row={simulationOnly:true,actuationEnabled:false,stage:this.stage,reason,nowMs:input.nowMs,proposal:childOutput?.proposal??null,capture:childOutput?.capture??null,childState:childOutput?.state??null,...extra};this.history.push(row);if(this.history.length>this.searchConfig.limits.maxMissionHistory){this.history.shift();this.droppedHistoryEvents++;}this.lastNowMs=input.nowMs;return row;}
  recordSchedule(entry){const previous=[...this.schedule].reverse().find(row=>row.leadId===entry.leadId);if(previous?.status===entry.status&&previous.reason===entry.reason&&entry.nowMs-previous.nowMs<this.searchConfig.queue.revisitMs)return;this.schedule.push(entry);if(this.schedule.length>this.searchConfig.limits.maxScheduleHistory){this.schedule.shift();this.droppedScheduleEvents++;}}
  startChild(points,stage){const config=clone(this.flightConfig);if(stage==='DETAIL')config.survey.captureYawOffsetsRad=[0];this.child=new SurveyMission(config,points);this.stage=stage;this.autoStart=3;}
  startDetail(plan,lead){this.activePlan=plan;const points=plan.points.slice(1).length?plan.points.slice(1):[plan.viewpoint];this.startChild(points.map(p=>({...p,yawRad:cameraAim(p,lead.position).yawRad})),'DETAIL');}
  retainPhoto(row){this.photoHistory.push(row);if(this.photoHistory.length>this.searchConfig.limits.maxPhotoHistory){this.photoHistory.shift();this.droppedPhotoHistory++;}}
  resumeScout(input,reason){if(!this.pausedScout)return null;this.child=this.pausedScout;this.pausedScout=null;this.child.lastNowMs=input.nowMs;this.child.settleSince=null;this.stage='SCOUT';this.autoStart=0;return this.output(input,reason,{proposal:this.child.holdProposal(input)},{remainingSurveyWaypoints:this.child.waypoints.length-this.child.index});}
  finishPhotoAttempt(input,outcome,ack=null){
    const lead=this.queue.leads.get(this.selected.id);lead.photoStatus=outcome;lead.inspectionRequest.status=outcome;lead.inspectionRequest.closeupPending=ack?.framingSatisfied!==true;lead.humanConfirmationRequired=true;
    if(ack){this.queue.finish(lead.id,input.nowMs,outcome);if(ack.synthetic)this.completedSimulationInspections++;else this.completedInspections++;}else this.queue.deferUnsafe(lead.id,input.nowMs,outcome);
    this.recordSchedule({leadId:lead.id,kind:lead.kind,status:ack?'capture_acknowledged':'deferred',reason:outcome,nowMs:input.nowMs,synthetic:ack?.synthetic??null,framingSatisfied:ack?.framingSatisfied??false});
    this.pendingPhoto=null;this.selected=null;this.stage='SELECT_INSPECTION';
  }
  childStep(input){let event=this.stage==='SCOUT'&&['preflight','begin_simulation'].includes(input.event)?input.event:undefined;if(this.autoStart===3)event='preflight';else if(this.autoStart===2)event=undefined;else if(this.autoStart===1)event='begin_simulation';if(this.autoStart>0)this.autoStart--;const out=this.child.step({...input,event});if(this.child.history.length>this.searchConfig.limits.maxMissionHistory)this.child.history.splice(0,this.child.history.length-this.searchConfig.limits.maxMissionHistory);return this.stage!=='SCOUT'&&['DISARMED','PREFLIGHT','READY'].includes(out.state)?{...out,proposal:this.child.holdProposal(input)}:out;}
  tryRetreat(input,reason){
    const healthy=this.child.health({...input,nowMs:input.nowMs});
    if(healthy){this.stage='MANUAL_REQUIRED';return this.output(input,reason,null,{manualTakeoverRequired:true,safetyReason:healthy});}
    if(!this.anchor){this.stage='MANUAL_REQUIRED';return this.output(input,reason,null,{manualTakeoverRequired:true});}
    const c=this.flightConfig,radiusM=planningRadius(c,input.telemetry.speedMps);
    const route=planCorridor3D(input.map,input.telemetry.position,this.anchor,{radiusM,nowMs:input.nowMs,maxVoxelAgeMs:c.safety.maxVoxelAgeMs,maxClockSkewMs:c.safety.maxClockSkewMs});
    if(route.status!=='ready'){this.stage='MANUAL_REQUIRED';return this.output(input,'retreat_corridor_unavailable',null,{manualTakeoverRequired:true,trigger:reason,clearance:route});}
    this.queue.deferUnsafe(this.selected?.id,input.nowMs,'unsafe');this.pendingPhoto=null;this.startChild(route.points.slice(1).length?route.points.slice(1):[this.anchor],'RETREAT');
    return this.output(input,'retreat_through_revalidated_corridor',{proposal:this.child.holdProposal(input)},{trigger:reason,route:route.points});
  }
  step(input){
    if(!input||!Number.isFinite(input.nowMs))throw new Error('Finite nowMs required');
    if(['MANUAL','ABORTED','MANUAL_REQUIRED','COMPLETE'].includes(this.stage))return this.output(input,'terminal_state');
    if(input.event==='manual'||input.event==='abort'){this.stage=input.event==='manual'?'MANUAL':'ABORTED';return this.output(input,input.event==='manual'?'manual_takeover':'operator_abort',null,{manualTakeoverRequired:true});}
    const observations=Array.isArray(input.observations)?input.observations:[];
    this.droppedInputObservations+=Math.max(0,observations.length-this.searchConfig.limits.maxInputObservations);
    for(const observation of observations.slice(0,this.searchConfig.limits.maxInputObservations))this.queue.ingest(observation,input.nowMs);
    if(this.lastNowMs!==null&&(input.nowMs<this.lastNowMs||input.nowMs-this.lastNowMs>this.flightConfig.safety.maxInputGapMs)){this.stage='MANUAL_REQUIRED';return this.output(input,'control_loop_gap',null,{manualTakeoverRequired:true});}
    if(this.stage==='SCOUT'){
      const out=this.childStep(input);
      if(out.capture&&out.state!=='COMPLETE'&&this.queue.withinBudget(input.nowMs)&&this.queue.snapshot(input.nowMs).some(l=>l.eligible&&l.routingCategory==='immediate_investigation')){this.pausedScout=this.child;this.stage='SELECT_INSPECTION';return this.output(input,'near_candidate_pauses_scout_at_settled_capture',out,{remainingSurveyWaypoints:this.child.waypoints.length-this.child.index});}
      if(out.state==='COMPLETE')this.stage='SELECT_INSPECTION';
      if(out.state==='HOLD'){this.stage='MANUAL_REQUIRED';return this.output(input,out.reason,out,{manualTakeoverRequired:true});}
      return this.output(input,out.capture?'distant_scout_capture':out.reason,out);
    }
    const health=this.child.health(input);
    if(health){this.stage='MANUAL_REQUIRED';return this.output(input,health,null,{manualTakeoverRequired:true});}
    if(this.stage==='SELECT_INSPECTION'||this.stage==='AWAITING_EVIDENCE'){
      this.child.lastNowMs=input.nowMs;
      const currentClearance=this.child.corridor(input,input.telemetry.position);
      if(!currentClearance.clear){this.stage='MANUAL_REQUIRED';return this.output(input,'current_aircraft_volume_unavailable',null,{manualTakeoverRequired:true,clearance:currentClearance});}
      if(!this.queue.withinBudget(input.nowMs)){for(const lead of this.queue.leads.values())if(lead.status==='queued'){lead.status='deferred';lead.lastOutcome='inspection_budget_exhausted';}const resumed=this.resumeScout(input,'inspection_budget_exhausted_resume_scout');if(resumed)return resumed;this.stage='COMPLETE';return this.output(input,'exploration_budget_exhausted');}
      const leads=this.queue.snapshot(input.nowMs),eligible=leads.filter(l=>l.eligible);
      for(const lead of eligible){
        const plan=inspectionPlan(this.flightConfig,this.searchConfig,input.map,input.telemetry.position,lead,input.nowMs);
        if(plan.status!=='ready'){this.queue.deferUnsafe(lead.id,input.nowMs,plan.reason);this.recordSchedule({leadId:lead.id,kind:lead.kind,status:'blocked',reason:plan.reason,nowMs:input.nowMs});continue;}
        if(!this.queue.reserve(lead.id,input.nowMs,plan.reservedTravelM)){this.recordSchedule({leadId:lead.id,kind:lead.kind,status:'deferred',reason:'exploration_budget',nowMs:input.nowMs});continue;}
        this.selected=lead;this.anchor=xyz(input.telemetry.position);this.recordSchedule({leadId:lead.id,kind:lead.kind,status:'scheduled',reason:plan.reason,priority:lead.priority,reasons:lead.reasons,nowMs:input.nowMs,route:plan.points,viewpoint:plan.viewpoint});
        this.approachSegments=1;this.startDetail(plan,lead);return this.output(input,'closer_inspection_scheduled',null,{leadId:lead.id,kind:lead.kind,plan});
      }
      const resumed=this.resumeScout(input,'inspection_queue_deferred_resume_scout');if(resumed)return resumed;this.stage='AWAITING_EVIDENCE';return this.output(input,'waiting_for_fresh_located_lead',{proposal:this.child.holdProposal(input)},{reacquireRequested:true,screenOnlyCount:leads.filter(l=>l.locationStatus==='screen_only').length});
    }
    if(['DETAIL','AIM_CAPTURE','WAIT_CAPTURE'].includes(this.stage)){
      if(this.queue.budget.startedMs!==null&&input.nowMs-this.queue.budget.startedMs>=this.searchConfig.budget.maxInspectionTimeMs)return this.tryRetreat(input,'inspection_time_budget_exhausted');
      const lead=this.queue.snapshot(input.nowMs).find(l=>l.id===this.selected?.id);
      if(!lead||lead.status==='expired'||lead.captureTimeKnown!==true||lead.ageMs>this.searchConfig.queue.maxActionEvidenceAgeMs||lead.ageMs < -this.searchConfig.queue.maxFutureSkewMs||lead.locationStatus!=='metric')return this.tryRetreat(input,'inspection_evidence_expired');
      if(lead.confidence<=0||(lead.kind==='habitat'&&lead.cueType!=='color_salience'&&!Object.values(lead.signals).some(value=>value>0)))return this.tryRetreat(input,'inspection_evidence_revoked');
      // A shifted target never silently redirects an active route.
      if(distance(lead.position,this.selected.position)>this.flightConfig.survey.arrivalToleranceM)return this.tryRetreat(input,'target_registration_changed');
      if(distance(input.telemetry.position,lead.position)<this.searchConfig.inspection.minStandoffM)return this.tryRetreat(input,'target_minimum_standoff_changed');
    }
    if(this.stage==='AIM_CAPTURE'||this.stage==='WAIT_CAPTURE'){
      this.child.lastNowMs=input.nowMs;
      const clearance=this.child.corridor(input,input.telemetry.position);if(!clearance.clear)return this.tryRetreat(input,'capture_station_no_longer_clear');
      const lead=this.queue.snapshot(input.nowMs).find(l=>l.id===this.selected.id),photoPlan=closePhotoPlan(this.flightConfig.camera,this.searchConfig.inspection,lead,input.telemetry.position),aim=photoPlan.aim;
      if(!photoPlan.cameraAimPossible){this.finishPhotoAttempt(input,'manual_closeup_camera_pitch_unreachable');return this.output(input,'fixed_camera_cannot_center_target',null,{photoPlan});}
      const proposal={...this.child.holdProposal(input),yawRad:aim.yawRad,cameraPitchRad:aim.pitchRad,cameraPitchIsRequirementOnly:true};
      const quality=this.child.captureQuality(input,aim.yawRad);
      if(this.stage==='AIM_CAPTURE'){
        if(!quality.ready)return this.output(input,'aim_and_settle_for_target_photo',{proposal},{photoPlan,imageQuality:quality});
        const sequence=++this.captureSequence,request={requestId:`inspection-photo-${sequence}`,frameId:`target-frame-${sequence}`,leadId:lead.id,requestedMs:input.nowMs,position:xyz(input.telemetry.position),targetPosition:xyz(lead.position),coordinateFrame:'map_enu',aim,photoPlan,requiredStreams:['rgb','aligned_depth','camera_info','odometry'],nativeUncroppedRequired:true,photoCaptured:false,syntheticProposal:true,humanConfirmationRequired:true};
        this.pendingPhoto=request;this.queue.leads.get(lead.id).photoStatus='waiting_for_capture_ack';this.stage='WAIT_CAPTURE';this.retainPhoto({type:'request',...clone(request)});return this.output(input,'target_photo_requested',{proposal},{photoRequest:request,imageQuality:quality});
      }
      if(input.nowMs-this.pendingPhoto.requestedMs>=this.searchConfig.inspection.captureAckTimeoutMs){this.finishPhotoAttempt(input,'capture_ack_missing_manual_closeup_required');return this.output(input,'target_photo_not_confirmed',{proposal});}
      if(input.photoAck){const result=validatePhotoAcknowledgment(input.photoAck,this.pendingPhoto,input,this.flightConfig,this.searchConfig);if(result.accepted&&quality.ready&&distance(input.telemetry.position,this.pendingPhoto.position)<=this.flightConfig.survey.arrivalToleranceM){const requestId=this.pendingPhoto.requestId;this.retainPhoto({type:'ack',requestId,...result});const outcome=result.synthetic?(result.framingSatisfied?'simulated_frame_filling_photo':'simulated_safe_photo_manual_closeup_required'):(result.framingSatisfied?'photo_acknowledged_pending_human_confirmation':'safe_photo_acknowledged_manual_closeup_required');this.finishPhotoAttempt(input,outcome,result);return this.output(input,'target_photo_acknowledged',{proposal},{captureAcknowledgment:result});}return this.output(input,'capture_ack_rejected',{proposal},{captureAcknowledgment:result.accepted?{accepted:false,reason:'current_capture_pose_not_settled'}:result});}
      return this.output(input,'waiting_for_native_photo_ack',{proposal},{photoRequestId:this.pendingPhoto.requestId});
    }
    const out=this.childStep(input);
    if(out.state==='HOLD')return this.stage==='DETAIL'?this.tryRetreat(input,out.reason):(this.stage='MANUAL_REQUIRED',this.output(input,out.reason,null,{manualTakeoverRequired:true}));
    if(out.state==='COMPLETE'){
      if(this.stage==='DETAIL'){
        const lead=this.queue.snapshot(input.nowMs).find(l=>l.id===this.selected.id),plan=inspectionPlan(this.flightConfig,this.searchConfig,input.map,input.telemetry.position,{...lead,eligible:true},input.nowMs);
        if(plan.status!=='ready')return this.tryRetreat(input,'fresh_segment_replan_failed');
        if(plan.routeDistanceM>=this.searchConfig.inspection.minimumRangeImprovementM&&this.approachSegments<this.searchConfig.inspection.maxApproachSegments&&this.queue.reserveSegment(plan.reservedTravelM,input.nowMs)){
          this.selected=lead;this.approachSegments++;this.startDetail(plan,lead);return this.output(input,'fresh_target_segment_replanned',out,{plan,approachSegments:this.approachSegments});
        }
        this.stage='AIM_CAPTURE';this.child.settleSince=null;return this.output(input,'target_viewpoint_reached_capture_not_yet_confirmed',out,{photoPlan:plan.photoPlan,approachSegments:this.approachSegments});
      }
      this.stage='SELECT_INSPECTION';this.selected=null;
    }
    return this.output(input,out.capture?'detail_or_retreat_capture':out.reason,out);
  }
  report(nowMs=this.lastNowMs??0){const leads=this.queue.snapshot(nowMs);return {schemaVersion:2,simulationOnly:true,weightsAreHeuristic:true,liveHabitatClassifierImplemented:false,stage:this.stage,mapTargets:leads.filter(l=>l.locationStatus==='metric'),screenLeads:leads.filter(l=>l.locationStatus==='screen_only'),habitatLog:clone(this.queue.habitatLog),photoHistory:clone(this.photoHistory),schedule:clone(this.schedule),budget:clone(this.queue.budget),completedInspections:this.completedInspections,completedSimulationInspections:this.completedSimulationInspections,physicalFlights:0,retention:{droppedHabitatLog:this.queue.droppedHabitatLog,droppedPhotoHistory:this.droppedPhotoHistory,droppedHistoryEvents:this.droppedHistoryEvents,droppedScheduleEvents:this.droppedScheduleEvents,droppedInputObservations:this.droppedInputObservations,rejectedObservations:this.queue.rejectedCount},events:clone(this.history)};}
}
