import { positive,point,distance,sweptClearance,blurLimitedSpeed,blurPixels } from './geometry.mjs';

const terminal=new Set(['MANUAL','ABORTED','COMPLETE']);
const stationary=new Set(['DISARMED','PREFLIGHT','READY','HOLD']);
const angleError=(a,b)=>Math.abs(Math.atan2(Math.sin(a-b),Math.cos(a-b)));

export function validateConfig(config) {
  if(config?.simulationOnly!==true || config.frame!=='map_enu') throw new Error('Only map_enu simulation configurations are accepted');
  for(const [section,fields] of Object.entries({vehicle:['radiusM','clearanceM','maxPoseSigmaM'],safety:['maxPoseAgeMs','maxDepthAgeMs','maxMapAgeMs','maxVoxelAgeMs','minBatteryFraction','maxPoseVarianceM2','maxInputGapMs','maxClockSkewMs'],survey:['maxSpeedMps','arrivalToleranceM','settleMs','maxCaptureSpeedMps','maxCaptureYawRateRadS','yawToleranceRad','maxDecelerationMps2','reactionTimeS'],camera:['fxPx','fyPx','widthPx','heightPx','exposureS','maxBlurPx','minTargetDiameterM','nearestSurveyRangeM','designTargetPixels'],candidate:['requiredDistinctFrames','maxObservationAgeMs','minStandoffM','maxApproachDistanceM']})) {
    for(const field of fields) positive(config[section]?.[field],`${section}.${field}`);
  }
  if(config.safety.minBatteryFraction>1) throw new Error('minBatteryFraction must be <= 1');
  if(Math.sqrt(config.safety.maxPoseVarianceM2)>config.vehicle.maxPoseSigmaM+1e-12) throw new Error('pose uncertainty inflation must cover configured variance gate');
  if(!Number.isInteger(config.candidate.requiredDistinctFrames)) throw new Error('requiredDistinctFrames must be an integer');
  if(!Array.isArray(config.survey.captureYawOffsetsRad) || !config.survey.captureYawOffsetsRad.length || !config.survey.captureYawOffsetsRad.every(Number.isFinite)) throw new Error('capture yaw offsets required');
  return config;
}

/** Produces proposed setpoints for deterministic software simulation only.
 * There is intentionally no arm command, socket, serial or MAVLink client. */
export class SurveyMission {
  constructor(config,waypoints) {
    this.config=structuredClone(validateConfig(config));
    if(!Array.isArray(waypoints) || !waypoints.length) throw new Error('survey waypoints required');
    this.waypoints=waypoints.map((p,i)=>({...point(p,`waypoint ${i}`),yawRad:Number.isFinite(p.yawRad)?p.yawRad:0}));
    this.state='DISARMED'; this.index=0; this.scanIndex=0; this.settleSince=null; this.lastNowMs=null; this.captureCount=0; this.candidate=null; this.history=[];
  }
  health(input) {
    const {nowMs,telemetry:t,map}=input, s=this.config.safety;
    if(!Number.isFinite(nowMs)) return 'invalid_clock';
    if(this.lastNowMs!==null && (nowMs<this.lastNowMs || nowMs-this.lastNowMs>s.maxInputGapMs)) return 'control_loop_gap';
    try { point(t?.position); } catch { return 'invalid_pose'; }
    const age=(v,max)=>Number.isFinite(v) && nowMs-v<=max && nowMs-v>=-s.maxClockSkewMs;
    if(t?.localization?.valid!==true || !age(t.localization.updatedMs,s.maxPoseAgeMs) || !Number.isFinite(t.localization.varianceM2) || t.localization.varianceM2<0 || t.localization.varianceM2>s.maxPoseVarianceM2) return 'localization_unreliable';
    if(t.depth?.valid!==true || !age(t.depth.updatedMs,s.maxDepthAgeMs)) return 'depth_unavailable';
    if(!age(map?.updatedMs,s.maxMapAgeMs)) return 'map_stale';
    if(!Number.isFinite(t.batteryFraction) || t.batteryFraction<s.minBatteryFraction || t.batteryFraction>1) return 'battery_reserve';
    if(t.manualLinkAvailable!==true) return 'manual_link_lost';
    if(!Number.isFinite(t.speedMps) || t.speedMps<0 || !Number.isFinite(t.yawRateRadS) || !Number.isFinite(t.yawRad)) return 'invalid_motion';
    return null;
  }
  corridor(input,to,extraRadiusM=0) {
    const c=this.config, radiusM=c.vehicle.radiusM+c.vehicle.clearanceM+c.vehicle.maxPoseSigmaM+extraRadiusM;
    return sweptClearance(input.map,input.telemetry.position,to,{radiusM,nowMs:input.nowMs,maxVoxelAgeMs:c.safety.maxVoxelAgeMs,maxClockSkewMs:c.safety.maxClockSkewMs});
  }
  output(input,reason,proposal=null,extra={}) {
    const result={simulationOnly:true,actuationEnabled:false,state:this.state,reason,nowMs:Number.isFinite(input.nowMs)?input.nowMs:null,waypointIndex:this.index,proposal,...extra};
    this.history.push(result); if(Number.isFinite(input.nowMs)) this.lastNowMs=input.nowMs; return result;
  }
  holdProposal(input) {
    const t=input.telemetry,s=this.config.safety,age=input.nowMs-t?.localization?.updatedMs;
    const trustworthy=t?.localization?.valid===true && Number.isFinite(age) && age>=-s.maxClockSkewMs && age<=s.maxPoseAgeMs && Number.isFinite(t.localization.varianceM2) && t.localization.varianceM2>=0 && t.localization.varianceM2<=s.maxPoseVarianceM2;
    try { point(t?.position); } catch { return null; }
    return trustworthy?{type:'position_hold',frame:'map_enu',position:{...t.position},speedMps:0}:null;
  }
  fail(input,reason,details={}) {
    this.state='HOLD'; this.settleSince=null;
    return this.output(input,reason,this.holdProposal(input),{manualTakeoverRequired:true,automaticResume:false,...details});
  }
  candidateEvidence(input,candidate) {
    try { point(candidate?.position); } catch { return {valid:false,reason:'candidate_has_no_metric_position'}; }
    if(candidate.frame!=='map_enu' || typeof candidate.id!=='string' || !candidate.id) return {valid:false,reason:'candidate_frame_or_id_invalid'};
    const unique=new Set(),c=this.config.candidate,s=this.config.safety;
    for(const observation of Array.isArray(candidate.observations)?candidate.observations:[]) {
      if(!observation || typeof observation!=='object') continue;
      const age=input.nowMs-observation.capturedMs;
      let poseAtCaptureValid=false;
      try { point(observation.capturePose?.position); poseAtCaptureValid=observation.capturePose.frame==='map_enu' && observation.capturePose.capturedMs===observation.capturedMs && Number.isFinite(observation.capturePose.yawRad); } catch { /* Missing capture-pose evidence cannot authorize an approach. */ }
      if(typeof observation.frameId==='string' && observation.frameId && observation.depthKnown===true && observation.qualityAccepted===true && poseAtCaptureValid && Number.isFinite(age) && age>=-s.maxClockSkewMs && age<=c.maxObservationAgeMs && Number.isFinite(observation.rangeM) && observation.rangeM>0) unique.add(observation.frameId);
    }
    return {valid:unique.size>=c.requiredDistinctFrames,reason:unique.size>=c.requiredDistinctFrames?'repeated_candidate_evidence':'candidate_needs_more_frames',distinctFrames:unique.size,identification:'unverified'};
  }
  step(input) {
    if(!input || typeof input!=='object') throw new Error('input object required');
    if(terminal.has(this.state)) return this.output(input,'terminal_state',null,{manualTakeoverRequired:this.state!=='COMPLETE'});
    if(input.event==='manual') { this.state='MANUAL'; return this.output(input,'manual_takeover',null); }
    if(input.event==='abort') { this.state='ABORTED'; return this.output(input,'operator_abort',this.holdProposal(input),{manualTakeoverRequired:true}); }
    if(this.state==='DISARMED') { if(input.event==='preflight') this.state='PREFLIGHT'; return this.output(input,this.state==='PREFLIGHT'?'preflight_requested':'awaiting_preflight'); }
    const issue=this.health(input);
    if(issue) { if(this.state==='PREFLIGHT') return this.output(input,issue,null,{ready:false}); return this.fail(input,issue); }
    const occupancy=this.corridor(input,input.telemetry.position);
    if(!occupancy.clear) { if(this.state==='PREFLIGHT') return this.output(input,occupancy.reason,null,{ready:false,clearance:occupancy}); return this.fail(input,occupancy.reason,{clearance:occupancy}); }
    if(this.state==='PREFLIGHT') { this.state='READY'; return this.output(input,'simulation_preflight_passed',null,{ready:true,calibrationStatus:this.config.camera.calibrationStatus}); }
    if(this.state==='READY') { if(input.event==='begin_simulation') this.state='SURVEY_TRANSIT'; else return this.output(input,'awaiting_simulation_start'); }
    if(this.state==='HOLD') {
      if(input.event!=='resume') return this.output(input,'explicit_resume_required',this.holdProposal(input),{manualTakeoverRequired:true,automaticResume:false});
      this.state='SURVEY_TRANSIT'; this.candidate=null; this.scanIndex=0; this.settleSince=null;
    }
    if(input.event==='candidate' && input.candidate && ['SURVEY_TRANSIT','SURVEY_SETTLE','SURVEY_CAPTURE'].includes(this.state)) {
      this.candidate=structuredClone(input.candidate); this.state='CANDIDATE_VERIFY'; this.settleSince=null;
    }
    if(this.state==='CANDIDATE_VERIFY') {
      if(input.candidate?.id===this.candidate.id) this.candidate=structuredClone(input.candidate);
      const evidence=this.candidateEvidence(input,this.candidate);
      if(input.event==='dismiss_candidate') { this.candidate=null; this.state='SURVEY_TRANSIT'; return this.output(input,'candidate_dismissed',this.holdProposal(input)); }
      if(input.event==='approach') {
        if(!evidence.valid) return this.output(input,evidence.reason,this.holdProposal(input),{evidence});
        const target=this.candidate.approachViewpoint, current=input.telemetry.position;
        try { point(target); } catch { return this.output(input,'no_verified_viewpoint',this.holdProposal(input)); }
        if(Math.abs(target.z-current.z)>1e-6) return this.output(input,'approach_altitude_change_blocked',this.holdProposal(input));
        if(distance(current,target)>this.config.candidate.maxApproachDistanceM || distance(target,this.candidate.position)<this.config.candidate.minStandoffM) return this.output(input,'viewpoint_standoff_or_distance_invalid',this.holdProposal(input));
        const clearance=this.corridor(input,target);
        if(!clearance.clear) return this.output(input,'approach_corridor_blocked',this.holdProposal(input),{clearance});
        this.approachTarget={...target,yawRad:Number.isFinite(target.yawRad)?target.yawRad:input.telemetry.yawRad}; this.state='APPROACH_TRANSIT';
      } else {
        const quality=this.captureQuality(input);
        return this.output(input,evidence.reason,this.holdProposal(input),{evidence,captureRequested:quality.ready,capture:quality.ready?this.captureRecord(input,'candidate_verification'):null});
      }
    }
    if(this.state==='SURVEY_TRANSIT' || this.state==='APPROACH_TRANSIT') {
      const approaching=this.state==='APPROACH_TRANSIT',target=approaching?this.approachTarget:this.waypoints[this.index],from=input.telemetry.position;
      if(approaching) {
        if(input.candidate?.id===this.candidate.id) this.candidate=structuredClone(input.candidate);
        const evidence=this.candidateEvidence(input,this.candidate);
        if(!evidence.valid) return this.fail(input,'candidate_evidence_expired',{evidence});
        if(distance(target,this.candidate.position)<this.config.candidate.minStandoffM) return this.fail(input,'candidate_standoff_changed');
      }
      const clearance=this.corridor(input,target);
      if(!clearance.clear) return this.fail(input,clearance.reason,{clearance});
      const remaining=distance(from,target),c=this.config, speedMps=Math.min(c.survey.maxSpeedMps,blurLimitedSpeed(c.camera,c.camera.nearestSurveyRangeM,input.telemetry.yawRateRadS));
      if(remaining<=c.survey.arrivalToleranceM) {
        if(approaching) { this.state='CANDIDATE_VERIFY'; this.settleSince=null; return this.output(input,'verification_viewpoint_reached',this.holdProposal(input)); }
        this.state='SURVEY_SETTLE'; this.settleSince=null;
      } else {
        const travelYaw=Math.atan2(target.y-from.y,target.x-from.x);
        if(angleError(input.telemetry.yawRad,travelYaw)>c.survey.yawToleranceRad || Math.abs(input.telemetry.yawRateRadS)>c.survey.maxCaptureYawRateRadS) return this.output(input,'orient_before_translation',{...this.holdProposal(input),yawRad:travelYaw});
        if(speedMps<=0) return this.fail(input,'rotation_exhausts_blur_budget');
        // Unknown velocity direction is handled by reserving braking distance in all axes.
        const brakingSpeed=Math.max(input.telemetry.speedMps,speedMps);
        const brakeM=brakingSpeed*c.survey.reactionTimeS+brakingSpeed**2/(2*c.survey.maxDecelerationMps2);
        const brakingClearance=this.corridor(input,target,brakeM);
        if(!brakingClearance.clear) return this.fail(input,'braking_corridor_blocked',{clearance:brakingClearance});
        return this.output(input,approaching?'verified_corridor_approach':'survey_translation',{type:'position',frame:'map_enu',position:{x:target.x,y:target.y,z:target.z},yawRad:travelYaw,speedMps:Math.min(speedMps,Math.sqrt(2*c.survey.maxDecelerationMps2*remaining))},{clearance,brakingClearance,brakingDistanceM:brakeM});
      }
    }
    if(this.state==='SURVEY_SETTLE' || this.state==='SURVEY_CAPTURE') {
      const waypoint=this.waypoints[this.index],desiredYaw=waypoint.yawRad+this.config.survey.captureYawOffsetsRad[this.scanIndex];
      if(distance(input.telemetry.position,waypoint)>this.config.survey.arrivalToleranceM) { this.state='SURVEY_TRANSIT'; this.settleSince=null; return this.output(input,'drifted_from_capture_station',this.holdProposal(input)); }
      const quality=this.captureQuality(input,desiredYaw),proposal={...this.holdProposal(input),yawRad:desiredYaw};
      if(!quality.ready) { this.state='SURVEY_SETTLE'; return this.output(input,quality.reason,proposal,{captureRequested:false,imageQuality:quality}); }
      this.state='SURVEY_CAPTURE'; const capture=this.captureRecord(input,'survey'); this.scanIndex++;
      this.settleSince=null;
      if(this.scanIndex>=this.config.survey.captureYawOffsetsRad.length) {
        this.scanIndex=0; this.index++;
        this.state=this.index>=this.waypoints.length?'COMPLETE':'SURVEY_TRANSIT';
      }
      return this.output(input,'stop_and_look_capture',proposal,{captureRequested:true,capture,imageQuality:quality});
    }
    if(stationary.has(this.state)) return this.output(input,'stationary',this.holdProposal(input));
    throw new Error(`Unhandled state ${this.state}`);
  }
  captureQuality(input,desiredYaw=input.telemetry.yawRad) {
    const {telemetry:t,nowMs}=input,c=this.config,blurPx=blurPixels(c.camera,{speedMps:t.speedMps,rangeM:c.camera.nearestSurveyRangeM,yawRateRadS:t.yawRateRadS});
    if(t.speedMps>c.survey.maxCaptureSpeedMps || Math.abs(t.yawRateRadS)>c.survey.maxCaptureYawRateRadS || angleError(t.yawRad,desiredYaw)>c.survey.yawToleranceRad || blurPx>c.camera.maxBlurPx) { this.settleSince=null; return {ready:false,reason:'waiting_for_still_image',blurPx}; }
    if(this.settleSince===null) this.settleSince=nowMs;
    return {ready:nowMs-this.settleSince>=c.survey.settleMs,reason:nowMs-this.settleSince>=c.survey.settleMs?'settled':'settling',blurPx,settledMs:nowMs-this.settleSince};
  }
  captureRecord(input,purpose) {
    this.captureCount++;
    return {frameId:`sim-${this.captureCount}`,purpose,capturedMs:input.nowMs,position:{...input.telemetry.position},yawRad:input.telemetry.yawRad,fixedCameraPitchRad:this.config.camera.fixedMountPitchRad,exposureS:this.config.camera.exposureS,requiredStreams:['rgb','aligned_depth','left_rectified','right_rectified','camera_info','odometry','imu'],synthetic:true,captureStatus:'proposal_only',photoCaptured:false,identification:'unverified'};
  }
}
