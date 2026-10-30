import {point,distance} from './geometry.mjs';

const finitePositive=n=>Number.isFinite(n)&&n>0;
const text=s=>typeof s==='string'&&s.length>0&&s.length<=256;
export const angleDifference=(a,b)=>Math.abs(Math.atan2(Math.sin(a-b),Math.cos(a-b)));

/** Range routing never estimates physical distance from color, image pixels or assumed cap size. */
export function observationRange(observation,located=false){
  const o=observation,r=o.measuredRange;
  if(r!==undefined){
    if(!finitePositive(r?.valueM)||!Number.isFinite(r.uncertaintyM)||r.uncertaintyM<0||!Number.isFinite(r.valueM+r.uncertaintyM)||!['laser_rangefinder','surveyed','stereo_depth'].includes(r.method)||!text(r.sourceId)||!text(r.calibrationId)||!Number.isFinite(o.capturedMs)||r.capturedMs!==o.capturedMs)
      return {status:'unknown',reason:'invalid_measured_range',valueM:null,lowerM:null,upperM:null};
    if(located&&Math.abs(distance(o.capturePose.position,o.metric.position)-r.valueM)>r.uncertaintyM+3*Math.sqrt(o.metric.positionVarianceM2))return {status:'unknown',reason:'range_and_metric_geometry_disagree',valueM:null,lowerM:null,upperM:null};
    return {status:'known',method:r.method,valueM:r.valueM,uncertaintyM:r.uncertaintyM,lowerM:Math.max(0,r.valueM-r.uncertaintyM),upperM:r.valueM+r.uncertaintyM,capturedMs:r.capturedMs,sourceId:r.sourceId,calibrationId:r.calibrationId,synthetic:o.synthetic===true};
  }
  if(located&&(o.synthetic===true||text(o.metric?.calibrationId)&&o.metric.calibrationId===o.capturePose?.calibrationId)){
    const valueM=distance(o.capturePose.position,o.metric.position),uncertaintyM=3*Math.sqrt(o.metric.positionVarianceM2);
    if(finitePositive(valueM))return {status:'known',method:o.synthetic===true?'simulation_metric':'synchronized_metric_pose',valueM,uncertaintyM,lowerM:Math.max(0,valueM-uncertaintyM),upperM:valueM+uncertaintyM,capturedMs:o.capturedMs,calibrationId:o.metric.calibrationId??'simulation-fixture',synthetic:o.synthetic===true};
  }
  return {status:'unknown',reason:'calibrated_synchronized_range_required',valueM:null,lowerM:null,upperM:null};
}

export function routeObservation(observation,range,thresholdM=50){
  const colorOnly=observation.cueType==='color_salience';
  if(colorOnly)return {routingCategory:'habitat_followup',routingReason:'color_cue_requires_context',label:'Unconfirmed color contrast — habitat follow-up'};
  if(observation.kind==='habitat')return {routingCategory:'habitat_followup',routingReason:'habitat_context',label:'Possible mushroom habitat'};
  if(range.status!=='known')return {routingCategory:'unlocated_followup',routingReason:range.reason,label:'Unconfirmed visual cue — range unknown'};
  if(range.upperM>=thresholdM)return {routingCategory:'habitat_followup',routingReason:range.lowerM<thresholdM?'range_interval_crosses_50m':'distant_visual_cue',label:'Distant visual cue — habitat follow-up'};
  return {routingCategory:'immediate_investigation',routingReason:'measured_range_below_50m',label:'Nearby visual candidate — investigate and confirm'};
}

export function cameraAim(from,target){
  point(from);point(target);const dx=target.x-from.x,dy=target.y-from.y,dz=target.z-from.z;
  return {yawRad:Math.atan2(dy,dx),pitchRad:Math.atan2(dz,Math.hypot(dx,dy)),rangeM:distance(from,target)};
}

/** Planning estimate only. A real native-frame bounding box is required to verify framing. */
export function closePhotoPlan(camera,inspection,lead,viewpoint){
  const aim=cameraAim(viewpoint,lead.position),extent=lead.targetExtent;
  const measured=extent?.measured===true&&finitePositive(extent.diameterM)&&extent.capturedMs===lead.originalCapturedMs&&text(extent.sourceId);
  const diameterM=measured?extent.diameterM:camera.minTargetDiameterM;
  const targetFillFraction=inspection.targetFrameFillFraction;
  const expectedNativeWidthPx=camera.fxPx*diameterM/aim.rangeM;
  const requiredRangeM=camera.fxPx*diameterM/(camera.widthPx*targetFillFraction);
  const requiredOpticalMagnification=camera.widthPx*targetFillFraction/expectedNativeWidthPx;
  const pitchErrorRad=angleDifference(aim.pitchRad,camera.fixedMountPitchRad);
  return {targetFillFraction,fillDimension:'native_frame_width',nativeWidthPx:camera.widthPx,nativeHeightPx:camera.heightPx,targetDiameterM:diameterM,targetDiameterMeasured:measured,calibrationStatus:camera.calibrationStatus,expectedNativeWidthPx,expectedFillFraction:expectedNativeWidthPx/camera.widthPx,requiredRangeM,minimumStandoffM:inspection.minStandoffM,requiredOpticalMagnification:Math.max(1,requiredOpticalMagnification),digitalCropAddsDetail:false,framingVerified:false,aim,pitchErrorRad,cameraAimPossible:pitchErrorRad<=inspection.pitchToleranceRad,closeupStatus:requiredRangeM<inspection.minStandoffM?'zoom_required_or_manual_closeup':requiredOpticalMagnification>1?'closer_view_required':'native_framing_estimate_only',humanConfirmationRequired:true};
}

export function validatePhotoAcknowledgment(ack,request,input,config,search){
  const reject=reason=>({accepted:false,reason});
  if(config.simulationOnly===true&&ack?.synthetic!==true)return reject('simulation_requires_explicit_synthetic_ack');
  if(!ack||ack.requestId!==request.requestId||ack.leadId!==request.leadId)return reject('capture_ack_identity_mismatch');
  if(ack.frameId!==request.frameId)return reject('capture_ack_frame_mismatch');
  if(!Number.isFinite(ack.capturedMs)||ack.capturedMs<request.requestedMs||ack.capturedMs>input.nowMs+config.safety.maxClockSkewMs||input.nowMs-ack.capturedMs>search.inspection.captureAckTimeoutMs)return reject('capture_ack_timestamp_invalid');
  if(!/^[a-f0-9]{64}$/.test(ack.photoSha256??'')||!text(ack.photoRef)||ack.nativeUncropped!==true||ack.widthPx!==config.camera.widthPx||ack.heightPx!==config.camera.heightPx)return reject('native_photo_evidence_required');
  try{point(ack.position);if(distance(ack.position,request.position)>config.survey.arrivalToleranceM)return reject('capture_pose_changed');}catch{return reject('capture_pose_required');}
  if(!Number.isFinite(ack.yawRad)||!Number.isFinite(ack.pitchRad)||angleDifference(ack.yawRad,request.aim.yawRad)>config.survey.yawToleranceRad||angleDifference(ack.pitchRad,request.aim.pitchRad)>search.inspection.pitchToleranceRad)return reject('capture_not_aimed_at_target');
  if(ack.settled!==true||ack.sharpnessAccepted!==true)return reject('capture_quality_not_acknowledged');
  const box=ack.targetBbox;
  const validBox=Array.isArray(box)&&box.length===4&&box.every(Number.isFinite)&&box[0]>=0&&box[1]>=0&&box[2]<=ack.widthPx&&box[3]<=ack.heightPx&&box[2]>box[0]&&box[3]>box[1];
  const fillFraction=validBox?(box[2]-box[0])/ack.widthPx:null;
  const centered=validBox&&Math.abs((box[0]+box[2])/2-ack.widthPx/2)<ack.widthPx*.1&&Math.abs((box[1]+box[3])/2-ack.heightPx/2)<ack.heightPx*.1;
  const framingSatisfied=validBox&&centered&&fillFraction>=search.inspection.targetFrameFillFraction;
  return {accepted:true,reason:ack.synthetic===true?'simulated_capture_acknowledged':'external_camera_capture_acknowledged',synthetic:ack.synthetic===true,framingSatisfied,fillFraction,frameId:ack.frameId,photoSha256:ack.photoSha256,photoRef:ack.photoRef,capturedMs:ack.capturedMs,realPhotoIndependentlyVerified:false,humanConfirmationRequired:true};
}
