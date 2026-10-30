import { distance, planCorridor3D, sweptClearance } from './geometry.mjs';
import { cameraAim, angleDifference } from './range-inspection.mjs';

// This reference executor produces proposals only. It contains no MAVLink,
// network, serial, arming or actuator transport. Live inputs never become flight commands.
export const LIVE_EXECUTOR_DEFAULTS = Object.freeze({
  maxSensorAgeMs: 1500, maxClockSkewMs: 30, minBatteryPercent: 30,
  radiusM: .65, standoffM: 2, arrivalToleranceM: .18,
  settledSpeedMps: .08, settleMs: 800, aimToleranceRad: .12,
  captureTimeoutMs: 8000, targetFillFraction: .7, maxTargetStdM: .75,
  maxTargetAgeMs: 10000,
});
const vec = a => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);
const xyz = a => ({x:a[0], y:a[1], z:a[2]});
const array = p => [p.x,p.y,p.z];
const fresh = (timestampMs, nowMs, config) => Number.isFinite(timestampMs) && nowMs - timestampMs <= config.maxSensorAgeMs && nowMs - timestampMs >= -config.maxClockSkewMs;
export const taskTerminal = task => ['completed','cancelled'].includes(task.status);

function hold(task, reason, nowMs) {
  task.status = 'blocked'; task.blockedReason = reason; task.updatedMs = nowMs;
  task.settledSinceMs = null;
  return {type:'hold',reason,taskId:task.id,actuationEnabled:false};
}

export function inspectPreconditions(state, task, nowMs, options={}) {
  const config={...LIVE_EXECUTOR_DEFAULTS,...options}, telemetry=state.telemetry, target=state.landmarks.find(x=>x.id===task.landmarkId);
  if(!target) return {ok:false,reason:'target_missing'};
  if(target.review?.status==='rejected') return {ok:false,reason:'target_rejected'};
  if(target.mapRevision!==state.mapRevision || task.mapRevision!==state.mapRevision) return {ok:false,reason:'map_revision_changed_reacquire_target'};
  if(!vec(target.positionEnuM) || !Number.isFinite(target.positionStdM) || target.positionStdM>config.maxTargetStdM) return {ok:false,reason:'target_position_unverified'};
  // A stationary mushroom landmark may outlive one slow inference cycle, but
  // navigation/depth must remain fresh and the target must be reacquired soon.
  if(!fresh(target.lastSeenMs,nowMs,{...config,maxSensorAgeMs:config.maxTargetAgeMs})) return {ok:false,reason:'target_observation_stale'};
  if(!telemetry?.localized || !vec(telemetry.positionEnuM) || !fresh(telemetry.timestampSeconds*1000,nowMs,config)) return {ok:false,reason:'fresh_localization_required'};
  if(!telemetry.depthValid || !fresh(telemetry.depthTimestampSeconds*1000,nowMs,config)) return {ok:false,reason:'fresh_depth_required'};
  if(!Number.isFinite(telemetry.batteryPercent) || telemetry.batteryPercent<config.minBatteryPercent) return {ok:false,reason:'battery_reserve_required'};
  if(![telemetry.speedMps,telemetry.yawRad,telemetry.pitchRad].every(Number.isFinite) || telemetry.speedMps<0) return {ok:false,reason:'motion_and_camera_aim_required'};
  if(!state.voxelMap || state.voxelMap.mapRevision!==state.mapRevision || !fresh(state.voxelMap.updatedMs,nowMs,config)) return {ok:false,reason:'fresh_observed_free_space_required'};
  return {ok:true,config,telemetry,target};
}

export function validateCaptureAck(ack, task, state, nowMs, options={}) {
  const config={...LIVE_EXECUTOR_DEFAULTS,...options}, request=task.captureRequest;
  const fail=reason=>({accepted:false,reason});
  if(!request || !ack || ack.requestId!==request.requestId || ack.taskId!==task.id || ack.sessionId!==state.sessionId || ack.landmarkId!==task.landmarkId || ack.mapRevision!==state.mapRevision) return fail('capture_identity_mismatch');
  if(state.mode==='live' && ack.synthetic===true) return fail('synthetic_capture_forbidden_in_live_session');
  if(state.mode==='replay' && ack.synthetic!==true) return fail('replay_capture_must_be_marked_synthetic');
  if(!Number.isFinite(ack.capturedMs) || ack.capturedMs<request.requestedMs || ack.capturedMs>nowMs+config.maxClockSkewMs || nowMs-ack.capturedMs>config.captureTimeoutMs) return fail('capture_timestamp_invalid');
  if(!/^[a-f0-9]{64}$/.test(ack.photoSha256??'') || typeof ack.photoRef!=='string' || ack.photoRef.length<1 || ack.photoRef.length>1000 || !ack.nativeUncropped || !Number.isInteger(ack.width) || ack.width<1 || !Number.isInteger(ack.height) || ack.height<1) return fail('saved_native_photo_evidence_required');
  if(!vec(ack.positionEnuM) || distance(xyz(ack.positionEnuM),xyz(request.positionEnuM))>config.arrivalToleranceM) return fail('capture_pose_changed');
  if(request.cameraPositionEnuM&&(!vec(ack.cameraPositionEnuM)||distance(xyz(ack.cameraPositionEnuM),xyz(request.cameraPositionEnuM))>config.arrivalToleranceM))return fail('capture_camera_pose_changed');
  if(!Number.isFinite(ack.yawRad) || !Number.isFinite(ack.pitchRad) || angleDifference(ack.yawRad,request.aim.yawRad)>config.aimToleranceRad || angleDifference(ack.pitchRad,request.aim.pitchRad)>config.aimToleranceRad) return fail('capture_aim_unverified');
  if(ack.settled!==true || ack.sharpnessAccepted!==true) return fail('capture_quality_unverified');
  const box=ack.targetBboxPixelsXYXY;
  const valid=Array.isArray(box)&&box.length===4&&box.every(Number.isFinite)&&box[0]>=0&&box[1]>=0&&box[2]>box[0]&&box[3]>box[1]&&box[2]<=ack.width&&box[3]<=ack.height;
  const fillFraction=valid?(box[2]-box[0])/ack.width:null;
  const centered=valid&&Math.abs((box[0]+box[2])/2-ack.width/2)<=ack.width*.1&&Math.abs((box[1]+box[3])/2-ack.height/2)<=ack.height*.1;
  return {accepted:true,synthetic:ack.synthetic===true,photoSha256:ack.photoSha256,photoRef:ack.photoRef,capturedMs:ack.capturedMs,fillFraction,
    framingSatisfied:centered&&fillFraction>=config.targetFillFraction,independentlyVerified:false,humanConfirmationRequired:true};
}

/** Mutates only the selected task; returns a proposal consumed by simulation or a
 * future separately verified controller. A live session cannot actuate anything. */
export function stepLiveExecutor(state, {nowMs,captureAck=null,options={}}={}) {
  if(!Number.isFinite(nowMs)) throw new Error('Executor nowMs must be finite');
  const tasks=state.tasks.filter(t=>!taskTerminal(t)&&t.status!=='paused'&&t.status!=='blocked');
  const task=tasks.find(t=>t.status==='running')??tasks.find(t=>t.status==='queued');
  if(!task) return {type:'idle',actuationEnabled:false,reason:'no_runnable_task'};
  const pre=inspectPreconditions(state,task,nowMs,options);
  if(!pre.ok) return hold(task,pre.reason,nowMs);
  const {config,telemetry,target}=pre;
  task.status='running'; task.updatedMs=nowMs; task.blockedReason=null;
  const from=xyz(telemetry.positionEnuM), targetPosition=xyz(target.positionEnuM);
  // Inspection is at the current observed flight altitude. No blind descent to
  // a ground mushroom, and the aircraft never occupies the target itself.
  if(!task.viewpointEnuM) {
    const dx=from.x-targetPosition.x,dy=from.y-targetPosition.y,len=Math.hypot(dx,dy);
    if(len<1e-6) return hold(task,'horizontal_approach_direction_unknown',nowMs);
    const horizontal=Math.sqrt(Math.max(0,config.standoffM**2-(from.z-targetPosition.z)**2));
    if(horizontal<.25) return hold(task,'camera_viewpoint_requires_operator',nowMs);
    const desired={x:targetPosition.x+dx/len*horizontal,y:targetPosition.y+dy/len*horizontal,z:from.z};
    task.viewpointEnuM=array(distance(from,targetPosition)<=config.standoffM?from:desired);
    task.phase='approach';
  }
  const goal=xyz(task.viewpointEnuM);
  if(distance(from,targetPosition)<config.standoffM-config.arrivalToleranceM) return hold(task,'target_inside_minimum_standoff',nowMs);
  if(distance(from,goal)>config.arrivalToleranceM) {
    task.settledSinceMs=null; task.phase='approach';
    const plan=planCorridor3D(state.voxelMap,from,goal,{radiusM:config.radiusM+3*target.positionStdM,nowMs,maxVoxelAgeMs:config.maxSensorAgeMs,maxClockSkewMs:config.maxClockSkewMs,maxExpanded:2000,allowVertical:false});
    if(plan.status!=='ready') return hold(task,plan.reason,nowMs);
    task.route=plan.points.map(array);
    return {type:'move_proposal',taskId:task.id,waypointsEnuM:task.route,simulationOnly:true,actuationEnabled:false};
  }
  const clearance=sweptClearance(state.voxelMap,from,from,{radiusM:config.radiusM,nowMs,maxVoxelAgeMs:config.maxSensorAgeMs,maxClockSkewMs:config.maxClockSkewMs});
  if(!clearance.clear) return hold(task,clearance.reason,nowMs);
  const cameraPosition=vec(telemetry.cameraPositionEnuM)?xyz(telemetry.cameraPositionEnuM):from;
  const aim=cameraAim(cameraPosition,targetPosition);
  if(telemetry.speedMps>config.settledSpeedMps || angleDifference(telemetry.yawRad,aim.yawRad)>config.aimToleranceRad || angleDifference(telemetry.pitchRad,aim.pitchRad)>config.aimToleranceRad) {
    task.phase='aim';task.settledSinceMs=null;
    return {type:'aim_proposal',taskId:task.id,aim,actuationEnabled:false};
  }
  if(task.settledSinceMs==null) task.settledSinceMs=nowMs;
  if(nowMs-task.settledSinceMs<config.settleMs) {
    task.phase='settle';return {type:'settle',taskId:task.id,remainingMs:config.settleMs-(nowMs-task.settledSinceMs),actuationEnabled:false};
  }
  if(!task.captureRequest) {
    task.phase='capture';
    task.captureRequest={requestId:`capture-${task.id}-${nowMs}`,taskId:task.id,sessionId:state.sessionId,landmarkId:task.landmarkId,mapRevision:state.mapRevision,requestedMs:nowMs,positionEnuM:array(from),...(vec(telemetry.cameraPositionEnuM)?{cameraPositionEnuM:[...telemetry.cameraPositionEnuM]}:{}),aim,nativeUncropped:true,targetFillFraction:config.targetFillFraction};
    return {type:'capture_request',...task.captureRequest,actuationEnabled:false};
  }
  if(captureAck) {
    const result=validateCaptureAck(captureAck,task,state,nowMs,options);
    if(!result.accepted) {task.lastCaptureRejection=result.reason;return {type:'capture_rejected',taskId:task.id,reason:result.reason,actuationEnabled:false};}
    task.capture=result;task.status='completed';task.phase='review';task.completedMs=nowMs;
    task.result=result.framingSatisfied?'photo_ready_for_human_review':'photo_saved_closer_framing_requires_optical_zoom_or_operator';
    return {type:'completed',taskId:task.id,result:task.result,capture:result,actuationEnabled:false};
  }
  if(nowMs-task.captureRequest.requestedMs>config.captureTimeoutMs) return hold(task,'capture_ack_timeout',nowMs);
  return {type:'awaiting_capture_ack',taskId:task.id,requestId:task.captureRequest.requestId,actuationEnabled:false};
}

export function createSimulationPacket({sessionId='simulation-demo',sequence=0,timeSeconds=100,positionEnuM=[-4,0,1.5],yawRad=0,pitchRad=0,blocked=false}={}) {
  const voxels={},size={x:24,y:16,z:12},origin={x:-6,y:-4,z:-1};
  for(let z=0;z<size.z;z++)for(let y=0;y<size.y;y++)for(let x=0;x<size.x;x++)voxels[`${x},${y},${z}`]={state:blocked&&x===7?'occupied':'free',observedMs:timeSeconds*1000};
  const rangeM=distance(xyz(positionEnuM),{x:1,y:0,z:0});
  return {schema:'mycoscout.live-frame.v1',sessionId,frameId:`sim-frame-${sequence}`,sequence,timestampSeconds:timeSeconds,simulationOnly:true,coordinateFrame:'map-enu',mapRevision:'simulation-map-1',actuationEnabled:false,
    image:{width:640,height:480,sha256:'0'.repeat(64),mimeType:'image/jpeg'},
    camera:{intrinsics:{fx:500,fy:500,cx:320,cy:240},cameraToMap4x4:[[1,0,0,positionEnuM[0]],[0,0,1,positionEnuM[1]],[0,-1,0,positionEnuM[2]],[0,0,0,1]],poseTimestampSeconds:timeSeconds,positionStdM:.02,orientationStdRad:.01,calibrationId:'synthetic-calibration'},
    depth:{timestampSeconds:timeSeconds,calibrationId:'synthetic-calibration',valid:true,method:'rgbd'},
    detections:[{detectionId:`d-${sequence}`,bboxPixelsXYXY:[290,210,350,270],confidence:.85,trackId:'sim-track-1',landmarkId:'sim-mushroom-1',positionEnuM:[1,0,0],positionStdM:.02,rangeM,locationStatus:'located'}],
    tracks:[{trackId:'sim-track-1',landmarkId:'sim-mushroom-1',bboxPixelsXYXY:[290,210,350,270],state:'confirmed',observationCount:sequence+1,lastSeenSeconds:timeSeconds}],
    landmarks:[{landmarkId:'sim-mushroom-1',trackId:'sim-track-1',positionEnuM:[1,0,0],positionStdM:.02,gps:null,rangeM,inspectionPriority:'immediate',humanVerified:false}],mapPoints:[{positionEnuM:[1,0,0]}],
    flight:{telemetry:{positionEnuM,timestampSeconds:timeSeconds,batteryPercent:85,speedMps:0,yawRad,pitchRad,localized:true,depthValid:true,depthTimestampSeconds:timeSeconds},voxelMap:{frame:'map_enu',unit:'m',mapRevision:'simulation-map-1',origin,resolutionM:.5,size,updatedMs:timeSeconds*1000,voxels}}};
}

export function createSyntheticCaptureAck(state, task, nowMs) {
  if(state.mode!=='replay' || !state.simulationDemo) throw new Error('Synthetic acknowledgments require an explicit simulation demo');
  const request=task.captureRequest;
  return {...request,capturedMs:nowMs,synthetic:true,photoSha256:'f'.repeat(64),photoRef:'simulation://synthetic-photo-no-real-image',nativeUncropped:true,width:640,height:480,positionEnuM:[...request.positionEnuM],yawRad:request.aim.yawRad,pitchRad:request.aim.pitchRad,settled:true,sharpnessAccepted:true,targetBboxPixelsXYXY:[80,60,560,420]};
}
