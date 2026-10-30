import { DatabaseSync } from 'node:sqlite';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { ApiError, shortText } from './validation.mjs';
import { validateVoxelMap } from '../flight/geometry.mjs';
import { createSimulationPacket, createSyntheticCaptureAck, stepLiveExecutor, taskTerminal } from '../flight/live-executor.mjs';

const ID=/^[A-Za-z0-9_.:-]{1,160}$/, HASH=/^[a-f0-9]{64}$/;
const limits={frames:120,landmarks:1000,tasks:300,taskKeys:10000,events:300,mapPoints:20000,frameIds:100000,sessions:64};
const copy=value=>structuredClone(value);
const vector=value=>Array.isArray(value)&&value.length===3&&value.every(n=>Number.isFinite(n)&&Math.abs(n)<1e7);
const validId=(value,name)=>{if(typeof value!=='string'||!ID.test(value))throw new ApiError(`Invalid ${name}.`);return value;};
const number=(value,name,min=0,max=1e15)=>{if(typeof value!=='number'||!Number.isFinite(value)||value<min||value>max)throw new ApiError(`Invalid ${name}.`);return value;};
const optionalText=(value,name,max=1000)=>value==null||value===''?'':shortText(value,name,undefined,max);
const revision=(state,input)=>{
  if(input.sessionId!==undefined&&input.sessionId!==state.sessionId)throw new ApiError('The selected mission session changed.',409,'LIVE_SESSION_MISMATCH');
  const useControl=input.baseControlRevision!==undefined,value=useControl?input.baseControlRevision:input.baseRevision,expected=useControl?(state.controlRevision??0):state.revision;
  if(value!==undefined&&(!Number.isInteger(value)||value!==expected))throw new ApiError('Mission changed. Reload before changing the queue.',409,'LIVE_REVISION_CONFLICT');
};
const touchControl=state=>{state.controlRevision=(state.controlRevision??0)+1;};
const taskControls=state=>JSON.stringify(state.tasks.map(t=>[t.id,t.status,t.phase,t.blockedReason??null]));
const fail=(message,code='LIVE_INVALID_PACKET',status=400)=>{throw new ApiError(message,status,code);};
const event=(state,type,details,nowMs)=>{state.events.push({id:randomUUID(),type,atMs:nowMs,...details});state.events=state.events.slice(-limits.events);};
const emptyState=(sessionId,mode,nowMs)=>({schema:'mycoscout.live-state.v1',sessionId,mode,revision:0,controlRevision:0,createdAtMs:nowMs,updatedAtMs:nowMs,actuationEnabled:false,realFlightSupported:false,simulationDemo:false,mapRevision:null,lastSequence:-1,lastCaptureSeconds:null,lastPoseSeconds:null,receivedAtMs:null,navigationReceivedAtMs:null,lastNavigationSeconds:null,latestFrame:null,frames:[],tracks:[],landmarks:[],tasks:[],events:[],telemetry:null,voxelMap:null,map:{coordinateFrame:'map-enu',points:[],clearanceVerified:false},lastAction:null});

function cameraEvidence(packet) {
  const camera=packet.camera,depth=packet.depth;
  if(!camera||!depth||depth.valid!==true||!['rgbd','stereo','surveyed'].includes(depth.method))return false;
  if(typeof camera.calibrationId!=='string'||!camera.calibrationId||depth.calibrationId!==camera.calibrationId)return false;
  if(!Number.isFinite(camera.poseTimestampSeconds)||Math.abs(camera.poseTimestampSeconds-packet.timestampSeconds)>.05||!Number.isFinite(depth.timestampSeconds)||Math.abs(depth.timestampSeconds-packet.timestampSeconds)>.05)return false;
  if(!Number.isFinite(camera.positionStdM)||camera.positionStdM<0||!Number.isFinite(camera.orientationStdRad)||camera.orientationStdRad<0)return false;
  const k=camera.intrinsics;if(!k||![k.fx,k.fy,k.cx,k.cy].every(Number.isFinite)||k.fx<=0||k.fy<=0)return false;
  const matrix=camera.cameraToMap4x4;
  if(!Array.isArray(matrix)||matrix.length!==4||matrix.some(row=>!Array.isArray(row)||row.length!==4||!row.every(Number.isFinite)))return false;
  if(matrix[3].some((v,i)=>Math.abs(v-(i===3?1:0))>1e-6))return false;
  for(let a=0;a<3;a++)for(let b=0;b<3;b++)if(Math.abs(matrix[a].slice(0,3).reduce((n,v,i)=>n+v*matrix[b][i],0)-(a===b?1:0))>1e-3)return false;
  const m=matrix,det=m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
  return Math.abs(det-1)<1e-3;
}

function validGeoreference(packet) {
  const g=packet.georeference,a=g?.anchor;
  return a&&Number.isFinite(a.latitude)&&Math.abs(a.latitude)<=90&&Number.isFinite(a.longitude)&&Math.abs(a.longitude)<=180&&Number.isFinite(a.altitudeM)&&Number.isFinite(g.horizontalStdM)&&g.horizontalStdM>0&&Number.isFinite(g.headingStdRad)&&g.headingStdRad>=0&&typeof g.sourceId==='string'&&g.sourceId.length>0;
}

/** Canonical packet validation deliberately refuses to infer metric location or
 * free volume from detector boxes, RGB pixels or sparse surface-map points. */
export function validateLivePacket(input, mode) {
  if(!input||input.schema!=='mycoscout.live-frame.v1'||input.coordinateFrame!=='map-enu'||input.actuationEnabled!==false)fail('Use a non-actuating mycoscout.live-frame.v1 packet in map-enu coordinates.');
  const p=copy(input);validId(p.sessionId,'session ID');validId(p.frameId,'frame ID');validId(p.mapRevision,'map revision');
  if(!Number.isSafeInteger(p.sequence)||p.sequence<0)fail('Frame sequence must be a nonnegative safe integer.');
  number(p.timestampSeconds,'capture timestamp',0,1e12);
  if(typeof p.simulationOnly!=='boolean'||p.simulationOnly!==(mode==='replay'))fail('Frame mode must match the active live or replay session.','LIVE_MODE_MISMATCH',409);
  if(!p.image||!Number.isInteger(p.image.width)||p.image.width<1||p.image.width>16384||!Number.isInteger(p.image.height)||p.image.height<1||p.image.height>16384||!HASH.test(p.image.sha256)||!['image/jpeg','image/png','image/webp'].includes(p.image.mimeType))fail('Frame image dimensions, SHA-256 and supported MIME type are required.');
  // HTTP ingestion owns storage and byte/hash verification. Do not persist large
  // base64 images in mission state; retain its reviewed local URL and digest.
  delete p.image.imageBase64;
  if(p.image.url!==undefined&&(!/^\/api\/live\/images\/[A-Za-z0-9_.-]+$/.test(p.image.url)&&!/^\/live-images\/[A-Za-z0-9_.-]+$/.test(p.image.url)))fail('Frame image URL must reference the local live-image store.');
  for(const field of ['detections','tracks','landmarks','mapPoints'])if(!Array.isArray(p[field])||p[field].length>(field==='mapPoints'?limits.mapPoints:1000))fail(`Invalid or oversized ${field}.`);
  const evidence=cameraEvidence(p),geo=validGeoreference(p),ids=new Set();
  if(p.camera&&(!Number.isFinite(p.camera.poseTimestampSeconds)||Math.abs(p.camera.poseTimestampSeconds-p.timestampSeconds)>.05))fail('Camera pose must be synchronized with the captured frame.');
  for(const d of p.detections){
    validId(d.detectionId,'detection ID');const b=d.bboxPixelsXYXY;
    if(!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||b[0]<0||b[1]<0||b[2]>p.image.width||b[3]>p.image.height||b[2]<=b[0]||b[3]<=b[1])fail('Detection boxes must be inside the native frame.');
    number(d.confidence,'detection confidence',0,1);
  }
  for(const landmark of p.landmarks){
    validId(landmark.landmarkId,'landmark ID');if(ids.has(landmark.landmarkId))fail('Duplicate landmark ID in frame.');ids.add(landmark.landmarkId);
    if(landmark.positionEnuM!=null&&(!evidence||!vector(landmark.positionEnuM)||!Number.isFinite(landmark.positionStdM)||landmark.positionStdM<0))fail('Metric landmarks require synchronized calibrated depth and camera pose.');
    if(landmark.rangeM!=null){
      number(landmark.rangeM,'landmark range',0,100000);
      if(landmark.positionEnuM!=null){const pose=p.camera.cameraToMap4x4,range=Math.hypot(...landmark.positionEnuM.map((v,i)=>v-pose[i][3]));
        if(Math.abs(range-landmark.rangeM)>Math.max(.05,3*(landmark.positionStdM+p.camera.positionStdM)))fail('Landmark range disagrees with synchronized camera geometry.');}
    }
    if(landmark.gps!=null){const g=landmark.gps;if(!evidence||!geo||!vector(landmark.positionEnuM)||![g.latitude,g.longitude,g.altitudeM,g.horizontalUncertaintyM].every(Number.isFinite)||Math.abs(g.latitude)>90||Math.abs(g.longitude)>180||g.horizontalUncertaintyM<=0)fail('GPS landmarks require georeferencing evidence and explicit uncertainty.');}
  }
  for(const pt of p.mapPoints)if(!evidence||!vector(pt.positionEnuM))fail('Map surface points require synchronized metric depth and pose.');
  if(p.flight){
    const t=p.flight.telemetry;
    if(!t||!vector(t.positionEnuM)||!Number.isFinite(t.timestampSeconds)||Math.abs(t.timestampSeconds-p.timestampSeconds)>.05||!Number.isFinite(t.batteryPercent)||t.batteryPercent<0||t.batteryPercent>100||![t.speedMps,t.yawRad,t.pitchRad].every(Number.isFinite)||t.speedMps<0||typeof t.localized!=='boolean'||typeof t.depthValid!=='boolean')fail('Flight telemetry must be complete and synchronized.');
    if(t.depthValid&&(!evidence||!Number.isFinite(t.depthTimestampSeconds)||Math.abs(t.depthTimestampSeconds-p.timestampSeconds)>.05))fail('Flight depth evidence must be synchronized.');
    const map=p.flight.voxelMap;
    if(map){
      try{validateVoxelMap(map);}catch(error){fail(error.message);}
      if(map.mapRevision!==p.mapRevision||Object.keys(map.voxels).length>100000||Math.abs(map.updatedMs-p.timestampSeconds*1000)>50)fail('Occupancy map must match this capture clock and map revision.');
      for(const [key,cell] of Object.entries(map.voxels)){
        const index=key.split(',').map(Number);
        if(index.length!==3||index.some((v,i)=>!Number.isInteger(v)||v<0||v>=map.size[['x','y','z'][i]])||!['free','occupied'].includes(cell.state)||!Number.isFinite(cell.observedMs)||cell.observedMs>map.updatedMs+30)fail('Invalid observed voxel evidence.');
      }
    }
  }
  return p;
}

export class LiveMissionStore {
  constructor(directory,{mode='replay',now=()=>Date.now(),executorOptions={}}={}){this.directory=resolve(directory);this.defaultMode=mode;this.now=now;this.executorOptions=executorOptions;this.db=null;}
  async initialize(){
    await mkdir(this.directory,{recursive:true});this.db=new DatabaseSync(resolve(this.directory,'live-mission.sqlite'));
    const version=this.db.prepare('PRAGMA user_version').get().user_version;if(![0,1].includes(version))throw new Error('Unsupported mission database version');
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS live_sessions(id TEXT PRIMARY KEY,data TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS live_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS live_frames(session_id TEXT NOT NULL,frame_id TEXT NOT NULL,hash TEXT NOT NULL,sequence INTEGER NOT NULL,PRIMARY KEY(session_id,frame_id),UNIQUE(session_id,sequence));
      CREATE TABLE IF NOT EXISTS live_idempotency(session_id TEXT NOT NULL,key TEXT NOT NULL,payload TEXT NOT NULL,task_id TEXT NOT NULL,PRIMARY KEY(session_id,key));
      PRAGMA user_version=1;`);return this;
  }
  close(){this.db?.close();this.db=null;}
  transaction(fn){this.db.exec('BEGIN IMMEDIATE');try{const result=fn();this.db.exec('COMMIT');return result;}catch(error){this.db.exec('ROLLBACK');throw error;}}
  active(){const id=this.db.prepare("SELECT value FROM live_meta WHERE key='active'").get()?.value;return id?{controlRevision:0,...JSON.parse(this.db.prepare('SELECT data FROM live_sessions WHERE id=?').get(id).data)}:null;}
  requireActive(){const state=this.active();if(!state)throw new ApiError('Start a replay or live session first.',409,'LIVE_SESSION_REQUIRED');return state;}
  save(state){state.revision++;state.updatedAtMs=this.now();this.db.prepare('INSERT INTO live_sessions(id,data) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET data=excluded.data').run(state.sessionId,JSON.stringify(state));}
  snapshot(){const state=this.active(),sessions=this.db.prepare('SELECT id,data FROM live_sessions').all().map(row=>{const s=JSON.parse(row.data);return {sessionId:row.id,mode:s.mode,revision:s.revision,updatedAtMs:s.updatedAtMs,simulationDemo:s.simulationDemo};});return {...(state??emptyState(null,this.defaultMode,this.now())),sessions,capabilities:{realFlight:false,hardwareActuation:false,simulatedExecutor:true,metricMappingFromDepth:true,ordinaryRgbProvidesMetricMap:false},connection:{connected:state?.receivedAtMs!=null&&this.now()-state.receivedAtMs<3000,ageMs:state?.receivedAtMs==null?null:Math.max(0,this.now()-state.receivedAtMs),captureAgeMs:state?.mode==='live'&&state.lastCaptureSeconds!=null?Math.max(0,this.now()-state.lastCaptureSeconds*1000):null}};}
  startSession(input){
    validId(input.sessionId,'session ID');if(!['replay','live'].includes(input.mode))throw new ApiError('Session mode must be replay or live.');
    return this.transaction(()=>{
      const active=this.active();if(active)revision(active,{...input,sessionId:undefined});
      const row=this.db.prepare('SELECT data FROM live_sessions WHERE id=?').get(input.sessionId);
      if(row){const old=JSON.parse(row.data);if(old.mode!==input.mode)throw new ApiError('A session cannot change between replay and live.',409,'LIVE_MODE_MISMATCH');}
      if(!row&&this.db.prepare('SELECT COUNT(*) AS n FROM live_sessions').get().n>=limits.sessions)throw new ApiError('Session capacity reached. Archive this database before collecting more.',409,'LIVE_CAPACITY');
      if(active&&active.sessionId!==input.sessionId){for(const task of active.tasks)if(task.status==='running'){task.status='paused';task.pauseReason='session_changed';}touchControl(active);this.save(active);}
      const state=row?JSON.parse(row.data):emptyState(input.sessionId,input.mode,this.now());
      if(!row)this.save(state);
      this.db.prepare("INSERT INTO live_meta(key,value) VALUES('active',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").run(input.sessionId);
      return this.snapshot();
    });
  }
  makeTask(state,input){
    const target=state.landmarks.find(l=>l.id===input.landmarkId);if(!target)throw new ApiError('Choose a landmark from the active session.',404,'LIVE_TARGET_NOT_FOUND');
    if(!['inspect','photograph','scan'].includes(input.type))throw new ApiError('Task type must be inspect, photograph or scan.');
    if(state.tasks.length>=limits.tasks){const removable=state.tasks.findIndex(taskTerminal);if(removable<0)throw new ApiError('Task queue is full.',409,'LIVE_CAPACITY');state.tasks.splice(removable,1);}
    const task={id:randomUUID(),sessionId:state.sessionId,landmarkId:target.id,mapRevision:target.mapRevision,type:input.type,status:'queued',phase:'pending',source:input.source??'operator',createdAtMs:this.now(),updatedMs:state.lastCaptureSeconds*1000,actuationEnabled:false};
    state.tasks.push(task);touchControl(state);event(state,'task_queued',{taskId:task.id,landmarkId:target.id,type:task.type},this.now());return task;
  }
  enqueue(input){return this.transaction(()=>{
    const state=this.requireActive(),key=validId(input.idempotencyKey,'idempotency key'),payload=JSON.stringify({type:input.type,landmarkId:input.landmarkId});
    const existing=this.db.prepare('SELECT * FROM live_idempotency WHERE session_id=? AND key=?').get(state.sessionId,key);
    if(existing){if(existing.payload!==payload)throw new ApiError('This request key belongs to a different task.',409,'LIVE_IDEMPOTENCY_CONFLICT');return {task:state.tasks.find(t=>t.id===existing.task_id)??{id:existing.task_id,status:'archived'},duplicate:true,state:this.snapshot()};}
    if(this.db.prepare('SELECT COUNT(*) AS n FROM live_idempotency WHERE session_id=?').get(state.sessionId).n>=limits.taskKeys)throw new ApiError('Task request history is full. Start a new session.',409,'LIVE_CAPACITY');
    revision(state,input);const task=this.makeTask(state,input);this.db.prepare('INSERT INTO live_idempotency(session_id,key,payload,task_id) VALUES(?,?,?,?)').run(state.sessionId,key,payload,task.id);this.save(state);return {task,duplicate:false,state:this.snapshot()};
  });}
  ingestFrame(input){return this.transaction(()=>{
    const state=this.requireActive(),packet=validateLivePacket(input,state.mode);
    if(packet.sessionId!==state.sessionId)throw new ApiError('Select this session before ingesting its frames.',409,'LIVE_SESSION_MISMATCH');
    if(state.mode==='live'&&(packet.timestampSeconds<1e9||packet.timestampSeconds*1000>this.now()+100))throw new ApiError('Live capture timestamps must use synchronized UTC epoch seconds and cannot be in the future.',409,'LIVE_CLOCK_INVALID');
    const hash=createHash('sha256').update(JSON.stringify(packet)).digest('hex'),known=this.db.prepare('SELECT hash FROM live_frames WHERE session_id=? AND frame_id=?').get(state.sessionId,packet.frameId);
    if(known){if(known.hash!==hash)throw new ApiError('This frame ID already describes different evidence.',409,'LIVE_FRAME_CONFLICT');return {duplicate:true,state:this.snapshot()};}
    if(packet.sequence<=state.lastSequence||state.lastCaptureSeconds!==null&&packet.timestampSeconds<=state.lastCaptureSeconds)throw new ApiError('Frames must arrive in capture order.',409,'LIVE_OUT_OF_ORDER');
    if(packet.camera&&state.lastPoseSeconds!==null&&packet.camera.poseTimestampSeconds<state.lastPoseSeconds)throw new ApiError('Camera pose moved backwards in time.',409,'LIVE_STALE_POSE');
    if(this.db.prepare('SELECT COUNT(*) AS n FROM live_frames WHERE session_id=?').get(state.sessionId).n>=limits.frameIds)throw new ApiError('Start a new session: frame history capacity reached.',409,'LIVE_CAPACITY');
    const previousMap=state.mapRevision;state.mapRevision=packet.mapRevision;
    if(previousMap&&previousMap!==packet.mapRevision){
      touchControl(state);state.telemetry=null;state.voxelMap=null;state.lastNavigationSeconds=null;state.navigationReceivedAtMs=null;
      for(const task of state.tasks)if(!taskTerminal(task)){task.status='blocked';task.blockedReason='map_revision_changed_reacquire_target';}
      // Without an explicit loop-closure transform, old metric coordinates must
      // not be drawn in the new map or reused for an approach.
      for(const landmark of state.landmarks){
        if(landmark.positionEnuM)landmark.previousLocation={mapRevision:landmark.mapRevision,positionEnuM:landmark.positionEnuM,positionStdM:landmark.positionStdM,gps:landmark.gps,lastSeenMs:landmark.lastSeenMs};
        Object.assign(landmark,{positionEnuM:null,positionStdM:null,gps:null,rangeM:null,inspectionPriority:'unlocated',locationStatus:'map_revision_changed'});
      }
      event(state,'map_revision_changed',{previousMap,mapRevision:packet.mapRevision},this.now());
    }
    state.lastSequence=packet.sequence;state.lastCaptureSeconds=packet.timestampSeconds;state.lastPoseSeconds=packet.camera?.poseTimestampSeconds??state.lastPoseSeconds;state.receivedAtMs=this.now();
    state.latestFrame={frameId:packet.frameId,sequence:packet.sequence,timestampSeconds:packet.timestampSeconds,receivedAtMs:state.receivedAtMs,simulationOnly:packet.simulationOnly,image:packet.image,detections:packet.detections,camera:packet.camera??null,detectorStatus:packet.detectorStatus??null,provenance:packet.provenance??null};
    state.frames.push({frameId:packet.frameId,timestampSeconds:packet.timestampSeconds,image:packet.image});state.frames=state.frames.slice(-limits.frames);state.tracks=packet.tracks;
    state.map={coordinateFrame:'map-enu',mapRevision:packet.mapRevision,points:packet.mapPoints,clearanceVerified:false,georeference:packet.georeference??null};
    // Slow semantic frames cannot overwrite newer independent navigation data.
    if(state.lastNavigationSeconds==null||packet.timestampSeconds>=state.lastNavigationSeconds){
      state.telemetry=packet.flight?.telemetry??null;state.voxelMap=packet.flight?.voxelMap??null;
      if(packet.flight){state.lastNavigationSeconds=packet.timestampSeconds;state.navigationReceivedAtMs=this.now();}
    }
    for(const incoming of packet.landmarks){
      const existing=state.landmarks.find(l=>l.id===incoming.landmarkId),hasPosition=vector(incoming.positionEnuM),upper=hasPosition&&Number.isFinite(incoming.rangeM)?incoming.rangeM+3*incoming.positionStdM:null;
      const priority=!hasPosition||upper===null?'unlocated':incoming.inspectionPriority==='habitat'||upper>=50||incoming.positionStdM>.75?'habitat':'immediate';
      const landmark={...incoming,id:incoming.landmarkId,sessionId:state.sessionId,mapRevision:packet.mapRevision,inspectionPriority:priority,humanVerified:existing?.review?.status==='confirmed',review:existing?.review??null,firstSeenMs:existing?.firstSeenMs??packet.timestampSeconds*1000,lastSeenMs:packet.timestampSeconds*1000,observationCount:(existing?.observationCount??0)+1,latestFrameId:packet.frameId,latestImageUrl:packet.image.url??null};
      if(existing)Object.assign(existing,landmark);else{if(state.landmarks.length>=limits.landmarks)throw new ApiError('Landmark capacity reached. Start a new session.',409,'LIVE_CAPACITY');state.landmarks.push(landmark);}
      if(priority==='immediate'&&landmark.review?.status!=='rejected'&&!state.tasks.some(t=>t.landmarkId===landmark.id))this.makeTask(state,{type:'inspect',landmarkId:landmark.id,source:'nearby_candidate'});
    }
    this.db.prepare('INSERT INTO live_frames(session_id,frame_id,hash,sequence) VALUES(?,?,?,?)').run(state.sessionId,packet.frameId,hash,packet.sequence);this.save(state);return {duplicate:false,state:this.snapshot()};
  });}
  updateNavigation(input){return this.transaction(()=>{
    const state=this.requireActive(),p=copy(input),t=p.telemetry,map=p.voxelMap,evidence=map?.evidence;
    if(p.schema!=='mycoscout.live-navigation.v1'||p.sessionId!==state.sessionId||p.simulationOnly!==(state.mode==='replay'))throw new ApiError('Navigation packet must match the active session and mode.',409,'LIVE_SESSION_MISMATCH');
    validId(p.mapRevision,'map revision');number(p.timestampSeconds,'navigation timestamp',0,1e12);
    if(state.mapRevision!==null&&state.mapRevision!==p.mapRevision)throw new ApiError('Navigation belongs to another map revision. Reacquire semantic observations in the new map first.',409,'LIVE_MAP_MISMATCH');
    if(state.mode==='live'&&(p.timestampSeconds<1e9||this.now()-p.timestampSeconds*1000>1500||p.timestampSeconds*1000>this.now()+100))throw new ApiError('Navigation requires fresh synchronized UTC capture timestamps.',409,'LIVE_NAVIGATION_STALE');
    const digest=createHash('sha256').update(JSON.stringify(p)).digest('hex');
    if(state.lastNavigationSeconds!=null&&p.timestampSeconds<=state.lastNavigationSeconds){
      if(p.timestampSeconds===state.lastNavigationSeconds&&state.navigationDigest===digest)return {duplicate:true,state:this.snapshot()};
      throw new ApiError('Navigation poses must arrive in capture order.',409,'LIVE_OUT_OF_ORDER');
    }
    if(!t||!vector(t.positionEnuM)||!vector(t.cameraPositionEnuM)||!['rigid_calibrated','measured_gimbal'].includes(t.cameraMountMode)||![t.timestampSeconds,t.depthTimestampSeconds,t.batteryPercent,t.speedMps,t.yawRad,t.pitchRad].every(Number.isFinite)||t.batteryPercent<0||t.batteryPercent>100||t.speedMps<0||typeof t.localized!=='boolean'||typeof t.depthValid!=='boolean'||Math.abs(t.timestampSeconds-p.timestampSeconds)>.05||Math.abs(t.depthTimestampSeconds-p.timestampSeconds)>.05)throw new ApiError('Navigation needs synchronized body pose, calibrated camera aim, depth and battery telemetry.');
    try{validateVoxelMap(map);}catch(error){throw new ApiError(error.message);}
    if(map.mapRevision!==p.mapRevision||Object.keys(map.voxels).length>100000||Math.abs(map.updatedMs-p.timestampSeconds*1000)>50)throw new ApiError('Navigation occupancy must match the capture timestamp and map revision.');
    if(evidence?.schema!=='mycoscout.observed-depth-volume.v1'||evidence.sessionId!==state.sessionId||evidence.freeSpaceMethod!=='full_projected_voxel_footprint'||typeof evidence.calibrationId!=='string'||!evidence.calibrationId||!Number.isFinite(evidence.depthTimestampSeconds)||!Number.isFinite(evidence.poseTimestampSeconds)||Math.abs(evidence.depthTimestampSeconds-p.timestampSeconds)>.05||Math.abs(evidence.poseTimestampSeconds-p.timestampSeconds)>.05||![evidence.positionStdM,evidence.orientationStdRad,evidence.depthStdM].every(n=>Number.isFinite(n)&&n>0))throw new ApiError('Navigation free space needs synchronized calibrated depth-volume evidence.');
    for(const [key,cell] of Object.entries(map.voxels)){
      const index=key.split(',').map(Number);
      if(index.length!==3||index.some((v,i)=>!Number.isInteger(v)||v<0||v>=map.size[['x','y','z'][i]])||!['free','occupied'].includes(cell.state)||!Number.isFinite(cell.observedMs)||cell.observedMs>map.updatedMs+30)throw new ApiError('Invalid observed navigation voxel.');
    }
    state.mapRevision=p.mapRevision;state.telemetry=t;state.voxelMap=map;state.lastNavigationSeconds=p.timestampSeconds;state.navigationReceivedAtMs=this.now();state.navigationDigest=digest;
    const controls=taskControls(state),action=state.lastCaptureSeconds==null?{type:'idle',reason:'awaiting_semantic_observation',actuationEnabled:false}:stepLiveExecutor(state,{nowMs:state.mode==='live'?this.now():p.timestampSeconds*1000,options:this.executorOptions});
    state.lastAction={...action,generatedAtMs:this.now()};if(taskControls(state)!==controls)touchControl(state);
    this.save(state);return {duplicate:false,action:state.lastAction,state:this.snapshot()};
  });}
  updateTask(id,input){return this.transaction(()=>{
    const state=this.requireActive();revision(state,input);const task=state.tasks.find(t=>t.id===id);if(!task)throw new ApiError('Task not found.',404,'LIVE_TASK_NOT_FOUND');
    if(taskTerminal(task))throw new ApiError('Completed or cancelled tasks cannot be changed.',409,'LIVE_TASK_TERMINAL');
    switch(input.action){
      case 'pause':task.status='paused';task.pauseReason='operator';break;
      case 'cancel':task.status='cancelled';task.cancelledAtMs=this.now();break;
      case 'resume':case 'retry':
        if(!['paused','blocked'].includes(task.status))throw new ApiError('Only paused or blocked tasks can be resumed.',409,'LIVE_TASK_STATE');
        task.status='queued';task.phase='pending';task.mapRevision=state.mapRevision;task.viewpointEnuM=null;task.captureRequest=null;task.settledSinceMs=null;task.blockedReason=null;break;
      case 'move':{
        if(task.status==='running')throw new ApiError('Pause the running task before reordering.',409,'LIVE_TASK_STATE');
        const before=input.beforeTaskId==null?null:state.tasks.find(t=>t.id===input.beforeTaskId);if(input.beforeTaskId!=null&&!before)throw new ApiError('Reorder target was not found.');
        if(before?.id!==id){state.tasks=state.tasks.filter(t=>t.id!==id);const index=before?state.tasks.findIndex(t=>t.id===before.id):state.tasks.length;state.tasks.splice(index,0,task);}break;
      }
      default:throw new ApiError('Unknown task action.');
    }
    if(['pause','cancel'].includes(input.action)&&state.lastAction?.taskId===id)state.lastAction={type:'hold',taskId:id,reason:`operator_${input.action}`,actuationEnabled:false};
    touchControl(state);event(state,'task_updated',{taskId:id,action:input.action},this.now());this.save(state);return this.snapshot();
  });}
  review(id,input){return this.transaction(()=>{
    const state=this.requireActive();revision(state,input);const target=state.landmarks.find(l=>l.id===id);if(!target)throw new ApiError('Landmark not found.',404,'LIVE_TARGET_NOT_FOUND');
    if(!['confirmed','rejected','uncertain'].includes(input.status))throw new ApiError('Choose confirmed, rejected or uncertain.');
    target.review={status:input.status,groupId:optionalText(input.groupId,'group',100),note:optionalText(input.note,'note',2000),reviewer:optionalText(input.reviewer,'reviewer',120)||'operator',reviewedAtMs:this.now(),trainingAllowed:false};target.humanVerified=input.status==='confirmed';
    if(input.status==='rejected')for(const task of state.tasks)if(task.landmarkId===id&&!taskTerminal(task)){task.status='cancelled';task.cancelReason='human_rejected_target';}
    touchControl(state);event(state,'landmark_reviewed',{landmarkId:id,status:input.status},this.now());this.save(state);return this.snapshot();
  });}
  tick(input={}){return this.transaction(()=>{
    const state=this.requireActive();revision(state,input);
    if(input.captureAck!=null){
      // Saving photo bytes is asynchronous. Recheck the exact capture target in
      // this transaction so a late ACK cannot advance a different queued task.
      const ack=input.captureAck,task=state.tasks.find(t=>t.status==='running'),request=task?.captureRequest;
      if(!request||ack.sessionId!==state.sessionId||ack.taskId!==task.id||ack.requestId!==request.requestId||ack.landmarkId!==task.landmarkId||ack.mapRevision!==state.mapRevision||request.sessionId!==state.sessionId||request.mapRevision!==state.mapRevision)
        throw new ApiError('The capture request is no longer the active running task.',409,'LIVE_CAPTURE_TASK_CHANGED');
    }
    if(state.lastCaptureSeconds===null)throw new ApiError('Ingest a synchronized frame first.',409,'LIVE_FRAME_REQUIRED');
    if(input.nowMs!==undefined)throw new ApiError('Executor time comes from the capture clock and receipt age.');
    // Delayed inference is still useful for review, but reception does not make
    // old sensor evidence fresh. Hardware sessions use synchronized epoch time.
    const captureClock=state.lastNavigationSeconds??state.lastCaptureSeconds,receiptClock=state.navigationReceivedAtMs??state.receivedAtMs;
    const nowMs=state.mode==='live'?this.now():captureClock*1000+Math.max(0,this.now()-receiptClock);
    const controls=taskControls(state);
    const action={...stepLiveExecutor(state,{nowMs,captureAck:input.captureAck??null,options:this.executorOptions}),generatedAtMs:this.now()};
    if(action.type==='completed'&&action.capture){const task=state.tasks.find(t=>t.id===action.taskId),landmark=state.landmarks.find(l=>l.id===task?.landmarkId);if(landmark)landmark.capture={...action.capture,taskId:action.taskId};}
    state.lastAction=action;if(taskControls(state)!==controls)touchControl(state);event(state,'executor_step',{action},this.now());this.save(state);return {action,state:this.snapshot()};
  });}
  seedSimulation(input={}){
    const current=this.active();if(current)revision(current,input);
    const sessionId=`demo-${randomUUID()}`;this.startSession({sessionId,mode:'replay'});
    this.ingestFrame(createSimulationPacket({sessionId,blocked:input.blocked===true}));
    const state=this.requireActive();state.simulationDemo=true;state.simulationBlocked=input.blocked===true;event(state,'simulation_created',{synthetic:true},this.now());this.save(state);return this.snapshot();
  }
  stepSimulation(input={}){
    const state=this.requireActive();revision(state,input);if(!state.simulationDemo||state.mode!=='replay')throw new ApiError('Simulation stepping requires an explicitly created demo session.',409,'LIVE_SIMULATION_REQUIRED');
    const previous=state.lastAction,telemetry=state.telemetry;let positionEnuM=telemetry.positionEnuM,yawRad=telemetry.yawRad,pitchRad=telemetry.pitchRad;
    const proposalActive=state.tasks.some(t=>t.id===previous?.taskId&&t.status==='running');
    if(proposalActive&&previous?.type==='move_proposal')positionEnuM=previous.waypointsEnuM.at(-1);
    if(proposalActive&&previous?.type==='aim_proposal'){yawRad=previous.aim.yawRad;pitchRad=previous.aim.pitchRad;}
    const timeSeconds=state.lastCaptureSeconds+1;
    this.ingestFrame(createSimulationPacket({sessionId:state.sessionId,sequence:state.lastSequence+1,timeSeconds,positionEnuM,yawRad,pitchRad,blocked:state.simulationBlocked}));
    const refreshed=this.requireActive(),task=refreshed.tasks.find(t=>t.status==='running'&&t.captureRequest);
    const captureAck=task&&previous?.type==='capture_request'?createSyntheticCaptureAck(refreshed,task,timeSeconds*1000):null;
    return this.tick({captureAck});
  }
}
