import { createHash, randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { ApiError, shortText, finite } from './validation.mjs';
import { validateImage } from './finds-images.mjs';

export const DEVICE_BODY_LIMIT = 6 * 1024 * 1024;
export const TASKS = Object.freeze(['image-quality-v1', 'fungi-candidates-v1']);
const MAX_BYTES = 4 * 1024 * 1024, MAX_QUEUE_BYTES = 32 * 1024 * 1024;
const digest = value => createHash('sha256').update(value).digest();
const fail = (message, status=400, code='DEVICE_INPUT') => { throw new ApiError(message,status,code); };
const integer = (value,name,min,max) => { finite(value,name,min,max); if(!Number.isInteger(value))fail(`${name} must be an integer.`); return value; };
const plain = value => value && typeof value==='object' && !Array.isArray(value);
const allowedKeys = (input, keys) => { if(!plain(input)||Object.keys(input).some(key=>!keys.includes(key)))fail('Unexpected device protocol fields.'); };

/** Ephemeral optional-analysis queue. This module has no mission/flight dependencies. */
export class DeviceComputeHub {
  constructor({profiles=[],clock=Date.now}={}) { this.profiles=profiles;this.clock=clock;this.devices=new Map();this.jobs=new Map(); }
  sweep() {
    const now=this.clock();
    for(const job of this.jobs.values()) {
      if(['queued','leased'].includes(job.status)&&now>=job.expiresAt){job.status='expired';delete job.image;}
    }
    // Retain only the most recent 64 finished jobs; never discard active work to admit more.
    const finished=[...this.jobs.values()].filter(j=>!['queued','leased'].includes(j.status));
    for(const job of finished.slice(0,Math.max(0,finished.length-64)))this.jobs.delete(job.id);
  }
  pair(input) {
    allowedKeys(input,['name','profileId']);this.sweep();
    if(this.devices.size>=16)fail('Remove a paired device before adding another.',429,'DEVICE_LIMIT');
    const name=shortText(input.name,'Device name',undefined,80),profileId=shortText(input.profileId,'Profile',undefined,80);
    if(!this.profiles.some(p=>p.id===profileId))fail('Choose a known device profile.');
    const id=randomUUID(),token=randomBytes(32).toString('base64url');
    this.devices.set(id,{id,name,profileId,tokenHash:digest(token),pairedAt:this.clock(),lastSeenAt:null,capabilities:[],runtime:null});
    return {deviceId:id,deviceToken:token,workerPath:'/api/device-compute/worker',expiresOnStationRestart:true};
  }
  revoke(input) {
    allowedKeys(input,['deviceId']);if(!this.devices.delete(input.deviceId))fail('Device not found.',404,'DEVICE_NOT_FOUND');
    for(const job of this.jobs.values())if(job.deviceId===input.deviceId&&job.status==='leased'){job.status='cancelled';delete job.image;}
    return {revoked:true};
  }
  authenticate(authorization) {
    if(typeof authorization!=='string'||!authorization.startsWith('Bearer ')||authorization.length>200)fail('A paired device token is required.',401,'DEVICE_UNAUTHORIZED');
    const tokenHash=digest(authorization.slice(7));
    for(const device of this.devices.values())if(timingSafeEqual(device.tokenHash,tokenHash))return device;
    fail('Device pairing is missing, revoked, or expired after a station restart.',401,'DEVICE_UNAUTHORIZED');
  }
  enqueue(input) {
    allowedKeys(input,['kind','sessionId','frameId','imageBase64','mimeType','ttlMs','targetDeviceId']);this.sweep();
    if(!TASKS.includes(input.kind))fail('This task is not supported. Flight and navigation jobs are never accepted.');
    const sessionId=shortText(input.sessionId,'Session',undefined,120),frameId=shortText(input.frameId,'Frame',undefined,120);
    const ttlMs=integer(input.ttlMs??30000,'Lifetime',1000,120000);
    if(input.targetDeviceId&&!this.devices.has(input.targetDeviceId))fail('Target device is not paired.',404,'DEVICE_NOT_FOUND');
    const image=validateImage(input);
    if(image.bytes.length>MAX_BYTES||image.width*image.height>8_000_000)fail('Device jobs allow at most 4 MiB and 8 megapixels per still image.',413,'DEVICE_IMAGE_LIMIT');
    const pending=[...this.jobs.values()].filter(j=>['queued','leased'].includes(j.status));
    if(pending.length>=32||pending.reduce((n,j)=>n+(j.image?.byteLength??0),0)+image.bytes.length>MAX_QUEUE_BYTES)fail('The optional vision queue is full.',429,'DEVICE_QUEUE_FULL');
    const job={id:randomUUID(),kind:input.kind,sessionId,frameId,imageSha256:image.sha256,
      width:image.width,height:image.height,createdAt:this.clock(),expiresAt:this.clock()+ttlMs,status:'queued',targetDeviceId:input.targetDeviceId??null,
      image:{imageBase64:image.bytes.toString('base64'),mimeType:image.mimeType,byteLength:image.bytes.length}};
    this.jobs.set(job.id,job);return this.publicJob(job);
  }
  publicJob(job) { const {image,leaseToken,...visible}=job;return structuredClone(visible); }
  worker(authorization,input) {
    this.sweep();const device=this.authenticate(authorization);
    if(!plain(input))fail('Device request must be an object.');
    if(input.op==='announce') {
      allowedKeys(input,['op','capabilities','runtime']);
      if(!Array.isArray(input.capabilities)||!input.capabilities.length||input.capabilities.length>TASKS.length||new Set(input.capabilities).size!==input.capabilities.length||input.capabilities.some(c=>!TASKS.includes(c)))fail('Declare implemented task capabilities.');
      device.capabilities=[...input.capabilities];device.runtime=shortText(input.runtime,'Runtime',undefined,120);device.lastSeenAt=this.clock();
      return {deviceId:device.id,acceptedCapabilities:device.capabilities,maxConcurrent:1,maxImageBytes:MAX_BYTES,maxPixels:8_000_000,humanReviewRequired:true,flightAuthority:false};
    }
    if(!device.capabilities.length)fail('Announce capabilities before requesting work.',409,'DEVICE_NOT_READY');
    device.lastSeenAt=this.clock();
    if(input.op==='claim') {
      allowedKeys(input,['op']);
      if([...this.jobs.values()].some(j=>j.status==='leased'&&j.deviceId===device.id))return {job:null,reason:'already_working'};
      const job=[...this.jobs.values()].find(j=>j.status==='queued'&&device.capabilities.includes(j.kind)&&(!j.targetDeviceId||j.targetDeviceId===device.id));
      if(!job)return {job:null,reason:'no_matching_work'};
      job.status='leased';job.deviceId=device.id;job.leaseToken=randomBytes(24).toString('base64url');job.claimedAt=this.clock();
      return {job:{...this.publicJob(job),image:structuredClone(job.image),leaseToken:job.leaseToken,remainingMs:job.expiresAt-this.clock(),flightAuthority:false}};
    }
    if(input.op==='result') {
      allowedKeys(input,['op','jobId','leaseToken','sessionId','frameId','imageSha256','result']);
      const job=this.jobs.get(input.jobId);
      if(!job||job.deviceId!==device.id||job.leaseToken!==input.leaseToken)fail('Result does not match this device lease.',409,'DEVICE_LEASE_MISMATCH');
      if(job.status!=='leased'||this.clock()>=job.expiresAt)fail('Late, duplicate, or cancelled result discarded.',409,'DEVICE_RESULT_STALE');
      if(job.sessionId!==input.sessionId||job.frameId!==input.frameId||job.imageSha256!==input.imageSha256)fail('Result image or session does not match its job.',409,'DEVICE_FRAME_MISMATCH');
      const result=validateResult(job,input.result);
      job.result={...result,unverified:true,humanReviewRequired:true,flightAuthority:false};job.status='completed';job.completedAt=this.clock();delete job.image;
      return {accepted:true,job:this.publicJob(job)};
    }
    fail('Unknown device operation.');
  }
  status() {
    this.sweep();return {schema:'mycoscout.device-compute.v1',profiles:this.profiles,volatile:true,flightAuthority:false,
      devices:[...this.devices.values()].map(({tokenHash,...device})=>({...device,online:device.lastSeenAt!==null&&this.clock()-device.lastSeenAt<15000})),
      jobs:[...this.jobs.values()].map(j=>this.publicJob(j)).reverse(),limits:{devices:16,pendingJobs:32,totalPendingBytes:MAX_QUEUE_BYTES,imageBytes:MAX_BYTES,concurrencyPerDevice:1}};
  }
}

export function validateResult(job,input) {
  if(job.kind==='image-quality-v1') {
    allowedKeys(input,['algorithm','width','height','sampleWidth','sampleHeight','meanLuminance','luminanceStd','laplacianVariance','meanChroma','colorOutlierFraction','processingMs']);
    if(input.algorithm!=='rgba-quality-v1')fail('Unknown quality algorithm.');
    if(input.width!==job.width||input.height!==job.height)fail('Quality result dimensions differ from source.');
    const result={algorithm:input.algorithm,width:input.width,height:input.height};
    for(const key of ['sampleWidth','sampleHeight'])result[key]=integer(input[key],key,1,640);
    if(result.sampleWidth>job.width||result.sampleHeight>job.height)fail('Quality samples cannot enlarge the source.');
    for(const [key,max]of [['meanLuminance',255],['luminanceStd',255],['laplacianVariance',1_100_000],['meanChroma',255],['colorOutlierFraction',1],['processingMs',120000]])result[key]=finite(input[key],key,0,max);
    return result;
  }
  allowedKeys(input,['model','processingMs','candidates']);
  const model=shortText(input.model,'Model provenance',undefined,200),processingMs=finite(input.processingMs,'Processing time',0,120000);
  if(!Array.isArray(input.candidates)||input.candidates.length>100)fail('At most 100 candidates per frame are accepted.');
  const candidates=input.candidates.map(candidate=>{
    allowedKeys(candidate,['bboxPixelsXYXY','confidence','modelLabel']);
    const b=candidate.bboxPixelsXYXY;
    if(!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||b[0]<0||b[1]<0||b[2]>job.width||b[3]>job.height||b[0]>=b[2]||b[1]>=b[3])fail('Candidate box must lie within source pixels.');
    return {bboxPixelsXYXY:[...b],confidence:finite(candidate.confidence,'Score',0,1),modelLabel:shortText(candidate.modelLabel,'Candidate label','fungus candidate',120),status:'unverified',species:null,position:null};
  });
  return {model,processingMs,candidates,scoreMeaning:'Uncalibrated model score, not species probability'};
}

/** Reproducible queue/network planning scenario, not measurements of a phone or radio. */
export function simulateOffload(input={}) {
  allowedKeys(input,['frames','fps','frameBytes','resultBytes','bandwidthMbps','rttMs','jitterMs','lossRate','computeMs','deadlineMs','seed']);
  const p={frames:integer(input.frames??100,'Frames',1,2000),fps:finite(input.fps??2,'Frame rate',.1,60),frameBytes:integer(input.frameBytes??200000,'Frame bytes',100,MAX_BYTES),resultBytes:integer(input.resultBytes??4000,'Result bytes',100,100000),bandwidthMbps:finite(input.bandwidthMbps??20,'Bandwidth',.1,1000),rttMs:finite(input.rttMs??40,'Round-trip latency',0,5000),jitterMs:finite(input.jitterMs??15,'Jitter',0,5000),lossRate:finite(input.lossRate??.02,'Failed transfer rate',0,1),computeMs:finite(input.computeMs??200,'Analysis time',.1,120000),deadlineMs:integer(input.deadlineMs??3000,'Deadline',100,120000),seed:integer(input.seed??42,'Random seed',0,0xffffffff)};
  let seed=p.seed>>>0;const rng=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed/4294967296;};
  let workerAvailable=0,networkAvailable=0;const outcomes=[],accepted=[];
  for(let i=0;i<p.frames;i++) {
    const arrival=i*1000/p.fps,uploadMs=p.frameBytes*8/(p.bandwidthMbps*1000),downloadMs=p.resultBytes*8/(p.bandwidthMbps*1000),networkMs=Math.max(0,p.rttMs+(rng()*2-1)*p.jitterMs);
    const uploadStart=Math.max(arrival,networkAvailable);networkAvailable=uploadStart+uploadMs;
    if(rng()<p.lossRate){outcomes.push({frame:i,outcome:'transfer_failed',latencyMs:null});continue;}
    const ready=networkAvailable+networkMs/2,computeStart=Math.max(ready,workerAvailable);
    if(computeStart+p.computeMs+networkMs/2+downloadMs>arrival+p.deadlineMs){outcomes.push({frame:i,outcome:'deadline_missed',latencyMs:null});continue;}
    workerAvailable=computeStart+p.computeMs;
    const latency=workerAvailable+networkMs/2+downloadMs-arrival;accepted.push(latency);outcomes.push({frame:i,outcome:'completed',latencyMs:Math.round(latency*10)/10});
  }
  accepted.sort((a,b)=>a-b);const percentile=q=>accepted.length?Math.round(accepted[Math.min(accepted.length-1,Math.floor((accepted.length-1)*q))]*10)/10:null;
  return {schema:'mycoscout.offload-simulation.v1',estimated:true,parameters:p,completed:accepted.length,failedTransfers:outcomes.filter(x=>x.outcome==='transfer_failed').length,missedDeadlines:outcomes.filter(x=>x.outcome==='deadline_missed').length,completionFraction:accepted.length/p.frames,latencyMs:{median:percentile(.5),p95:percentile(.95)},offeredBandwidthMbps:p.fps*p.frameBytes*8/1e6,outcomes,
    limits:['Single worker, serialized upload link; no retransmission, camera encoding, thermal throttling or contention model.','Loss rate represents a whole failed transfer, not raw Wi-Fi packet loss.','Results cannot measure detection accuracy or guarantee a radio link or flight clearance.'],flightAuthority:false};
}

/** Call owner routes only AFTER existing station authentication; worker owns its token gate. */
export async function handleDeviceCompute({hub,request,response,url,readBody,sendJson}) {
  const path=url.pathname;
  if(request.method==='GET'&&path==='/api/device-compute/status'){sendJson(response,200,hub.status());return true;}
  if(request.method!=='POST'||!path.startsWith('/api/device-compute/'))return false;
  const input=await readBody(request,path.endsWith('/jobs')?DEVICE_BODY_LIMIT:128*1024);
  const action=path.slice('/api/device-compute/'.length);let result;
  if(action==='worker')result=hub.worker(request.headers.authorization,input);
  else if(action==='pair')result=hub.pair(input);
  else if(action==='revoke')result=hub.revoke(input);
  else if(action==='jobs')result=hub.enqueue(input);
  else if(action==='simulate')result=simulateOffload(input);
  else return false;
  sendJson(response,200,result);return true;
}
