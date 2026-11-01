import { readFile,stat } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { ApiError } from './validation.mjs';
import { validateImage } from './finds-images.mjs';
import { testbenchResult,testbenchSettings } from './testbench-validation.mjs';
import { trackingProfile } from '../public/tracking-profile.mjs';

const HASH=/^[a-f0-9]{64}$/;
const number=(n,min,max)=>typeof n==='number'&&Number.isFinite(n)&&n>=min&&n<=max;
export function validateTrackingDemo(data){
  const fail=()=>{throw new ApiError('The recorded tracking demonstration failed its integrity checks.',409,'TRACKING_DEMO_INVALID');};
  if(data?.schema!=='mushroom-vision.tracking-demo.v1'||data.inferenceMode!=='precomputed_replay'||!HASH.test(data.videoSha256)||
    !Number.isInteger(data.videoBytes)||data.videoBytes<1||data.videoBytes>16*1024*1024||!HASH.test(data.source?.videoHash)||
    data.actuationEnabled!==false||data.humanVerified!==false||data.trainingAllowed!==false||
    !Number.isInteger(data.width)||!Number.isInteger(data.height)||data.width<1||data.height<1||data.width*data.height>2000000||
    !number(data.durationSeconds,.1,120)||!Array.isArray(data.frames)||data.frames.length<2||data.frames.length>120)fail();
  for(const key of ['title','attribution','license','sourceUrl','licenseUrl'])if(typeof data.source[key]!=='string'||data.source[key].length>2000)fail();
  for(const key of ['sourceUrl','licenseUrl']){try{const url=new URL(data.source[key]);if(url.protocol!=='https:'||url.username||url.password)fail();}catch{fail();}}
  let previous=-1;
  for(const frame of data.frames){
    if(!number(frame.timestampSeconds,0,data.durationSeconds)||frame.timestampSeconds<=previous||frame.width!==data.width||frame.height!==data.height||
      !HASH.test(frame.imageSha256)||!number(frame.processingSeconds,0,3600)||!Array.isArray(frame.detections)||frame.detections.length>64)fail();
    previous=frame.timestampSeconds;
    for(const detection of frame.detections){const b=detection.bbox;if(!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||b[0]<0||b[1]<0||b[2]>data.width||b[3]>data.height||b[2]<=b[0]||b[3]<=b[1]||!number(detection.confidence,0,1))fail();}
  }
  return data;
}
export async function readTrackingDemo(root){
  const path=resolve(root,'data/tracking-demo-v1/manifest.json');
  try{if((await stat(path)).size>1024*1024)throw new ApiError('Tracking demo manifest is too large.',409);
    const data=validateTrackingDemo(JSON.parse(await readFile(path,'utf8')));
    const {videoSha256,videoBytes,...publicData}=data;
    return {...publicData,available:true,videoUrl:'/api/tracking-demo/video'};
  }catch(error){if(error.code==='ENOENT')return {schema:'mushroom-vision.tracking-demo.v1',available:false,frames:[],actuationEnabled:false,
    notice:'The licensed replay is not installed. The labeled synthetic example is available; run scripts/build-tracking-demo.py after acquiring the documented recording to install the real replay.'};throw error;}
}
export async function readTrackingDemoVideo(root){
  const directory=resolve(root,'data/tracking-demo-v1'),manifestPath=resolve(directory,'manifest.json');
  if((await stat(manifestPath)).size>1024*1024)throw new ApiError('Tracking demo manifest is too large.',409);
  const data=validateTrackingDemo(JSON.parse(await readFile(manifestPath,'utf8'))),file=resolve(directory,'replay.mp4');
  if((await stat(file)).size!==data.videoBytes)throw new ApiError('Tracking video size changed.',409,'TRACKING_DEMO_INVALID');
  const bytes=await readFile(file);if(createHash('sha256').update(bytes).digest('hex')!==data.videoSha256)throw new ApiError('Tracking video integrity check failed.',409,'TRACKING_DEMO_INVALID');
  return {bytes,mimeType:'video/mp4'};
}
export async function detectTrackingFrame(input,worker){
  let profile;try{profile=trackingProfile(input);}catch(error){throw new ApiError(error.message);}
  const image=validateImage(input);
  if(image.width*image.height>1920*1080)throw new ApiError('Sample tracking frames at no more than 1920 × 1080 pixels.');
  const threshold=input.boxThreshold??.3;if(!number(threshold,.1,.8))throw new ApiError('Use a detection threshold between 0.1 and 0.8.');
  const settings=testbenchSettings({methods:profile.methods,tileGrid:profile.tileGrid,scale:profile.scale,boxThreshold:threshold,...(profile.tileRegion?{tileRegion:profile.tileRegion}:{})});
  const started=performance.now(),raw=await worker.run(image,settings),result=testbenchResult(raw,image,settings);
  const method=result.methods[0];
  if(!Number.isInteger(result.modelPasses)||result.modelPasses!==method.passes||profile.tileRegion&&(!method.fullFrameContext||!method.passTimings))throw new ApiError('The tracking worker omitted the actual scan-pass metadata.',502,'TESTBENCH_PROTOCOL');
  return {schema:'mushroom-vision.tracking-detection.v1',image:{width:result.image.width,height:result.image.height,sha256:image.sha256},
    detections:method.candidates.map(item=>({bbox:item.bboxPixelsXYXY,confidence:item.confidence,label:'Unverified mushroom candidate'})),
    processingSeconds:result.seconds,requestSeconds:(performance.now()-started)/1000,model:result.model,
    profile:{id:profile.id,label:profile.label,expectedPasses:profile.expectedPasses,semanticGroundSegmentation:false},settings,
    scan:{method:method.id,passes:method.passes,modelPasses:result.modelPasses,fullFrameContext:method.id==='original'||method.fullFrameContext===true,
      tileRegion:method.tileRegion??null,tileBoundsPixelsXYXY:method.tileBoundsPixelsXYXY??[],passTimings:method.passTimings??[{kind:'context',bboxPixelsXYXY:[0,0,result.image.width,result.image.height],seconds:method.seconds}],seconds:method.seconds},
    humanVerified:false,trainingAllowed:false,actuationEnabled:false,stored:false,
    notice:'Sampled RGB inference. Scores are uncalibrated; no depth, GPS, verified species identity or flight command is produced.'};
}
