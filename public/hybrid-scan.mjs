import { trackingProfile } from './tracking-profile.mjs';

export const HYBRID_SCAN_MODES=Object.freeze(['hybrid','fast','ground-tiles','full-tiles']);
const validTime=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=86400;

/** Source-time scheduling only. Plans do not queue work or change model thresholds. */
export class HybridScanScheduler{
  constructor({detailIntervalSeconds=5,fullSweepEvery=3,minFastFrames=2}={}){
    if(!Number.isInteger(detailIntervalSeconds)||detailIntervalSeconds<2||detailIntervalSeconds>60||!Number.isInteger(fullSweepEvery)||fullSweepEvery<1||fullSweepEvery>20||!Number.isInteger(minFastFrames)||minFastFrames<2||minFastFrames>20)throw new Error('Choose a detail interval from 2 to 60 seconds and at least two fast samples between sweeps.');
    this.detailIntervalSeconds=detailIntervalSeconds;this.fullSweepEvery=fullSweepEvery;this.minFastFrames=minFastFrames;this.reset();
  }
  reset(){this.generation=(this.generation??0)+1;this.revision=0;this.firstTime=null;this.lastTime=null;this.lastDetailTime=null;this.detailCount=0;this.lastStage=null;this.fastFramesSinceDetail=0;}
  plan({timestampSeconds,mode='hybrid',groundTop=.35}={}){
    if(!validTime(timestampSeconds)||this.lastTime!==null&&timestampSeconds<this.lastTime)throw new Error('Reset the scan schedule before seeking backward.');
    if(!HYBRID_SCAN_MODES.includes(mode)||typeof groundTop!=='number'||!Number.isFinite(groundTop)||groundTop<0||groundTop>.8)throw new Error('Choose a valid scan mode and ground boundary.');
    let profileId=mode,stage=mode==='fast'?'fast':mode==='full-tiles'?'whole-frame-detail':'ground-detail';
    if(mode==='hybrid'){
      const due=this.firstTime!==null&&timestampSeconds-(this.lastDetailTime??this.firstTime)>=this.detailIntervalSeconds;
      const detail=due&&this.fastFramesSinceDetail>=this.minFastFrames;
      profileId=detail?((this.detailCount+1)%this.fullSweepEvery===0?'full-tiles':'ground-tiles'):'fast';
      stage=profileId==='fast'?'fast':profileId==='full-tiles'?'whole-frame-detail':'ground-detail';
    }
    const profile=trackingProfile({profileId,...(profileId==='ground-tiles'?{tileRegion:[0,groundTop,1,1]}:{})});
    return {generation:this.generation,revision:this.revision,timestampSeconds,mode,stage,profile,
      reason:stage==='fast'?'Whole-frame search; detail follows when due.':stage==='whole-frame-detail'?'Detail sweep includes trunks, logs and higher clusters.':'Detail sweep of the selected lower image area.'};
  }
  commit(plan){
    if(!plan||plan.generation!==this.generation||plan.revision!==this.revision||!validTime(plan.timestampSeconds))return false;
    if(this.lastTime!==null&&plan.timestampSeconds<this.lastTime)return false;
    const distinctFrame=plan.timestampSeconds!==this.lastTime;
    this.firstTime??=plan.timestampSeconds;this.lastTime=plan.timestampSeconds;this.lastStage=plan.stage;
    if(plan.stage!=='fast'){this.lastDetailTime=plan.timestampSeconds;this.detailCount++;this.fastFramesSinceDetail=0;}
    else if(distinctFrame)this.fastFramesSinceDetail++;
    this.revision++;return true;
  }
}

/** Evidence is attached to its captured image; no association, location or label is inferred. */
export function investigationCues({detections,timestampSeconds,stage,width,height,sourceId}){
  if(!validTime(timestampSeconds)||!['fast','ground-detail','whole-frame-detail'].includes(stage)||!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||typeof sourceId!=='string'||!sourceId||!Array.isArray(detections)||detections.length>100)throw new Error('Investigation cues need the captured frame and measured detections.');
  return detections.filter(item=>Array.isArray(item?.bbox)&&item.bbox.length===4&&item.bbox.every(Number.isFinite)&&item.bbox[0]>=0&&item.bbox[1]>=0&&item.bbox[2]<=width&&item.bbox[3]<=height&&item.bbox[0]<item.bbox[2]&&item.bbox[1]<item.bbox[3]&&typeof item.confidence==='number'&&Number.isFinite(item.confidence)&&item.confidence>=.3&&item.confidence<=1).map((item,index)=>({
    id:`${sourceId}:${timestampSeconds}:${stage}:${index}`,sourceId,timestampSeconds,stage,
    status:'unconfirmed_image_cue',imageBox:[...item.bbox],imageSize:[width,height],score:item.confidence,
    species:null,actualDistanceM:null,position:null,gps:null,humanVerified:false,trainingAllowed:false,actuationEnabled:false,
  }));
}

/** Bounded historical evidence, independent of the live candidate tracker. */
export class InvestigationLog{
  constructor({maxFrames=20,maxCues=100}={}){
    if(!Number.isInteger(maxFrames)||maxFrames<1||maxFrames>20||!Number.isInteger(maxCues)||maxCues<1||maxCues>100)throw new Error('The investigation log holds at most 20 frames and 100 cues.');
    this.maxFrames=maxFrames;this.maxCues=maxCues;this.reset();
  }
  reset(){this.frames=[];}
  add({cues,thumbnail,thumbnailSize}){
    if(!Array.isArray(cues)||cues.length>100)throw new Error('Invalid investigation frame.');
    if(!cues.length)return;
    const first=cues[0];
    if(!first||cues.some(cue=>cue.sourceId!==first.sourceId||cue.timestampSeconds!==first.timestampSeconds||cue.stage!==first.stage||cue.status!=='unconfirmed_image_cue'||cue.actuationEnabled!==false||cue.position!==null||cue.gps!==null)||typeof thumbnail!=='string'||!thumbnail.startsWith('data:image/jpeg;base64,')||thumbnail.length>2000000||!Array.isArray(thumbnailSize)||thumbnailSize.length!==2||thumbnailSize.some(n=>!Number.isInteger(n)||n<1||n>2048))throw new Error('Investigation evidence must belong to one captured frame.');
    const id=`${first.sourceId}:${first.timestampSeconds}:${first.stage}`;
    const frame={id,sourceId:first.sourceId,timestampSeconds:first.timestampSeconds,stage:first.stage,thumbnail,thumbnailSize:[...thumbnailSize],cues:structuredClone(cues.slice(0,this.maxCues))};
    this.frames=this.frames.filter(item=>item.id!==id);this.frames.push(frame);
    while(this.frames.length>this.maxFrames||this.frames.reduce((sum,item)=>sum+item.cues.length,0)>this.maxCues)this.frames.shift();
  }
  get cues(){return this.frames.flatMap(frame=>structuredClone(frame.cues));}
}
