const clamp=(value,min,max)=>Math.max(min,Math.min(max,value));
const finite=(value,min,max)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max;
export function boxIou(a,b){const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));const union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection;return union>0?intersection/union:0;}
function duplicateBox(a,b){const overlap=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));const aa=(a[2]-a[0])*(a[3]-a[1]),bb=(b[2]-b[0])*(b[3]-b[1]);return boxIou(a,b)>.8||(overlap/Math.min(aa,bb)>.9&&Math.max(aa,bb)/Math.min(aa,bb)<3.5);}
export function validTrackingDetection(item,width,height){const b=item?.bbox;return Array.isArray(b)&&b.length===4&&b.every(Number.isFinite)&&b[0]>=0&&b[1]>=0&&b[2]<=width&&b[3]<=height&&b[2]>b[0]&&b[3]>b[1]&&finite(item.confidence,0,1);}

/** Conservative image-plane association. No claim of metric 3D or specimen identity. */
export class CandidateTracker{
  constructor({observedWindowSeconds=.15,predictionSeconds=1.2,lostSeconds=4,maxTracks=64}={}){
    if(!finite(observedWindowSeconds,0,5)||!finite(predictionSeconds,observedWindowSeconds,30)||!finite(lostSeconds,predictionSeconds,120)||!Number.isInteger(maxTracks)||maxTracks<1||maxTracks>128)throw new Error('Invalid tracker limits.');
    Object.assign(this,{observedWindowSeconds,predictionSeconds,lostSeconds,maxTracks});this.reset('initial');
  }
  reset(sourceId='source'){this.session=(this.session??0)+1;this.sourceId=String(sourceId).slice(0,160);this.tracks=[];this.nextId=1;this.lastTime=-1;this.width=1;this.height=1;}
  update({detections,timestampSeconds,width,height}){
    if(!finite(timestampSeconds,0,86400)||!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||width*height>20000000||!Array.isArray(detections)||detections.length>128)throw new Error('Invalid sampled frame.');
    if(timestampSeconds<this.lastTime)throw new Error('Reset tracking before seeking backward.');
    if(timestampSeconds===this.lastTime)return this.snapshot(timestampSeconds);
    if(this.lastTime>=0&&(this.width!==width||this.height!==height))throw new Error('Reset tracking when image dimensions change.');
    const candidates=detections.filter(item=>validTrackingDetection(item,width,height)).sort((a,b)=>b.confidence-a.confidence).filter((item,index,items)=>!items.slice(0,index).some(prior=>duplicateBox(prior.bbox,item.bbox))).slice(0,this.maxTracks);
    this.width=width;this.height=height;this.lastTime=timestampSeconds;
    this.tracks=this.tracks.filter(track=>timestampSeconds-track.lastSeen<=this.lostSeconds*2);
    const pairs=[],diagonal=Math.hypot(width,height);
    for(const [ti,track]of this.tracks.entries()){
      const age=timestampSeconds-track.lastSeen;if(age>this.lostSeconds)continue;
      for(const [di,item]of candidates.entries()){
        const overlap=boxIou(track.bbox,item.bbox),a=track.bbox,b=item.bbox;
        const distance=Math.hypot((a[0]+a[2]-b[0]-b[2])/2,(a[1]+a[3]-b[1]-b[3])/2)/diagonal;
        const areaRatio=((b[2]-b[0])*(b[3]-b[1]))/((a[2]-a[0])*(a[3]-a[1]));
        if((overlap>=.08||distance<=.18)&&areaRatio>=.2&&areaRatio<=5)pairs.push({ti,di,score:overlap*.7+(1-distance)*.3});
      }
    }
    pairs.sort((a,b)=>b.score-a.score);const usedTracks=new Set(),usedDetections=new Set();
    for(const pair of pairs){if(usedTracks.has(pair.ti)||usedDetections.has(pair.di))continue;usedTracks.add(pair.ti);usedDetections.add(pair.di);
      const track=this.tracks[pair.ti],detection=candidates[pair.di],dt=timestampSeconds-track.lastSeen;
      track.velocity=detection.bbox.map((coordinate,index)=>(coordinate-track.bbox[index])/Math.max(.01,dt));
      track.bbox=[...detection.bbox];track.confidence=detection.confidence;track.lastSeen=timestampSeconds;track.observations++;
    }
    for(const [index,detection]of candidates.entries())if(!usedDetections.has(index)&&this.tracks.length<this.maxTracks)this.tracks.push({id:'R'+this.session+'-T'+String(this.nextId++).padStart(2,'0'),sourceId:this.sourceId,bbox:[...detection.bbox],velocity:[0,0,0,0],confidence:detection.confidence,lastSeen:timestampSeconds,firstSeen:timestampSeconds,observations:1});
    return this.snapshot(timestampSeconds);
  }
  snapshot(timestampSeconds){
    if(!finite(timestampSeconds,0,86400))throw new Error('Invalid tracking clock.');
    return this.tracks.map(track=>{
      const age=Math.max(0,timestampSeconds-track.lastSeen),status=age<=this.observedWindowSeconds?'observed':age<=this.lostSeconds?'predicted':'lost';
      const dt=Math.min(age,this.predictionSeconds),predicted=track.bbox.map((value,index)=>clamp(value+track.velocity[index]*dt,0,index%2?this.height:this.width));
      const valid=predicted[2]>predicted[0]&&predicted[3]>predicted[1];
      return {id:track.id,sourceId:track.sourceId,status,stale:age>this.predictionSeconds,ageSeconds:age,lastSeenSeconds:track.lastSeen,
        bbox:status==='observed'||!valid?[...track.bbox]:predicted,lastObservedBbox:[...track.bbox],confidence:track.confidence,
        confidenceMeaning:'uncalibrated_detection_score',observations:track.observations,observedPixelWidth:track.bbox[2]-track.bbox[0],
        observedPixelHeight:track.bbox[3]-track.bbox[1],speciesVerified:false,position:null,gps:null};
    });
  }
}

export function closeLookProposal(track,{sourceId,timestampSeconds,actualDistanceM=null}={}){
  if(typeof sourceId!=='string'||!sourceId||sourceId.length>160||!finite(timestampSeconds,0,86400))throw new Error('A proposal needs its source and recording timestamp.');
  if(!track||track.status==='lost'||track.stale||track.observations<2||track.confidence<.3)throw new Error('Wait for at least two recent observations before proposing a closer look.');
  if(actualDistanceM!==null&&!finite(actualDistanceM,0,100000))throw new Error('Range must be a measured nonnegative distance or unknown.');
  return {schema:'mushroom-vision.close-look-proposal.v1',sourceId,trackId:track.id,timestampSeconds,
    kind:actualDistanceM!==null&&actualDistanceM>=50?'habitat_followup':'closer_photo',status:'proposal_only',
    reason:actualDistanceM===null?'Image-plane candidate; location and safe approach remain unresolved.':'Measured-range candidate requires human review and safe path validation.',
    observedPixelSize:[track.observedPixelWidth,track.observedPixelHeight],imageBox:[...track.lastObservedBbox],actualDistanceM,
    desiredFrameFill:.7,position:null,gps:null,requires:['human_review','calibrated_depth','time_aligned_pose','obstacle_coverage','safe_approach_plan'],
    actuationEnabled:false,trainingAllowed:false,speciesVerified:false};
}

/** One active promise; callers skip sampling while it runs, so no backlog grows. */
export class LatestFrameGate{
  constructor(){this.generation=0;this.pending=false;}
  reset(){this.generation++;}
  async run(work,accept){if(this.pending)return {skipped:true};const generation=this.generation;this.pending=true;
    try{const result=await work();if(generation!==this.generation)return {discarded:true};accept(result);return {accepted:true};}
    finally{this.pending=false;}
  }
}

export function syntheticFrame(time){
  const x=120+time*16,y=220-Math.sin(time*.5)*25;
  return {width:640,height:480,timestampSeconds:time,detections:time>8&&time<13.5?[]:[{bbox:[x,y,x+45+time*2,y+55+time*2],confidence:.7,label:'Synthetic candidate'}]};
}
