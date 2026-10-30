import { createHash } from 'node:crypto';
import { testbenchMetrics } from './testbench-validation.mjs';

const BINS=[['under4','< 4 px',0,4],['4to8','4–7 px',4,8],['8to16','8–15 px',8,16],['16to32','16–31 px',16,32],['32to64','32–63 px',32,64],['64plus','64+ px',64,Infinity]];
const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,stable(value[key])])):value;
const signature=value=>createHash('sha256').update(JSON.stringify(stable(value))).digest('hex');
const totals=()=>({frames:0,truePositives:0,falsePositives:0,falseNegatives:0,groundTruth:0,predictions:0,negativeFrames:0,
  pixelWidthBins:BINS.map(([id,label])=>({id,label,total:0,matched:0,missed:0,recall:null}))});
function add(target,metric,review){
  target.frames++;for(const key of ['truePositives','falsePositives','falseNegatives','groundTruth','predictions'])target[key]+=metric[key];
  if(!review.boxes.length)target.negativeFrames++;
  const matched=new Set(metric.matches.map(match=>match.annotationId));
  for(const box of review.boxes){const width=box.bboxPixelsXYXY[2]-box.bboxPixelsXYXY[0],index=BINS.findIndex(([, ,min,max])=>width>=min&&width<max),bin=target.pixelWidthBins[index];
    bin.total++;if(matched.has(box.id))bin.matched++;else bin.missed++;}
}
function finish(target){target.precision=target.predictions?target.truePositives/target.predictions:null;target.recall=target.groundTruth?target.truePositives/target.groundTruth:null;
  for(const bin of target.pixelWidthBins)bin.recall=bin.total?bin.matched/bin.total:null;return target;}

// Only actual results on the exact original frame can be compared with current
// exhaustive human labels. Repeated runs do not inflate denominators. Frames
// remain correlated within source groups.
export function videoDetectionMetrics(frames,snapshot,completedRuns){
  const byId=new Map(frames.map(frame=>[frame.id,frame])),eligible=new Map(),excluded={unreviewed:0,uncertain:0,incomplete:0,provenanceMismatchRuns:0,duplicateRuns:0};
  for(const frame of frames){const entry=snapshot.reviews[frame.id],review=entry?.review;
    if(!review){excluded.unreviewed++;continue;}if(review.status==='uncertain'){excluded.uncertain++;continue;}
    if(!review.complete||(review.status==='visible'&&!review.boxes.length)){excluded.incomplete++;continue;}
    eligible.set(frame.id,{frame,review,reviewRevision:entry.revision});}
  const selected=new Map();
  for(const run of [...completedRuns].sort((a,b)=>(b.createdAt??'').localeCompare(a.createdAt??'')||b.id.localeCompare(a.id))){
    if(run.status!=='completed'||!run.result||!run.videoProvenance)continue;
    const p=run.videoProvenance,frame=byId.get(p.id);
    if(!frame||p.sha256!==frame.sha256||run.image.sha256!==frame.sha256||run.result.image.sha256!==frame.sha256||
      run.result.image.width!==frame.width||run.result.image.height!==frame.height||p.sourceId!==frame.sourceId||
      p.source?.id!==frame.sourceId||p.source?.videoHash!==frame.source.videoHash||p.sourceGroupId!==frame.sourceGroupId||
      p.source?.sourceGroupId!==frame.sourceGroupId||p.timestampSeconds!==frame.timestampSeconds||p.width!==frame.width||p.height!==frame.height){excluded.provenanceMismatchRuns++;continue;}
    if(!eligible.has(frame.id))continue;
    for(const method of run.result.methods){
      const settings={...run.settings};delete settings.methods;
      const config={model:run.result.model,settings,methodId:method.id},configurationId=signature(config),key=frame.id+':'+configurationId;
      if(selected.has(key)){excluded.duplicateRuns++;continue;}
      selected.set(key,{...eligible.get(frame.id),run,method,config,configurationId});
    }
  }
  const grouped=new Map();
  for(const item of selected.values()){
    const {frame,review,reviewRevision,run,method,config,configurationId}=item;
    if(!grouped.has(configurationId))grouped.set(configurationId,{id:configurationId,...config,...totals(),sourceIds:new Set(),sourceGroupIds:new Set(),bySourceGroup:new Map(),observations:[]});
    const group=grouped.get(configurationId),metric=testbenchMetrics({...run.result,methods:[method]},review).methods[0];
    add(group,metric,review);group.sourceIds.add(frame.sourceId);group.sourceGroupIds.add(frame.sourceGroupId);
    if(!group.bySourceGroup.has(frame.sourceGroupId))group.bySourceGroup.set(frame.sourceGroupId,{sourceGroupId:frame.sourceGroupId,...totals()});
    add(group.bySourceGroup.get(frame.sourceGroupId),metric,review);
    group.observations.push({frameId:frame.id,runId:run.id,reviewRevision,sourceId:frame.sourceId,sourceGroupId:frame.sourceGroupId,timestampSeconds:frame.timestampSeconds,
      originalWidth:frame.width,originalHeight:frame.height,workingFrameWidth:run.result.image.workingWidth,workingFrameHeight:run.result.image.workingHeight,
      requestedScale:run.settings.scale,groundTruth:metric.groundTruth,predictions:metric.predictions,matched:metric.matched,falsePositives:metric.falsePositives,
      boxes:review.boxes.map(box=>({annotationId:box.id,originalPixelWidth:box.bboxPixelsXYXY[2]-box.bboxPixelsXYXY[0],
        workingFramePixelWidth:(box.bboxPixelsXYXY[2]-box.bboxPixelsXYXY[0])*run.result.image.workingWidth/frame.width,
        matched:metric.matches.some(match=>match.annotationId===box.id)}))});
  }
  const groups=[...grouped.values()].map(group=>{const {sourceIds,sourceGroupIds,bySourceGroup,...result}=group;return {...finish(result),sourceVideos:sourceIds.size,sourceGroups:sourceGroupIds.size,
    bySourceGroup:[...bySourceGroup.values()].map(finish)};});
  return {schema:'mycoscout.video-detection-metrics.v1',status:groups.length?'available':eligible.size?'awaiting_matching_tests':'awaiting_complete_labels',
    iouThreshold:.5,pixelWidthUnit:'original-frame pixels',completeReviewedFrames:eligible.size,groups,excluded,
    matching:'Maximum-cardinality one-to-one detection, class-agnostic; latest run per exact frame, method, model and settings.',
    workingPixelMeaning:'Working-frame pixel widths reflect the requested image scale, not the detector tensor or tile resampling. Bins always use original-frame widths.',
    notice:'These are descriptive rates on correlated reviewed video frames, not independent field trials, species-ID accuracy, or measured flight-distance accuracy. Color-only cues are excluded.'};
}
