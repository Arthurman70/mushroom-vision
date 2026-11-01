import { ApiError, shortText } from './validation.mjs';
import { normalizedTileRegion } from '../public/tracking-profile.mjs';

export const METHOD_IDS=['original','color','tiles'];
export const DEFAULT_SETTINGS={methods:['original','color','tiles'],tileGrid:2,scale:1,boxThreshold:.25};
const object=value=>value&&typeof value==='object'&&!Array.isArray(value);
const finite=(value,min,max)=>typeof value==='number'&&Number.isFinite(value)&&value>=min&&value<=max;
const optionalText=(value,name,max=500)=>value==null||value===''?'':shortText(value,name,undefined,max);

export function testbenchSettings(input=DEFAULT_SETTINGS){
  if(object(input)&&Object.keys(input).some(key=>!['methods','tileGrid','scale','boxThreshold','tileRegion'].includes(key)))throw new ApiError('Unsupported test settings.');
  if(!object(input)||!Array.isArray(input.methods)||input.methods.length<1||input.methods.length>3||
      input.methods.some(id=>!METHOD_IDS.includes(id))||new Set(input.methods).size!==input.methods.length)
    throw new ApiError('Choose one or more distinct methods: original, color, tiles.');
  const settings={methods:METHOD_IDS.filter(id=>input.methods.includes(id)),tileGrid:input.tileGrid??2,
    scale:input.scale??1,boxThreshold:input.boxThreshold??.25};
  if(![2,3].includes(settings.tileGrid)||![1,.5,.25,.125].includes(settings.scale)||!finite(settings.boxThreshold,.1,.8))
    throw new ApiError('Use tileGrid 2 or 3, scale 1/.5/.25/.125, and boxThreshold 0.1–0.8.');
  if(input.tileRegion!==undefined){
    if(!settings.methods.includes('tiles'))throw new ApiError('A tile area requires the tiles method.');
    try{settings.tileRegion=normalizedTileRegion(input.tileRegion);}catch(error){throw new ApiError(error.message);}
  }
  return settings;
}

export function validTestBox(box,width,height){return Array.isArray(box)&&box.length===4&&box.every(n=>finite(n,0,Math.max(width,height)))&&
  box[0]<box[2]&&box[1]<box[3]&&box[2]<=width&&box[3]<=height;}

export function testbenchProgress(raw){
  if(!object(raw)||typeof raw.stage!=='string'||!/^[a-z0-9_]{1,48}$/.test(raw.stage)||
      !Number.isInteger(raw.completed)||!Number.isInteger(raw.total)||raw.total<1||raw.total>16||raw.completed<0||raw.completed>raw.total)
    throw new ApiError('Test worker sent invalid progress.',502,'TESTBENCH_PROTOCOL');
  return {stage:raw.stage,completed:raw.completed,total:raw.total};
}

export function testbenchResult(raw,image,settings){
  const fail=()=>{throw new ApiError('Test worker returned an invalid result.',502,'TESTBENCH_PROTOCOL');};
  if(!object(raw)||raw.schema!=='mycoscout.testbench-result.v1'||!object(raw.image)||raw.image.sha256!==image.sha256||
      !Number.isInteger(raw.image.width)||!Number.isInteger(raw.image.height)||raw.image.width<1||raw.image.height<1||
      !((raw.image.width===image.width&&raw.image.height===image.height)||(raw.image.width===image.height&&raw.image.height===image.width))||
      !Array.isArray(raw.methods)||raw.methods.length!==settings.methods.length||!Array.isArray(raw.colorCues)||raw.colorCues.length>12||
      !finite(raw.seconds,0,3600)||!object(raw.model))fail();
  let parsedSettings;try{parsedSettings=testbenchSettings(raw.settings);}catch{fail();}
  if(JSON.stringify(parsedSettings)!==JSON.stringify(settings))fail();
  for(const key of ['workingWidth','workingHeight'])if(!Number.isInteger(raw.image[key])||raw.image[key]<1||raw.image[key]>20000)fail();
  const {width,height}=raw.image,seen=new Set();
  const methods=raw.methods.map(method=>{
    if(!object(method)||!settings.methods.includes(method.id)||seen.has(method.id)||!Array.isArray(method.candidates)||method.candidates.length>100||
        !finite(method.seconds,0,3600)||!Number.isInteger(method.passes)||method.passes<1||method.passes>10)fail();
    seen.add(method.id);const ids=new Set();
    const candidates=method.candidates.map((candidate,index)=>{
      if(!object(candidate)||!validTestBox(candidate.bboxPixelsXYXY,width,height)||!finite(candidate.confidence,0,1))fail();
      const id=typeof candidate.id==='string'?candidate.id:`${method.id}-${index}`;
      if(!id||id.length>128||ids.has(id))fail();ids.add(id);
      return {id,bboxPixelsXYXY:[...candidate.bboxPixelsXYXY],confidence:candidate.confidence,
        label:'Unverified mushroom candidate',speciesConfirmed:false};
    });
    const detail={};
    if(method.reusedPasses!==undefined){if(!Number.isInteger(method.reusedPasses)||method.reusedPasses<0||method.reusedPasses>1)fail();detail.reusedPasses=method.reusedPasses;}
    if(method.additionalSeconds!==undefined){if(!finite(method.additionalSeconds,0,3600))fail();detail.additionalSeconds=method.additionalSeconds;}
    if(method.preNmsCandidateCount!==undefined){if(!Number.isInteger(method.preNmsCandidateCount)||method.preNmsCandidateCount<method.candidates.length||method.preNmsCandidateCount>10000)fail();detail.preNmsCandidateCount=method.preNmsCandidateCount;}
    if(method.id==='tiles'&&(settings.tileRegion!==undefined||method.tileRegion!==undefined)){
      let region;try{region=normalizedTileRegion(method.tileRegion);}catch{fail();}
      if(JSON.stringify(region)!==JSON.stringify(settings.tileRegion??[0,0,1,1])||method.fullFrameContext!==true||
        !Array.isArray(method.tileBoundsPixelsXYXY)||method.tileBoundsPixelsXYXY.length<1||method.tileBoundsPixelsXYXY.length>settings.tileGrid**2||
        method.passes!==method.tileBoundsPixelsXYXY.length+1||!Array.isArray(method.passTimings)||method.passTimings.length!==method.passes)fail();
      const bounds=method.tileBoundsPixelsXYXY.map(box=>{if(!validTestBox(box,width,height))fail();return [...box];});
      const wx=width/raw.image.workingWidth,wy=height/raw.image.workingHeight;
      const regionPixels=[Math.floor(region[0]*raw.image.workingWidth)*wx,Math.floor(region[1]*raw.image.workingHeight)*wy,
        Math.ceil(region[2]*raw.image.workingWidth)*wx,Math.ceil(region[3]*raw.image.workingHeight)*wy];
      if(bounds.some(box=>box[0]<regionPixels[0]-.0001||box[1]<regionPixels[1]-.0001||box[2]>regionPixels[2]+.0001||box[3]>regionPixels[3]+.0001))fail();
      const passTimings=method.passTimings.map((pass,index)=>{
        const expected=index===0?[0,0,width,height]:bounds[index-1];
        if(!object(pass)||pass.kind!==(index===0?'context':'tile')||!validTestBox(pass.bboxPixelsXYXY,width,height)||
          pass.bboxPixelsXYXY.some((value,i)=>Math.abs(value-expected[i])>.0001)||!finite(pass.seconds,0,3600))fail();
        return {kind:pass.kind,bboxPixelsXYXY:[...pass.bboxPixelsXYXY],seconds:pass.seconds};
      });
      if(Math.abs(passTimings.reduce((sum,pass)=>sum+pass.seconds,0)-method.seconds)>.01)fail();
      Object.assign(detail,{tileRegion:region,tileBoundsPixelsXYXY:bounds,fullFrameContext:true,passTimings});
    }
    return {id:method.id,label:({original:'Original RGB',color:'Color-enhanced RGB',tiles:'Tiled RGB'})[method.id],candidates,seconds:method.seconds,passes:method.passes,...detail};
  });
  const colorCues=raw.colorCues.map((cue,index)=>{
    if(!object(cue)||!validTestBox(cue.bboxPixelsXYXY,width,height)||!finite(cue.contrastScore,0,1))fail();
    return {id:`color-cue-${index}`,bboxPixelsXYXY:[...cue.bboxPixelsXYXY],contrastScore:cue.contrastScore,
      label:'Color clue — fungal presence unknown',cueType:'color_salience',speciesConfirmed:false};
  });
  const model={};
  for(const key of ['model','resolvedRevision','device','prompt','scoreMeaning']){
    const value=raw.model[key];if(value!=null&&typeof value!=='string')fail();model[key]=value?.slice(0,500)??null;
  }
  for(const key of ['threads','boxThreshold','textThreshold','nmsIou'])if(raw.model[key]!==undefined){if(!finite(raw.model[key],0,100))fail();model[key]=raw.model[key];}
  if(raw.model.downloadsEnabled!==false)fail();model.downloadsEnabled=false;
  const warnings=Array.isArray(raw.warnings)?raw.warnings.slice(0,20).map(value=>String(value).slice(0,1000)):[];
  if(raw.modelPasses!==undefined&&(!Number.isInteger(raw.modelPasses)||raw.modelPasses<1||raw.modelPasses>11))fail();
  return {schema:raw.schema,image:{width,height,workingWidth:raw.image.workingWidth,workingHeight:raw.image.workingHeight,sha256:image.sha256},
    settings,model,methods,colorCues,seconds:raw.seconds,modelPasses:raw.modelPasses??null,warnings,actualDistanceM:null,actuationEnabled:false,
    humanConfirmationRequired:true,autoTraining:false};
}

export function testbenchAnnotation(input,run,groupIds){
  if(!object(input)||!Number.isSafeInteger(input.baseRevision)||input.baseRevision<0)throw new ApiError('baseRevision must be the current annotation revision.');
  if(typeof input.complete!=='boolean'||typeof input.trainingAllowed!=='boolean')throw new ApiError('complete and trainingAllowed must be explicit booleans.');
  if(!Array.isArray(input.boxes)||input.boxes.length>200)throw new ApiError('Provide at most 200 annotation boxes.');
  const ids=new Set(),boxes=input.boxes.map(box=>{
    if(!object(box)||!validTestBox(box.bboxPixelsXYXY,run.image.width,run.image.height))throw new ApiError('Annotation box must be inside the original oriented image.');
    const id=shortText(box.id,'box.id',undefined,128);if(ids.has(id))throw new ApiError('Annotation box IDs must be distinct.');ids.add(id);
    const groupId=box.groupId??'unknown';if(!groupIds.has(groupId))throw new ApiError('Choose an annotation group from the catalog.');
    return {id,bboxPixelsXYXY:[...box.bboxPixelsXYXY],groupId,label:optionalText(box.label,'box.label',240)};
  });
  const measuredDistanceM=input.measuredDistanceM??null;
  if(measuredDistanceM!==null&&!finite(measuredDistanceM,.01,10000))throw new ApiError('Measured distance must be 0.01–10000 m, or null when unknown.');
  return {reviewer:shortText(input.reviewer,'reviewer',undefined,160),complete:input.complete,
    trainingAllowed:run.evaluationOnly?false:input.trainingAllowed,trainingPermissionRequested:input.trainingAllowed,
    boxes,notes:optionalText(input.notes,'notes',2000),measuredDistanceM,
    distanceProvenance:measuredDistanceM===null?'unknown':'human_reported_not_independently_verified',
    emptySceneConfirmed:input.complete&&boxes.length===0,labelsFrom:'explicit_human_annotations',
    rightsAttestation:input.trainingAllowed?'Reviewer permits these submitted photos and labels to enter a future training export.':null,
    professionallyVerified:false,salesApproval:false,autoTraining:false};
}

export function boxIou(a,b){const overlap=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  const union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-overlap;return union>0?overlap/union:0;}

export function testbenchMetrics(result,annotation){
  if(!result)return {status:'unavailable',iouThreshold:.5,methods:[]};
  const complete=annotation?.complete===true,truth=annotation?.boxes??[];
  const methods=result.methods.map(method=>{
    const edges=method.candidates.map(candidate=>truth.map((box,index)=>({index,iou:boxIou(candidate.bboxPixelsXYXY,box.bboxPixelsXYXY)}))
      .filter(edge=>edge.iou>=.5).sort((a,b)=>b.iou-a.iou||a.index-b.index));
    const assigned=new Map();
    function assign(candidate,seen){for(const edge of edges[candidate]){if(seen.has(edge.index))continue;seen.add(edge.index);
      if(!assigned.has(edge.index)||assign(assigned.get(edge.index),seen)){assigned.set(edge.index,candidate);return true;}}return false;}
    for(let index=0;index<edges.length;index++)assign(index,new Set());
    const tp=assigned.size,fp=method.candidates.length-tp,fn=truth.length-tp;
    const precision=method.candidates.length?tp/method.candidates.length:null,recall=truth.length?tp/truth.length:null;
    return {id:method.id,predictions:method.candidates.length,groundTruth:truth.length,matched:tp,
      truePositives:annotation?tp:null,falsePositives:complete?fp:null,falseNegatives:complete?fn:null,
      precision:complete?precision:null,recall:complete?recall:null,f1:complete&&precision!==null&&recall!==null?(precision+recall?2*precision*recall/(precision+recall):0):null,
      matches:[...assigned].map(([truthIndex,predictionIndex])=>({annotationId:truth[truthIndex].id,predictionId:method.candidates[predictionIndex].id,
        iou:boxIou(truth[truthIndex].bboxPixelsXYXY,method.candidates[predictionIndex].bboxPixelsXYXY)}))};
  });
  return {status:!annotation?'unannotated':complete?'complete':'incomplete',iouThreshold:.5,
    matching:'maximum-cardinality one-to-one, class-agnostic detection',colorCuesScoredAsMushrooms:false,
    unknownRegionsAreNegatives:false,methods};
}
