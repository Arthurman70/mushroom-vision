import { open } from 'node:fs/promises';
import { resolve } from 'node:path';

const FILES=['experiments/results/video-distance-v3/report.json','experiments/results/video-distance-v2/report.json','experiments/results/video-distance-v1/report.json'];
const MAX_BYTES=4*1024*1024;
const count=value=>Number.isSafeInteger(value)&&value>=0;
const fraction=value=>Number.isFinite(value)&&value>=0&&value<=1;
const METHODS=['raw','color','tiles'];
function realizedWidths(row){
  const value=row.realizedTargetWidthPixels;
  if(!value||!Number.isFinite(value.min)||!Number.isFinite(value.max)||value.min<=0||value.max<value.min||value.max>65536)return null;
  if(row.realizedTargetWidths!==undefined){
    const widths=row.realizedTargetWidths;
    if(!Array.isArray(widths)||!widths.length||widths.length>10000||widths.some(width=>!Number.isFinite(width)||width<=0||width>65536)||
      Math.abs(Math.min(...widths)-value.min)>1e-9||Math.abs(Math.max(...widths)-value.max)>1e-9)return null;
  }
  return {min:value.min,max:value.max};
}
const empty=availability=>({schema:'mycoscout.video-distance-summary.v1',availability,status:'unavailable',generatedAt:null,
  readOnly:true,actualDistanceMeasured:false,automaticTraining:false,automaticActivation:false,sample:null,progress:null,rows:[],
  nativeBaseline:null,nativeSmallPositiveOnly:null,classificationRows:[],cropContextRows:[],aiCandidateDistance:null,corpusCandidateDistance:null,runVersion:null,methodNote:'No comparable measurements are available.',limitations:[]});

async function saved(root,file){
  let handle;
  try{
    handle=await open(resolve(root,file),'r');const info=await handle.stat();
    if(!info.isFile()||info.size>MAX_BYTES)return {availability:'invalid'};
    const bytes=Buffer.alloc(MAX_BYTES+1);let length=0;
    while(length<bytes.length){const {bytesRead}=await handle.read(bytes,length,bytes.length-length,null);if(!bytesRead)break;length+=bytesRead;}
    if(length>MAX_BYTES)return {availability:'invalid'};
    return {availability:'available',data:JSON.parse(bytes.subarray(0,length).toString('utf8'))};
  }catch(error){return {availability:error.code==='ENOENT'?'missing':'invalid'};}
  finally{await handle?.close();}
}

function classificationRows(data,detectorRows,sample){
  if(!Array.isArray(data)||data.length>32)return [];
  const result=[],seen=new Set();
  for(const row of data){
    const condition=detectorRows.find(value=>(value.native?'native':`${value.targetPixels}`)===String(row?.id).replace(/px$/,'').replace('p','.'));
    if(!condition||seen.has(condition.id)||row.referenceAuthority!=='assistant'||row.referenceClass!=='true_morels'||row.humanSpeciesGroundTruth!==false||
      !count(row.targets)||row.targets>sample.targets||!count(row.productionEligible)||row.productionEligible>row.targets)continue;
    const clean={id:condition.id,label:condition.label,targets:row.targets,productionEligible:row.productionEligible,
      referenceAuthority:'assistant',humanSpeciesGroundTruth:false};
    let valid=true;
    for(const name of ['baseline','expanded']){
      const value=row[name],keys=['top1Agreement','top3Agreement','retained','retainedAgreement','abstained'];
      if(!value||keys.some(key=>!count(value[key])||value[key]>row.targets)||value.top1Agreement>value.top3Agreement||
        value.retainedAgreement>value.retained||value.retainedAgreement>value.top1Agreement||value.retained+value.abstained!==row.targets){valid=false;break;}
      clean[name]=Object.fromEntries(keys.map(key=>[key,value[key]]));
    }
    if(valid){result.push(clean);seen.add(condition.id);}
  }
  return result;
}

function cropContextRows(data,detectorRows,sample){
  if(data?.schema!=='mycoscout.crop-context-report.v1'||data.status!=='completed'||data.protocol?.referenceAuthority!=='assistant'||
    data.protocol?.referenceClass!=='true_morels'||data.protocol?.humanSpeciesGroundTruth!==false||data.protocol?.knownTargetLocation!==true||
    data.protocol?.localizationEvaluated!==false||data.protocol?.modelActivation!==false||!Array.isArray(data.rows)||data.rows.length>32)return [];
  const result=[],seen=new Set();
  for(const row of data.rows){
    const condition=detectorRows.find(value=>(value.native?'native':`${value.targetPixels}`)===String(row?.condition).replace(/px$/,'').replace('p','.'));
    if(!condition||seen.has(condition.id)||!count(row.targets)||row.targets>sample.targets||!count(row.productionEligible)||row.productionEligible>row.targets)continue;
    const clean={id:condition.id,label:condition.label,targets:row.targets,productionEligible:row.productionEligible,policies:{}};
    let valid=true;
    for(const policy of ['tight','fitted','context']){
      clean.policies[policy]={};
      for(const name of ['baseline','expanded']){
        const value=row.policies?.[policy]?.[name],keys=['top1Agreement','retained','retainedAgreement'];
        if(!value||keys.some(key=>!count(value[key])||value[key]>row.targets)||value.retainedAgreement>value.retained||value.retainedAgreement>value.top1Agreement){valid=false;break;}
        clean.policies[policy][name]=Object.fromEntries(keys.map(key=>[key,value[key]]));
      }
      if(!valid)break;
    }
    if(valid){result.push(clean);seen.add(condition.id);}
  }
  return result;
}

function candidateDistance(data,detectorRows,sample,corpus=false){
  const protocol=data?.protocol;
  const schema=corpus?'mushroom-vision.ai-corpus-distance-report.v1':'mushroom-vision.ai-candidate-distance-report.v1';
  if(data?.schema!==schema||data.status!=='completed'||data.modelActivation!==false||data.productionModelChanged!==false||
    protocol?.trainingLabelAuthority!=='ai-visual-review'||protocol.evaluationBoxAuthority!=='human'||protocol.activationEligible!==false||
    protocol.thresholdSelectionOnTest!==false||protocol.additionalTraining!==false||(!corpus&&protocol.epochs!==1)||protocol.scoreThreshold!==.3||protocol.iouThreshold!==.5||
    protocol.model!=='fasterrcnn_mobilenet_v3_large_320_fpn'||['trainingFrames','trainingSourceGroups','trainingBoxes'].some(key=>!count(protocol[key])||protocol[key]>10000)||
    protocol.trainingSourceGroups>protocol.trainingFrames||!Array.isArray(data.rows)||data.rows.length>32)return null;
  if(corpus&&(!count(protocol.epochs)||protocol.epochs<1||protocol.epochs>12||!count(protocol.selectedEpoch)||protocol.selectedEpoch<1||protocol.selectedEpoch>protocol.epochs||
    !count(protocol.optimizerSteps)||protocol.optimizerSteps<1||protocol.optimizerSteps>10000))return null;
  const result={trainingFrames:protocol.trainingFrames,trainingSourceGroups:protocol.trainingSourceGroups,trainingBoxes:protocol.trainingBoxes,
    scoreThreshold:.3,iouThreshold:.5,activationEligible:false,trainingLabelAuthority:'ai-visual-review',evaluationBoxAuthority:'human',rows:[]};
  if(corpus)Object.assign(result,{epochs:protocol.epochs,selectedEpoch:protocol.selectedEpoch,optimizerSteps:protocol.optimizerSteps,
    evaluationPurpose:'Previously inspected regression benchmark; not independent field validation',hardNegativeCrops:Object.fromEntries(
      Object.entries(protocol.hardNegativeCrops??{}).filter(([key,value])=>['hat','stump','tree','leaf','rock','wood','person','animal','other'].includes(key)&&count(value)&&value<=10000))});
  const seen=new Set();
  for(const row of data.rows){
    const condition=detectorRows.find(value=>(value.native?'native':`${value.targetPixels}`)===String(row?.condition).replace(/px$/,'').replace('p','.'));
    if(!condition||seen.has(condition.id)||['frames','targets','matched','unmatchedDetections','negativeFrames','negativeFalsePositives'].some(key=>!count(row[key]))||
      row.frames>sample.positiveFrames||row.targets>sample.targets||row.matched>row.targets||!row.frames&&(row.targets||row.unmatchedDetections)||
      row.negativeFrames>sample.negativeFrames||!row.negativeFrames&&row.negativeFalsePositives)continue;
    result.rows.push({id:condition.id,label:condition.label,...Object.fromEntries(['frames','targets','matched','unmatchedDetections','negativeFrames','negativeFalsePositives'].map(key=>[key,row[key]]))});
    seen.add(condition.id);
  }
  return result.rows.length?result:null;
}

function metric(value,sample){
  if(!value||['frames','matched','targets','falsePositives','negativeFrames','negativeFalsePositives','negativeFramesWithDetections'].some(key=>!count(value[key]))||
    value.frames>sample.positiveFrames||value.targets>sample.targets||value.matched>value.targets||value.negativeFrames>sample.negativeFrames||
    value.negativeFramesWithDetections>value.negativeFrames||value.negativeFramesWithDetections>value.negativeFalsePositives||
    !value.negativeFrames&&(value.negativeFalsePositives||value.negativeFramesWithDetections)||!value.frames&&(value.targets||value.falsePositives))return null;
  return {frames:value.frames,matched:value.matched,targets:value.targets,recall:value.targets?value.matched/value.targets:null,
    falsePositives:value.falsePositives,negativeFrames:value.negativeFrames,negativeFalsePositives:value.negativeFalsePositives,
    negativeFramesWithDetections:value.negativeFramesWithDetections};
}

function nativeEvidence(value,positiveOnly){
  if(!value||value.labelAuthority!=='human'||['frames','matched','targets','sourceVideos','sourceGroups'].some(key=>!count(value[key]))||
    !value.frames||!value.targets||value.matched>value.targets||value.sourceGroups>value.sourceVideos||value.sourceVideos>value.frames||
    !fraction(value.scoreThreshold)||!fraction(value.iouThreshold)||!value.iouThreshold||
    positiveOnly&&value.falsePositives!==null&&value.falsePositives!==undefined||!positiveOnly&&!count(value.falsePositives))return null;
  const widths=value.targetWidthPixels;
  if(!widths||!Number.isFinite(widths.min)||!Number.isFinite(widths.max)||widths.min<=0||widths.max<widths.min)return null;
  return {frames:value.frames,matched:value.matched,targets:value.targets,recall:value.matched/value.targets,
    falsePositives:positiveOnly?null:value.falsePositives,precision:null,sourceVideos:value.sourceVideos,sourceGroups:value.sourceGroups,
    targetWidthPixels:{min:widths.min,max:widths.max},scoreThreshold:value.scoreThreshold,iouThreshold:value.iouThreshold,
    labelAuthority:'human',positiveOnly,reused:value.reused===true,actualDistanceM:null,
    labelCautions:Array.isArray(value.labelCautions)?value.labelCautions.slice(0,6).filter(note=>typeof note==='string').map(note=>note.slice(0,1000)):[]};
}

// Only bounded aggregate evidence is exposed. Source paths, frame records,
// contributor identities and image bytes remain in the local experiment.
export async function readVideoDistanceReport(root){
  let artifact;
  for(const file of FILES){artifact=await saved(root,file);if(artifact.availability!=='missing')break;}
  if(artifact.availability!=='available')return empty(artifact.availability);
  const data=artifact.data,sample=data?.sample,progress=data?.progress,protocol=data?.protocol;
  if(data?.schema!=='mycoscout.video-distance-report.v1'||!['running','completed','completed_with_errors'].includes(data.status)||
    !sample||['positiveFrames','negativeFrames','targets','sourceVideos','sourceGroups'].some(key=>!count(sample[key]))||
    sample.sourceGroups>sample.sourceVideos||sample.sourceVideos>sample.positiveFrames+sample.negativeFrames||
    !['human','assistant'].includes(sample.labelAuthority)||!progress||!count(progress.completed)||!count(progress.total)||progress.completed>progress.total||
    data.status==='completed'&&progress.completed!==progress.total||!Array.isArray(data.rows)||data.rows.length>32||
    !protocol||!fraction(protocol.scoreThreshold)||!fraction(protocol.iouThreshold)||protocol.iouThreshold<=0||!count(protocol.tileGrid)||protocol.tileGrid<1||protocol.tileGrid>8)return empty('invalid');
  const rows=[],seen=new Set();let invalidConditions=0;
  for(const row of data.rows){
    if(!row||typeof row.native!=='boolean'||row.native&&row.targetPixels!==null||!row.native&&(!Number.isFinite(row.targetPixels)||row.targetPixels<=0||row.targetPixels>65536)){
      invalidConditions++;continue;
    }
    const key=row.native?'native':String(row.targetPixels);
    if(seen.has(key)){invalidConditions++;continue;}seen.add(key);
    const result={id:key,label:row.native?'Native video frames':`Requested ${row.targetPixels} px ${row.targetPixels>=32?'control':'stress test'}`,native:row.native,targetPixels:row.targetPixels,
      realizedTargetWidthPixels:realizedWidths(row)};
    for(const name of METHODS)result[name]=metric(row[name],sample);
    const measured=METHODS.map(name=>result[name]).filter(Boolean);
    if(measured.some(value=>value.frames!==measured[0].frames||value.targets!==measured[0].targets||value.negativeFrames!==measured[0].negativeFrames)){
      invalidConditions++;continue;
    }
    if(!measured.length&&data.status==='completed'){invalidConditions++;continue;}
    rows.push(result);
  }
  rows.sort((a,b)=>Number(b.native)-Number(a.native)||(b.targetPixels??0)-(a.targetPixels??0));
  const limitations=[
    'Native results use original video frames. Resized conditions measure synthetic pixel sensitivity, not a real camera approach.',
    'Frames from the same video are correlated. This selected sample is not an independent estimate of woodland performance.',
    'Detection scores and broad visual labels do not verify a mushroom species or edibility.',
  ];
  if(sample.labelAuthority==='assistant')limitations.unshift('Assistant visual labels are provisional and await a human check. These are agreement measurements against those labels.');
  if(Array.isArray(data.limitations))for(const note of data.limitations.slice(0,12))if(typeof note==='string'&&note.trim())limitations.push(note.slice(0,1000));
  if(invalidConditions)limitations.push(`${invalidConditions} condition(s) were omitted because their metrics or comparison denominators were invalid.`);
  const cropContext=await saved(root,'experiments/results/video-distance-context-v1/report.json');
  const candidate=await saved(root,'experiments/results/ai-candidate-distance-v1/report.json');
  const corpusCandidate=await saved(root,'experiments/results/ai-corpus-distance-v1/report.json');
  return {schema:'mycoscout.video-distance-summary.v1',availability:'available',status:data.status,
    generatedAt:typeof data.generatedAt==='string'&&Number.isFinite(Date.parse(data.generatedAt))?data.generatedAt:null,
    runVersion:['video-distance-v2','video-distance-v3'].includes(data.runVersion)?data.runVersion:'video-distance-v1',
    readOnly:true,actualDistanceMeasured:false,automaticTraining:false,automaticActivation:false,
    sample:Object.fromEntries(['positiveFrames','negativeFrames','targets','sourceVideos','sourceGroups','labelAuthority'].map(key=>[key,sample[key]])),
    progress:{completed:progress.completed,total:progress.total,unit:['method-results','model-forwards'].includes(progress.unit)?progress.unit:'steps'},rows,limitations,
    nativeBaseline:nativeEvidence(data.nativeBaseline,false),nativeSmallPositiveOnly:nativeEvidence(data.nativeSmallPositiveOnly,true),
    classificationRows:classificationRows(data.classificationRows,rows,sample),
    cropContextRows:cropContext.availability==='available'?cropContextRows(cropContext.data,rows,sample):[],
    aiCandidateDistance:candidate.availability==='available'?candidateDistance(candidate.data,rows,sample):null,
    corpusCandidateDistance:corpusCandidate.availability==='available'?candidateDistance(corpusCandidate.data,rows,sample,true):null,
    methodNote:`Fixed detector score threshold ${protocol.scoreThreshold}; box-overlap threshold ${protocol.iouThreshold}; tiled view ${protocol.tileGrid} × ${protocol.tileGrid}. Missing method results remain unavailable.`,
    negativeNote:'Negative controls are labeled video frames with no visible mushrooms. False detections are separate from unmatched boxes on positive frames; zero is shown only for a completed measured result.',
  };
}
