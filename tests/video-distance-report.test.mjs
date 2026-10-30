import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,dirname } from 'node:path';
import { readVideoDistanceReport } from '../core/video-distance-report.mjs';

const metric=()=>({frames:1,matched:0,targets:1,falsePositives:2,negativeFrames:2,negativeFalsePositives:3,negativeFramesWithDetections:1});
const sample=()=>({schema:'mycoscout.video-distance-report.v1',status:'running',generatedAt:'2026-10-29T18:00:00Z',
  sample:{positiveFrames:1,negativeFrames:2,targets:1,sourceVideos:2,sourceGroups:2,labelAuthority:'human',privateFrame:'PRIVATE'},
  progress:{completed:3,total:24},protocol:{scoreThreshold:.3,iouThreshold:.5,tileGrid:2},
  rows:[{id:'PRIVATE',native:false,targetPixels:.69,raw:metric(),color:metric(),tiles:metric()}],
  limitations:['Selected source sample.'],frames:[{frameId:'PRIVATE',path:'PRIVATE'}]});
async function fixture(callback){
  const root=await mkdtemp(join(tmpdir(),'video-distance-summary-')),file=join(root,'experiments/results/video-distance-v1/report.json');
  const save=async value=>{await mkdir(dirname(file),{recursive:true});await writeFile(file,typeof value==='string'?value:JSON.stringify(value));};
  try{await callback(root,save);}finally{await rm(root,{recursive:true,force:true});}
}

test('missing, malformed and oversized video tests do not claim zero-result success',()=>fixture(async(root,save)=>{
  let result=await readVideoDistanceReport(root);assert.equal(result.availability,'missing');assert.equal(result.sample,null);assert.deepEqual(result.rows,[]);
  for(const value of ['{broken',JSON.stringify({...sample(),schema:'wrong'}),' '.repeat(4*1024*1024+1)]){
    await save(value);result=await readVideoDistanceReport(root);assert.equal(result.availability,'invalid');assert.deepEqual(result.rows,[]);
  }
}));

test('video summaries retain fractional pixels and measured denominators, never physical-distance claims',()=>fixture(async(root,save)=>{
  const data=sample();data.actualDistanceMeasured=true;data.rows[0].raw.recall=1;await save(data);
  const result=await readVideoDistanceReport(root);
  assert.equal(result.availability,'available');assert.equal(result.status,'running');assert.equal(result.rows[0].targetPixels,.69);
  assert.equal(result.rows[0].raw.recall,0);assert.equal(result.rows[0].raw.negativeFalsePositives,3);assert.equal(result.rows[0].raw.negativeFramesWithDetections,1);
  assert.equal(result.actualDistanceMeasured,false);assert.equal(result.automaticActivation,false);assert.equal(JSON.stringify(result).includes('PRIVATE'),false);
}));

test('requested pixel widths stay distinct from validated realized integer-resize bounds',()=>fixture(async(root,save)=>{
  const data=sample(),row=data.rows[0];
  row.realizedTargetWidthPixels={min:.653125,max:.731953125};row.realizedTargetWidths=[.653125,.7,.731953125];
  await save(data);let result=await readVideoDistanceReport(root);
  assert.equal(result.rows[0].targetPixels,.69);assert.match(result.rows[0].label,/Requested 0\.69 px/);
  assert.deepEqual(result.rows[0].realizedTargetWidthPixels,row.realizedTargetWidthPixels);
  delete row.realizedTargetWidths;await save(data);assert.deepEqual((await readVideoDistanceReport(root)).rows[0].realizedTargetWidthPixels,row.realizedTargetWidthPixels);
  for(const bounds of [{min:0,max:.7},{min:.8,max:.7},{min:.7,max:Infinity},{min:.7,max:65537},null]){
    row.realizedTargetWidthPixels=bounds;await save(data);result=await readVideoDistanceReport(root);
    assert.equal(result.rows[0].realizedTargetWidthPixels,null);assert.equal(result.rows[0].targetPixels,.69);
  }
  row.realizedTargetWidthPixels={min:.6,max:.8};row.realizedTargetWidths=[.6,.9];await save(data);
  assert.equal((await readVideoDistanceReport(root)).rows[0].realizedTargetWidthPixels,null);
  row.native=true;row.targetPixels=null;row.realizedTargetWidthPixels={min:66.93,max:259.85};row.realizedTargetWidths=[66.93,167.32,259.85];await save(data);
  result=await readVideoDistanceReport(root);assert.equal(result.rows[0].label,'Native video frames');assert.deepEqual(result.rows[0].realizedTargetWidthPixels,{min:66.93,max:259.85});
}));

test('incomparable method denominators are omitted and unavailable negative controls stay distinct from zero',()=>fixture(async(root,save)=>{
  const data=sample();data.rows[0].color.targets=0;data.rows.push({native:true,targetPixels:null,raw:{...metric(),negativeFrames:0,negativeFalsePositives:0,negativeFramesWithDetections:0}});
  await save(data);const result=await readVideoDistanceReport(root);
  assert.equal(result.rows.length,1);assert.equal(result.rows[0].native,true);assert.equal(result.rows[0].raw.negativeFrames,0);
  assert.equal(result.rows[0].color,null);assert.equal(result.rows[0].tiles,null);assert.match(result.limitations.at(-1),/1 condition/);
  const invalid=sample();invalid.rows[0].raw.negativeFramesWithDetections=3;await save(invalid);
  assert.equal((await readVideoDistanceReport(root)).rows[0].raw,null);
}));

test('invalid progress and contradictory sample counts fail closed',()=>fixture(async(root,save)=>{
  for(const mutate of [data=>{data.progress.completed=25},data=>{data.status='completed'},data=>{data.sample.sourceGroups=3},data=>{data.sample.labelAuthority='confirmed-by-ai'}]){
    const data=sample();mutate(data);await save(data);assert.equal((await readVideoDistanceReport(root)).availability,'invalid');
  }
}));

test('native partial-label evidence shows positive recall only and remains separate from the complete baseline',()=>fixture(async(root,save)=>{
  const data=sample(),native={frames:3,matched:1,targets:3,sourceVideos:1,sourceGroups:1,targetWidthPixels:{min:41.25,max:61.5},
    scoreThreshold:.3,iouThreshold:.5,labelAuthority:'human',reused:true,actualDistanceM:10};
  data.nativeBaseline={...native,frames:19,targets:12,matched:10,falsePositives:4,targetWidthPixels:{min:64,max:500}};
  data.nativeSmallPositiveOnly={...native,precision:.99,falsePositives:null};
  data.nativeEvidence=[{frameId:'PRIVATE',imageSha256:'PRIVATE',reviewer:'PRIVATE'}];await save(data);
  let result=await readVideoDistanceReport(root);assert.equal(result.nativeBaseline.falsePositives,4);assert.equal(result.nativeSmallPositiveOnly.recall,1/3);
  assert.equal(result.nativeSmallPositiveOnly.falsePositives,null);assert.equal(result.nativeSmallPositiveOnly.precision,null);assert.equal(result.nativeSmallPositiveOnly.actualDistanceM,null);
  assert.deepEqual(result.nativeSmallPositiveOnly.targetWidthPixels,{min:41.25,max:61.5});assert.equal(JSON.stringify(result).includes('PRIVATE'),false);
  data.nativeSmallPositiveOnly.falsePositives=2;await save(data);result=await readVideoDistanceReport(root);
  assert.equal(result.nativeSmallPositiveOnly,null);assert.equal(result.nativeBaseline.matched,10);
}));

test('the fresh run takes precedence and an invalid new report never falls back to stale success',()=>fixture(async(root,save)=>{
  await save(sample());const file=join(root,'experiments/results/video-distance-v2/report.json');
  await mkdir(dirname(file),{recursive:true});
  const data={...sample(),runVersion:'video-distance-v2'};data.progress.completed=2;
  await writeFile(file,JSON.stringify(data));let result=await readVideoDistanceReport(root);
  assert.equal(result.runVersion,'video-distance-v2');assert.equal(result.progress.completed,2);
  await writeFile(file,'{broken');result=await readVideoDistanceReport(root);
  assert.equal(result.availability,'invalid');assert.deepEqual(result.rows,[]);
}));

test('known-box category agreement preserves provisional authority and rejects impossible counts',()=>fixture(async(root,save)=>{
  const data=sample();data.rows[0].targetPixels=6.9;
  const counts={top1Agreement:1,top3Agreement:1,retained:1,retainedAgreement:1,abstained:0};
  data.classificationRows=[{id:'6p9px',targets:1,productionEligible:0,referenceAuthority:'assistant',referenceClass:'true_morels',
    humanSpeciesGroundTruth:false,baseline:counts,expanded:{...counts,retainedAgreement:0,top1Agreement:0},privatePath:'PRIVATE'}];
  await save(data);let result=await readVideoDistanceReport(root);
  assert.equal(result.classificationRows.length,1);assert.equal(result.classificationRows[0].referenceAuthority,'assistant');
  assert.equal(result.classificationRows[0].productionEligible,0);assert.equal(result.classificationRows[0].expanded.retainedAgreement,0);
  assert.equal(JSON.stringify(result).includes('PRIVATE'),false);
  for(const mutate of [row=>row.referenceAuthority='human',row=>row.humanSpeciesGroundTruth=true,row=>row.productionEligible=2,
    row=>row.baseline={...counts,retainedAgreement:2},row=>row.expanded={...counts,abstained:1}]){
    const bad=structuredClone(data);mutate(bad.classificationRows[0]);await save(bad);
    assert.deepEqual((await readVideoDistanceReport(root)).classificationRows,[]);
  }
}));

test('context diagnostic exposes bounded aggregates without activating a crop policy',()=>fixture(async(root,save)=>{
  await save(sample());const file=join(root,'experiments/results/video-distance-context-v1/report.json');
  await mkdir(dirname(file),{recursive:true});
  const score={top1Agreement:1,retained:1,retainedAgreement:1},data={schema:'mycoscout.crop-context-report.v1',status:'completed',
    protocol:{referenceAuthority:'assistant',referenceClass:'true_morels',humanSpeciesGroundTruth:false,knownTargetLocation:true,localizationEvaluated:false,modelActivation:false},
    rows:[{condition:'0p69px',targets:1,productionEligible:0,policies:Object.fromEntries(['tight','fitted','context'].map(p=>[p,{baseline:score,expanded:score}]))}],records:[{path:'PRIVATE'}]};
  await writeFile(file,JSON.stringify(data));let result=await readVideoDistanceReport(root);
  assert.equal(result.cropContextRows.length,1);assert.equal(result.cropContextRows[0].productionEligible,0);
  assert.equal(result.automaticActivation,false);assert.equal(JSON.stringify(result).includes('PRIVATE'),false);
  data.protocol.modelActivation=true;await writeFile(file,JSON.stringify(data));result=await readVideoDistanceReport(root);
  assert.deepEqual(result.cropContextRows,[]);
}));

test('AI candidate results retain separate detector authority and reject activation or threshold changes',()=>fixture(async(root,save)=>{
  await save(sample());const file=join(root,'experiments/results/ai-candidate-distance-v1/report.json');
  await mkdir(dirname(file),{recursive:true});
  const data={schema:'mushroom-vision.ai-candidate-distance-report.v1',status:'completed',modelActivation:false,productionModelChanged:false,
    protocol:{trainingLabelAuthority:'ai-visual-review',evaluationBoxAuthority:'human',activationEligible:false,thresholdSelectionOnTest:false,additionalTraining:false,
      scoreThreshold:.3,iouThreshold:.5,epochs:1,model:'fasterrcnn_mobilenet_v3_large_320_fpn',trainingFrames:6,trainingSourceGroups:3,trainingBoxes:5},
    rows:[{condition:'0p69px',frames:1,targets:1,matched:0,unmatchedDetections:2,negativeFrames:0,negativeFalsePositives:0}],records:[{path:'PRIVATE'}]};
  await writeFile(file,JSON.stringify(data));let result=await readVideoDistanceReport(root);
  assert.equal(result.aiCandidateDistance.trainingLabelAuthority,'ai-visual-review');assert.equal(result.aiCandidateDistance.evaluationBoxAuthority,'human');
  assert.equal(result.aiCandidateDistance.activationEligible,false);assert.equal(JSON.stringify(result).includes('PRIVATE'),false);
  for(const change of [d=>d.modelActivation=true,d=>d.protocol.scoreThreshold=.25,d=>d.protocol.trainingLabelAuthority='human',d=>d.protocol.additionalTraining=true,d=>d.protocol.epochs=2]){
    const invalid=structuredClone(data);change(invalid);await writeFile(file,JSON.stringify(invalid));assert.equal((await readVideoDistanceReport(root)).aiCandidateDistance,null);
  }
}));

test('corpus candidate remains separate and exposes only bounded aggregate training evidence',()=>fixture(async(root,save)=>{
  await save(sample());const file=join(root,'experiments/results/ai-corpus-distance-v1/report.json');
  await mkdir(dirname(file),{recursive:true});
  const data={schema:'mushroom-vision.ai-corpus-distance-report.v1',status:'completed',modelActivation:false,productionModelChanged:false,
    protocol:{trainingLabelAuthority:'ai-visual-review',evaluationBoxAuthority:'human',activationEligible:false,thresholdSelectionOnTest:false,additionalTraining:false,
      scoreThreshold:.3,iouThreshold:.5,epochs:8,selectedEpoch:3,optimizerSteps:816,model:'fasterrcnn_mobilenet_v3_large_320_fpn',
      trainingFrames:82,trainingSourceGroups:3,trainingBoxes:32,hardNegativeCrops:{hat:16,wood:1,PRIVATE:9}},
    rows:[{condition:'0p69px',frames:1,targets:1,matched:0,unmatchedDetections:2,negativeFrames:0,negativeFalsePositives:0}],records:[{path:'PRIVATE'}]};
  await writeFile(file,JSON.stringify(data));let result=await readVideoDistanceReport(root);
  assert.equal(result.aiCandidateDistance,null);assert.equal(result.corpusCandidateDistance.epochs,8);
  assert.deepEqual(result.corpusCandidateDistance.hardNegativeCrops,{hat:16,wood:1});assert.equal(JSON.stringify(result).includes('PRIVATE'),false);
  for(const change of [d=>d.protocol.selectedEpoch=9,d=>d.protocol.epochs=13,d=>d.protocol.optimizerSteps=0,d=>d.protocol.activationEligible=true]){
    const invalid=structuredClone(data);change(invalid);await writeFile(file,JSON.stringify(invalid));
    assert.equal((await readVideoDistanceReport(root)).corpusCandidateDistance,null);
  }
}));
