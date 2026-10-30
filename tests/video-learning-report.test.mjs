import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,dirname } from 'node:path';
import { readVideoLearningReport } from '../core/video-learning-report.mjs';

const evaluation=(matched,missed,falsePositives)=>({
  metrics:{frames:1,truePositives:matched,falseNegatives:missed,falsePositives,scoreThreshold:.3,iouThreshold:.5,
    sourceGroups:{'PRIVATE-SOURCE':{}},actualDistanceMeasured:false,
    pixelWidthBins:Object.fromEntries(['under4','4to8','8to16','16to32','32to64'].map(name=>[name,{total:0}]))},
  frames:[{frameId:'PRIVATE-FRAME',collection:'main',imageSha256:'a'.repeat(64),humanReviewRevision:2,notes:'PRIVATE-NOTE'}],
});
const base='experiments/results/assistant-video-pilot-comparison-v1/comparison.json';
const comparison=()=>({schema:'mycoscout.assistant-pilot-comparison.v1',humanFrames:1,snapshotSha256:'b'.repeat(64),baseline:evaluation(1,0,2),candidate:evaluation(0,1,0)});
async function fixture(callback) {
  const root=await mkdtemp(join(tmpdir(),'video-learning-summary-'));
  const save=async(path,value)=>{const file=join(root,path);await mkdir(dirname(file),{recursive:true});await writeFile(file,typeof value==='string'?value:JSON.stringify(value));};
  try{await callback(root,save);}finally{await rm(root,{recursive:true,force:true});}
}

test('missing saved reports stay unavailable, not zero-result success',()=>fixture(async root=>{
  const result=await readVideoLearningReport(root);
  assert.equal(result.pass.status,'unavailable');assert.equal(result.pass.processedFrames,null);assert.equal(result.pass.totalFrames,null);
  assert.equal(result.audit.reviewedFrames,null);assert.equal(result.training,null);assert.deepEqual(result.comparison.rows,[]);
  assert.equal(result.comparison.evidence.targetsBelow64,null);assert.ok(Object.values(result.availability).every(value=>value==='missing'));
}));

test('summarizes same-frame counts and strips all frame, reviewer and source details',()=>fixture(async(root,save)=>{
  await save('data/ai-video-review-v1/inputs.json',{schema:'mycoscout.ai-video-pass-inputs.v1',collections:[{frames:372,sources:7},{frames:60,sources:2}]});
  await save('data/ai-video-review-v1/status.json',{schema:'mycoscout.ai-video-pass-status.v1',status:'running',processedFrames:100,completedFrames:99,failedFrames:1,candidateBoxes:150,totalFrames:432,latestFrame:'PRIVATE-FRAME'});
  await save('data/ai-video-review-v1/assistant-audit.json',{schema:'mycoscout.assistant-audit.v1',reviews:[{collection:'main',frameId:'PRIVATE-FRAME',status:'visible',complete:true,reviewer:'PRIVATE-REVIEWER'}]});
  await save('data/assistant-video-pilot-dataset-v1/manifest.json',{schema:'mycoscout.assistant-video-training.v1',images:12,positiveImages:7,explicitNegativeImages:5,sourceGroups:['PRIVATE-SOURCE'],humanVerified:false});
  await save(base,comparison());
  await save('experiments/results/proposal-verifier-pilot-v1/model-card.json',{schema:'mycoscout.assistant-verifier-model-card.v1',snapshotSha256:'b'.repeat(64),baseline:evaluation(1,0,2),candidate:evaluation(1,0,1),activationEligible:false,automaticActivation:false});
  const result=await readVideoLearningReport(root);
  assert.equal(result.pass.totalFrames,432);assert.equal(result.pass.totalVideos,9);assert.equal(result.pass.failedFrames,1);
  assert.equal(result.training.frames,12);assert.equal(result.audit.reviewedFrames,1);
  assert.deepEqual(result.comparison.rows.map(row=>[row.id,row.matched,row.missed,row.falsePositives]),[['baseline',1,0,2],['pilot',0,1,0],['verifier',1,0,1]]);
  assert.equal(result.comparison.evidence.targetsBelow64,0);assert.equal(result.comparison.evidence.actualDistanceMeasured,false);
  assert.equal(JSON.stringify(result).includes('PRIVATE-'),false);assert.ok(JSON.stringify(result).length<6000);
}));

test('malformed reports fail locally while independent artifacts still load',()=>fixture(async(root,save)=>{
  await save('data/ai-video-review-v1/status.json','{"schema":');
  await save('data/ai-video-review-v1/assistant-audit.json',{schema:'mycoscout.assistant-audit.v1',reviews:'not an array'});
  await save(base,comparison());
  const result=await readVideoLearningReport(root);
  assert.equal(result.availability.pass,'invalid');assert.equal(result.pass.completedFrames,null);
  assert.equal(result.availability.audit,'invalid');assert.equal(result.audit.reviewedFrames,null);
  assert.equal(result.comparison.rows.length,2);
}));

test('different image versions, thresholds or snapshots are never combined',()=>fixture(async(root,save)=>{
  const changed=comparison();changed.candidate.frames[0].imageSha256='c'.repeat(64);await save(base,changed);
  assert.deepEqual((await readVideoLearningReport(root)).comparison.rows,[]);
  const threshold=comparison();threshold.candidate.metrics.scoreThreshold=.25;await save(base,threshold);
  assert.equal((await readVideoLearningReport(root)).comparison.status,'incomparable');
  await save(base,comparison());
  await save('experiments/results/proposal-verifier-pilot-v1/model-card.json',{schema:'mycoscout.assistant-verifier-model-card.v1',snapshotSha256:'d'.repeat(64),baseline:evaluation(1,0,2),candidate:evaluation(1,0,1)});
  const result=await readVideoLearningReport(root);assert.equal(result.comparison.rows.length,2);assert.equal(result.availability.verifier,'incomparable');
}));

test('contradictory progress counts are unavailable rather than clamped',()=>fixture(async(root,save)=>{
  await save('data/ai-video-review-v1/status.json',{schema:'mycoscout.ai-video-pass-status.v1',status:'completed',totalFrames:10,processedFrames:11,completedFrames:11,failedFrames:0,candidateBoxes:2});
  const result=await readVideoLearningReport(root);assert.equal(result.availability.pass,'invalid');assert.equal(result.pass.processedFrames,null);
}));

test('known disputed labels are cautioned only for their exact saved pixels and human revision',()=>fixture(async(root,save)=>{
  const data=comparison(),frame={collection:'main',frameId:'youtube-m9aGEh33w0I-024',humanReviewRevision:1,imageSha256:'3181fbf365d79f4716dd2e0bd42b349bfd1cd8e6cf75ca5537a5554f1ea45ef7'};
  data.baseline.frames=[frame];data.candidate.frames=[{...frame}];await save(base,data);
  let result=await readVideoLearningReport(root);assert.equal(result.comparison.labelCautions.length,1);assert.match(result.comparison.labelCautions[0],/disputed/);
  data.baseline.frames[0].humanReviewRevision=2;data.candidate.frames[0].humanReviewRevision=2;await save(base,data);
  result=await readVideoLearningReport(root);assert.deepEqual(result.comparison.labelCautions,[]);assert.equal(result.comparison.rows.length,2);
}));
