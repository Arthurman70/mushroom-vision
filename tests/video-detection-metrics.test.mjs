import test from 'node:test';
import assert from 'node:assert/strict';
import { videoDetectionMetrics } from '../core/video-detection-metrics.mjs';

function frame(id='frame1',sourceId='video1',group='session:1'){
  return {id,sourceId,sourceGroupId:group,width:100,height:100,sha256:'a'.repeat(64),timestampSeconds:1,
    source:{id:sourceId,videoHash:'b'.repeat(64),sourceGroupId:group}};
}
const box=(id='truth1',width=7)=>({id,bboxPixelsXYXY:[0,0,width,10]});
function snapshot(frames,review={complete:true,status:'visible',boxes:[box()]}){
  return {reviews:Object.fromEntries(frames.map(f=>[f.id,{revision:1,review:structuredClone(review)}]))};
}
function run(f,{id='run1',createdAt='2026-01-01',candidates=[{id:'pred1',bboxPixelsXYXY:[0,0,7,10],confidence:.5}],scale=1,threshold=.25,model='Fixture model',revision='abc',methods=['original']}={}){
  return {id,createdAt,status:'completed',image:{sha256:f.sha256},videoProvenance:structuredClone(f),settings:{methods,scale,boxThreshold:threshold,tileGrid:2},
    result:{image:{sha256:f.sha256,width:f.width,height:f.height,workingWidth:100*scale,workingHeight:100*scale},model:{model,resolvedRevision:revision},
      methods:methods.map(id=>({id,candidates})),colorCues:[{id:'cue1',bboxPixelsXYXY:[0,0,7,10]}]}};
}

test('metrics await complete reviews and exact matching tests without fabricating percentages',()=>{
  const f=frame();let result=videoDetectionMetrics([f],{reviews:{frame1:{review:null,revision:0}}},[run(f)]);
  assert.equal(result.status,'awaiting_complete_labels');assert.equal(result.groups.length,0);assert.equal(result.excluded.unreviewed,1);
  result=videoDetectionMetrics([f],snapshot([f]),[]);assert.equal(result.status,'awaiting_matching_tests');
  for(const review of [{complete:false,status:'visible',boxes:[box()]},{complete:false,status:'uncertain',boxes:[]},{complete:false,status:'visible',boxes:[]}])
    assert.equal(videoDetectionMetrics([f],snapshot([f],review),[run(f)]).groups.length,0);
});

test('actual IoU matches produce original-width recall bins and explicitly scaled frame widths',()=>{
  const f=frame(),truth={complete:true,status:'visible',boxes:[box(),{id:'truth2',bboxPixelsXYXY:[20,20,36,40]}]},result=videoDetectionMetrics([f],snapshot([f],truth),[run(f,{scale:.5})]),group=result.groups[0];
  assert.equal(group.frames,1);assert.equal(group.groundTruth,2);assert.equal(group.truePositives,1);assert.equal(group.falseNegatives,1);
  assert.equal(group.recall,.5);assert.equal(group.pixelWidthBins[1].recall,1);assert.equal(group.pixelWidthBins[3].recall,0);assert.equal(group.pixelWidthBins[0].recall,null);
  assert.equal(group.observations[0].boxes[0].originalPixelWidth,7);assert.equal(group.observations[0].boxes[0].workingFramePixelWidth,3.5);
  assert.equal(result.pixelWidthUnit,'original-frame pixels');
});

test('duplicate detections remain false positives; color cues cannot become matches',()=>{
  const f=frame(),pred={id:'one',bboxPixelsXYXY:[0,0,7,10],confidence:.5};
  let g=videoDetectionMetrics([f],snapshot([f]),[run(f,{candidates:[pred,{...pred,id:'two'}]})]).groups[0];
  assert.equal(g.truePositives,1);assert.equal(g.falsePositives,1);assert.equal(g.precision,.5);
  g=videoDetectionMetrics([f],snapshot([f]),[run(f,{candidates:[]})]).groups[0];assert.equal(g.truePositives,0);assert.equal(g.recall,0);assert.equal(g.precision,null);
});

test('fully reviewed negative frames count false alarms without a recall denominator',()=>{
  const f=frame(),g=videoDetectionMetrics([f],snapshot([f],{complete:true,status:'absent',boxes:[]}),[run(f)]).groups[0];
  assert.equal(g.negativeFrames,1);assert.equal(g.falsePositives,1);assert.equal(g.precision,0);assert.equal(g.recall,null);
  assert.ok(g.pixelWidthBins.every(bin=>bin.total===0&&bin.recall===null));
});

test('latest completed repeat per configuration is chosen and distinct methods/settings/models remain separate',()=>{
  const f=frame(),runs=[run(f,{id:'old',candidates:[],createdAt:'2025-01-01'}),run(f,{id:'new'}),run(f,{id:'scale',scale:.5}),run(f,{id:'threshold',threshold:.6}),
    run(f,{id:'model',model:'Another model'}),run(f,{id:'revision',revision:'new'}),run(f,{id:'method',methods:['tiles']})];
  const result=videoDetectionMetrics([f],snapshot([f]),runs);assert.equal(result.groups.length,6);assert.equal(result.excluded.duplicateRuns,1);
  assert.ok(result.groups.every(g=>g.frames===1&&g.truePositives===1));assert.ok(!result.groups.some(g=>g.observations.some(x=>x.runId==='old')));
});

test('pixel and source-group counts retain correlated frame membership',()=>{
  const frames=[frame('f1'),frame('f2','video2'),frame('f3','video3','session:2')],g=videoDetectionMetrics(frames,snapshot(frames),frames.map(f=>run(f,{id:f.id}))).groups[0];
  assert.equal(g.frames,3);assert.equal(g.sourceVideos,3);assert.equal(g.sourceGroups,2);assert.equal(g.bySourceGroup.find(x=>x.sourceGroupId==='session:1').frames,2);
  assert.equal(g.pixelWidthBins[1].total,3);
});

test('mismatched hash, source video, geometry, group or timestamp never joins labels',()=>{
  const f=frame(),base=run(f),mutations=[r=>r.image.sha256='c'.repeat(64),r=>r.result.image.width=99,r=>r.videoProvenance.id='wrong',r=>r.videoProvenance.sha256='d'.repeat(64),
    r=>r.videoProvenance.source.videoHash='e'.repeat(64),r=>r.videoProvenance.sourceId='video2',r=>r.videoProvenance.sourceGroupId='session:wrong',r=>r.videoProvenance.timestampSeconds=2];
  for(const mutate of mutations){const modified=structuredClone(base);mutate(modified);const result=videoDetectionMetrics([f],snapshot([f]),[modified]);
    assert.equal(result.groups.length,0);assert.equal(result.excluded.provenanceMismatchRuns,1);}
});

test('current human revision is used and pending model results are excluded',()=>{
  const f=frame(),s=snapshot([f]);s.reviews.frame1.revision=7;s.reviews.frame1.review.boxes=[box('corrected',64)];
  const r=run(f);let result=videoDetectionMetrics([f],s,[r]);assert.equal(result.groups[0].observations[0].reviewRevision,7);assert.equal(result.groups[0].pixelWidthBins[5].total,1);
  r.status='running';result=videoDetectionMetrics([f],s,[r]);assert.equal(result.groups.length,0);
});
