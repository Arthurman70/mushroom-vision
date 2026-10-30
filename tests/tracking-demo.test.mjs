import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,readFile,rm } from 'node:fs/promises';
import { join,resolve,sep } from 'node:path';
import { tmpdir } from 'node:os';
import { createHash } from 'node:crypto';
import { CandidateTracker,LatestFrameGate,closeLookProposal,syntheticFrame } from '../public/tracking-logic.mjs';
import { readTrackingDemo,readTrackingDemoVideo,validateTrackingDemo } from '../core/tracking-demo.mjs';

const box=(x=20,y=20,w=40,h=50,confidence=.7)=>({bbox:[x,y,x+w,y+h],confidence});
const frame=(timestampSeconds,detections=[box()])=>({timestampSeconds,detections,width:640,height:480});
test('candidate IDs persist across nearby detections; duplicate samples do not invent observations',()=>{
  const tracker=new CandidateTracker(),first=tracker.update(frame(0));const id=first[0].id;
  const second=tracker.update(frame(.5,[box(25,21)]));assert.equal(second[0].id,id);assert.equal(second[0].observations,2);
  assert.equal(tracker.update(frame(.5,[box(100,100)]))[0].observations,2);
  const three=tracker.update(frame(1,[box(30,22),box(450,350)]));assert.equal(three.length,2);assert.equal(three[0].observations,3);
  assert.equal(new Set(three.map(t=>t.id)).size,2);assert.equal(three[1].observations,1);
});
test('observed, predicted, stale and lost states have bounded motion and retain measured pixel size',()=>{
  const tracker=new CandidateTracker();tracker.update(frame(0));tracker.update(frame(.5,[box(25,20)]));
  assert.equal(tracker.snapshot(.5)[0].status,'observed');assert.equal(tracker.snapshot(1)[0].status,'predicted');
  assert.equal(tracker.snapshot(2)[0].stale,true);assert.equal(tracker.snapshot(5)[0].status,'lost');
  assert.equal(tracker.snapshot(2)[0].observedPixelWidth,40);assert.equal(tracker.snapshot(2)[0].gps,null);
  assert.deepEqual(tracker.snapshot(2)[0].bbox,tracker.snapshot(4)[0].bbox);
});
test('backward or dimension-changing samples require reset; reset invalidates old track IDs',()=>{
  const tracker=new CandidateTracker(),old=tracker.update(frame(5))[0].id;
  assert.throws(()=>tracker.update(frame(4)),/Reset/);assert.throws(()=>tracker.update({...frame(6),width:1280}),/dimensions/);
  tracker.reset('after-seek');const fresh=tracker.update(frame(0))[0];assert.notEqual(fresh.id,old);assert.equal(fresh.sourceId,'after-seek');
  assert.equal(tracker.update(frame(1,[box(-1),box(635),{bbox:[0,0,1,1],confidence:2}])).filter(t=>t.status==='observed').length,0);
});
test('associations are one-to-one and input/history capacity stays bounded',()=>{
  const tracker=new CandidateTracker({maxTracks:2});tracker.update(frame(0,[box(10),box(300)]));
  const result=tracker.update(frame(.5,[box(15),box(305),box(500)]));assert.equal(result.length,2);assert.ok(result.every(t=>t.observations===2));
  assert.equal(tracker.update(frame(20,[])).length,0);assert.throws(()=>new CandidateTracker({maxTracks:1000}));
});
test('nested duplicate detector boxes do not create two identities for one sample',()=>{
  const tracker=new CandidateTracker(),detections=[box(20,20,40,40,.8),box(15,15,50,50,.5)];
  assert.equal(tracker.update(frame(0,detections)).length,1);assert.equal(detections.length,2);
});
test('closer-look tasks are proposals with no invented coordinates or authority',()=>{
  const tracker=new CandidateTracker();const first=tracker.update(frame(0))[0];assert.throws(()=>closeLookProposal(first,{sourceId:'test',timestampSeconds:0}),/two recent/);
  const current=tracker.update(frame(.5))[0],proposal=closeLookProposal(current,{sourceId:'test',timestampSeconds:.5});
  assert.equal(proposal.kind,'closer_photo');assert.equal(proposal.actualDistanceM,null);assert.equal(proposal.gps,null);assert.equal(proposal.position,null);assert.equal(proposal.actuationEnabled,false);
  assert.equal(closeLookProposal(current,{sourceId:'test',timestampSeconds:.5,actualDistanceM:50}).kind,'habitat_followup');
  assert.throws(()=>closeLookProposal(tracker.snapshot(3)[0],{sourceId:'test',timestampSeconds:3}),/recent/);
});
test('latest-frame gate skips overlapping work and discards stale source responses',async()=>{
  const gate=new LatestFrameGate();let finish,calls=0,accepted=0;
  const first=gate.run(()=>{calls++;return new Promise(resolve=>{finish=resolve;});},()=>accepted++);
  assert.deepEqual(await gate.run(()=>{calls++;},()=>accepted++),{skipped:true});gate.reset();finish('old');
  assert.deepEqual(await first,{discarded:true});assert.equal(calls,1);assert.equal(accepted,0);assert.equal(gate.pending,false);
  assert.deepEqual(await gate.run(async()=>1,()=>accepted++),{accepted:true});assert.equal(accepted,1);
  await assert.rejects(gate.run(async()=>{throw new Error('offline');},()=>{}));assert.equal(gate.pending,false);
});
test('synthetic example explicitly exercises occlusion and loss',()=>{
  const tracker=new CandidateTracker();tracker.update(syntheticFrame(8));tracker.update(syntheticFrame(12.5));
  assert.equal(tracker.snapshot(12.5)[0].status,'lost');assert.equal(syntheticFrame(13.5).detections.length,1);
});
function manifest(videoBytes){return {schema:'mushroom-vision.tracking-demo.v1',inferenceMode:'precomputed_replay',videoSha256:createHash('sha256').update(videoBytes).digest('hex'),videoBytes:videoBytes.length,
  source:{title:'Licensed fixture',attribution:'Test author',license:'CC BY-SA 2.0',sourceUrl:'https://example.org/video',licenseUrl:'https://creativecommons.org/licenses/by-sa/2.0/',videoHash:'a'.repeat(64)},
  actuationEnabled:false,humanVerified:false,trainingAllowed:false,width:640,height:480,durationSeconds:2,
  frames:[0,1].map(t=>({timestampSeconds:t,width:640,height:480,imageSha256:'b'.repeat(64),processingSeconds:7,detections:[box()]}))};}
test('replay endpoint preserves recorded provenance, rejects tampering and gives an explicit missing-asset result',async()=>{
  const root=await mkdtemp(join(tmpdir(),'tracking-demo-')),bytes=Buffer.from('media-fixture');
  try{assert.equal((await readTrackingDemo(root)).available,false);await mkdir(join(root,'data/tracking-demo-v1'),{recursive:true});const data=manifest(bytes);
    await writeFile(join(root,'data/tracking-demo-v1/manifest.json'),JSON.stringify(data));await writeFile(join(root,'data/tracking-demo-v1/replay.mp4'),bytes);
    const publicData=await readTrackingDemo(root);assert.equal(publicData.available,true);assert.equal(publicData.inferenceMode,'precomputed_replay');assert.equal(publicData.videoSha256,undefined);
    assert.deepEqual((await readTrackingDemoVideo(root)).bytes,bytes);
    await writeFile(join(root,'data/tracking-demo-v1/replay.mp4'),'modified');await assert.rejects(readTrackingDemoVideo(root),/changed|integrity/);
    for(const changed of [{...data,humanVerified:true},{...data,frames:[data.frames[1],data.frames[0]]},{...data,frames:[{...data.frames[0],detections:[box(-1)]},data.frames[1]]}])assert.throws(()=>validateTrackingDemo(changed));
  }finally{assert.ok(resolve(root).startsWith(resolve(tmpdir())+sep));await rm(root,{recursive:true,force:true});}
});
test('page and client identify replay/synthetic modes and keep camera capture behind a user click',async()=>{
  const html=await readFile(new URL('../public/tracking-lab.html',import.meta.url),'utf8'),js=await readFile(new URL('../public/tracking-lab.js',import.meta.url),'utf8');
  assert.match(html,/PRECOMPUTED REPLAY/);assert.match(js,/SYNTHETIC · NOT REAL FOOTAGE/);assert.match(js,/\$\('start-camera'\)\.onclick=async/);
  assert.equal(js.includes('/api/live/tasks'),false);assert.match(js,/requestCount>=60/);
});
