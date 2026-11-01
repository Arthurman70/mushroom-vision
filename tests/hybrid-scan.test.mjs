import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { HybridScanScheduler,InvestigationLog,investigationCues } from '../public/hybrid-scan.mjs';

function accept(scheduler,timestampSeconds,options={}){const plan=scheduler.plan({timestampSeconds,...options});assert.equal(scheduler.commit(plan),true);return plan;}
test('hybrid starts fast and uses source time for periodic detail',()=>{
  const scheduler=new HybridScanScheduler();assert.equal(accept(scheduler,50).stage,'fast');
  assert.equal(accept(scheduler,54.99).stage,'fast');assert.equal(accept(scheduler,55).stage,'ground-detail');
  assert.equal(accept(scheduler,55.1).stage,'fast');assert.equal(accept(scheduler,57).stage,'fast');assert.equal(accept(scheduler,60).stage,'ground-detail');
  assert.equal(accept(scheduler,60.1).stage,'fast');assert.equal(accept(scheduler,62).stage,'fast');const whole=accept(scheduler,65);
  assert.equal(whole.stage,'whole-frame-detail');assert.equal(whole.profile.id,'full-tiles');assert.deepEqual(whole.profile.tileRegion,[0,0,1,1]);
});
test('slow camera processing cannot create all-detail scans or catch-up backlog',()=>{
  const scheduler=new HybridScanScheduler();assert.equal(accept(scheduler,0).stage,'fast');
  assert.equal(accept(scheduler,8).stage,'fast');assert.equal(accept(scheduler,16).stage,'ground-detail');assert.equal(accept(scheduler,40).stage,'fast');
  assert.equal(accept(scheduler,48).stage,'fast');assert.equal(accept(scheduler,56).stage,'ground-detail');assert.equal(accept(scheduler,1000).stage,'fast');
  assert.equal(accept(scheduler,1008).stage,'fast');assert.equal(accept(scheduler,1016).stage,'whole-frame-detail');assert.equal(scheduler.detailCount,3);
});
test('paused recording does not advance detail time, and seeking requires reset',()=>{
  const scheduler=new HybridScanScheduler();accept(scheduler,10);assert.equal(accept(scheduler,10).stage,'fast');
  assert.equal(scheduler.fastFramesSinceDetail,1);assert.equal(accept(scheduler,12).stage,'fast');assert.equal(accept(scheduler,15).stage,'ground-detail');assert.throws(()=>scheduler.plan({timestampSeconds:14}));
  scheduler.reset();assert.equal(accept(scheduler,2).stage,'fast');assert.equal(scheduler.detailCount,0);
});
test('discarded or repeated result cannot advance the detail schedule',()=>{
  const scheduler=new HybridScanScheduler();accept(scheduler,0);accept(scheduler,2);const pending=scheduler.plan({timestampSeconds:5});
  assert.equal(scheduler.detailCount,0);assert.equal(scheduler.plan({timestampSeconds:8}).stage,'ground-detail');
  scheduler.reset();assert.equal(scheduler.commit(pending),false);assert.equal(accept(scheduler,10).stage,'fast');
  accept(scheduler,12);const next=scheduler.plan({timestampSeconds:15});assert.equal(scheduler.commit(next),true);assert.equal(scheduler.commit(next),false);
});
test('only effective ground profile uses selected boundary; backend never receives hybrid',()=>{
  const scheduler=new HybridScanScheduler();accept(scheduler,0);accept(scheduler,2);const ground=accept(scheduler,5,{groundTop:.7});assert.deepEqual(ground.profile.tileRegion,[0,.7,1,1]);
  for(const mode of ['fast','ground-tiles','full-tiles']){const plan=scheduler.plan({timestampSeconds:6,mode,groundTop:.7});assert.equal(plan.profile.id,mode);if(mode==='fast')assert.equal(plan.profile.tileRegion,undefined);if(mode==='full-tiles')assert.deepEqual(plan.profile.tileRegion,[0,0,1,1]);}
});
test('invalid scheduler times and controls fail closed',()=>{
  for(const interval of [1,61,NaN,'5',2.5])assert.throws(()=>new HybridScanScheduler({detailIntervalSeconds:interval}));
  assert.throws(()=>new HybridScanScheduler({minFastFrames:1}));
  const scheduler=new HybridScanScheduler();for(const timestampSeconds of [NaN,-1,'3',86401])assert.throws(()=>scheduler.plan({timestampSeconds}));
  for(const options of [{mode:'unknown'},{groundTop:-.1},{groundTop:.81},{groundTop:'0'}])assert.throws(()=>scheduler.plan({timestampSeconds:0,...options}));
});
const evidence=(timestampSeconds=1,stage='ground-detail')=>({cues:investigationCues({sourceId:'test',timestampSeconds,stage,width:100,height:80,detections:[{bbox:[2,3,30,20],confidence:.8}]}),thumbnail:'data:image/jpeg;base64,fixture',thumbnailSize:[100,80]});
test('persistent investigation log survives empty fast scans and deduplicates exact frame/stage',()=>{
  const log=new InvestigationLog();log.add(evidence());log.add({cues:[]});assert.equal(log.frames.length,1);assert.equal(log.cues.length,1);
  log.add(evidence());assert.equal(log.frames.length,1);log.add(evidence(1,'fast'));assert.equal(log.frames.length,2);
  assert.equal(log.frames[0].timestampSeconds,1);assert.equal(log.frames[0].thumbnail,'data:image/jpeg;base64,fixture');
  log.reset();assert.deepEqual(log.cues,[]);assert.deepEqual(log.frames,[]);
});
test('investigation evidence remains bounded and never mixes timestamps into one captured image',()=>{
  const log=new InvestigationLog({maxFrames:2,maxCues:2});for(const time of [1,2,3])log.add(evidence(time));
  assert.deepEqual(log.frames.map(frame=>frame.timestampSeconds),[2,3]);assert.equal(log.cues.length,2);
  const mixed=evidence(4);mixed.cues.push(...evidence(5).cues);assert.throws(()=>log.add(mixed));
  const wrong=evidence(4);wrong.cues[0].gps={lat:1,lng:2};assert.throws(()=>log.add(wrong));
  const original=evidence(4);log.add(original);original.cues[0].imageBox[0]=99;assert.equal(log.frames.at(-1).cues[0].imageBox[0],2);
});
test('investigation cues retain only frame evidence and never create species, range or actionable tasks',()=>{
  const cues=investigationCues({sourceId:'test',timestampSeconds:15,stage:'whole-frame-detail',width:100,height:80,detections:[{bbox:[2,3,30,20],confidence:.8,label:'chicken of the woods'},{bbox:[5,5,20,20],confidence:.29},{bbox:[2,3,130,20],confidence:.9}]});
  assert.equal(cues.length,1);assert.equal(cues[0].timestampSeconds,15);assert.equal(cues[0].status,'unconfirmed_image_cue');
  for(const field of ['species','actualDistanceM','position','gps'])assert.equal(cues[0][field],null);
  for(const field of ['humanVerified','trainingAllowed','actuationEnabled'])assert.equal(cues[0][field],false);
  assert.deepEqual(investigationCues({sourceId:'test',timestampSeconds:1,stage:'fast',width:100,height:80,detections:[]}),[]);
});
test('hybrid UI exposes cadence, effective stage and captured-frame investigation cues',async()=>{
  const html=await readFile(new URL('../public/tracking-lab.html',import.meta.url),'utf8'),js=await readFile(new URL('../public/tracking-lab.js',import.meta.url),'utf8');
  assert.match(html,/<option value="hybrid" selected>/);assert.match(html,/<label for="detail-interval">/);assert.match(html,/id="detail-interval"[^>]*value="5"/);
  for(const id of ['stage-status','cue-list','cue-count'])assert.match(html,new RegExp(`id="${id}"`));
  assert.match(js,/scheduler\.commit\(scanPlan\)/);assert.match(js,/profileId:profile\.id/);assert.match(js,/lastActualStage/);
});
