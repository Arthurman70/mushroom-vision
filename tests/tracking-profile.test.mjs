import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { trackingProfile,trackingCaptureSize,trackingTilePreview,sampleCanResume,validateTrackingSample } from '../public/tracking-profile.mjs';
import { testbenchSettings,testbenchResult } from '../core/testbench-validation.mjs';
import { detectTrackingFrame } from '../core/tracking-demo.mjs';
import { LatestFrameGate,CandidateTracker } from '../public/tracking-logic.mjs';

const imageBase64='iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwCAIAAAAuKetIAAAAaElEQVR4nNXOMREAIBDAsFJdzMz4V4OIH7hGQda5mzKJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkziJkzh/B6YenRIBMk8i1kIAAAAASUVORK5CYII=';
const image={width:64,height:48,sha256:'a'.repeat(64)};
function rawResult(image,settings){
  const tiled=settings.methods.includes('tiles'),bounds=trackingTilePreview(image.width,image.height,{tileRegion:settings.tileRegion??[0,0,1,1]});
  const passTimings=[{kind:'context',bboxPixelsXYXY:[0,0,image.width,image.height],seconds:.1},...(tiled?bounds.map(bboxPixelsXYXY=>({kind:'tile',bboxPixelsXYXY,seconds:.1})):[])];
  return {schema:'mycoscout.testbench-result.v1',image:{...image,workingWidth:image.width,workingHeight:image.height},settings,
    model:{model:'fixture',downloadsEnabled:false},colorCues:[],seconds:passTimings.length*.1,modelPasses:passTimings.length,
    methods:[{id:tiled?'tiles':'original',candidates:[{bboxPixelsXYXY:[4,4,12,12],confidence:.7}],seconds:passTimings.length*.1,passes:passTimings.length,
      ...(tiled?{fullFrameContext:true,tileRegion:settings.tileRegion??[0,0,1,1],tileBoundsPixelsXYXY:bounds,passTimings}:{})}]};
}

test('tracking defaults keep the full context while tiling the selected lower area',()=>{
  const profile=trackingProfile();assert.equal(profile.id,'ground-tiles');assert.deepEqual(profile.tileRegion,[0,.35,1,1]);
  assert.equal(profile.expectedPasses,5);assert.equal(profile.semanticGroundSegmentation,false);assert.equal(profile.fullFrameContext,true);
  assert.deepEqual(trackingCaptureSize(3840,2160,profile),{width:1920,height:1080});
  assert.deepEqual(trackingCaptureSize(2160,3840,profile),{width:607,height:1080});
  assert.deepEqual(trackingCaptureSize(3840,2160,trackingProfile({profileId:'fast'})),{width:960,height:540});
  assert.deepEqual(trackingCaptureSize(64,48,profile),{width:64,height:48});
  const tiles=trackingTilePreview(1920,1080,profile);assert.equal(tiles.length,4);assert.equal(Math.min(...tiles.map(b=>b[1])),378);
  assert.equal(Math.max(...tiles.map(b=>b[3])),1080);assert.ok(tiles[0][2]>tiles[1][0]);assert.ok(tiles[0][3]>tiles[2][1]);
  assert.equal(trackingTilePreview(1920,1080,trackingProfile({profileId:'full-tiles'}))[0][1],0);
});

test('legacy settings preserve absence of ROI; malformed/unsupported profile areas fail closed',()=>{
  assert.equal(Object.hasOwn(testbenchSettings({methods:['tiles']}),'tileRegion'),false);
  for(const tileRegion of [null,[0,0,1],[0,.5,1,.5],[-.1,0,1,1],[0,0,Infinity,1],[0,'0',1,1]]){
    assert.throws(()=>testbenchSettings({methods:['tiles'],tileRegion}));assert.throws(()=>trackingProfile({tileRegion}));
  }
  assert.throws(()=>testbenchSettings({methods:['original'],tileRegion:[0,0,1,1]}));
  assert.throws(()=>testbenchSettings({methods:['tiles'],profileId:'ground-tiles'}));
  assert.throws(()=>trackingProfile({profileId:'ground'}));assert.throws(()=>trackingProfile({profileId:'fast',tileRegion:[0,0,1,1]}));
  assert.throws(()=>trackingProfile({profileId:'full-tiles',tileRegion:[0,.35,1,1]}));
});

test('tracking endpoint returns actual passes, timing and original-coordinate context detections',async()=>{
  const requests=[],worker={run:async(image,settings)=>{requests.push(settings);return rawResult(image,settings);}};
  const result=await detectTrackingFrame({imageBase64,mimeType:'image/png'},worker);
  assert.deepEqual(requests[0],{methods:['tiles'],tileGrid:2,scale:1,boxThreshold:.3,tileRegion:[0,.35,1,1]});
  assert.equal(result.profile.id,'ground-tiles');assert.equal(result.scan.passes,5);assert.equal(result.scan.modelPasses,5);
  assert.equal(result.scan.fullFrameContext,true);assert.equal(result.scan.passTimings.length,5);assert.equal(result.scan.seconds,.5);
  assert.deepEqual(result.detections[0].bbox,[4,4,12,12]); // Above ROI: full-context evidence is retained.
  for(const flag of ['humanVerified','trainingAllowed','actuationEnabled','stored'])assert.equal(result[flag],false);
  const fast=await detectTrackingFrame({imageBase64,mimeType:'image/png',profileId:'fast'},worker);
  assert.equal(fast.scan.passes,1);assert.equal(fast.scan.tileRegion,null);assert.equal(fast.scan.passTimings[0].seconds,.1);
  assert.equal(requests[1].boxThreshold,.3);assert.deepEqual(requests[1].methods,['original']);
  let calls=0;await assert.rejects(detectTrackingFrame({imageBase64,profileId:'invented'},{run:()=>calls++}));assert.equal(calls,0);
});

test('worker ROI and per-pass evidence must match the requested geometry and timing',()=>{
  const settings=testbenchSettings({methods:['tiles'],tileRegion:[0,.35,1,1]}),raw=rawResult(image,settings);
  const result=testbenchResult(raw,image,settings);assert.equal(result.methods[0].fullFrameContext,true);assert.equal(result.methods[0].passTimings.length,5);
  for(const mutate of [r=>r.methods[0].tileRegion=[0,0,1,1],r=>r.methods[0].fullFrameContext=false,
    r=>r.methods[0].passTimings.pop(),r=>r.methods[0].passTimings[0].kind='tile',
    r=>r.methods[0].tileBoundsPixelsXYXY[0][1]=0,r=>r.methods[0].passTimings[1].seconds=50,
    r=>r.methods[0].passTimings[1].bboxPixelsXYXY=[0,0,64,48],r=>delete r.methods[0].tileBoundsPixelsXYXY]){
    const bad=structuredClone(raw);mutate(bad);assert.throws(()=>testbenchResult(bad,image,settings),error=>error.code==='TESTBENCH_PROTOCOL');
  }
});

test('legacy testbench results stay readable without inventing new scan metadata',()=>{
  const settings=testbenchSettings({methods:['tiles']}),legacy=rawResult(image,settings);
  for(const key of ['fullFrameContext','tileRegion','tileBoundsPixelsXYXY','passTimings'])delete legacy.methods[0][key];
  delete legacy.modelPasses;
  const result=testbenchResult(legacy,image,settings);assert.equal(result.modelPasses,null);assert.equal(result.methods[0].fullFrameContext,undefined);
  assert.equal(result.methods[0].tileRegion,undefined);assert.equal(result.methods[0].candidates.length,1);
});

test('changing scan while work is pending drops old evidence without permitting concurrent work',async()=>{
  const gate=new LatestFrameGate(),tracker=new CandidateTracker();let done,accepted=0;
  tracker.update({timestampSeconds:0,width:64,height:48,detections:[{bbox:[4,4,12,12],confidence:.7}]});
  const pending=gate.run(()=>new Promise(resolve=>{done=resolve;}),()=>accepted++);
  gate.reset();tracker.reset('new scan');assert.equal(gate.pending,true);assert.deepEqual(tracker.snapshot(0),[]);
  assert.deepEqual(await gate.run(async()=>{},()=>accepted++),{skipped:true});done({});assert.deepEqual(await pending,{discarded:true});assert.equal(accepted,0);
});

test('only an accepted result from the unchanged tracking generation can resume playback',()=>{
  const state={resume:true,outcome:{accepted:true},capturedEpoch:2,currentEpoch:2,capturedGeneration:4,currentGeneration:4};
  assert.equal(sampleCanResume(state),true);
  for(const changed of [{resume:false},{outcome:{discarded:true}},{outcome:undefined},{currentEpoch:3},{currentGeneration:5}])assert.equal(sampleCanResume({...state,...changed}),false);
});

test('client rejects wrong profile, altered dimensions or missing pass evidence before tracking',async()=>{
  const result=await detectTrackingFrame({imageBase64},{run:async(image,settings)=>rawResult(image,settings)}),profile=trackingProfile();
  assert.equal(validateTrackingSample(result,{profile,width:64,height:48}),result);
  for(const mutate of [r=>r.profile.id='fast',r=>r.settings.tileRegion=[0,0,1,1],r=>r.image.width=128,r=>delete r.scan.passTimings,
    r=>r.scan.passTimings[0].kind='tile',r=>r.scan.modelPasses=1,r=>r.scan.fullFrameContext=false,r=>r.detections[0].bbox[0]=-1]){
    const bad=structuredClone(result);mutate(bad);assert.throws(()=>validateTrackingSample(bad,{profile,width:64,height:48}));
  }
});

test('custom top edges preserve full context and put every detail tile in the selected area',()=>{
  for(const top of [0,.35,.8]){
    const profile=trackingProfile({profileId:'ground-tiles',tileRegion:[0,top,1,1]}),tiles=trackingTilePreview(640,480,profile);
    assert.equal(profile.fullFrameContext,true);assert.equal(tiles.length,4);assert.equal(Math.min(...tiles.map(b=>b[1])),Math.floor(top*480));
    assert.equal(Math.max(...tiles.map(b=>b[3])),480);
  }
});

test('tracking UI keeps saved replay separate and provides accessible scan controls and measured timing',async()=>{
  const html=await readFile(new URL('../public/tracking-lab.html',import.meta.url),'utf8'),js=await readFile(new URL('../public/tracking-lab.js',import.meta.url),'utf8');
  assert.match(html,/<label for="vision-profile">/);assert.match(html,/<select id="vision-profile"/);
  assert.match(html,/not automatic ground recognition/);assert.match(html,/Analyze recording afresh/);
  assert.match(html,/<label for="ground-top">/);assert.match(html,/<input id="ground-top" type="range" min="0" max="80" value="35"/);
  assert.match(js,/mode==='replay'&&!replayFresh&&demo/);assert.match(js,/epoch\+\+;automatic=false;inferenceSeconds=null;resetTracking/);
  assert.match(js,/mode==='replay'&&!replayFresh\?'Play until/);assert.match(js,/\$\('ground-top'\)\.oninput=changeScan/);
  assert.match(js,/result\.scan\.modelPasses/);assert.match(js,/result\.scan\.passTimings/);
});
