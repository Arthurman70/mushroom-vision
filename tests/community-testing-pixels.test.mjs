import test from 'node:test';
import assert from 'node:assert/strict';
import {publicPixels,pixelAssetUrl,pixelLevelSelection,pixelViewSelection,pixelInspectorUrl,pixelRecoveryNote,PIXEL_LEVELS} from '../community/public/testing-pixels.js';

const a='a'.repeat(64),b='b'.repeat(64),c='c'.repeat(64);
const method=()=>({candidates:[{bbox:[10,10,20,20],score:.6}],metrics:{hits:0,targetCount:1,unmatchedPositiveBoxes:1},seconds:1.2,modelPasses:1});
const input=()=>({schema:'mushroom-vision.public-pixels.v1',completedAt:'2026-11-01T00:00:00Z',protocol:{weightsChanged:false,detectorName:'Fixture only',scoreThreshold:.1,iouThreshold:.3},cohorts:[{id:'human-video',label:'Human video',referenceAuthority:'human',targetCount:1,conditions:[{id:'native',mode:'raw',hits:0,targetCount:1,unmatchedPositiveBoxes:1,negativeAlertCount:0,negativeFrameCount:1,negativeFramesWithAlerts:0,meanSecondsPerFrame:1.2,measurementFrames:2}]},{id:'ai-photo',label:'AI photo',referenceAuthority:'ai-incomplete',targetCount:1,conditions:[{id:'0p69px',mode:'raw',hits:0,targetCount:1,unmatchedPositiveBoxes:1,negativeAlertCount:null,negativeFrameCount:null,negativeFramesWithAlerts:null,meanSecondsPerFrame:1.2,measurementFrames:1}]}],frames:[{frameId:'video-fixture',cohort:'human-video',imageSha256:a,width:100,height:80,referenceAuthority:'human',source:{title:'Synthetic test fixture',attribution:'Fixture',license:'CC0'}},{frameId:'photo-fixture',cohort:'ai-photo',imageSha256:b,width:200,height:160,referenceAuthority:'ai-incomplete',source:{title:'Synthetic test fixture',attribution:'Fixture',license:'CC0'}}],records:[{frameId:'video-fixture',cohort:'human-video',conditionId:'native',derivedImageSha256:a,width:100,height:80,originalTargetPixelsXYXY:[10,20,30,40],targetPixelsXYXY:[10,20,30,40],methods:{raw:method()}},{frameId:'photo-fixture',cohort:'ai-photo',conditionId:'0p69px',derivedImageSha256:c,assetName:'pixel-view-photo-fixture--0p69px.png',width:1920,height:1080,originalTargetPixelsXYXY:[20,30,120,80],targetPixelsXYXY:[500,400,500.69,400.345],realizedTargetWidthPixels:.69,realizedTargetHeightPixels:.345,geometry:{sourceSize:[200,160],outputSize:[1920,1080],sceneScaleXY:[.0069,.0069],supersampling:16},methods:{raw:method(),tiles:{...method(),modelPasses:5}}}]});

test('pixel results preserve subpixel regions and separate human and AI references',()=>{
  const result=publicPixels(input());assert.equal(result.cohorts[0].referenceAuthority,'human');assert.equal(result.cohorts[1].referenceAuthority,'ai-incomplete');assert.equal(result.records[1].realizedTargetWidthPixels,.69);assert.deepEqual(result.records[1].targetPixelsXYXY,[500,400,500.69,400.345]);assert.equal(result.records[1].methods.tiles.modelPasses,5);
});
test('missing control measurements remain unavailable, independently of actual zero alerts',()=>{
  const result=publicPixels(input());assert.equal(result.cohorts[0].conditions[0].negativeAlertCount,0);assert.equal(result.cohorts[1].conditions[0].negativeAlertCount,null);assert.equal(result.records[0].methods.tiles,undefined);
});
test('best reference overlap retains real zero and near misses but rejects invalid values',()=>{
  for(const value of [0,.49,.5,1]){const source=input();source.records[0].methods.raw.metrics.bestIoU=value;assert.equal(publicPixels(source).records[0].methods.raw.metrics.bestIoU,value);}
  for(const value of [undefined,null,-.1,1.1,NaN,Infinity,'0.5']){const source=input();source.records[0].methods.raw.metrics.bestIoU=value;assert.equal(publicPixels(source).records[0].methods.raw.metrics.bestIoU,null);}
});
test('synthetic images require exact saved assets on the fixed canvas',()=>{
  for(const patch of [{width:200,height:160},{assetName:null},{assetName:'https://example.org/other.png'},{assetName:'../private.png'},{assetName:'pixel-view-video-fixture--0p69px.png'}]){const source=input();Object.assign(source.records[1],patch);assert.throws(()=>publicPixels(source));}
});
test('native inputs bind to the original bytes and dimensions',()=>{
  for(const patch of [{width:1920,height:1080},{derivedImageSha256:c},{derivedImageSha256:'bad'}]){const source=input();Object.assign(source.records[0],patch);assert.throws(()=>publicPixels(source));}
  const result=publicPixels(input());assert.equal(pixelAssetUrl(result.frames[0],result.records[0],'https://example.org/mushroom-lab/testing-pixels.html'),`https://example.org/mushroom-lab/images/video-fixture?v=${a}`);
});
test('image URL builder independently rejects arbitrary assets and mismatched identity',()=>{
  const source=input();assert.equal(pixelAssetUrl(source.frames[1],source.records[1],'https://example.org/mushroom-lab/testing-pixels.html'),`https://example.org/mushroom-lab/pixel-view-photo-fixture--0p69px.png?v=${c}`);
  for(const patch of [{assetName:'//evil.invalid/photo.png'},{assetName:'../private.txt'},{frameId:'another-frame'},{conditionId:'unknown'},{derivedImageSha256:'bad'},{derivedImageSha256:undefined}])assert.throws(()=>pixelAssetUrl(source.frames[1],{...source.records[1],...patch},'https://example.org/mushroom-lab/'));
});
test('rerun bytes produce a distinct synthetic URL without changing the bounded asset path',()=>{
  const source=input(),old=pixelAssetUrl(source.frames[1],source.records[1],'https://example.org/mushroom-lab/'),updated=pixelAssetUrl(source.frames[1],{...source.records[1],derivedImageSha256:a},'https://example.org/mushroom-lab/');assert.notEqual(old,updated);assert.equal(new URL(old).pathname,new URL(updated).pathname);assert.equal(new URL(updated).searchParams.get('v'),a);
});
test('native-only controls fall back from tiny levels and disable unavailable measurements',()=>{
  const records=PIXEL_LEVELS.map(level=>({frameId:'positive',conditionId:level.id}));records.push({frameId:'control',conditionId:'native'});
  const tiny=pixelLevelSelection(records,'positive','0p69px');assert.equal(tiny.selectedId,'0p69px');assert.equal(tiny.levels.filter(level=>level.available).length,8);
  const control=pixelLevelSelection(records,'control',tiny.selectedId);assert.equal(control.selectedId,'native');assert.deepEqual(control.levels.filter(level=>level.available).map(level=>level.id),['native']);
  const restored=pixelLevelSelection(records,'positive',control.selectedId);assert.equal(restored.selectedId,'native');assert.equal(restored.levels.filter(level=>level.available).length,8);
});
test('source navigation reports no saved test when no level exists',()=>{
  const selection=pixelLevelSelection(input().records,'missing-frame','32px');assert.equal(selection.selectedId,null);assert.equal(selection.levels.some(level=>level.available),false);
});
test('shareable inspector views resolve only known frame and saved level combinations',()=>{
  const run=publicPixels(input()),url=pixelInspectorUrl(run,'photo-fixture','0p69px','tiles','https://example.org/mushroom-lab/testing-pixels.html?old=1');
  assert.deepEqual(pixelViewSelection(run,url),{cohortId:'ai-photo',frameId:'photo-fixture',level:'0p69px',mode:'tiles'});assert.equal(new URL(url).hash,'#inspection-title');assert.equal(new URL(url).searchParams.has('old'),false);
  assert.deepEqual(pixelViewSelection(run,'https://example.org/?frame=video-fixture&level=0p69px&mode=tiles'),{cohortId:'human-video',frameId:'video-fixture',level:'native',mode:'tiles'});
});
test('untrusted inspector parameters cannot choose arbitrary media or missing output',()=>{
  const run=publicPixels(input());assert.deepEqual(pixelViewSelection(run,'https://example.org/?frame=https://evil.invalid/photo.png&level=0p69px&mode=tiles'),{cohortId:'human-video',frameId:null,level:'native',mode:'raw'});assert.equal(pixelViewSelection(run,'https://example.org/?frame=video-fixture&mode=unknown').mode,'raw');
  for(const [frame,level,mode] of [['missing','native','raw'],['video-fixture','0p69px','raw'],['photo-fixture','0p69px','unknown']])assert.throws(()=>pixelInspectorUrl(run,frame,level,mode,'https://example.org/mushroom-lab/'));
});
test('regions, scores and reference collections cannot escape their declared inputs',()=>{
  for(const patch of [{cohort:'human-video'},{targetPixelsXYXY:[0,0,1921,20]},{originalTargetPixelsXYXY:[0,0,201,20]},{targetPixelsXYXY:[10,10,10,11]}]){const source=input();Object.assign(source.records[1],patch);assert.throws(()=>publicPixels(source));}
  for(const box of [{bbox:[0,0,1921,10],score:.3},{bbox:[0,0,20,10],score:1.1},{bbox:[0,0,20,10],score:NaN}]){const source=input();source.records[1].methods.raw.candidates=[box];assert.throws(()=>publicPixels(source));}
});
test('duplicate views and inconsistent measurement counts are rejected',()=>{
  for(const change of [s=>s.cohorts[1].id='human-video',s=>s.cohorts[0].conditions.push(s.cohorts[0].conditions[0]),s=>s.records.push(s.records[0]),s=>s.frames.push(s.frames[0]),s=>s.cohorts[0].conditions[0].hits=2,s=>s.cohorts[0].conditions[0].negativeFramesWithAlerts=2,s=>s.records[0].methods.raw.metrics.hits=2]){const source=input();change(source);assert.throws(()=>publicPixels(source));}
});
test('public projection excludes private paths and unsafe source URLs',()=>{
  const source=input();source.privatePath='C:/secret/report.json';source.frames[1].source.sourceUrl='file:///C:/secret/image.png';source.frames[1].source.licenseUrl='https://user:secret@example.org/license';source.records[1].imagePath='C:/secret/image.png';source.records[1].geometry.imagePath='C:/secret/image.png';source.records[1].geometry.outputSize=['C:/secret/image.png',1080];
  const result=publicPixels(source);assert.equal(JSON.stringify(result).includes('secret'),false);assert.equal(result.frames[1].source.sourceUrl,null);assert.equal(result.frames[1].source.licenseUrl,null);assert.equal(result.records[1].geometry.supersampling,16);
});
test('public output copies reference, candidate and geometry arrays',()=>{
  const source=input(),result=publicPixels(source);result.records[1].targetPixelsXYXY[0]=1;result.records[1].methods.raw.candidates[0].bbox[0]=1;result.records[1].geometry.sourceSize[0]=1;assert.equal(source.records[1].targetPixelsXYXY[0],500);assert.equal(source.records[1].methods.raw.candidates[0].bbox[0],10);assert.equal(source.records[1].geometry.sourceSize[0],200);
});
test('pixel labels use stated geometric examples without deriving accuracy from distances',()=>{
  assert.equal(PIXEL_LEVELS.length,8);for(const level of PIXEL_LEVELS.filter(row=>row.distance))assert.ok(Math.abs(level.pixels-1380*.05/level.distance)<1e-10);assert.deepEqual(PIXEL_LEVELS.filter(row=>row.distance).map(row=>row.distance),[10,20,50,100]);
});
test('report never treats a weight update as this frozen-weight comparison',()=>{
  const source=input();source.protocol.weightsChanged=true;assert.throws(()=>publicPixels(source));source.protocol.weightsChanged=false;source.protocol.scoreThreshold=2;assert.equal(publicPixels(source).protocol.scoreThreshold,null);
});
test('recovery provenance distinguishes saved model passes from unknown attempted work',()=>{
  const source=input();source.protocol.recovery={occurred:true,reason:'file-lock',retainedScenes:1,recoveredScenes:1,totalScenes:2,uniquePersistedModelPasses:10,totalAttemptedModelPasses:null,totalAttemptedWorkKnown:false,notes:'Test fixture only.'};
  const recovery=publicPixels(source).protocol.recovery;assert.equal(recovery.totalAttemptedModelPasses,null);assert.match(pixelRecoveryNote(recovery),/2 unique image cases and 10 saved model passes/);assert.match(pixelRecoveryNote(recovery),/Total attempted work, including any unsaved attempts, is unknown/);assert.equal(pixelRecoveryNote(publicPixels(input()).protocol.recovery),'');
});
test('recovery counts cannot contradict records or invent a known attempted total',()=>{
  for(const patch of [{totalScenes:3},{retainedScenes:0},{uniquePersistedModelPasses:11},{totalAttemptedModelPasses:10},{totalAttemptedWorkKnown:true},{reason:'unsupported'},{recoveredScenes:Infinity}]){
    const source=input();source.protocol.recovery={occurred:true,reason:'file-lock',retainedScenes:1,recoveredScenes:1,totalScenes:2,uniquePersistedModelPasses:10,totalAttemptedModelPasses:null,totalAttemptedWorkKnown:false,...patch};assert.throws(()=>publicPixels(source));
  }
});
