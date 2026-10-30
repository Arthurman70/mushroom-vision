import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,rm,mkdir,writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,resolve,sep } from 'node:path';
import { createHash } from 'node:crypto';
import { VideoGradesStore } from '../core/video-grades.mjs';
import { createApp } from '../server.mjs';

const collectionId='video-fixture';
const source={id:'video1',videoHash:'a'.repeat(64),sourceGroupId:'session:1',licenseAllowsTraining:true,evaluationOnly:false};
const frame=(id='frame1',sourceOverride=source)=>({id,sourceId:sourceOverride.id,source:sourceOverride,sha256:'b'.repeat(64),width:1000,height:500,timestampSeconds:id==='frame1'?1:2,evaluationOnly:false});
const review=(fields={})=>({baseRevision:0,reviewer:'Human test fixture',status:'visible',boxes:[],notes:'Fixture only',...fields});
const box=(fields={})=>({id:'box1',bboxPixelsXYXY:[100,100,107,110],groupId:'unknown',label:'Mushroom',...fields});
async function fixture(t){
  const directory=await mkdtemp(join(tmpdir(),'mycoscout-video-grades-')),store=await new VideoGradesStore(directory).initialize();
  t.after(async()=>{store.close();assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});});
  return {store,directory};
}

test('quick visible, absent and uncertain grades remain explicit and independent',async t=>{
  const {store}=await fixture(t),frames=[frame(),frame('frame2'),frame('frame3'),frame('frame4')];
  store.save(collectionId,frames[0],review());store.save(collectionId,frames[1],review({status:'absent'}));store.save(collectionId,frames[2],review({status:'uncertain'}));
  const result=store.snapshot(collectionId,frames);
  assert.deepEqual(result.report.statusCounts,{visible:1,absent:1,uncertain:1});assert.equal(result.report.unreviewedFrames,1);
  assert.equal(result.report.verifiedBoxes,0);assert.equal(result.report.visibleWithoutBoxes,1);assert.equal(result.report.detectionAccuracy,null);
  assert.equal(result.automaticPropagation,false);assert.equal(result.automaticTraining,false);assert.equal(result.reviews.frame4.review,null);
});

test('assistance provenance persists with the human review and older clients cannot erase it by omission',async t=>{
  const {store}=await fixture(t),f=frame();
  const saved=store.save(collectionId,f,review({boxes:[box()],assistance:{aiAssisted:true,communityReviewIds:[7,7,12]}}));
  assert.deepEqual(saved.review.assistance,{aiAssisted:true,communityReviewIds:[7,12]});
  assert.equal(saved.review.boxes[0].trackId,null);assert.equal(saved.review.trainingAllowed,false);
  store.close();await store.initialize();
  assert.deepEqual(store.get(collectionId,f).review.assistance,saved.review.assistance);
  const newer=store.save(collectionId,f,review({baseRevision:1,status:'uncertain'}));
  assert.deepEqual(newer.review.assistance,saved.review.assistance);
  assert.equal(store.get(collectionId,f,{history:true}).history[1].boxes.length,1);
  for(const assistance of [{aiAssisted:'yes',communityReviewIds:[]},{aiAssisted:false,communityReviewIds:[0]},{aiAssisted:false,communityReviewIds:[1.5]},{aiAssisted:false,communityReviewIds:['7']},{aiAssisted:true,communityReviewIds:Array(101).fill(1)}])
    assert.throws(()=>store.save(collectionId,f,review({baseRevision:2,assistance})),/assistance provenance/);
  assert.equal(store.get(collectionId,f).revision,2);
});

test('frame revisions survive shuffled feed and restart; stale saves conflict',async t=>{
  const {store,directory}=await fixture(t),first=frame(),second=frame('frame2');
  store.save(collectionId,first,review({boxes:[box()]}));
  store.save(collectionId,first,review({baseRevision:1,status:'uncertain'}));
  assert.throws(()=>store.save(collectionId,first,review()),{code:'VIDEO_GRADE_CONFLICT'});
  assert.equal(store.snapshot(collectionId,[second,first]).reviews.frame1.revision,2);
  store.close();await store.initialize();const result=store.get(collectionId,first,{history:true});
  assert.equal(result.history.length,2);assert.equal(result.history[1].boxes[0].pixelWidth,7);assert.equal(result.review.status,'uncertain');
  assert.equal(store.directory,resolve(directory));
});

test('same source track allows earlier manual review but never propagates boxes',async t=>{
  const {store}=await fixture(t),track=store.createTrack(collectionId,source,{label:'One mushroom',reviewer:'Fixture'}),early=frame(),late=frame('frame2');
  store.save(collectionId,late,review({boxes:[box({trackId:track.id,bboxPixelsXYXY:[100,100,300,300]})]}));
  let timeline=store.snapshot(collectionId,[late,early]).report.tracks[0];
  assert.equal(timeline.frames[0].status,'unreviewed');assert.deepEqual(timeline.frames[0].boxes,[]);
  store.save(collectionId,early,review({status:'uncertain'}));timeline=store.snapshot(collectionId,[early,late]).report.tracks[0];
  assert.equal(timeline.frames[0].status,'uncertain');assert.equal(timeline.frames[0].boxes.length,0);
  store.save(collectionId,early,review({baseRevision:1,boxes:[box({trackId:track.id})]}));
  const report=store.snapshot(collectionId,[late,early]).report;
  assert.equal(report.pixelWidthBins['4to8'],1);assert.equal(report.pixelWidthBins['64plus'],1);
  assert.equal(report.tracks[0].frames[0].boxes[0].pixelWidth,7);
});

test('track cannot cross videos even when clips share a split group',async t=>{
  const {store}=await fixture(t),track=store.createTrack(collectionId,source,{label:'One mushroom',reviewer:'Fixture'});
  for(const foreign of [{...source,id:'video2'},{...source,videoHash:'c'.repeat(64)}])
    assert.throws(()=>store.save(collectionId,frame('foreign',foreign),review({boxes:[box({trackId:track.id})]})),{code:'VIDEO_TRACK_SOURCE_MISMATCH'});
  assert.throws(()=>store.save('another-collection',frame(),review({boxes:[box({trackId:track.id})]})),{code:'VIDEO_TRACK_SOURCE_MISMATCH'});
});

test('a tracked mushroom appears once per frame and unknown tracks are rejected',async t=>{
  const {store}=await fixture(t),track=store.createTrack(collectionId,source,{label:'One mushroom',reviewer:'Fixture'});
  assert.throws(()=>store.save(collectionId,frame(),review({boxes:[box({trackId:'missing'})]})),{code:'VIDEO_TRACK_SOURCE_MISMATCH'});
  assert.throws(()=>store.save(collectionId,frame(),review({boxes:[box({trackId:track.id}),box({id:'box2',trackId:track.id})]})),/only once/);
});

test('human box validation rejects impossible or nonvisible truth',async t=>{
  const {store}=await fixture(t);
  for(const invalid of [[NaN,0,1,1],[-1,0,1,1],[0,0,1001,1],[0,0,1,501],[2,0,1,1],[0,0,0,1],['0',0,1,1]])
    assert.throws(()=>store.save(collectionId,frame(),review({boxes:[box({bboxPixelsXYXY:invalid})]})));
  assert.throws(()=>store.save(collectionId,frame(),review({boxes:[null]})),/object/);
  assert.throws(()=>store.save(collectionId,frame(),review({boxes:[box(),box()]})),/unique/);
  for(const status of ['absent','uncertain'])assert.throws(()=>store.save(collectionId,frame(),review({status,boxes:[box()]})),/Only visible/);
  assert.throws(()=>store.save(collectionId,frame(),review({complete:true})),/at least one box/);
  assert.throws(()=>store.save(collectionId,frame(),review({status:'uncertain',complete:true})),/cannot be completely/);
  assert.throws(()=>store.save(collectionId,frame(),review({trainingAllowed:'true'})),/boolean/);
  assert.throws(()=>store.save(collectionId,frame(),review({boxes:Array.from({length:129},(_,i)=>box({id:'box'+i}))})),/128/);
});

test('all pixel bins count only current visible human boxes; uncertain excluded',async t=>{
  const {store}=await fixture(t),f=frame(),boxes=[3,4,8,16,32,64].map((width,i)=>box({id:'b'+i,bboxPixelsXYXY:[0,i*10,width,i*10+5]}));
  store.save(collectionId,f,review({boxes,complete:true}));
  assert.deepEqual(Object.values(store.snapshot(collectionId,[f]).report.pixelWidthBins),[1,1,1,1,1,1]);
  store.save(collectionId,f,review({baseRevision:1,status:'uncertain'}));
  assert.equal(store.snapshot(collectionId,[f]).report.verifiedBoxes,0);
});

test('provenance cannot change under saved review IDs',async t=>{
  const {store}=await fixture(t),f=frame();store.save(collectionId,f,review());
  for(const modified of [{...f,sha256:'c'.repeat(64)},{...f,width:999},{...f,timestampSeconds:3},{...f,source:{...source,videoHash:'c'.repeat(64)}}])
    assert.throws(()=>store.get(collectionId,modified),{code:'VIDEO_GRADE_INTEGRITY'});
  assert.throws(()=>store.db.prepare('UPDATE grade_reviews SET data=?').run('{}'),/immutable/);
});

test('training permission stays explicit and restricted by source rights and uncertainty',async t=>{
  const {store}=await fixture(t);
  assert.equal(store.save(collectionId,frame('permitted'),review({trainingAllowed:true,boxes:[box()],complete:true})).review.trainingAllowed,true);
  for(const f of [{...frame('evaluation'),evaluationOnly:true},frame('unlicensed',{...source,licenseAllowsTraining:false})])
    assert.equal(store.save(collectionId,f,review({trainingAllowed:true})).review.trainingAllowed,false);
  assert.equal(store.save(collectionId,frame('uncertain'),review({status:'uncertain',trainingAllowed:true})).review.trainingAllowed,false);
  assert.equal(store.save(collectionId,frame('default'),review()).review.trainingAllowed,false);
});

const png=Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2ZQkAAAAASUVORK5CYII=','base64');
async function apiFixture(t){
  const directory=await mkdtemp(join(tmpdir(),'mycoscout-video-grades-api-')),videoDir=join(directory,'videos');
  await mkdir(join(videoDir,'frames'),{recursive:true});await writeFile(join(videoDir,'frames','frame1.png'),png);
  await writeFile(join(videoDir,'manifest.json'),JSON.stringify({schema:'mycoscout.video-collection.v1',sources:[{...source,title:'Fixture source',sourceUrl:'https://example.org/video',attribution:'Fixture only',license:'CC0',licenseUrl:'https://creativecommons.org/publicdomain/zero/1.0/'}],
    frames:[{id:'frame1',sourceId:source.id,path:'frames/frame1.png',sha256:createHash('sha256').update(png).digest('hex'),width:1,height:1,timestampSeconds:1,blurScore:0,blurMetric:'fixture'}]}));
  const token='video-grades-test-token-123456789',app=await createApp({host:'127.0.0.1',token,stateFile:join(directory,'state.json'),videoCollectionDir:videoDir,
    testbenchWorker:{stop(){},status:async()=>({available:false})}}),address=await app.listen(0),base='http://127.0.0.1:'+address.port;
  t.after(async()=>{app.server.closeAllConnections();await new Promise(done=>app.server.close(done));await app.testbench.pending;await app.finds.pending;
    assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});});
  const request=async(path,body,headers={Authorization:'Bearer '+token})=>{const response=await fetch(base+path,{method:body===undefined?'GET':'POST',headers:{...headers,...(body===undefined?{}:{'Content-Type':'application/json'})},body:body===undefined?undefined:JSON.stringify(body)});return {status:response.status,data:await response.json()};};
  return {request,token};
}

test('grading APIs save versioned human reviews and expose timelines without inference',async t=>{
  const {request}=await apiFixture(t);let response=await request('/api/video-grades');assert.equal(response.status,200);assert.equal(response.data.report.unreviewedFrames,1);
  const track=(await request('/api/video-grades/tracks',{sourceId:'video1',label:'Fixture track',reviewer:'Fixture'})).data.track;
  response=await request('/api/video-grades/frames/frame1',review({boxes:[box({trackId:track.id,bboxPixelsXYXY:[0,0,1,1]})]}));
  assert.equal(response.status,201);assert.equal(response.data.revision,1);
  assert.equal((await request('/api/video-grades/frames/frame1',review())).status,409);
  assert.equal((await request('/api/video-grades/frames/frame1')).data.history.length,1);
  response=await request('/api/video-grades');assert.equal(response.data.report.tracks[0].frames[0].boxes[0].pixelWidth,1);
  assert.equal(response.data.automaticTraining,false);assert.equal((await request('/api/video-grades/frames/missing',review())).status,404);
  assert.equal((await request('/api/video-grades/tracks',{sourceId:'missing',label:'x',reviewer:'Fixture'})).status,404);
});

test('grading APIs inherit token and origin protection',async t=>{
  const {request,token}=await apiFixture(t);
  for(const path of ['/api/video-grades','/api/video-grades/frames/frame1'])assert.equal((await request(path,undefined,{})).status,401);
  assert.equal((await request('/api/video-grades/frames/frame1',review(),{})).status,401);
  assert.equal((await request('/api/video-grades/tracks',{sourceId:'video1',label:'x',reviewer:'Fixture'},{})).status,401);
  assert.equal((await request('/api/video-grades/frames/frame1',review(),{Authorization:'Bearer '+token,Origin:'https://foreign.example'})).status,403);
});
