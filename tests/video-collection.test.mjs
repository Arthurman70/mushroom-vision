import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,resolve,sep } from 'node:path';
import { createHash } from 'node:crypto';
import { deflateSync } from 'node:zlib';
import { VideoCollection } from '../core/video-collection.mjs';
import { createApp } from '../server.mjs';
import { TestbenchStore } from '../core/testbench-store.mjs';
import { DatabaseSync } from 'node:sqlite';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const settings={methods:['original'],tileGrid:2,scale:1,boxThreshold:.25};
const worker=()=>({status:async()=>({available:true}),stop(){},run:async(image,settings)=>({schema:'mycoscout.testbench-result.v1',image:{sha256:image.sha256,width:1,height:1,workingWidth:1,workingHeight:1},settings,
  model:{model:'test fixture',downloadsEnabled:false},methods:[{id:'original',candidates:[],seconds:.01,passes:1}],colorCues:[],seconds:.01})});
function png(color){
  const chunk=(kind,data)=>{const type=Buffer.from(kind),payload=Buffer.concat([type,data]),length=Buffer.alloc(4),checksum=Buffer.alloc(4);length.writeUInt32BE(data.length);let crc=0xffffffff;
    for(const byte of payload){crc^=byte;for(let bit=0;bit<8;bit++)crc=(crc>>>1)^((crc&1)?0xedb88320:0);}checksum.writeUInt32BE((crc^0xffffffff)>>>0);return Buffer.concat([length,payload,checksum]);};
  const header=Buffer.alloc(13);header.writeUInt32BE(1,0);header.writeUInt32BE(1,4);header[8]=8;header[9]=2;
  return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(Buffer.from([0,...color]))),chunk('IEND',Buffer.alloc(0))]);
}
async function fixture(t,{eligible=false,cleanup=true}={}){
  const directory=await mkdtemp(join(tmpdir(),'mycoscout-video-')),collectionDir=join(directory,'collection');await mkdir(join(collectionDir,'frames'),{recursive:true});
  const source={id:'video-a',title:'Fixture source',sourceUrl:'https://example.org/video',attribution:'Software fixture',license:'CC0',licenseUrl:'https://creativecommons.org/publicdomain/zero/1.0/',
    videoHash:'a'.repeat(64),sourceGroupId:'video:'+'a'.repeat(64),evaluationOnly:!eligible,licenseAllowsTraining:eligible};
  const frames=[];for(let index=0;index<2;index++){const bytes=png([index*100,50,80]),id='frame-'+index,path='frames/'+id+'.png';await writeFile(join(collectionDir,path),bytes);frames.push({id,sourceId:source.id,path,sha256:sha(bytes),width:1,height:1,timestampSeconds:index*5,blurScore:30-index*20,blurMetric:'laplacian_variance',partitionGroup:source.sourceGroupId,evaluationOnly:!eligible});}
  const manifest={schema:'mycoscout.video-collection.v1',sources:[source],frames},file=join(collectionDir,'manifest.json');await writeFile(file,JSON.stringify(manifest));
  if(cleanup)t.after(async()=>{assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});});
  return {directory,collectionDir,file,manifest,collection:new VideoCollection(collectionDir)};
}
test('video gallery has bounded pagination, attribution and distinct same-video group; no labels or filesystem paths',async t=>{
  const {collection}=await fixture(t),gallery=await collection.list({limit:1,sort:'blur'});assert.equal(gallery.frames[0].id,'frame-1');assert.equal(gallery.total,2);assert.equal(gallery.frames[0].path,undefined);
  assert.equal(gallery.frames[0].source.path,undefined);assert.equal(gallery.frames[0].actualDistanceM,null);assert.equal(gallery.labelsVerified,false);assert.equal(gallery.frames[0].source.attribution,'Software fixture');
  const a=await collection.frame('frame-0'),b=await collection.frame('frame-1');assert.equal(a.metadata.sceneId,b.metadata.sceneId);assert.notEqual(a.image.sha256,b.image.sha256);
  await assert.rejects(collection.list({limit:1000}));await assert.rejects(collection.frame('../../secrets'));assert.equal(a.metadata.evaluationOnly,true);
});
test('video manifest traversal, changed image bytes and forged grouping are rejected before inference',async t=>{
  const {collection,file,manifest,collectionDir}=await fixture(t);
  manifest.frames[0].path='../secret.png';await writeFile(file,JSON.stringify(manifest));await assert.rejects(collection.read(),error=>error.code==='VIDEO_COLLECTION_INVALID');
  manifest.frames[0].path='frames/frame-0.png';manifest.frames[0].partitionGroup='other';await writeFile(file,JSON.stringify(manifest));await assert.rejects(collection.read());
  manifest.frames[0].partitionGroup=manifest.sources[0].sourceGroupId;await writeFile(file,JSON.stringify(manifest));await writeFile(join(collectionDir,manifest.frames[0].path),png([255,0,0]));
  await assert.rejects(collection.frame('frame-0'),error=>error.code==='VIDEO_FRAME_INTEGRITY');
});
test('video API preserves provenance and same-source splits; human permission is required and clients cannot override research quarantine',async t=>{
  const f=await fixture(t,{eligible:true,cleanup:false});const worker={status:async()=>({available:true}),stop(){},run:async(image,settings)=>({schema:'mycoscout.testbench-result.v1',image:{sha256:image.sha256,width:1,height:1,workingWidth:1,workingHeight:1},settings,
    model:{model:'test fixture',downloadsEnabled:false},methods:[{id:'original',candidates:[],seconds:.01,passes:1}],colorCues:[],seconds:.01})};
  const app=await createApp({stateFile:join(f.directory,'state.json'),videoCollectionDir:f.collectionDir,testbenchWorker:worker,token:'fixture-video-token-123456789'});
  const address=await app.listen(0),base='http://127.0.0.1:'+address.port,headers={'Content-Type':'application/json',Authorization:'Bearer fixture-video-token-123456789'};
  t.after(async()=>{app.server.closeAllConnections();await new Promise(done=>app.server.close(done));await app.closed;await app.testbench.pending;await app.finds.pending;assert.ok(resolve(f.directory).startsWith(resolve(tmpdir())+sep));await rm(f.directory,{recursive:true,force:true});});
  assert.equal((await fetch(base+'/api/testbench/videos')).status,401);assert.equal((await fetch(base+'/api/testbench/videos/frames/frame-0/image')).status,401);
  const runs=[];for(const frame of f.manifest.frames){const response=await fetch(base+'/api/testbench/videos/frames/'+frame.id+'/run',{method:'POST',headers,body:JSON.stringify({settings,evaluationOnly:false})});assert.equal(response.status,202);const {run}=await response.json();await app.testbench.pending;runs.push(app.testbench.get(run.id));}
  assert.equal(runs[0].annotation,null);assert.equal(runs[0].sceneId,runs[1].sceneId);assert.equal(runs[1].videoProvenance.timestampSeconds,5);assert.equal((await app.testbench.exportDataset()).records.length,0);
  for(const run of runs)app.testbench.annotate(run.id,{baseRevision:0,reviewer:'Software fixture',complete:true,trainingAllowed:true,boxes:[]});
  const exported=await app.testbench.exportDataset();assert.equal(exported.records.length,2);assert.equal(exported.records[0].partition,exported.records[1].partition);assert.equal(exported.records[0].provenance.sceneGrouping,'source-video-group');
  const rerun=await app.testbench.create({sourceRunId:runs[0].id,settings});await app.testbench.pending;assert.equal(rerun.sceneId,runs[0].sceneId);assert.equal(rerun.videoProvenance.id,'frame-0');
  f.manifest.sources[0].evaluationOnly=true;await writeFile(f.file,JSON.stringify(f.manifest));
  const quarantined=await fetch(base+'/api/testbench/videos/frames/frame-1/run',{method:'POST',headers,body:JSON.stringify({settings,evaluationOnly:false,trainingAllowed:true})});const {run}=await quarantined.json();await app.testbench.pending;
  assert.equal(run.evaluationOnly,true);assert.equal(app.testbench.annotate(run.id,{baseRevision:0,reviewer:'Fixture',complete:true,trainingAllowed:true,boxes:[]}).annotation.trainingAllowed,false);
  assert.equal((await app.testbench.exportDataset()).records.length,1);
});

test('identical pixels at different video times and sources retain immutable per-run identity and quarantine conflicting source groups',async t=>{
  const f=await fixture(t,{eligible:true,cleanup:false}),directory=join(f.directory,'testbench'),store=await new TestbenchStore(directory,resolve('.'),worker()).initialize();
  t.after(async()=>{await store.stop();assert.ok(resolve(f.directory).startsWith(resolve(tmpdir())+sep));await rm(f.directory,{recursive:true,force:true});});
  const {image,metadata}=await f.collection.frame('frame-0'),first=await store.create({imageBase64:image.bytes.toString('base64'),settings},metadata);await store.pending;
  const later={...metadata,id:'same-image-later',timestampSeconds:20};
  const second=await store.create({imageBase64:image.bytes.toString('base64'),settings},later);await store.pending;
  assert.equal(store.get(first.id).videoProvenance.timestampSeconds,0);assert.equal(store.get(second.id).videoProvenance.timestampSeconds,20);
  const secondSource={...metadata.source,id:'other-video',videoHash:'b'.repeat(64),sourceGroupId:'video:'+'b'.repeat(64),attribution:'Second video credit'};
  const thirdMetadata={...metadata,id:'other-video-frame',sourceId:secondSource.id,source:secondSource,sceneId:secondSource.sourceGroupId};
  const third=await store.create({imageBase64:image.bytes.toString('base64'),settings},thirdMetadata);await store.pending;
  assert.equal(store.get(first.id).sourceCredits.attribution,'Software fixture');assert.equal(store.get(third.id).sourceCredits.attribution,'Second video credit');
  assert.equal(store.get(third.id).videoProvenance.source.videoHash,'b'.repeat(64));
  const freshPhoto=await store.create({imageBase64:image.bytes.toString('base64'),settings});await store.pending;assert.equal(freshPhoto.videoProvenance,null);
  assert.equal(store.completedVideoRuns().length,3);assert.deepEqual(store.completedVideoRuns().map(run=>run.videoProvenance.timestampSeconds).sort((a,b)=>a-b),[0,0,20]);
  const rerun=await store.create({sourceRunId:second.id,settings});await store.pending;assert.equal(rerun.videoProvenance.timestampSeconds,20);
  assert.throws(()=>store.db.prepare('UPDATE runs SET video_provenance=? WHERE id=?').run('{}',first.id),/immutable/);
  for(const run of [first,second,third])store.annotate(run.id,{baseRevision:0,reviewer:'Fixture',complete:true,trainingAllowed:true,boxes:[]});
  const unrelatedPixels=await f.collection.frame('frame-1'),fourth=await store.create({imageBase64:unrelatedPixels.image.bytes.toString('base64'),settings},unrelatedPixels.metadata);await store.pending;
  store.annotate(fourth.id,{baseRevision:0,reviewer:'Fixture',complete:true,trainingAllowed:true,boxes:[]});
  const exported=await store.exportDataset();assert.equal(exported.records.length,0);assert.equal(exported.summary.excluded.ambiguousSourceGroups,2);
});

test('known research sample bytes keep their quarantine while video provenance remains joinable',async t=>{
  const f=await fixture(t,{eligible:true,cleanup:false}),{image,metadata}=await f.collection.frame('frame-0');
  const sample={sha256:image.sha256,path:join(f.collectionDir,'frames/frame-0.png'),name:'Known sample',attribution:'Sample origin credit',licenseUrl:'https://example.org/license',sourceUrl:'https://example.org/sample'};
  const store=await new TestbenchStore(join(f.directory,'testbench'),resolve('.'),worker(),{sample}).initialize();
  t.after(async()=>{await store.stop();assert.ok(resolve(f.directory).startsWith(resolve(tmpdir())+sep));await rm(f.directory,{recursive:true,force:true});});
  const baseline=await store.createSample({settings});await store.pending;assert.equal(baseline.videoProvenance,null);
  const frame=await store.create({imageBase64:image.bytes.toString('base64'),settings},metadata);await store.pending;
  assert.equal(frame.evaluationOnly,true);assert.equal(frame.videoProvenance.id,'frame-0');assert.equal(frame.sourceCredits.attribution,'Software fixture');
  assert.equal(store.get(baseline.id).sourceCredits.attribution,'Sample origin credit');assert.equal(store.completedVideoRuns()[0].videoProvenance.id,'frame-0');
  const rerun=await store.create({sourceRunId:frame.id,settings});await store.pending;assert.equal(rerun.evaluationOnly,true);assert.equal(rerun.videoProvenance.id,'frame-0');
  assert.equal(store.annotate(frame.id,{baseRevision:0,reviewer:'Fixture',complete:true,trainingAllowed:true,boxes:[]}).annotation.trainingAllowed,false);
});

test('v1 migration preserves records and research quarantine without inventing per-run video timestamps',async t=>{
  const f=await fixture(t,{cleanup:false}),{image,metadata}=await f.collection.frame('frame-0'),directory=join(f.directory,'testbench');
  let store=await new TestbenchStore(directory,resolve('.'),worker()).initialize();
  t.after(async()=>{await store.stop();assert.ok(resolve(f.directory).startsWith(resolve(tmpdir())+sep));await rm(f.directory,{recursive:true,force:true});});
  const first=await store.create({imageBase64:image.bytes.toString('base64'),settings},metadata);await store.pending;await store.stop();
  const old=new DatabaseSync(join(directory,'testbench.sqlite'));
  old.exec('DROP TRIGGER runs_inputs_immutable; ALTER TABLE runs DROP COLUMN video_provenance; PRAGMA user_version=1;');
  old.prepare('UPDATE photos SET credits=? WHERE hash=?').run(JSON.stringify({attribution:'Legacy credit',videoProvenance:metadata}),image.sha256);old.close();
  store=await new TestbenchStore(directory,resolve('.'),worker()).initialize();
  assert.equal(store.db.prepare('PRAGMA user_version').get().user_version,2);assert.equal(store.get(first.id).videoProvenance,null);assert.equal(store.get(first.id).legacyVideoProvenanceNeedsRerun,true);
  assert.equal(store.get(first.id).evaluationOnly,true);assert.equal(store.get(first.id).sourceCredits.attribution,'Legacy credit');assert.equal(store.completedVideoRuns().length,0);
  const fresh=await store.create({imageBase64:image.bytes.toString('base64'),settings},metadata);await store.pending;
  assert.equal(fresh.videoProvenance.id,'frame-0');assert.equal(store.completedVideoRuns().length,1);assert.equal(store.get(first.id).videoProvenance,null);
});
