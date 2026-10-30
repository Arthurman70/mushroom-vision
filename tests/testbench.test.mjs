import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,resolve,sep } from 'node:path';
import { TestbenchStore } from '../core/testbench-store.mjs';
import { testbenchMetrics,testbenchSettings,testbenchResult } from '../core/testbench-validation.mjs';

export const imageBase64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2ZQkAAAAASUVORK5CYII=';
export function result(image,settings){return {schema:'mycoscout.testbench-result.v1',image:{sha256:image.sha256,width:image.width,height:image.height,workingWidth:1,workingHeight:1},
  settings,model:{model:'fixture detector',resolvedRevision:'fixture',device:'cpu',threads:1,boxThreshold:settings.boxThreshold,textThreshold:.25,prompt:'fixture',downloadsEnabled:false,scoreMeaning:'uncalibrated'},
  methods:settings.methods.map(id=>({id,label:id,candidates:[{id:id+'-1',bboxPixelsXYXY:[0,0,1,1],confidence:.6}],seconds:.01,passes:1})),
  colorCues:[{bboxPixelsXYXY:[0,0,1,1],contrastScore:.5}],seconds:.03,warnings:[],actualDistanceM:null,actuationEnabled:false};}
export class FakeWorker{constructor(){this.requests=[];}async status(){return {available:true,busy:false};}async run(image,settings,progress){this.requests.push({image,settings});progress({stage:'original',completed:1,total:1});return result(image,settings);}stop(){}}
const annotation=(baseRevision=0,extra={})=>({baseRevision,reviewer:'Software test human fixture',complete:true,trainingAllowed:false,
  boxes:[{id:'human-box',bboxPixelsXYXY:[0,0,1,1],groupId:'unknown',label:'fungus'}],notes:'fixture',...extra});
async function fixture(t,worker=new FakeWorker()){
  const directory=await mkdtemp(join(tmpdir(),'mycoscout-testbench-')),store=await new TestbenchStore(directory,resolve('.'),worker).initialize();
  t.after(async()=>{await store.stop();assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});});
  return {store,worker,directory};
}

test('testbench persists actual originals, immutable annotations, settings and exact-image rerun lineage',async t=>{
  const {store}=await fixture(t),created=await store.create({imageBase64,name:'One photo',settings:{methods:['original'],tileGrid:2,scale:.5,boxThreshold:.4}});
  await store.pending;const run=store.get(created.id);assert.equal(run.status,'completed');assert.equal(run.image.orientationVerified,true);
  assert.deepEqual((await store.image(run.id)).bytes,Buffer.from(imageBase64,'base64'));assert.equal(run.metrics.status,'unannotated');
  const saved=store.annotate(run.id,annotation());assert.equal(saved.annotation.revision,1);assert.equal(saved.run.metrics.methods[0].precision,1);
  assert.throws(()=>store.annotate(run.id,annotation()),error=>error.code==='ANNOTATION_CONFLICT');
  assert.throws(()=>store.db.prepare('UPDATE annotations SET data=?').run('{}'),/immutable/);
  const rerun=await store.create({sourceRunId:run.id,settings:{methods:['tiles'],tileGrid:3,scale:.25,boxThreshold:.2}});await store.pending;
  const inherited=store.get(rerun.id);assert.equal(inherited.sceneId,run.sceneId);assert.equal(inherited.annotation.inheritedFrom.runId,run.id);
  assert.equal(inherited.annotation.inheritedFrom.revision,1);assert.equal(inherited.settings.scale,.25);
  assert.equal((await store.exportDataset()).records.length,0);
  store.annotate(run.id,annotation(1,{trainingAllowed:true,measuredDistanceM:20}));
  const exported=await store.exportDataset();assert.equal(exported.records.length,1);assert.equal(exported.records[0].annotation.measuredDistanceM,20);
  assert.equal(exported.records[0].provenance.relatedRunIds.length,2);assert.equal(exported.policies.automaticTraining,false);
  assert.equal((await store.exportDataset()).releaseId,exported.releaseId);
  store.annotate(run.id,annotation(2,{trainingAllowed:false}));assert.equal((await store.exportDataset()).records.length,0);
});

test('incomplete truth cannot become false negatives or false positives and a complete empty scene is an explicit negative',()=>{
  const predictions={methods:[{id:'original',candidates:[{id:'a',bboxPixelsXYXY:[0,0,1,1],confidence:.8},{id:'duplicate',bboxPixelsXYXY:[0,0,1,1],confidence:.7}]}]};
  const partial=testbenchMetrics(predictions,{complete:false,boxes:annotation().boxes});assert.equal(partial.methods[0].matched,1);
  for(const key of ['precision','recall','f1','falsePositives','falseNegatives'])assert.equal(partial.methods[0][key],null);
  const complete=testbenchMetrics(predictions,{complete:true,boxes:annotation().boxes});assert.equal(complete.methods[0].truePositives,1);assert.equal(complete.methods[0].falsePositives,1);
  const negative=testbenchMetrics(predictions,{complete:true,boxes:[]});assert.equal(negative.methods[0].falsePositives,2);assert.equal(negative.methods[0].recall,null);
  assert.equal(negative.colorCuesScoredAsMushrooms,false);
});

test('maximum matching reassigns overlaps instead of counting a duplicate as a second mushroom',()=>{
  const metrics=testbenchMetrics({methods:[{id:'original',candidates:[
    {id:'flexible',bboxPixelsXYXY:[0,0,15,10],confidence:.9},{id:'first-only',bboxPixelsXYXY:[0,0,10,10],confidence:.8}]}]},
    {complete:true,boxes:[{id:'first',bboxPixelsXYXY:[0,0,10,10]},{id:'second',bboxPixelsXYXY:[5,0,15,10]}]});
  assert.equal(metrics.methods[0].truePositives,2);assert.equal(new Set(metrics.methods[0].matches.map(x=>x.predictionId)).size,2);
});

test('one running test is enforced and interrupted jobs are durable failures after restart',async t=>{
  let finish;const worker=new FakeWorker();worker.run=(image,settings)=>new Promise(done=>{finish=()=>done(result(image,settings));});
  const {store}=await fixture(t,worker),run=await store.create({imageBase64});assert.equal(run.status,'running');
  await assert.rejects(store.create({imageBase64}),error=>error.code==='TESTBENCH_BUSY');
  assert.throws(()=>store.annotate(run.id,annotation()),error=>error.code==='TESTBENCH_NOT_COMPLETE');
  finish();await store.pending;store.db.prepare("UPDATE runs SET status='running' WHERE id=?").run(run.id);
  await store.stop();store.closed=false;await store.initialize();assert.equal(store.get(run.id).status,'failed');assert.equal(store.get(run.id).error.code,'TESTBENCH_INTERRUPTED');
});

test('malformed worker result fails closed and annotation boxes/settings reject unsupported coordinates',async t=>{
  const worker=new FakeWorker();worker.run=async(image,settings)=>({...result(image,settings),image:{...result(image,settings).image,sha256:'a'.repeat(64)}});
  const {store}=await fixture(t,worker),run=await store.create({imageBase64});await store.pending;
  assert.equal(store.get(run.id).status,'failed');assert.equal(store.get(run.id).error.code,'TESTBENCH_PROTOCOL');
  for(const settings of [{methods:['original','original']},{methods:['remote']},{methods:['tiles'],tileGrid:4},{methods:['color'],scale:.75},{methods:['original'],boxThreshold:.9}])assert.throws(()=>testbenchSettings(settings));
  const prepared=result({sha256:'a'.repeat(64),width:1,height:1},testbenchSettings());prepared.methods[0].candidates[0].bboxPixelsXYXY=[0,0,5,1];
  assert.throws(()=>testbenchResult(prepared,{sha256:'a'.repeat(64),width:1,height:1},testbenchSettings()));
});

test('conflicting independent human labels are quarantined and a permission revocation excludes duplicate runs',async t=>{
  const {store}=await fixture(t),first=await store.create({imageBase64});await store.pending;
  store.annotate(first.id,annotation(0,{trainingAllowed:true}));
  const second=await store.create({sourceRunId:first.id});await store.pending;
  store.annotate(second.id,annotation(1,{trainingAllowed:true,boxes:[]}));
  let exported=await store.exportDataset();assert.equal(exported.records.length,0);assert.equal(exported.summary.excluded.conflictingAnnotations,1);
  store.annotate(second.id,annotation(2,{trainingAllowed:true}));assert.equal((await store.exportDataset()).records.length,1);
  store.annotate(first.id,annotation(1,{trainingAllowed:false}));exported=await store.exportDataset();assert.equal(exported.records.length,0);assert.equal(exported.summary.excluded.rightsNotGranted,1);
});

test('explicit evaluation quarantine persists across reruns and repeated original uploads',async t=>{
  const {store}=await fixture(t);
  await assert.rejects(store.create({imageBase64,evaluationOnly:'true'}));
  const first=await store.create({imageBase64,evaluationOnly:true});await store.pending;
  const rerun=await store.create({sourceRunId:first.id,evaluationOnly:false});await store.pending;
  const duplicate=await store.create({imageBase64,evaluationOnly:false});await store.pending;
  for(const run of [first,rerun,duplicate])assert.equal(store.get(run.id).evaluationOnly,true);
  const saved=store.annotate(duplicate.id,annotation(0,{trainingAllowed:true}));assert.equal(saved.annotation.trainingAllowed,false);
  assert.equal((await store.exportDataset()).records.length,0);
});
