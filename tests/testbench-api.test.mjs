import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,rm,writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,resolve,sep } from 'node:path';
import { createHash } from 'node:crypto';
import { createApp } from '../server.mjs';

const imageBase64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2ZQkAAAAASUVORK5CYII=';
const settings={methods:['original'],tileGrid:2,scale:1,boxThreshold:.25};
function fakeResult(image,input){return {schema:'mycoscout.testbench-result.v1',image:{sha256:image.sha256,width:image.width,height:image.height,workingWidth:1,workingHeight:1},settings:input,
  model:{model:'API fixture',downloadsEnabled:false},methods:input.methods.map(id=>({id,candidates:[{id:'predicted-box',bboxPixelsXYXY:[0,0,1,1],confidence:.7}],seconds:.01,passes:1})),colorCues:[],seconds:.01,warnings:[]};}
async function fixture(t,{token='',sample=false}={}){
  const directory=await mkdtemp(join(tmpdir(),'mycoscout-testbench-api-'));
  const worker={requests:[],status:async()=>({available:true,busy:false}),run:async function(image,input,progress){this.requests.push({image,input});progress({stage:'original',completed:1,total:1});return fakeResult(image,input);},stop(){}};
  let testbenchSample;
  if(sample){const path=join(directory,'sample.png');await writeFile(path,Buffer.from(imageBase64,'base64'));testbenchSample={path,sha256:createHash('sha256').update(Buffer.from(imageBase64,'base64')).digest('hex'),name:'Test sample',attribution:'Test credit',licenseUrl:'https://example.org/license',sourceUrl:'https://example.org/source'};}
  const app=await createApp({host:'127.0.0.1',token,stateFile:join(directory,'state.json'),testbenchWorker:worker,testbenchSample});
  const address=await app.listen(0),base='http://127.0.0.1:'+address.port;
  t.after(async()=>{app.server.closeAllConnections();await new Promise(done=>app.server.close(done));await app.testbench.pending;await app.finds.pending;
    assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});});
  const request=async(path,body,headers={})=>{const response=await fetch(base+path,{method:body===undefined?'GET':'POST',headers:{...(body===undefined?{}:{'Content-Type':'application/json'}),...headers},body:body===undefined?undefined:JSON.stringify(body)});
    return {status:response.status,data:await response.json()};};
  return {app,worker,base,request};
}

test('testbench upload runs asynchronously, exposes original image and accepts explicit human truth',async t=>{
  const {app,request,base}=await fixture(t);
  const created=await request('/api/testbench/runs',{imageBase64,mimeType:'image/png',name:'My image',settings});assert.equal(created.status,202);
  await app.testbench.pending;const id=created.data.run.id,run=(await request('/api/testbench/runs/'+id)).data.run;
  assert.equal(run.status,'completed');assert.equal(run.image.width,1);assert.equal(run.metrics.status,'unannotated');
  const image=await fetch(base+run.imageUrl);assert.deepEqual(Buffer.from(await image.arrayBuffer()),Buffer.from(imageBase64,'base64'));
  const annotated=await request(`/api/testbench/runs/${id}/annotations`,{baseRevision:0,reviewer:'UI test fixture',complete:true,trainingAllowed:false,boxes:[],notes:'Explicit negative fixture'});
  assert.equal(annotated.status,201);assert.equal(annotated.data.annotation.emptySceneConfirmed,true);assert.equal(annotated.data.run.metrics.methods[0].falsePositives,1);
  assert.equal((await request('/api/testbench/runs')).data.runs[0].hasAnnotation,true);
  assert.equal((await request('/api/testbench/export')).data.records.length,0);
  const conflict=await request(`/api/testbench/runs/${id}/annotations`,{baseRevision:0,reviewer:'Old browser',complete:false,trainingAllowed:false,boxes:[]});
  assert.equal(conflict.status,409);assert.equal(conflict.data.code,'ANNOTATION_CONFLICT');
});

test('testbench auth and origin checks protect run details, photos, export and sample creation',async t=>{
  const token='testbench-secret-fixture-token-12345',{app,request,base}=await fixture(t,{token}),headers={Authorization:'Bearer '+token};
  for(const path of ['/api/testbench/status','/api/testbench/runs','/api/testbench/export','/api/testbench/sample'])assert.equal((await request(path)).status,401);
  assert.equal((await request('/api/testbench/runs',{imageBase64,settings})).status,401);
  assert.equal((await request('/api/testbench/runs',{imageBase64,settings},{...headers,Origin:'https://attacker.example'})).status,403);
  const created=await request('/api/testbench/runs',{imageBase64,settings},headers);await app.testbench.pending;
  assert.equal((await fetch(base+created.data.run.imageUrl)).status,401);assert.equal((await fetch(base+created.data.run.imageUrl,{headers})).status,200);
});

test('fixed sample, its reruns and exact-byte reuploads cannot enter training exports',async t=>{
  const {app,request}=await fixture(t,{sample:true});assert.equal((await request('/api/testbench/sample')).data.available,true);
  const created=await request('/api/testbench/sample',{settings:{...settings,scale:.5}});await app.testbench.pending;const id=created.data.run.id;
  assert.equal(created.data.run.settings.scale,.5);assert.deepEqual(created.data.run.settings.methods,['original']);
  assert.equal((await request(`/api/testbench/runs/${id}`)).data.run.annotation,null);
  const saved=await request(`/api/testbench/runs/${id}/annotations`,{baseRevision:0,reviewer:'Sample reviewer',complete:true,trainingAllowed:true,boxes:[]});
  assert.equal(saved.data.annotation.trainingAllowed,false);assert.equal(saved.data.run.evaluationOnly,true);
  const rerun=await request('/api/testbench/runs',{sourceRunId:id,settings});await app.testbench.pending;assert.equal(rerun.data.run.evaluationOnly,true);
  const reupload=await request('/api/testbench/runs',{imageBase64,settings});await app.testbench.pending;assert.equal(reupload.data.run.evaluationOnly,true);
  assert.equal((await request('/api/testbench/export')).data.records.length,0);
});

test('invalid images, settings, annotation fields and path requests do not start inference',async t=>{
  const {app,worker,request}=await fixture(t);
  assert.equal((await request('/api/testbench/runs',{imageBase64:'broken',settings})).status,400);
  assert.equal((await request('/api/testbench/runs',{imageBase64,settings:{...settings,tileGrid:20}})).status,400);
  assert.equal((await request('/api/testbench/runs',{sourceRunId:'../../data/state.json',settings})).status,404);
  assert.equal(worker.requests.length,0);
  const created=await request('/api/testbench/runs',{imageBase64,settings});await app.testbench.pending;
  const id=created.data.run.id,body={baseRevision:0,reviewer:'Fixture',complete:true,trainingAllowed:true,boxes:[]};
  assert.equal((await request(`/api/testbench/runs/${id}/annotations`,{...body,measuredDistanceM:-1})).status,400);
  assert.equal((await request(`/api/testbench/runs/${id}/annotations`,{...body,boxes:[{id:'x',bboxPixelsXYXY:[0,0,2,2],groupId:'unknown'}]})).status,400);
  assert.equal((await request(`/api/testbench/runs/${id}/annotations`,{...body,complete:'true'})).status,400);
  assert.equal((await request('/api/testbench/runs?limit=500')).status,400);
});
