import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { TestbenchWorker } from '../core/testbench-worker.mjs';

function fixture(timeoutMs=1000){const child=new EventEmitter();child.stdout=new PassThrough();child.stderr=new PassThrough();child.stdin=new PassThrough();child.killed=false;child.kill=()=>{child.killed=true;};
  let launch;const worker=new TestbenchWorker('/trusted-root',{timeoutMs,spawnWorker:(...args)=>{launch=args;return child;}});
  return {worker,child,launch:()=>launch};}
const image={bytes:Buffer.from('fixture'),mimeType:'image/png'};
test('worker preserves one bounded request and progress while invoking only the fixed local module',async()=>{
  const {worker,child,launch}=fixture(),progress=[];
  const promise=worker.run(image,{methods:['original']},value=>progress.push(value));const request=JSON.parse(child.stdin.read().toString());
  assert.deepEqual(launch()[1],['-m','vision.testbench_worker']);assert.equal(launch()[2].windowsHide,true);assert.equal(launch()[2].env.HF_HUB_OFFLINE,'1');
  await assert.rejects(worker.run(image,{}),error=>error.code==='TESTBENCH_BUSY');
  child.stdout.write(JSON.stringify({id:request.id,status:'progress',progress:{stage:'original',completed:1,total:1}})+'\n');
  child.stdout.write(JSON.stringify({id:request.id,status:'ok',result:{fixture:true}})+'\n');assert.deepEqual(await promise,{fixture:true});assert.equal(progress.length,1);worker.stop();
});
test('worker wrong-ID and unbounded progress fail closed without passing a result',async()=>{
  for(const invalid of ['id','progress']){const {worker,child}=fixture();const promise=worker.run(image,{});const request=JSON.parse(child.stdin.read().toString());
    child.stdout.write(JSON.stringify(invalid==='id'?{id:'wrong',status:'ok',result:{}}:{id:request.id,status:'progress',progress:{stage:'tiles',completed:1,total:999}})+'\n');
    await assert.rejects(promise,error=>error.code==='TESTBENCH_PROTOCOL');assert.equal(child.killed,true);}
});
test('worker total timeout terminates inference and does not leak diagnostic text',async()=>{
  const {worker,child}=fixture(15),promise=worker.run(image,{});await assert.rejects(promise,error=>error.code==='TESTBENCH_TIMEOUT');assert.equal(child.killed,true);
  const next=fixture(),failed=next.worker.run(image,{}),request=JSON.parse(next.child.stdin.read().toString());
  next.child.stdout.write(JSON.stringify({id:request.id,status:'error',error:'private machine path and diagnostic data'})+'\n');
  await assert.rejects(failed,error=>error.code==='TESTBENCH_INFERENCE_FAILED'&&!error.message.includes('private'));next.worker.stop();
});
