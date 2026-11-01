import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { randomUUID } from 'node:crypto';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { LiveStation } from '../core/live-station.mjs';

function fixture(t){
  const calls=[],sessions=[];
  const spawnProcess=(executable,args,options)=>{
    calls.push({executable,args,options});
    const child=new EventEmitter();
    child.stdout=new EventEmitter();child.stdout.pause=()=>{};child.stdout.resume=()=>{};
    child.stderr=new EventEmitter();child.kill=()=>child.emit('close',0);
    return child;
  };
  const root=join(tmpdir(),'mycoscout-profile-'+randomUUID());
  const station=new LiveStation(root,join(root,'live'),{startSession:value=>sessions.push(value)},{spawnProcess});
  t.after(()=>station.close());
  return {station,calls,sessions};
}

test('fresh inference explicitly uses hybrid scans by default without changing saved replay',async t=>{
  const {station,calls}=fixture(t);
  const result=await station.startReplay({source:'fresh'});
  const index=calls[0].args.indexOf('--vision-profile');
  assert.ok(index>=0);assert.equal(calls[0].args[index+1],'hybrid');
  assert.equal(result.tileRegion,null);assert.equal(result.groundRegionVerified,false);
  assert.deepEqual(result.cadence,{detailIntervalSeconds:5,minimumFastFrames:2,fullDetailEvery:3,clock:'source_capture',backlog:false});
  assert.equal(calls[0].options.windowsHide,true);
  station.stopReplay();await station.replayFinished;
  const replay=await station.startReplay({source:'collection'});
  assert.equal(calls[1].args.includes('--vision-profile'),false);
  assert.equal(replay.visionProfile,undefined);
});

test('whole-view and fast modes reach fresh inference explicitly',async t=>{
  const {station,calls}=fixture(t);
  for(const mode of ['ground-tiles','full-tiles','fast']){
    const result=await station.startReplay({source:'fresh',visionProfile:mode});
    const args=calls.at(-1).args;
    assert.equal(args[args.indexOf('--vision-profile')+1],mode);
    assert.deepEqual(result.tileRegion,mode==='full-tiles'?[0,0,1,1]:mode==='ground-tiles'?[0,.35,1,1]:null);
    station.stopReplay();await station.replayFinished;
  }
});

test('unknown vision profiles fail before spawning or replacing a session',async t=>{
  const {station,calls,sessions}=fixture(t);
  for(const visionProfile of ['automatic-ground',null,{},'ground-tiles --input 0']){
    await assert.rejects(station.startReplay({source:'fresh',visionProfile}),/Choose hybrid search/);
  }
  assert.equal(calls.length,0);assert.equal(sessions.length,0);
});
