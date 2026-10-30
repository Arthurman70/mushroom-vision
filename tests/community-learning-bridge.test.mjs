import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm, writeFile, readFile, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { resolve } from 'node:path';
import { CommunityLearningBridge } from '../core/community-learning-bridge.mjs';

async function fixture(t,options={}){
  const root=await mkdtemp(resolve(tmpdir(),'mycoscout-community-bridge-'));
  t.after(()=>rm(root,{recursive:true,force:true}));
  const calls=[];let time=Date.parse('2026-10-29T12:00:00Z');
  const bridge=await new CommunityLearningBridge(root,resolve(root,'bridge'),{
    now:()=>time,run:async args=>{calls.push(args);if(args[0]==='export-reviews')await writeFile(args[2],JSON.stringify({schema:'mycoscout.community-reviews.v1',reviews:[]}));},
    importReviews:async()=>({imported:2,total:3}),...options}).initialize();
  return {bridge,root,calls,advance:ms=>{time+=ms;}};
}
test('disabled bridge does no network work; enabled bridge syncs and preserves only imported evidence',async t=>{
  const {bridge,root,calls}=await fixture(t);await bridge.sync();assert.equal(calls.length,0);
  await bridge.configure({enabled:true});await bridge.sync();assert.equal(calls.length,1);assert.equal(bridge.snapshot().lastImported,2);
  assert.equal((await readdir(resolve(root,'bridge'))).some(n=>n.startsWith('transfer-')),false);
  const resumed=await new CommunityLearningBridge(root,resolve(root,'bridge')).initialize();assert.equal(resumed.snapshot().enabled,true);assert.equal(resumed.snapshot().lastTotal,3);
});
test('sync deduplicates concurrent requests, throttles idle polling, and force checks permissions before work',async t=>{
  const {bridge,calls,advance}=await fixture(t);await bridge.configure({enabled:true});
  await Promise.all([bridge.sync(),bridge.sync(),bridge.sync()]);assert.equal(calls.length,1);
  await bridge.sync();assert.equal(calls.length,1);await bridge.beforeWork();assert.equal(calls.length,2);
  advance(300001);await bridge.sync();assert.equal(calls.length,3);
});
test('failed permission import fails closed and never stamps a successful synchronization',async t=>{
  const {bridge,root}=await fixture(t,{importReviews:async()=>{throw new Error('revocation evidence invalid');}});
  await bridge.configure({enabled:true});await assert.rejects(bridge.beforeWork(),/revocation/);
  assert.equal(bridge.snapshot().lastSyncAt,null);assert.match(bridge.snapshot().error,/Training waits/);
  assert.equal((await readdir(resolve(root,'bridge'))).some(n=>n.startsWith('transfer-')),false);
});
test('disabling synchronization cannot reuse stale community permissions',async t=>{
  const {bridge}=await fixture(t);await bridge.beforeWork();
  await bridge.configure({enabled:true});await bridge.sync();await bridge.configure({enabled:false});
  await assert.rejects(bridge.beforeWork(),/Reconnect community synchronization/);
});
test('public publication only accepts its aggregate schema and leaves private bridge details local',async t=>{
  const {bridge,calls,root}=await fixture(t);await bridge.configure({enabled:true});
  await assert.rejects(bridge.publish({schema:'mycoscout.community-reviews.v1'}),/public learning summary/);
  await bridge.publish({schema:'mycoscout.community-learning-status.v1',updatedAt:'2026-10-29T12:00:00Z',worker:{state:'waiting_data'}});
  const output=JSON.parse(await readFile(resolve(root,'bridge','public-status.json'),'utf8'));
  assert.equal(output.lastExportSha256,undefined);assert.equal(calls[0][0],'update-training-status');
  await bridge.publish(output);assert.equal(calls.length,1);
});
