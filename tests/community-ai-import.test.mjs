import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {AIContributionStore,AI_TERMS} from '../community/ai-store.mjs';
import {importAIContributions} from '../community/import-ai-contributions.mjs';
async function fixture(evaluationOnly=false){
  const dir=await mkdtemp(join(tmpdir(),'ai-import-')),collectionDir=join(dir,'collection'),output=join(dir,'inbox'),file=join(dir,'export.json');
  const bytes=Buffer.from('synthetic published original'),sha=createHash('sha256').update(bytes).digest('hex');
  const source={id:'source',license:'CC BY 4.0',licenseUrl:'https://creativecommons.org/licenses/by/4.0/',sourceUrl:'https://example.org/image',attribution:'Synthetic author'};
  const frame={id:'frame',sourceId:'source',sourceGroupId:'group',width:300,height:200,sha256:sha,path:`images/${sha}.jpg`,sourceLicenseAllowsTraining:true,evaluationOnly};
  const manifest={schema:'mycoscout.community-collection.v1',sources:[source],frames:[frame]};await mkdir(join(collectionDir,'images'),{recursive:true});await writeFile(join(collectionDir,frame.path),bytes);
  await writeFile(join(collectionDir,'manifest.json'),JSON.stringify(manifest));
  const store=new AIContributionStore(join(dir,'db')),registered=store.register({reviewerKind:'ai',model:{provider:'Synthetic',name:'Fixture',version:'1'},participationTerms:AI_TERMS}),agent=store.authenticate(registered.token),task=store.lease(agent,[frame],new Map([['source',source]]),'/mushroom-lab/');
  store.markDelivered(agent,task.taskId,sha);store.submit(agent,{taskId:task.taskId,imageSha256:sha,reviewerKind:'ai',status:'absent',complete:true,boxes:[],confounders:[{bbox:[10,10,30,30],kind:'hat',confidence:.8}],observation:{imageViewed:true,mode:'original-image',imageSha256:sha,note:'Synthetic fixture.'},participationTerms:AI_TERMS},'import-test-key-0001',frame,source,'a'.repeat(32));
  const exported=store.export();store.close();await writeFile(file,JSON.stringify(exported));
  return {dir,collectionDir,output,file,frame,manifest,exported,run:()=>importAIContributions(file,output,{publicCollectionDir:collectionDir}),write:async value=>writeFile(file,JSON.stringify(value)),async close(){await rm(dir,{recursive:true,force:true});}};
}
test('AI exports import into a separate quarantine inbox and replay idempotently',async()=>{
  const f=await fixture();try{
    assert.deepEqual(await f.run(),{imported:1,total:1,trainingApproved:false,humanVerified:false,automaticTraining:false});assert.equal((await f.run()).imported,0);
    const inbox=JSON.parse(await readFile(join(f.output,'contributions.json'),'utf8'));assert.equal(inbox.schema,'mushroom-vision.community-ai-inbox.v1');assert.equal(inbox.contributions[0].source.license,'CC BY 4.0');assert.equal(inbox.contributions[0].split,'train_candidate');
    await assert.rejects(readFile(join(f.output,'reviews.json')),{code:'ENOENT'});
  }finally{await f.close();}
});
test('evaluation-only AI opinions retain their split and never gain training approval',async()=>{
  const f=await fixture(true);try{await f.run();const row=JSON.parse(await readFile(join(f.output,'contributions.json'),'utf8')).contributions[0];assert.equal(row.split,'evaluation_only');assert.equal(row.evaluationOnly,true);assert.equal(row.trainingApproved,false);}finally{await f.close();}
});
test('human exports and elevated AI authority are rejected before any inbox write',async()=>{
  const f=await fixture();try{
    for(const mutate of [d=>d.schema='mycoscout.community-reviews.v1',d=>d.humanVerified=true,d=>d.contributions[0].humanVerified=true,d=>d.contributions[0].trainingApproved=true,d=>d.contributions[0].reviewerKind='human',d=>d.contributions[0].reviewStatus='approved']){const value=structuredClone(f.exported);mutate(value);await f.write(value);await assert.rejects(f.run());}
    await assert.rejects(readFile(join(f.output,'contributions.json')),{code:'ENOENT'});
  }finally{await f.close();}
});
test('immutable duplicates, task reuse, changed source provenance and bad bounds fail closed',async()=>{
  const f=await fixture();try{
    await f.run();const before=await readFile(join(f.output,'contributions.json'),'utf8');
    for(const mutate of [d=>d.contributions.push(structuredClone(d.contributions[0])),d=>d.contributions[0].id='f'.repeat(48),d=>d.contributions[0].observation.note='Changed evidence',d=>d.contributions[0].imageWidth=301,d=>d.contributions[0].imageSha256='a'.repeat(64),d=>d.contributions[0].sourceGroupId='other',d=>d.contributions[0].source.license='CC0',d=>d.contributions[0].evaluationOnly=true,d=>d.contributions[0].confounders[0].bbox=[0,0,999,999],d=>d.contributions[0].imageDeliveredAt='2000-01-01T00:00:00.000Z',d=>d.contributions[0].participationTerms.noticeSha256='f'.repeat(64)]){
      const value=structuredClone(f.exported);mutate(value);await f.write(value);await assert.rejects(f.run());assert.equal(await readFile(join(f.output,'contributions.json'),'utf8'),before);
    }
  }finally{await f.close();}
});
test('current published image bytes and permission must still match exported records',async()=>{
  const f=await fixture();try{
    await writeFile(join(f.collectionDir,f.frame.path),'changed bytes');await assert.rejects(f.run(),/integrity/);
    f.manifest.sources[0].license='CC BY-NC 4.0';await writeFile(join(f.collectionDir,'manifest.json'),JSON.stringify(f.manifest));await assert.rejects(f.run(),/provenance/);
  }finally{await f.close();}
});
test('group split conflicts, external paths and concurrent import locks are refused',async()=>{
  const f=await fixture();try{
    f.manifest.frames.push({...f.frame,id:'other-frame',evaluationOnly:true});await writeFile(join(f.collectionDir,'manifest.json'),JSON.stringify(f.manifest));await assert.rejects(f.run(),/provenance/);
    f.manifest.frames.pop();f.manifest.frames[0].path='../export.json';await writeFile(join(f.collectionDir,'manifest.json'),JSON.stringify(f.manifest));await assert.rejects(f.run(),/identity/);
    f.manifest.frames[0].path=f.frame.path=`images/${f.frame.sha256}.jpg`;await writeFile(join(f.collectionDir,'manifest.json'),JSON.stringify(f.manifest));await mkdir(f.output,{recursive:true});await writeFile(join(f.output,'import.lock'),'another importer');await assert.rejects(f.run(),/in progress/);
  }finally{await f.close();}
});
