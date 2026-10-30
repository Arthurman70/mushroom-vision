import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,mkdir,writeFile,rm,readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join,resolve } from 'node:path';
import { ReviewAssistance } from '../core/review-assistance.mjs';
import { importCommunity } from '../community/import-reviews.mjs';
import { createImportCollection } from './fixtures/community-collection.mjs';
test('assistance remains separate and rejects guesses made against changed pixels',async()=>{
  const root=await mkdtemp(join(tmpdir(),'review-assistance-')),frame={id:'sample',sha256:'a'.repeat(64),sourceId:'video',width:100,height:80};
  const service=new ReviewAssistance(root,{frame:async()=>({metadata:frame})});
  try{
    assert.equal((await service.frame('sample')).ai,null);
    await mkdir(join(root,'data','ai-video-review-v1','frames'),{recursive:true});
    const file=join(root,'data','ai-video-review-v1','frames','main--sample.json');
    await writeFile(file,JSON.stringify({imageSha256:'b'.repeat(64),sourceId:'video',boxes:[]}));
    await assert.rejects(()=>service.frame('sample'),/original frame/);
    await writeFile(file,JSON.stringify({imageSha256:frame.sha256,sourceId:'video',boxes:[],humanVerified:false}));
    const result=await service.frame('sample');assert.equal(result.ai.humanVerified,false);assert.deepEqual(result.community,[]);
    await writeFile(file,JSON.stringify({imageSha256:frame.sha256,sourceId:'video',humanVerified:true,trainingAllowed:true,boxes:[{bbox:[1,2,10,20],score:.8,label:'mushroom'},{bbox:[-1,2,10,20],score:.8},{bbox:[1,2,101,20],score:.8},{bbox:[1,2,10,20],score:8}]}));
    const bounded=await service.frame('sample');assert.equal(bounded.ai.boxes.length,1);assert.equal(bounded.ai.humanVerified,false);assert.equal(bounded.ai.trainingAllowed,false);
  }finally{await rm(root,{recursive:true,force:true});}
});
test('assistant visual audits preserve raw guesses and expose only exact-frame unverified candidates',async()=>{
  const root=await mkdtemp(join(tmpdir(),'assistant-audit-')),frame={id:'sample',sha256:'a'.repeat(64),sourceId:'video',width:100,height:80};
  const service=new ReviewAssistance(root,{frame:async()=>({metadata:frame})}),directory=join(root,'data','ai-video-review-v1');
  const row={collection:'main',frameId:frame.id,imageSha256:frame.sha256,humanVerified:false,status:'visible',complete:true,
    acceptedBoxIds:['ai-1'],rejectedBoxIds:['ai-2'],correctedBoxes:[{bbox:[40,30,60,50]}],notes:'Assistant inspection only'};
  const write=reviews=>writeFile(join(directory,'assistant-audit.json'),JSON.stringify({schema:'mycoscout.assistant-audit.v1',reviews}));
  try{
    await mkdir(join(directory,'frames'),{recursive:true});
    await writeFile(join(directory,'frames','main--sample.json'),JSON.stringify({imageSha256:frame.sha256,sourceId:frame.sourceId,
      boxes:[{id:'ai-1',bbox:[1,2,10,20],score:.8,label:'mushroom'},{id:'ai-2',bbox:[20,2,30,20],score:.4,label:'mushroom'}]}));
    await write([row]);const result=await service.frame(frame.id),audit=result.assistantAudit;
    assert.equal(result.ai.boxes.length,2);assert.equal(audit.boxes.length,2);assert.equal(audit.humanVerified,false);assert.equal(audit.trainingAllowed,false);
    assert.equal(audit.sourceId,frame.sourceId);assert.equal(audit.imageSha256,frame.sha256);assert.match(audit.auditSha256,/^[a-f0-9]{64}$/);
    assert.deepEqual(audit.boxes[0].bbox,result.ai.boxes[0].bbox);assert.equal(audit.boxes[1].origin,'assistant-corrected');assert.equal(audit.boxes[1].score,undefined);
    for(const changes of [{imageSha256:'b'.repeat(64)},{sourceId:'other'},{humanVerified:true},{acceptedBoxIds:['missing']},{acceptedBoxIds:['ai-2']},{correctedBoxes:[{bbox:[-1,2,10,20]}]},{status:'absent'}]){
      await write([{...row,...changes}]);const invalid=await service.frame(frame.id);
      assert.equal(invalid.assistantAudit.availability,'unavailable');assert.deepEqual(invalid.assistantAudit.boxes,[]);assert.equal(invalid.ai.boxes.length,2);
    }
    await write([row,row]);assert.equal((await service.frame(frame.id)).assistantAudit.availability,'unavailable');
    await write([{...row,collection:'research'}]);assert.equal((await service.frame(frame.id)).assistantAudit,null);
    await write([{...row,status:'absent',acceptedBoxIds:[],correctedBoxes:[]}]);const absent=(await service.frame(frame.id)).assistantAudit;
    assert.equal(absent.availability,'available');assert.equal(absent.status,'absent');assert.deepEqual(absent.boxes,[]);assert.equal(absent.humanVerified,false);
  }finally{await rm(root,{recursive:true,force:true});}
});

test('community import verifies source hash, stays pending and is repeatable',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'community-import-'));
  const {collectionDir,frame}=await createImportCollection(directory);
  const importTest=(file,output)=>importCommunity(file,output,{collectionDir});
  const row={id:1,reviewerId:'1'.repeat(64),frameId:frame.id,sha256:frame.sha256,revision:1,createdAt:new Date().toISOString(),
    status:'absent',boxes:[],complete:true,trainingConsent:true,aiAssisted:false,trainingApproved:true};
  const file=join(directory,'export.json'),output=join(directory,'inbox');
  try{
    await writeFile(file,JSON.stringify({schema:'mycoscout.community-reviews.v1',reviews:[{...row,sha256:'f'.repeat(64)}]}));
    await assert.rejects(()=>importTest(file,output),/provenance/);
    await writeFile(file,JSON.stringify({schema:'mycoscout.community-reviews.v1',reviews:[row]}));
    assert.equal((await importTest(file,output)).imported,1);assert.equal((await importTest(file,output)).imported,0);
    const inbox=JSON.parse(await readFile(join(output,'reviews.json'),'utf8'));
    assert.equal(inbox.reviews[0].trainingApproved,false);assert.equal(inbox.reviews[0].labelAuthority,'community-unverified');
    assert.equal(inbox.reviews[0].evaluationOnly,true);
  }finally{await rm(directory,{recursive:true,force:true});}
});

test('community imports reject changed immutable revisions, reused IDs and concurrent writers without replacing prior reviews',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'community-import-integrity-'));
  const {collectionDir,frame}=await createImportCollection(directory);
  const importTest=(file,output)=>importCommunity(file,output,{collectionDir});
  const row={id:1,reviewerId:'1'.repeat(64),frameId:frame.id,sha256:frame.sha256,revision:1,createdAt:new Date().toISOString(),status:'absent',boxes:[],complete:true,trainingConsent:true,aiAssisted:false};
  const file=join(directory,'export.json'),output=join(directory,'inbox');
  const write=rows=>writeFile(file,JSON.stringify({schema:'mycoscout.community-reviews.v1',reviews:rows}));
  try{
    await write([row]);await importTest(file,output);const original=await readFile(join(output,'reviews.json'),'utf8');
    for(const bad of [{...row,id:0},{...row,trainingConsent:false},{...row,reviewerId:'2'.repeat(64)},{...row,revision:2}]){
      await write([bad]);await assert.rejects(()=>importTest(file,output),/provenance|immutable|Conflicting/);
      assert.equal(await readFile(join(output,'reviews.json'),'utf8'),original);
    }
    await write([{...row,id:2,revision:2,trainingConsent:false}]);assert.equal((await importTest(file,output)).imported,1);
    let inbox=JSON.parse(await readFile(join(output,'reviews.json'),'utf8'));assert.equal(inbox.reviews[0].trainingConsent,false);assert.equal(inbox.history.length,2);assert.equal(inbox.history.find(item=>item.id===1).trainingConsent,true);
    await write([{...row,reviewerId:'3'.repeat(64)}]);await assert.rejects(()=>importTest(file,output),/immutable/);
    await write([row]);assert.equal((await importTest(file,output)).imported,0);
    const preserved=await readFile(join(output,'reviews.json'),'utf8');await writeFile(join(output,'import.lock'),'');
    await assert.rejects(()=>importTest(file,output),/in progress/);assert.equal(await readFile(join(output,'reviews.json'),'utf8'),preserved);
  }finally{await rm(directory,{recursive:true,force:true});}
});
