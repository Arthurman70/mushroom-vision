import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {AIContributionStore,AI_TERMS,validateAIContribution,aiSourceAllowed,aiEligibleFrames} from '../community/ai-store.mjs';
import {createCommunityServer} from '../community/server.mjs';

const bytes=Buffer.from('synthetic-image-byte-fixture'),sha=createHash('sha256').update(bytes).digest('hex');
const source={id:'public-source',title:'Synthetic original',license:'CC BY 4.0',attribution:'Synthetic test author',sourceUrl:'https://example.org/source',licenseUrl:'https://creativecommons.org/licenses/by/4.0/'};
const frame={id:'public-frame',sourceId:source.id,sourceGroupId:'PRIVATE-GROUP-ID',sha256:sha,path:`images/${sha}.jpg`,width:300,height:200,timestampSeconds:0,evaluationOnly:false,sourceLicenseAllowsTraining:true};
const registration={reviewerKind:'ai',model:{provider:'Test',name:'Synthetic model',version:'1'},participationTerms:AI_TERMS};
const submission=task=>({taskId:task.taskId,imageSha256:task.imageSha256,reviewerKind:'ai',status:'visible',complete:true,
  boxes:[{bbox:[20,30,60,70],group:'unknown',confidence:.6}],confounders:[{bbox:[80,20,120,70],kind:'hat',confidence:.8}],
  observation:{imageViewed:true,mode:'original-image',imageSha256:task.imageSha256,note:'Synthetic test evidence, not a real image review.'},participationTerms:AI_TERMS});
async function fixture(){const directory=await mkdtemp(join(tmpdir(),'ai-inbox-'));const store=new AIContributionStore(directory);const credentials=store.register(registration),agent=store.authenticate(credentials.token);const task=store.lease(agent,[frame],new Map([[source.id,source]]),'/mushroom-lab/');return {store,directory,credentials,agent,task,async close(){store.close();await rm(directory,{recursive:true,force:true});}};}

test('AI queue restricts licenses and explicit split/training rights',()=>{
  assert.equal(aiSourceAllowed(frame,source),true);
  for(const license of ['CC BY-NC 4.0','All rights reserved','CC BY-ND 4.0','fake CC BY 4.0'])assert.equal(aiSourceAllowed(frame,{...source,license}),false);
  for(const license of ['CC0','CC0 1.0','CC BY-SA 4.0','CDLA Permissive 1.0'])assert.equal(aiSourceAllowed(frame,{...source,license}),true);
  assert.equal(aiSourceAllowed({...frame,evaluationOnly:undefined},source),false);
  assert.equal(aiSourceAllowed({...frame,sourceLicenseAllowsTraining:false},source),false);
  assert.equal(aiSourceAllowed(frame,{...source,sourceUrl:'file:///private'}),false);
});
test('registration stores only hashed expiring bearer credentials, no human session',async()=>{
  const f=await fixture();try{
    assert.equal(f.store.db.prepare('SELECT token_hash FROM ai_agents').get().token_hash,f.agent.id);
    assert.notEqual(f.agent.id,f.credentials.token);
    assert.throws(()=>f.store.authenticate(f.credentials.token,Date.now()+25*3600000),e=>e.status===401);
    assert.throws(()=>f.store.register({...registration,reviewerKind:'human'}));
    assert.throws(()=>f.store.register({...registration,model:{...registration.model,callbackUrl:'http://127.0.0.1'}}));
    assert.throws(()=>f.store.register({...registration,participationTerms:{}}),e=>e.status===428);
  }finally{await f.close();}
});
test('a source group with contradictory train and evaluation splits is withheld',()=>{
  const mixed=[frame,{...frame,id:'heldout-frame',evaluationOnly:true}],sources=new Map([[source.id,source]]);
  assert.deepEqual(aiEligibleFrames(mixed,sources),[]);
  assert.equal(aiEligibleFrames(mixed.map(f=>({...f,evaluationOnly:true})),sources).length,2);
});
test('leases one task with opaque group and preserves evaluation-only separation',async()=>{
  const f=await fixture();try{
    assert.equal(f.store.lease(f.agent,[frame],new Map([[source.id,source]]),'/mushroom-lab/').taskId,f.task.taskId);
    assert.equal(JSON.stringify(f.task).includes('PRIVATE-GROUP-ID'),false);
    assert.equal(f.task.split,'train_candidate');assert.equal(f.task.trainingApproved,false);
    const other=f.store.authenticate(f.store.register(registration).token);
    const heldout=f.store.lease(other,[{...frame,evaluationOnly:true}],new Map([[source.id,source]]),'/mushroom-lab/');
    assert.equal(heldout.split,'evaluation_only');assert.equal(heldout.evaluationOnly,true);
    assert.throws(()=>f.store.task(other,f.task.taskId),e=>e.status===404);
    assert.throws(()=>f.store.task(f.agent,f.task.taskId,Date.now()+31*60000),e=>e.status===410);
  }finally{await f.close();}
});
test('original retrieval and truthful image-view attestation are required separately',async()=>{
  const f=await fixture();try{
    assert.throws(()=>f.store.submit(f.agent,submission(f.task),'submission-key-0001',frame,source,'req'),e=>e.status===428);
    f.store.markDelivered(f.agent,f.task.taskId,sha);
    assert.throws(()=>f.store.submit(f.agent,{...submission(f.task),observation:{...submission(f.task).observation,imageViewed:false}},'submission-key-0001',frame,source,'req'));
    const saved=f.store.submit(f.agent,submission(f.task),'submission-key-0001',frame,source,'req').contribution;
    assert.equal(saved.observation.verification,'agent-self-attestation-with-server-image-delivery');assert.ok(saved.imageDeliveredAt);
    assert.equal(saved.reviewStatus,'quarantined');assert.equal(saved.trainingApproved,false);assert.equal(saved.humanVerified,false);
  }finally{await f.close();}
});
test('authority fields, arbitrary URLs, non-finite and out-of-bounds boxes are rejected',async()=>{
  const f=await fixture();try{
    const valid=submission(f.task),validate=input=>validateAIContribution(input,f.task,new Date().toISOString());
    for(const patch of [{humanVerified:true},{trainingApproved:true},{reviewerKind:'human'},{callbackUrl:'http://localhost'},{modelWeights:'run this'}])assert.throws(()=>validate({...valid,...patch}));
    for(const bbox of [[-1,0,1,1],[0,0,301,10],[0,0,10,201],[2,2,1,3],[0,0,Infinity,9],[0,0,NaN,9]])assert.throws(()=>validate({...valid,boxes:[{bbox,group:'unknown',confidence:.8}]}));
    assert.throws(()=>validate({...valid,boxes:[{...valid.boxes[0],humanVerified:true}]}));
    assert.throws(()=>validate({...valid,confounders:[{...valid.confounders[0],kind:'definitely-edible'}]}));
    assert.throws(()=>validate({...valid,observation:{...valid.observation,mode:'contact-sheet'}}));
    assert.throws(()=>validate({...valid,status:'uncertain'}));
    const abstain=validate({...valid,status:'uncertain',complete:false,boxes:[]});assert.equal(abstain.status,'uncertain');
  }finally{await f.close();}
});
test('stale image hashes, rights and split changes cannot be submitted',async()=>{
  const f=await fixture();try{
    f.store.markDelivered(f.agent,f.task.taskId,sha);const send=(input,fr=frame,s=source)=>f.store.submit(f.agent,input,'submission-key-0001',fr,s,'req');
    assert.throws(()=>send({...submission(f.task),imageSha256:'f'.repeat(64)}),e=>e.status===409);
    assert.throws(()=>send(submission(f.task),{...frame,sha256:'f'.repeat(64)}),e=>e.status===409);
    assert.throws(()=>send(submission(f.task),{...frame,evaluationOnly:true}),e=>e.status===409);
    assert.throws(()=>send(submission(f.task),{...frame,sourceGroupId:'different-group'}),e=>e.status===409);
    assert.throws(()=>send(submission(f.task),{...frame,width:301}),e=>e.status===409);
    assert.throws(()=>send(submission(f.task),frame,{...source,attribution:'different-author'}),e=>e.status===409);
    assert.throws(()=>send(submission(f.task),frame,{...source,license:'CC BY-NC 4.0'}),e=>e.status===409);
  }finally{await f.close();}
});
test('idempotent retries preserve one immutable record and receipt ownership',async()=>{
  const f=await fixture();try{
    f.store.markDelivered(f.agent,f.task.taskId,sha);const input=submission(f.task),send=(data,key='submission-key-0001')=>f.store.submit(f.agent,data,key,frame,source,'req');
    const first=send(input),again=send(input);assert.equal(first.duplicate,false);assert.equal(again.duplicate,true);assert.deepEqual(first.contribution,again.contribution);
    assert.throws(()=>send({...input,complete:false}),e=>e.status===409);assert.throws(()=>send(input,'submission-key-0002'),e=>e.status===409);
    assert.throws(()=>f.store.db.exec('UPDATE ai_contributions SET sha256="changed"'));
    assert.throws(()=>f.store.db.exec('DELETE FROM ai_contributions'),/immutable/);
    assert.equal(f.store.export().contributions.length,1);assert.equal(f.store.export().contributions[0].trainingApproved,false);
    assert.equal(f.store.get(f.agent,first.contribution.id).id,first.contribution.id);
    const other=f.store.authenticate(f.store.register(registration).token);assert.throws(()=>f.store.get(other,first.contribution.id),e=>e.status===404);
    const publicData=JSON.stringify(f.store.publicReviews(frame.id,sha));assert.equal(publicData.includes(f.credentials.token),false);assert.equal(publicData.includes(f.agent.id),false);assert.equal(publicData.includes(f.task.taskId),false);
    assert.equal(f.store.publicReviews(frame.id,'f'.repeat(64)).length,0);
  }finally{await f.close();}
});
test('credential daily contribution cap is persisted across leases',async()=>{
  const f=await fixture();try{
    const frames=Array.from({length:101},(_,i)=>({...frame,id:`f-${String(i).padStart(3,'0')}`})),sources=new Map([[source.id,source]]);
    // Use a distinct agent, since fixture agent already has an outstanding original task.
    const agent=f.store.authenticate(f.store.register(registration).token);
    for(let i=0;i<100;i++){
      const task=f.store.lease(agent,frames,sources,'/mushroom-lab/'),original=frames.find(fr=>fr.id===task.frameId);
      f.store.markDelivered(agent,task.taskId,sha);
      f.store.submit(agent,submission(task),'daily-submission-'+String(i).padStart(4,'0'),original,source,'request');
    }
    assert.throws(()=>f.store.lease(agent,frames,sources,'/mushroom-lab/'),e=>e.status===429);
    assert.equal(f.store.summary().contributions,100);
  }finally{await f.close();}
});
test('AI API performs retrieval, accepts webhook, protects human authority and emits safe receipts',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'ai-api-'));const collection=join(directory,'collection');await mkdir(join(collection,'images'),{recursive:true});
  await writeFile(join(collection,'manifest.json'),JSON.stringify({schema:'mycoscout.community-collection.v1',sources:[source],frames:[frame]}));await writeFile(join(collection,frame.path),bytes);
  const app=await createCommunityServer({dataDir:join(directory,'data'),collectionDir:collection,origin:'https://thegreatlakesgleaner.press'});
  await new Promise(r=>app.server.listen(0,'127.0.0.1',r));const base=`http://127.0.0.1:${app.server.address().port}/mushroom-lab/`;
  const jsonHeaders={'Content-Type':'application/json'},post=(path,data,headers={})=>fetch(base+path,{method:'POST',headers:{...jsonHeaders,...headers},body:JSON.stringify(data)});
  try{
    const info=await (await fetch(base+'api/ai/info')).json();assert.equal(info.eligibleFrames,1);assert.equal(info.trainingApproved,false);
    const registered=await post('api/ai/agents',registration);assert.equal(registered.status,201);assert.equal(registered.headers.has('set-cookie'),false);
    const agent=await registered.json(),headers={Authorization:'Bearer '+agent.token};
    assert.equal((await fetch(base+'api/ai/tasks')).status,401);
    const task=(await (await fetch(base+'api/ai/tasks?limit=1',{headers})).json()).tasks[0];
    assert.equal((await fetch(base+'api/ai/tasks?limit=2',{headers})).status,400);
    const input=submission(task),writeHeaders={...headers,'Idempotency-Key':'api-submission-00001'};
    assert.equal((await post('api/ai/contributions',input,writeHeaders)).status,428);
    const image=await fetch(new URL(task.imageUrl,base),{headers});assert.equal(image.status,200);assert.equal(image.headers.get('x-image-sha256'),sha);assert.deepEqual(Buffer.from(await image.arrayBuffer()),bytes);
    const response=await post('api/ai/contributions',input,writeHeaders),receipt=await response.json();assert.equal(response.status,201);assert.ok(response.headers.get('x-request-id'));assert.equal(receipt.contribution.humanVerified,false);
    assert.equal((await post('api/ai/contributions',input,writeHeaders)).status,200);
    assert.equal((await fetch(base+'api/ai/contributions/'+receipt.contribution.id)).status,401);
    assert.equal((await fetch(base+'api/ai/contributions/'+receipt.contribution.id,{headers})).status,200);
    const reviews=await (await fetch(base+'api/ai/reviews?frameId='+frame.id)).json();assert.equal(reviews.reviews.length,1);assert.equal(reviews.reviews[0].reviewStatus,'quarantined');
    assert.equal(app.store.summary().reviews,0);assert.equal(app.store.export().reviews.length,0);
    assert.equal((await post('api/reviews/'+frame.id,input,headers)).status,403);
    assert.equal((await post('api/ai/agents',registration,{Origin:'https://evil.example'})).status,403);
    assert.equal((await fetch(base+'data/ai-contributions.sqlite')).status,404);
    const spec=await (await fetch(base+'ai-openapi.json')).json();assert.equal(spec.openapi,'3.1.0');
    assert.equal((await post('api/ai/contributions',{...input,huge:'x'.repeat(40000)},writeHeaders)).status,413);
    for(let n=0;n<3;n++)assert.equal((await post('api/ai/agents',registration)).status,201);
    const limited=await post('api/ai/agents',registration);assert.equal(limited.status,429);assert.equal(limited.headers.get('retry-after'),'60');
  }finally{await app.close();await rm(directory,{recursive:true,force:true});}
});
test('failed original integrity check does not mint image-view receipt',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'ai-integrity-'));await mkdir(join(directory,'collection','images'),{recursive:true});
  await writeFile(join(directory,'collection','manifest.json'),JSON.stringify({schema:'mycoscout.community-collection.v1',sources:[source],frames:[frame]}));await writeFile(join(directory,'collection',frame.path),'tampered');
  const app=await createCommunityServer({dataDir:join(directory,'data'),collectionDir:join(directory,'collection')});await new Promise(r=>app.server.listen(0,'127.0.0.1',r));
  const base=`http://127.0.0.1:${app.server.address().port}`,credentials=app.aiStore.register(registration),agent=app.aiStore.authenticate(credentials.token),task=app.aiStore.lease(agent,[frame],new Map([[source.id,source]]),'/mushroom-lab/');
  try{assert.equal((await fetch(base+task.imageUrl,{headers:{Authorization:'Bearer '+credentials.token}})).status,409);assert.equal(app.aiStore.task(agent,task.taskId).deliveredAt,null);}
  finally{await app.close();await rm(directory,{recursive:true,force:true});}
});
