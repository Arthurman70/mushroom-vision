import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp,writeFile,mkdir,rm,readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { CommunityStore,validateReview,PARTICIPATION_TERMS } from '../community/store.mjs';
import { createCommunityServer,readTrainingStatus,STATUS_MAX_AGE_MS } from '../community/server.mjs';
import { priorityFrameIndex } from '../community/public/app.js';
import { publicSourceAllowed } from '../community/package-collection.mjs';
import { importCommunity } from '../community/import-reviews.mjs';
import { createImportCollection } from './fixtures/community-collection.mjs';
const imageBytes=Buffer.from('test-jpeg'),imageHash=createHash('sha256').update(imageBytes).digest('hex');
const frame={id:'example-001',sourceId:'example',width:320,height:240,timestampSeconds:1,sha256:imageHash,path:'images/'+imageHash+'.jpg'};
const terms={version:PARTICIPATION_TERMS.version,noticeSha256:PARTICIPATION_TERMS.noticeSha256};
const valid={status:'visible',boxes:[{bbox:[10,20,30,50],group:'morels'}],complete:true,participationTerms:terms,aiAssisted:false,baseRevision:0};
test('community submissions cannot manufacture confirmation or training approval',()=>{
  const saved=validateReview({...valid,humanVerified:true,trainingApproved:true},frame);
  assert.equal(saved.labelAuthority,'community-unverified');assert.equal(saved.trainingApproved,false);assert.equal(saved.humanVerified,undefined);
});
test('invalid coordinates, group, incomplete truths and uncertain negatives are rejected',()=>{
  for(const data of [{...valid,boxes:[{bbox:[0,0,500,10]}]},{...valid,boxes:[{bbox:[2,2,1,3]}]},
    {...valid,boxes:[{bbox:[1,1,2,2],group:'poison-free'}]}, {...valid,status:'absent'}, {...valid,boxes:[]}, {...valid,status:'uncertain',boxes:[]}])
    assert.throws(()=>validateReview(data,frame));
});
test('reviews retain independent reviewers, immutable history and stale revision conflicts',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-store-')),store=new CommunityStore(dir);
  try{
    const a=store.createSession(),b=store.createSession();assert.equal(store.session(a.token).reviewer,a.reviewer);
    store.save(a.reviewer,frame,valid);store.save(b.reviewer,frame,{...valid,status:'absent',boxes:[]});
    assert.throws(()=>store.save(a.reviewer,frame,valid),/another tab/);
    store.save(a.reviewer,frame,{...valid,baseRevision:1,aiAssisted:true});
    assert.equal(store.get(a.reviewer,frame.id).revision,2);assert.equal(store.get(b.reviewer,frame.id).status,'absent');
    assert.equal(store.summary().reviews,2);assert.equal(store.export().reviews.length,2);
    assert.throws(()=>store.db.exec('UPDATE reviews SET revision=9'),/immutable/);
  }finally{store.close();await rm(dir,{recursive:true,force:true});}
});
test('public API protects private reviews, CSRF, routes and media; saving is durable',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-api-'));await mkdir(join(dir,'collection','images'),{recursive:true});
  await writeFile(join(dir,'collection','manifest.json'),JSON.stringify({schema:'mycoscout.community-collection.v1',sources:[{id:'example'}],frames:[frame]}));
  await writeFile(join(dir,'collection',frame.path),imageBytes);
  const app=await createCommunityServer({dataDir:join(dir,'db'),collectionDir:join(dir,'collection'),origin:'https://thegreatlakesgleaner.press'});
  await new Promise(r=>app.server.listen(0,'127.0.0.1',r));const url=`http://127.0.0.1:${app.server.address().port}/mushroom-lab/`;
  const headers={'Content-Type':'application/json',Origin:'https://thegreatlakesgleaner.press'};
  try{
    assert.equal((await fetch(url+'api/session',{method:'POST',headers:{...headers,Origin:'https://evil.example'},body:'{}'})).status,403);
    const response=await fetch(url+'api/session',{method:'POST',headers,body:'{}'}),session=await response.json(),{csrfToken}=session;
    assert.deepEqual(session.participationTerms,PARTICIPATION_TERMS);
    const cookie=response.headers.get('set-cookie').split(';')[0];assert.match(response.headers.get('set-cookie'),/HttpOnly; SameSite=Strict/);
    assert.equal((await fetch(url+'api/reviews/example-001',{method:'POST',headers:{...headers,Cookie:cookie},body:JSON.stringify(valid)})).status,403);
    const save=await fetch(url+'api/reviews/example-001',{method:'POST',headers:{...headers,Cookie:cookie,'X-CSRF-Token':csrfToken},body:JSON.stringify(valid)});
    assert.equal(save.status,201);const saved=(await save.json()).review;assert.equal(saved.trainingApproved,false);assert.equal(saved.trainingConsent,true);
    assert.equal(saved.participationTerms.acceptedAt,saved.createdAt);assert.equal(saved.participationTerms.noticeSha256,terms.noticeSha256);
    const staleTerms={...valid,baseRevision:1,participationTerms:undefined};
    assert.equal((await fetch(url+'api/reviews/example-001',{method:'POST',headers:{...headers,Cookie:cookie,'X-CSRF-Token':csrfToken},body:JSON.stringify(staleTerms)})).status,428);
    assert.equal((await (await fetch(url+'api/reviews/example-001')).json()).review,null);
    assert.equal((await (await fetch(url+'api/reviews/example-001',{headers:{Cookie:cookie}})).json()).review.boxes.length,1);
    assert.equal((await fetch(url+'api/live/tasks')).status,404);assert.equal((await fetch(url+'data/community.sqlite')).status,404);
    assert.equal((await fetch(url+'api/training/start',{method:'POST',headers:{...headers,Cookie:cookie,'X-CSRF-Token':csrfToken},body:'{}'})).status,404);
    await writeFile(join(dir,'db','training-status.json'),JSON.stringify({...learningStatus(),reviewer:'PRIVATE-PROGRESS',path:'PRIVATE-PROGRESS'}));
    const publicProgress=await (await fetch(url+'api/training-status')).json();assert.equal(publicProgress.availability,'available');assert.equal(JSON.stringify(publicProgress).includes('PRIVATE-PROGRESS'),false);
    assert.equal((await fetch(url+'data/training-status.json')).status,404);
    assert.equal((await fetch(url+'api/reviews/example-001/withdraw',{method:'POST',headers:{...headers,Cookie:cookie},body:'{"baseRevision":1}'})).status,403);
    const another=await fetch(url+'api/session',{method:'POST',headers,body:'{}'}),otherSession=await another.json(),otherCookie=another.headers.get('set-cookie').split(';')[0];
    assert.equal((await fetch(url+'api/reviews/example-001/withdraw',{method:'POST',headers:{...headers,Cookie:otherCookie,'X-CSRF-Token':otherSession.csrfToken},body:'{"baseRevision":1}'})).status,404);
    const withdrew=await fetch(url+'api/reviews/example-001/withdraw',{method:'POST',headers:{...headers,Cookie:cookie,'X-CSRF-Token':csrfToken},body:'{"baseRevision":1}'});
    assert.equal(withdrew.status,201);assert.equal((await withdrew.json()).review.trainingConsent,false);
    assert.equal((await fetch(url+'images/example-001')).status,200);
    const collection=await (await fetch(url+'api/collection')).json();assert.equal(collection.frames[0].path,undefined);assert.equal(collection.frames[0].sha256,undefined);assert.match(collection.frames[0].imageUrl,new RegExp('v='+frame.sha256));
    const researchCatalog=await (await fetch(url+'api/research-catalog')).json();assert.ok(researchCatalog.taxa.length>=100);assert.equal(researchCatalog.model.speciesModelTrained,false);
    for(const asset of ['research-taxonomy.js','research-taxon-editor.js']){const response=await fetch(url+asset);assert.equal(response.status,200);assert.match(response.headers.get('content-type'),/javascript/);}
    const suggestions={frames:[{frameId:frame.id,sourceId:frame.sourceId,imageSha256:'f'.repeat(64),status:'completed',boxes:[{bbox:[1,1,5,5],score:.8,label:'mushroom'}]}]};
    await writeFile(join(dir,'collection','ai-suggestions.json'),JSON.stringify(suggestions));
    assert.deepEqual((await (await fetch(url+'api/collection')).json()).frames[0].ai.boxes,[]);
    suggestions.frames[0].imageSha256=frame.sha256;suggestions.frames[0].boxes.push({bbox:[0,0,999,5],score:.8,label:'outside'});
    await writeFile(join(dir,'collection','ai-suggestions.json'),JSON.stringify(suggestions));
    assert.equal((await (await fetch(url+'api/collection')).json()).frames[0].ai.boxes.length,1);
    assert.equal((await fetch(url+'images/example-001?v='+ 'f'.repeat(64))).status,409);
    await writeFile(join(dir,'collection',frame.path),'changed image pixels');
    assert.equal((await fetch(url+'images/example-001')).status,409);
  }finally{await app.close();await rm(dir,{recursive:true,force:true});}
});

test('new participation terms are bound to displayed notice and server time, without manufacturing human approval',()=>{
  const acceptedAt='2026-10-29T18:00:00.000Z';
  const result=validateReview({...valid,trainingConsent:true,participationTerms:{...terms,acceptedAt:'2000-01-01T00:00:00.000Z'},humanVerified:true},frame,{acceptedAt});
  assert.equal(result.trainingConsent,true);assert.equal(result.participationTerms.acceptedAt,acceptedAt);
  assert.equal(result.participationTerms.noticeSha256,createHash('sha256').update(PARTICIPATION_TERMS.notice).digest('hex'));
  assert.equal(result.humanVerified,undefined);assert.equal(result.trainingApproved,false);
  for(const input of [{...valid,participationTerms:undefined},{...valid,participationTerms:{...terms,version:'old'}},{...valid,participationTerms:{...terms,noticeSha256:'f'.repeat(64)}}])
    assert.throws(()=>validateReview(input,frame),error=>error.status===428);
  assert.throws(()=>validateReview({...valid,trainingConsent:false},frame),/withdrawal/);
});

test('historical false consent and withdrawal history survive reopening and later submission',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-terms-'));let store=new CommunityStore(dir);
  try {
    const user=store.createSession(),legacy={status:'visible',boxes:valid.boxes,complete:true,trainingConsent:false,aiAssisted:false,labelAuthority:'community-unverified',trainingApproved:false};
    store.db.prepare('INSERT INTO reviews(reviewer,frame,sha256,revision,created_at,payload) VALUES(?,?,?,?,?,?)').run(user.reviewer,frame.id,frame.sha256,1,'2026-10-28T00:00:00.000Z',JSON.stringify(legacy));
    const untouched=store.db.prepare('SELECT payload FROM reviews WHERE revision=1').get().payload;store.close();store=new CommunityStore(dir);
    assert.equal(store.get(user.reviewer,frame.id).trainingConsent,false);
    assert.deepEqual(validateReview({...legacy,baseRevision:0},frame,{historical:true}),legacy);
    const current=store.save(user.reviewer,frame,{...valid,baseRevision:1});assert.equal(current.trainingConsent,true);
    const withdrawn=store.withdraw(user.reviewer,frame,{baseRevision:2,status:'absent',boxes:[]});
    assert.equal(withdrawn.revision,3);assert.equal(withdrawn.trainingConsent,false);assert.deepEqual(withdrawn.boxes,valid.boxes);
    assert.equal(withdrawn.permissionWithdrawal.priorRevision,2);assert.equal(withdrawn.participationTerms.acceptedAt,current.createdAt);
    const historical=validateReview({...withdrawn,baseRevision:2},frame,{historical:true});assert.deepEqual(historical.permissionWithdrawal,withdrawn.permissionWithdrawal);
    assert.equal(store.db.prepare('SELECT payload FROM reviews WHERE revision=1').get().payload,untouched);
    assert.throws(()=>store.withdraw(user.reviewer,frame,{baseRevision:2}),/another tab/);
    assert.throws(()=>store.withdraw(user.reviewer,frame,{baseRevision:3}),/already withheld/);
    assert.equal(store.export().reviews[0].trainingConsent,false);assert.equal(store.db.prepare('SELECT COUNT(*) AS n FROM reviews').get().n,3);
  }finally{store.close();await rm(dir,{recursive:true,force:true});}
});

test('community import preserves legacy refusal, participation provenance and withdrawal as separate immutable revisions',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-terms-import-')),output=join(dir,'inbox'),file=join(dir,'export.json');
  const {collectionDir,frame:source}=await createImportCollection(dir),at='2026-10-29T18:00:00.000Z';
  const importTest=(file,output)=>importCommunity(file,output,{collectionDir});
  const base={id:1,revision:1,reviewerId:'a'.repeat(64),frameId:source.id,sha256:source.sha256,createdAt:at,status:'absent',boxes:[],complete:true,trainingConsent:false,aiAssisted:false};
  const write=reviews=>writeFile(file,JSON.stringify({schema:'mycoscout.community-reviews.v1',reviews}));
  try{
    await write([base]);await importTest(file,output);
    const granted={...base,id:2,revision:2,trainingConsent:true,participationTerms:{...terms,acceptedAt:at,acceptance:'submission'}};
    await write([granted]);await importTest(file,output);
    const withdrawn={...granted,id:3,revision:3,trainingConsent:false,permissionWithdrawal:{withdrawnAt:at,scope:'future-training-and-evaluation',priorRevision:2}};
    await write([withdrawn]);await importTest(file,output);
    const snapshot=await readFile(join(output,'reviews.json'),'utf8'),inbox=JSON.parse(snapshot);
    assert.deepEqual(inbox.history.map(r=>r.trainingConsent),[false,true,false]);assert.equal(inbox.history[0].participationTerms,undefined);
    assert.deepEqual(inbox.history[1].participationTerms,granted.participationTerms);assert.deepEqual(inbox.reviews[0].permissionWithdrawal,withdrawn.permissionWithdrawal);
    assert.ok(inbox.history.every(r=>r.trainingApproved===false&&r.labelAuthority==='community-unverified'));
    await write([{...withdrawn,participationTerms:{...withdrawn.participationTerms,noticeSha256:'f'.repeat(64)}}]);
    await assert.rejects(()=>importTest(file,output),/Conflicting/);assert.equal(await readFile(join(output,'reviews.json'),'utf8'),snapshot);
  }finally{await rm(dir,{recursive:true,force:true});}
});

const learningStatus=()=>({schema:'mycoscout.community-learning-status.v1',updatedAt:new Date().toISOString(),worker:{state:'waiting_data',enabled:true,paused:false},approvedFrames:25,newFrames:12,newSessions:2,thresholds:{newFrames:50,newSessions:3,maxRunsPerDay:1},latestCandidate:null,automaticActivation:false});
test('training status is bounded, allowlisted and offline when missing or stale',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-status-')),path=join(dir,'training-status.json');
  try {
    assert.equal((await readTrainingStatus(dir)).availability,'missing');
    const input=learningStatus();input.path='SECRET';input.reviewerIds=['SECRET'];input.worker.error='SECRET';input.latestCandidate={status:'candidate_not_better',completedAt:input.updatedAt,path:'SECRET',comparison:{baselineRecall:.4,candidateRecall:.3,baselineFalsePositivesPerFrame:2,candidateFalsePositivesPerFrame:1,raw:'SECRET'}};
    await writeFile(path,JSON.stringify(input));let output=await readTrainingStatus(dir);assert.equal(output.availability,'available');assert.equal(JSON.stringify(output).includes('SECRET'),false);
    output=await readTrainingStatus(dir,Date.parse(input.updatedAt)+STATUS_MAX_AGE_MS+1);assert.equal(output.availability,'stale');assert.equal(output.worker.state,'offline');assert.equal(output.worker.enabled,false);assert.equal(output.approvedFrames,25);
    for(const bad of ['{bad',' '.repeat(65537),JSON.stringify({...input,newFrames:-1}),JSON.stringify({...input,automaticActivation:true}),JSON.stringify({...input,updatedAt:'2099-01-01T00:00:00.000Z'})]){
      await writeFile(path,bad);output=await readTrainingStatus(dir);assert.equal(output.availability,'invalid');assert.equal(output.worker.enabled,false);assert.equal(output.approvedFrames,null);
    }
    await writeFile(path,JSON.stringify({...learningStatus(),approvedFrames:null,newFrames:null,newSessions:null,worker:{state:'waiting_sync',enabled:true,paused:false}}));
    assert.equal((await readTrainingStatus(dir)).approvedFrames,null);
  }finally{await rm(dir,{recursive:true,force:true});}
});

test('priority choices preserve source navigation and do not expose reviewer identities',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-priority-')),store=new CommunityStore(dir);
  try{const a=store.createSession(),b=store.createSession();store.save(a.reviewer,frame,valid);assert.equal(store.priorityFrames().get(frame.id),2);
    store.save(b.reviewer,frame,{...valid,status:'absent',boxes:[]});assert.equal(store.priorityFrames().get(frame.id),3);
    store.withdraw(a.reviewer,frame,{baseRevision:1});assert.equal(store.priorityFrames().get(frame.id),0);
    store.withdraw(b.reviewer,frame,{baseRevision:1});assert.equal(store.priorityFrames().has(frame.id),false);
    const rows=[{id:'same-1',reviewPriority:1},{id:'same-2',reviewPriority:3},{id:'other-1',reviewPriority:2}],original=JSON.stringify(rows);
    assert.equal(priorityFrameIndex(rows,'same-1'),1);assert.equal(priorityFrameIndex(rows,'same-1',new Set(['same-2'])),2);assert.equal(JSON.stringify(rows),original);
  }finally{store.close();await rm(dir,{recursive:true,force:true});}
});

test('participation notice is beside Submit without a training checkbox and keeps a separate withdrawal',async()=>{
  const html=await readFile(new URL('../community/public/index.html',import.meta.url),'utf8');
  assert.equal(html.includes('id="training-consent"'),false);assert.ok(html.includes(PARTICIPATION_TERMS.notice));assert.match(html,/id="withdraw-permission"/);
  assert.ok(html.indexOf('id="participation-notice"')<html.indexOf('id="save-next"'));assert.match(html,/id="review-complete"/);
});

test('public source selection excludes noncommercial and no-derivatives variants',()=>{
  for(const license of ['CC BY 4.0','CC BY-SA 2.0','CC BY (YouTube)'])assert.equal(publicSourceAllowed({license,licenseAllowsTraining:true}),true);
  for(const license of ['CC BY-ND 4.0','CC BY-NC 4.0','CC BY-SA-ND 4.0','All rights reserved','CC BY 4.0 private'])assert.equal(publicSourceAllowed({license,licenseAllowsTraining:true}),false);
  assert.equal(publicSourceAllowed({license:'CC BY 4.0',licenseAllowsTraining:false}),false);
});

test('saved community reviews cannot attach to replacement pixels under the same frame ID',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-pixels-')),store=new CommunityStore(dir);
  try{const user=store.createSession();store.save(user.reviewer,frame,valid);const changed={...frame,sha256:'c'.repeat(64)};
    assert.throws(()=>store.get(user.reviewer,changed.id,changed.sha256),/original pixels changed/);
    assert.throws(()=>store.save(user.reviewer,changed,{...valid,baseRevision:1}),/original pixels changed/);
    assert.throws(()=>store.save(store.createSession().reviewer,changed,valid),/original pixels changed/);
    assert.equal(store.get(user.reviewer,frame.id,frame.sha256).revision,1);
  }finally{store.close();await rm(dir,{recursive:true,force:true});}
});

test('invalid public dimensions and disconnected sources fail before service startup',async()=>{
  const dir=await mkdtemp(join(tmpdir(),'community-manifest-'));
  try{for(const bad of [{...frame,width:undefined},{...frame,height:0},{...frame,width:1e9},{...frame,sourceId:'private-other'},{...frame,path:'images/'+ 'f'.repeat(64)+'.jpg'}]){
    await writeFile(join(dir,'manifest.json'),JSON.stringify({schema:'mycoscout.community-collection.v1',sources:[{id:'example'}],frames:[bad]}));
    await assert.rejects(()=>createCommunityServer({collectionDir:dir,dataDir:join(dir,'db')}),/Invalid public frame/);
  }}finally{await rm(dir,{recursive:true,force:true});}
});
