import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile,mkdtemp,rm } from 'node:fs/promises';
import { join,resolve,sep } from 'node:path';
import { tmpdir } from 'node:os';
import { RESEARCH_CATALOG,COMMUNITY_GROUPS,normalizeTaxonCandidate,researchTaxon } from '../community/public/research-taxonomy.js';
import { CommunityStore,validateReview,PARTICIPATION_TERMS } from '../community/store.mjs';
import { VideoGradesStore } from '../core/video-grades.mjs';

const candidate={taxonId:'morchella_americana',confidence:.5,evidenceSource:'image_only',evidenceNote:'Visible pits; stem base not visible.'};
const frame={id:'frame1',sourceId:'video1',width:640,height:480,sha256:'a'.repeat(64),timestampSeconds:1,evaluationOnly:false,
  source:{id:'video1',videoHash:'b'.repeat(64),sourceGroupId:'session:1',licenseAllowsTraining:true,evaluationOnly:false}};
const valid={status:'visible',boxes:[{bbox:[10,10,30,50],group:'morels',taxonCandidate:candidate}],complete:true,
  aiAssisted:false,baseRevision:0,participationTerms:{version:PARTICIPATION_TERMS.version,noticeSha256:PARTICIPATION_TERMS.noticeSha256}};

test('research names are source-linked, distinct from species capability and Michigan scope',async()=>{
  const canonical=JSON.parse(await readFile(new URL('../config/research-taxa-v1.json',import.meta.url),'utf8'));
  assert.deepEqual(RESEARCH_CATALOG,canonical);assert.ok(canonical.taxa.length>=100);
  const groups=new Set(canonical.groups.map(g=>g.id)),sources=new Set(canonical.sources.map(s=>s.id));
  assert.equal(new Set(canonical.taxa.map(t=>t.id)).size,canonical.taxa.length);
  for(const taxon of canonical.taxa){assert.ok(groups.has(taxon.broadGroupId));assert.ok(taxon.sourceIds.length);assert.ok(taxon.sourceIds.every(id=>sources.has(id)));
    assert.equal(taxon.modelSupport,'catalog_only');assert.equal(taxon.speciesAccuracyMeasured,false);assert.equal(taxon.salesEligibility,'not_assessed');}
  assert.equal(researchTaxon('gyromitra_esculenta').broadGroupId,'ascomycetes');
  assert.equal(researchTaxon('omphalotus_illudens').broadGroupId,'gilled');
  assert.equal(researchTaxon('amanita').rank,'genus');assert.equal(researchTaxon('amanita').nameResolution,'inat_exact_name_resolved');
  const curriculum=JSON.parse(await readFile(new URL('../config/michigan-mushrooms-v1.json',import.meta.url),'utf8'));
  assert.equal(curriculum.documentedIndexEntryCount,20);assert.equal(curriculum.entries.some(e=>e.id==='amanita'),false);
});
test('browser catalogs and editors are identical in standalone and community packages',async()=>{
  for(const name of ['research-taxonomy.js','research-taxon-editor.js'])assert.equal(
    await readFile(new URL('../community/public/'+name,import.meta.url),'utf8'),await readFile(new URL('../public/'+name,import.meta.url),'utf8'));
});
test('candidate validation does not accept forged verification, training, confidence or group',()=>{
  const saved=normalizeTaxonCandidate({...candidate,speciesVerified:true,status:'confirmed',modelSupport:'trained',trainingApproved:true},{groupId:'morels'});
  assert.equal(saved.status,'tentative');assert.equal(saved.speciesVerified,false);assert.equal(saved.modelSupport,'catalog_only');assert.equal(saved.trainingApproved,false);
  assert.equal(saved.confidenceMeaning,'reviewer_reported_not_calibrated');
  for(const change of [{taxonId:'safe_to_eat'},{confidence:NaN},{confidence:1.1},{confidence:'0.9'},{evidenceSource:'model_verified'},{evidenceSource:'dna',evidenceNote:''},{evidenceNote:'x'.repeat(1001)}])
    assert.throws(()=>normalizeTaxonCandidate({...candidate,...change}));
  assert.throws(()=>normalizeTaxonCandidate(candidate,{groupId:'gilled'}),/match the broad group/);
  assert.equal(normalizeTaxonCandidate(null),null);
});
test('community legacy groups remain accepted and structured research evidence round-trips',async()=>{
  for(const group of ['unknown','morels','chanterelles','boletes','black-trumpets','oysters','lion-mane','hen-of-the-woods','chicken-of-the-woods','hedgehogs','puffballs','other']){
    assert.ok(COMMUNITY_GROUPS.some(([id])=>id===group));
    const legacy=validateReview({...valid,boxes:[{bbox:[1,1,3,3],group}]},frame);
    assert.deepEqual(legacy.boxes,[{bbox:[1,1,3,3],group}]);
  }
  const directory=await mkdtemp(join(tmpdir(),'taxonomy-community-')),store=new CommunityStore(directory);
  try{const auth=store.createSession();store.save(auth.reviewer,frame,valid);const row=store.export().reviews[0];
    assert.equal(row.boxes[0].taxonCandidate.taxonId,'morchella_americana');assert.equal(row.trainingApproved,false);
    const imported=validateReview({...row,baseRevision:0},frame,{historical:true});assert.deepEqual(imported.boxes,row.boxes);
    store.withdraw(auth.reviewer,frame,{baseRevision:1});assert.equal(store.export().reviews[0].boxes[0].taxonCandidate.status,'tentative');
  }finally{store.close();assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});}
});
test('local video grading preserves tentative identities through save and reopen',async()=>{
  const directory=await mkdtemp(join(tmpdir(),'taxonomy-video-'));let store=await new VideoGradesStore(directory).initialize();
  try{const saved=store.save('collection',frame,{reviewer:'Test reviewer',status:'visible',baseRevision:0,complete:true,trainingAllowed:true,
    boxes:[{id:'b1',bboxPixelsXYXY:[10,10,30,50],groupId:'morels',taxonCandidate:candidate}]});
    assert.equal(saved.review.boxes[0].taxonCandidate.speciesVerified,false);
    store.close();store=await new VideoGradesStore(directory).initialize();
    assert.deepEqual(store.get('collection',frame).review.boxes[0].taxonCandidate,saved.review.boxes[0].taxonCandidate);
    assert.throws(()=>store.save('collection',frame,{reviewer:'Test reviewer',status:'visible',baseRevision:1,boxes:[{id:'b1',bboxPixelsXYXY:[10,10,30,50],groupId:'gilled',taxonCandidate:candidate}]}),/match the broad group/);
  }finally{store.close();assert.ok(resolve(directory).startsWith(resolve(tmpdir())+sep));await rm(directory,{recursive:true,force:true});}
});
