import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtemp,mkdir,readFile,writeFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createHash} from 'node:crypto';
import {importCommunity} from '../community/import-reviews.mjs';
import {createImportCollection} from './fixtures/community-collection.mjs';
import { publicAudit } from '../community/public/ai-review-data.js';
import { validateReview,PARTICIPATION_TERMS } from '../community/store.mjs';
import { matchesReviewFilter,auditScope } from '../community/public/review-evidence.js';
const frame={id:'image-a',width:100,height:80};
const audit={reviewId:'bair-a',revision:1,reviewerKind:'ai',humanVerified:false,status:'visible',complete:true,observation:{mode:'contact-sheet',displayedWidth:50,displayedHeight:40},evidence:'A pale cap is visible; small preview.',boxes:[{bboxPixelsXYXY:[1,2,30,40],confidence:.8,groupId:'unknown',evidence:'Cap'}],confounders:[{bboxPixelsXYXY:[50,20,80,60],kind:'hat',explicitNonFungal:true,confidence:.9}],confounderTags:['hat']};
const review={status:'absent',boxes:[],negativeBoxes:[{bbox:[1,2,30,40],kind:'hat',trainingApproved:true}],complete:true,aiAssisted:true,baseRevision:0,participationTerms:PARTICIPATION_TERMS,aiAssessment:{reviewId:'bair-a',revision:1,decision:'corrected'}};
test('AI preview evidence stays partial, unverified, and public fields are allowlisted',()=>{
 const result=publicAudit({...audit,path:'PRIVATE',reviewer:'PRIVATE',gps:'PRIVATE'},frame);
 assert.equal(result.complete,false);assert.equal(result.humanVerified,false);assert.equal(result.trainingAllowed,false);assert.equal(result.boxes.length,1);assert.equal(result.confounders.length,1);assert.equal(JSON.stringify(result).includes('PRIVATE'),false);
 assert.match(auditScope(result),/reduced contact-sheet/);assert.equal(publicAudit({...audit,humanVerified:true},frame),null);
 assert.equal(publicAudit({...audit,observation:{mode:'fake'}},frame),null);
 assert.equal(publicAudit({...audit,boxes:[{bboxPixelsXYXY:[0,0,300,4]}]},frame).boxes.length,0);
 const malformed=publicAudit({...audit,boxes:{},confounders:'bad',confounderTags:{}},frame);
 assert.deepEqual(malformed.boxes,[]);assert.deepEqual(malformed.confounders,[]);assert.deepEqual(malformed.confounderTags,[]);
});
test('corrected hard-negative regions and exact AI review reference survive without becoming ground truth',()=>{
 const saved=validateReview(review,frame);assert.equal(saved.negativeBoxes[0].kind,'hat');assert.equal(saved.negativeBoxes[0].trainingApproved,false);assert.equal(saved.aiAssessment.reviewId,'bair-a');assert.equal(saved.trainingApproved,false);
 const historical=validateReview({...saved,baseRevision:1},frame,{historical:true});assert.deepEqual(historical,saved);
 assert.throws(()=>validateReview({...review,negativeBoxes:[{bbox:[0,0,3,3],kind:'human_verified_mushroom'}]},frame));
 assert.throws(()=>validateReview({...review,aiAssisted:false},frame));
 assert.throws(()=>validateReview({...review,negativeBoxes:[{bbox:[0,0,300,3],kind:'hat'}]},frame));
});
test('review filters distinguish pending, uncertain, source, image type, and confounders',()=>{
 const row={kind:'photo',sourceId:'source-1',ai:{audit:{...audit,status:'uncertain'}}};
 assert.equal(matchesReviewFilter(row,{status:'uncertain',kind:'photo',confounder:'hat',source:'source-1'}),true);
 for(const filter of [{status:'pending'},{kind:'video-frame'},{source:'source-2'},{confounder:'leaves'}])assert.equal(matchesReviewFilter(row,filter),false);
 assert.equal(matchesReviewFilter({...row,ai:{audit:null}},{status:'pending'}),true);
});
test('new public photo reviews import only against the published image bytes and preserve AI corrections',async()=>{
 const directory=await mkdtemp(join(tmpdir(),'community-photo-review-'));
 try{
  const {collectionDir}=await createImportCollection(directory),publicCollectionDir=join(directory,'public'),output=join(directory,'inbox'),file=join(directory,'export.json');
  const bytes=Buffer.from('synthetic-public-photo'),sha256=createHash('sha256').update(bytes).digest('hex');
  const photo={id:'image-'+sha256,path:`images/${sha256}.jpg`,sha256,width:100,height:80,sourceId:'public-photo',sourceGroupId:'public-observation',evaluationOnly:true};
  await mkdir(join(publicCollectionDir,'images'),{recursive:true});await writeFile(join(publicCollectionDir,photo.path),bytes);
  await writeFile(join(publicCollectionDir,'manifest.json'),JSON.stringify({schema:'mycoscout.community-collection.v1',frames:[photo]}));
  const row={...review,id:1,revision:1,reviewerId:'a'.repeat(64),frameId:photo.id,sha256,createdAt:'2026-10-30T00:00:00.000Z',trainingConsent:false,participationTerms:undefined,negativeBoxes:[{bbox:[1,2,30,40],kind:'animal'}]};
  await writeFile(file,JSON.stringify({schema:'mycoscout.community-reviews.v1',reviews:[row]}));
  assert.equal((await importCommunity(file,output,{collectionDir,publicCollectionDir})).imported,1);
  const saved=JSON.parse(await readFile(join(output,'reviews.json'),'utf8')).reviews[0];
  assert.equal(saved.negativeBoxes[0].kind,'animal');assert.equal(saved.negativeBoxes[0].trainingApproved,false);assert.equal(saved.aiAssessment.reviewId,'bair-a');assert.equal(saved.evaluationOnly,true);
  await writeFile(join(publicCollectionDir,photo.path),'replacement pixels');
  await assert.rejects(()=>importCommunity(file,output,{collectionDir,publicCollectionDir}),/integrity validation/);
 }finally{await rm(directory,{recursive:true,force:true});}
});
