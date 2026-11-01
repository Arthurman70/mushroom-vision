import test from 'node:test';
import assert from 'node:assert/strict';
import { selectExternalReviews } from '../community/public/app.js';

const hash = 'a'.repeat(64);
const frame = { id:'frame-one', width:100, height:80, imageUrl:`/mushroom-lab/images/frame-one?v=${hash}` };
const row = { id:'contribution-one', frameId:frame.id, imageSha256:hash, humanVerified:false, trainingApproved:false, reviewStatus:'quarantined', status:'visible', complete:true, model:{provider:'test',name:'fixture',version:'1'}, observation:{note:'Test fixture only.'}, boxes:[{bbox:[10,20,30,40],group:'unknown'}], confounders:[] };
const response = (changes={}) => ({frameId:frame.id,imageSha256:hash,reviews:[{...row,...changes}]});

test('external opinions retain explicit separate evidence and pixel boxes',()=>{
  const actual=selectExternalReviews(response(),frame);
  assert.equal(actual.length,1);assert.equal(actual[0].model,'test · fixture · 1');assert.deepEqual(actual[0].boxes,[{bbox:[10,20,30,40],group:'unknown'}]);
  actual[0].boxes[0].bbox[0]=50;assert.equal(row.boxes[0].bbox[0],10);
});
test('external opinions fail closed on frame or image changes',()=>{
  assert.deepEqual(selectExternalReviews({...response(),frameId:'another-frame'},frame),[]);
  assert.deepEqual(selectExternalReviews({...response(),imageSha256:'b'.repeat(64)},frame),[]);
  assert.deepEqual(selectExternalReviews(response({imageSha256:'b'.repeat(64)}),frame),[]);
  assert.deepEqual(selectExternalReviews(response(),{...frame,imageUrl:'/image-without-hash'}),[]);
});
test('external opinions cannot claim human or training authority',()=>{
  for(const changes of [{humanVerified:true},{trainingApproved:true},{reviewStatus:'approved'},{status:'confirmed'}])assert.deepEqual(selectExternalReviews(response(changes),frame),[]);
});
test('external region bounds are validated, never silently clamped',()=>{
  const result=selectExternalReviews(response({boxes:[{bbox:[-1,0,20,30]},{bbox:[0,0,110,80]},{bbox:[0,0,0,10]},{bbox:[0,0,20,30],group:'invented'}]}),frame);
  assert.deepEqual(result[0].boxes,[{bbox:[0,0,20,30],group:'unknown'}]);
});
test('external opinion count and visual notes are bounded',()=>{
  const input=response({observation:{note:'x'.repeat(5000)}});input.reviews=Array(30).fill(input.reviews[0]);
  const result=selectExternalReviews(input,frame);assert.equal(result.length,20);assert.equal(result[0].note.length,2000);
});
