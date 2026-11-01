import test from 'node:test';
import assert from 'node:assert/strict';
import {publicRound4} from '../community/public/testing-round4.js';

const input=()=>({schema:'mushroom-vision.public-testing-round4.v1',title:'Test fixture',completedAt:'2026-10-31T00:00:00Z',summary:{positiveReferences:1,negativeReferences:0,referenceAuthority:'ai-incomplete',weightsChanged:false},conditions:[{id:'native',label:'Native',selectedTargetHits:0,selectedTargetCount:1,negativeFramesWithAlerts:null,negativeFrameCount:null,negativeAlertCount:null,meanSecondsPerFrame:1.2,measurementFrames:1}],frames:[{frameId:'public-frame',imageSha256:'a'.repeat(64),width:100,height:80,status:'visible',note:'Incomplete AI reference',groupHint:'unknown',selectedTargetPixelsXYXY:[1,2,20,30],source:{attribution:'Fixture',license:'CC0',sourceUrl:'https://example.org/image',licenseUrl:'https://creativecommons.org/publicdomain/zero/1.0/'},results:[{conditionId:'native',selectedTargetHit:false,alertCount:0,elapsedSeconds:1.2}]}]});

test('round4 public view retains missing measurements as unavailable',()=>{
  const result=publicRound4(input());assert.equal(result.conditions[0].selectedTargetHits,0);assert.equal(result.conditions[0].negativeAlertCount,null);assert.equal(result.frames[0].results[0].selectedTargetHit,false);assert.equal(result.summary.referenceAuthority,'ai-incomplete');
});
test('round4 references require explicit incomplete AI authority and unchanged weights',()=>{
  for(const patch of [{referenceAuthority:'human-confirmed'},{weightsChanged:true}]){const source=input();Object.assign(source.summary,patch);assert.throws(()=>publicRound4(source));}
});
test('round4 rejects mismatched or unbounded image references',()=>{
  for(const patch of [{frameId:'../../private'},{imageSha256:'bad'},{width:Infinity},{height:300000},{selectedTargetPixelsXYXY:[-1,0,10,20]},{selectedTargetPixelsXYXY:[0,0,110,80]}]){const source=input();Object.assign(source.frames[0],patch);assert.throws(()=>publicRound4(source));}
});
test('round4 rejects duplicate frames, conditions and inconsistent counts',()=>{
  const frames=input();frames.frames.push(frames.frames[0]);assert.throws(()=>publicRound4(frames));
  const conditions=input();conditions.conditions.push(conditions.conditions[0]);assert.throws(()=>publicRound4(conditions));
  const counts=input();counts.conditions[0].selectedTargetHits=2;assert.throws(()=>publicRound4(counts));
});
test('round4 output discards private ledger paths and unsafe source links',()=>{
  const source=input();source.privateLedgerPath='private/ledger.json';source.frames[0].privateOriginal='C:/secret/photo.jpg';source.frames[0].source.sourceUrl='file:///C:/secret/photo.jpg';source.frames[0].source.licenseUrl='https://user:secret@example.org/license';
  const result=publicRound4(source);assert.equal(JSON.stringify(result).includes('secret'),false);assert.equal(result.frames[0].source.sourceUrl,null);assert.equal(result.frames[0].source.licenseUrl,null);assert.equal('privateLedgerPath' in result,false);
});
test('round4 copies region coordinates and only includes declared conditions',()=>{
  const source=input();source.frames[0].results.push({conditionId:'unannounced',alertCount:10});const result=publicRound4(source);result.frames[0].selectedTargetPixelsXYXY[0]=50;assert.equal(source.frames[0].selectedTargetPixelsXYXY[0],1);assert.equal(result.frames[0].results.length,1);
});
