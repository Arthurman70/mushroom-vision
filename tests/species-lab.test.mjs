import test from 'node:test';
import assert from 'node:assert/strict';
import { visibleResearchGuesses,filteredResearchTaxa } from '../public/species-lab.js';
test('research shortlist ignores invalid model scores and keeps only three ranked guesses',()=>{
  const result=visibleResearchGuesses({guesses:[{label:'Bad',modelScore:2},{label:'NaN',modelScore:NaN},{modelScore:.99},
    ...[.2,.8,.3,.6].map(score=>({label:String(score),modelScore:score}))]});
  assert.deepEqual(result.map(guess=>guess.modelScore),[.8,.6,.3]);assert.deepEqual(visibleResearchGuesses(null),[]);
});
test('catalog browsing separates false morel research group and supports source-name search',()=>{
  assert.ok(filteredResearchTaxa('Morchella').length>=9);
  assert.ok(filteredResearchTaxa('', 'ascomycetes').some(t=>t.id==='gyromitra_esculenta'));
  assert.equal(filteredResearchTaxa('Gyromitra','morels').length,0);
  assert.equal(filteredResearchTaxa('amanita')[0].rank,'genus');assert.deepEqual(filteredResearchTaxa('no_such_taxon'),[]);
});
