import test from 'node:test';
import assert from 'node:assert/strict';
import {investigationView,confirmedIdentification} from '../public/investigation-view.mjs';
test('a color-only claim cannot appear as an immediate identified mushroom',()=>{
  const view=investigationView({metadata:{cueType:'color_salience',investigation:{routingCategory:'immediate_investigation',rangeEvidence:{status:'synthetic',rangeM:10},photoStatus:'complete'}}});
  assert.equal(view.category,'habitat_followup');assert.match(view.rangeText,/unknown/);assert.match(view.photoText,/not been verified/);
});
test('absent and unknown investigation formats do not fabricate a flight request',()=>{
  assert.equal(investigationView({}),null);assert.equal(investigationView({metadata:{investigation:{routingCategory:'fly_now'}}}),null);
});
test('unresolved or rejected species text cannot title an unconfirmed cue as an identified mushroom',()=>{
  for(const reviewStatus of ['pending','unresolved','rejected'])assert.equal(confirmedIdentification({reviewStatus,latestReview:{species:'Morel'}}),null);
  assert.equal(confirmedIdentification({reviewStatus:'reviewed',latestReview:{species:'Morel'}}),'Morel');
});
test('simulated capture and incomplete native framing stay distinct from a physical close-up',()=>{
  const view=photoStatus=>investigationView({metadata:{investigation:{routingCategory:'habitat_followup',photoStatus}}});
  assert.match(view('simulated_frame_filling_photo').photoText,/No physical photo/);
  assert.match(view('simulated_safe_photo_manual_closeup_required').photoText,/still needs zoom/);
});
