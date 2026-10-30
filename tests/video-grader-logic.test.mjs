import test from 'node:test';
import assert from 'node:assert/strict';
import { trackChoices, nextBoxTrack, reviewError, feedAfterReview } from '../public/video-grader-logic.mjs';

const tracks = [{ id: 'morel-1', label: 'Morel by log' }, { id: 'morel-2', label: 'Morel by fern' }];

test('one physical mushroom cannot be offered for another box but remains available on its own box', () => {
  const boxes = [{ id: 'box-1', trackId: 'morel-1' }, { id: 'box-2', trackId: null }];
  assert.deepEqual(trackChoices(tracks, boxes).map(track => track.disabled), [true, false]);
  assert.deepEqual(trackChoices(tracks, boxes, 'box-1').map(track => track.disabled), [false, false]);
  assert.deepEqual(trackChoices(tracks, boxes, 'box-2').map(track => track.disabled), [true, false]);
});

test('drawing another mushroom with a stale selected track preserves the box but leaves it unlinked', () => {
  const boxes = [{ id: 'first', trackId: nextBoxTrack('morel-1', []) }];
  const second = { id: 'second', trackId: nextBoxTrack('morel-1', boxes) };
  boxes.push(second);
  assert.equal(second.trackId, null);
  assert.equal(boxes[0].trackId, 'morel-1');
  assert.equal(reviewError('visible', boxes, true), '');
  assert.equal(nextBoxTrack('morel-2', boxes), 'morel-2');
});

test('existing duplicate drafts produce an actionable save error without modifying boxes or labels', () => {
  const boxes = [{ id: 'first', trackId: 'morel-1', groupId: 'unknown' }, { id: 'second', trackId: 'morel-1', groupId: 'unknown' }];
  const before = structuredClone(boxes);
  assert.match(reviewError('visible', boxes, true), /set the other to Unlinked/);
  assert.deepEqual(boxes, before);
  boxes[1].trackId = null;
  assert.equal(reviewError('visible', boxes, true), '');
});

test('removing a box makes its individual track available again', () => {
  const boxes = [{ id: 'first', trackId: 'morel-1' }];
  assert.equal(trackChoices(tracks, boxes)[0].disabled, true);
  boxes.pop();
  assert.equal(trackChoices(tracks, boxes)[0].disabled, false);
});

test('complete visible reviews need boxes, while quick visibility and negative reviews remain possible', () => {
  assert.match(reviewError('visible', [], true), /Draw a box/);
  assert.equal(reviewError('visible', [], false), '');
  assert.equal(reviewError('absent', [], true), '');
});

test('auto-advance respects source/status filtering when the just-reviewed frame drops out', () => {
  const old = [{ id: 'a' }, { id: 'b' }, { id: 'c' }];
  const reviews = { a: { review: { status: 'visible' } }, b: { review: { status: 'visible' } } };
  assert.equal(feedAfterReview(old, 'b', [old[2]], reviews).id, 'c');
  assert.equal(feedAfterReview(old, 'c', [], reviews), null);
  assert.equal(feedAfterReview([old[0]], 'a', [old[0]], reviews).id, 'a');
});
