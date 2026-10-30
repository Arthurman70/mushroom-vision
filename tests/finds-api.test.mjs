import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { createApp } from '../server.mjs';

const imageBase64 = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2ZQkAAAAASUVORK5CYII=';
async function fixture(t, token = '') {
  const directory = await mkdtemp(join(tmpdir(), 'mycoscout-finds-api-'));
  const classifier = { requests: [], classify: async function(image, context) {
    this.requests.push({ bytes: image.bytes.length, ...context });
    return { schema: 'mycoscout.group-guess.v1', evidenceVersion: 'fixture', abstained: true, abstentionReasons: ['fixture'],
      guesses: [{ classId: 'black_trumpets', groupId: 'chanterelles_trumpets', modelScore: 0.6 }], salesApproval: true };
  } };
  const app = await createApp({ host: '127.0.0.1', token, stateFile: join(directory, 'state.json'), classifier });
  const address = await app.listen(0), base = 'http://127.0.0.1:' + address.port;
  t.after(async () => { app.server.closeAllConnections(); await new Promise(done => app.server.close(done)); await app.closed; await app.finds.pending;
    assert.ok(resolve(directory).startsWith(resolve(tmpdir()) + sep)); await rm(directory, { recursive: true, force: true }); });
  const request = async (path, body, headers = {}) => {
    const response = await fetch(base + path, { method: body === undefined ? 'GET' : 'POST', headers: { ...body !== undefined && { 'Content-Type': 'application/json' }, ...headers }, body: body === undefined ? undefined : JSON.stringify(body) });
    return { status: response.status, data: await response.json() };
  };
  return { app, base, request, classifier };
}

test('phone upload, prediction, headset polling and human review produce separate records', async t => {
  const { request, base, classifier } = await fixture(t);
  const created = await request('/api/finds', { imageBase64, source: 'photo_upload', sessionId: 'test-session',
    metadata: { actualLiveHardware: false }, rights: { trainingOptIn: true, license: 'Fixture consent', attestedBy: 'Photographer' } });
  assert.equal(created.status, 201); const id = created.data.find.id;
  const photo = await fetch(base + `/api/finds/${id}/image`);
  assert.equal(photo.headers.get('content-type'), 'image/png'); assert.deepEqual(Buffer.from(await photo.arrayBuffer()), Buffer.from(imageBase64, 'base64'));
  const classified = await request('/api/vision/classify', { findId: id });
  assert.equal(classified.status, 200); assert.equal(classified.data.find.reviewStatus, 'pending');
  assert.equal(classified.data.result.guesses[0].classId, 'black_trumpets');
  assert.equal(classifier.requests[0].findId, id); assert.equal(classified.data.result.salesApproval, false);
  const feed = await request('/api/finds?since=0'); assert.equal(feed.data.finds[0].prediction.guesses.length, 1);
  const events = await request('/api/events?since=0'); assert.equal(events.data.events[1].type, 'prediction');
  assert.equal((await request('/api/training/export')).data.records.length, 0);
  const reviewed = await request(`/api/finds/${id}/reviews`, { action: 'confirm_group', groupId: 'chanterelles_trumpets',
    classId: 'black_trumpets', reviewerName: 'Human fixture', credentialReference: 'Self-reported', specimen: { cap: 'Funnel' } });
  assert.equal(reviewed.status, 201); assert.equal(reviewed.data.find.reviewStatus, 'reviewed');
  assert.equal(reviewed.data.find.professionallyVerified, false); assert.equal(reviewed.data.find.salesApproval, false);
  assert.equal((await request('/api/training/export')).data.records[0].labels.classId, 'black_trumpets');
  assert.equal((await request(`/api/finds/${id}`)).data.find.latestReview.reviewerName, 'Human fixture');
  assert.equal((await request('/api/catalog')).data.groups.length, 10);
  assert.equal((await request('/api/vision/status')).data.humanVerificationRequired, true);
});

test('Bearer authentication and allowed origins also protect private images and model requests', async t => {
  const token = 'fixture-authentication-token-123456', { request, base } = await fixture(t, token);
  const headers = { Authorization: 'Bearer ' + token };
  assert.equal((await request('/api/finds', { imageBase64 })).status, 401);
  assert.equal((await request('/api/catalog')).status, 401);
  assert.equal((await request('/api/training/export')).status, 401);
  const created = await request('/api/finds', { imageBase64 }, headers);
  assert.equal((await fetch(base + created.data.find.imageUrl)).status, 401);
  assert.equal((await fetch(base + created.data.find.imageUrl, { headers })).status, 200);
  assert.equal((await request('/api/events', undefined, { ...headers, Origin: 'https://attacker.example' })).status, 403);
  assert.equal((await request('/api/vision/classify', { imageBase64 }, { ...headers, Origin: 'https://appassets.androidplatform.net' })).status, 200);
});

test('invalid uploads, labels and cursor parameters return bounded client errors', async t => {
  const { request } = await fixture(t);
  assert.equal((await request('/api/finds', { imageBase64: 'not-an-image' })).status, 400);
  assert.equal((await request('/api/finds', { imageBase64, mimeType: 'image/jpeg' })).status, 415);
  assert.equal((await request('/api/finds?since=-1')).status, 400);
  assert.equal((await request('/api/events?limit=0')).status, 400);
  assert.equal((await request('/api/vision/classify', { findId: '../../config' })).status, 404);
  const created = await request('/api/finds', { imageBase64 });
  assert.equal((await request(`/api/finds/${created.data.find.id}/reviews`, { action: 'confirm_group', groupId: 'morels' })).status, 400);
  assert.equal((await request('/api/arm', {})).status, 404);
});

test('review API returns a conflict when the form is based on an older review', async t => {
  const { request } = await fixture(t);
  const created = await request('/api/finds', { imageBase64 });
  const path = `/api/finds/${created.data.find.id}/reviews`;
  const input = { action: 'confirm_group', groupId: 'morels', reviewerName: 'Human fixture', expectedReviewId: null };
  const first = await request(path, input); assert.equal(first.status, 201);
  const stale = await request(path, input); assert.equal(stale.status, 409); assert.equal(stale.data.code, 'REVIEW_CONFLICT');
  const fresh = await request(path, { ...input, expectedReviewId: first.data.review.id }); assert.equal(fresh.status, 201);
  assert.equal((await request(path, { ...input, expectedReviewId: false })).status, 400);
});

test('a saved color cue cannot be promoted into a mushroom guess through the API', async t => {
  const { request, classifier } = await fixture(t);
  const created = await request('/api/finds', { imageBase64, metadata: { cueType: 'color_salience', classificationEligible: true,
    investigation: { routingCategory: 'habitat_followup' } } });
  const classified = await request('/api/vision/classify', { findId: created.data.find.id });
  assert.equal(classified.status, 409); assert.equal(classified.data.code, 'CLOSER_VIEW_REQUIRED');
  assert.equal(classifier.requests.length, 0);
  const stored = (await request(`/api/finds/${created.data.find.id}`)).data.find;
  assert.equal(stored.reviewStatus, 'pending'); assert.equal(stored.prediction, null);
  assert.equal((await request('/api/training/export')).data.records.length, 0);
});
