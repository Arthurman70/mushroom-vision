import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve, sep } from 'node:path';
import { deflateSync } from 'node:zlib';
import { FindsStore, normalizePrediction } from '../core/finds.mjs';
import { validateImage } from '../core/finds-images.mjs';

const catalogFile = new URL('../config/michigan-mushrooms-v1.json', import.meta.url);
const catalog = JSON.parse(await readFile(catalogFile, 'utf8'));
function crc32(bytes) { let crc = -1; for (const byte of bytes) { crc ^= byte; for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1)); } return (crc ^ -1) >>> 0; }
export function tinyImage(color = 0, width = 1, height = 1) {
  const chunk = (type, data) => { const out = Buffer.alloc(data.length + 12); out.writeUInt32BE(data.length); out.write(type, 4); data.copy(out, 8); out.writeUInt32BE(crc32(out.subarray(4, -4)), out.length - 4); return out; };
  const header = Buffer.alloc(13); header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 2;
  const png = Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]), chunk('IHDR', header), chunk('IDAT', deflateSync(Buffer.from([0, color, 0, 0]))), chunk('IEND', Buffer.alloc(0))]);
  return { imageBase64: png.toString('base64'), mimeType: 'image/png' };
}
const permission = { trainingOptIn: true, license: 'Photographer permits local training', attestedBy: 'Fixture photographer' };
const review = extra => ({ action: 'confirm_group', groupId: 'chanterelles_trumpets', reviewerName: 'Human reviewer', ...extra });
const guess = () => ({ schema: 'mycoscout.group-guess.v1', evidenceVersion: 'fixture-only', abstained: false,
  guesses: [{ classId: 'black_trumpets', groupId: 'chanterelles_trumpets', modelScore: 0.9 }, { classId: 'chanterelles', groupId: 'chanterelles_trumpets', modelScore: 0.08 }], speciesConfirmed: true, salesApproval: true });

async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'mycoscout-finds-'));
  const store = await new FindsStore(directory, catalogFile).initialize();
  t.after(async () => { await store.pending; store.close(); assert.ok(resolve(directory).startsWith(resolve(tmpdir()) + sep)); await rm(directory, { recursive: true, force: true }); });
  return { directory, store };
}

test('valid images are checked by bytes, bounded by dimensions, and stored by SHA256', async t => {
  const { store, directory } = await fixture(t), input = tinyImage();
  assert.equal(validateImage(input).width, 1);
  const a = await store.create(input), b = await store.create(input);
  assert.equal(a.contentHash, b.contentHash); assert.notEqual(a.id, b.id);
  assert.equal(store.db.prepare('SELECT count(*) AS n FROM photos').get().n, 1);
  assert.deepEqual((await store.image(a.id)).bytes, Buffer.from(input.imageBase64, 'base64'));
  assert.equal(a.reviewStatus, 'pending'); assert.equal(a.prediction, null); assert.equal(a.capturedAt, null);
  assert.equal((await readFile(join(directory, 'photos', a.contentHash + '.png'))).length, a.image.bytes);
  assert.throws(() => validateImage({ ...input, mimeType: 'image/jpeg' }), /does not match/);
  assert.throws(() => validateImage(tinyImage(0, 10000, 10000)), /20 megapixels/);
  assert.throws(() => validateImage({ imageBase64: 'PHN2Zz48L3N2Zz4=' }), /Only JPEG/);
  assert.throws(() => validateImage({ imageBase64: input.imageBase64 + '\n' }), /base64/);
});

test('research taxa persist as tentative evidence through find review and export',async t=>{
  const {store}=await fixture(t),find=await store.create({...tinyImage(38),rights:permission});
  const taxonCandidate={taxonId:'amanita',confidence:.5,evidenceSource:'field_notes',evidenceNote:'Gills and stem base photographed; species unresolved.',speciesVerified:true};
  const saved=await store.review(find.id,review({groupId:'gilled',taxonCandidate}));
  assert.equal(saved.find.latestReview.taxonCandidate.status,'tentative');assert.equal(saved.find.latestReview.taxonCandidate.speciesVerified,false);
  const exported=await store.exportTraining();
  assert.equal(exported.records[0].labels.taxonCandidate.taxonId,'amanita');
  assert.equal(exported.records[0].labels.classId,null);assert.equal(exported.records[0].labels.speciesProfessionallyVerified,false);
  await assert.rejects(store.review(find.id,review({groupId:'morels',taxonCandidate})),/match the broad group/);
});

test('predictions retain internal classes but cannot confirm species or create human review', async t => {
  const { store } = await fixture(t), find = await store.create(tinyImage());
  const saved = await store.prediction(find.id, guess());
  assert.equal(saved.find.reviewStatus, 'pending'); assert.equal(saved.find.latestReview, null);
  assert.equal(saved.result.guesses.length, 2); assert.equal(saved.result.guesses[0].classId, 'black_trumpets');
  assert.equal(saved.result.speciesConfirmed, false); assert.equal(saved.result.salesApproval, false);
  assert.equal(saved.result.flightGateEnabled, false);
  assert.equal((await store.exportTraining()).records.length, 0);
  assert.throws(() => normalizePrediction({ ...guess(), guesses: [{ groupId: 'morels', classId: 'black_trumpets', modelScore: 0.8 }] }, catalog), /catalog/);
});

test('reviews append without changing earlier decisions and survive restart', async t => {
  const { store, directory } = await fixture(t), find = await store.create({ ...tinyImage(), rights: permission });
  const first = await store.review(find.id, review({ classId: 'black_trumpets', credentialReference: 'Self-reported credential', specimen: { underside: 'Smooth wrinkles', size: { capMm: 20 } } }));
  assert.equal(first.find.humanReviewed, true); assert.equal(first.find.professionallyVerified, false);
  assert.equal(first.review.credentialVerified, false);
  const release = await store.exportTraining(); assert.equal(release.records.length, 1);
  assert.equal(release.records[0].labels.classId, 'black_trumpets');
  assert.equal(release.records[0].labels.labelSource, 'human_review');
  const revised = await store.review(find.id, review({ action: 'unresolved', notes: 'Need spore print' }));
  assert.equal(revised.find.reviewStatus, 'unresolved');
  assert.equal(store.events().events.filter(event => event.type === 'review').length, 2);
  assert.equal((await store.exportTraining()).records.length, 0);
  assert.equal(JSON.parse(await readFile(join(directory, 'exports', release.releaseId + '.json'), 'utf8')).records.length, 1);
  assert.throws(() => store.db.prepare("UPDATE events SET type='changed'").run(), /append-only/);
  assert.throws(() => store.db.prepare('DELETE FROM events').run(), /append-only/);
  store.close(); await store.initialize();
  assert.equal(store.get(find.id).reviewStatus, 'unresolved'); assert.equal(store.get(find.id).latestReview.notes, 'Need spore print');
});

test('rights opt-in is required independently from review and can be revoked', async t => {
  const { store } = await fixture(t), find = await store.create(tinyImage());
  await store.review(find.id, review()); assert.equal((await store.exportTraining()).records.length, 0);
  await assert.rejects(store.review(find.id, review({ trainingOptIn: true })), /rights.license/);
  await store.review(find.id, review({ rights: permission }));
  const release = await store.exportTraining(); assert.equal(release.records.length, 1);
  assert.equal((await store.exportTraining()).releaseId, release.releaseId);
  await store.review(find.id, review({ trainingOptIn: false }));
  assert.equal((await store.exportTraining()).records.length, 0);
  await assert.rejects(store.create({ ...tinyImage(), rights: { trainingOptIn: true } }), /requires/);
});

test('unknown groups, missing names and incompatible fine classes cannot become training labels', async t => {
  const { store } = await fixture(t), find = await store.create({ ...tinyImage(), rights: permission });
  await assert.rejects(store.review(find.id, review({ groupId: 'edible_everything' })), /catalog/);
  await assert.rejects(store.review(find.id, review({ reviewerName: '' })), /reviewerName/);
  await assert.rejects(store.review(find.id, review({ action: 'correct_species' })), /species name/);
  await assert.rejects(store.review(find.id, review({ classId: 'true_morels' })), /match/);
  await store.review(find.id, review({ groupId: 'unknown' }));
  assert.equal((await store.exportTraining()).records.length, 0);
});

test('session groups and exact-photo duplicates keep a common held-out partition', async t => {
  const { store } = await fixture(t);
  const a = await store.create({ ...tinyImage(1), sessionId: 'held-out-session', evaluationOnly: true, rights: permission });
  const b = await store.create({ ...tinyImage(2), sessionId: 'held-out-session', rights: permission });
  const c = await store.create({ ...tinyImage(2), sessionId: 'another-session', rights: permission });
  for (const find of [a, b, c]) await store.review(find.id, review());
  const release = await store.exportTraining();
  assert.equal(release.records.length, 3); assert.ok(release.records.every(row => row.partition === 'test'));
  assert.equal(new Set(release.records.map(row => row.sourceGroupId)).size, 1);
});

test('a new connection across previously different splits quarantines the entire component', async t => {
  const { store } = await fixture(t);
  const trainingPhoto = await store.create({ ...tinyImage(3), rights: permission });
  // Force a deterministic fixture of an already exported train assignment.
  store.db.prepare("UPDATE partitions SET partition='train'").run();
  const testPhoto = await store.create({ ...tinyImage(4), sessionId: 'test-session', evaluationOnly: true, rights: permission });
  await store.review(trainingPhoto.id, review()); await store.review(testPhoto.id, review());
  assert.equal((await store.exportTraining()).records.length, 2);
  const connection = await store.create({ ...tinyImage(3), sessionId: 'test-session', rights: permission });
  await store.review(connection.id, review());
  const result = await store.exportTraining(); assert.equal(result.records.length, 0); assert.equal(result.summary.excluded.partitionConflict, 3);
});

test('event pagination advances without losing repeated changes to the same find', async t => {
  const { store } = await fixture(t), a = await store.create(tinyImage()), b = await store.create(tinyImage(2));
  await store.review(a.id, review()); await store.prediction(b.id, guess());
  const first = store.list(0, 2), second = store.list(first.cursor, 2);
  assert.equal(first.hasMore, true); assert.equal(second.hasMore, false);
  assert.equal(first.finds.length, 2); assert.equal(second.finds.length, 2);
  assert.equal(second.cursor, 4); assert.equal(store.list(second.cursor).finds.length, 0);
  assert.throws(() => store.events(-1), /since/); assert.throws(() => store.events(0, 201), /limit/);
});

test('stored image tampering and arbitrary find paths are rejected', async t => {
  const { store, directory } = await fixture(t), find = await store.create(tinyImage());
  assert.throws(() => store.get('../../config'), /not found/);
  await writeFile(join(directory, 'photos', find.contentHash + '.png'), Buffer.from('tampered'));
  await assert.rejects(store.image(find.id), /integrity/);
});

test('specimen IDs supplied during review link phone/drone views and persist through later omissions', async t => {
  const { store } = await fixture(t);
  const a = await store.create({ ...tinyImage(7), source: 'drone_camera', rights: permission, evaluationOnly: true });
  const b = await store.create({ ...tinyImage(8), source: 'phone_upload', rights: permission, evaluationOnly: true });
  await store.review(a.id, review({ specimen: { specimenId: 'field-specimen-1', viewType: 'top', features: ['funnel'] } }));
  await store.review(b.id, review({ specimen: { specimenId: 'field-specimen-1', viewType: 'underside' } }));
  await store.review(a.id, review({ notes: 'Reviewed again without replacing character fields' }));
  const release = await store.exportTraining();
  assert.equal(release.records.length, 2); assert.equal(new Set(release.records.map(row => row.sourceGroupId)).size, 1);
  assert.ok(release.records.every(row => row.partition === 'test'));
  assert.equal(store.get(a.id).specimen.viewType, 'top'); assert.deepEqual(store.get(a.id).specimen.features, ['funnel']);
});

test('contradictory human labels for the same image are excluded from a release', async t => {
  const { store } = await fixture(t);
  const a = await store.create({ ...tinyImage(9), rights: permission }), b = await store.create({ ...tinyImage(9), rights: permission });
  await store.review(a.id, review({ classId: 'black_trumpets' }));
  await store.review(b.id, review({ groupId: 'morels', classId: 'true_morels' }));
  const result = await store.exportTraining(); assert.equal(result.records.length, 0); assert.equal(result.summary.excluded.conflictingLabels, 2);
});

test('taxon-level research guesses keep form uncertainty without trusting arbitrary model labels', () => {
  const result = normalizePrediction({ ...guess(), guesses: [
    { classId: 'aborted_entoloma', groupId: 'crust_parasitized', label: 'Guaranteed aborted form', modelScore: 0.7 },
    { classId: 'chaga', groupId: 'polypores', label: 'Confirmed harvestable conk', modelScore: 0.2 },
  ] }, catalog);
  for (const predicted of result.guesses) {
    assert.match(predicted.label, /\(form unverified\)/);
    assert.doesNotMatch(predicted.label, /Guaranteed|Confirmed harvestable/);
    assert.equal(predicted.scopeNotes.length, 1); assert.match(predicted.scopeNotes[0], /not confirmed/);
  }
  assert.match(result.guesses[0].scopeNotes[0], /normal gilled/);
  assert.equal(result.speciesConfirmed, false);
});

test('training release preserves bounded photo source attribution as provenance, not labels', async t => {
  const { store } = await fixture(t);
  const metadata = { photoLicense: 'CC-BY-4.0', attribution: 'Fixture photographer', sourceUrl: 'https://example.org/fixture-photo', sourceObservationId: 'public-fixture-1', guessedClass: 'true_morels' };
  const find = await store.create({ ...tinyImage(12), metadata, rights: permission });
  await store.review(find.id, review({ classId: 'black_trumpets' }));
  const release = await store.exportTraining(), item = release.records[0];
  assert.deepEqual(item.provenance.sourceMetadata, metadata);
  assert.equal(item.labels.classId, 'black_trumpets'); assert.equal(item.labels.guessedClass, undefined);
  assert.equal((await store.exportTraining()).releaseId, release.releaseId);
});

test('stale device review cannot undo a newer rights revocation or corrected label', async t => {
  const { store } = await fixture(t), find = await store.create({ ...tinyImage(13), rights: permission });
  const initial = await store.review(find.id, review({ expectedReviewId: null, classId: 'black_trumpets' }));
  const revoked = await store.review(find.id, review({ expectedReviewId: initial.review.id, groupId: 'morels', classId: 'true_morels', trainingOptIn: false }));
  const count = store.events().events.length;
  await assert.rejects(store.review(find.id, review({ expectedReviewId: initial.review.id, classId: 'black_trumpets', rights: permission })),
    error => error.status === 409 && error.code === 'REVIEW_CONFLICT');
  assert.equal(store.get(find.id).rights.trainingOptIn, false);
  assert.equal(store.get(find.id).latestReview.id, revoked.review.id);
  assert.equal(store.get(find.id).latestReview.classId, 'true_morels');
  assert.equal(store.events().events.length, count);
  assert.equal((await store.exportTraining()).records.length, 0);
  await assert.rejects(store.review(find.id, review({ expectedReviewId: null })), error => error.code === 'REVIEW_CONFLICT');
});

test('expected review IDs are validated and concurrent saves serialize their conflict check', async t => {
  const { store } = await fixture(t), find = await store.create(tinyImage(14));
  for (const expectedReviewId of ['', 'not-a-uuid', 123, {}, undefined]) {
    await assert.rejects(store.review(find.id, review({ expectedReviewId })), error => error.status === 400);
  }
  const initial = await store.review(find.id, review({ expectedReviewId: null }));
  const concurrent = await Promise.allSettled([
    store.review(find.id, review({ expectedReviewId: initial.review.id.toUpperCase(), notes: 'first device' })),
    store.review(find.id, review({ expectedReviewId: initial.review.id, notes: 'second device' })),
  ]);
  assert.equal(concurrent.filter(result => result.status === 'fulfilled').length, 1);
  const rejected = concurrent.find(result => result.status === 'rejected');
  assert.equal(rejected.reason.code, 'REVIEW_CONFLICT');
  assert.equal(store.events().events.filter(event => event.type === 'review').length, 2);
  await store.review(find.id, review({ notes: 'legacy client compatibility' }));
});

test('prediction review policy retains validated thresholds in the stored prediction', async t => {
  const { store } = await fixture(t), find = await store.create(tinyImage(15));
  const reviewPolicy = { scoreThreshold: 0.5, marginThreshold: 0.1, targetValidationAccuracy: 0.8, validationTargetMet: false };
  const result = await store.prediction(find.id, { ...guess(), reviewPolicy: { ...reviewPolicy, arbitraryApproval: true } });
  assert.deepEqual(result.result.reviewPolicy, reviewPolicy);
  assert.deepEqual(store.get(find.id).prediction.reviewPolicy, reviewPolicy);
  assert.equal(result.find.reviewStatus, 'pending'); assert.equal(result.result.salesApproval, false);
  assert.equal(normalizePrediction(guess(), catalog).reviewPolicy, null);
  assert.equal(normalizePrediction({ ...guess(), reviewPolicy: { ...reviewPolicy, scoreThreshold: 1.01, marginThreshold: 1.01 } }, catalog).reviewPolicy.scoreThreshold, 1.01);
  for (const invalid of [
    { scoreThreshold: -0.1 }, { scoreThreshold: 1.02 }, { scoreThreshold: '0.5' },
    { marginThreshold: NaN }, { marginThreshold: 1.02 }, { marginThreshold: -1 },
    { targetValidationAccuracy: Infinity }, { targetValidationAccuracy: 1.01 }, { targetValidationAccuracy: -0.1 },
    { validationTargetMet: 'true' },
  ]) {
    assert.throws(() => normalizePrediction({ ...guess(), reviewPolicy: { ...reviewPolicy, ...invalid } }, catalog), error => error.status === 502 && error.code === 'CLASSIFIER_PROTOCOL');
  }
});
