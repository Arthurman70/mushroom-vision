import test from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { FindsClassifier } from '../core/finds-worker.mjs';

function workerFixture(onRequest, timeoutMs = 1000) {
  let child, spawnArguments;
  const broker = new FindsClassifier('C:/fixture', { timeoutMs, spawnWorker: (...args) => {
    spawnArguments = args; child = new EventEmitter(); child.stdout = new PassThrough(); child.stderr = new PassThrough(); child.stdin = new PassThrough();
    child.killed = false; child.kill = () => { child.killed = true; };
    child.stdin.on('data', bytes => onRequest(JSON.parse(bytes.toString()), child)); return child;
  } });
  return { broker, child: () => child, arguments: () => spawnArguments };
}
const photo = { bytes: Buffer.from('image fixture'), mimeType: 'image/png' };

test('classifier starts only the fixed local worker and matches NDJSON response IDs', async () => {
  const fixture = workerFixture((request, child) => setImmediate(() => child.stdout.write(JSON.stringify({ id: request.id, status: 'ok', result: { marker: true } }) + '\n')));
  const result = await fixture.broker.classify(photo, { findId: 'find-1', capturedAt: null });
  assert.equal(result.marker, true); assert.match(fixture.arguments()[0], /python(?:\.exe)?$/);
  assert.equal(fixture.broker.loaded, true);
  assert.deepEqual(fixture.arguments()[1], ['-m', 'vision.multiclass', 'worker', '--checkpoint', 'vision/checkpoints/groups-v1.json']);
  assert.equal(fixture.arguments()[2].windowsHide, true); assert.equal(fixture.arguments()[2].env.HF_HUB_OFFLINE, '1');
  fixture.broker.stop(); assert.equal(fixture.child().killed, true);
});

test('classifier permits one in-flight image and kills stalled workers', async () => {
  const fixture = workerFixture(() => {}, 20), first = fixture.broker.classify(photo);
  await assert.rejects(fixture.broker.classify(photo), error => error.code === 'CLASSIFIER_BUSY');
  await assert.rejects(first, error => error.code === 'CLASSIFIER_TIMEOUT');
  assert.equal(fixture.child().killed, true); assert.equal(fixture.broker.pending, null);
  fixture.broker.stop();
});

test('malformed JSON and mismatched replies cannot satisfy another request', async () => {
  for (const output of ['not-json\n', JSON.stringify({ id: 'wrong', status: 'ok', result: {} }) + '\n']) {
    const fixture = workerFixture((request, child) => setImmediate(() => child.stdout.write(output)));
    await assert.rejects(fixture.broker.classify(photo), error => error.code === 'CLASSIFIER_PROTOCOL');
    assert.equal(fixture.child().killed, true); fixture.broker.stop();
  }
});

test('worker errors and oversized responses remain bounded', async () => {
  const failure = workerFixture((request, child) => setImmediate(() => child.stdout.write(JSON.stringify({ id: request.id, status: 'error', error: { message: 'private file path' } }) + '\n')));
  await assert.rejects(failure.broker.classify(photo), error => error.code === 'CLASSIFIER_FAILED' && !error.message.includes('private'));
  failure.broker.stop();
  const large = workerFixture((request, child) => setImmediate(() => child.stdout.write('x'.repeat(256 * 1024 + 1))));
  await assert.rejects(large.broker.classify(photo), error => error.code === 'CLASSIFIER_PROTOCOL'); large.broker.stop();
});
