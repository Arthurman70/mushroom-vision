import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { INPUT_SCHEMA, replayPhotoLeads, writePhotoReplay } from '../flight/replay-photo-leads.mjs';

const config = JSON.parse(await readFile(new URL('../config/x500-habitat-search.json', import.meta.url), 'utf8'));
function observation(extra = {}) {
  return { id: 'box-1', frameId: 'photo-1', kind: 'fungus', confidence: 0.18,
    capturedMs: null, archivalPhoto: true, source: 'saved detector inference',
    bbox: [12, 20, 40, 65], ...extra };
}
const archive = observations => ({ schema: INPUT_SCHEMA, simulationOnly: true, actuationEnabled: false, observations });

test('real-photo candidates enter the existing queue without map targets or commands', () => {
  const report = replayPhotoLeads(archive([observation(), observation({ id: 'habitat-1', kind: 'habitat', signals: { forest_context: 0.7 } })]), config);
  assert.equal(report.summary.acceptedObservationCount, 2);
  assert.equal(report.summary.fungusLeadCount, 1);
  assert.equal(report.summary.habitatLeadCount, 1);
  assert.deepEqual(report.mapTargets, []);
  assert.equal(report.summary.flightCommands, 0);
  for (const lead of report.screenLeads) {
    assert.equal(lead.eligible, false); assert.equal(lead.position, null);
    assert.equal(lead.originalCapturedMs, null); assert.equal(lead.receivedMs, 0);
    assert.equal(lead.evidenceMode, 'archival_inference'); assert.equal(lead.identifiedSpecies, null);
  }
  assert.deepEqual(report.screenLeads.find(lead => lead.kind === 'fungus').bbox, { x1: 12, y1: 20, x2: 40, y2: 65, units: 'pixel' });
});

test('photo ranking is retained as review context and cannot change detector score or eligibility', () => {
  const metadata = { photoRank: 0.99, predictedSpecies: 'black trumpet', eligible: true, confidence: 1 };
  const report = replayPhotoLeads(archive([observation({ metadata, ranking: { score: 0.999 }, eligible: true, priority: 1 })]), config);
  const lead = report.screenLeads[0];
  assert.deepEqual(lead.reviewMetadata.metadata, metadata);
  assert.equal(lead.reviewMetadata.ranking.score, 0.999);
  assert.equal(lead.confidence, 0.18); assert.equal(lead.priority, config.weights.visual * 0.18);
  assert.equal(lead.eligible, false); assert.equal(lead.reviewMetadataAffectsEligibility, false);
});

test('exporter xyxy boxes and canonical review metadata survive replay', () => {
  const report = replayPhotoLeads(archive([observation({
    bbox: { xyxy: [10, 20, 60, 90], imageWidth: 768, imageHeight: 1024 },
    reviewMetadata: { blackTrumpetPatternScore: 0.92, prioritizeHumanReview: true, flightGateEnabled: false },
  })]), config);
  const lead = report.screenLeads[0];
  assert.deepEqual(lead.bbox, { x1: 10, y1: 20, x2: 60, y2: 90, units: 'pixel' });
  assert.equal(lead.reviewMetadata.blackTrumpetPatternScore, 0.92);
  assert.deepEqual(lead.reviewMetadata.imageDimensions, { width: 768, height: 1024 });
  assert.equal(lead.eligible, false);
});

test('archived records cannot introduce capture time or metric geometry', () => {
  const observations = [observation({ capturedMs: 0 }), observation({ archivalPhoto: false }),
    observation({ metric: { position: { x: 1, y: 2, z: 3 } } }), observation({ capturePose: null })];
  const report = replayPhotoLeads(archive(observations), config);
  assert.equal(report.summary.acceptedObservationCount, 0);
  assert.equal(report.summary.rejectedObservationCount, 4);
  assert.equal(report.rejectionCounts.archive_requires_unknown_capture_time, 2);
  assert.equal(report.rejectionCounts.archive_cannot_supply_navigation_geometry, 2);
});

test('wrong archive schema and enabled actuation fail before ingestion', () => {
  for (const change of [{ schema: 'unknown' }, { actuationEnabled: true }, { simulationOnly: false }, { observations: null }]) {
    assert.throws(() => replayPhotoLeads({ ...archive([]), ...change }, config), /Expected/);
  }
});

test('duplicate boxes are reported rather than counted as new evidence', () => {
  const report = replayPhotoLeads(archive([observation(), observation()]), config);
  assert.equal(report.summary.screenLeadCount, 1);
  assert.equal(report.summary.acceptedObservationCount, 1);
  assert.equal(report.rejectionCounts.duplicate_frame_does_not_refresh_evidence, 1);
});

test('queue capacity and rejection detail retention remain bounded', () => {
  const bounded = structuredClone(config); bounded.queue.maxLeads = 1; bounded.limits.maxRejectedHistory = 2;
  const report = replayPhotoLeads(archive(Array.from({ length: 6 }, (_, index) => observation({ id: `box-${index}` }))), bounded);
  assert.equal(report.summary.screenLeadCount, 1); assert.equal(report.summary.rejectedObservationCount, 5);
  assert.equal(report.rejections.length, 2); assert.equal(report.summary.omittedRejectionDetails, 3);
  assert.equal(report.rejectionCounts.queue_capacity, 5);
});

test('invalid confidence cannot be replaced by a photo-ranking score', () => {
  const report = replayPhotoLeads(archive([observation({ confidence: null, ranking: { confidence: 0.9 } })]), config);
  assert.equal(report.summary.screenLeadCount, 0);
  assert.equal(report.rejectionCounts.malformed_observation, 1);
});

test('file replay creates its report and refuses to overwrite the original archive', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'mycoscout-photo-replay-'));
  try {
    const input = join(directory, 'input.json'), output = join(directory, 'reports', 'replay.json');
    await writeFile(input, JSON.stringify(archive([observation()])));
    await assert.rejects(writePhotoReplay(input, input), /differ/);
    const report = await writePhotoReplay(input, output);
    assert.deepEqual(JSON.parse(await readFile(output, 'utf8')), report);
    assert.equal(report.summary.actionEligibleCount, 0);
    assert.equal(JSON.parse(await readFile(input, 'utf8')).schema, INPUT_SCHEMA);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
