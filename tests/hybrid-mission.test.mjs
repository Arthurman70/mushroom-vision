import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, join, resolve, sep } from 'node:path';
import { LiveMissionStore } from '../core/live-mission.mjs';
import { createSimulationPacket } from '../flight/live-executor.mjs';

// These are synthetic sensor fixtures exercising the real live-session rules;
// simulationOnly:false selects those rules, not a claim of physical capture.
async function fixture(t) {
  const directory = await mkdtemp(join(tmpdir(), 'mycoscout-hybrid-mission-'));
  let clock = 1800000000000;
  const store = await new LiveMissionStore(directory, { now: () => clock }).initialize();
  t.after(async () => {
    store.close();
    assert.ok(resolve(directory).startsWith(resolve(tmpdir()) + sep));
    assert.ok(basename(directory).startsWith('mycoscout-hybrid-mission-'));
    await rm(directory, { recursive: true, force: true });
  });
  store.startSession({ sessionId: 'hybrid-test', mode: 'live' });
  const packet = ({ ageSeconds = 0, sequence = 0, positionEnuM = [-4, 0, 1.5] } = {}) => {
    const value = createSimulationPacket({ sessionId: 'hybrid-test', sequence,
      timeSeconds: clock / 1000 - ageSeconds, positionEnuM });
    value.simulationOnly = false;
    return value;
  };
  return { store, packet, now: () => clock, advance: ms => { clock += ms; } };
}

function navigation(packet) {
  const data = structuredClone(packet), telemetry = data.flight.telemetry, voxelMap = data.flight.voxelMap;
  telemetry.cameraPositionEnuM = [...telemetry.positionEnuM];
  telemetry.cameraMountMode = 'rigid_calibrated';
  voxelMap.evidence = { schema: 'mycoscout.observed-depth-volume.v1', sessionId: data.sessionId,
    freeSpaceMethod: 'full_projected_voxel_footprint', calibrationId: data.camera.calibrationId,
    depthTimestampSeconds: data.timestampSeconds, poseTimestampSeconds: data.timestampSeconds,
    positionStdM: .02, orientationStdRad: .01, depthStdM: .02 };
  return { schema: 'mycoscout.live-navigation.v1', sessionId: data.sessionId,
    mapRevision: data.mapRevision, timestampSeconds: data.timestampSeconds,
    simulationOnly: false, telemetry, voxelMap };
}

test('30-second tile result retains review evidence without replacing fresh navigation or dispatching stale inspection', async t => {
  const { store, packet, now } = await fixture(t);
  const current = navigation(packet({ positionEnuM: [-3, 0, 1.5] }));
  assert.equal(store.updateNavigation(current).action.reason, 'awaiting_semantic_observation');
  const tiled = packet({ ageSeconds: 30 });
  tiled.detectorStatus = { status: 'completed', processingSeconds: 30, modelPasses: 5 };
  store.ingestFrame(tiled);
  const visible = store.snapshot();
  assert.equal(visible.landmarks.length, 1);
  assert.equal(visible.tasks.length, 1);
  assert.equal(visible.landmarks[0].lastSeenMs, now() - 30000);
  assert.equal(visible.latestFrame.timestampSeconds, tiled.timestampSeconds);
  assert.equal(visible.latestFrame.detectorStatus.processingSeconds, 30);
  assert.equal(visible.connection.captureAgeMs, 30000);
  assert.deepEqual(visible.telemetry, current.telemetry);
  assert.deepEqual(visible.voxelMap, current.voxelMap);
  assert.equal(visible.lastNavigationSeconds, current.timestampSeconds);
  const result = store.tick();
  assert.equal(result.action.type, 'hold');
  assert.equal(result.action.reason, 'target_observation_stale');
  assert.equal(result.action.actuationEnabled, false);
  assert.equal(store.snapshot().tasks[0].status, 'blocked');
});

test('fresh navigation alone cannot rehabilitate tile evidence; fresh target capture and explicit retry allow only a proposal', async t => {
  const { store, packet, advance } = await fixture(t);
  const old = packet({ ageSeconds: 30 });
  delete old.flight;
  store.ingestFrame(old);
  assert.equal(store.updateNavigation(navigation(packet())).action.reason, 'target_observation_stale');
  const taskId = store.snapshot().tasks[0].id;
  advance(100);
  store.updateNavigation(navigation(packet({ sequence: 1 })));
  assert.equal(store.snapshot().tasks[0].status, 'blocked');
  assert.equal(store.snapshot().landmarks[0].lastSeenMs, old.timestampSeconds * 1000);

  const reacquired = packet({ sequence: 1 });
  delete reacquired.flight;
  store.ingestFrame(reacquired);
  assert.equal(store.snapshot().tasks[0].status, 'blocked');
  assert.equal(store.snapshot().landmarks[0].lastSeenMs, reacquired.timestampSeconds * 1000);
  store.updateTask(taskId, { action: 'retry' });
  const result = store.tick();
  assert.equal(result.action.type, 'move_proposal', result.action.reason);
  assert.equal(result.action.actuationEnabled, false);
  assert.equal(result.state.capabilities.realFlight, false);
  assert.ok(result.action.waypointsEnuM.every(position => position[2] === 1.5));
});

test('late tile completion cannot be relabeled as a newer capture after a newer full-frame observation', async t => {
  const { store, packet } = await fixture(t);
  store.ingestFrame(packet({ sequence: 0 }));
  const before = store.snapshot();
  const late = packet({ ageSeconds: 30, sequence: 1 });
  late.detectorStatus = { status: 'completed', processingSeconds: 30, modelPasses: 5 };
  assert.throws(() => store.ingestFrame(late), { code: 'LIVE_OUT_OF_ORDER' });
  const after = store.snapshot();
  assert.equal(after.revision, before.revision);
  assert.deepEqual(after.landmarks, before.landmarks);
  assert.deepEqual(after.latestFrame, before.latestFrame);
});

test('semantic-only observations at the same or later capture time preserve independent navigation evidence', async t => {
  for (const delayMs of [0, 100]) await t.test(`semantic delay ${delayMs} ms`, async st => {
    const { store, packet, advance } = await fixture(st);
    const nav = navigation(packet());
    store.updateNavigation(nav);
    advance(delayMs);
    const semantic = packet();
    delete semantic.flight;
    store.ingestFrame(semantic);
    const state = store.snapshot();
    assert.deepEqual(state.telemetry, nav.telemetry);
    assert.deepEqual(state.voxelMap, nav.voxelMap);
    assert.equal(state.lastNavigationSeconds, nav.timestampSeconds);
    assert.equal(state.lastCaptureSeconds, semantic.timestampSeconds);
    assert.equal(store.tick().action.type, 'move_proposal');
  });
});

test('explicit flight evidence at the same or newer capture time still updates navigation', async t => {
  for (const delayMs of [0, 100]) await t.test(`embedded evidence delay ${delayMs} ms`, async st => {
    const { store, packet, advance } = await fixture(st);
    store.updateNavigation(navigation(packet({ positionEnuM: [-3, 0, 1.5] })));
    advance(delayMs);
    const semantic = packet({ positionEnuM: [-4, 0, 1.5] });
    store.ingestFrame(semantic);
    const state = store.snapshot();
    assert.deepEqual(state.telemetry, semantic.flight.telemetry);
    assert.deepEqual(state.voxelMap, semantic.flight.voxelMap);
    assert.equal(state.lastNavigationSeconds, semantic.timestampSeconds);
  });
});

test('hybrid frame routing uses measured range and uncertainty, never box size or detector confidence', async t => {
  for (const item of [
    { name: 'unknown', rangeM: null, expected: 'unlocated' },
    { name: 'near', rangeM: 20, expected: 'immediate' },
    { name: 'exact boundary', rangeM: 50, expected: 'habitat' },
    { name: 'far', rangeM: 100, expected: 'habitat' },
    { name: 'uncertainty crosses boundary', rangeM: 49.95, expected: 'habitat' },
    { name: 'explicit habitat remains habitat nearby', rangeM: 5, habitat: true, expected: 'habitat' },
  ]) await t.test(item.name, async st => {
    const { store, packet } = await fixture(st);
    const value = packet();
    value.detections[0].confidence = .99;
    value.detections[0].bboxPixelsXYXY = [1, 1, 639, 479];
    const landmark = value.landmarks[0];
    if (item.rangeM === null) {
      delete value.camera; delete value.depth; delete value.flight;
      value.mapPoints = [];
      Object.assign(landmark, { positionEnuM: null, positionStdM: null, rangeM: null, gps: null });
    } else {
      Object.assign(landmark, { positionEnuM: [-4 + item.rangeM, 0, 1.5], positionStdM: .02,
        rangeM: item.rangeM, inspectionPriority: item.habitat ? 'habitat' : 'immediate' });
    }
    store.ingestFrame(value);
    const state = store.snapshot();
    assert.equal(state.landmarks[0].inspectionPriority, item.expected);
    assert.equal(state.landmarks[0].humanVerified, false);
    assert.equal(state.tasks.length, item.expected === 'immediate' ? 1 : 0);
  });
});

test('live scan display retains bounded timing fields without model paths or authority claims', async t => {
  const { store, packet } = await fixture(t);
  const value = packet({ ageSeconds: 30 });
  value.detector = { modelPath: 'private-model-path', secret: 'private-value',
    scan: { profileId: 'hybrid', effectiveProfileId: 'ground-tiles', modelPasses: 6, seconds: 30,
      cadence: { exceedsRequestedInterval: true, modelPath: 'private-nested-path' },
      actuationEnabled: true, positionVerified: true, arbitrary: 'do-not-copy' } };
  store.ingestFrame(value);
  const frame = store.snapshot().latestFrame;
  assert.deepEqual(frame.detector, { scan: { profileId: 'hybrid', effectiveProfileId: 'ground-tiles',
    modelPasses: 6, seconds: 30, cadence: { exceedsRequestedInterval: true } } });
  assert.equal(frame.timestampSeconds, value.timestampSeconds);
  assert.equal(store.tick().action.reason, 'target_observation_stale');
});

test('legacy observations have no scan display and older nonhybrid scan metadata stays compatible', async t => {
  const { store, packet, advance } = await fixture(t);
  store.ingestFrame(packet());
  assert.equal(store.snapshot().latestFrame.detector, null);
  advance(100);
  const olderFormat = packet({ sequence: 1 });
  olderFormat.detector = { scan: { profileId: 'fast', modelPasses: 1, seconds: 0 } };
  store.ingestFrame(olderFormat);
  assert.deepEqual(store.snapshot().latestFrame.detector, {
    scan: { profileId: 'fast', effectiveProfileId: 'fast', modelPasses: 1, seconds: 0 } });
});

test('malformed optional scan display is ignored without corrupting valid observation evidence', async t => {
  const { store, packet, advance } = await fixture(t);
  const valid = { profileId: 'hybrid', effectiveProfileId: 'fast', modelPasses: 1, seconds: 8 };
  const invalid = [null, [], 'bad',
    { ...valid, profileId: 'unknown' }, { ...valid, effectiveProfileId: 'hybrid' },
    { ...valid, effectiveProfileId: undefined }, { ...valid, modelPasses: 0 },
    { ...valid, modelPasses: 7 }, { ...valid, modelPasses: 1.5 }, { ...valid, modelPasses: true },
    { ...valid, seconds: -1 }, { ...valid, seconds: 3601 }, { ...valid, seconds: '8' },
    { ...valid, seconds: NaN }, { ...valid, seconds: Infinity },
    { ...valid, cadence: { exceedsRequestedInterval: 'true' } }, { ...valid, cadence: [] }];
  for (const [sequence, scan] of invalid.entries()) {
    advance(100);
    const value = packet({ sequence });
    value.detector = { scan };
    store.ingestFrame(value);
    const state = store.snapshot();
    assert.equal(state.latestFrame.detector, null, `Invalid metadata ${sequence} was retained`);
    assert.equal(state.latestFrame.timestampSeconds, value.timestampSeconds);
    assert.equal(state.landmarks[0].humanVerified, false);
    assert.equal(state.tasks.length, 1);
    assert.equal(state.actuationEnabled, false);
  }
});
