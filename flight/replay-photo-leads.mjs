import { readFile, stat, mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { InspectionQueue } from './habitat-search.mjs';

export const INPUT_SCHEMA = 'mycoscout.archival-scout-leads.v2';
const OUTPUT_SCHEMA = 'mycoscout.archival-scout-replay.v2';
const MAX_INPUT_BYTES = 32 * 1024 * 1024;
const MAX_OBSERVATIONS = 50000;
const forbiddenGeometry = ['capturePose', 'pose', 'metric', 'position', 'mapPosition'];

// Retain bounded review context separately from the observation passed to the queue.
// A photo ranking is not a detector score, a metric position, or flight permission.
function reviewValue(value, depth = 0) {
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  if (typeof value === 'string') return value.slice(0, 512);
  if (depth >= 3) return null;
  if (Array.isArray(value)) return value.slice(0, 16).map(item => reviewValue(item, depth + 1));
  if (value && typeof value === 'object') {
    return Object.fromEntries(Object.entries(value).slice(0, 24)
      .filter(([key]) => !['__proto__', 'prototype', 'constructor'].includes(key))
      .map(([key, item]) => [key.slice(0, 100), reviewValue(item, depth + 1)]));
  }
  return null;
}

function reviewMetadata(observation) {
  const rawReview = observation.reviewMetadata;
  const result = rawReview && typeof rawReview === 'object' && !Array.isArray(rawReview)
    ? reviewValue(rawReview) : {};
  for (const key of ['metadata', 'ranking', 'rankingMetadata', 'photoRanking']) {
    if (observation[key] !== undefined) result[key] = reviewValue(observation[key]);
  }
  if (Number.isInteger(observation.bbox?.imageWidth) && observation.bbox.imageWidth > 0 &&
      Number.isInteger(observation.bbox?.imageHeight) && observation.bbox.imageHeight > 0) {
    result.imageDimensions = { width: observation.bbox.imageWidth, height: observation.bbox.imageHeight };
  }
  // Also bound the serialized size, including nested arrays and objects.
  return JSON.stringify(result).length <= 16384 ? result : { omitted: 'Review metadata exceeded 16 KiB' };
}

function normalizedBbox(bbox) {
  const coordinates = Array.isArray(bbox) ? bbox : bbox?.xyxy;
  if (Array.isArray(coordinates) && coordinates.length === 4 && coordinates.every(Number.isFinite)) {
    const [x1, y1, x2, y2] = coordinates;
    return { x1, y1, x2, y2, units: 'pixel' };
  }
  return bbox;
}

/** Archive queue replay only: no mission, setpoints, telemetry, or aircraft transport. */
export function replayPhotoLeads(document, searchConfig) {
  if (!document || (document.schema ?? document.schemaVersion) !== INPUT_SCHEMA ||
      document.simulationOnly !== true || document.actuationEnabled !== false ||
      !Array.isArray(document.observations)) {
    throw new Error(`Expected ${INPUT_SCHEMA}, simulationOnly:true, actuationEnabled:false, and observations[]`);
  }
  if (document.observations.length > MAX_OBSERVATIONS) throw new Error('Archive exceeds 50000 observations');
  const queue = new InspectionQueue(searchConfig);
  const metadataByLead = new Map();
  const rejections = [], rejectionCounts = {};
  let acceptedCount = 0, rejectedCount = 0;
  const reject = (index, reason) => {
    rejectedCount++;
    rejectionCounts[reason] = (rejectionCounts[reason] ?? 0) + 1;
    if (rejections.length < searchConfig.limits.maxRejectedHistory) rejections.push({ index, reason });
  };
  for (const [index, raw] of document.observations.entries()) {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) { reject(index, 'malformed_observation'); continue; }
    if (raw.capturedMs !== null || raw.archivalPhoto !== true) {
      reject(index, 'archive_requires_unknown_capture_time'); continue;
    }
    if (forbiddenGeometry.some(key => Object.hasOwn(raw, key))) {
      reject(index, 'archive_cannot_supply_navigation_geometry'); continue;
    }
    // Explicit whitelist: upstream ranking and any unexpected fields cannot alter gates.
    const observation = {
      id: raw.id, frameId: raw.frameId, kind: raw.kind, cueType: raw.cueType, confidence: raw.confidence,
      source: raw.source, capturedMs: null, archivalPhoto: true,
      bbox: normalizedBbox(raw.bbox), signals: raw.signals,
      semanticFractions: raw.semanticFractions, month: raw.month,
    };
    const result = queue.ingest(observation, 0);
    if (!result.accepted) { reject(index, result.reason); continue; }
    acceptedCount++;
    metadataByLead.set(result.leadId, reviewMetadata(raw));
  }
  const screenLeads = queue.snapshot(0).map(lead => ({
    ...lead, archivalPhoto: true, reviewMetadata: metadataByLead.get(lead.id) ?? {},
    reviewMetadataAffectsEligibility: false,
  }));
  if (screenLeads.some(lead => lead.locationStatus !== 'screen_only' || lead.eligible ||
      lead.position !== null || lead.captureTimeKnown || lead.originalCapturedMs !== null)) {
    throw new Error('Archive replay invariant failed: a photo became a navigation target');
  }
  return {
    schema: OUTPUT_SCHEMA, schemaVersion: 2, simulationOnly: true, actuationEnabled: false,
    replayTimeMs: 0, captureTimeKnown: false, mapTargets: [], screenLeads, habitatLog: structuredClone(queue.habitatLog), rejections, rejectionCounts,
    summary: {
      inputObservationCount: document.observations.length, acceptedObservationCount: acceptedCount,
      rejectedObservationCount: rejectedCount, screenLeadCount: screenLeads.length,
      fungusLeadCount: screenLeads.filter(lead => lead.kind === 'fungus').length,
      habitatLeadCount: screenLeads.filter(lead => lead.kind === 'habitat').length,
      mapTargetCount: 0, actionEligibleCount: 0, flightCommands: 0,
      omittedRejectionDetails: Math.max(0, rejectedCount - rejections.length),
    },
    notes: [
      'Saved photo detections were ingested into the inspection queue at replay time zero.',
      'Original capture times remain unknown; replay time is not a capture or inference timestamp.',
      'All leads are screen-only. No map targets, flight state, setpoints, or flight commands were created.',
      'Detector scores and photo-ranking metadata do not confirm mushroom presence or species.',
      'Habitat signals only affect heuristic review priority; they do not establish clearance or ground safety.',
    ],
  };
}

export async function writePhotoReplay(inputPath, outputPath) {
  const input = resolve(inputPath), output = resolve(outputPath);
  if (input.toLowerCase() === output.toLowerCase()) throw new Error('Output must differ from the input file');
  if ((await stat(input)).size > MAX_INPUT_BYTES) throw new Error('Archive input exceeds 32 MiB');
  const document = JSON.parse(await readFile(input, 'utf8'));
  const config = JSON.parse(await readFile(new URL('../config/x500-habitat-search.json', import.meta.url), 'utf8'));
  const report = replayPhotoLeads(document, config);
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + '\n', 'utf8');
  return report;
}

function cliArguments(args) {
  const result = {};
  for (let index = 0; index < args.length; index += 2) {
    const flag = args[index], value = args[index + 1];
    if (!['--input', '--output'].includes(flag) || !value || value.startsWith('--') || result[flag]) {
      throw new Error('Usage: node flight/replay-photo-leads.mjs --input archive.json --output replay.json');
    }
    result[flag] = value;
  }
  if (!result['--input'] || !result['--output']) throw new Error('Both --input and --output are required');
  return result;
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const args = cliArguments(process.argv.slice(2));
    const report = await writePhotoReplay(args['--input'], args['--output']);
    console.log(JSON.stringify(report.summary, null, 2));
  } catch (error) {
    console.error(`Photo replay failed: ${error.message}`);
    process.exitCode = 1;
  }
}
