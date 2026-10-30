import { DatabaseSync } from 'node:sqlite';
import { mkdir, readFile, writeFile, realpath } from 'node:fs/promises';
import { resolve, sep } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { ApiError, shortText } from './validation.mjs';
import { validateImage } from './finds-images.mjs';
import { RESEARCH_CATALOG, normalizeTaxonCandidate } from '../community/public/research-taxonomy.js';

const hash = value => createHash('sha256').update(value).digest('hex');
const parse = value => JSON.parse(value);
const optionalText = (value, name, limit = 160) => value === undefined || value === null || value === '' ? null : shortText(value, name, undefined, limit);
const record = value => value && typeof value === 'object' && !Array.isArray(value);
const MAX_EVENTS = 100000;
const MAX_STORAGE = 2 * 1024 * 1024 * 1024;
const specimenKeys = ['cap', 'underside', 'stem', 'substrate', 'odor', 'sporePrint', 'size', 'environment', 'bruising', 'texture', 'habitat', 'locationNotes', 'other', 'viewType', 'features', 'specimenId'];
const researchFormNotes = {
  aborted_entoloma: 'Research photos cover Entoloma abortivum, including normal gilled forms; the aborted form is not confirmed.',
  chaga: 'Research photos cover Inonotus obliquus; the sterile conk or other growth form is not confirmed.',
};

function smallJson(value, name, maxBytes = 16384, depth = 0) {
  if (depth > 6) throw new ApiError(`${name} is nested too deeply.`);
  if (value === undefined) return null;
  if (value === null || typeof value === 'boolean') return value;
  if (typeof value === 'number') { if (!Number.isFinite(value)) throw new ApiError(`${name} contains a non-finite number.`); return value; }
  if (typeof value === 'string') { if (value.length > 2048) throw new ApiError(`${name} text is too long.`); return value; }
  if (!record(value) && !Array.isArray(value)) throw new ApiError(`${name} must contain JSON values.`);
  const entries = Object.entries(value);
  if (entries.length > 64) throw new ApiError(`${name} contains too many fields.`);
  const result = Array.isArray(value) ? value.map(item => smallJson(item, name, maxBytes, depth + 1))
    : Object.fromEntries(entries.filter(([key]) => !['__proto__', 'constructor', 'prototype'].includes(key))
      .map(([key, item]) => [shortText(key, name + ' key', undefined, 100), smallJson(item, name, maxBytes, depth + 1)]));
  if (JSON.stringify(result).length > maxBytes) throw new ApiError(`${name} is too large.`);
  return result;
}

function specimen(value) {
  if (value == null) return {};
  if (!record(value)) throw new ApiError('specimen must be an object.');
  if (value.specimenId !== undefined && value.specimenId !== null && value.specimenId !== '') shortText(value.specimenId, 'specimen.specimenId', undefined, 160);
  return smallJson(Object.fromEntries(specimenKeys.filter(key => value[key] !== undefined).map(key => [key, value[key]])), 'specimen');
}

function rights(value, fallback = null) {
  if (value === undefined) return fallback ?? { trainingOptIn: false, license: null, attestedBy: null };
  if (!record(value) || typeof value.trainingOptIn !== 'boolean') throw new ApiError('rights.trainingOptIn must be a boolean.');
  const result = { trainingOptIn: value.trainingOptIn,
    license: optionalText(value.license ?? fallback?.license, 'rights.license', 500),
    attestedBy: optionalText(value.attestedBy ?? fallback?.attestedBy, 'rights.attestedBy', 160) };
  if (result.trainingOptIn && (!result.license || !result.attestedBy)) throw new ApiError('Training opt-in requires rights.license and rights.attestedBy.');
  return result;
}

function timestamp(value) {
  if (value == null || value === '') return null;
  const date = new Date(value);
  if (!['string', 'number'].includes(typeof value) || !Number.isFinite(date.getTime()) || date.getUTCFullYear() < 1900 || date.getTime() > Date.now() + 86400000) throw new ApiError('captureTimestamp must be an ISO date or Unix milliseconds, or null when unknown.');
  return date.toISOString();
}

function pageNumber(value, name, fallback, max) {
  if (value === undefined || value === null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0 || parsed > max) throw new ApiError(`${name} must be an integer from 0 to ${max}.`);
  return parsed;
}

function predictionReviewPolicy(value) {
  if (value == null) return null;
  if (!record(value) || typeof value.validationTargetMet !== 'boolean' ||
      !Number.isFinite(value.scoreThreshold) || value.scoreThreshold < 0 || value.scoreThreshold > 1.01 ||
      !Number.isFinite(value.marginThreshold) || value.marginThreshold < 0 || value.marginThreshold > 1.01 ||
      !Number.isFinite(value.targetValidationAccuracy) || value.targetValidationAccuracy < 0 || value.targetValidationAccuracy > 1) {
    throw new ApiError('Classifier review policy contains invalid thresholds.', 502, 'CLASSIFIER_PROTOCOL');
  }
  return { scoreThreshold: value.scoreThreshold, marginThreshold: value.marginThreshold,
    targetValidationAccuracy: value.targetValidationAccuracy, validationTargetMet: value.validationTargetMet };
}

export function normalizePrediction(raw, catalog) {
  if (!record(raw) || raw.schema !== 'mycoscout.group-guess.v1' || !Array.isArray(raw.guesses) || raw.guesses.length > 50 || typeof raw.abstained !== 'boolean') throw new ApiError('Classifier result has an invalid schema.', 502, 'CLASSIFIER_PROTOCOL');
  const groups = Array.isArray(catalog) ? catalog : catalog.groups;
  const classes = new Map((catalog.entries ?? []).map(entry => [entry.id, entry]));
  classes.set('other_fungi', { id: 'other_fungi', label: 'Other or unsupported fungi', groupId: 'unknown' });
  const known = new Map(groups.map(group => [group.id, group.label]));
  const seen = new Set();
  const guesses = raw.guesses.map(guess => {
    const classId = guess?.classId ?? null, identity = classId ?? guess?.groupId;
    if (!record(guess) || !known.has(guess.groupId) || seen.has(identity) ||
        classId && classes.get(classId)?.groupId !== guess.groupId ||
        !Number.isFinite(guess.modelScore) || guess.modelScore < 0 || guess.modelScore > 1) throw new ApiError('Classifier guesses do not match the catalog.', 502, 'CLASSIFIER_PROTOCOL');
    seen.add(identity);
    let label = classId ? classes.get(classId).label : known.get(guess.groupId);
    const scopeNotes = researchFormNotes[classId] ? [researchFormNotes[classId]] : [];
    if (scopeNotes.length && !label.toLowerCase().includes('form unverified')) label += ' (form unverified)';
    return { classId, groupId: guess.groupId, label, modelScore: guess.modelScore, scopeNotes };
  }).sort((a, b) => b.modelScore - a.modelScore).slice(0, 3);
  return { schema: raw.schema, evidenceVersion: optionalText(raw.evidenceVersion, 'evidenceVersion', 240), guesses,
    abstained: raw.abstained, abstentionReasons: Array.isArray(raw.abstentionReasons) ? raw.abstentionReasons.slice(0, 12).map(reason => shortText(reason, 'abstention reason', undefined, 500)) : [],
    scoreCalibration: smallJson(raw.scoreCalibration ?? {}, 'scoreCalibration'),
    reviewPolicy: predictionReviewPolicy(raw.reviewPolicy),
    humanVerificationRequired: true, speciesConfirmed: false, edibility: 'not assessed',
    professionallyVerified: false, salesApproval: false, flightGateEnabled: false };
}

export class FindsStore {
  constructor(directory, catalogFile) { this.directory = resolve(directory); this.catalogFile = catalogFile; this.db = null; this.pending = Promise.resolve(); this.queuedWrites = 0; }
  async initialize() {
    await mkdir(resolve(this.directory, 'photos'), { recursive: true });
    this.db = new DatabaseSync(resolve(this.directory, 'finds.sqlite'));
    const schemaVersion = this.db.prepare('PRAGMA user_version').get().user_version;
    if (![0, 1].includes(schemaVersion)) { this.db.close(); this.db = null; throw new Error('Finds database schema is newer than this application; preserve it and use a compatible version.'); }
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS photos(hash TEXT PRIMARY KEY, extension TEXT NOT NULL, mime TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL, bytes INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS finds(id TEXT PRIMARY KEY, created_at TEXT NOT NULL, photo_hash TEXT NOT NULL REFERENCES photos(hash), data TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT NOT NULL UNIQUE, find_id TEXT NOT NULL REFERENCES finds(id), type TEXT NOT NULL, created_at TEXT NOT NULL, data TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS events_find_seq ON events(find_id,seq);
      CREATE TABLE IF NOT EXISTS partitions(token TEXT PRIMARY KEY, partition TEXT NOT NULL);
      CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
      CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events BEGIN SELECT RAISE(ABORT,'Events are append-only'); END;
      CREATE TRIGGER IF NOT EXISTS finds_no_update BEFORE UPDATE ON finds BEGIN SELECT RAISE(ABORT,'Finds are immutable'); END;
      CREATE TRIGGER IF NOT EXISTS finds_no_delete BEFORE DELETE ON finds BEGIN SELECT RAISE(ABORT,'Finds are immutable'); END;
      PRAGMA user_version=1;`);
    return this;
  }
  close() { if (this.db) { this.db.close(); this.db = null; } }
  serialize(operation) {
    if (this.queuedWrites >= 8) return Promise.reject(new ApiError('The review database is busy. Retry shortly.', 429, 'DATABASE_BUSY'));
    this.queuedWrites++;
    const result = this.pending.then(operation).finally(() => { this.queuedWrites--; });
    this.pending = result.catch(() => {}); return result;
  }
  transaction(operation) { this.db.exec('BEGIN IMMEDIATE'); try { const result = operation(); this.db.exec('COMMIT'); return result; } catch (error) { this.db.exec('ROLLBACK'); throw error; } }
  async catalog() {
    let catalog;
    try { catalog = parse(await readFile(this.catalogFile, 'utf8')); }
    catch { throw new ApiError('The local mushroom catalog is unavailable.', 503, 'CATALOG_UNAVAILABLE'); }
    if (!Array.isArray(catalog.groups) || catalog.groups.some(group => typeof group.id !== 'string' || typeof group.label !== 'string')) throw new ApiError('Catalog groups are invalid.', 503, 'CATALOG_UNAVAILABLE');
    return catalog;
  }
  append(findId, type, data, createdAt = new Date().toISOString()) {
    if (this.db.prepare('SELECT count(*) AS n FROM events').get().n >= MAX_EVENTS) throw new ApiError('Review event capacity reached. Export this database before continuing.', 409, 'DATABASE_CAPACITY');
    const id = randomUUID();
    const result = this.db.prepare('INSERT INTO events(id,find_id,type,created_at,data) VALUES(?,?,?,?,?)').run(id, findId, type, createdAt, JSON.stringify(data));
    return { id, seq: Number(result.lastInsertRowid), findId, type, createdAt, ...data };
  }
  grouping(data, photoHash) {
    const tokens = [`photo:${photoHash}`];
    if (data.sessionId) tokens.push('session:' + hash(data.sessionId));
    if (data.groupId) tokens.push('specimen:' + hash(data.groupId));
    if (data.specimen?.specimenId) tokens.push('specimen:' + hash(data.specimen.specimenId));
    return tokens;
  }
  allGroupTokens(id) {
    const initial = this.db.prepare('SELECT data FROM finds WHERE id=?').get(id);
    const tokens = new Set(parse(initial.data).groupTokens);
    for (const event of this.db.prepare("SELECT data FROM events WHERE find_id=? AND type='review'").all(id)) {
      for (const token of parse(event.data).groupTokens ?? []) tokens.add(token);
    }
    return [...tokens];
  }
  assignPartition(tokens, evaluationOnly) {
    const existing = new Set(tokens.map(token => this.db.prepare('SELECT partition FROM partitions WHERE token=?').get(token)?.partition).filter(Boolean));
    if (evaluationOnly) existing.add('test');
    let partition;
    if (existing.size > 1 || existing.has('quarantine')) partition = 'quarantine';
    else if (existing.size === 1) partition = [...existing][0];
    else { const bucket = parseInt(hash(tokens.at(-1)).slice(0, 8), 16) % 10; partition = bucket < 8 ? 'train' : bucket === 8 ? 'validation' : 'test'; }
    for (const token of tokens) {
      this.db.prepare('INSERT OR IGNORE INTO partitions(token,partition) VALUES(?,?)').run(token, partition);
      if (partition === 'quarantine') this.db.prepare('UPDATE partitions SET partition=? WHERE token=?').run(partition, token);
    }
    return partition;
  }
  async create(input) {
    const image = validateImage(input);
    if (input.evaluationOnly !== undefined && typeof input.evaluationOnly !== 'boolean') throw new ApiError('evaluationOnly must be boolean.');
    const data = { source: optionalText(input.source, 'source') ?? 'phone_upload',
      capturedAt: timestamp(input.captureTimestamp), deviceId: optionalText(input.deviceId, 'deviceId'),
      sessionId: optionalText(input.sessionId, 'sessionId'), groupId: optionalText(input.groupId, 'groupId'),
      metadata: smallJson(input.metadata ?? {}, 'metadata'), camera: smallJson(input.camera ?? null, 'camera'),
      depth: smallJson(input.depth ?? null, 'depth'), specimen: specimen(input.specimen), rights: rights(input.rights),
      evaluationOnly: input.evaluationOnly === true };
    return this.serialize(async () => {
      if (this.db.prepare('SELECT count(*) AS n FROM finds').get().n >= 10000) throw new ApiError('Find capacity reached.', 409, 'DATABASE_CAPACITY');
      const photo = this.db.prepare('SELECT * FROM photos WHERE hash=?').get(image.sha256);
      const used = this.db.prepare('SELECT coalesce(sum(bytes),0) AS n FROM photos').get().n;
      if (!photo && used + image.bytes.length > MAX_STORAGE) throw new ApiError('Photo storage reached 2 GiB. Export and archive this database.', 409, 'DATABASE_CAPACITY');
      const target = resolve(this.directory, 'photos', `${image.sha256}.${image.extension}`);
      try { await writeFile(target, image.bytes, { flag: 'wx' }); }
      catch (error) { if (error.code !== 'EEXIST') throw error; if (hash(await readFile(target)) !== image.sha256) throw new Error('Stored photo hash mismatch'); }
      return this.transaction(() => {
        this.db.prepare('INSERT OR IGNORE INTO photos(hash,extension,mime,width,height,bytes) VALUES(?,?,?,?,?,?)')
          .run(image.sha256, image.extension, image.mimeType, image.width, image.height, image.bytes.length);
        const id = randomUUID(), createdAt = new Date().toISOString();
        data.groupTokens = this.grouping(data, image.sha256);
        data.initialPartition = this.assignPartition(data.groupTokens, data.evaluationOnly);
        this.db.prepare('INSERT INTO finds(id,created_at,photo_hash,data) VALUES(?,?,?,?)').run(id, createdAt, image.sha256, JSON.stringify(data));
        this.append(id, 'find_created', { reviewStatus: 'pending' }, createdAt);
        return this.get(id);
      });
    });
  }
  get(id) {
    if (typeof id !== 'string' || !/^[a-f0-9-]{36}$/.test(id)) throw new ApiError('Find was not found.', 404, 'NOT_FOUND');
    const row = this.db.prepare('SELECT f.*,p.extension,p.mime,p.width,p.height,p.bytes FROM finds f JOIN photos p ON p.hash=f.photo_hash WHERE f.id=?').get(id);
    if (!row) throw new ApiError('Find was not found.', 404, 'NOT_FOUND');
    const data = parse(row.data), events = this.db.prepare('SELECT * FROM events WHERE find_id=? ORDER BY seq').all(id);
    let prediction = null, latestReview = null, currentRights = data.rights, currentSpecimen = { ...data.specimen };
    for (const event of events) {
      const payload = parse(event.data);
      if (event.type === 'prediction') prediction = { ...payload.prediction, eventId: event.id, createdAt: event.created_at };
      if (event.type === 'review') { latestReview = { ...payload, id: event.id, seq: event.seq, createdAt: event.created_at }; currentRights = payload.rights; currentSpecimen = { ...currentSpecimen, ...payload.specimen }; }
    }
    const reviewStatus = !latestReview ? 'pending' : latestReview.action === 'unresolved' ? 'unresolved' : latestReview.action === 'reject' ? 'rejected' : 'reviewed';
    const { groupTokens, initialPartition, ...visible } = data;
    return { id: row.id, ...visible, rights: currentRights, createdAt: row.created_at, updatedAt: events.at(-1)?.created_at ?? row.created_at,
      revision: events.at(-1)?.seq ?? 0, imageUrl: `/api/finds/${row.id}/image`, contentHash: row.photo_hash,
      image: { sha256: row.photo_hash, mimeType: row.mime, width: row.width, height: row.height, bytes: row.bytes },
      reviewStatus, prediction, latestReview, humanVerificationRequired: true, speciesConfirmed: false,
      professionallyVerified: false, salesApproval: false, flightGateEnabled: false,
      humanReviewed: !!latestReview && ['confirm_group', 'correct_species'].includes(latestReview.action),
      specimen: currentSpecimen };
  }
  list(since = 0, limit = 100) {
    const page = this.events(since, limit), ids = [...new Set(page.events.map(event => event.findId))];
    return { finds: ids.map(id => this.get(id)), cursor: page.cursor, hasMore: page.hasMore };
  }
  events(since = 0, limit = 100) {
    since = pageNumber(since, 'since', 0, Number.MAX_SAFE_INTEGER); limit = pageNumber(limit, 'limit', 100, 200);
    if (limit < 1) throw new ApiError('limit must be at least 1.');
    const rows = this.db.prepare('SELECT * FROM events WHERE seq>? ORDER BY seq LIMIT ?').all(since, limit + 1), hasMore = rows.length > limit;
    const page = rows.slice(0, limit);
    return { events: page.map(row => ({ seq: row.seq, id: row.id, findId: row.find_id, type: row.type, createdAt: row.created_at, ...parse(row.data) })),
      cursor: page.at(-1)?.seq ?? since, hasMore };
  }
  async image(id) {
    const find = this.get(id), extension = find.image.mimeType === 'image/png' ? 'png' : 'jpg';
    const path = await realpath(resolve(this.directory, 'photos', `${find.contentHash}.${extension}`));
    const photoDirectory = await realpath(resolve(this.directory, 'photos'));
    if (!path.startsWith(photoDirectory + sep)) throw new ApiError('Photo path is invalid.', 403);
    const bytes = await readFile(path);
    if (hash(bytes) !== find.contentHash) throw new ApiError('Stored photo failed its integrity check.', 409, 'PHOTO_INTEGRITY');
    return { ...find.image, byteLength: find.image.bytes, bytes };
  }
  async review(id, input) {
    const catalog = await this.catalog(), groupIds = new Set([...catalog.groups,...RESEARCH_CATALOG.groups].map(group => group.id));
    const action = input.action;
    if (!['confirm_group', 'correct_species', 'unresolved', 'reject'].includes(action)) throw new ApiError('Review action must be confirm_group, correct_species, unresolved, or reject.');
    const groupId = optionalText(input.groupId, 'groupId');
    if (['confirm_group', 'correct_species'].includes(action) && (!groupId || !groupIds.has(groupId))) throw new ApiError('A reviewed groupId from the catalog is required.');
    if (groupId && !groupIds.has(groupId)) throw new ApiError('groupId is not in the catalog.');
    const classId = optionalText(input.classId, 'classId');
    if (classId && ![...(catalog.entries ?? []), { id: 'other_fungi', groupId: 'unknown' }].some(entry => entry.id === classId && entry.groupId === groupId)) throw new ApiError('classId must match the reviewed catalog group.');
    let taxonCandidate;
    try{taxonCandidate=normalizeTaxonCandidate(input.taxonCandidate,{groupId:groupId??'unknown'});}catch(error){throw new ApiError(error.message);}
    const review = { action, groupId, classId, species: optionalText(input.species, 'species', 240),
      ...(taxonCandidate?{taxonCandidate}:{}),
      reviewerName: shortText(input.reviewerName, 'reviewerName', undefined, 160),
      credentialReference: optionalText(input.credentialReference, 'credentialReference', 500), credentialVerified: false,
      notes: optionalText(input.notes, 'notes', 2000), specimen: specimen(input.specimen),
      speciesProfessionallyVerified: false, salesApproval: false, flightGateEnabled: false };
    if (action === 'correct_species' && !review.species) throw new ApiError('A corrected species name is required.');
    if (input.trainingOptIn !== undefined && typeof input.trainingOptIn !== 'boolean') throw new ApiError('trainingOptIn must be boolean.');
    const checkReviewVersion = Object.hasOwn(input, 'expectedReviewId');
    if (checkReviewVersion && input.expectedReviewId !== null &&
        (typeof input.expectedReviewId !== 'string' || !/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(input.expectedReviewId))) {
      throw new ApiError('expectedReviewId must be a UUID or null for an initially unreviewed find.');
    }
    const expectedReviewId = input.expectedReviewId?.toLowerCase() ?? null;
    return this.serialize(() => this.transaction(() => {
      const find = this.get(id);
      if (checkReviewVersion && (find.latestReview?.id ?? null) !== expectedReviewId) {
        throw new ApiError('Another device changed this review. Reload the latest review before saving your changes.', 409, 'REVIEW_CONFLICT');
      }
      const count = this.db.prepare("SELECT count(*) AS n FROM events WHERE find_id=? AND type='review'").get(id).n;
      if (count >= 200) throw new ApiError('Review history capacity reached for this find.', 409, 'DATABASE_CAPACITY');
      review.rights = rights(input.rights, find.rights);
      if (input.trainingOptIn !== undefined) review.rights = rights({ ...review.rights, trainingOptIn: input.trainingOptIn }, find.rights);
      review.groupTokens = this.allGroupTokens(id);
      if (review.specimen.specimenId) review.groupTokens.push('specimen:' + hash(review.specimen.specimenId));
      review.groupTokens = [...new Set(review.groupTokens)];
      this.assignPartition(review.groupTokens, find.evaluationOnly);
      const event = this.append(id, 'review', review);
      return { find: this.get(id), review: { ...review, id: event.id, seq: event.seq, createdAt: event.createdAt } };
    }));
  }
  async prediction(id, raw) {
    const catalog = await this.catalog(), prediction = normalizePrediction(raw, catalog);
    return this.serialize(() => this.transaction(() => {
      this.get(id);
      if (this.db.prepare("SELECT count(*) AS n FROM events WHERE find_id=? AND type='prediction'").get(id).n >= 500) throw new ApiError('Prediction history capacity reached for this find.', 409, 'DATABASE_CAPACITY');
      this.append(id, 'prediction', { prediction }); return { find: this.get(id), result: prediction };
    }));
  }
  async exportTraining() {
    return this.serialize(async () => {
      const rows = this.db.prepare('SELECT * FROM finds ORDER BY id').all();
      // Connected source groups include non-exported finds, so revoked labels cannot break split grouping.
      const parent = new Map(), connect = token => { if (!parent.has(token)) parent.set(token, token); let root = token; while (parent.get(root) !== root) root = parent.get(root); let next = token; while (parent.get(next) !== next) { const old = parent.get(next); parent.set(next, root); next = old; } return root; };
      for (const row of rows) { const tokens = this.allGroupTokens(row.id); for (const token of tokens) { const a = connect(tokens[0]), b = connect(token); if (a !== b) parent.set(b, a); } }
      const components = new Map();
      for (const token of parent.keys()) { const key = connect(token); if (!components.has(key)) components.set(key, []); components.get(key).push(token); }
      const grouped = new Map();
      for (const [key, tokens] of components) {
        const partitions = new Set(tokens.map(token => this.db.prepare('SELECT partition FROM partitions WHERE token=?').get(token)?.partition));
        grouped.set(key, { sourceGroupId: 'group-' + hash(tokens.sort().join('\n')).slice(0, 24), partition: partitions.size === 1 ? [...partitions][0] : 'quarantine' });
      }
      const records = [], excluded = { unreviewedOrUnresolved: 0, rightsNotGranted: 0, partitionConflict: 0, unknownGroup: 0, conflictingLabels: 0 };
      const active = rows.map(row => ({ row, data: parse(row.data), find: this.get(row.id) }));
      const labelsByPhoto = new Map();
      for (const { find } of active) {
        if (!find.humanReviewed || !find.rights.trainingOptIn) continue;
        if (!labelsByPhoto.has(find.contentHash)) labelsByPhoto.set(find.contentHash, { groups: new Set(), classes: new Set(), species: new Set() });
        const labels = labelsByPhoto.get(find.contentHash), review = find.latestReview;
        labels.groups.add(review.groupId);
        if (review.classId) labels.classes.add(review.classId);
        if (review.species) labels.species.add(review.species.toLowerCase());
      }
      for (const { row, find, data } of active) {
        const review = find.latestReview;
        if (!review || !['confirm_group', 'correct_species'].includes(review.action)) { excluded.unreviewedOrUnresolved++; continue; }
        if (!find.rights.trainingOptIn || !find.rights.license || !find.rights.attestedBy) { excluded.rightsNotGranted++; continue; }
        if (review.groupId === 'unknown') { excluded.unknownGroup++; continue; }
        const labels = labelsByPhoto.get(find.contentHash);
        if (labels && [labels.groups, labels.classes, labels.species].some(values => values.size > 1)) { excluded.conflictingLabels++; continue; }
        const group = grouped.get(connect(data.groupTokens[0]));
        if (!['train', 'validation', 'test'].includes(group.partition)) { excluded.partitionConflict++; continue; }
        records.push({ findId: find.id, image: { ...find.image, relativePath: `photos/${find.contentHash}.${find.image.mimeType === 'image/png' ? 'png' : 'jpg'}`, url: find.imageUrl },
          ...group, labels: { groupId: review.groupId, classId: review.classId, speciesReported: review.species, labelSource: 'human_review', speciesProfessionallyVerified: false,
            ...(review.taxonCandidate?{taxonCandidate:review.taxonCandidate}:{}) },
          review: { id: review.id, action: review.action, reviewerName: review.reviewerName, credentialReference: review.credentialReference, credentialVerified: false, createdAt: review.createdAt },
          specimen: find.specimen, rights: find.rights,
          provenance: { source: find.source, sessionId: find.sessionId, deviceId: find.deviceId, groupId: find.groupId,
            capturedAt: find.capturedAt, evaluationOnly: find.evaluationOnly,
            sourceMetadata: smallJson(find.metadata ?? {}, 'sourceMetadata') } });
      }
      const catalog = await this.catalog();
      const releaseId = 'reviewed-' + hash(JSON.stringify({ records, excluded, catalogSchema: catalog.schema, catalogVersion: catalog.version }));
      const manifest = { schema: 'mycoscout.reviewed-training-export.v1', releaseId, catalogSchema: catalog.schema ?? null,
        labelsFrom: 'explicit_human_reviews_only', imagePathBase: 'finds database directory', records,
        summary: { records: records.length, train: records.filter(row => row.partition === 'train').length,
          validation: records.filter(row => row.partition === 'validation').length, test: records.filter(row => row.partition === 'test').length, excluded },
        policies: { automaticTraining: false, aiGuessesAreLabels: false, sourceGroupingEnforced: true, testUsedForTraining: false,
          credentialsVerified: false, speciesConfirmed: false, salesApproval: false },
        notes: ['Training uses only the train partition; validation tunes models; test remains held out.',
          'Photo rights and reviewer credentials are self-asserted. A review does not constitute professional identification or sale approval.',
          'Duplicate photos and all historically declared specimen/session groups share partitions. Conflicting existing assignments or photo labels are quarantined.',
          'A training importer must deduplicate identical image SHA256 values and reconcile them against prior corpus test/validation assignments.',
          'Earlier release manifests remain immutable. Revocation or correction creates a different release; rebuild datasets from the current release.'] };
      await mkdir(resolve(this.directory, 'exports'), { recursive: true });
      const filename = resolve(this.directory, 'exports', `${releaseId}.json`);
      try { await writeFile(filename, JSON.stringify(manifest, null, 2) + '\n', { flag: 'wx' }); }
      catch (error) { if (error.code !== 'EEXIST') throw error; }
      return manifest;
    });
  }
}
