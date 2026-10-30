import { DatabaseSync } from 'node:sqlite';
import { randomBytes, createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { COMMUNITY_GROUPS, normalizeTaxonCandidate } from './public/research-taxonomy.js';
import { validConfounder } from './public/review-evidence.js';

export class ReviewError extends Error {
  constructor(message, status = 400) { super(message); this.status = status; }
}
export const GROUPS = COMMUNITY_GROUPS.map(([id])=>id);
const hash = value => createHash('sha256').update(value).digest('hex');
const notice = 'By submitting this review, you permit Great Lakes Gleaner and MycoScout to use your annotations to train, evaluate and improve mushroom-vision software, including commercial software. This permission covers your annotations only; source images keep their existing licenses. You can withdraw permission for future use of this review from this browser session. Withdrawal cannot undo completed training.';
export const PARTICIPATION_TERMS = Object.freeze({version:'community-annotations-2026-10-29-v1',notice,noticeSha256:hash(notice)});
const validTime = value => typeof value==='string' && Number.isFinite(Date.parse(value)) && new Date(value).toISOString()===value;
function historicalPermission(input) {
  if(typeof input.trainingConsent!=='boolean') throw new ReviewError('Historical training permission must be preserved.');
  const result={trainingConsent:input.trainingConsent};
  if(input.participationTerms!==undefined){
    const terms=input.participationTerms;
    if(!terms||!/^community-annotations-[a-z0-9.-]{1,70}$/.test(terms.version)||!(/^[a-f0-9]{64}$/).test(terms.noticeSha256)||!validTime(terms.acceptedAt)||terms.acceptance!=='submission') throw new ReviewError('Invalid historical participation terms.');
    result.participationTerms={version:terms.version,noticeSha256:terms.noticeSha256,acceptedAt:terms.acceptedAt,acceptance:'submission'};
  }
  if(input.permissionWithdrawal!==undefined){
    const withdrawal=input.permissionWithdrawal;
    if(input.trainingConsent!==false||!withdrawal||!validTime(withdrawal.withdrawnAt)||withdrawal.scope!=='future-training-and-evaluation'||!Number.isSafeInteger(withdrawal.priorRevision)||withdrawal.priorRevision<1)throw new ReviewError('Invalid historical permission withdrawal.');
    result.permissionWithdrawal={withdrawnAt:withdrawal.withdrawnAt,scope:withdrawal.scope,priorRevision:withdrawal.priorRevision};
  }
  return result;
}
export function validateReview(input, frame, {historical=false,acceptedAt=new Date().toISOString()}={}) {
  if (!input || !['visible','absent','uncertain'].includes(input.status)) throw new ReviewError('Choose what you can see first.');
  if (!Array.isArray(input.boxes) || input.boxes.length > 64) throw new ReviewError('Use at most 64 boxes.');
  for (const field of ['complete','aiAssisted']) if (typeof input[field] !== 'boolean') throw new ReviewError(`${field} must be true or false.`);
  let permission;
  if(historical)permission=historicalPermission(input);
  else {
    if(input.participationTerms?.version!==PARTICIPATION_TERMS.version||input.participationTerms?.noticeSha256!==PARTICIPATION_TERMS.noticeSha256)throw new ReviewError('Participation terms changed or were not loaded. Reload this page and read the terms beside Submit before submitting.',428);
    if((input.trainingConsent!==undefined&&input.trainingConsent!==true)||input.permissionWithdrawal!==undefined)throw new ReviewError('Use the separate withdrawal action to withdraw permission for an earlier review.');
    permission={trainingConsent:true,participationTerms:{version:PARTICIPATION_TERMS.version,noticeSha256:PARTICIPATION_TERMS.noticeSha256,acceptedAt,acceptance:'submission'}};
  }
  if (!Number.isInteger(input.baseRevision) || input.baseRevision < 0) throw new ReviewError('Reload your saved review before submitting.');
  if (input.status !== 'visible' && input.boxes.length) throw new ReviewError('Only visible mushrooms can have boxes.');
  if (input.complete && (input.status === 'uncertain' || (input.status === 'visible' && !input.boxes.length))) throw new ReviewError('Complete labels need a box for every visible mushroom, or a clear None visible rating.');
  const boxes = input.boxes.map(box => {
    const b = box?.bbox;
    if (!Array.isArray(b) || b.length !== 4 || b.some(n => typeof n !== 'number' || !Number.isFinite(n)) ||
      b[0] < 0 || b[1] < 0 || b[2] > frame.width || b[3] > frame.height || b[0] >= b[2] || b[1] >= b[3]) throw new ReviewError('Draw boxes inside the original image.');
    if (!GROUPS.includes(box.group ?? 'unknown')) throw new ReviewError('Choose a supported mushroom group or Unknown.');
    let taxonCandidate;
    try { taxonCandidate=normalizeTaxonCandidate(box.taxonCandidate,{groupId:box.group??'unknown'}); }
    catch(error){throw new ReviewError(error.message);}
    return {bbox: [...b], group: box.group ?? 'unknown',...(taxonCandidate?{taxonCandidate}:{})};
  });
  let negativeBoxes,aiAssessment;
  if(input.negativeBoxes!==undefined){
    if(!Array.isArray(input.negativeBoxes)||input.negativeBoxes.length>64)throw new ReviewError('Use at most 64 confusing-object boxes.');
    negativeBoxes=input.negativeBoxes.map(item=>{
      const b=item?.bbox;
      if(!validConfounder(item?.kind)||!Array.isArray(b)||b.length!==4||!b.every(Number.isFinite)||b[0]<0||b[1]<0||b[2]>frame.width||b[3]>frame.height||b[2]<=b[0]||b[3]<=b[1])throw new ReviewError('Mark a supported non-mushroom object inside the image.');
      return {kind:item.kind,bbox:[...b],explicitNonFungal:true,labelAuthority:'community-unverified',trainingApproved:false};
    });
  }
  if(input.aiAssessment!==undefined){
    const a=input.aiAssessment;
    if(!input.aiAssisted||!a||!['agree','corrected','uncertain'].includes(a.decision)||typeof a.reviewId!=='string'||a.reviewId.length>180||!/^[-A-Za-z0-9_:]+$/.test(a.reviewId)||!Number.isSafeInteger(a.revision)||a.revision<1)throw new ReviewError('Choose a valid comparison with the displayed AI review.');
    aiAssessment={decision:a.decision,reviewId:a.reviewId,revision:a.revision,labelAuthority:'community-unverified'};
  }
  return {status:input.status, boxes, complete:input.complete, trainingConsent:permission.trainingConsent,
    ...(negativeBoxes?{negativeBoxes}:{}),...(aiAssessment?{aiAssessment}:{}),
    aiAssisted:input.aiAssisted, labelAuthority:'community-unverified', trainingApproved:false,
    ...(permission.participationTerms?{participationTerms:permission.participationTerms}:{}),
    ...(permission.permissionWithdrawal?{permissionWithdrawal:permission.permissionWithdrawal}:{})};
}

export class CommunityStore {
  constructor(directory) {
    mkdirSync(directory,{recursive:true});
    this.db = new DatabaseSync(resolve(directory,'community.sqlite'));
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY, csrf TEXT NOT NULL, expires INTEGER NOT NULL);
      CREATE TABLE IF NOT EXISTS reviews(id INTEGER PRIMARY KEY AUTOINCREMENT, reviewer TEXT NOT NULL, frame TEXT NOT NULL,
        sha256 TEXT NOT NULL, revision INTEGER NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(reviewer,frame,revision));
      CREATE INDEX IF NOT EXISTS reviews_latest ON reviews(reviewer,frame,revision DESC);
      CREATE TRIGGER IF NOT EXISTS review_immutable_update BEFORE UPDATE ON reviews BEGIN SELECT RAISE(ABORT,'Review history is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS review_immutable_delete BEFORE DELETE ON reviews BEGIN SELECT RAISE(ABORT,'Review history is immutable'); END;`);
  }
  session(token) {
    if (!/^[a-f0-9]{64}$/.test(token ?? '')) return null;
    const row=this.db.prepare('SELECT * FROM sessions WHERE token_hash=? AND expires>?').get(hash(token),Date.now());
    return row ? {reviewer:row.token_hash,csrfToken:row.csrf} : null;
  }
  createSession() {
    this.db.prepare('DELETE FROM sessions WHERE expires<?').run(Date.now());
    if(this.db.prepare('SELECT COUNT(*) AS n FROM sessions').get().n > 50000) throw new ReviewError('Review service is at capacity. Please try later.',503);
    const token=randomBytes(32).toString('hex'), csrfToken=randomBytes(32).toString('hex');
    this.db.prepare('INSERT INTO sessions VALUES(?,?,?)').run(hash(token),csrfToken,Date.now()+90*86400000);
    return {token,csrfToken,reviewer:hash(token)};
  }
  get(reviewer,frame,expectedSha256=null) {
    const row=this.db.prepare('SELECT * FROM reviews WHERE reviewer=? AND frame=? ORDER BY revision DESC LIMIT 1').get(reviewer,frame);
    if(row&&expectedSha256!==null&&row.sha256!==expectedSha256)throw new ReviewError('This frame’s original pixels changed. Keep the original image or publish it with a new frame ID before reviewing.',409);
    return row?{...JSON.parse(row.payload),revision:row.revision,id:row.id,createdAt:row.created_at}:null;
  }
  save(reviewer,frame,input) {
    const createdAt=new Date().toISOString(),payload=validateReview(input,frame,{acceptedAt:createdAt});
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const original=this.db.prepare('SELECT sha256 FROM reviews WHERE frame=? LIMIT 1').get(frame.id);
      if(original&&original.sha256!==frame.sha256)throw new ReviewError('This frame’s original pixels changed. Keep the original image or publish it with a new frame ID before reviewing.',409);
      const previous=this.get(reviewer,frame.id,frame.sha256);
      if ((previous?.revision??0)!==input.baseRevision) throw new ReviewError('This review changed in another tab. Reload it before saving again.',409);
      const revision=input.baseRevision+1;
      this.db.prepare('INSERT INTO reviews(reviewer,frame,sha256,revision,created_at,payload) VALUES(?,?,?,?,?,?)')
        .run(reviewer,frame.id,frame.sha256,revision,createdAt,JSON.stringify(payload));
      this.db.exec('COMMIT');return this.get(reviewer,frame.id);
    } catch(error) {this.db.exec('ROLLBACK');throw error;}
  }
  withdraw(reviewer,frame,input) {
    if(!Number.isSafeInteger(input?.baseRevision)||input.baseRevision<1)throw new ReviewError('Reload your saved review before withdrawing permission.');
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const previous=this.get(reviewer,frame.id,frame.sha256);
      if(!previous)throw new ReviewError('No earlier review exists in this browser session.',404);
      if(previous.revision!==input.baseRevision)throw new ReviewError('This review changed in another tab. Reload it before withdrawing permission.',409);
      if(previous.trainingConsent!==true)throw new ReviewError('Permission for this review is already withheld.',409);
      const {revision,id,createdAt,...saved}=previous,now=new Date().toISOString();
      const payload={...saved,trainingConsent:false,trainingApproved:false,labelAuthority:'community-unverified',
        permissionWithdrawal:{withdrawnAt:now,scope:'future-training-and-evaluation',priorRevision:revision}};
      this.db.prepare('INSERT INTO reviews(reviewer,frame,sha256,revision,created_at,payload) VALUES(?,?,?,?,?,?)')
        .run(reviewer,frame.id,frame.sha256,revision+1,now,JSON.stringify(payload));
      this.db.exec('COMMIT');return this.get(reviewer,frame.id);
    }catch(error){this.db.exec('ROLLBACK');throw error;}
  }
  priorityFrames() {
    const rows=this.db.prepare(`SELECT r.frame,r.payload FROM reviews r JOIN
      (SELECT reviewer,frame,MAX(revision) AS revision FROM reviews GROUP BY reviewer,frame) v
      ON r.reviewer=v.reviewer AND r.frame=v.frame AND r.revision=v.revision`).iterate();
    const aggregate=new Map();
    for(const row of rows){const payload=JSON.parse(row.payload);if(payload.trainingConsent!==true)continue;
      const entry=aggregate.get(row.frame)??{visible:false,absent:false,uncertain:false,small:false};
      if(['visible','absent','uncertain'].includes(payload.status))entry[payload.status]=true;
      entry.small ||= payload.boxes?.some(box=>Array.isArray(box.bbox)&&box.bbox[2]-box.bbox[0]<64)===true;aggregate.set(row.frame,entry);}
    return new Map([...aggregate].map(([id,value])=>[id,value.visible&&value.absent?3:value.small?2:value.uncertain?1:0]));
  }
  summary() {
    return {reviews:this.db.prepare('SELECT COUNT(*) AS n FROM (SELECT reviewer,frame FROM reviews GROUP BY reviewer,frame)').get().n,
      reviewedFrames:this.db.prepare('SELECT COUNT(DISTINCT frame) AS n FROM reviews').get().n};
  }
  export() {
    return {schema:'mycoscout.community-reviews.v1',exportedAt:new Date().toISOString(),trainingApproved:false,
      reviews:this.db.prepare(`SELECT r.* FROM reviews r JOIN (SELECT reviewer,frame,MAX(revision) AS revision FROM reviews GROUP BY reviewer,frame) v
        ON r.reviewer=v.reviewer AND r.frame=v.frame AND r.revision=v.revision ORDER BY r.id`).all().map(row=>({
        id:row.id,reviewerId:row.reviewer,frameId:row.frame,sha256:row.sha256,revision:row.revision,createdAt:row.created_at,...JSON.parse(row.payload)}))};
  }
  close(){this.db.close();}
}
