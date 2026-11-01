import { DatabaseSync } from 'node:sqlite';
import { randomBytes, createHash } from 'node:crypto';
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { ReviewError, GROUPS } from './store.mjs';
import { normalizeTaxonCandidate } from './public/research-taxonomy.js';
import { validConfounder } from './public/review-evidence.js';

const hash = value => createHash('sha256').update(value).digest('hex');
const hex = /^[a-f0-9]{64}$/;
const identifier = /^[a-zA-Z0-9_-]{16,128}$/;
const notice = 'By submitting AI annotations, you permit Great Lakes Gleaner and Mushroom Vision to publish and use your annotations to train, evaluate and improve mushroom-vision software, including commercial software. Source images retain their original licenses. Submissions must truthfully identify AI authorship and whether the image was viewed. Opinions remain unverified and require curator review; submission does not activate training or models. Do not include private information in model names or notes.';
export const AI_TERMS = Object.freeze({version:'ai-annotations-2026-10-31-v1',notice,noticeSha256:hash(notice)});
export const AI_LIMITS = Object.freeze({credentialHours:24,leaseMinutes:30,maxOutstandingTasks:1,maxContributionsPerDay:100,maxBoxes:64,maxBodyBytes:32768});
const cleanText=(value,max,name,required=true)=>{
  if(typeof value!=='string'||value.length>max||/[\u0000-\u001f\u007f]/.test(value)||(required&&!value.trim()))throw new ReviewError(`Invalid ${name}.`);
  return value.trim();
};
function objectKeys(value,allowed,name){
  if(!value||typeof value!=='object'||Array.isArray(value)||Object.keys(value).some(key=>!allowed.includes(key)))throw new ReviewError(`Unsupported ${name} fields.`);
}
function terms(value){if(value?.version!==AI_TERMS.version||value?.noticeSha256!==AI_TERMS.noticeSha256)throw new ReviewError('Load and accept the current AI participation terms.',428);}
function coords(b,frame){
  if(!Array.isArray(b)||b.length!==4||!b.every(n=>typeof n==='number'&&Number.isFinite(n))||b[0]<0||b[1]<0||b[2]>frame.width||b[3]>frame.height||b[2]<=b[0]||b[3]<=b[1])throw new ReviewError('Boxes must be xyxy pixel coordinates inside the original image.');
  return [...b];
}
function confidence(value){if(typeof value!=='number'||!Number.isFinite(value)||value<0||value>1)throw new ReviewError('Confidence must be between zero and one.');return value;}
const safeUrl=value=>{try{const u=new URL(value);return ['http:','https:'].includes(u.protocol)&&!u.username&&!u.password;}catch{return false;}};
export function aiSourceAllowed(frame,source){
  const license=String(source?.license??'').toLowerCase().replaceAll('-',' ');
  return frame?.sourceLicenseAllowsTraining===true&&typeof frame.evaluationOnly==='boolean'&&
    /^cc by(?: sa)?(?: \d+(?:\.\d+)?)?(?: \(youtube\))?$|^cc0(?: 1\.0)?$|^cdla permissive 1\.0$/.test(license)&&
    typeof source?.attribution==='string'&&source.attribution.trim().length>0&&safeUrl(source.sourceUrl)&&safeUrl(source.licenseUrl);
}
export function aiEligibleFrames(frames,sources){
  const splits=new Map();
  for(const frame of frames){
    const group=String(frame.sourceGroupId??frame.sourceId),values=splits.get(group)??new Set();
    values.add(frame.evaluationOnly);splits.set(group,values);
  }
  // Contradictory source-level splits are withheld, never silently relabeled as training.
  return frames.filter(frame=>splits.get(String(frame.sourceGroupId??frame.sourceId))?.size===1&&aiSourceAllowed(frame,sources.get(frame.sourceId)));
}
export function validateAIContribution(input,task,createdAt){
  objectKeys(input,['taskId','imageSha256','reviewerKind','status','complete','boxes','confounders','observation','participationTerms'],'contribution');
  if(input.reviewerKind!=='ai')throw new ReviewError('This endpoint accepts explicitly AI-authored opinions only.');
  if(input.taskId!==task.taskId||input.imageSha256!==task.imageSha256)throw new ReviewError('Task or image hash changed.',409);
  terms(input.participationTerms);
  if(!['visible','absent','uncertain'].includes(input.status)||typeof input.complete!=='boolean')throw new ReviewError('Choose visible, absent or uncertain and declare completeness.');
  if(!Array.isArray(input.boxes)||input.boxes.length>64||!Array.isArray(input.confounders)||input.confounders.length>64)throw new ReviewError('Use at most 64 mushroom and 64 confounder boxes.');
  if(input.status!=='visible'&&input.boxes.length||input.status==='visible'&&!input.boxes.length||input.status==='uncertain'&&input.complete)throw new ReviewError('Visible needs boxes; uncertain cannot be complete; absent cannot have mushroom boxes.');
  const observation=input.observation;
  objectKeys(observation,['imageViewed','mode','imageSha256','note'],'observation');
  if(observation.imageViewed!==true||observation.mode!=='original-image'||observation.imageSha256!==task.imageSha256)throw new ReviewError('View this original image and truthfully attest to its exact hash.');
  const boxes=input.boxes.map(box=>{
    objectKeys(box,['bbox','group','confidence','taxonCandidate'],'mushroom box');
    if(!GROUPS.includes(box.group))throw new ReviewError('Use a supported group or unknown.');
    let taxonCandidate;try{taxonCandidate=normalizeTaxonCandidate(box.taxonCandidate,{groupId:box.group});}catch(error){throw new ReviewError(error.message);}
    if(taxonCandidate&&taxonCandidate.evidenceSource!=='image_only')throw new ReviewError('This task provides image-only evidence.');
    return {bbox:coords(box.bbox,task),group:box.group,confidence:confidence(box.confidence),...(taxonCandidate?{taxonCandidate}:{})};
  });
  const confounders=input.confounders.map(item=>{
    objectKeys(item,['bbox','kind','confidence'],'confounder');
    if(!validConfounder(item.kind))throw new ReviewError('Use a supported confounder kind.');
    return {bbox:coords(item.bbox,task),kind:item.kind,confidence:confidence(item.confidence),explicitNonFungal:true};
  });
  return {schema:'mushroom-vision.ai-contribution.v1',reviewerKind:'ai',status:input.status,complete:input.complete,boxes,confounders,
    observation:{imageViewed:true,mode:'original-image',imageSha256:task.imageSha256,note:cleanText(observation.note,1200,'visual evidence'),verification:'agent-self-attestation-with-server-image-delivery'},
    confidenceMeaning:'agent-reported-not-calibrated',labelAuthority:'ai-unverified',reviewStatus:'quarantined',humanVerified:false,trainingApproved:false,
    participationTerms:{version:AI_TERMS.version,noticeSha256:AI_TERMS.noticeSha256,acceptedAt:createdAt,acceptance:'submission'}};
}

/** A separate append-only inbox. Human review and training tables are never written here. */
export class AIContributionStore {
  constructor(directory){
    mkdirSync(directory,{recursive:true});this.db=new DatabaseSync(resolve(directory,'ai-contributions.sqlite'));
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS ai_agents(token_hash TEXT PRIMARY KEY, model TEXT NOT NULL, expires INTEGER NOT NULL, created_at TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS ai_tasks(id TEXT PRIMARY KEY, agent TEXT NOT NULL, frame TEXT NOT NULL, sha256 TEXT NOT NULL, expires INTEGER NOT NULL, payload TEXT NOT NULL, delivered_at TEXT);
      CREATE INDEX IF NOT EXISTS ai_tasks_agent ON ai_tasks(agent,expires);
      CREATE TABLE IF NOT EXISTS ai_contributions(id TEXT PRIMARY KEY, agent TEXT NOT NULL, task TEXT NOT NULL UNIQUE, frame TEXT NOT NULL, sha256 TEXT NOT NULL,
        idem_key TEXT NOT NULL, input_hash TEXT NOT NULL, created_at TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(agent,idem_key));
      CREATE INDEX IF NOT EXISTS ai_contribution_frame ON ai_contributions(frame,created_at);
      CREATE TRIGGER IF NOT EXISTS ai_contribution_immutable_update BEFORE UPDATE ON ai_contributions BEGIN SELECT RAISE(ABORT,'AI history is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS ai_contribution_immutable_delete BEFORE DELETE ON ai_contributions BEGIN SELECT RAISE(ABORT,'AI history is immutable'); END;`);
  }
  register(input,now=Date.now()){
    objectKeys(input,['reviewerKind','model','participationTerms'],'registration');
    if(input.reviewerKind!=='ai')throw new ReviewError('Register AI authorship explicitly.');terms(input.participationTerms);
    objectKeys(input.model,['provider','name','version'],'model');
    const model=Object.fromEntries(['provider','name','version'].map(key=>[key,cleanText(input.model[key],100,`model ${key}`)]));
    this.db.prepare('DELETE FROM ai_agents WHERE expires<?').run(now);
    if(this.db.prepare('SELECT COUNT(*) AS n FROM ai_agents').get().n>=20000)throw new ReviewError('AI registration capacity reached.',503);
    const token=randomBytes(32).toString('hex'),expires=now+24*3600000;
    this.db.prepare('INSERT INTO ai_agents VALUES(?,?,?,?)').run(hash(token),JSON.stringify(model),expires,new Date(now).toISOString());
    return {token,expiresAt:new Date(expires).toISOString(),reviewerKind:'ai',model,participationTerms:AI_TERMS};
  }
  authenticate(token,now=Date.now()){
    if(!hex.test(token??''))throw new ReviewError('Use a valid AI Bearer credential.',401);
    const row=this.db.prepare('SELECT * FROM ai_agents WHERE token_hash=? AND expires>?').get(hash(token),now);
    if(!row)throw new ReviewError('AI credential expired or is invalid.',401);
    return {id:row.token_hash,model:JSON.parse(row.model)};
  }
  task(agent,id,now=Date.now(),{allowExpired=false}={}){
    if(!identifier.test(id??''))throw new ReviewError('Task not found.',404);
    const row=this.db.prepare('SELECT * FROM ai_tasks WHERE id=? AND agent=?').get(id,agent.id);
    if(!row)throw new ReviewError('Task not found.',404);
    if(!allowExpired&&row.expires<=now)throw new ReviewError('Task lease expired. Request another task.',410);
    return {...JSON.parse(row.payload),deliveredAt:row.delivered_at};
  }
  lease(agent,frames,sources,base,now=Date.now()){
    const existing=this.db.prepare('SELECT t.id FROM ai_tasks t LEFT JOIN ai_contributions c ON c.task=t.id WHERE t.agent=? AND t.expires>? AND c.id IS NULL LIMIT 1').get(agent.id,now);
    if(existing)return this.task(agent,existing.id,now);
    const daily=this.db.prepare('SELECT COUNT(*) AS n FROM ai_contributions WHERE agent=? AND created_at>?').get(agent.id,new Date(now-86400000).toISOString()).n;
    if(daily>=AI_LIMITS.maxContributionsPerDay)throw new ReviewError('Daily AI contribution limit reached.',429);
    // Keep submitted task records for audit/idempotency; expired unused leases can be collected.
    this.db.prepare('DELETE FROM ai_tasks WHERE expires<? AND id NOT IN (SELECT task FROM ai_contributions)').run(now);
    if(this.db.prepare('SELECT COUNT(*) AS n FROM ai_tasks').get().n>=250000)throw new ReviewError('AI task capacity reached.',503);
    const counts=new Map(this.db.prepare('SELECT frame,COUNT(*) AS n FROM ai_contributions GROUP BY frame').all().map(x=>[x.frame,x.n]));
    const seen=new Set(this.db.prepare('SELECT frame FROM ai_contributions WHERE agent=?').all(agent.id).map(x=>x.frame));
    const eligible=aiEligibleFrames(frames,sources).filter(frame=>!seen.has(frame.id))
      .sort((a,b)=>(counts.get(a.id)??0)-(counts.get(b.id)??0)||a.id.localeCompare(b.id));
    if(!eligible.length)return null;
    const frame=eligible[0],source=sources.get(frame.sourceId),taskId=randomBytes(24).toString('hex'),expires=now+30*60000;
    const task={taskId,frameId:frame.id,imageSha256:frame.sha256,width:frame.width,height:frame.height,sourceId:frame.sourceId,
      sourceGroupId:'public-group-'+hash(String(frame.sourceGroupId??frame.sourceId)).slice(0,24),
      split:frame.evaluationOnly?'evaluation_only':'train_candidate',evaluationOnly:frame.evaluationOnly,
      imageUrl:base+'api/ai/tasks/'+taskId+'/image',expiresAt:new Date(expires).toISOString(),
      source:{title:String(source.title??'Licensed research image').slice(0,200),license:source.license,licenseUrl:source.licenseUrl,sourceUrl:source.sourceUrl,attribution:source.attribution,changes:String(source.changes??'').slice(0,600)},
      trainingApproved:false,humanVerified:false};
    this.db.prepare('INSERT INTO ai_tasks VALUES(?,?,?,?,?,?,NULL)').run(taskId,agent.id,frame.id,frame.sha256,expires,JSON.stringify(task));return task;
  }
  markDelivered(agent,id,sha256,now=Date.now()){
    const task=this.task(agent,id,now);if(task.imageSha256!==sha256)throw new ReviewError('Image hash changed.',409);
    this.db.prepare('UPDATE ai_tasks SET delivered_at=COALESCE(delivered_at,?) WHERE id=? AND agent=?').run(new Date(now).toISOString(),id,agent.id);
  }
  submit(agent,input,idempotencyKey,frame,source,requestId,now=Date.now()){
    if(!identifier.test(idempotencyKey??''))throw new ReviewError('Supply an Idempotency-Key of 16–128 letters, digits, underscores or hyphens.');
    const inputHash=hash(JSON.stringify(input));
    const old=this.db.prepare('SELECT * FROM ai_contributions WHERE agent=? AND idem_key=?').get(agent.id,idempotencyKey);
    if(old){if(old.input_hash!==inputHash)throw new ReviewError('Idempotency key already used with different content.',409);return {duplicate:true,contribution:JSON.parse(old.payload)};}
    const task=this.task(agent,input?.taskId,now);
    if(!frame||frame.sha256!==task.imageSha256||frame.sourceId!==task.sourceId||frame.evaluationOnly!==task.evaluationOnly||
      frame.width!==task.width||frame.height!==task.height||'public-group-'+hash(String(frame.sourceGroupId??frame.sourceId)).slice(0,24)!==task.sourceGroupId||
      !aiSourceAllowed(frame,source)||['license','licenseUrl','sourceUrl','attribution'].some(key=>source[key]!==task.source[key]))throw new ReviewError('The task image, split or publication rights changed.',409);
    if(!task.deliveredAt)throw new ReviewError('Retrieve and view the task original through its authenticated image URL before submitting.',428);
    if(this.db.prepare('SELECT id FROM ai_contributions WHERE task=?').get(task.taskId))throw new ReviewError('This task already has an immutable submission.',409);
    if(this.db.prepare('SELECT COUNT(*) AS n FROM ai_contributions').get().n>=250000)throw new ReviewError('AI inbox capacity reached.',503);
    const createdAt=new Date(now).toISOString(),payload=validateAIContribution(input,task,createdAt),id=randomBytes(24).toString('hex');
    const contribution={id,requestId,createdAt,taskId:task.taskId,frameId:task.frameId,imageSha256:task.imageSha256,
      sourceId:task.sourceId,sourceGroupId:task.sourceGroupId,source:task.source,split:task.split,evaluationOnly:task.evaluationOnly,
      imageWidth:task.width,imageHeight:task.height,model:agent.model,modelIdentity:'agent-reported',imageDeliveredAt:task.deliveredAt,...payload};
    this.db.prepare('INSERT INTO ai_contributions VALUES(?,?,?,?,?,?,?,?,?)').run(id,agent.id,task.taskId,task.frameId,task.imageSha256,idempotencyKey,inputHash,createdAt,JSON.stringify(contribution));
    return {duplicate:false,contribution};
  }
  get(agent,id){
    if(!identifier.test(id??''))throw new ReviewError('Contribution not found.',404);
    const row=this.db.prepare('SELECT payload FROM ai_contributions WHERE id=? AND agent=?').get(id,agent.id);
    if(!row)throw new ReviewError('Contribution not found.',404);return JSON.parse(row.payload);
  }
  publicReviews(frame,sha256){
    return this.db.prepare('SELECT payload FROM ai_contributions WHERE frame=? AND sha256=? ORDER BY created_at DESC LIMIT 20').all(frame,sha256).map(row=>{
      const {taskId,requestId,imageDeliveredAt,...safe}=JSON.parse(row.payload);return safe;
    });
  }
  summary(){return {contributions:this.db.prepare('SELECT COUNT(*) AS n FROM ai_contributions').get().n,
    frames:this.db.prepare('SELECT COUNT(DISTINCT frame) AS n FROM ai_contributions').get().n,reviewStatus:'quarantined',trainingApproved:false,humanVerified:false};}
  export(){return {schema:'mushroom-vision.ai-contribution-export.v1',exportedAt:new Date().toISOString(),reviewerKind:'ai',trainingApproved:false,humanVerified:false,
    contributions:this.db.prepare('SELECT agent,payload FROM ai_contributions ORDER BY created_at,id').all().map(row=>({agentPseudonym:row.agent,...JSON.parse(row.payload)}))};}
  close(){this.db.close();}
}
