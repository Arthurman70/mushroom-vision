import { DatabaseSync } from 'node:sqlite';
import { mkdir,readFile,realpath,stat } from 'node:fs/promises';
import { resolve,sep } from 'node:path';
import { createHash,randomUUID } from 'node:crypto';
import { ApiError,shortText } from './validation.mjs';
import { COMMUNITY_GROUPS } from '../community/public/research-taxonomy.js';

const ID=/^[A-Za-z0-9_-]{1,160}$/,HASH=/^[a-f0-9]{64}$/;
const rank={unassigned:0,train:1,validation:2,test:3},groups=new Set(COMMUNITY_GROUPS.map(([id])=>id));
const sha=value=>createHash('sha256').update(JSON.stringify(value)).digest('hex');
const fail=(message,code='AI_REVIEW_INVALID')=>{throw new ApiError(message,409,code);};
const note=(value,name='evidence',max=4000)=>shortText(value,name,undefined,max);
const optional=(value,name,max=4000)=>value==null||value===''?'':note(value,name,max);
const identity=frame=>({frameId:frame.id,sourceId:frame.sourceId,sourceGroupId:frame.sourceGroupId,sha256:frame.sha256,
  videoHash:frame.source.videoHash,width:frame.width,height:frame.height,timestampSeconds:frame.timestampSeconds});
const same=(a,b)=>JSON.stringify(a)===JSON.stringify(b);
function coordinates(box,frame){
  if(!Array.isArray(box)||box.length!==4||box.some(n=>typeof n!=='number'||!Number.isFinite(n))||box[0]<0||box[1]<0||box[2]>frame.width||box[3]>frame.height||box[2]<=box[0]||box[3]<=box[1])throw new ApiError('Use a nonempty box inside the original frame, in original pixels.');
  return [...box];
}

// One fixed local experiment, reduced to an explicit public-safe allowlist.
// No source IDs, frame IDs, file paths, model weights or free-form report text
// leave this helper. Reading a report never enables its checkpoint.
export async function readAiCandidateSummary(root){
  const base={schema:'mycoscout.ai-candidate-summary.v1',available:false,humanVerified:false,activationEligible:false,productionModelChanged:false};
  const file=resolve(root,'experiments/results/ai-review-candidate-v1/report.json');let bytes;
  try{
    const actualRoot=await realpath(root),actual=await realpath(file);if(!actual.startsWith(actualRoot+sep))throw new Error('outside');
    if((await stat(actual)).size>1024*1024)throw new Error('size');bytes=await readFile(actual);
  }catch(error){return {...base,status:error.code==='ENOENT'?'not-run':'unavailable',notice:error.code==='ENOENT'?'This station has no completed AI training experiment yet.':'The saved AI experiment report is unavailable.'};}
  try{
    const data=JSON.parse(bytes),count=value=>Number.isSafeInteger(value)&&value>=0&&value<=100000,number=value=>typeof value==='number'&&Number.isFinite(value)&&value>=0;
    if(data.schema!=='mushroom-vision.ai-candidate-run.v1'||data.completed!==true||data.epochs!==1||data.labelAuthority!=='ai-visual-review'||data.humanVerified!==false||data.activationEligible!==false||data.productionModelChanged!==false||data.testEvaluationPerformed!==false||data.checkpointReloadVerified!==true)throw new Error('authority');
    for(const key of ['optimizerSteps','trainFrames','trainSources','trainPositiveFrames','trainBoxes','validationFrames'])if(!count(data[key]))throw new Error('count');
    if(data.optimizerSteps!==data.trainFrames||data.trainFrames<1||data.trainSources<1||data.trainSources>data.trainFrames||data.trainPositiveFrames>data.trainFrames||data.trainBoxes<data.trainPositiveFrames||!number(data.elapsedSeconds)||!Array.isArray(data.losses)||data.losses.length!==data.optimizerSteps||data.losses.some(row=>!number(row?.loss)))throw new Error('consistency');
    const meanLoss=data.losses.reduce((sum,row)=>sum+row.loss,0)/data.losses.length;if(!Number.isFinite(meanLoss))throw new Error('loss');
    return {...base,available:true,status:'completed',epochs:1,optimizerSteps:data.optimizerSteps,trainFrames:data.trainFrames,trainSources:data.trainSources,trainPositiveFrames:data.trainPositiveFrames,trainBoxes:data.trainBoxes,
      validationFrames:data.validationFrames,validationAvailable:data.validationFrames>0&&!!data.validationBefore&&!!data.validationAfter,meanLoss,elapsedSeconds:data.elapsedSeconds,checkpointReloadVerified:true,inactive:true,
      notice:data.validationFrames===0?'One experimental pass completed. No eligible validation frames; recognition improvement has not been measured.':'One experimental pass completed with AI-labeled validation. This is not independent human accuracy.',
      lossNotice:'Training loss describes optimizer fit on these AI labels. It is not recognition accuracy.'};
  }catch{return {...base,status:'unavailable',notice:'The saved AI experiment report did not pass its summary checks.'};}
}

// AI reviews never enter VideoGradesStore. This append-only database has no
// endpoint for changing reviewer authority, promoting labels or activating models.
export class AiReviewStore {
  constructor(directory,{videos,collectionId='video-discovery-v1',partitionsFile,sourceSplits={},holdoutSnapshot=async()=>({})}={}){
    this.directory=resolve(directory);Object.assign(this,{videos,collectionId,partitionsFile,sourceSplits,holdoutSnapshot});this.db=null;
  }
  async initialize(){
    if(!this.videos||!ID.test(this.collectionId))throw new Error('AI review needs a video collection.');
    await mkdir(this.directory,{recursive:true});this.db=new DatabaseSync(resolve(this.directory,'ai-visual-review.sqlite'));
    const version=this.db.prepare('PRAGMA user_version').get().user_version;if(![0,1].includes(version))throw new Error('Unsupported AI review database.');
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS ai_runs(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,data TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS ai_reviews(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,frame_id TEXT NOT NULL,revision INTEGER NOT NULL,data TEXT NOT NULL,UNIQUE(run_id,frame_id,revision));
      CREATE INDEX IF NOT EXISTS ai_review_latest ON ai_reviews(run_id,frame_id,revision);
      CREATE TRIGGER IF NOT EXISTS ai_runs_no_update BEFORE UPDATE ON ai_runs BEGIN SELECT RAISE(ABORT,'Run provenance is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS ai_runs_no_delete BEFORE DELETE ON ai_runs BEGIN SELECT RAISE(ABORT,'Run provenance is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS ai_reviews_no_update BEFORE UPDATE ON ai_reviews BEGIN SELECT RAISE(ABORT,'AI reviews are immutable'); END;
      CREATE TRIGGER IF NOT EXISTS ai_reviews_no_delete BEFORE DELETE ON ai_reviews BEGIN SELECT RAISE(ABORT,'AI reviews are immutable'); END;
      PRAGMA user_version=1;`);return this;
  }
  async policy(collection){
    let assignments=this.sourceSplits;
    if(this.partitionsFile){const data=JSON.parse(await readFile(this.partitionsFile,'utf8'));if(data.schema!=='mycoscout.ai-review-partitions.v1')fail('Unrecognized AI review partition policy.');assignments=data.sourceGroups;}
    if(!assignments||typeof assignments!=='object'||Array.isArray(assignments)||Object.values(assignments).some(value=>!['train','validation','test'].includes(value)))fail('Invalid AI source partitions.');
    const snapshot=await this.holdoutSnapshot(),frozen=snapshot.frozenPartitions??{},blockedHashes=new Set(snapshot.evaluationHashes??[]),evaluationGroups=new Set();
    for(const row of snapshot.evaluationSources??[])for(const key of row.groupKeys??[])evaluationGroups.add(key);
    const frames=collection.frames.map(frame=>{
      const keys=['source:'+frame.sourceGroupId,'video:'+frame.source.videoHash];let split=assignments[frame.sourceGroupId]??'unassigned';
      for(const key of keys)if((rank[frozen[key]]??0)>rank[split])split=frozen[key];
      if(frame.evaluationOnly||blockedHashes.has(frame.sha256)||keys.some(key=>evaluationGroups.has(key)))split='test';
      return {...identity(frame),split,licenseAllowsTraining:frame.source.licenseAllowsTraining===true,evaluationOnly:frame.evaluationOnly,
        source:frame.source,imageUrl:frame.imageUrl,blurScore:frame.blurScore,blurStratum:frame.blurStratum,
        experimentalTrainingEligible:split==='train'&&frame.source.licenseAllowsTraining===true&&!frame.evaluationOnly};
    });
    // Pixel duplicates join whole source/video groups, so another frame from a
    // duplicated clip cannot sneak into training after its twin becomes test.
    const parents=frames.map((_,i)=>i),find=i=>parents[i]===i?i:(parents[i]=find(parents[i])),owners=new Map();
    for(const [i,frame]of frames.entries())for(const key of ['source:'+frame.sourceGroupId,'video:'+frame.videoHash,'pixels:'+frame.sha256]){if(owners.has(key))parents[find(i)]=find(owners.get(key));else owners.set(key,i);}
    const componentSplits=new Map();for(const [i,frame]of frames.entries()){const key=find(i);if(rank[frame.split]>(rank[componentSplits.get(key)]??-1))componentSplits.set(key,frame.split);}
    for(const [i,frame]of frames.entries()){frame.split=componentSplits.get(find(i));frame.experimentalTrainingEligible=frame.experimentalTrainingEligible&&frame.split==='train';}
    return {frames,fingerprint:sha({assignments,frozen,evaluationHashes:[...blockedHashes].sort(),evaluationGroups:[...evaluationGroups].sort()})};
  }
  listRuns(){return this.db.prepare('SELECT data FROM ai_runs ORDER BY created_at DESC LIMIT 100').all().map(row=>{const run=JSON.parse(row.data);return {id:run.id,title:run.title,createdAt:run.createdAt,reviewer:run.reviewer,totalFrames:run.frames.length,reviewedFrames:this.db.prepare('SELECT COUNT(DISTINCT frame_id) AS n FROM ai_reviews WHERE run_id=?').get(run.id).n};});}
  async status(){const collection=await this.videos.read(),policy=await this.policy(collection);return {schema:'mycoscout.ai-review-status.v1',available:collection.available,
    sources:collection.sources.map(source=>({id:source.id,title:source.title,sourceGroupId:source.sourceGroupId,frames:policy.frames.filter(f=>f.sourceId===source.id).length,split:policy.frames.find(f=>f.sourceId===source.id)?.split??'unassigned'})),
    counts:Object.fromEntries(Object.keys(rank).map(split=>[split,policy.frames.filter(f=>f.split===split).length])),runs:this.listRuns(),humanVerified:false,automaticTraining:false,automaticActivation:false};}
  run(id){if(!ID.test(id))throw new ApiError('Unknown AI review run.',404);const row=this.db.prepare('SELECT data FROM ai_runs WHERE id=?').get(id);if(!row)throw new ApiError('Unknown AI review run.',404);return JSON.parse(row.data);}
  getRun(id){const run=this.run(id),reviews=this.db.prepare('SELECT frame_id,data FROM ai_reviews WHERE run_id=? ORDER BY revision').all(id),latest={};for(const row of reviews)latest[row.frame_id]=JSON.parse(row.data);
    return {...run,reviews:latest,summary:{reviewedFrames:Object.keys(latest).length,totalFrames:run.frames.length,statusCounts:Object.fromEntries(['visible','absent','uncertain'].map(status=>[status,Object.values(latest).filter(r=>r.status===status).length]))}};}
  async createRun(input){
    if(!input||typeof input!=='object'||Array.isArray(input))throw new ApiError('Supply an AI review run.');
    if(input.reviewerKind&&input.reviewerKind!=='ai'||input.humanVerified===true)throw new ApiError('This interface records AI reviews only.');
    const reviewer={kind:'ai',name:note(input.reviewer?.name??'Codex visual reviewer','reviewer name',120),model:note(input.reviewer?.model,'model identity',160),version:optional(input.reviewer?.version,'model version',160)};
    const split=input.split??'train';if(!['train','validation','test'].includes(split))throw new ApiError('Choose train, validation or test review scope.');
    const limit=input.limit??18;if(!Number.isInteger(limit)||limit<1||limit>100)throw new ApiError('Choose a finite run of 1–100 frames.');
    const collection=await this.videos.read(),policy=await this.policy(collection);let frames;
    if(input.frameIds!==undefined){if(!Array.isArray(input.frameIds)||!input.frameIds.length||input.frameIds.length>100||new Set(input.frameIds).size!==input.frameIds.length||input.frameIds.some(id=>!ID.test(id)))throw new ApiError('Use 1–100 unique frame IDs.');
      frames=input.frameIds.map(id=>{const f=policy.frames.find(f=>f.frameId===id);if(!f)throw new ApiError('Unknown run frame.');if(f.split!==split)fail('A requested frame belongs to another reserved partition.','AI_REVIEW_PARTITION');return f;});
    }else{
      const sources=[...new Set(policy.frames.filter(f=>f.split===split&&(split!=='train'||f.experimentalTrainingEligible)).map(f=>f.sourceId))].sort();
      const perSource=sources.map(id=>policy.frames.filter(f=>f.sourceId===id&&f.split===split).sort((a,b)=>a.timestampSeconds-b.timestampSeconds));frames=[];
      for(let round=0;frames.length<limit&&round<100;round++)for(const list of perSource){const slots=Math.min(list.length,Math.ceil(limit/Math.max(1,sources.length))),index=Math.min(list.length-1,Math.floor((round+.5)*list.length/slots));if(round<slots&&frames.length<limit)frames.push(list[index]);}
    }
    if(!frames.length)throw new ApiError('No frames are eligible for this review scope.',409,'AI_REVIEW_EMPTY');
    if(split==='train'&&frames.some(f=>!f.experimentalTrainingEligible))fail('A training review cannot contain a held-out or unlicensed frame.','AI_REVIEW_PARTITION');
    // Check the actual bytes before freezing a run; no arbitrary paths enter the API.
    for(const frame of frames){const original=await this.videos.frame(frame.frameId);if(!same(identity(original.metadata),Object.fromEntries(Object.keys(identity(original.metadata)).map(key=>[key,frame[key]]))))fail('Run frame identity changed.','AI_REVIEW_INTEGRITY');}
    const run={schema:'mycoscout.ai-visual-review-run.v1',id:'ai-'+randomUUID(),createdAt:new Date().toISOString(),collectionId:this.collectionId,title:note(input.title??'Independent AI visual review','title',200),reviewer,reviewerKind:'ai',humanVerified:false,
      split,policyFingerprint:policy.fingerprint,selection:'Fixed original-frame queue; no model scores used by the default selector.',protocol:'Inspect original pixels independently. Record uncertainty. Temporal evidence never creates earlier-frame visibility.',frames,automaticTraining:false,automaticActivation:false};
    if(this.db.prepare('SELECT COUNT(*) AS n FROM ai_runs').get().n>=1000)fail('AI run capacity reached.');
    this.db.prepare('INSERT INTO ai_runs(id,created_at,data) VALUES(?,?,?)').run(run.id,run.createdAt,JSON.stringify(run));return this.getRun(run.id);
  }
  async saveReview(runId,frameId,input){
    const run=this.run(runId),frame=run.frames.find(f=>f.frameId===frameId);if(!frame)throw new ApiError('Frame is not in this AI review run.',404);
    if(!input||typeof input!=='object'||Array.isArray(input)||input.humanVerified===true||input.reviewerKind&&input.reviewerKind!=='ai')throw new ApiError('This interface stores AI candidates, never human confirmations.');
    const {metadata}=await this.videos.frame(frameId);if(!same(identity(metadata),Object.fromEntries(Object.keys(identity(metadata)).map(key=>[key,frame[key]])))||input.sha256!==frame.sha256||input.videoHash!==frame.videoHash)fail('Review hashes do not match the frozen original.','AI_REVIEW_INTEGRITY');
    if(input.viewedOriginal!==true)throw new ApiError('Inspect the original frame before submitting an AI review.');
    if(typeof input.modelSuggestionsViewed!=='boolean')throw new ApiError('Record whether model suggestions were viewed.');
    const status=({yes:'visible',no:'absent',uncertain:'uncertain'})[input.visibility]??input.status;if(!['visible','absent','uncertain'].includes(status))throw new ApiError('Choose visible, absent or uncertain.');
    if(typeof input.complete!=='boolean'||status==='uncertain'&&input.complete)throw new ApiError('Uncertain reviews cannot claim complete labeling.');
    if(!Array.isArray(input.boxes)||input.boxes.length>128||status!=='visible'&&input.boxes.length||status==='visible'&&input.complete&&!input.boxes.length)throw new ApiError('Only visible frames can have boxes; complete visible reviews need a box.');
    const boxes=input.boxes.map((box,index)=>{if(!box||typeof box!=='object'||!groups.has(box.groupId??'unknown')||!Number.isFinite(box.confidence)||box.confidence<0||box.confidence>1)throw new ApiError('Each box needs a supported broad group and confidence from zero to one.');
      return {id:'box-'+(index+1),bboxPixelsXYXY:coordinates(box.bboxPixelsXYXY,frame),groupId:box.groupId??'unknown',label:optional(box.label,'label',160)||'Mushroom candidate',confidence:box.confidence,evidence:note(box.evidence,'box evidence'),reviewerKind:'ai',humanVerified:false};});
    const crops=input.viewedCrops??[];if(!Array.isArray(crops)||crops.length>128)throw new ApiError('Use at most 128 crop views.');const viewedCrops=crops.map(crop=>coordinates(crop,frame));
    const temporal=input.temporalEvidence??[];if(!Array.isArray(temporal)||temporal.length>20)throw new ApiError('Use at most 20 temporal evidence links.');const temporalEvidence=[];
    for(const item of temporal){if(!item||!ID.test(item.frameId)||item.frameId===frameId||!HASH.test(item.sha256??''))throw new ApiError('Invalid temporal evidence.');const other=await this.videos.frame(item.frameId);if(other.metadata.sourceId!==frame.sourceId||other.metadata.source.videoHash!==frame.videoHash||other.metadata.sha256!==item.sha256)fail('Temporal evidence must match this exact source video.','AI_REVIEW_TEMPORAL');temporalEvidence.push({frameId:item.frameId,sha256:item.sha256,note:note(item.note,'temporal evidence'),inferredVisibility:false});}
    if(!Number.isInteger(input.baseRevision)||input.baseRevision<0)throw new ApiError('Supply the current review revision.');
    const review={schema:'mycoscout.ai-visual-review.v1',id:'air-'+randomUUID(),runId,frameId,sourceId:frame.sourceId,sourceGroupId:frame.sourceGroupId,sha256:frame.sha256,videoHash:frame.videoHash,width:frame.width,height:frame.height,
      reviewer:run.reviewer,reviewerKind:'ai',humanVerified:false,trainingAllowed:false,split:frame.split,status,visibility:({visible:'yes',absent:'no',uncertain:'uncertain'})[status],complete:input.complete,
      boxes,evidence:note(input.evidence,'frame evidence'),viewedOriginal:true,viewedCrops,modelSuggestionsViewed:input.modelSuggestionsViewed,temporalEvidence,automaticPropagation:false,revision:input.baseRevision+1,createdAt:new Date().toISOString()};
    this.db.exec('BEGIN IMMEDIATE');try{const current=this.db.prepare('SELECT MAX(revision) AS n FROM ai_reviews WHERE run_id=? AND frame_id=?').get(runId,frameId).n??0;
      if(current!==input.baseRevision)fail('A newer AI review exists. Reload before saving.','AI_REVIEW_CONFLICT');if(current>=100)fail('Review history capacity reached.');
      this.db.prepare('INSERT INTO ai_reviews(id,run_id,frame_id,revision,data) VALUES(?,?,?,?,?)').run(review.id,runId,frameId,review.revision,JSON.stringify(review));this.db.exec('COMMIT');return review;
    }catch(error){this.db.exec('ROLLBACK');throw error;}
  }
  history(runId,frameId){this.run(runId);return this.db.prepare('SELECT data FROM ai_reviews WHERE run_id=? AND frame_id=? ORDER BY revision DESC').all(runId,frameId).map(row=>JSON.parse(row.data));}
  async exportRun(runId){
    const run=this.getRun(runId),policy=await this.policy(await this.videos.read()),records=[],excluded=[];
    for(const frame of run.frames){const review=run.reviews[frame.frameId],current=policy.frames.find(f=>f.frameId===frame.frameId);let reason='';
      if(!current||['frameId','sourceId','sourceGroupId','sha256','videoHash','width','height','timestampSeconds'].some(key=>current[key]!==frame[key]))reason='source_changed';
      else if(current.split==='test'||frame.split==='test')reason='held_out_test';
      else if(current.split!==frame.split||!['train','validation'].includes(frame.split))reason='partition_changed_or_unassigned';
      else if(!current.licenseAllowsTraining||current.evaluationOnly)reason='rights_or_evaluation_only';
      else if(!review)reason='unreviewed';else if(!review.complete||review.status==='uncertain')reason='incomplete_or_uncertain';
      if(reason){excluded.push({frameId:frame.frameId,reason});continue;}
      const {metadata}=await this.videos.frame(frame.frameId);if(metadata.sha256!==frame.sha256)fail('Export frame identity changed.');
      records.push({...review,collectionId:run.collectionId,groupKeys:['source:'+frame.sourceGroupId,'video:'+frame.videoHash],sourceCredits:frame.source,
        originalSha256:frame.sha256,imageUrl:frame.imageUrl,labelAuthority:'ai-visual-review',trainingAllowed:false,experimentalTrainingAllowed:frame.split==='train',automaticActivation:false});
    }
    return {schema:'mycoscout.ai-visual-review-export.v1',createdAt:new Date().toISOString(),runId,reviewer:run.reviewer,reviewerKind:'ai',humanVerified:false,
      labelAuthority:'ai-visual-review',automaticActivation:false,productionTrainingAllowed:false,policyFingerprint:policy.fingerprint,runPolicyFingerprint:run.policyFingerprint,records,excluded,
      notice:'AI annotations are experimental candidates. They do not enter human training automatically. Test sources are excluded; validation is for comparison only.'};
  }
  async dispatch(method,path,input){
    if(method==='GET'&&path==='/api/ai-review/status')return {status:200,data:await this.status()};
    if(method==='POST'&&path==='/api/ai-review/runs')return {status:201,data:await this.createRun(input)};
    const route=/^\/api\/ai-review\/runs\/([A-Za-z0-9_-]{1,160})(?:\/(export|reviews)(?:\/([A-Za-z0-9_-]{1,160}))?)?$/.exec(path);if(!route)return null;
    if(method==='GET'&&!route[2])return {status:200,data:this.getRun(route[1])};
    if(method==='GET'&&route[2]==='export'&&!route[3])return {status:200,data:await this.exportRun(route[1])};
    if(route[2]==='reviews'&&route[3]){if(method==='POST')return {status:201,data:await this.saveReview(route[1],route[3],input)};if(method==='GET')return {status:200,data:{history:this.history(route[1],route[3])}};}
    return null;
  }
  close(){this.db?.close();this.db=null;}
}
