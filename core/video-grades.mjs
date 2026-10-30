import { DatabaseSync } from 'node:sqlite';
import { mkdir } from 'node:fs/promises';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { ApiError,shortText } from './validation.mjs';
import { normalizeTaxonCandidate } from '../community/public/research-taxonomy.js';

const ID=/^[A-Za-z0-9_-]{1,160}$/,HASH=/^[a-f0-9]{64}$/;
const statuses=['visible','absent','uncertain'];
const conflict=()=>{throw new ApiError('This frame or source changed. Keep the original collection and reload before grading.',409,'VIDEO_GRADE_INTEGRITY');};
function context(collectionId,frame){
  const source=frame?.source;
  if(!ID.test(collectionId)||!ID.test(frame?.id)||!ID.test(source?.id)||frame.sourceId!==source.id||!HASH.test(frame.sha256)||!HASH.test(source.videoHash)||
    !Number.isInteger(frame.width)||!Number.isInteger(frame.height)||frame.width<1||frame.height<1||!Number.isFinite(frame.timestampSeconds)||frame.timestampSeconds<0)
    throw new ApiError('Invalid video grading frame.',409,'VIDEO_GRADE_INTEGRITY');
  return {collectionId,frameId:frame.id,sha256:frame.sha256,width:frame.width,height:frame.height,sourceId:source.id,videoHash:source.videoHash,
    sourceGroupId:source.sourceGroupId,timestampSeconds:frame.timestampSeconds};
}
const optionalText=(value,name,max)=>value==null||value===''?'':shortText(value,name,undefined,max);
const widthBin=width=>width<4?'under4':width<8?'4to8':width<16?'8to16':width<32?'16to32':width<64?'32to64':'64plus';

// Reviews describe exactly one original frame. Track links are human assertions;
// neither neighboring frames nor model guesses become labels automatically.
export class VideoGradesStore {
  constructor(directory){this.directory=resolve(directory);this.db=null;}
  async initialize(){
    await mkdir(this.directory,{recursive:true});this.db=new DatabaseSync(resolve(this.directory,'video-grades.sqlite'));
    const version=this.db.prepare('PRAGMA user_version').get().user_version;
    if(![0,1].includes(version))throw new Error('Unsupported video grading database version');
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS grade_frames(collection_id TEXT NOT NULL,frame_id TEXT NOT NULL,context TEXT NOT NULL,PRIMARY KEY(collection_id,frame_id));
      CREATE TABLE IF NOT EXISTS grade_reviews(id TEXT PRIMARY KEY,collection_id TEXT NOT NULL,frame_id TEXT NOT NULL,revision INTEGER NOT NULL,created_at TEXT NOT NULL,data TEXT NOT NULL,
        FOREIGN KEY(collection_id,frame_id) REFERENCES grade_frames(collection_id,frame_id),UNIQUE(collection_id,frame_id,revision));
      CREATE TABLE IF NOT EXISTS grade_tracks(id TEXT PRIMARY KEY,collection_id TEXT NOT NULL,source_id TEXT NOT NULL,video_hash TEXT NOT NULL,created_at TEXT NOT NULL,data TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS grade_latest ON grade_reviews(collection_id,frame_id,revision);
      CREATE INDEX IF NOT EXISTS grade_source_tracks ON grade_tracks(collection_id,source_id);
      CREATE TRIGGER IF NOT EXISTS grade_reviews_no_update BEFORE UPDATE ON grade_reviews BEGIN SELECT RAISE(ABORT,'Reviews are immutable'); END;
      CREATE TRIGGER IF NOT EXISTS grade_reviews_no_delete BEFORE DELETE ON grade_reviews BEGIN SELECT RAISE(ABORT,'Reviews are immutable'); END;
      CREATE TRIGGER IF NOT EXISTS grade_frames_no_update BEFORE UPDATE ON grade_frames BEGIN SELECT RAISE(ABORT,'Frame provenance is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS grade_tracks_no_update BEFORE UPDATE ON grade_tracks BEGIN SELECT RAISE(ABORT,'Tracks are immutable'); END;
      PRAGMA user_version=1;`);
    return this;
  }
  transaction(fn){this.db.exec('BEGIN IMMEDIATE');try{const result=fn();this.db.exec('COMMIT');return result;}catch(error){this.db.exec('ROLLBACK');throw error;}}
  check(collectionId,frame){const ctx=context(collectionId,frame),stored=this.db.prepare('SELECT context FROM grade_frames WHERE collection_id=? AND frame_id=?').get(collectionId,frame.id);
    if(stored&&stored.context!==JSON.stringify(ctx))conflict();return ctx;}
  get(collectionId,frame,{history=false}={}){
    this.check(collectionId,frame);const rows=this.db.prepare('SELECT * FROM grade_reviews WHERE collection_id=? AND frame_id=? ORDER BY revision DESC').all(collectionId,frame.id);
    const decode=row=>({...JSON.parse(row.data),id:row.id,revision:row.revision,createdAt:row.created_at});
    return {frameId:frame.id,revision:rows[0]?.revision??0,review:rows[0]?decode(rows[0]):null,historyCount:rows.length,...(history?{history:rows.map(decode)}:{})};
  }
  createTrack(collectionId,source,input){
    if(!ID.test(collectionId)||!ID.test(source?.id)||!HASH.test(source.videoHash))throw new ApiError('Invalid video track source.');
    const label=shortText(input.label,'label',undefined,160),reviewer=shortText(input.reviewer,'reviewer',undefined,120);
    if(this.db.prepare('SELECT COUNT(*) AS n FROM grade_tracks WHERE collection_id=? AND source_id=?').get(collectionId,source.id).n>=128)
      throw new ApiError('This video already has the maximum of 128 tracks.',409,'VIDEO_TRACK_CAPACITY');
    const track={id:randomUUID(),collectionId,sourceId:source.id,videoHash:source.videoHash,sourceGroupId:source.sourceGroupId,label,reviewer,createdAt:new Date().toISOString(),humanCreated:true};
    this.db.prepare('INSERT INTO grade_tracks(id,collection_id,source_id,video_hash,created_at,data) VALUES(?,?,?,?,?,?)')
      .run(track.id,collectionId,source.id,source.videoHash,track.createdAt,JSON.stringify(track));return track;
  }
  listTracks(collectionId,sourceId=null){
    if(!ID.test(collectionId)||(sourceId!==null&&!ID.test(sourceId)))throw new ApiError('Invalid video track query.');
    const rows=sourceId===null?this.db.prepare('SELECT data FROM grade_tracks WHERE collection_id=? ORDER BY created_at,id').all(collectionId):
      this.db.prepare('SELECT data FROM grade_tracks WHERE collection_id=? AND source_id=? ORDER BY created_at,id').all(collectionId,sourceId);
    return rows.map(row=>JSON.parse(row.data));
  }
  save(collectionId,frame,input){
    const ctx=this.check(collectionId,frame);
    if(!Number.isInteger(input.baseRevision)||input.baseRevision<0||!statuses.includes(input.status))throw new ApiError('Choose visible, absent or uncertain and supply the current review revision.');
    if(!Array.isArray(input.boxes)||input.boxes.length>128)throw new ApiError('Supply at most 128 human boxes.');
    for(const key of ['complete','trainingAllowed'])if(input[key]!==undefined&&typeof input[key]!=='boolean')throw new ApiError(key+' must be a boolean.');
    if(input.status!=='visible'&&input.boxes.length)throw new ApiError('Only visible frames can contain verified mushroom boxes.');
    if(input.complete===true&&input.status==='visible'&&!input.boxes.length)throw new ApiError('A completely labeled visible frame needs at least one box.');
    if(input.complete===true&&input.status==='uncertain')throw new ApiError('An uncertain frame cannot be completely labeled.');
    const ids=new Set(),trackIds=new Set(),boxes=input.boxes.map(box=>{
      if(!box||typeof box!=='object'||Array.isArray(box))throw new ApiError('Each human box must be an object.');
      const id=box.id??randomUUID(),bbox=box.bboxPixelsXYXY;
      if(!ID.test(id)||ids.has(id)||!Array.isArray(bbox)||bbox.length!==4||bbox.some(x=>typeof x!=='number'||!Number.isFinite(x))||
        bbox[0]<0||bbox[1]<0||bbox[2]>ctx.width||bbox[3]>ctx.height||bbox[2]<=bbox[0]||bbox[3]<=bbox[1])throw new ApiError('Each box needs a unique ID and valid coordinates inside the original frame.');
      ids.add(id);let trackId=null;
      if(box.trackId!=null&&box.trackId!==''){
        if(typeof box.trackId!=='string'||box.trackId.length>160)throw new ApiError('Invalid track ID.');
        const track=this.db.prepare('SELECT * FROM grade_tracks WHERE id=? AND collection_id=?').get(box.trackId,collectionId);
        if(!track||track.source_id!==ctx.sourceId||track.video_hash!==ctx.videoHash)throw new ApiError('A track must belong to this exact source video.',409,'VIDEO_TRACK_SOURCE_MISMATCH');
        if(trackIds.has(box.trackId))throw new ApiError('A mushroom track can appear only once in a frame.');trackId=box.trackId;trackIds.add(trackId);
      }
      const groupId=optionalText(box.groupId,'groupId',80)||'unknown';
      let taxonCandidate;
      try{taxonCandidate=normalizeTaxonCandidate(box.taxonCandidate,{groupId});}catch(error){throw new ApiError(error.message);}
      return {id,bboxPixelsXYXY:[...bbox],pixelWidth:bbox[2]-bbox[0],pixelHeight:bbox[3]-bbox[1],trackId,
        groupId,label:optionalText(box.label,'label',160),humanVerified:true,...(taxonCandidate?{taxonCandidate}:{})};
    });
    const assistance=input.assistance??{aiAssisted:false,communityReviewIds:[]};
    if(typeof assistance.aiAssisted!=='boolean'||!Array.isArray(assistance.communityReviewIds)||assistance.communityReviewIds.length>100||assistance.communityReviewIds.some(id=>!Number.isSafeInteger(id)||id<1))throw new ApiError('Invalid assistance provenance.');
    const review={reviewer:shortText(input.reviewer,'reviewer',undefined,120),status:input.status,boxes,notes:optionalText(input.notes,'notes',3000),
      assistance:{aiAssisted:assistance.aiAssisted,communityReviewIds:[...new Set(assistance.communityReviewIds)]},
      complete:input.complete===true,trainingAllowed:input.trainingAllowed===true&&frame.evaluationOnly===false&&frame.source.licenseAllowsTraining===true&&input.status!=='uncertain',
      trainingPermissionRequested:input.trainingAllowed===true,humanVerified:true,automaticPropagation:false};
    return this.transaction(()=>{
      const current=this.get(collectionId,frame);if(current.revision!==input.baseRevision)throw new ApiError('Another review was saved. Reload this frame before saving.',409,'VIDEO_GRADE_CONFLICT');
      // An older grader may omit this newly added field. Keep the provenance of
      // its existing draft instead of silently converting it into a blind review.
      if(input.assistance===undefined&&current.review?.assistance)review.assistance=structuredClone(current.review.assistance);
      if(current.revision>=200)throw new ApiError('This frame has reached its review history limit.',409,'VIDEO_GRADE_CAPACITY');
      this.db.prepare('INSERT OR IGNORE INTO grade_frames(collection_id,frame_id,context) VALUES(?,?,?)').run(collectionId,frame.id,JSON.stringify(ctx));
      this.db.prepare('INSERT INTO grade_reviews(id,collection_id,frame_id,revision,created_at,data) VALUES(?,?,?,?,?,?)')
        .run(randomUUID(),collectionId,frame.id,current.revision+1,new Date().toISOString(),JSON.stringify(review));
      return this.get(collectionId,frame);
    });
  }
  snapshot(collectionId,frames){
    if(!Array.isArray(frames)||frames.length>2000)throw new ApiError('Invalid grading collection.');
    const reviews=Object.fromEntries(frames.map(frame=>[frame.id,this.get(collectionId,frame)])),tracks=this.listTracks(collectionId);
    const statusCounts={visible:0,absent:0,uncertain:0},pixelWidthBins=Object.fromEntries(['under4','4to8','8to16','16to32','32to64','64plus'].map(k=>[k,0]));
    let reviewedFrames=0,verifiedBoxes=0,visibleWithoutBoxes=0;
    for(const entry of Object.values(reviews)){const review=entry.review;if(!review)continue;reviewedFrames++;statusCounts[review.status]++;
      if(review.status==='visible'){if(!review.boxes.length)visibleWithoutBoxes++;for(const box of review.boxes){pixelWidthBins[widthBin(box.pixelWidth)]++;verifiedBoxes++;}}}
    const timelines=tracks.map(track=>({...track,frames:frames.filter(frame=>frame.sourceId===track.sourceId&&frame.source.videoHash===track.videoHash)
      .sort((a,b)=>a.timestampSeconds-b.timestampSeconds||a.id.localeCompare(b.id)).map(frame=>({frameId:frame.id,timestampSeconds:frame.timestampSeconds,
        status:reviews[frame.id].review?.status??'unreviewed',boxes:reviews[frame.id].review?.boxes.filter(box=>box.trackId===track.id)??[]}))}));
    return {schema:'mycoscout.video-grades.v1',collectionId,reviews,tracks,report:{totalFrames:frames.length,reviewedFrames,unreviewedFrames:frames.length-reviewedFrames,
      statusCounts,verifiedBoxes,visibleWithoutBoxes,pixelWidthBins,pixelWidthUnit:'original-frame pixels',tracks:timelines,
      detectionAccuracy:null,notice:'Counts describe human reviews, not detector accuracy. Uncertain and unreviewed frames are excluded from verified boxes. Pixel size does not establish distance.'},
      automaticTraining:false,automaticPropagation:false};
  }
  close(){this.db?.close();this.db=null;}
}
