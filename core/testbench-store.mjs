import { DatabaseSync } from 'node:sqlite';
import { mkdir,readFile,writeFile,realpath,stat } from 'node:fs/promises';
import { resolve,sep } from 'node:path';
import { createHash,randomUUID } from 'node:crypto';
import { ApiError,shortText } from './validation.mjs';
import { validateImage } from './finds-images.mjs';
import { DEFAULT_SETTINGS,testbenchSettings,testbenchResult,testbenchAnnotation,testbenchMetrics } from './testbench-validation.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const parse=JSON.parse;
const validId=id=>typeof id==='string'&&/^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/.test(id);
const MAX_RUNS=1000,MAX_STORAGE=2*1024*1024*1024;
export const SAMPLE={path:'data/black-trumpet-discovery-v2/images/inat-photo-745660170.jpg',
  sha256:'3b5ab0e9d6f741d9a395e27312fe563cc5dc6007347762a8cb6519d0ea4faf26',name:'Licensed black-trumpet research sample',
  attribution:'no rights reserved (CC0)',licenseUrl:'https://creativecommons.org/publicdomain/zero/1.0/',sourceUrl:'https://www.inaturalist.org/observations/405910182'};

export class TestbenchStore {
  constructor(directory,root,worker,{sample=SAMPLE,catalogFile=resolve(root,'config','michigan-mushrooms-v1.json')}={}){
    this.directory=resolve(directory);this.root=root;this.worker=worker;this.sample=sample;this.catalogFile=catalogFile;
    this.db=null;this.activeRunId=null;this.creating=false;this.closed=false;this.activeTask=null;this.pending=Promise.resolve();
  }
  async initialize(){
    await mkdir(resolve(this.directory,'photos'),{recursive:true});
    const catalog=parse(await readFile(this.catalogFile,'utf8'));this.groupIds=new Set(catalog.groups.map(group=>group.id));
    this.db=new DatabaseSync(resolve(this.directory,'testbench.sqlite'));
    const version=this.db.prepare('PRAGMA user_version').get().user_version;if(![0,1,2].includes(version))throw new Error('Unsupported testbench database version');
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA synchronous=FULL;
      CREATE TABLE IF NOT EXISTS photos(hash TEXT PRIMARY KEY,extension TEXT NOT NULL,mime TEXT NOT NULL,width INTEGER NOT NULL,height INTEGER NOT NULL,bytes INTEGER NOT NULL,evaluation_only INTEGER NOT NULL,credits TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY,created_at TEXT NOT NULL,updated_at TEXT NOT NULL,name TEXT NOT NULL,photo_hash TEXT NOT NULL REFERENCES photos(hash),settings TEXT NOT NULL,source_run_id TEXT,scene_id TEXT NOT NULL,status TEXT NOT NULL,progress TEXT,result TEXT,error TEXT);
      CREATE TABLE IF NOT EXISTS annotations(id TEXT PRIMARY KEY,run_id TEXT NOT NULL REFERENCES runs(id),revision INTEGER NOT NULL,created_at TEXT NOT NULL,data TEXT NOT NULL,UNIQUE(run_id,revision));
      CREATE INDEX IF NOT EXISTS annotation_run ON annotations(run_id,revision);
      CREATE TRIGGER IF NOT EXISTS annotations_no_update BEFORE UPDATE ON annotations BEGIN SELECT RAISE(ABORT,'Annotations are immutable'); END;
      CREATE TRIGGER IF NOT EXISTS annotations_no_delete BEFORE DELETE ON annotations BEGIN SELECT RAISE(ABORT,'Annotations are immutable'); END;
      CREATE TRIGGER IF NOT EXISTS runs_no_delete BEFORE DELETE ON runs BEGIN SELECT RAISE(ABORT,'Test runs are retained'); END;
      CREATE TRIGGER IF NOT EXISTS runs_inputs_immutable BEFORE UPDATE OF photo_hash,settings,source_run_id,scene_id,name,created_at ON runs BEGIN SELECT RAISE(ABORT,'Run inputs are immutable'); END;`);
    // Source identity belongs to a particular observation, not deduplicated pixels.
    // Legacy photo-level metadata cannot prove which timestamp each run used; do
    // not backfill guessed frame identity into historical runs.
    this.transaction(()=>{
      if(!this.db.prepare('PRAGMA table_info(runs)').all().some(column=>column.name==='video_provenance'))
        this.db.exec('ALTER TABLE runs ADD COLUMN video_provenance TEXT');
      this.db.exec(`DROP TRIGGER IF EXISTS runs_inputs_immutable;
        CREATE TRIGGER runs_inputs_immutable BEFORE UPDATE OF photo_hash,settings,source_run_id,scene_id,name,created_at,video_provenance ON runs BEGIN SELECT RAISE(ABORT,'Run inputs are immutable'); END;
        PRAGMA user_version=2;`);
    });
    this.db.prepare("UPDATE runs SET status='failed',error=?,updated_at=? WHERE status IN ('queued','running')")
      .run(JSON.stringify({code:'TESTBENCH_INTERRUPTED',message:'The station restarted during this test. Rerun the original image.'}),new Date().toISOString());
    return this;
  }
  transaction(action){this.db.exec('BEGIN IMMEDIATE');try{const result=action();this.db.exec('COMMIT');return result;}catch(error){this.db.exec('ROLLBACK');throw error;}}
  async status(){return {schema:'mycoscout.testbench-status.v1',worker:await this.worker.status(),busy:this.creating||!!this.activeRunId,
    activeRunId:this.activeRunId,counts:{runs:this.db.prepare('SELECT COUNT(*) AS n FROM runs').get().n,photos:this.db.prepare('SELECT COUNT(*) AS n FROM photos').get().n},
    defaultSettings:DEFAULT_SETTINGS,automaticTraining:false,liveCameraConnected:false,actuationEnabled:false};}
  row(id){if(!validId(id))throw new ApiError('Test run not found.',404,'NOT_FOUND');const row=this.db.prepare('SELECT * FROM runs WHERE id=?').get(id);
    if(!row)throw new ApiError('Test run not found.',404,'NOT_FOUND');return row;}
  annotation(id){const row=this.db.prepare('SELECT rowid AS sequence,* FROM annotations WHERE run_id=? ORDER BY revision DESC LIMIT 1').get(id);
    return row?{...parse(row.data),id:row.id,revision:row.revision,createdAt:row.created_at,sequence:row.sequence}:null;}
  get(id){const row=this.row(id),photo=this.db.prepare('SELECT * FROM photos WHERE hash=?').get(row.photo_hash),result=row.result?parse(row.result):null,annotation=this.annotation(id);
    const video=row.video_provenance?parse(row.video_provenance):null,{videoProvenance:legacyVideo,...photoCredits}=parse(photo.credits);
    const sourceCredits=video?{attribution:video.source.attribution,licenseUrl:video.source.licenseUrl,sourceUrl:video.source.sourceUrl}:photoCredits;
    return {id:row.id,name:row.name,createdAt:row.created_at,updatedAt:row.updated_at,status:row.status,settings:parse(row.settings),sourceRunId:row.source_run_id,
      sceneId:row.scene_id,evaluationOnly:!!photo.evaluation_only,sourceCredits,videoProvenance:video,legacyVideoProvenanceNeedsRerun:!video&&!!legacyVideo,
      image:{sha256:row.photo_hash,width:result?.image.width??photo.width,height:result?.image.height??photo.height,
        mimeType:photo.mime,bytes:photo.bytes,orientationVerified:!!result},imageUrl:`/api/testbench/runs/${row.id}/image`,
      progress:row.progress?parse(row.progress):null,result,error:row.error?parse(row.error):null,
      annotationRevision:annotation?.revision??0,annotation,metrics:testbenchMetrics(result,annotation),
      autoTraining:false,actuationEnabled:false};}
  list(limit=50,offset=0){limit=Number(limit??50);offset=Number(offset??0);
    if(!Number.isInteger(limit)||limit<1||limit>100||!Number.isInteger(offset)||offset<0||offset>MAX_RUNS)throw new ApiError('Use limit 1–100 and a nonnegative offset.');
    const rows=this.db.prepare('SELECT id FROM runs ORDER BY created_at DESC,id DESC LIMIT ? OFFSET ?').all(limit,offset);
    return {runs:rows.map(({id})=>{const {result,annotation,...run}=this.get(id);return {...run,hasAnnotation:!!annotation,
      completeAnnotation:annotation?.complete===true,predictionCounts:result?.methods.map(method=>({id:method.id,count:method.candidates.length}))??[]};}),
      total:this.db.prepare('SELECT COUNT(*) AS n FROM runs').get().n,limit,offset};}
  async image(id){const run=this.get(id),photo=this.db.prepare('SELECT * FROM photos WHERE hash=?').get(run.image.sha256);
    const directory=await realpath(resolve(this.directory,'photos')),path=await realpath(resolve(directory,`${photo.hash}.${photo.extension}`));
    if(!path.startsWith(directory+sep))throw new ApiError('Invalid stored test image path.',409,'PHOTO_INTEGRITY');
    const bytes=await readFile(path);if(sha(bytes)!==photo.hash)throw new ApiError('Stored test image failed its integrity check.',409,'PHOTO_INTEGRITY');
    return {bytes,sha256:photo.hash,mimeType:photo.mime,extension:photo.extension,width:photo.width,height:photo.height};}
  async sampleInfo(){let available=false;try{const file=resolve(this.root,this.sample.path);available=(await stat(file)).size<=8*1024*1024&&sha(await readFile(file))===this.sample.sha256;}catch{}
    return {available,name:this.sample.name,attribution:this.sample.attribution,licenseUrl:this.sample.licenseUrl,sourceUrl:this.sample.sourceUrl,evaluationOnly:true,
      annotationsIncluded:false,notice:'Research sample. Supply your own annotations; existing assistant labels are not imported.'};}
  async createSample(input={}){const settings=testbenchSettings(input.settings??DEFAULT_SETTINGS);
    if(!(await this.sampleInfo()).available)throw new ApiError('The licensed sample is not installed on this station.',404,'SAMPLE_UNAVAILABLE');
    const bytes=await readFile(resolve(this.root,this.sample.path));return this.create({imageBase64:bytes.toString('base64'),name:this.sample.name,settings});}
  async create(input,trustedVideo=null){
    if(this.closed)throw new ApiError('Testbench is stopping.',503,'TESTBENCH_STOPPED');
    if(this.creating||this.activeRunId)throw new ApiError('Another image test is running. Wait for it to finish.',409,'TESTBENCH_BUSY');
    this.creating=true;
    try{
      if(input.evaluationOnly!==undefined&&typeof input.evaluationOnly!=='boolean')throw new ApiError('evaluationOnly must be a boolean when supplied.');
      const settings=testbenchSettings(input.settings),name=input.name==null?'Untitled image test':shortText(input.name,'name',undefined,160);
      if(input.sourceRunId!==undefined&&input.imageBase64!==undefined)throw new ApiError('Choose a source run or upload an image, not both.');
      let source=null,image;
      if(input.sourceRunId!==undefined){source=this.get(input.sourceRunId);image=await this.image(source.id);}
      else image=validateImage(input);
      if(this.db.prepare('SELECT COUNT(*) AS n FROM runs').get().n>=MAX_RUNS)throw new ApiError('Testbench run capacity reached. Export this collection before starting a new store.',409,'TESTBENCH_CAPACITY');
      const known=this.db.prepare('SELECT * FROM photos WHERE hash=?').get(image.sha256),sample=image.sha256===this.sample.sha256;
      const video=trustedVideo??source?.videoProvenance??null;
      const evaluationOnly=input.evaluationOnly===true||sample||source?.evaluationOnly||!!known?.evaluation_only||video?.evaluationOnly===true;
      const credits=sample?{attribution:this.sample.attribution,licenseUrl:this.sample.licenseUrl,sourceUrl:this.sample.sourceUrl}:known?parse(known.credits):{};
      const sceneId=video?.sceneId??source?.sceneId??'scene-'+image.sha256;
      if(!known){const used=this.db.prepare('SELECT COALESCE(SUM(bytes),0) AS bytes FROM photos').get().bytes;
        if(used+image.bytes.length>MAX_STORAGE)throw new ApiError('Testbench photo storage limit reached.',409,'TESTBENCH_CAPACITY');
        const photoPath=resolve(this.directory,'photos',`${image.sha256}.${image.extension}`);
        try{await writeFile(photoPath,image.bytes,{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;if(sha(await readFile(photoPath))!==image.sha256)throw new ApiError('Test image collision.',409,'PHOTO_INTEGRITY');}
      }
      const id=randomUUID(),now=new Date().toISOString();
      this.transaction(()=>{
        this.db.prepare('INSERT OR IGNORE INTO photos(hash,extension,mime,width,height,bytes,evaluation_only,credits) VALUES(?,?,?,?,?,?,?,?)')
          .run(image.sha256,image.extension,image.mimeType,image.width,image.height,image.bytes.length,evaluationOnly?1:0,JSON.stringify(credits));
        if(evaluationOnly)this.db.prepare('UPDATE photos SET evaluation_only=1 WHERE hash=?').run(image.sha256);
        this.db.prepare('INSERT INTO runs(id,created_at,updated_at,name,photo_hash,settings,source_run_id,scene_id,status,video_provenance) VALUES(?,?,?,?,?,?,?,?,?,?)')
          .run(id,now,now,input.name==null&&source?source.name:name,image.sha256,JSON.stringify(settings),source?.id??null,sceneId,'queued',video?JSON.stringify(video):null);
        if(source?.annotation){const {id:oldId,revision,createdAt,...truth}=source.annotation;
          this.db.prepare('INSERT INTO annotations(id,run_id,revision,created_at,data) VALUES(?,?,?,?,?)').run(randomUUID(),id,1,now,JSON.stringify({...truth,
            trainingAllowed:evaluationOnly?false:truth.trainingAllowed,inheritedFrom:{runId:source.id,annotationId:oldId,revision,createdAt},
            inheritanceReason:'Exact original image hash; no model labels copied'}));}
      });
      this.activeRunId=id;
      const work=this.execute(id,image,settings);this.activeTask=work;this.pending=work.finally(()=>{if(this.activeTask===work)this.activeTask=null;});
      return this.get(id);
    }finally{this.creating=false;}
  }
  async execute(id,image,settings){
    try{
      this.db.prepare("UPDATE runs SET status='running',updated_at=? WHERE id=?").run(new Date().toISOString(),id);
      const raw=await this.worker.run(image,settings,progress=>{if(!this.closed)this.db.prepare('UPDATE runs SET progress=?,updated_at=? WHERE id=?').run(JSON.stringify(progress),new Date().toISOString(),id);});
      const result=testbenchResult(raw,image,settings);
      this.db.prepare("UPDATE runs SET status='completed',result=?,progress=?,updated_at=? WHERE id=?")
        .run(JSON.stringify(result),JSON.stringify({stage:'complete',completed:1,total:1}),new Date().toISOString(),id);
    }catch(error){if(this.db){this.db.prepare("UPDATE runs SET status='failed',error=?,updated_at=? WHERE id=?")
      .run(JSON.stringify({code:error instanceof ApiError?error.code:'TESTBENCH_FAILED',message:error instanceof ApiError?error.message:'The local image test failed.'}),new Date().toISOString(),id);}}
    finally{if(this.activeRunId===id)this.activeRunId=null;}
  }
  annotate(id,input){
    const run=this.get(id);if(run.status!=='completed')throw new ApiError('Wait for a completed, decoded image test before saving annotations.',409,'TESTBENCH_NOT_COMPLETE');
    const annotation=testbenchAnnotation(input,run,this.groupIds);
    return this.transaction(()=>{
      const current=this.annotation(id),revision=current?.revision??0;
      if(revision!==input.baseRevision)throw new ApiError('Another annotation was saved. Reload this run before saving.',409,'ANNOTATION_CONFLICT');
      if(revision>=200)throw new ApiError('Annotation revision capacity reached for this run.',409,'ANNOTATION_CAPACITY');
      const annotationId=randomUUID(),now=new Date().toISOString();
      this.db.prepare('INSERT INTO annotations(id,run_id,revision,created_at,data) VALUES(?,?,?,?,?)').run(annotationId,id,revision+1,now,JSON.stringify(annotation));
      this.db.prepare('UPDATE runs SET updated_at=? WHERE id=?').run(now,id);
      const updated=this.get(id);return {run:updated,annotation:updated.annotation};
    });
  }
  completedVideoRuns(){
    return this.db.prepare("SELECT id,status,created_at,settings,result,photo_hash,video_provenance FROM runs WHERE status='completed' AND result IS NOT NULL AND video_provenance IS NOT NULL")
      .all().map(row=>({id:row.id,status:row.status,createdAt:row.created_at,settings:parse(row.settings),result:parse(row.result),image:{sha256:row.photo_hash},videoProvenance:parse(row.video_provenance)}));
  }
  async exportDataset(){
    // Inherited copies never supersede source corrections. Current independent
    // human labels must agree; competing labels are quarantined, not overwritten.
    const all=this.db.prepare('SELECT id FROM runs ORDER BY created_at,id').all().map(row=>this.get(row.id));
    const byImage=new Map();
    for(const run of all){if(!byImage.has(run.image.sha256))byImage.set(run.image.sha256,[]);byImage.get(run.image.sha256).push(run);}
    const conflictingGroups=new Set();
    for(const runs of byImage.values()){
      const groups=new Set(runs.filter(run=>run.videoProvenance).map(run=>run.sceneId));
      if(groups.size>1)for(const group of groups)conflictingGroups.add(group);
    }
    const records=[],excluded={incompleteOrUnreviewed:0,rightsNotGranted:0,evaluationOnly:0,failedOrPending:0,conflictingAnnotations:0,ambiguousSourceGroups:0};
    for(const runs of byImage.values()){
      const direct=runs.filter(run=>run.annotation&&!run.annotation.inheritedFrom).sort((a,b)=>b.annotation.sequence-a.annotation.sequence);
      const selected=direct[0]??runs.filter(run=>run.annotation).sort((a,b)=>b.annotation.createdAt.localeCompare(a.annotation.createdAt))[0];
      if(!selected||!selected.annotation.complete){excluded.incompleteOrUnreviewed++;continue;}
      if(selected.evaluationOnly){excluded.evaluationOnly++;continue;}
      if(!selected.annotation.trainingAllowed){excluded.rightsNotGranted++;continue;}
      if(selected.legacyVideoProvenanceNeedsRerun||runs.some(run=>run.videoProvenance&&conflictingGroups.has(run.sceneId))||
        (!selected.videoProvenance&&runs.some(run=>run.videoProvenance))){excluded.ambiguousSourceGroups++;continue;}
      const labelSignatures=new Set(direct.filter(run=>run.annotation.complete).map(run=>JSON.stringify(run.annotation.boxes
        .map(box=>({bboxPixelsXYXY:box.bboxPixelsXYXY,groupId:box.groupId,label:box.label})).sort((a,b)=>JSON.stringify(a).localeCompare(JSON.stringify(b))))));
      if(labelSignatures.size>1){excluded.conflictingAnnotations++;continue;}
      if(selected.status!=='completed'){excluded.failedOrPending++;continue;}
      const video=selected.videoProvenance,sceneId=video?.sceneId??selected.sceneId;
      const splitNumber=parseInt((video?sha(sceneId):selected.image.sha256).slice(0,8),16)%100,partition=splitNumber<80?'train':splitNumber<90?'validation':'test';
      records.push({image:{...selected.image,url:selected.imageUrl},sceneId,partition,runId:selected.id,evaluationOnly:selected.evaluationOnly,videoProvenance:video,
        sourceRunId:selected.sourceRunId,annotation:selected.annotation,sourceCredits:selected.sourceCredits,
        settings:selected.settings,model:selected.result?.model??null,
        provenance:{originalImageSha256:selected.image.sha256,relatedRunIds:runs.map(run=>run.id),labelsFrom:'explicit_human_annotations',
          sceneGrouping:video?'source-video-group':'exact-original-sha256',nearDuplicateGroupingVerified:false},metrics:selected.metrics});
    }
    const content={schema:'mycoscout.testbench-dataset.v1',records,summary:{records:records.length,uniqueImages:byImage.size,runs:all.length,excluded},
      policies:{automaticTraining:false,automaticModelDeployment:false,modelScoresAreLabels:false,rightsRequired:true,completeAnnotationsRequired:true,
        knownResearchSampleExcluded:true,conflictingHumanLabelsQuarantined:true,originalImageAndRerunsShareSplit:true,videoFramesShareSourceGroup:true,ambiguousVideoGroupsQuarantined:true,nearDuplicateSceneGroupingImplemented:false},
      notes:['Exact duplicates and reruns share the original scene split. Video frames share their source video group. Separate videos and photographs of the same scene still need grouping before training.',
        'Photo permission and measured distance are asserted by the reviewer; neither is independently verified.',
        'Image references require station access. Export does not upload data or retrain a model.']};
    const releaseId='testbench-'+sha(JSON.stringify(content));const manifest={releaseId,...content};
    const directory=resolve(this.directory,'exports');await mkdir(directory,{recursive:true});
    try{await writeFile(resolve(directory,releaseId+'.json'),JSON.stringify(manifest,null,2)+'\n',{flag:'wx'});}catch(error){if(error.code!=='EEXIST')throw error;}
    return manifest;
  }
  stop(){this.closed=true;this.worker.stop?.();const close=()=>{if(this.db){this.db.close();this.db=null;}};
    if(!this.activeTask){close();return Promise.resolve();}return this.pending.finally(close);}
}
