import { spawn } from 'node:child_process';
import { access, mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { resolve, dirname, basename } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { ApiError, shortText } from './validation.mjs';

const sha=value=>createHash('sha256').update(value).digest('hex');
const ID=/^[a-z]+-[a-f0-9-]{12,64}$/;
const read=async path=>JSON.parse(await readFile(path,'utf8'));
const exists=path=>access(path).then(()=>true,()=>false);
const write=async(path,value)=>writeFile(path,JSON.stringify(value,null,2)+'\n',{flag:'wx'});
const signature=row=>sha(JSON.stringify(row));

// Fixed local executables/modules only. API callers select bounded settings and
// immutable IDs, never command strings, arbitrary checkpoints, paths or URLs.
export class TrainingPipeline {
  constructor(root,directory,{testbench,videos,videoGrades,collectionId='video-discovery-v1',timeoutMs=3600000,spawnWorker=spawn,communityInboxFile=resolve(root,'data/community-inbox-v1/reviews.json'),releaseCheck=null}={}){
    this.root=resolve(root);this.directory=resolve(directory);Object.assign(this,{testbench,videos,videoGrades,collectionId,timeoutMs,spawnWorker});
    this.communityInboxFile=resolve(communityInboxFile);
    this.releaseCheck=releaseCheck;
    this.jobs=new Map();this.datasets=new Map();this.active=null;this.child=null;this.busy=false;this.closed=false;
  }
  executable(){return process.env.MYCOSCOUT_PYTHON??resolve(this.root,'runtime-vision',process.platform==='win32'?'Scripts':'bin',process.platform==='win32'?'python.exe':'python');}
  async initialize(){
    for(const name of ['datasets','snapshots','jobs','requests'])await mkdir(resolve(this.directory,name),{recursive:true});
    for(const name of await readdir(resolve(this.directory,'datasets'))){if(!ID.test(name))continue;
      try{const path=resolve(this.directory,'datasets',name),manifest=await read(resolve(path,'manifest.json'));if(await exists(resolve(path,'READY.json')))this.datasets.set(manifest.datasetId,{...manifest,directory:path});}catch{}}
    for(const name of await readdir(resolve(this.directory,'jobs'))){if(!name.endsWith('.json')||!ID.test(name.slice(0,-5)))continue;
      try{const job=await read(resolve(this.directory,'jobs',name));if(['running','queued'].includes(job.status)){job.status='interrupted';job.error='Station stopped before the job completed.';await this.persistJob(job);}this.jobs.set(job.id,job);}catch{}}
    try{this.active=await read(resolve(this.directory,'active-model.json'));}catch{}
    return this;
  }
  async sourceSnapshot(groupAssignments={}){
    if(!groupAssignments||typeof groupAssignments!=='object'||Array.isArray(groupAssignments)||Object.keys(groupAssignments).length>1000)throw new ApiError('Use a record-ID to session-name mapping.');
    const rows=[],excluded={incompleteVideo:0,videoPermission:0,communityPermission:0},evaluationHashes=new Set(),evaluationSources=[];
    let communityIndex=null;
    const workbench=await this.testbench.exportDataset();
    for(const row of workbench.records){
      const video=row.videoProvenance;
      rows.push({id:'workbench:'+row.runId,kind:'workbench',sourceId:row.runId,originalSha256:row.image.sha256,width:row.image.width,height:row.image.height,
        status:row.annotation.boxes.length?'visible':'absent',complete:true,humanVerified:true,reviewer:row.annotation.reviewer,trainingAllowed:true,evaluationOnly:false,
        boxes:row.annotation.boxes,reviewRevision:row.annotation.revision,reviewId:row.annotation.id,
        groupKeys:video?['source:'+video.source.sourceGroupId,'video:'+video.source.videoHash]:['scene:'+row.sceneId],
        sourceCredits:row.sourceCredits,videoProvenance:video??null,rightsAttestation:row.annotation.rightsAttestation});
    }
    // Include every known evaluation hash, even if its review is incomplete.
    for(let offset=0;;offset+=100){const page=this.testbench.list(100,offset);for(const run of page.runs)if(run.evaluationOnly){evaluationHashes.add(run.image.sha256);evaluationSources.push({kind:'workbench',sourceId:run.id,originalSha256:run.image.sha256});}if(offset+page.runs.length>=page.total||!page.runs.length)break;}
    const collection=await this.videos.read();
    for(const frame of collection.frames){
      if(frame.evaluationOnly){evaluationHashes.add(frame.sha256);evaluationSources.push({kind:'video',sourceId:frame.id,originalSha256:frame.sha256,groupKeys:['source:'+frame.sourceGroupId,'video:'+frame.source.videoHash]});}
      const current=this.videoGrades.get(this.collectionId,frame),review=current.review;
      if(!review?.complete||review.status==='uncertain'){excluded.incompleteVideo++;continue;}
      if(!frame.evaluationOnly&&(!review.trainingAllowed||!frame.source.licenseAllowsTraining)){excluded.videoPermission++;continue;}
      const assistance=structuredClone(review.assistance??{aiAssisted:false,communityReviewIds:[]}),communityEvidence=[];
      if(assistance.communityReviewIds.length){
        if(!communityIndex)communityIndex=await this.communityEvidenceIndex();
        let permitted=true;
        for(const id of assistance.communityReviewIds){
          const cited=communityIndex.byId.get(id),latest=cited&&communityIndex.latest.get(cited.reviewerId+':'+cited.frameId);
          if(!cited||!latest||cited.frameId!==frame.id||cited.sha256!==frame.sha256||cited.sourceId!==frame.sourceId||
            latest.sha256!==frame.sha256||latest.sourceId!==frame.sourceId||cited.trainingConsent!==true||latest.trainingConsent!==true){permitted=false;break;}
          communityEvidence.push({id:cited.id,revision:cited.revision,frameId:cited.frameId,sourceId:cited.sourceId,sha256:cited.sha256,
            reviewerId:cited.reviewerId,createdAt:cited.createdAt,trainingConsent:true,labelAuthority:'community-unverified',
            currentConsentReviewId:latest.id,currentConsentRevision:latest.revision});
        }
        // Viewing and copying are both recorded as assistance today. Treat all
        // cited evidence conservatively; consent alone never approves its boxes.
        if(!permitted){excluded.communityPermission++;continue;}
      }
      rows.push({id:'video:'+this.collectionId+':'+frame.id,kind:'video',sourceId:frame.id,originalSha256:frame.sha256,width:frame.width,height:frame.height,
        status:review.status,complete:true,humanVerified:true,reviewer:review.reviewer,trainingAllowed:review.trainingAllowed,evaluationOnly:frame.evaluationOnly,
        boxes:review.boxes,reviewRevision:current.revision,reviewId:review.id,assistance,communityEvidence,
        groupKeys:['source:'+frame.sourceGroupId,'video:'+frame.source.videoHash],
        sourceCredits:frame.source,videoProvenance:{frameId:frame.id,timestampSeconds:frame.timestampSeconds,source:frame.source}});
    }
    const knownIds=new Set(rows.map(row=>row.id));for(const [id,group] of Object.entries(groupAssignments)){
      if(!knownIds.has(id))throw new ApiError('Session grouping references a missing or ineligible record.');shortText(group,'session name',undefined,160);
    }
    for(const row of rows)if(groupAssignments[row.id])row.groupKeys.push('session:'+groupAssignments[row.id].trim());
    const sourceRevisionSignatures=Object.fromEntries(rows.map(row=>[row.id,signature({...row,groupKeys:row.groupKeys.filter(k=>!k.startsWith('session:'))})]));
    // Earlier validation/test sources can never migrate back into training in
    // a later release. Explicit session links may promote a group to holdout.
    const frozenPartitions={};
    const rank={train:0,validation:1,test:2};
    for(const dataset of this.datasets.values())for(const row of dataset.records){for(const key of row.groupKeys){
      if(rank[row.split]>(rank[frozenPartitions[key]]??-1))frozenPartitions[key]=row.split;
    }if(row.split==='test')evaluationHashes.add(row.originalSha256);}
    return {schema:'mycoscout.reviewed-detection-snapshot.v1',createdAt:new Date().toISOString(),records:rows,sourceRevisionSignatures,
      evaluationHashes:[...evaluationHashes].sort(),evaluationSources:[...new Map(evaluationSources.map(row=>[row.originalSha256,row])).values()],
      frozenPartitions,excluded:{...excluded,workbench:workbench.summary.excluded}};
  }
  async communityEvidenceIndex(){
    let inbox;try{inbox=await read(this.communityInboxFile);}catch(error){if(error.code==='ENOENT')return {byId:new Map(),latest:new Map()};throw error;}
    if(inbox.schema!=='mycoscout.community-inbox.v1'||!Array.isArray(inbox.reviews)||!Array.isArray(inbox.history??inbox.reviews))throw new ApiError('Community consent records are unreadable. Import a verified export before training.',409,'LEARNING_COMMUNITY_EVIDENCE');
    if((inbox.history?.length??0)+inbox.reviews.length>250000)throw new ApiError('Community evidence exceeds its import limit.',409,'LEARNING_COMMUNITY_EVIDENCE');
    const byId=new Map(),latest=new Map(),versions=new Map(),conflicts=new Set(),conflictedContributors=new Set();
    for(const row of [...(inbox.history??[]),...inbox.reviews]){
      if(!Number.isSafeInteger(row?.id)||row.id<1||!Number.isSafeInteger(row.revision)||row.revision<1||typeof row.frameId!=='string'||
        !/^[a-f0-9]{64}$/.test(row.reviewerId)||!/^[a-f0-9]{64}$/.test(row.sha256)||typeof row.sourceId!=='string'||!row.sourceId||typeof row.trainingConsent!=='boolean')
        throw new ApiError('Community consent history contains invalid evidence. Import a verified export before training.',409,'LEARNING_COMMUNITY_EVIDENCE');
      const contributor=row.reviewerId+':'+row.frameId,version=contributor+':'+row.revision;
      if(versions.has(version)&&versions.get(version)!==row.id)conflictedContributors.add(contributor);
      versions.set(version,row.id);
      const prior=byId.get(row.id);
      if(prior&&JSON.stringify(prior)!==JSON.stringify(row)){conflicts.add(row.id);conflictedContributors.add(prior.reviewerId+':'+prior.frameId);conflictedContributors.add(row.reviewerId+':'+row.frameId);}
      byId.set(row.id,row);
    }
    for(const id of conflicts)byId.delete(id);
    for(const row of byId.values()){
      const key=row.reviewerId+':'+row.frameId,previous=latest.get(key);
      if(!previous||row.revision>previous.revision)latest.set(key,row);
      else if(row.revision===previous.revision&&row.id!==previous.id){latest.set(key,{...row,trainingConsent:false});}
    }
    for(const key of conflictedContributors)if(latest.has(key))latest.set(key,{...latest.get(key),trainingConsent:false});
    return {byId,latest};
  }
  async status(){
    const snapshot=await this.sourceSnapshot();const latest=[...this.datasets.values()].sort((a,b)=>a.createdAt.localeCompare(b.createdAt)).at(-1);
    const datasetStale=!!latest&&latest.records.some(row=>(row.relatedRecordIds??[row.id]).some(id=>snapshot.sourceRevisionSignatures[id]!==latest.sourceRevisionSignatures[id]));
    return {schema:'mycoscout.learning-status.v1',busy:this.busy,available:await exists(this.executable()),maxActiveJobs:1,
      eligibleImages:snapshot.records.filter(r=>!r.evaluationOnly).length,completeEvaluationImages:snapshot.records.filter(r=>r.evaluationOnly).length,
      eligibleBoxes:snapshot.records.filter(r=>!r.evaluationOnly).reduce((n,r)=>n+r.boxes.length,0),excluded:snapshot.excluded,
      records:snapshot.records.map(r=>({id:r.id,kind:r.kind,boxes:r.boxes.length,evaluationOnly:r.evaluationOnly,groupKeys:r.groupKeys})),
      lastDataset:latest?this.datasetSummary(latest):null,datasetStale,readiness:datasetStale?{...latest.readiness,ready:false,reasons:['Labels or permissions changed. Export a new dataset before training.']}:latest?.readiness??{ready:false,reasons:['Export reviewed labels to measure independent training, validation and test partitions.']},
      jobs:[...this.jobs.values()].slice(-20).reverse(),active:this.active,
      pretrainedAvailable:await exists(resolve(this.root,'models','reviewed-detector','fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth')),
      automaticTraining:false,automaticActivation:false,notice:'Saved grades become training labels only after complete review and explicit permission. Evaluation-only footage never trains.'};
  }
  datasetSummary(data){return {datasetId:data.datasetId,createdAt:data.createdAt,images:data.records.length,readiness:data.readiness,
    duplicatesCollapsed:data.duplicatesCollapsed,excludedConflictingImages:data.excludedConflictingImages.length};}
  manifest(id){const dataset=this.datasets.get(id);if(!dataset)throw new ApiError('Dataset not found.',404,'DATASET_NOT_FOUND');const {directory,...manifest}=dataset;return structuredClone(manifest);}
  checkIdle(){if(this.closed)throw new ApiError('Learning service is stopped.',503,'LEARNING_STOPPED');if(this.busy)throw new ApiError('Another export or training job is running.',409,'LEARNING_BUSY');}
  requestKey(value){if(value===undefined)return null;if(typeof value!=='string'||!/^[-A-Za-z0-9_.:]{1,160}$/.test(value))throw new ApiError('Use a bounded learning idempotency key.');return value;}
  async persistRequest(key,value){const file=resolve(this.directory,'requests',sha(key)+'.json'),temporary=file+'.tmp';await writeFile(temporary,JSON.stringify(value,null,2)+'\n');await rename(temporary,file);}
  async export(input={}){
    const key=this.requestKey(input.idempotencyKey),parameters=signature(input.groupAssignments??{});let request=null;
    if(key){try{request=await read(resolve(this.directory,'requests',sha(key)+'.json'));}catch(error){if(error.code!=='ENOENT')throw error;}
      if(request&&(request.kind!=='export'||request.parameters!==parameters))throw new ApiError('Learning idempotency key was used for another request.',409,'LEARNING_IDEMPOTENCY_CONFLICT');
      if(request?.output&&(typeof request.output!=='string'||dirname(resolve(request.output))!==resolve(this.directory,'datasets')||!ID.test(basename(request.output))))throw new ApiError('Recovered export location is invalid.',409,'LEARNING_SOURCE_CHANGED');
      if(request?.output&&await exists(resolve(request.output,'READY.json'))){
        const manifest=await read(resolve(request.output,'manifest.json')),ready=await read(resolve(request.output,'READY.json'));
        if(sha(await readFile(resolve(request.output,'manifest.json')))!==ready.manifestSha256)throw new ApiError('Recovered dataset integrity failed.',409,'LEARNING_SOURCE_CHANGED');
        const dataset={...manifest,directory:request.output};this.datasets.set(dataset.datasetId,dataset);await this.recheckDataset(dataset);return this.datasetSummary(dataset);
      }
    }
    this.checkIdle();this.busy=true;
    try{
      const snapshot=await this.sourceSnapshot(input.groupAssignments??{}),id='snapshot-'+randomUUID(),directory=resolve(this.directory,'snapshots',id);
      const sourceSignature=signature({sources:snapshot.sourceRevisionSignatures,evaluation:snapshot.evaluationHashes});
      if(request?.sourceSignature&&request.sourceSignature!==sourceSignature)throw new ApiError('Labels changed since this automatic export began.',409,'LEARNING_IDEMPOTENCY_CONFLICT');
      const output=resolve(this.directory,'datasets','release-'+randomUUID());
      if(key){request={kind:'export',parameters,sourceSignature,output,createdAt:request?.createdAt??new Date().toISOString()};await this.persistRequest(key,request);}
      await mkdir(resolve(directory,'images'),{recursive:true});let total=0;
      for(const row of [...snapshot.records,...snapshot.evaluationSources]){const image=row.kind==='workbench'?await this.testbench.image(row.sourceId):(await this.videos.frame(row.sourceId)).image;
        if(image.sha256!==row.originalSha256&&sha(image.bytes)!==row.originalSha256)throw new ApiError('Source image integrity changed.',409,'LEARNING_SOURCE_CHANGED');
        total+=image.bytes.length;if(total>512*1024*1024)throw new ApiError('Reviewed export exceeds the 512 MiB image budget.');
        row.stagedPath='images/'+row.originalSha256+'.img';const path=resolve(directory,row.stagedPath);if(!await exists(path))await writeFile(path,image.bytes,{flag:'wx'});
      }
      const current=await this.sourceSnapshot(input.groupAssignments??{});
      if(JSON.stringify(snapshot.sourceRevisionSignatures)!==JSON.stringify(current.sourceRevisionSignatures)||JSON.stringify(snapshot.evaluationHashes)!==JSON.stringify(current.evaluationHashes))
        throw new ApiError('Labels or permission changed while exporting. Export again.',409,'LEARNING_SOURCE_CHANGED');
      const path=resolve(directory,'snapshot.json');await write(path,snapshot);
      await this.runProcess(['-m','adapters.export_reviewed_detection_dataset','--snapshot',path,'--output',output],120000);
      const manifest=await read(resolve(output,'manifest.json'));this.datasets.set(manifest.datasetId,{...manifest,directory:output});
      return this.datasetSummary(manifest);
    }finally{this.busy=false;}
  }
  async recheckDataset(dataset){
    const current=await this.sourceSnapshot(),byId=new Map(current.records.map(row=>[row.id,row]));
    for(const row of dataset.records)for(const id of row.relatedRecordIds??[row.id]){
      if(!byId.has(id)||current.sourceRevisionSignatures[id]!==dataset.sourceRevisionSignatures[id])throw new ApiError('A label, source or permission changed. Export a new dataset.',409,'LEARNING_SOURCE_CHANGED');
    }
    for(const row of dataset.records)if(row.split!=='test'&&current.evaluationHashes.includes(row.originalSha256))throw new ApiError('A dataset image is now reserved for evaluation. Export again.',409,'LEARNING_SOURCE_CHANGED');
  }
  async startJob(input={}){
    const key=this.requestKey(input.idempotencyKey),validationOnly=input.validationOnly??false;
    if(typeof validationOnly!=='boolean')throw new ApiError('validationOnly must be a boolean.');
    const requestParameters=signature({datasetId:input.datasetId,epochs:input.epochs??3,learningRate:input.learningRate??.005,scoreThreshold:input.scoreThreshold??.3,validationOnly});
    if(key){const previous=[...this.jobs.values()].find(job=>job.idempotencyKey===key);
      if(previous){if(previous.requestParameters!==requestParameters)throw new ApiError('Learning idempotency key was used for another job.',409,'LEARNING_IDEMPOTENCY_CONFLICT');return structuredClone(previous);}}
    this.checkIdle();const dataset=this.datasets.get(input.datasetId);if(!dataset)throw new ApiError('Export and select an existing reviewed dataset.',404,'DATASET_NOT_FOUND');
    if(!dataset.readiness.ready)throw new ApiError('Not enough independent reviewed data: '+dataset.readiness.reasons.join('; '),409,'LEARNING_INSUFFICIENT_DATA');
    const epochs=input.epochs??3,learningRate=input.learningRate??.005,scoreThreshold=input.scoreThreshold??.3;
    if(!Number.isInteger(epochs)||epochs<1||epochs>30||typeof learningRate!=='number'||!Number.isFinite(learningRate)||learningRate<.00001||learningRate>.02||
      typeof scoreThreshold!=='number'||!Number.isFinite(scoreThreshold)||scoreThreshold<.05||scoreThreshold>.95)throw new ApiError('Use 1–30 epochs, learning rate .00001–.02 and threshold .05–.95.');
    this.busy=true;
    try{await this.recheckDataset(dataset);const id='job-'+randomUUID(),job={id,status:'queued',datasetId:dataset.datasetId,epochs,learningRate,scoreThreshold,
      createdAt:new Date().toISOString(),outputDirectory:resolve(this.directory,'jobs',id),progress:null,automaticActivation:false,validationOnly,
      ...(key?{idempotencyKey:key,requestParameters}:{})};
      this.jobs.set(id,job);await this.persistJob(job);this.running=this.execute(job,dataset);return structuredClone(job);
    }catch(error){this.busy=false;throw error;}
  }
  async execute(job,dataset){
    try{job.status='running';await this.persistJob(job);
      // This fixed official download runs only after a legitimate dataset has
      // passed readiness and a user explicitly starts a training job.
      await this.runProcess(['-m','vision.training_pipeline','download-pretrained'],180000,job);
      if(job.status==='cancelled')return;
      await this.runProcess(['-m','vision.training_pipeline','train','--dataset',dataset.directory,'--output',job.outputDirectory,
        '--epochs',String(job.epochs),'--learning-rate',String(job.learningRate),'--score-threshold',String(job.scoreThreshold),...(job.validationOnly?['--validation-only']:[])],this.timeoutMs,job);
      if(job.status==='cancelled')return;
      job.modelCard=await read(resolve(job.outputDirectory,'model-card.json'));job.status='completed';
    }catch(error){if(job.status!=='cancelled'){job.status='failed';job.error=error.message;}}
    finally{job.finishedAt=new Date().toISOString();await this.persistJob(job);this.busy=false;this.running=null;}
  }
  runProcess(args,timeoutMs,job=null){return new Promise((resolvePromise,reject)=>{
    let output='',stderr='',lastError='',settled=false;
    const child=this.spawnWorker(this.executable(),args,{cwd:this.root,stdio:['ignore','pipe','pipe'],windowsHide:true,env:{...process.env,PYTHONUNBUFFERED:'1'}});this.child=child;
    const finish=(error)=>{if(settled)return;settled=true;clearTimeout(timer);if(this.child===child)this.child=null;error?reject(error):resolvePromise();};
    const timer=setTimeout(()=>{child.kill();finish(new ApiError('Learning job exceeded its time limit.',504,'LEARNING_TIMEOUT'));},timeoutMs);
    child.stdout.setEncoding('utf8');child.stderr.setEncoding('utf8');
    child.stdout.on('data',chunk=>{output+=chunk;if(output.length>2*1024*1024){child.kill();finish(new Error('Learning output exceeded its size limit.'));return;}
      let end;while((end=output.indexOf('\n'))>=0){const line=output.slice(0,end);output=output.slice(end+1);try{const item=JSON.parse(line);if(job&&item.status==='progress')job.progress=item;if(item.status==='failed'&&typeof item.error==='string')lastError=item.error.slice(0,2000);}catch{}}});
    child.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-6000);});
    child.on('error',error=>finish(new ApiError('Local learning runtime could not start: '+error.message,503,'LEARNING_UNAVAILABLE')));
    child.on('exit',code=>finish(code===0?null:new ApiError(job?.status==='cancelled'?'Learning job cancelled.':'Local learning process failed. '+(lastError||stderr.slice(-2000)),422,'LEARNING_FAILED')));
  });}
  async persistJob(job){const path=resolve(this.directory,'jobs',job.id+'.json'),temporary=path+'.tmp';await writeFile(temporary,JSON.stringify(job,null,2)+'\n');await rename(temporary,path);}
  job(id){const job=this.jobs.get(id);if(!job)throw new ApiError('Learning job not found.',404,'NOT_FOUND');return structuredClone(job);}
  cancel(id){const job=this.jobs.get(id);if(!job)throw new ApiError('Learning job not found.',404,'NOT_FOUND');if(['running','queued'].includes(job.status)){job.status='cancelled';this.child?.kill();}return structuredClone(job);}
  async activate(input={}){
    this.checkIdle();const job=this.jobs.get(input.jobId),reviewer=shortText(input.reviewer,'reviewer',undefined,120);
    if(!job||job.status!=='completed'||!job.modelCard||(!job.validationOnly&&!job.modelCard.activationEligible))throw new ApiError('Only a completed eligible candidate can be activated.',409,'LEARNING_NOT_ACTIVATABLE');
    const dataset=this.datasets.get(job.datasetId);if(!dataset)throw new ApiError('Candidate dataset is missing.',409,'LEARNING_SOURCE_CHANGED');
    // Reserve the worker before any asynchronous permission or release checks.
    // The release hook may verify files using the same bounded Python runner.
    this.busy=true;
    try{
    await this.recheckDataset(dataset);
    const checkpoint=resolve(job.outputDirectory,'candidate.pt'),bytes=await readFile(checkpoint);if(sha(bytes)!==job.modelCard.checkpointSha256)throw new ApiError('Candidate checkpoint integrity failed.',409,'LEARNING_CHECKPOINT_CHANGED');
    let releaseComparison=null;
    if(job.validationOnly){
      if(typeof this.releaseCheck!=='function')throw new ApiError('This candidate still needs an explicit held-out release comparison.',409,'LEARNING_RELEASE_CHECK_REQUIRED');
      releaseComparison=await this.releaseCheck(structuredClone(job),dataset);
      if(releaseComparison?.approved!==true)throw new ApiError('The held-out release comparison did not approve this candidate.',409,'LEARNING_RELEASE_CHECK_FAILED');
      await this.recheckDataset(dataset);
      if(sha(await readFile(checkpoint))!==job.modelCard.checkpointSha256)throw new ApiError('Candidate checkpoint changed during release checking.',409,'LEARNING_CHECKPOINT_CHANGED');
    }
    const active={schema:'mycoscout.active-detector.v1',jobId:job.id,datasetId:job.datasetId,checkpoint,checkpointSha256:sha(bytes),reviewer,
      activatedAt:new Date().toISOString(),previousJobId:this.active?.jobId??null,modelCard:job.modelCard,releaseComparison,experimental:true,operationallyValidated:false};
    const path=resolve(this.directory,'active-model.json'),temp=path+'.tmp';await writeFile(temp,JSON.stringify(active,null,2)+'\n');await rename(temp,path);this.active=active;return structuredClone(active);
    }finally{this.busy=false;}
  }
  async stop(){this.closed=true;for(const job of this.jobs.values())if(['running','queued'].includes(job.status))job.status='cancelled';this.child?.kill();if(this.running)await this.running;}
}
