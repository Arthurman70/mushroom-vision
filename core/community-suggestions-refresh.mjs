import { mkdir, readFile, writeFile, rename, stat, unlink, realpath } from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { resolve, relative, isAbsolute } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';

const SCHEMA='mycoscout.community-suggestions-refresh.v1';
const HASH=/^[a-f0-9]{64}$/;
const OUTPUT=/^suggestions-[a-f0-9-]{36}\.json$/;
const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const inside=(base,path)=>{const result=resolve(base,path),rel=relative(base,result);if(rel==='..'||rel.startsWith('..\\')||rel.startsWith('../')||isAbsolute(rel))throw new Error('Path escaped its collection.');return result;};
const readBounded=async(path,limit)=>{if((await stat(path)).size>limit)throw new Error('File exceeded its bound.');const bytes=await readFile(path);if(bytes.length>limit)throw new Error('File exceeded its bound.');return bytes;};
const json=async(path,limit=8*1024*1024)=>{const bytes=await readBounded(path,limit);return {value:JSON.parse(bytes),sha256:sha(bytes)};};
const fileSha=async(path,limit)=>{if((await stat(path)).size>limit)throw new Error('File exceeded its bound.');const hash=createHash('sha256');let count=0;for await(const chunk of createReadStream(path)){count+=chunk.length;if(count>limit)throw new Error('File exceeded its bound.');hash.update(chunk);}return hash.digest('hex');};
const activeIdentity=active=>sha(JSON.stringify(active));
const finite=value=>typeof value==='number'&&Number.isFinite(value);

// This worker only refreshes unverified suggestions from a manually active
// detector. It cannot train, promote a model, or write human review records.
export class CommunitySuggestionsRefresh {
  constructor(root,directory,{learning,bridge,collectionDir=resolve(root,'data/community-publish-v1'),now=()=>Date.now(),retryMs=60000,timeoutMs=3660000,maxSeconds=3600}={}){
    if(!learning||!bridge)throw new Error('Learning and community bridge are required.');
    this.root=resolve(root);this.directory=resolve(directory);this.collectionDir=resolve(collectionDir);
    Object.assign(this,{learning,bridge,now,retryMs,timeoutMs,maxSeconds});
    this.state={schema:SCHEMA,status:'idle',pending:null,completed:[],failures:0,nextAttemptAt:0,lastCompletedAt:null,lastFrameCount:0,error:null};
    this.running=null;this.closed=false;this.child=null;this.progress={};this.writes=Promise.resolve();
  }
  async initialize(){
    await mkdir(this.directory,{recursive:true});
    try{
      const {value}=await json(resolve(this.directory,'refresh.json'),2*1024*1024);
      if(value.schema!==SCHEMA||!Array.isArray(value.completed)||value.completed.length>4096||
        value.completed.some(row=>!HASH.test(row?.key)||!HASH.test(row?.outputSha256))||
        new Set(value.completed.map(row=>row.key)).size!==value.completed.length||
        !Number.isSafeInteger(value.failures)||value.failures<0||value.failures>30||!finite(value.nextAttemptAt)||
        (value.pending&&(!HASH.test(value.pending.key)||!OUTPUT.test(value.pending.output)||!HASH.test(value.pending.contextSha256))))throw new Error('Invalid suggestion refresh state.');
      this.state={...this.state,...value,status:value.pending?'waiting':'idle',error:value.error?'Suggestion refresh is waiting for a successful retry.':null};
    }catch(error){if(error.code!=='ENOENT')throw error;}
    return this;
  }
  snapshot(){
    return {schema:SCHEMA,enabled:this.bridge.state.enabled===true,activeModelAvailable:!!this.learning.active,
      status:this.closed?'stopped':this.state.status,running:!!this.running,completedRefreshes:this.state.completed.length,
      pending:!!this.state.pending,lastCompletedAt:this.state.lastCompletedAt,lastFrameCount:this.state.lastFrameCount,
      nextAttemptAt:this.state.nextAttemptAt||null,error:this.state.error,
      progress:this.running&&Number.isSafeInteger(this.progress.progress?.completed)?{
        completed:this.progress.progress.completed,total:this.progress.progress.total}:null};
  }
  persist(){
    const value=structuredClone(this.state);
    const pending=this.writes.then(async()=>{const path=resolve(this.directory,'refresh.json'),temp=path+'.'+randomUUID()+'.tmp';await writeFile(temp,JSON.stringify(value,null,2)+'\n',{flag:'wx'});await rename(temp,path);});
    this.writes=pending.catch(()=>{});return pending;
  }
  available(){if(this.closed||this.learning.closed||this.bridge.state.enabled!==true)throw new Error('Refresh is unavailable.');}
  async context(){
    this.available();const active=structuredClone(this.learning.active);
    if(!active||active.schema!=='mycoscout.active-detector.v1'||!HASH.test(active.checkpointSha256))throw new Error('Active detector is missing.');
    const job=this.learning.job(active.jobId),dataset=this.learning.datasets.get(active.datasetId);
    if(job.status!=='completed'||job.datasetId!==active.datasetId||!dataset||
      job.modelCard?.checkpointSha256!==active.checkpointSha256||active.modelCard?.checkpointSha256!==active.checkpointSha256||
      resolve(active.checkpoint)!==resolve(job.outputDirectory,'candidate.pt'))throw new Error('Active detector binding changed.');
    if(job.validationOnly&&(active.releaseComparison?.approved!==true||active.releaseComparison?.candidateCheckpointSha256!==active.checkpointSha256))throw new Error('Active detector release approval is missing.');
    const card=await json(resolve(job.outputDirectory,'model-card.json')),complete=await json(resolve(job.outputDirectory,'COMPLETE.json'));
    const datasetManifest=await json(resolve(dataset.directory,'manifest.json'),32*1024*1024);
    if(card.value.schema!=='mycoscout.detector-model-card.v1'||card.value.datasetId!==active.datasetId||
      card.value.datasetManifestSha256!==datasetManifest.sha256||datasetManifest.value.datasetId!==active.datasetId||
      card.value.validationOnly!==!!job.validationOnly||(!job.validationOnly&&card.value.activationEligible!==true)||
      card.value.checkpoint!=='candidate.pt'||card.value.checkpointSha256!==active.checkpointSha256||
      complete.value.checkpointSha256!==active.checkpointSha256||complete.value.datasetId!==active.datasetId||
      complete.value.modelCardSha256!==card.sha256||
      await fileSha(active.checkpoint,1024*1024*1024)!==active.checkpointSha256)throw new Error('Active detector files changed.');
    const manifest=await json(resolve(this.collectionDir,'manifest.json'));
    if(manifest.value.schema!=='mycoscout.community-collection.v1'||!Array.isArray(manifest.value.frames)||
      !manifest.value.frames.length||manifest.value.frames.length>1000||
      new Set(manifest.value.frames.map(frame=>frame.id)).size!==manifest.value.frames.length||
      manifest.value.frames.some(frame=>typeof frame.id!=='string'||!HASH.test(frame.sha256)||typeof frame.sourceId!=='string'||
        !Number.isSafeInteger(frame.width)||!Number.isSafeInteger(frame.height)||frame.width<16||frame.height<16||frame.width*frame.height>20000000))throw new Error('Public collection is invalid.');
    const identity=activeIdentity(active),key=sha(active.checkpointSha256+':'+manifest.sha256);
    if(identity!==activeIdentity(this.learning.active))throw new Error('Active pointer changed.');
    const contextSha256=sha(JSON.stringify({identity,manifest:manifest.sha256,card:card.sha256,complete:complete.sha256}));
    return {active,job,dataset,card,complete,manifest,identity,key,contextSha256};
  }
  async recheck(expected,{permissions=true,images=false}={}){
    this.available();
    if(permissions){await this.bridge.beforeWork();this.available();await this.learning.recheckDataset(expected.dataset);}
    const current=await this.context();
    if(current.contextSha256!==expected.contextSha256)throw new Error('Refresh inputs changed.');
    if(images){
      const base=await realpath(this.collectionDir);
      for(const frame of current.manifest.value.frames){const path=await realpath(inside(base,frame.path));inside(base,path);if(await fileSha(path,8*1024*1024)!==frame.sha256)throw new Error('Public image bytes changed.');}
    }
    this.available();if(activeIdentity(this.learning.active)!==expected.identity)throw new Error('Active pointer changed.');
  }
  async output(context,path){
    const {value,sha256}=await json(path,48*1024*1024),run=value.run,frames=context.manifest.value.frames;
    if(value.schema!=='mycoscout.public-ai-suggestions.v1'||run?.status!=='completed'||
      run.manifestSha256!==context.manifest.sha256||run.checkpointSha256!==context.active.checkpointSha256||
      run.modelVersion!=='reviewed-'+context.active.checkpointSha256.slice(0,16)||run.humanVerified!==false||run.trainingAllowed!==false||
      run.reviewedFrames!==frames.length||run.totalFrames!==frames.length||!Array.isArray(value.frames)||value.frames.length!==frames.length)throw new Error('Incomplete suggestions.');
    let modelIdentity=null;
    for(let i=0;i<frames.length;i++){
      const frame=frames[i],row=value.frames[i],model=row?.model;
      if(row?.frameId!==frame.id||row.imageSha256!==frame.sha256||row.sourceId!==frame.sourceId||row.status!=='completed'||
        model?.checkpointSha256!==context.active.checkpointSha256||model.modelCardSha256!==context.card.sha256||
        model.completeSha256!==context.complete.sha256||model.version!==run.modelVersion||
        model.model!==context.card.value.architecture||model.boxThreshold!==context.card.value.scoreThreshold||
        model.resolvedRevision!=='sha256:'+context.active.checkpointSha256||model.cropPolicy!=='full_frame'||model.nmsIou!==0.5||model.maxBoxes!==64||
        model.humanVerified!==false||model.trainingAllowed!==false||!finite(model.boxThreshold)||model.boxThreshold<0||model.boxThreshold>1||
        !Array.isArray(row.boxes)||row.boxes.length>64)throw new Error('Suggestion provenance changed.');
      modelIdentity??=JSON.stringify(model);if(JSON.stringify(model)!==modelIdentity)throw new Error('Suggestion models are inconsistent.');
      for(const box of row.boxes){
        const b=box?.bbox;
        if(!Array.isArray(b)||b.length!==4||!b.every(finite)||b[0]<0||b[1]<0||b[0]>=b[2]||b[1]>=b[3]||b[2]>frame.width||b[3]>frame.height||
          !finite(box.score)||box.score<model.boxThreshold||box.score>1||box.humanVerified!==false||box.trainingAllowed!==false||
          box.label!=='Unverified mushroom candidate')throw new Error('Invalid unverified suggestion.');
      }
    }
    return sha256;
  }
  tick(){
    const reason=this.closed||this.learning.closed?'stopped':!this.bridge.state.enabled?'disabled':!this.learning.active?'no_active_model':
      this.running||this.learning.busy?'busy':this.now()<this.state.nextAttemptAt?'retry_wait':null;
    if(reason)return Promise.resolve({skipped:true,reason});
    // Reserve synchronously so training, activation and concurrent ticks cannot
    // start while permission checks yield control.
    this.learning.busy=true;
    this.running=this.perform().finally(()=>{this.learning.busy=false;this.running=null;this.child=null;});
    return this.running;
  }
  async perform(){
    try{
      const context=await this.context();
      if(this.state.completed.some(row=>row.key===context.key)){this.state.status='current';return {skipped:true,reason:'current'};}
      if(this.state.completed.length>=4096)throw new Error('Completed refresh history is full.');
      await this.recheck(context);
      if(this.state.pending?.contextSha256!==context.contextSha256){
        this.state.pending={key:context.key,contextSha256:context.contextSha256,output:'suggestions-'+randomUUID()+'.json',createdAt:new Date(this.now()).toISOString()};
      }
      const pending=this.state.pending,path=resolve(this.directory,pending.output),cancel=path+'.cancel';
      await unlink(cancel).catch(error=>{if(error.code!=='ENOENT')throw error;});
      this.state.status='refreshing';this.state.error=null;this.progress={};await this.persist();
      this.available();
      const work=this.learning.runProcess(['-m','vision.refresh_community_suggestions','--collection',this.collectionDir,
        '--active-job',context.job.outputDirectory,'--output',path,'--cache-dir',resolve(this.directory,'cache'),
        '--cancel-file',cancel,'--max-frames','1000','--max-seconds',String(this.maxSeconds)],this.timeoutMs,this.progress);
      this.child=this.learning.child;await work;this.child=null;
      const outputSha256=await this.output(context,path);
      await this.recheck(context,{images:true});
      if(await fileSha(path,48*1024*1024)!==outputSha256)throw new Error('Suggestions changed before publication.');
      this.state.status='publishing';pending.outputSha256=outputSha256;await this.persist();
      this.available();if(activeIdentity(this.learning.active)!==context.identity)throw new Error('Active pointer changed.');
      // Fixed existing publisher only; repeat after unknown outcome uploads the
      // same validated bytes. No public endpoint controls this private worker.
      await this.bridge.run(['update-ai','--file',path]);
      await this.recheck(context);
      this.state.completed.push({key:context.key,outputSha256,completedAt:new Date(this.now()).toISOString()});
      Object.assign(this.state,{pending:null,status:'current',failures:0,nextAttemptAt:0,error:null,lastCompletedAt:new Date(this.now()).toISOString(),lastFrameCount:context.manifest.value.frames.length});
      await this.persist();return {completed:true,frames:this.state.lastFrameCount};
    }catch{
      this.state.failures=Math.min(30,this.state.failures+1);
      this.state.nextAttemptAt=this.now()+Math.min(1800000,this.retryMs*2**Math.min(10,this.state.failures-1));
      this.state.status=this.closed?'stopped':'retry_wait';this.state.error=this.closed?null:'Suggestion refresh did not complete. Existing suggestions remain available; the private worker will retry.';
      await this.persist().catch(()=>{});return {failed:true,stopped:this.closed};
    }
  }
  async close(){
    this.closed=true;
    if(this.state.pending)await writeFile(resolve(this.directory,this.state.pending.output+'.cancel'),'stop\n').catch(()=>{});
    const child=this.child,timer=child?setTimeout(()=>{if(this.child===child)child.kill();},5000):null;
    try{await Promise.allSettled([this.running,this.writes]);}finally{if(timer)clearTimeout(timer);}
  }
}
