import { mkdir,open,readFile,rename,unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash,randomUUID } from 'node:crypto';
import { ApiError } from './validation.mjs';

const DAY=86400000,MAX_HISTORY=200,MAX_FRAMES=100000,MAX_KEYS=100000;
const HASH=/^[a-f0-9]{64}$/;
const clone=value=>structuredClone(value);
const stable=value=>Array.isArray(value)?value.map(stable):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,stable(value[key])])):value;
const hash=value=>createHash('sha256').update(JSON.stringify(stable(value))).digest('hex');
const DEFAULTS={enabled:false,minimumNewFrames:50,minimumNewSessions:3,maxRunsPerDay:1,maxAttempts:2,retryDelayMs:300000,
  recipe:{epochs:3,learningRate:.005,scoreThreshold:.3,validationOnly:true}};
const STATES=new Set(['disabled','paused','waiting_data','waiting_daily_limit','waiting_resource','waiting_sync','exporting','training','comparing','retry_wait','candidate_ready','failed']);
const MESSAGES={disabled:'Automatic candidate training is off.',paused:'Automatic candidate training is paused.',waiting_data:'Waiting for enough newly approved frames and source sessions.',
  waiting_daily_limit:'Waiting for the next permitted training window.',waiting_resource:'Waiting for the local training worker.',waiting_sync:'Waiting for current review permissions.',
  exporting:'Preparing a reviewed dataset.',training:'Training a candidate model.',comparing:'Comparing the candidate on validation data.',retry_wait:'Waiting before a bounded retry.',
  candidate_ready:'A candidate comparison is ready for human review.',failed:'The automatic cycle needs local review.'};

function config(input,prior=DEFAULTS){
  if(!input||typeof input!=='object'||Array.isArray(input))throw new ApiError('Use a learning-loop configuration object.');
  const allowed=new Set(Object.keys(DEFAULTS));if(Object.keys(input).some(key=>!allowed.has(key)))throw new ApiError('Unknown learning-loop setting.');
  const result={...clone(prior),...input,recipe:{...prior.recipe,...input.recipe,validationOnly:true}};
  if(typeof result.enabled!=='boolean')throw new ApiError('enabled must be a boolean.');
  for(const [key,min,max] of [['minimumNewFrames',1,100000],['minimumNewSessions',1,10000],['maxRunsPerDay',1,10],['maxAttempts',1,3],['retryDelayMs',1000,DAY]])
    if(!Number.isInteger(result[key])||result[key]<min||result[key]>max)throw new ApiError(`Invalid ${key}.`);
  if(input.recipe!==undefined&&(!input.recipe||typeof input.recipe!=='object'||Array.isArray(input.recipe)||Object.keys(input.recipe).some(key=>!['epochs','learningRate','scoreThreshold','validationOnly'].includes(key))||input.recipe.validationOnly===false))throw new ApiError('Automatic recipes must use validation only.');
  const recipe=result.recipe;
  if(!Number.isInteger(recipe.epochs)||recipe.epochs<1||recipe.epochs>30||!Number.isFinite(recipe.learningRate)||recipe.learningRate<.00001||recipe.learningRate>.02||
    !Number.isFinite(recipe.scoreThreshold)||recipe.scoreThreshold<.05||recipe.scoreThreshold>.95)throw new ApiError('Invalid automatic training recipe.');
  return result;
}

function initial(now){return {schema:'mycoscout.learning-loop.v1',revision:0,updatedAt:new Date(now).toISOString(),config:clone(DEFAULTS),paused:false,
  workerState:'disabled',reason:null,counts:{approvedFrames:null,newFrames:null,newSessions:null},consumedFrames:[],consumedSessions:[],handled:{},launches:[],history:[],
  active:null,latestCandidate:null,lastReadiness:null,nextCheckAt:0};}
const alive=pid=>{try{process.kill(pid,0);return true;}catch(error){return error.code!=='ESRCH';}};

// Aggregate changes without counting duplicate pixels or repeated frames from
// one source as new independent capture sessions. All identities stay local.
export function learningLoopEvidence(snapshot,state,recipe){
  if(!snapshot||!Array.isArray(snapshot.records)||snapshot.records.length>MAX_FRAMES)throw new ApiError('Reviewed snapshot is missing or too large.',409,'LOOP_SNAPSHOT');
  const blockedHashes=new Set(snapshot.evaluationHashes??[]),blockedKeys=new Set();
  for(const source of snapshot.evaluationSources??[])for(const key of source.groupKeys??[])blockedKeys.add(key);
  for(const [key,split] of Object.entries(snapshot.frozenPartitions??{}))if(split==='validation'||split==='test')blockedKeys.add(key);
  const rows=snapshot.records.filter(row=>row?.complete===true&&row.humanVerified===true&&row.trainingAllowed===true&&row.evaluationOnly===false&&
    ['visible','absent'].includes(row.status)&&HASH.test(row.originalSha256??'')&&Array.isArray(row.groupKeys)&&row.groupKeys.length<=30);
  const parents=rows.map((_,index)=>index),find=index=>parents[index]===index?index:(parents[index]=find(parents[index])),owners=new Map();
  for(const [index,row] of rows.entries())for(const key of [...row.groupKeys,'pixels:'+row.originalSha256]){
    if(typeof key!=='string'||key.length>500)throw new ApiError('Invalid source grouping in reviewed snapshot.',409,'LOOP_SNAPSHOT');
    if(owners.has(key))parents[find(index)]=find(owners.get(key));else owners.set(key,index);
  }
  const groups=new Map();for(const [index,row] of rows.entries()){const id=find(index);if(!groups.has(id))groups.set(id,[]);groups.get(id).push(row);}
  const consumed=new Set(state.consumedFrames),consumedSessions=new Set(state.consumedSessions),approved=new Set(),newFrames=new Set(),sessionKeys=new Set();let newSessions=0;
  for(const members of groups.values()){
    if(members.some(row=>blockedHashes.has(row.originalSha256)||row.groupKeys.some(key=>blockedKeys.has(key))))continue;
    const keys=[...new Set(members.flatMap(row=>row.groupKeys).filter(key=>/^(source|video|session):/.test(key)))].map(hash);
    let fresh=false;for(const row of members){approved.add(row.originalSha256);if(!consumed.has(row.originalSha256)){newFrames.add(row.originalSha256);fresh=true;}}
    for(const key of keys)sessionKeys.add(key);
    if(fresh&&keys.length&&!keys.some(key=>consumedSessions.has(key)))newSessions++;
  }
  const sourceIdentity={sources:snapshot.sourceRevisionSignatures??Object.fromEntries(snapshot.records.map(row=>[row.id,hash(row)])),evaluationHashes:[...blockedHashes].sort(),
    evaluationGroups:[...blockedKeys].sort()};
  return {approvedFrames:approved.size,newFrames:newFrames.size,newSessions,frameKeys:[...approved].sort(),sessionKeys:[...sessionKeys].sort(),
    dataFingerprint:hash(sourceIdentity),fingerprint:hash({sourceIdentity,recipe,trainer:'reviewed-detector-validation-v1'})};
}

export class LearningLoop{
  constructor(directory,{learning,compareCandidate,beforeWork=async()=>{},now=()=>Date.now(),processAlive=alive}={}){
    if(!learning)throw new Error('A learning service is required.');
    this.directory=resolve(directory);this.learning=learning;this.compareCandidate=compareCandidate;this.beforeWork=beforeWork;this.now=now;this.processAlive=processAlive;
    this.state=initial(now());this.writeQueue=Promise.resolve();this.tickPromise=null;this.closed=false;this.token=randomUUID();this.locked=false;this.timer=null;
  }
  async initialize(){
    await mkdir(this.directory,{recursive:true});const path=resolve(this.directory,'learning-loop.lock');
    for(let attempt=0;attempt<2;attempt++){
      try{const handle=await open(path,'wx');try{await handle.writeFile(JSON.stringify({pid:process.pid,token:this.token}));await handle.sync();}finally{await handle.close();}this.locked=true;break;}
      catch(error){if(error.code!=='EEXIST')throw error;const original=await readFile(path,'utf8');let owner;try{owner=JSON.parse(original);}catch{throw new ApiError('Learning-loop lock needs local inspection.',409,'LOOP_LOCKED');}
        if(!Number.isSafeInteger(owner.pid)||owner.pid<1||this.processAlive(owner.pid))throw new ApiError('Another learning coordinator owns this directory.',409,'LOOP_LOCKED');
        if(await readFile(path,'utf8')!==original)throw new ApiError('Learning coordinator ownership changed.',409,'LOOP_LOCKED');await unlink(path);}
    }
    if(!this.locked)throw new ApiError('Could not acquire learning coordinator ownership.',409,'LOOP_LOCKED');
    try{
      let loaded;try{loaded=JSON.parse(await readFile(resolve(this.directory,'learning-loop.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
      if(loaded){
        if(loaded.schema!=='mycoscout.learning-loop.v1'||!STATES.has(loaded.workerState)||!Number.isSafeInteger(loaded.revision)||!Array.isArray(loaded.consumedFrames)||loaded.consumedFrames.length>MAX_FRAMES||
          !Array.isArray(loaded.consumedSessions)||loaded.consumedSessions.length>MAX_KEYS||[...loaded.consumedFrames,...loaded.consumedSessions].some(key=>!HASH.test(key))||
          !Array.isArray(loaded.history)||loaded.history.length>MAX_HISTORY||!loaded.handled||Object.keys(loaded.handled).length>10000||!Array.isArray(loaded.launches))throw new Error('Saved learning-loop state is invalid; it was not replaced.');
        if(loaded.active&&(!HASH.test(loaded.active.fingerprint??'')||!['export','train','monitor','compare'].includes(loaded.active.stage)||!Number.isInteger(loaded.active.attempt)||loaded.active.attempt<0||loaded.active.attempt>2||
          !Array.isArray(loaded.active.frameKeys)||loaded.active.frameKeys.length>MAX_FRAMES||!Array.isArray(loaded.active.sessionKeys)||loaded.active.sessionKeys.length>MAX_KEYS||
          [...loaded.active.frameKeys,...loaded.active.sessionKeys].some(key=>!HASH.test(key))||loaded.active.recipe?.validationOnly!==true))throw new Error('Saved automatic cycle is invalid; it was not replaced.');
        loaded.config=config(loaded.config);this.state=loaded;
      }else await this.persist();
      return this;
    }catch(error){await this.close();throw error;}
  }
  async persist(){
    this.state.revision++;this.state.updatedAt=new Date(this.now()).toISOString();const value=JSON.stringify(this.state,null,2)+'\n';
    const write=async()=>{const temporary=resolve(this.directory,'learning-loop.'+randomUUID()+'.tmp'),handle=await open(temporary,'wx');try{await handle.writeFile(value);await handle.sync();}finally{await handle.close();}await rename(temporary,resolve(this.directory,'learning-loop.json'));};
    const pending=this.writeQueue.then(write);this.writeQueue=pending.catch(()=>{});await pending;
  }
  status(value,code,message){this.state.workerState=value;this.state.reason=code?{code,message:String(message??MESSAGES[value]).slice(0,1000)}:null;}
  snapshot(){return clone({...this.state,automaticActivation:false});}
  async configure(input){this.state.config=config(input,this.state.config);if(!this.state.config.enabled)this.status('disabled');await this.persist();return this.snapshot();}
  async pause(){this.state.paused=true;this.status('paused');await this.persist();return this.snapshot();}
  async resume(){this.state.paused=false;this.state.config.enabled=true;this.state.nextCheckAt=0;this.status('waiting_data');await this.persist();return this.snapshot();}
  async cancel(){const active=this.state.active;if(active){active.cancelRequested=true;await this.persist();if(active.jobId)await this.learning.cancel(active.jobId);if(this.state.active?.id===active.id)await this.finish(active,'cancelled','The automatic cycle was cancelled.');}return this.snapshot();}
  start(intervalMs=60000){if(this.timer)return;if(!Number.isInteger(intervalMs)||intervalMs<1000)throw new Error('Use a poll interval of at least one second.');this.timer=setInterval(()=>{this.tick().catch(()=>{});},intervalMs);this.timer.unref?.();return this;}
  async close(){this.closed=true;if(this.timer)clearInterval(this.timer);this.timer=null;if(this.tickPromise)await this.tickPromise;await this.writeQueue;
    if(this.locked){const path=resolve(this.directory,'learning-loop.lock');try{const value=JSON.parse(await readFile(path,'utf8'));if(value.token===this.token)await unlink(path);}catch(error){if(error.code!=='ENOENT')throw error;}this.locked=false;}
  }
  tick(){if(this.tickPromise)return this.tickPromise;if(this.closed)return Promise.resolve(this.snapshot());
    this.tickPromise=this.runTick().finally(()=>{this.tickPromise=null;});return this.tickPromise;
  }
  async observe(){const evidence=learningLoopEvidence(await this.learning.sourceSnapshot(),this.state,this.state.config.recipe);
    this.state.counts={approvedFrames:evidence.approvedFrames,newFrames:evidence.newFrames,newSessions:evidence.newSessions};return evidence;}
  async preflight(stage,active){
    try{await this.beforeWork(stage,clone(active));return this.state.active?.id===active.id&&!active.cancelRequested&&!this.state.paused&&this.state.config.enabled;}
    catch(error){this.state.nextCheckAt=this.now()+this.state.config.retryDelayMs;this.status('waiting_sync','SYNC_UNAVAILABLE',error.message);await this.persist();return false;}
  }
  async finish(active,outcome,message,consume=false){
    if(this.state.active?.id!==active.id)return;
    if(consume){this.state.consumedFrames=[...new Set([...this.state.consumedFrames,...active.frameKeys])];this.state.consumedSessions=[...new Set([...this.state.consumedSessions,...active.sessionKeys])];}
    this.state.handled[active.fingerprint]={outcome,at:new Date(this.now()).toISOString()};
    this.state.history.push({id:active.id,fingerprint:active.fingerprint,outcome,message:String(message??'').slice(0,1000),jobId:active.jobId??null,datasetId:active.datasetId??null,at:new Date(this.now()).toISOString()});
    this.state.history=this.state.history.slice(-MAX_HISTORY);this.state.active=null;this.state.nextCheckAt=0;
    this.status(outcome==='completed'?'candidate_ready':outcome==='rejected'?'waiting_data':'failed',outcome.toUpperCase(),message);await this.persist();
  }
  async fail(active,error){
    if(this.state.active?.id!==active.id)return;
    if(['LEARNING_SOURCE_CHANGED','LEARNING_IDEMPOTENCY_CONFLICT','LEARNING_INSUFFICIENT_DATA'].includes(error.code))return this.finish(active,'rejected',error.message);
    if(error.code==='LEARNING_BUSY'){this.status('waiting_resource','LEARNING_BUSY',error.message);await this.persist();return;}
    active.errors=(active.errors??0)+1;
    if(active.errors>=this.state.config.maxAttempts){
      if(active.stage==='compare')this.state.latestCandidate={jobId:active.jobId,status:'comparison_failed',completedAt:new Date(this.now()).toISOString(),result:null};
      return this.finish(active,'failed',error.message);
    }
    this.state.nextCheckAt=this.now()+this.state.config.retryDelayMs;this.status('retry_wait',error.code??'WORK_FAILED',error.message);await this.persist();
  }
  async runTick(){
    try{
      const evidence=await this.observe();
      if(!this.state.config.enabled){this.status('disabled');await this.persist();return this.snapshot();}
      if(this.state.paused){this.status('paused');await this.persist();return this.snapshot();}
      if(this.now()<this.state.nextCheckAt){await this.persist();return this.snapshot();}
      let active=this.state.active;
      if(!active){
        if(evidence.newFrames<this.state.config.minimumNewFrames||evidence.newSessions<this.state.config.minimumNewSessions){this.status('waiting_data','INSUFFICIENT_NEW_DATA');await this.persist();return this.snapshot();}
        if(this.state.handled[evidence.fingerprint]){const prior=this.state.handled[evidence.fingerprint];this.status(prior.outcome==='completed'?'candidate_ready':prior.outcome==='rejected'?'waiting_data':'failed','ALREADY_HANDLED','This dataset and recipe were already handled. New eligible evidence is required.');await this.persist();return this.snapshot();}
        if(Object.keys(this.state.handled).length>=10000||new Set([...this.state.consumedFrames,...evidence.frameKeys]).size>MAX_FRAMES||new Set([...this.state.consumedSessions,...evidence.sessionKeys]).size>MAX_KEYS){this.status('failed','HISTORY_FULL','The local evidence ledger requires archival.');await this.persist();return this.snapshot();}
        active={id:'cycle-'+randomUUID(),...evidence,recipe:clone(this.state.config.recipe),stage:'export',attempt:0,errors:0,createdAt:new Date(this.now()).toISOString(),cancelRequested:false};
        this.state.active=active;await this.persist();
      }
      if(active.cancelRequested){await this.finish(active,'cancelled','The automatic cycle was cancelled.');return this.snapshot();}
      if(active.stage==='export'){
        if(this.learning.busy){this.status('waiting_resource');await this.persist();return this.snapshot();}
        if(!await this.preflight('export',active))return this.snapshot();
        const fresh=await this.observe();if(fresh.fingerprint!==active.fingerprint){await this.finish(active,'rejected','Reviewed labels or permissions changed before export.');return this.snapshot();}
        this.status('exporting');await this.persist();
        const dataset=await this.learning.export({idempotencyKey:'loop-export-'+active.fingerprint});
        if(this.state.active?.id!==active.id)return this.snapshot();
        this.state.lastReadiness=clone(dataset.readiness??null);active.datasetId=dataset.datasetId;
        if(dataset.readiness?.ready!==true){await this.finish(active,'rejected',(dataset.readiness?.reasons??['Dataset is not ready.']).join('; '));return this.snapshot();}
        active.stage='train';active.errors=0;await this.persist();return this.snapshot();
      }
      if(active.stage==='train'){
        this.state.launches=this.state.launches.filter(row=>row.at>this.now()-DAY);
        const launchKey=active.id+':'+active.attempt,reserved=this.state.launches.some(row=>row.key===launchKey);
        if(!reserved&&this.state.launches.length>=this.state.config.maxRunsPerDay){this.status('waiting_daily_limit');await this.persist();return this.snapshot();}
        if(this.learning.busy&&!reserved){this.status('waiting_resource');await this.persist();return this.snapshot();}
        if(!await this.preflight('train',active))return this.snapshot();
        if(!reserved)this.state.launches.push({key:launchKey,at:this.now()});
        this.status('training');await this.persist();
        const job=await this.learning.startJob({...active.recipe,datasetId:active.datasetId,idempotencyKey:'loop-job-'+active.fingerprint+'-'+active.attempt,validationOnly:true});
        if(this.state.active?.id!==active.id){if(['running','queued'].includes(job.status))await this.learning.cancel(job.id);return this.snapshot();}
        active.jobId=job.id;active.stage='monitor';active.errors=0;await this.persist();return this.snapshot();
      }
      if(active.stage==='monitor'){
        const job=await this.learning.job(active.jobId);
        if(['queued','running'].includes(job.status)){this.status('training');await this.persist();return this.snapshot();}
        if(job.status==='completed'){active.stage='compare';active.errors=0;this.status('comparing');await this.persist();return this.snapshot();}
        if(job.status==='cancelled'){await this.finish(active,'cancelled','The candidate job was cancelled.');return this.snapshot();}
        if(['failed','interrupted'].includes(job.status)){
          if(active.attempt+1>=this.state.config.maxAttempts){await this.finish(active,'failed',job.error??'Candidate training did not finish.');return this.snapshot();}
          active.attempt++;active.jobId=null;active.stage='train';this.state.nextCheckAt=this.now()+this.state.config.retryDelayMs;this.status('retry_wait','TRAINING_RETRY',job.error);await this.persist();return this.snapshot();
        }
        throw new Error('Training worker returned an unknown state.');
      }
      if(active.stage==='compare'){
        if(!await this.preflight('compare',active))return this.snapshot();
        if(typeof this.compareCandidate!=='function')throw new Error('Candidate comparison is not configured.');
        this.status('comparing');await this.persist();
        const result=await this.compareCandidate(await this.learning.job(active.jobId));
        if(this.state.active?.id!==active.id)return this.snapshot();
        if(result?.status!=='completed'||!['candidate_better','candidate_not_better','insufficient_evidence'].includes(result.decision)||result.activationAllowed!==false)throw new Error('Candidate comparison did not return a valid unactivated decision.');
        this.state.latestCandidate={jobId:active.jobId,status:result.decision,completedAt:new Date(this.now()).toISOString(),result:clone(result)};
        await this.finish(active,'completed','Validation comparison completed; human release review is required.',true);return this.snapshot();
      }
      throw new Error('Saved automatic cycle stage is invalid.');
    }catch(error){
      if(this.state.active)await this.fail(this.state.active,error);else{this.status('failed',error.code??'LOOP_FAILED',error.message);await this.persist();}
      return this.snapshot();
    }
  }
  publicSummary(){return sanitizeLearningLoopSummary(this.state);}
}

export function sanitizeLearningLoopSummary(state){
  const number=value=>Number.isSafeInteger(value)&&value>=0?value:null,fraction=value=>Number.isFinite(value)&&value>=0&&value<=1?value:null;
  const rate=value=>Number.isFinite(value)&&value>=0?value:null,candidate=state.latestCandidate,result=candidate?.result;
  const metrics=value=>value?.metrics??value;
  const baseline=metrics(result?.baseline),proposed=metrics(result?.candidate),validCandidate=candidate&&['candidate_better','candidate_not_better','insufficient_evidence','comparison_failed'].includes(candidate.status);
  const workerState=STATES.has(state.workerState)?state.workerState:'failed';
  return {schema:'mycoscout.community-learning-status.v1',updatedAt:state.updatedAt,worker:{state:workerState,enabled:state.config.enabled===true,paused:state.paused===true},message:MESSAGES[workerState],
    approvedFrames:number(state.counts?.approvedFrames),newFrames:number(state.counts?.newFrames),newSessions:number(state.counts?.newSessions),
    thresholds:{newFrames:number(state.config.minimumNewFrames),newSessions:number(state.config.minimumNewSessions),maxRunsPerDay:number(state.config.maxRunsPerDay)},
    latestCandidate:validCandidate?{status:candidate.status,completedAt:candidate.completedAt,comparison:{baselineRecall:fraction(baseline?.recall),candidateRecall:fraction(proposed?.recall),
      baselineFalsePositivesPerFrame:rate(baseline?.falsePositivesPerFrame),candidateFalsePositivesPerFrame:rate(proposed?.falsePositivesPerFrame)}}:null,automaticActivation:false};
}
