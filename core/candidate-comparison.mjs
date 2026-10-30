import { mkdir, readFile, readdir, writeFile, rename, access } from 'node:fs/promises';
import { resolve } from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import { ApiError } from './validation.mjs';

const sha=bytes=>createHash('sha256').update(bytes).digest('hex');
const exists=path=>access(path).then(()=>true,()=>false);
const read=async path=>JSON.parse(await readFile(path,'utf8'));
const safeId=/^comparison-[a-f0-9-]{36}$/;
const summary=report=>({baseline:report.baseline,candidate:report.candidate,support:report.support,reasons:report.reasons,limitations:report.limitations});

export class CandidateComparisons {
  constructor(directory,{learning,beforeWork=async()=>{}}){this.directory=resolve(directory);this.learning=learning;this.beforeWork=beforeWork;this.requests=new Map();this.pending=null;}
  async initialize(){
    await mkdir(this.directory,{recursive:true});
    for(const name of await readdir(this.directory))if(name.endsWith('.json')&&safeId.test(name.slice(0,-5))){
      try{const item=await read(resolve(this.directory,name));if(item.id!==name.slice(0,-5))continue;if(item.status==='running')item.status='interrupted';this.requests.set(item.id,item);}catch{}
    }
    return this;
  }
  async persist(item){const path=resolve(this.directory,item.id+'.json'),tmp=path+'.tmp';await writeFile(tmp,JSON.stringify(item,null,2)+'\n');await rename(tmp,path);}
  context(jobId,split){
    if(!['validation','test'].includes(split))throw new ApiError('Choose validation or the explicit release test.');
    const job=this.learning.job(jobId),dataset=this.learning.datasets.get(job.datasetId);
    if(job.status!=='completed'||!dataset||!job.modelCard?.checkpointSha256)throw new ApiError('Comparison needs a completed candidate and its frozen dataset.',409,'COMPARISON_NOT_READY');
    const active=this.learning.active,baseline=active?this.learning.job(active.jobId):null;
    if(active&&(baseline.status!=='completed'||baseline.modelCard?.checkpointSha256!==active.checkpointSha256))throw new ApiError('The active baseline could not be verified.',409,'BASELINE_CHANGED');
    const baselineDataset=baseline?this.learning.datasets.get(baseline.datasetId):null;
    if(baseline&&!baselineDataset)throw new ApiError('The active baseline dataset is missing.',409,'BASELINE_CHANGED');
    const key=sha(JSON.stringify({jobId,checkpoint:job.modelCard.checkpointSha256,dataset:dataset.datasetId,split,baselineId:baseline?.id??'installed-grounding-dino',baselineHash:active?.checkpointSha256??null}));
    return {job,dataset,baseline,baselineDataset,key,split};
  }
  args(context,output,verify=false){return ['-m','vision.compare_candidate','--dataset',context.dataset.directory,'--candidate-job',context.job.outputDirectory,'--output',output,
    '--split',context.split,...(context.baseline?['--baseline-job',context.baseline.outputDirectory,'--baseline-dataset',context.baselineDataset.directory]:[]),...(verify?['--validate-only']:[])];}
  async verify(item,context){
    const output=resolve(this.directory,item.id);
    await this.learning.runProcess(this.args(context,output,true),120000);
    const bytes=await readFile(resolve(output,'comparison.json'));if(bytes.length>4*1024*1024)throw new Error('Comparison report exceeded its limit.');
    const report=JSON.parse(bytes);
    if(report.schema!=='mycoscout.candidate-baseline-comparison.v1'||report.status!=='completed'||report.split!==context.split||report.datasetId!==context.job.datasetId||
      report.candidateCheckpointSha256!==context.job.modelCard.checkpointSha256||!['candidate_better','candidate_not_better','insufficient_evidence'].includes(report.decision)||
      report.activationAllowed!==false||report.automaticActivation!==false)throw new ApiError('Comparison identity or evidence changed.',409,'COMPARISON_CHANGED');
    const baselineKind=report.models?.baseline?.kind;
    if(context.baseline?report.models?.baseline?.checkpointSha256!==context.baseline.modelCard.checkpointSha256:baselineKind!=='installed_grounding_dino')throw new ApiError('Comparison baseline changed.',409,'BASELINE_CHANGED');
    return {...report,comparisonId:item.id,reportSha256:sha(bytes),completedAt:item.finishedAt??report.completedAt??new Date().toISOString()};
  }
  async compare(jobOrId,{split='validation'}={}){
    const jobId=typeof jobOrId==='string'?jobOrId:jobOrId.id;
    if(this.pending)throw new ApiError('Another candidate comparison is running.',409,'LEARNING_BUSY');
    this.learning.checkIdle();this.learning.busy=true;
    let item;
    try{
      await this.beforeWork();const context=this.context(jobId,split);await this.learning.recheckDataset(context.dataset);
      if(split==='test'){
        const validation=this.context(jobId,'validation');
        const prior=[...this.requests.values()].reverse().find(row=>row.key===validation.key&&row.status==='completed'&&row.decision==='candidate_better');
        if(!prior||(await this.verify(prior,validation)).decision!=='candidate_better')throw new ApiError('A passing validation comparison is required before the explicit release test.',409,'VALIDATION_COMPARISON_REQUIRED');
      }
      for(const prior of [...this.requests.values()].reverse())if(prior.key===context.key&&await exists(resolve(this.directory,prior.id,'COMPLETE.json'))){
        const report=await this.verify(prior,context);await this.beforeWork();await this.learning.recheckDataset(context.dataset);prior.status='completed';prior.finishedAt??=new Date().toISOString();prior.decision=report.decision;prior.summary=summary(report);await this.persist(prior);return report;
      }
      item={id:'comparison-'+randomUUID(),key:context.key,jobId,split,baselineJobId:context.baseline?.id??null,status:'running',createdAt:new Date().toISOString()};
      this.requests.set(item.id,item);await this.persist(item);this.pending=item.id;
      await this.learning.runProcess(this.args(context,resolve(this.directory,item.id)),this.learning.timeoutMs);
      const report=await this.verify(item,context);await this.beforeWork();await this.learning.recheckDataset(context.dataset);item.status='completed';item.decision=report.decision;item.summary=summary(report);item.finishedAt=new Date().toISOString();await this.persist(item);return {...report,completedAt:item.finishedAt};
    }catch(error){if(item){item.status='failed';item.error='Candidate comparison failed; inspect the private worker log.';item.finishedAt=new Date().toISOString();await this.persist(item);}throw error;}
    finally{this.pending=null;this.learning.busy=false;}
  }
  status(){return [...this.requests.values()].slice(-30).reverse().map(({id,jobId,split,status,createdAt,finishedAt,decision,error,summary})=>({id,jobId,split,status,createdAt,finishedAt,decision,error,summary}));}
  async releaseCheck(job){
    await this.beforeWork();const context=this.context(job.id,'test');
    const item=[...this.requests.values()].reverse().find(row=>row.key===context.key&&row.status==='completed'&&row.decision==='candidate_better');
    if(!item)throw new ApiError('Run the explicit release comparison and review a passing result before activation.',409,'RELEASE_COMPARISON_REQUIRED');
    const report=await this.verify(item,context);
    await this.beforeWork();await this.learning.recheckDataset(context.dataset);
    if(report.decision!=='candidate_better'||report.testEvaluationPerformed!==true)throw new ApiError('The candidate has not passed its release comparison.',409,'RELEASE_COMPARISON_FAILED');
    return {approved:true,comparisonId:item.id,reportSha256:report.reportSha256,candidateCheckpointSha256:report.candidateCheckpointSha256,
      datasetManifestSha256:report.datasetManifestSha256,baselineJobId:context.baseline?.id??null,decision:report.decision,reviewedTestMetrics:report.candidate};
  }
}
