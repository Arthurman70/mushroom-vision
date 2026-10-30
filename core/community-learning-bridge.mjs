import { spawn } from 'node:child_process';
import { mkdir, readFile, writeFile, rename, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { randomUUID, createHash } from 'node:crypto';
import { importCommunity } from '../community/import-reviews.mjs';

const digest=bytes=>createHash('sha256').update(bytes).digest('hex');
const save=async(path,value)=>{const temp=path+'.tmp';await writeFile(temp,JSON.stringify(value,null,2)+'\n');await rename(temp,path);};

// Private SSH export/import and an aggregate-only outbound status. There is no
// public administrative API, and no contributor data is sent to a model vendor.
export class CommunityLearningBridge {
  constructor(root,directory,{python,run,importReviews=importCommunity,now=()=>Date.now(),syncIntervalMs=300000,inboxDir=resolve(root,'data/community-inbox-v1')}={}){
    this.root=resolve(root);this.directory=resolve(directory);this.python=python??resolve(root,'runtime-vision',process.platform==='win32'?'Scripts/python.exe':'bin/python');
    this.run=run??((args)=>this.execute(args));this.importReviews=importReviews;this.now=now;this.syncIntervalMs=syncIntervalMs;this.inboxDir=resolve(inboxDir);
    this.state={schema:'mycoscout.community-learning-bridge.v1',enabled:false,lastSyncAt:null,lastPublishAt:null,lastExportSha256:null,lastImported:0,lastTotal:0,error:null};
    this.syncing=null;this.publishing=null;this.writes=Promise.resolve();this.children=new Set();this.nextSyncAt=0;this.closed=false;
  }
  async initialize(){
    await mkdir(this.directory,{recursive:true});
    try{const saved=JSON.parse(await readFile(resolve(this.directory,'bridge.json'),'utf8'));if(saved.schema!==this.state.schema||typeof saved.enabled!=='boolean')throw new Error('Invalid community bridge configuration');this.state={...this.state,...saved};}catch(error){if(error.code!=='ENOENT')throw error;}
    return this;
  }
  async configure({enabled}){if(typeof enabled!=='boolean')throw new Error('Choose whether community synchronization is enabled.');this.state.enabled=enabled;await this.persist();return this.snapshot();}
  snapshot(){return structuredClone(this.state);}
  persist(){const state=structuredClone(this.state);const pending=this.writes.then(()=>save(resolve(this.directory,'bridge.json'),state));this.writes=pending.catch(()=>{});return pending;}
  execute(args){return new Promise((resolvePromise,reject)=>{
    if(this.closed)return reject(new Error('Community worker is stopping.'));
    const child=spawn(this.python,['deploy/community/deploy.py',...args],{cwd:this.root,windowsHide:true,stdio:['ignore','pipe','pipe']});this.children.add(child);let stdout='',stderr='',settled=false;
    const finish=(error)=>{if(settled)return;settled=true;this.children.delete(child);clearTimeout(timer);error?reject(error):resolvePromise(stdout);};
    const timer=setTimeout(()=>{child.kill();finish(new Error('Community synchronization timed out.'));},120000);
    child.stdout.on('data',chunk=>{stdout+=chunk;if(stdout.length>65536){child.kill();finish(new Error('Community synchronization output exceeded its limit.'));}});
    child.stderr.on('data',chunk=>{stderr=(stderr+chunk).slice(-2000);});
    child.on('error',()=>finish(new Error('Community synchronization could not start.')));
    child.on('exit',code=>finish(code===0?null:new Error('Community synchronization failed. Check the private worker log.')));
  });}
  async sync({force=false}={}){
    if(!this.state.enabled)return {skipped:true,reason:'disabled'};
    if(this.syncing)return this.syncing;
    if(!force&&this.now()<this.nextSyncAt)return {skipped:true,reason:'retry_wait'};
    const age=this.now()-Date.parse(this.state.lastSyncAt??'');
    if(!force&&Number.isFinite(age)&&age>=0&&age<this.syncIntervalMs)return {skipped:true,reason:'current'};
    this.nextSyncAt=this.now()+this.syncIntervalMs;
    this.syncing=this.performSync();try{return await this.syncing;}finally{this.syncing=null;}
  }
  async performSync(){
    const file=resolve(this.directory,'transfer-'+randomUUID()+'.json');
    try{
      await this.run(['export-reviews','--output',file]);
      const raw=await readFile(file);if(raw.length>32*1024*1024)throw new Error('Community review export exceeded its limit.');
      const hash=digest(raw),result=await this.importReviews(file,this.inboxDir);
      Object.assign(this.state,{lastSyncAt:new Date(this.now()).toISOString(),lastExportSha256:hash,lastImported:result.imported,lastTotal:result.total,error:null});
      await this.persist();return result;
    }catch(error){this.state.error='Review synchronization failed. Training waits for a fresh permissions check.';await this.persist();throw error;}
    finally{await unlink(file).catch(error=>{if(error.code!=='ENOENT')throw error;});}
  }
  async beforeWork(){
    if(this.state.enabled)return this.sync({force:true});
    let priorContributions=this.state.lastTotal>0;
    try{const inbox=JSON.parse(await readFile(resolve(this.inboxDir,'reviews.json'),'utf8'));priorContributions||=(inbox.reviews?.length??0)>0;}catch(error){if(error.code!=='ENOENT')throw error;}
    if(priorContributions)throw new Error('Reconnect community synchronization before using imported reviews; current permissions must be checked.');
  }
  async publish(summary,{force=false}={}){
    if(!this.state.enabled)return {skipped:true};
    if(this.publishing)return this.publishing;
    const age=this.now()-Date.parse(this.state.lastPublishAt??'');if(!force&&Number.isFinite(age)&&age>=0&&age<60000)return {skipped:true};
    if(summary?.schema!=='mycoscout.community-learning-status.v1')throw new Error('Only a public learning summary can be published.');
    this.publishing=(async()=>{
      const file=resolve(this.directory,'public-status.json');await save(file,summary);
      await this.run(['update-training-status','--file',file]);this.state.lastPublishAt=new Date(this.now()).toISOString();await this.persist();return {published:true};
    })();
    try{return await this.publishing;}finally{this.publishing=null;}
  }
  async close(){this.closed=true;for(const child of this.children)child.kill();await Promise.allSettled([this.syncing,this.publishing,this.writes]);}
}
