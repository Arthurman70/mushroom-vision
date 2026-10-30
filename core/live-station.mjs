import { mkdir, readFile, writeFile, readdir, stat, unlink, rename } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { ApiError } from './validation.mjs';
import { validateImage } from './finds-images.mjs';

// This station accepts observations and high-level tasks. It has no aircraft transport.
export class LiveStation {
  constructor(root,directory,mission,{python,collectionDir,testbenchDb,learningDir}={}) {
    Object.assign(this,{root:resolve(root),directory:resolve(directory),mission,python,collectionDir,testbenchDb,learningDir});
    this.replay=null;this.child=null;this.pending=Promise.resolve();this.replayFinished=null;this.closed=false;
  }
  async initialize(){this.imageDir=join(this.directory,'images');await mkdir(this.imageDir,{recursive:true});return this;}
  snapshot(){return {...this.mission.snapshot(),replay:this.replay};}
  async ingest(packet){
    const work=async()=>{
      if(this.closed)throw new ApiError('Live station is stopping.',503,'LIVE_STOPPED');
      if(this.child&&packet.sessionId!==(this.replay?.activeSessionId??this.replay?.sessionId))throw new ApiError('Stop the recorded replay before connecting another source.',409,'REPLAY_ACTIVE');
      const input=structuredClone(packet);let photo=null;
      if(input.image?.imageBase64){
        photo=validateImage(input.image);
        if(input.image.sha256&&input.image.sha256!==photo.sha256)throw new ApiError('Frame image hash does not match its bytes.');
        if(input.image.width!==photo.width||input.image.height!==photo.height)throw new ApiError('Frame image dimensions do not match its bytes.');
        input.image={width:photo.width,height:photo.height,sha256:photo.sha256,mimeType:photo.mimeType,url:'/api/live/images/'+photo.sha256,imageUrl:'/api/live/images/'+photo.sha256};
      }
      if(photo){const target=join(this.imageDir,photo.sha256+'.'+photo.extension),temp=target+'.tmp';await writeFile(temp,photo.bytes);await rename(temp,target);}
      const result=this.mission.ingestFrame(input);
      if(photo)await this.pruneImages();
      if(!result.duplicate&&input.simulationOnly===false&&input.flight)return {...result,...this.mission.tick({})};
      return result;
    };
    const result=this.pending.then(work);this.pending=result.catch(()=>{});return result;
  }
  async pruneImages(){
    const entries=await readdir(this.imageDir);if(entries.length<=256)return;
    const files=await Promise.all(entries.filter(n=>/^[a-f0-9]{64}\.(jpg|png)$/.test(n)).map(async name=>({name,time:(await stat(join(this.imageDir,name))).mtimeMs})));
    files.sort((a,b)=>a.time-b.time);for(const file of files.slice(0,Math.max(0,files.length-256)))await unlink(join(this.imageDir,file.name));
  }
  async image(hash){
    if(!/^[a-f0-9]{64}$/.test(hash))throw new ApiError('Invalid frame image.',400);
    for(const [extension,mimeType]of [['jpg','image/jpeg'],['png','image/png']])try{return {bytes:await readFile(join(this.imageDir,hash+'.'+extension)),mimeType};}catch(e){if(e.code!=='ENOENT')throw e;}
    throw new ApiError('This preview has expired. Original captures stay on the recording computer.',404,'IMAGE_EXPIRED');
  }
  async startReplay({source='collection',interval=0.8,colorSensitive=false}={}){
    if(this.closed)throw new ApiError('Live station is stopping.',503,'LIVE_STOPPED');
    if(this.child)throw new ApiError('A replay is already running.',409,'REPLAY_ACTIVE');
    if(!['collection','fixture','fresh'].includes(source)||typeof colorSensitive!=='boolean'||!Number.isFinite(interval)||interval<0.1||interval>10)throw new ApiError('Choose a replay source and an interval from 0.1 to 10 seconds.');
    const sessionId='replay-'+randomUUID();
    this.mission.startSession({sessionId,mode:'replay'});
    const args=source==='fresh'?['-u','-m','adapters.live_capture','--input',join(this.root,'data','archive-video-v1','originals','Freestonew-theHugeMushroom948.flv'),
      '--emit','--session-id',sessionId,'--sample-every','20','--max-frames','12','--classify-groups',...(colorSensitive?['--color-sensitive']:[])]:
      ['-u','-m','adapters.live_replay','--emit','--session-id',sessionId,'--interval',String(interval)];
    if(source==='fixture')args.push('--fixture',join(this.root,'examples','live-session-synthetic.json'));
    else if(source==='collection')args.push('--classify-groups','--collection',this.collectionDir??join(this.root,'data','video-discovery-v1'),'--testbench-db',this.testbenchDb??join(this.root,'data','testbench','testbench.sqlite'));
    else if(source==='fresh'){
      const pointer=join(this.learningDir??join(this.root,'data','learning'),'active-model.json');
      try{await stat(pointer);args.push('--active-model',pointer);}catch(error){if(error.code!=='ENOENT')throw error;}
    }
    const child=spawn(this.python??join(this.root,'runtime-vision',process.platform==='win32'?'Scripts':'bin',process.platform==='win32'?'python.exe':'python'),args,{cwd:this.root,windowsHide:true,stdio:['ignore','pipe','pipe']});
    this.child=child;this.replay={sessionId,source,status:'running',frames:0,startedAt:new Date().toISOString(),error:null};
    let buffer='',diagnostics='',failure=null;const job=this.replay;
    const consume=async line=>{if(!line.trim())return;const packet=JSON.parse(line);
      if(packet.sessionId!==sessionId&&!packet.sessionId?.startsWith(sessionId+'-'))throw new Error('Replay emitted an unrelated session.');
      if(this.mission.snapshot().sessionId!==packet.sessionId)this.mission.startSession({sessionId:packet.sessionId,mode:'replay'});
      job.activeSessionId=packet.sessionId;await this.ingest(packet);job.frames++;
    };
    child.stdout.on('data',chunk=>{
      buffer+=chunk.toString('utf8');
      if(buffer.length>16*1024*1024){failure='Replay frame exceeded its size limit.';child.kill();return;}
      const lines=buffer.split('\n');buffer=lines.pop();
      child.stdout.pause();
      queue=queue.then(async()=>{for(const line of lines)await consume(line);}).catch(error=>{failure=error.message;child.kill();}).finally(()=>child.stdout.resume());
    });
    let queue=Promise.resolve();
    child.stderr.on('data',chunk=>{diagnostics=(diagnostics+chunk.toString('utf8')).slice(-4000);});
    child.on('error',error=>{failure=error.message;});
    this.replayFinished=new Promise(resolveFinished=>child.on('close',async code=>{
      try{
      await queue;await this.pending;
      if(buffer.trim()&&!failure)try{await consume(buffer);}catch(error){failure=error.message;}
      job.status=job.status==='stopping'?'cancelled':failure||code!==0?'failed':'completed';
      job.error=failure||(code!==0?diagnostics||'Replay process failed.':null);job.finishedAt=new Date().toISOString();
      if(this.child===child)this.child=null;
      }finally{resolveFinished();}
    }));
    return this.replay;
  }
  stopReplay(){if(this.child){this.replay.status='stopping';this.child.kill();}return this.replay;}
  async close(){this.stopReplay();if(this.replayFinished)await this.replayFinished;await this.pending;this.closed=true;}
}
