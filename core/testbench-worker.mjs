import { spawn } from 'node:child_process';
import { access } from 'node:fs/promises';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { ApiError } from './validation.mjs';
import { testbenchProgress } from './testbench-validation.mjs';

export class TestbenchWorker {
  constructor(root,{timeoutMs=240000,spawnWorker=spawn}={}){this.root=root;this.timeoutMs=timeoutMs;this.spawnWorker=spawnWorker;this.process=null;this.pending=null;this.buffer='';this.closed=false;this.loaded=false;}
  executable(){return process.env.MYCOSCOUT_PYTHON??resolve(this.root,'runtime-vision',process.platform==='win32'?'Scripts':'bin',process.platform==='win32'?'python.exe':'python');}
  async status(){const paths=[this.executable(),resolve(this.root,'vision','testbench_worker.py'),resolve(this.root,'models','grounding-dino-tiny','model.safetensors')];
    const checks=await Promise.all(paths.map(path=>access(path).then(()=>true,()=>false)));return {available:checks.every(Boolean),busy:!!this.pending,started:!!this.process,loaded:this.loaded,
      localOnly:true,maxActive:1,timeoutSeconds:this.timeoutMs/1000};}
  fail(error){const pending=this.pending;this.pending=null;if(pending){clearTimeout(pending.timer);pending.reject(error);}const child=this.process;this.process=null;this.buffer='';this.loaded=false;if(child)child.kill();}
  stop(){this.closed=true;this.fail(new ApiError('Test worker stopped.',503,'TESTBENCH_STOPPED'));}
  start(){if(this.process)return;
    const child=this.spawnWorker(this.executable(),['-m','vision.testbench_worker'],{cwd:this.root,stdio:['pipe','pipe','pipe'],windowsHide:true,
      env:{...process.env,PYTHONUNBUFFERED:'1',HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1'}});
    this.process=child;child.stdout.setEncoding('utf8');
    child.stdout.on('data',chunk=>{
      if(this.process!==child)return;this.buffer+=chunk;
      if(this.buffer.length>4*1024*1024)return this.fail(new ApiError('Test worker response exceeded its limit.',502,'TESTBENCH_PROTOCOL'));
      let newline;while((newline=this.buffer.indexOf('\n'))>=0){const line=this.buffer.slice(0,newline);this.buffer=this.buffer.slice(newline+1);if(!line.trim())continue;
        let message;try{message=JSON.parse(line);}catch{return this.fail(new ApiError('Test worker returned invalid JSON.',502,'TESTBENCH_PROTOCOL'));}
        if(!this.pending||message.id!==this.pending.id)return this.fail(new ApiError('Test worker response did not match its request.',502,'TESTBENCH_PROTOCOL'));
        if(message.status==='progress'){
          try{const progress=testbenchProgress(message.progress);this.pending.onProgress?.(progress);}catch{return this.fail(new ApiError('Test worker sent invalid progress.',502,'TESTBENCH_PROTOCOL'));}
          continue;
        }
        const pending=this.pending;this.pending=null;clearTimeout(pending.timer);
        if(message.status==='ok'&&message.result&&typeof message.result==='object'){this.loaded=true;pending.resolve(message.result);}
        else if(message.status==='error')pending.reject(new ApiError('The local test could not process this image. Check the image and local model runtime.',422,'TESTBENCH_INFERENCE_FAILED'));
        else{pending.reject(new ApiError('Test worker result protocol is invalid.',502,'TESTBENCH_PROTOCOL'));this.fail(new ApiError('Test worker protocol error.',502,'TESTBENCH_PROTOCOL'));}
      }
    });
    child.stderr.resume();
    child.on('error',()=>{if(this.process===child)this.fail(new ApiError('The local test runtime could not start.',503,'TESTBENCH_UNAVAILABLE'));});
    child.on('exit',()=>{if(this.process===child)this.fail(new ApiError('The local test runtime exited.',503,'TESTBENCH_UNAVAILABLE'));});
    child.stdin.on('error',()=>{if(this.process===child)this.fail(new ApiError('The local test runtime input failed.',503,'TESTBENCH_UNAVAILABLE'));});
  }
  run(image,settings,onProgress){
    if(this.closed)return Promise.reject(new ApiError('Test worker stopped.',503,'TESTBENCH_STOPPED'));
    if(this.pending)return Promise.reject(new ApiError('Another test is running.',409,'TESTBENCH_BUSY'));
    return new Promise((resolveRequest,reject)=>{
      const id=randomUUID();this.pending={id,resolve:resolveRequest,reject,onProgress,timer:setTimeout(()=>this.fail(new ApiError('The image test exceeded its time limit.',504,'TESTBENCH_TIMEOUT')),this.timeoutMs)};
      try{this.start();this.process.stdin.write(JSON.stringify({id,imageBase64:image.bytes.toString('base64'),mimeType:image.mimeType,settings})+'\n');}
      catch{this.fail(new ApiError('The local test runtime could not start.',503,'TESTBENCH_UNAVAILABLE'));}
    });
  }
}
