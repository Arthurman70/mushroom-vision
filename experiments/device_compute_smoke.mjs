import http from 'node:http';
import {spawn} from 'node:child_process';
import {readFile,writeFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {DeviceComputeHub,handleDeviceCompute} from '../core/device-compute.mjs';

const root=resolve('.'),output=process.argv[2];if(!output)throw new Error('Supply a new JSON output path.');
const profiles=JSON.parse(await readFile('config/device-profiles.json','utf8')).profiles;
const hub=new DeviceComputeHub({profiles}),reports=[];
const sendJson=(response,status,value)=>{response.writeHead(status,{'Content-Type':'application/json'});response.end(JSON.stringify(value));};
const readBody=async(request,max)=>{let size=0;const parts=[];for await(const p of request){size+=p.length;if(size>max)throw new Error('Body too large');parts.push(p);}return JSON.parse(Buffer.concat(parts).toString('utf8'));};
const server=http.createServer(async(request,response)=>{try{const url=new URL(request.url,'http://localhost');if(url.pathname!=='/api/device-compute/worker')return sendJson(response,404,{error:'Worker test only'});if(!await handleDeviceCompute({hub,request,response,url,readBody,sendJson}))sendJson(response,404,{error:'Unknown test route'});}catch(error){sendJson(response,error.status??400,{error:error.message,code:error.code});}});
await new Promise((ok,fail)=>{server.once('error',fail);server.listen(0,'127.0.0.1',ok);});
try {
 const origin='http://127.0.0.1:'+server.address().port,image=await readFile('data/video-discovery-v1/frames/youtube-1Z9FgVpYris-003.jpg');
 for(const kind of ['image-quality-v1','fungi-candidates-v1']){
  const paired=hub.pair({name:'Integration test '+kind,profileId:'this-pc'}),queued=hub.enqueue({kind,sessionId:'device-worker-smoke-v1',frameId:'youtube-1Z9FgVpYris-003',imageBase64:image.toString('base64'),mimeType:'image/jpeg',targetDeviceId:paired.deviceId,ttlMs:120000});
  const executable=resolve(root,'runtime-vision',process.platform==='win32'?'Scripts/python.exe':'bin/python'),args=['-m','adapters.device_worker','--station',origin,'--once'];if(kind==='fungi-candidates-v1')args.push('--detector-model','models/grounding-dino-tiny');
  const start=performance.now();let lines='';
  await new Promise((ok,fail)=>{const child=spawn(executable,args,{cwd:root,windowsHide:true,stdio:['ignore','pipe','pipe'],env:{...process.env,MYCOSCOUT_DEVICE_TOKEN:paired.deviceToken,HF_HUB_OFFLINE:'1',TRANSFORMERS_OFFLINE:'1'}}),timer=setTimeout(()=>{child.kill();fail(new Error('Worker integration timed out'));},115000);child.stdout.on('data',chunk=>{lines+=chunk.toString();if(lines.length>100000){child.kill();fail(new Error('Worker output exceeded test bound'));}});child.stderr.resume();child.once('error',error=>{clearTimeout(timer);fail(error);});child.once('exit',code=>{clearTimeout(timer);code===0?ok():fail(new Error('Worker exited '+code));});});
  const job=hub.status().jobs.find(j=>j.id===queued.id);if(job?.status!=='completed'||!job.result?.unverified||job.result.flightAuthority!==false)throw new Error('Worker did not produce a bounded unverified result');
  reports.push({kind,status:job.status,elapsedIncludingStartupMs:performance.now()-start,result:job.result,sourceFrame:job.frameId,sha256:job.imageSha256,workerMessages:lines.trim().split('\n').filter(Boolean).map(line=>JSON.parse(line))});hub.revoke({deviceId:paired.deviceId});console.log(JSON.stringify({kind,status:job.status,processingMs:job.result.processingMs}));
 }
 await writeFile(output,JSON.stringify({schema:'mycoscout.device-worker-smoke.v1',createdAt:new Date().toISOString(),transport:'loopback HTTP, paired device token',physicalRemoteDevice:false,reports},null,2)+'\n',{flag:'wx'});
} finally {server.closeAllConnections();await new Promise(resolve=>server.close(resolve));}
