import {readFile,writeFile,mkdir,rename,open,unlink,realpath,lstat} from 'node:fs/promises';
import {resolve,dirname,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {AI_TERMS,aiEligibleFrames,validateAIContribution} from './ai-store.mjs';
const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const hash=value=>createHash('sha256').update(value).digest('hex');
const canonical=value=>JSON.stringify(value&&typeof value==='object'?Array.isArray(value)?value.map(v=>JSON.parse(canonical(v))):Object.fromEntries(Object.keys(value).sort().map(k=>[k,JSON.parse(canonical(value[k]))])):value);
const stamp=value=>typeof value==='string'&&Number.isFinite(Date.parse(value))&&new Date(value).toISOString()===value;
const fields=['agentPseudonym','id','requestId','createdAt','taskId','frameId','imageSha256','sourceId','sourceGroupId','source','split','evaluationOnly','imageWidth','imageHeight','model','modelIdentity','imageDeliveredAt','schema','reviewerKind','status','complete','boxes','confounders','observation','confidenceMeaning','labelAuthority','reviewStatus','humanVerified','trainingApproved','participationTerms'];
async function boundedJSON(file,max){const handle=await open(file,'r');try{if((await handle.stat()).size>max)throw new Error('JSON exceeds bounded import size.');const raw=await handle.readFile();if(raw.length>max)throw new Error('JSON exceeds bounded import size.');return JSON.parse(raw);}finally{await handle.close();}}

export async function importAIContributions(file,output=resolve(ROOT,'data/community-ai-inbox-v1'),{publicCollectionDir=resolve(ROOT,'data/community-publish-all-v1')}={}){
  const data=await boundedJSON(file,64*1024*1024);
  if(data.schema!=='mushroom-vision.ai-contribution-export.v1'||data.reviewerKind!=='ai'||data.trainingApproved!==false||data.humanVerified!==false||!stamp(data.exportedAt)||Date.parse(data.exportedAt)>Date.now()+300000||!Array.isArray(data.contributions)||data.contributions.length>50000)throw new Error('Invalid AI-only contribution export.');
  const directory=await realpath(publicCollectionDir),collection=await boundedJSON(resolve(directory,'manifest.json'),12*1024*1024);
  if(collection.schema!=='mycoscout.community-collection.v1'||!Array.isArray(collection.frames)||collection.frames.length>10000||!Array.isArray(collection.sources)||collection.sources.length>10000)throw new Error('Invalid published collection.');
  const allFrames=new Map(),sources=new Map();
  for(const source of collection.sources){if(!/^[A-Za-z0-9_-]{1,160}$/.test(source.id??'')||sources.has(source.id))throw new Error('Duplicate or invalid published source.');sources.set(source.id,source);}
  for(const frame of collection.frames){
    if(!/^[A-Za-z0-9_-]{1,160}$/.test(frame.id??'')||allFrames.has(frame.id)||!sources.has(frame.sourceId)||!/^[a-f0-9]{64}$/.test(frame.sha256??'')||!['jpg','png'].some(ext=>frame.path===`images/${frame.sha256}.${ext}`)||!Number.isInteger(frame.width)||!Number.isInteger(frame.height)||frame.width<1||frame.height<1||frame.width*frame.height>20000000)throw new Error('Invalid published image identity.');
    allFrames.set(frame.id,frame);
  }
  const frames=new Map(aiEligibleFrames(collection.frames,sources).map(frame=>[frame.id,frame])),verified=new Set();
  async function validate(row,{fromPrior=false}={}){
    const frame=frames.get(row?.frameId),source=sources.get(frame?.sourceId);
    if(!row||typeof row!=='object'||Array.isArray(row)||Object.keys(row).some(key=>!fields.includes(key))||!frame||
      row.schema!=='mushroom-vision.ai-contribution.v1'||row.reviewerKind!=='ai'||row.labelAuthority!=='ai-unverified'||row.reviewStatus!=='quarantined'||row.humanVerified!==false||row.trainingApproved!==false||
      row.modelIdentity!=='agent-reported'||row.confidenceMeaning!=='agent-reported-not-calibrated'||!/^[a-f0-9]{48}$/.test(row.id??'')||!/^[a-f0-9]{48}$/.test(row.taskId??'')||!/^[a-f0-9]{32}$/.test(row.requestId??'')||!/^[a-f0-9]{64}$/.test(row.agentPseudonym??'')||
      !stamp(row.createdAt)||!stamp(row.imageDeliveredAt)||Date.parse(row.imageDeliveredAt)>Date.parse(row.createdAt)||Date.parse(row.createdAt)>(fromPrior?Date.now():Date.parse(data.exportedAt))+300000||
      row.imageSha256!==frame.sha256||row.imageWidth!==frame.width||row.imageHeight!==frame.height||row.sourceId!==frame.sourceId||row.evaluationOnly!==frame.evaluationOnly||row.split!==(frame.evaluationOnly?'evaluation_only':'train_candidate')||
      row.sourceGroupId!=='public-group-'+hash(String(frame.sourceGroupId??frame.sourceId)).slice(0,24)||!row.source||Object.keys(row.source).some(key=>!['title','license','licenseUrl','sourceUrl','attribution','changes'].includes(key))||['license','licenseUrl','sourceUrl','attribution'].some(key=>row.source[key]!==source[key]))throw new Error('AI annotation authority, rights, split or image provenance is invalid.');
    if(!row.model||Object.keys(row.model).sort().join(',')!=='name,provider,version'||Object.values(row.model).some(v=>typeof v!=='string'||!v.trim()||v.length>100||/[\u0000-\u001f\u007f]/.test(v)))throw new Error('Invalid self-reported AI model identity.');
    if(row.observation?.verification!=='agent-self-attestation-with-server-image-delivery'||row.participationTerms?.acceptance!=='submission'||row.participationTerms?.acceptedAt!==row.createdAt||row.participationTerms?.version!==AI_TERMS.version||row.participationTerms?.noticeSha256!==AI_TERMS.noticeSha256)throw new Error('AI viewing or annotation permission provenance changed.');
    const {verification,...observation}=row.observation;
    const validated=validateAIContribution({taskId:row.taskId,imageSha256:row.imageSha256,reviewerKind:row.reviewerKind,status:row.status,complete:row.complete,boxes:row.boxes,
      confounders:row.confounders.map(({explicitNonFungal,...box})=>{if(explicitNonFungal!==true)throw new Error('Missing explicit confounder evidence.');return box;}),observation,participationTerms:row.participationTerms},
      {taskId:row.taskId,imageSha256:frame.sha256,width:frame.width,height:frame.height},row.createdAt);
    for(const key of Object.keys(validated))if(canonical(validated[key])!==canonical(row[key]))throw new Error('AI annotation contains unrecognized or changed evidence.');
    if(!verified.has(frame.id)){
      const declared=resolve(directory,frame.path);if((await lstat(declared)).isSymbolicLink())throw new Error('Symlinked published image is not accepted.');
      const path=await realpath(declared);if(!path.startsWith(directory+sep))throw new Error('Published image escapes collection.');
      const handle=await open(path,'r');try{if((await handle.stat()).size>8*1024*1024)throw new Error('Oversized published image.');const bytes=await handle.readFile();if(bytes.length>8*1024*1024||hash(bytes)!==frame.sha256)throw new Error('Published original image integrity mismatch.');}finally{await handle.close();}
      verified.add(frame.id);
    }
    return row;
  }
  await mkdir(output,{recursive:true});const target=resolve(output,'contributions.json'),lockPath=resolve(output,'import.lock'),temporary=target+'.tmp';let lock;
  try{lock=await open(lockPath,'wx');}catch(error){if(error.code==='EEXIST')throw new Error('Another AI import is in progress.');throw error;}
  try{
    let prior={contributions:[]};try{if((await lstat(target)).isSymbolicLink())throw new Error('AI inbox must not be a symlink.');prior=await boundedJSON(target,64*1024*1024);if(prior.schema!=='mushroom-vision.community-ai-inbox.v1'||prior.trainingApproved!==false||prior.humanVerified!==false)throw new Error('Existing AI inbox is invalid.');}catch(error){if(error.code!=='ENOENT')throw error;}
    if(!Array.isArray(prior.contributions)||prior.contributions.length>50000)throw new Error('Existing AI inbox is invalid or full.');
    const merged=new Map(),tasks=new Map(),incoming=new Set();
    for(const row of prior.contributions){await validate(row,{fromPrior:true});if(merged.has(row.id)||tasks.has(row.taskId))throw new Error('Existing AI inbox identities conflict.');merged.set(row.id,row);tasks.set(row.taskId,row.id);}
    let imported=0;
    for(const row of data.contributions){
      await validate(row);if(incoming.has(row.id))throw new Error('Duplicate AI contribution ID in export.');incoming.add(row.id);
      if(tasks.has(row.taskId)&&tasks.get(row.taskId)!==row.id)throw new Error('An AI task was reused for another immutable annotation.');
      const existing=merged.get(row.id);if(existing&&canonical(existing)!==canonical(row))throw new Error('Changed content for an immutable AI contribution.');
      if(!existing){if(merged.size>=50000)throw new Error('AI inbox capacity reached.');merged.set(row.id,row);tasks.set(row.taskId,row.id);imported++;}
    }
    const payload={schema:'mushroom-vision.community-ai-inbox.v1',importedAt:new Date().toISOString(),reviewerKind:'ai',reviewStatus:'quarantined',humanVerified:false,trainingApproved:false,automaticTraining:false,
      contributions:[...merged.values()]},raw=JSON.stringify(payload,null,2)+'\n';if(Buffer.byteLength(raw)>64*1024*1024)throw new Error('Combined AI inbox exceeds64MiB.');
    await writeFile(temporary,raw,{flag:'wx'});await rename(temporary,target);return {imported,total:merged.size,trainingApproved:false,humanVerified:false,automaticTraining:false};
  }finally{await lock.close();await unlink(lockPath);}
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)){
  if(!process.argv[2])throw new Error('Usage: node community/import-ai-contributions.mjs EXPORT_JSON [OUTPUT_DIRECTORY] [PUBLIC_COLLECTION_DIRECTORY]');
  console.log(JSON.stringify(await importAIContributions(resolve(process.argv[2]),process.argv[3]?resolve(process.argv[3]):undefined,process.argv[4]?{publicCollectionDir:resolve(process.argv[4])}:{})));
}
