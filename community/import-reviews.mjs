import { readFile,writeFile,mkdir,rename,open,unlink,realpath } from 'node:fs/promises';
import { resolve,dirname,sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { VideoCollection } from '../core/video-collection.mjs';
import { validateReview } from './store.mjs';
const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'..');
export async function importCommunity(file,output=resolve(ROOT,'data/community-inbox-v1'),{collectionDir=resolve(ROOT,'data/video-discovery-v1'),publicCollectionDir=resolve(ROOT,'data/community-publish-all-v1')}={}) {
  const raw=await readFile(file);if(raw.length>32*1024*1024)throw new Error('Community export exceeds32MiB.');
  const data=JSON.parse(raw);if(data.schema!=='mycoscout.community-reviews.v1'||!Array.isArray(data.reviews)||data.reviews.length>50000)throw new Error('Invalid community export.');
  const collection=await new VideoCollection(collectionDir).read();
  const frames=new Map(collection.frames.map(f=>[f.id,f]));
  // The private bridge accepts new public photo IDs only from our explicit,
  // hash-verified publication package, never from an incoming review export.
  try{
    const publicManifest=JSON.parse(await readFile(resolve(publicCollectionDir,'manifest.json'),'utf8'));
    if(publicManifest.schema!=='mycoscout.community-collection.v1'||!Array.isArray(publicManifest.frames)||publicManifest.frames.length>10000)throw new Error('Invalid public review collection.');
    for(const frame of publicManifest.frames){
      if(frames.has(frame.id)){if(frames.get(frame.id).sha256!==frame.sha256)throw new Error('Public image conflicts with source collection.');continue;}
      if(!/^[A-Za-z0-9_-]{1,160}$/.test(frame.id)||!/^images\/[a-f0-9]{64}\.jpg$/.test(frame.path)||frame.path!==`images/${frame.sha256}.jpg`||!Number.isInteger(frame.width)||!Number.isInteger(frame.height)||frame.width<1||frame.height<1||frame.width*frame.height>20000000)throw new Error('Invalid published review image.');
      frames.set(frame.id,{...frame,publicationDirectory:publicCollectionDir});
    }
  }catch(error){if(error.code!=='ENOENT')throw error;}
  await mkdir(output,{recursive:true});const lockPath=resolve(output,'import.lock');let lock;
  try{lock=await open(lockPath,'wx');}catch(error){if(error.code==='EEXIST')throw new Error('Another community import is in progress. Retry when it finishes.');throw error;}
  try{
  const target=resolve(output,'reviews.json');let prior={reviews:[]};
  try{prior=JSON.parse(await readFile(target,'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
  if(!Array.isArray(prior.reviews)||prior.reviews.length>50000)throw new Error('Existing community inbox is invalid; it has not been replaced.');
  const history=prior.history??prior.reviews;
  if(!Array.isArray(history)||history.length>200000)throw new Error('Existing community import history is invalid or full; archive it before importing more.');
  const merged=new Map(prior.reviews.map(r=>[r.reviewerId+':'+r.frameId,r])),ids=new Map(),versions=new Map();let imported=0;
  for(const row of history){
    const identity=row.reviewerId+':'+row.frameId+':'+row.revision;
    if(!Number.isSafeInteger(row.id)||row.id<1||ids.has(row.id)||versions.has(identity))throw new Error('Existing community review identities conflict; the inbox has not been replaced.');
    ids.set(row.id,identity);versions.set(identity,row);
  }
  for(const row of data.reviews){
    const frame=frames.get(row.frameId);
    if(!frame||row.sha256!==frame.sha256||!Number.isSafeInteger(row.id)||row.id<1||!Number.isSafeInteger(row.revision)||row.revision<1||!/^[a-f0-9]{64}$/.test(row.reviewerId)||!Number.isFinite(Date.parse(row.createdAt)))throw new Error('Community review provenance is invalid.');
    if(frame.publicationDirectory&&!frame.bytesVerified){const directory=await realpath(frame.publicationDirectory),path=await realpath(resolve(directory,frame.path));if(!path.startsWith(directory+sep))throw new Error('Published review image escapes its collection.');const bytes=await readFile(path);if(bytes.length>8*1024*1024||createHash('sha256').update(bytes).digest('hex')!==frame.sha256)throw new Error('Published review image failed integrity validation.');frame.bytesVerified=true;}
    const payload=validateReview({...row,baseRevision:row.revision-1},frame,{historical:true}),key=row.reviewerId+':'+row.frameId;
    const identity=key+':'+row.revision;
    if(ids.has(row.id)&&ids.get(row.id)!==identity)throw new Error('An immutable community review ID was reused for different evidence.');
    const candidate={...payload,id:row.id,revision:row.revision,reviewerId:row.reviewerId,frameId:row.frameId,sha256:row.sha256,createdAt:row.createdAt,
      sourceId:frame.sourceId,sourceGroupId:frame.sourceGroupId,evaluationOnly:frame.evaluationOnly},existing=merged.get(key);
    const recorded=versions.get(identity);
    if(recorded&&JSON.stringify(recorded)!==JSON.stringify(candidate))throw new Error('Conflicting content for the same community review revision.');
    if(existing?.revision===row.revision&&JSON.stringify(existing)!==JSON.stringify(candidate))throw new Error('Conflicting content for the same community review revision.');
    if(!recorded){if(versions.size>=200000)throw new Error('Community import history is full; archive it before importing more.');ids.set(row.id,identity);versions.set(identity,candidate);}
    if((existing?.revision??0)>=row.revision)continue;
    merged.set(key,candidate);imported++;
  }
  await writeFile(target+'.tmp',JSON.stringify({schema:'mycoscout.community-inbox.v1',importedAt:new Date().toISOString(),trainingApproved:false,reviews:[...merged.values()],history:[...versions.values()]},null,2)+'\n');await rename(target+'.tmp',target);
  return {imported,total:merged.size,trainingApproved:false};
  }finally{await lock.close();await unlink(lockPath);}
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  if(!process.argv[2])throw new Error('Usage: node community/import-reviews.mjs EXPORTED_JSON');
  console.log(JSON.stringify(await importCommunity(resolve(process.argv[2]))));
}
