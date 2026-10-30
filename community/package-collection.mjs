import { mkdir, readFile, writeFile, rename } from 'node:fs/promises';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { VideoCollection } from '../core/video-collection.mjs';
const ROOT=resolve(dirname(fileURLToPath(import.meta.url)),'..');
export const publicSourceAllowed=source=>source?.licenseAllowsTraining===true&&/^CC BY(?:-SA)?(?:\s+\d+(?:\.\d+)?)?(?: \(YouTube\))?$/.test(source.license);
export async function packageCollection(output=resolve(ROOT,'data/community-publish-v1'), aiDirectory=resolve(ROOT,'data/ai-video-review-v1')) {
  // Only the public licensed video collection is eligible for publication.
  // Facebook/private sources and the NC research collection are never traversed.
  const collection=new VideoCollection(resolve(ROOT,'data/video-discovery-v1'));
  const data=await collection.read();
  const allowed=new Set(data.sources.filter(publicSourceAllowed).map(s=>s.id));
  await mkdir(resolve(output,'images'),{recursive:true});
  const frames=[];
  for(const frame of data.frames.filter(f=>allowed.has(f.sourceId))) {
    const {image}=await collection.frame(frame.id), hash=createHash('sha256').update(image.bytes).digest('hex');
    const path='images/'+hash+extname(frame.path);
    // Publish the bytes just verified by VideoCollection, not a second read of
    // a source file that might have changed while another import was running.
    await writeFile(resolve(output,path),image.bytes);
    frames.push({id:frame.id,sourceId:frame.sourceId,timestampSeconds:frame.timestampSeconds,width:frame.width,height:frame.height,
      sha256:hash,path,sourceGroupId:frame.sourceGroupId,videoHash:frame.source.videoHash,evaluationOnly:frame.evaluationOnly,
      sourceLicenseAllowsTraining:true,actualDistanceM:null});
  }
  const sources=data.sources.filter(s=>allowed.has(s.id)).map(s=>({...s,changes:'Frames extracted from source video and converted to JPEG; original resolution retained.'}));
  const manifest={schema:'mycoscout.community-collection.v1',createdAt:new Date().toISOString(),sources,frames,
    privacy:'Only licensed public footage. No private photos, precise location, local reviewer names or flight controls.',
    training:'Volunteer and AI labels remain unverified until curator acceptance; evaluation sources cannot train.'};
  const manifestPath=resolve(output,'manifest.json');await writeFile(manifestPath+'.tmp',JSON.stringify(manifest,null,2)+'\n');await rename(manifestPath+'.tmp',manifestPath);
  const suggestions=[];
  for(const frame of frames) {
    let row;try{row=JSON.parse(await readFile(resolve(aiDirectory,'frames','main--'+frame.id+'.json'),'utf8'));}catch(error){if(error.code==='ENOENT')continue;throw error;}
    if(row.imageSha256!==frame.sha256||row.sourceId!==frame.sourceId)throw new Error('AI suggestion source hash mismatch: '+frame.id);
    const boxes=(row.boxes??[]).slice(0,64).filter(b=>Array.isArray(b.bbox)&&b.bbox.length===4&&b.bbox.every(Number.isFinite)&&
      b.bbox[0]>=0&&b.bbox[1]>=0&&b.bbox[2]<=frame.width&&b.bbox[3]<=frame.height&&b.bbox[2]>b.bbox[0]&&b.bbox[3]>b.bbox[1]&&Number.isFinite(b.score)&&b.score>=0&&b.score<=1)
      .map(b=>({bbox:b.bbox,score:b.score,label:String(b.label??'mushroom candidate').slice(0,120),humanVerified:false,trainingAllowed:false}));
    suggestions.push({frameId:frame.id,imageSha256:frame.sha256,sourceId:frame.sourceId,status:row.status,model:row.model,boxes});
  }
  let status={};try{status=JSON.parse(await readFile(resolve(aiDirectory,'status.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
  const ai={schema:'mycoscout.public-ai-suggestions.v1',run:{status:suggestions.length===frames.length?'completed':'in-progress',
    reviewedFrames:suggestions.length,totalFrames:frames.length,updatedAt:new Date().toISOString(),humanVerified:false},frames:suggestions};
  const aiPath=resolve(output,'ai-suggestions.json');await writeFile(aiPath+'.tmp',JSON.stringify(ai)+'\n');await rename(aiPath+'.tmp',aiPath);
  return {output,frames:frames.length,sources:sources.length,aiFrames:suggestions.length};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url))console.log(JSON.stringify(await packageCollection(process.argv[2]),null,2));
