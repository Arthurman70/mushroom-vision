import { readFile,realpath,stat } from 'node:fs/promises';
import { resolve,sep } from 'node:path';
import { createHash } from 'node:crypto';
import { ApiError } from './validation.mjs';
import { validateImage } from './finds-images.mjs';

const HASH=/^[a-f0-9]{64}$/,ID=/^[A-Za-z0-9_-]{1,160}$/,GROUP=/^[A-Za-z0-9:_-]{1,160}$/;
const fail=()=>{throw new ApiError('The video collection manifest is invalid.',409,'VIDEO_COLLECTION_INVALID');};
const text=(v,max=500)=>typeof v==='string'&&v.length<=max?v:fail();
function url(value){if(!value)return '';try{const u=new URL(text(value,2000));if(u.protocol!=='https:'||u.username||u.password)fail();return u.href;}catch{fail();}}

// A collection is installed locally by the video extraction adapter. No request
// can nominate an arbitrary source file or fetch a remote video through this API.
export class VideoCollection {
  constructor(directory){this.directory=resolve(directory);}
  async read(){
    const file=resolve(this.directory,'manifest.json');let info;
    try{info=await stat(file);}catch(error){if(error.code==='ENOENT')return {available:false,sources:[],frames:[]};throw error;}
    if(info.size>8*1024*1024)fail();
    const root=await realpath(this.directory),actual=await realpath(file);if(!actual.startsWith(root+sep))fail();
    let data;try{data=JSON.parse(await readFile(actual,'utf8'));}catch{fail();}
    if(!data||data.schema!=='mycoscout.video-collection.v1'||!Array.isArray(data.sources)||data.sources.length>200||!Array.isArray(data.frames)||data.frames.length>2000)fail();
    const sources=new Map();
    for(const source of data.sources){
      if(!source||!ID.test(source.id)||sources.has(source.id)||!HASH.test(source.videoHash??source.sha256))fail();
      const sourceGroupId=source.sourceGroupId??'video:'+(source.videoHash??source.sha256);if(!GROUP.test(sourceGroupId))fail();
      sources.set(source.id,{id:source.id,title:text(source.title),sourceUrl:url(source.sourceUrl),attribution:text(source.attribution,2000),
        license:text(source.license,160),licenseUrl:url(source.licenseUrl),videoHash:source.videoHash??source.sha256,sourceGroupId,
        licenseAllowsTraining:source.licenseAllowsTraining===true,evaluationOnly:source.evaluationOnly!==false||source.licenseAllowsTraining!==true});
    }
    const ids=new Set(),frames=data.frames.map(frame=>{
      if(!frame||typeof frame!=='object')fail();const source=sources.get(frame.sourceId);
      if(!source||!ID.test(frame.id)||ids.has(frame.id)||!HASH.test(frame.sha256)||!/^frames\/[A-Za-z0-9._-]+\.(jpg|png)$/.test(frame.path)||
        !Number.isInteger(frame.width)||!Number.isInteger(frame.height)||frame.width<1||frame.height<1||frame.width*frame.height>20000000||
        !Number.isFinite(frame.timestampSeconds)||frame.timestampSeconds<0||frame.timestampSeconds>86400||!Number.isFinite(frame.blurScore)||frame.blurScore<0||
        (frame.partitionGroup!=null&&frame.partitionGroup!==source.sourceGroupId)||(frame.sourceGroupId!=null&&frame.sourceGroupId!==source.sourceGroupId))fail();
      ids.add(frame.id);return {id:frame.id,sourceId:source.id,sourceGroupId:source.sourceGroupId,partitionGroup:source.sourceGroupId,path:frame.path,sha256:frame.sha256,width:frame.width,height:frame.height,
        timestampSeconds:frame.timestampSeconds,blurScore:frame.blurScore,blurMetric:text(frame.blurMetric,160),
        blurStratum:['low','middle','high'].includes(frame.blurStratum)?frame.blurStratum:null,
        evaluationOnly:source.evaluationOnly||frame.evaluationOnly===true,actualDistanceM:null,source,
        imageUrl:'/api/testbench/videos/frames/'+frame.id+'/image'};
    });
    return {available:true,sources:[...sources.values()],frames};
  }
  async list({offset=0,limit=12,sourceId='',sort='time'}={}){
    offset=Number(offset);limit=Number(limit);if(!Number.isInteger(offset)||offset<0||offset>2000||!Number.isInteger(limit)||limit<1||limit>24||!['time','blur'].includes(sort))throw new ApiError('Use a valid video page and sort order.');
    const collection=await this.read();if(sourceId&&!collection.sources.some(source=>source.id===sourceId))throw new ApiError('Video source not found.',404,'NOT_FOUND');
    const frames=collection.frames.filter(frame=>!sourceId||frame.sourceId===sourceId).sort(sort==='blur'?(a,b)=>a.blurScore-b.blurScore||a.id.localeCompare(b.id):(a,b)=>a.sourceId.localeCompare(b.sourceId)||a.timestampSeconds-b.timestampSeconds);
    return {schema:'mycoscout.video-gallery.v1',available:collection.available,sources:collection.sources,total:frames.length,offset,limit,
      frames:frames.slice(offset,offset+limit).map(({path,...frame})=>frame),automaticTraining:false,labelsVerified:false,
      notice:'Blur is a whole-frame sharpness measure. Mushroom identity, visibility and distance need human review. Frames from one source video share a dataset group.'};
  }
  async frame(id){
    if(!ID.test(id))throw new ApiError('Video frame not found.',404,'NOT_FOUND');
    const collection=await this.read(),frame=collection.frames.find(frame=>frame.id===id);if(!frame)throw new ApiError('Video frame not found.',404,'NOT_FOUND');
    const directory=await realpath(this.directory),path=await realpath(resolve(directory,frame.path));if(!path.startsWith(directory+sep))fail();
    const info=await stat(path);if(info.size>8*1024*1024)fail();const bytes=await readFile(path);
    const hash=createHash('sha256').update(bytes).digest('hex');if(hash!==frame.sha256)throw new ApiError('Video frame integrity check failed.',409,'VIDEO_FRAME_INTEGRITY');
    const image=validateImage({imageBase64:bytes.toString('base64')});if(image.width!==frame.width||image.height!==frame.height)fail();
    const {path:privatePath,source,...publicFrame}=frame;
    return {image,metadata:{...publicFrame,source,sceneId:source.sourceGroupId}};
  }
}
