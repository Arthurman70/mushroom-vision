import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createHash } from 'node:crypto';
import { ApiError } from './validation.mjs';
async function optional(path,fallback){try{return JSON.parse(await readFile(path,'utf8'));}catch(error){if(error.code==='ENOENT')return fallback;throw error;}}
export class ReviewAssistance {
  constructor(root,videos){this.root=root;this.videos=videos;this.aiDirectory=resolve(root,'data/ai-video-review-v1');}
  async status(){
    const ai=await optional(resolve(this.aiDirectory,'status.json'),{status:'not-started',processedFrames:0,totalFrames:0});
    const inbox=await optional(resolve(this.root,'data/community-inbox-v1/reviews.json'),{reviews:[]});
    return {ai,communityReviews:inbox.reviews.length,communityFrames:new Set(inbox.reviews.map(r=>r.frameId)).size,
      communityUrl:'https://thegreatlakesgleaner.press/mushroom-lab/',automaticTraining:false};
  }
  async frame(id){
    const {metadata}=await this.videos.frame(id);
    const ai=await optional(resolve(this.aiDirectory,'frames','main--'+id+'.json'),null);
    if(ai&&(ai.imageSha256!==metadata.sha256||ai.sourceId!==metadata.sourceId)) throw new ApiError('AI suggestions do not match this original frame.',409,'AI_SOURCE_CHANGED');
    const inbox=await optional(resolve(this.root,'data/community-inbox-v1/reviews.json'),{reviews:[]});
    const validBox=box=>Array.isArray(box?.bbox)&&box.bbox.length===4&&box.bbox.every(Number.isFinite)&&box.bbox[0]>=0&&box.bbox[1]>=0&&box.bbox[2]>box.bbox[0]&&box.bbox[3]>box.bbox[1]&&box.bbox[2]<=metadata.width&&box.bbox[3]<=metadata.height;
    const suggestions=ai?{...ai,humanVerified:false,trainingAllowed:false,boxes:(Array.isArray(ai.boxes)?ai.boxes:[]).slice(0,128).filter(box=>validBox(box)&&Number.isFinite(box.score)&&box.score>=0&&box.score<=1).map(box=>({id:String(box.id??'').slice(0,160),bbox:[...box.bbox],score:box.score,label:String(box.label??'mushroom candidate').slice(0,160)}))}:null;
    const assistantAudit=await this.audit(id,metadata,suggestions,validBox);
    return {frameId:id,ai:suggestions,assistantAudit,community:inbox.reviews.filter(r=>r.frameId===id&&r.sha256===metadata.sha256&&r.sourceId===metadata.sourceId),
      notice:'Suggestions are not confirmed labels. Inspect the original frame, correct every box, then save your own review.'};
  }
  async audit(id,metadata,ai,validBox){
    let raw;
    try{raw=await readFile(resolve(this.aiDirectory,'assistant-audit.json'));}catch(error){if(error.code==='ENOENT')return null;throw error;}
    const unavailable=reason=>({availability:'unavailable',humanVerified:false,trainingAllowed:false,boxes:[],error:reason});
    try{
      if(raw.length>4*1024*1024)throw new Error('Audit file exceeds its size limit.');
      const data=JSON.parse(raw);
      if(data.schema!=='mycoscout.assistant-audit.v1'||!Array.isArray(data.reviews)||data.reviews.length>10000)throw new Error('Audit format is invalid.');
      const matches=data.reviews.filter(row=>row?.collection==='main'&&row.frameId===id);
      if(!matches.length)return null;
      if(matches.length!==1)throw new Error('More than one audit exists for this frame.');
      const row=matches[0];
      if(row.imageSha256!==metadata.sha256||(row.sourceId!==undefined&&row.sourceId!==metadata.sourceId))throw new Error('Audit pixels or source do not match this original frame.');
      if(row.humanVerified!==false||!['visible','absent','uncertain'].includes(row.status)||typeof row.complete!=='boolean')throw new Error('Audit authority or visibility is invalid.');
      const accepted=row.acceptedBoxIds??[],rejected=row.rejectedBoxIds??[],corrected=row.correctedBoxes??[];
      const validIds=ids=>Array.isArray(ids)&&ids.length<=128&&ids.every(value=>typeof value==='string'&&value.length>0&&value.length<=160)&&new Set(ids).size===ids.length;
      if(!validIds(accepted)||!validIds(rejected)||accepted.some(value=>rejected.includes(value))||!Array.isArray(corrected)||corrected.length+accepted.length>128||corrected.some(box=>!validBox(box)))throw new Error('Audit boxes or proposal references are invalid.');
      if(row.status!=='visible'&&(accepted.length||corrected.length))throw new Error('Audit visibility conflicts with its boxes.');
      const boxes=[];
      for(const proposalId of accepted){
        const proposals=(ai?.boxes??[]).filter(box=>box.id===proposalId);
        if(proposals.length!==1)throw new Error('An accepted proposal is missing or ambiguous in the matching AI result.');
        boxes.push({...proposals[0],id:'accepted-'+proposalId,origin:'accepted-ai',group:'unknown'});
      }
      corrected.forEach((box,index)=>boxes.push({id:'corrected-'+(index+1),bbox:[...box.bbox],label:'Assistant-corrected mushroom candidate',group:'unknown',origin:'assistant-corrected'}));
      return {availability:'available',frameId:id,sourceId:metadata.sourceId,imageSha256:metadata.sha256,
        auditSha256:createHash('sha256').update(raw).digest('hex'),reviewer:'assistant-visual-audit',
        status:row.status,complete:row.complete,notes:String(row.notes??'').slice(0,4000),
        acceptedBoxIds:[...accepted],rejectedBoxIds:[...rejected],boxes,humanVerified:false,trainingAllowed:false};
    }catch(error){return unavailable(error.message);}
  }
}
