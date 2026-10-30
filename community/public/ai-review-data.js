import { validConfounder } from './review-evidence.js';
const text=(value,limit=600)=>typeof value==='string'?value.slice(0,limit):'';
const list=value=>Array.isArray(value)?value:[];
const bbox=(b,f)=>Array.isArray(b)&&b.length===4&&b.every(Number.isFinite)&&b[0]>=0&&b[1]>=0&&b[2]>b[0]&&b[3]>b[1]&&b[2]<=f.width&&b[3]<=f.height;
export function publicAudit(value,frame){
  if(!value||value.reviewerKind!=='ai'||value.humanVerified!==false||!['visible','absent','uncertain'].includes(value.status)||!['original-image','contact-sheet','crop'].includes(value.observation?.mode)||!Number.isSafeInteger(value.revision)||value.revision<1)return null;
  const mode=value.observation.mode;
  const reviewId=text(value.reviewId??value.imageId??frame.id,180);if(!/^[-A-Za-z0-9_:]+$/.test(reviewId))return null;
  const cleanBox=box=>({bbox:[...(box.bboxPixelsXYXY??box.bbox)],groupId:text(box.groupId,80)||'unknown',confidence:Number.isFinite(box.confidence)&&box.confidence>=0&&box.confidence<=1?box.confidence:null,evidence:text(box.evidence),humanVerified:false,trainingAllowed:false});
  return {reviewId,revision:value.revision,status:value.status,reviewerKind:'ai',humanVerified:false,trainingAllowed:false,
    complete:mode==='original-image'&&value.complete===true,evidence:text(value.evidence,1200),
    observation:{mode,displayedWidth:Number.isInteger(value.observation.displayedWidth)?value.observation.displayedWidth:null,displayedHeight:Number.isInteger(value.observation.displayedHeight)?value.observation.displayedHeight:null},
    boxes:list(value.boxes).slice(0,64).filter(b=>bbox(b?.bboxPixelsXYXY??b?.bbox,frame)).map(cleanBox),
    confounders:list(value.confounders).slice(0,64).filter(b=>validConfounder(b?.kind)&&b.explicitNonFungal===true&&bbox(b.bboxPixelsXYXY??b.bbox,frame)).map(b=>({...cleanBox(b),kind:b.kind,explicitNonFungal:true})),
    confounderTags:[...new Set(list(value.confounderTags).filter(validConfounder))]};
}
