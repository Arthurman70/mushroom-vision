/** Shared scan choices: manually selected image areas, never semantic ground segmentation. */
export const TRACKING_PROFILE_IDS=Object.freeze(['ground-tiles','full-tiles','fast']);
export function normalizedTileRegion(value){
  if(!Array.isArray(value)||value.length!==4||!value.every(n=>typeof n==='number'&&Number.isFinite(n)&&n>=0&&n<=1)||value[0]>=value[2]||value[1]>=value[3])throw new Error('Choose a nonempty normalized tile area inside the image.');
  return [...value];
}
export function trackingProfile(input={}){
  const id=input.profileId===undefined?'ground-tiles':input.profileId;
  if(!TRACKING_PROFILE_IDS.includes(id))throw new Error('Choose ground-tiles, full-tiles or fast.');
  if(id==='fast'){
    if(input.tileRegion!==undefined)throw new Error('The fast profile uses the whole frame and accepts no tile area.');
    return {id,label:'Fast full frame',methods:['original'],tileGrid:2,scale:1,expectedPasses:1,maxWidth:960,maxHeight:720,fullFrameContext:true,semanticGroundSegmentation:false};
  }
  const tileRegion=normalizedTileRegion(input.tileRegion===undefined?(id==='ground-tiles'?[0,.35,1,1]:[0,0,1,1]):input.tileRegion);
  if(id==='full-tiles'&&tileRegion.some((value,index)=>value!==[0,0,1,1][index]))throw new Error('The full-tiles profile requires the whole image area.');
  return {id,label:id==='ground-tiles'?'Lower ground area':'Whole frame tiles',methods:['tiles'],tileGrid:2,scale:1,tileRegion,expectedPasses:5,maxWidth:1920,maxHeight:1080,fullFrameContext:true,semanticGroundSegmentation:false};
}
export function trackingCaptureSize(width,height,profile){
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1)throw new Error('The source has no image dimensions.');
  const scale=Math.min(1,profile.maxWidth/width,profile.maxHeight/height);
  return {width:Math.max(1,Math.floor(width*scale)),height:Math.max(1,Math.floor(height*scale))};
}
export function trackingTilePreview(width,height,profile){
  if(!profile.tileRegion)return [];
  const [x1,y1,x2,y2]=profile.tileRegion,left=Math.floor(x1*width),top=Math.floor(y1*height),right=Math.ceil(x2*width),bottom=Math.ceil(y2*height);
  const intervals=length=>{const tile=Math.min(length,Math.ceil(length/1.8));return [...new Set([0,length-tile])].map(start=>[start,start+tile]);};
  return intervals(bottom-top).flatMap(([a,b])=>intervals(right-left).map(([c,d])=>[left+c,top+a,left+d,top+b]));
}
export function sampleCanResume({resume,outcome,capturedEpoch,currentEpoch,capturedGeneration,currentGeneration}){
  return resume===true&&outcome?.accepted===true&&capturedEpoch===currentEpoch&&capturedGeneration===currentGeneration;
}
/** Reject mismatched or legacy responses before they can create fresh tracks. */
export function validateTrackingSample(result,{profile,width,height}){
  const fail=()=>{throw new Error('The station did not confirm this captured frame, selected scan and its measured pass evidence.');};
  const finite=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0;
  const box=b=>Array.isArray(b)&&b.length===4&&b.every(finite)&&b[0]<b[2]&&b[1]<b[3]&&b[2]<=width&&b[3]<=height;
  if(result?.image?.width!==width||result.image.height!==height||result.profile?.id!==profile.id||
    JSON.stringify(result.settings?.tileRegion??null)!==JSON.stringify(profile.tileRegion??null)||
    JSON.stringify(result.settings?.methods)!==JSON.stringify(profile.methods)||result.settings.scale!==1||result.settings.boxThreshold!==.3||
    !Array.isArray(result.detections)||result.detections.length>100||result.detections.some(item=>!box(item?.bbox)||!finite(item.confidence)||item.confidence>1)||
    !finite(result.processingSeconds)||!finite(result.requestSeconds))fail();
  const scan=result.scan;
  if(!scan||!Number.isInteger(scan.modelPasses)||scan.modelPasses<1||scan.modelPasses>5||scan.passes!==scan.modelPasses||
    scan.fullFrameContext!==true||scan.method!==profile.methods[0]||!Array.isArray(scan.tileBoundsPixelsXYXY)||
    JSON.stringify(scan.tileRegion)!==JSON.stringify(profile.tileRegion??null)||!Array.isArray(scan.passTimings)||scan.passTimings.length!==scan.modelPasses||
    scan.tileBoundsPixelsXYXY.length!==scan.modelPasses-1)fail();
  for(const [index,pass]of scan.passTimings.entries()){
    const expected=index===0?[0,0,width,height]:scan.tileBoundsPixelsXYXY[index-1];
    if(!box(expected)||!pass||pass.kind!==(index===0?'context':'tile')||JSON.stringify(pass.bboxPixelsXYXY)!==JSON.stringify(expected)||!finite(pass.seconds))fail();
  }
  return result;
}
