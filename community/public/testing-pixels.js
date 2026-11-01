export const PIXEL_LEVELS = [
  {id:'native',label:'Native original',pixels:null}, {id:'64px',label:'64 px',pixels:64},
  {id:'32px',label:'32 px',pixels:32}, {id:'16px',label:'16 px',pixels:16},
  {id:'6p9px',label:'6.9 px',pixels:6.9,distance:10}, {id:'3p45px',label:'3.45 px',pixels:3.45,distance:20},
  {id:'1p38px',label:'1.38 px',pixels:1.38,distance:50}, {id:'0p69px',label:'0.69 px',pixels:.69,distance:100},
];
const HASH=/^[a-f0-9]{64}$/,ID=/^[A-Za-z0-9_-]{1,160}$/,COHORTS=['human-video','ai-photo'];
const text=(value,max=2000)=>typeof value==='string'?value.slice(0,max):'';
const count=value=>Number.isSafeInteger(value)&&value>=0&&value<=100000?value:null;
const finite=value=>Number.isFinite(value)&&value>=0?value:null;
const probability=value=>finite(value)!==null&&value<=1?value:null;
const seconds=value=>finite(value)!==null&&value<=86400?value:null;
const webUrl=value=>{try{const url=new URL(value);return ['http:','https:'].includes(url.protocol)&&!url.username&&!url.password?url.href:null;}catch{return null;}};
const dimensions=(width,height)=>Number.isInteger(width)&&Number.isInteger(height)&&width>0&&height>0&&width*height<=20000000;
const validBox=(box,width,height)=>Array.isArray(box)&&box.length===4&&box.every(Number.isFinite)&&box[0]>=0&&box[1]>=0&&box[2]>box[0]&&box[3]>box[1]&&box[2]<=width&&box[3]<=height;
const levelFor=id=>PIXEL_LEVELS.find(level=>level.id===id);

function numericGeometry(value,depth=0){
  if(depth>5)return null;
  if(Number.isFinite(value))return value;
  if(typeof value==='boolean')return value;
  if(['context','tile-1','tile-2','tile-3','tile-4'].includes(value))return value;
  if(Array.isArray(value))return value.slice(0,20).map(v=>numericGeometry(v,depth+1)).filter(v=>v!==null);
  if(value&&typeof value==='object'){
    const keys=['sourceSize','outputSize','canvasSize','sourceWidth','sourceHeight','outputWidth','outputHeight','sceneScaleXY','sceneBounds','sceneBoundsPixelsXYXY','supersampling','requestedTargetWidthPixels','realizedTargetWidthPixels','requestedTargetHeightPixels','realizedTargetHeightPixels','requestedPlacement','realizedPlacement','requestedFocalPlacement','realizedFocalPlacement','requestedFocalPositionNormalized','realizedFocalPositionNormalized','focalPositionClamped','patchSupersampledSize','sceneReEnlarged','tileCoverageCount','tileCoverage','tiles','processorInputDimensions','processorAndTileCoverage','passes','kind','boundsPixelsXYXY','processorScaleXY','selectedTargetCoverageFraction','selectedTargetInputWidthPixels','selectedTargetInputHeightPixels','intersectingTileCount','fullyCoveringTileCount','inputWidth','inputHeight','width','height','x','y','x1','y1','x2','y2','bbox','index','coverage','targetCoverage','targetWidthPixels','targetHeightPixels'];
    return Object.fromEntries(keys.filter(key=>key in value).map(key=>[key,numericGeometry(value[key],depth+1)]).filter(([,v])=>v!==null));
  }
  return null;
}

/** A bounded public projection: no private paths, arbitrary asset URLs, or merged authorities. */
export function publicPixels(input){
  if(input?.schema!=='mushroom-vision.public-pixels.v1'||!Array.isArray(input.cohorts)||input.cohorts.length!==2||!Array.isArray(input.frames)||input.frames.length<1||input.frames.length>64||!Array.isArray(input.records)||input.records.length<1||input.records.length>512||input.protocol?.weightsChanged!==false)throw new Error('Unsupported public pixel-test record.');
  const cohorts=input.cohorts.map(row=>{
    if(!COHORTS.includes(row?.id)||!Array.isArray(row.conditions)||row.conditions.length>16)throw new Error('Invalid reference collection.');
    const conditions=row.conditions.map(condition=>{
      if(!levelFor(condition?.id)||!['raw','tiles'].includes(condition.mode))throw new Error('Invalid comparison view.');
      const hits=count(condition.hits),targetCount=count(condition.targetCount),negativeAlertCount=count(condition.negativeAlertCount),negativeFrameCount=count(condition.negativeFrameCount),negativeFramesWithAlerts=count(condition.negativeFramesWithAlerts);
      if(hits!==null&&targetCount!==null&&hits>targetCount||negativeFrameCount!==null&&negativeFramesWithAlerts!==null&&negativeFramesWithAlerts>negativeFrameCount)throw new Error('Inconsistent comparison counts.');
      return {id:condition.id,mode:condition.mode,targetPixels:levelFor(condition.id).pixels,hits,targetCount,unmatchedPositiveBoxes:count(condition.unmatchedPositiveBoxes),negativeAlertCount,negativeFrameCount,negativeFramesWithAlerts,meanSecondsPerFrame:seconds(condition.meanSecondsPerFrame),measurementFrames:count(condition.measurementFrames)};
    });
    if(new Set(conditions.map(c=>`${c.id}/${c.mode}`)).size!==conditions.length)throw new Error('Duplicate comparison view.');
    return {id:row.id,label:text(row.label,200),referenceAuthority:text(row.referenceAuthority,160),targetCount:count(row.targetCount),sourceGroupCount:count(row.sourceGroupCount),notes:text(row.notes,3000),conditions};
  });
  if(new Set(cohorts.map(row=>row.id)).size!==2)throw new Error('Human and AI collections must remain separate.');
  const frames=input.frames.map(row=>{
    if(!row||!ID.test(row.frameId)||!COHORTS.includes(row.cohort)||!HASH.test(row.imageSha256)||!dimensions(row.width,row.height))throw new Error('Invalid original image reference.');
    const source=row.source??{};
    return {frameId:row.frameId,cohort:row.cohort,width:row.width,height:row.height,imageSha256:row.imageSha256,referenceAuthority:text(row.referenceAuthority,160),notes:text(row.notes,5000),isNegativeControl:row.isNegativeControl===true,source:{title:text(source.title,300),attribution:text(source.attribution,1500),license:text(source.license,160),sourceUrl:webUrl(source.sourceUrl),licenseUrl:webUrl(source.licenseUrl)}};
  });
  if(new Set(frames.map(row=>row.frameId)).size!==frames.length)throw new Error('Duplicate public image.');
  const frameMap=new Map(frames.map(row=>[row.frameId,row]));
  const records=input.records.map(row=>{
    const frame=frameMap.get(row?.frameId),level=levelFor(row?.conditionId);
    if(!frame||row.cohort!==frame.cohort||!level||!HASH.test(row.derivedImageSha256)||!dimensions(row.width,row.height))throw new Error('Invalid saved input.');
    if(level.id!=='native'&&(row.width!==1920||row.height!==1080))throw new Error('Synthetic inputs must preserve the recorded 1920 × 1080 canvas.');
    if(level.id==='native'&&(row.width!==frame.width||row.height!==frame.height))throw new Error('Native input dimensions changed.');
    const expectedAsset=`pixel-view-${frame.frameId}--${level.id}.png`;
    if(row.assetName!=null&&row.assetName!==expectedAsset)throw new Error('Unexpected public image asset.');
    if(level.id!=='native'&&!row.assetName||!row.assetName&&row.derivedImageSha256!==frame.imageSha256)throw new Error('Saved input is not bound to its published bytes.');
    const target=row.targetPixelsXYXY,original=row.originalTargetPixelsXYXY;
    if(target!=null&&!validBox(target,row.width,row.height)||original!=null&&!validBox(original,frame.width,frame.height))throw new Error('Reference region is out of bounds.');
    const methods={};
    for(const mode of ['raw','tiles']){
      const method=row.methods?.[mode];if(method==null)continue;
      if(!Array.isArray(method.candidates)||method.candidates.length>512)throw new Error('Invalid detector output.');
      const candidates=method.candidates.map(box=>{if(!validBox(box?.bbox,row.width,row.height)||!Number.isFinite(box.score)||box.score<0||box.score>1)throw new Error('Detector box is out of bounds.');return {bbox:[...box.bbox],score:box.score};});
      const metrics=method.metrics??{},hits=count(metrics.hits),targetCount=count(metrics.targetCount);
      if(hits!==null&&targetCount!==null&&hits>targetCount)throw new Error('Invalid saved target count.');
      methods[mode]={candidates,metrics:{hits,targetCount,unmatchedPositiveBoxes:count(metrics.unmatchedPositiveBoxes),bestIoU:probability(metrics.bestIoU)},seconds:seconds(method.seconds),modelPasses:count(method.modelPasses)};
    }
    return {frameId:frame.frameId,cohort:frame.cohort,conditionId:level.id,width:row.width,height:row.height,derivedImageSha256:row.derivedImageSha256,assetName:row.assetName??null,targetPixelsXYXY:target?[...target]:null,originalTargetPixelsXYXY:original?[...original]:null,realizedTargetWidthPixels:finite(row.realizedTargetWidthPixels),realizedTargetHeightPixels:finite(row.realizedTargetHeightPixels),geometry:numericGeometry(row.geometry??{}),methods};
  });
  if(new Set(records.map(row=>`${row.frameId}/${row.conditionId}`)).size!==records.length)throw new Error('Duplicate saved input.');
  const p=input.protocol;let recovery=null;
  if(p.recovery!=null){
    const r=p.recovery,retainedScenes=count(r.retainedScenes),recoveredScenes=count(r.recoveredScenes),totalScenes=count(r.totalScenes),uniquePersistedModelPasses=count(r.uniquePersistedModelPasses),totalAttemptedModelPasses=count(r.totalAttemptedModelPasses);
    if(r.occurred!==true||r.reason!=='file-lock'||[retainedScenes,recoveredScenes,totalScenes,uniquePersistedModelPasses].some(value=>value===null)||retainedScenes+recoveredScenes!==totalScenes||totalScenes!==records.length||uniquePersistedModelPasses!==totalScenes*5||typeof r.totalAttemptedWorkKnown!=='boolean'||r.totalAttemptedWorkKnown&&(totalAttemptedModelPasses===null||totalAttemptedModelPasses<uniquePersistedModelPasses)||!r.totalAttemptedWorkKnown&&r.totalAttemptedModelPasses!==null)throw new Error('Inconsistent recovery provenance.');
    recovery={occurred:true,reason:'file-lock',retainedScenes,recoveredScenes,totalScenes,uniquePersistedModelPasses,totalAttemptedModelPasses,totalAttemptedWorkKnown:r.totalAttemptedWorkKnown,notes:text(r.notes,3000)};
  }
  return {completedAt:text(input.completedAt,60),cohorts,frames,records,protocol:{weightsChanged:false,detectorName:text(p.detectorName,200),scoreThreshold:probability(p.scoreThreshold),iouThreshold:probability(p.iouThreshold),transformDescription:text(p.transformDescription,3000),notes:text(p.notes,3000),recovery}};
}

export function pixelRecoveryNote(recovery){
  if(!recovery)return '';
  return `Recovered run: ${recovery.retainedScenes} saved image cases were retained; the remaining ${recovery.recoveredScenes} were completed after a Windows file lock. This record contains ${recovery.totalScenes} unique image cases and ${recovery.uniquePersistedModelPasses} saved model passes. ${recovery.totalAttemptedWorkKnown?`Total attempted work: ${recovery.totalAttemptedModelPasses} model passes.`:'Total attempted work, including any unsaved attempts, is unknown.'}`;
}

export function pixelAssetUrl(frame,record,base){
  if(!frame||!record||!ID.test(frame.frameId)||!HASH.test(frame.imageSha256)||!HASH.test(record.derivedImageSha256)||record.frameId!==frame.frameId||!levelFor(record.conditionId))throw new Error('Image identity mismatch.');
  if(record.assetName!=null&&record.assetName!==`pixel-view-${frame.frameId}--${record.conditionId}.png`)throw new Error('Unexpected public image asset.');
  if(record.conditionId!=='native'&&!record.assetName||!record.assetName&&record.derivedImageSha256!==frame.imageSha256)throw new Error('Saved input is not bound to its published bytes.');
  return new URL(record.assetName?`${record.assetName}?v=${record.derivedImageSha256}`:`images/${encodeURIComponent(frame.frameId)}?v=${frame.imageSha256}`,base).href;
}

/** Source-specific navigation never presents a missing test as a measured failure. */
export function pixelLevelSelection(records,frameId,requested='native'){
  const available=new Set(records.filter(record=>record.frameId===frameId).map(record=>record.conditionId));
  const levels=PIXEL_LEVELS.map(level=>({...level,available:available.has(level.id)}));
  const selectedId=levels.find(level=>level.id===requested&&level.available)?.id??levels.find(level=>level.id==='native'&&level.available)?.id??levels.find(level=>level.available)?.id??null;
  return {levels,selectedId};
}

export function pixelViewSelection(run,url){
  const parameters=new URL(url).searchParams,frame=run.frames.find(row=>row.frameId===parameters.get('frame'));
  if(!frame)return {cohortId:'human-video',frameId:null,level:'native',mode:'raw'};
  const level=pixelLevelSelection(run.records,frame.frameId,parameters.get('level')).selectedId;
  return {cohortId:frame.cohort,frameId:frame.frameId,level:level??'native',mode:parameters.get('mode')==='tiles'?'tiles':'raw'};
}

export function pixelInspectorUrl(run,frameId,level,mode,base){
  if(!run.frames.some(frame=>frame.frameId===frameId)||!run.records.some(record=>record.frameId===frameId&&record.conditionId===level)||!['raw','tiles'].includes(mode))throw new Error('Unknown inspector view.');
  const url=new URL('./testing-pixels.html',base);url.searchParams.set('frame',frameId);url.searchParams.set('level',level);url.searchParams.set('mode',mode);url.hash='inspection-title';return url.href;
}

if(typeof document!=='undefined'){
  const $=id=>document.getElementById(id),create=(tag,cls,value)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(value!=null)node.textContent=value;return node;};
  const ratio=(a,b)=>a===null||b===null?'Not tested':`${a} / ${b}`;
  const timing=value=>value===null?'Not tested':`${value.toFixed(2)} s`;
  const number=value=>value===null?'Not tested':String(value);
  let run=null,cohortId='human-video',selectedFrame=null,selectedRecord=null,imageReady=false,scale=1,imageController=null,imageSequence=0;
  const cohort=()=>run?.cohorts.find(row=>row.id===cohortId),mode=()=>$('pixel-mode').value;

  function renderTable(){
    const current=cohort();$('cohort-title').textContent=current.label||(cohortId==='human-video'?'Fixed human-reviewed video references':'Selected AI photo references');
    const authority=cohortId==='human-video'?'Human annotations from a small fixed video set. Related video frames do not represent independent woodland trials.':'Approximate regions selected by AI from photographs. They are incomplete references, not human-confirmed ground truth.';
    $('cohort-note').textContent=[`${current.targetCount??'Unreported'} selected targets.`,authority,current.notes].filter(Boolean).join(' ');
    $('pixel-protocol').textContent=[run.protocol.detectorName,run.protocol.scoreThreshold!==null?`score threshold ${run.protocol.scoreThreshold}`:'',run.protocol.iouThreshold!==null?`reference overlap (IoU) ≥ ${run.protocol.iouThreshold}`:'','Same saved input for both modes. No weight update or automatic activation.'].filter(Boolean).join(' · ');
    $('pixel-rows').replaceChildren();
    for(const level of PIXEL_LEVELS){
      const raw=current.conditions.find(c=>c.id===level.id&&c.mode==='raw'),tiles=current.conditions.find(c=>c.id===level.id&&c.mode==='tiles'),row=create('tr');if($('pixel-level').value===level.id)row.className='pixel-row-selected';
      const name=create('td','',level.label);name.append(create('small','',level.distance?`Illustrative ${level.distance} m · assumptions below`:level.id==='native'?'Original image dimensions':'1920 × 1080 synthetic canvas'));row.append(name,create('td','',raw?ratio(raw.hits,raw.targetCount):'Not tested'),create('td','',tiles?ratio(tiles.hits,tiles.targetCount):'Not tested'),create('td','',number(raw?.unmatchedPositiveBoxes??null)),create('td','',number(tiles?.unmatchedPositiveBoxes??null)));
      const negatives=create('td','',`${number(raw?.negativeAlertCount??null)} / ${number(tiles?.negativeAlertCount??null)}`);if(raw?.negativeFrameCount!==null&&raw?.negativeFrameCount!==undefined)negatives.append(create('small','',`Frames with alerts: ${ratio(raw.negativeFramesWithAlerts,raw.negativeFrameCount)} whole; ${tiles?ratio(tiles.negativeFramesWithAlerts,tiles.negativeFrameCount):'not tested'} tiled`));row.append(negatives);
      const elapsed=create('td','',`${timing(raw?.meanSecondsPerFrame??null)} / ${timing(tiles?.meanSecondsPerFrame??null)}`);if(raw?.measurementFrames!==null&&raw?.measurementFrames!==undefined)elapsed.append(create('small','',`${raw.measurementFrames} whole-frame / ${tiles?.measurementFrames??'unreported'} tiled measured images`));row.append(elapsed);$('pixel-rows').append(row);
    }
    $('negative-scope').textContent=(cohortId==='human-video'?'The video control uses a separately human-reviewed absence reference.':'The negative controls use older AI absence observations from video, separate from these new positive photographs.')+' Controls were tested at native size only. Per-image timing describes work on this test computer, not camera frame rate or real-time flight; parallel-worker wall time is different.';
  }

  function chooseCohort(id,preferredFrameId=null){
    if(!run||!COHORTS.includes(id))return;cohortId=id;
    for(const key of COHORTS){const button=$(`${key}-tab`);button.setAttribute('aria-selected',String(key===id));button.tabIndex=key===id?0:-1;}
    $('cohort-panel').setAttribute('aria-labelledby',`${id}-tab`);const select=$('pixel-source');select.replaceChildren();
    for(const frame of run.frames.filter(row=>row.cohort===id)){const option=create('option','',`${frame.isNegativeControl?'Control · ':''}${frame.source.title?`${frame.source.title} · `:''}${frame.frameId}`);option.value=frame.frameId;select.append(option);}if(preferredFrameId&&Array.from(select.options).some(option=>option.value===preferredFrameId))select.value=preferredFrameId;
    select.disabled=!select.options.length;$('pixel-level').disabled=!select.options.length;$('pixel-mode').disabled=!select.options.length;renderTable();selectInput();
  }

  function drawOverlay(){
    const overlay=$('pixel-overlay');overlay.replaceChildren();if(!imageReady||!selectedRecord)return;overlay.setAttribute('viewBox',`0 0 ${selectedRecord.width} ${selectedRecord.height}`);
    const rectangle=(bbox,cls,title)=>{const [x1,y1,x2,y2]=bbox,rect=document.createElementNS('http://www.w3.org/2000/svg','rect');for(const [key,value] of Object.entries({x:x1,y:y1,width:x2-x1,height:y2-y1,class:cls}))rect.setAttribute(key,String(value));const label=document.createElementNS('http://www.w3.org/2000/svg','title');label.textContent=title;rect.append(label);overlay.append(rect);};
    if($('pixel-predictions-toggle').checked)for(const prediction of selectedRecord.methods[mode()]?.candidates??[])rectangle(prediction.bbox,'pixel-prediction',`Detector score ${prediction.score.toFixed(3)}; not a calibrated probability`);
    if($('pixel-reference-toggle').checked&&selectedRecord.targetPixelsXYXY)rectangle(selectedRecord.targetPixelsXYXY,'pixel-reference','Selected reference region');
  }

  function zoom(){
    if(!selectedRecord)return;const value=$('pixel-zoom').value,viewport=$('pixel-viewport');scale=value==='fit'?Math.min(1,viewport.clientWidth/selectedRecord.width):Number(value);const stage=$('pixel-stage');stage.style.width=`${selectedRecord.width*scale}px`;stage.style.height=`${selectedRecord.height*scale}px`;stage.classList.toggle('pixelated',Number(value)>=2);if(value==='fit'){viewport.scrollLeft=0;viewport.scrollTop=0;}
  }

  function centerTarget(){
    if(!selectedRecord?.targetPixelsXYXY||!imageReady)return;if($('pixel-zoom').value==='fit'){$('pixel-zoom').value='8';zoom();}const [x1,y1,x2,y2]=selectedRecord.targetPixelsXYXY,viewport=$('pixel-viewport');viewport.scrollLeft=(x1+x2)/2*scale-viewport.clientWidth/2;viewport.scrollTop=(y1+y2)/2*scale-viewport.clientHeight/2;
  }

  function renderDetails(){
    const record=selectedRecord,frame=selectedFrame;if(!record||!frame)return;
    $('pixel-source-title').textContent=frame.source.title||frame.frameId;$('pixel-source-id').textContent=`Source image: ${frame.frameId}`;$('pixel-source-note').textContent=frame.notes||(cohortId==='ai-photo'?'Selected AI reference; incomplete and unverified.':'Fixed human-reviewed video reference; limited to this source.');$('pixel-share-link').href=pixelInspectorUrl(run,frame.frameId,record.conditionId,mode(),document.baseURI);
    const credit=$('pixel-credit');credit.replaceChildren(create('p','',[frame.source.attribution,frame.source.license?`License: ${frame.source.license}`:''].filter(Boolean).join(' · ')));
    for(const [label,url] of [['Original source ↗',frame.source.sourceUrl],['License ↗',frame.source.licenseUrl]]){if(!url)continue;const link=create('a','',label);link.href=url;link.target='_blank';link.rel='noopener noreferrer';credit.append(link);}
    $('pixel-lab-link').href=new URL(`./?frame=${encodeURIComponent(frame.frameId)}#review-workspace`,document.baseURI).href;
    const method=record.methods[mode()],result=frame.isNegativeControl||!record.targetPixelsXYXY?`No positive reference; ${method?.candidates.length??0} alerts.`:`${method?ratio(method.metrics.hits,method.metrics.targetCount):'Not tested'} selected targets matched. ${method?.candidates.length??0} saved detector boxes.`;$('pixel-result').textContent=method?`${result} ${timing(method.seconds)} of measured method work; ${method.modelPasses??'unreported'} model passes. Scores are not calibrated species probabilities.`:'This method has no saved result for the selected input.';
    if(method&&record.targetPixelsXYXY&&!frame.isNegativeControl)$('pixel-result').append(create('span','processor-size',`Best reference overlap: ${method.metrics.bestIoU===null?'unavailable':method.metrics.bestIoU.toFixed(3)}${run.protocol.iouThreshold===null?'.':`; match requires ≥ ${run.protocol.iouThreshold.toFixed(2)}.`}`));
    const box=record.targetPixelsXYXY,w=record.realizedTargetWidthPixels??(box?box[2]-box[0]:null),h=record.realizedTargetHeightPixels??(box?box[3]-box[1]:null);
    $('pixel-geometry').textContent=box?`Realized reference size: ${w.toFixed(3)} × ${h.toFixed(3)} pixels. ${record.conditionId==='native'?'Native original; no synthetic transform.':'Whole scene reduced on the fixed gray canvas; no re-enlargement.'}`:'Negative control: no positive target is marked. An alert still needs inspection.';
    const passes=record.geometry?.processorAndTileCoverage?.passes??[],context=passes.find(pass=>pass.kind==='context'),covered=passes.filter(pass=>pass.kind?.startsWith('tile-')&&pass.selectedTargetCoverageFraction>0&&Number.isFinite(pass.selectedTargetInputWidthPixels));
    if(box&&Number.isFinite(context?.selectedTargetInputWidthPixels))$('pixel-geometry').append(create('span','processor-size',`After the model’s input resizing, the whole-frame reference is ${context.selectedTargetInputWidthPixels.toFixed(3)} pixels wide.${covered.length?` Visible portions in ${covered.length} covering tiles range from ${Math.min(...covered.map(pass=>pass.selectedTargetInputWidthPixels)).toFixed(3)} to ${Math.max(...covered.map(pass=>pass.selectedTargetInputWidthPixels)).toFixed(3)} pixels wide.`:''} The saved canvas and the model’s resized input are different coordinate spaces.`));
    $('pixel-dimensions').textContent=`${record.width} × ${record.height} saved pixels${record.conditionId==='native'?' · native':' · synthetic canvas'}`;
    $('pixel-identity').textContent=JSON.stringify({frameId:frame.frameId,referenceAuthority:frame.referenceAuthority||cohort().referenceAuthority,input:record.conditionId,derivedImageSha256:record.derivedImageSha256,originalImageSha256:frame.imageSha256,targetPixelsXYXY:record.targetPixelsXYXY,geometry:record.geometry},null,2);$('pixel-center').disabled=!box;$('pixel-reference-toggle').disabled=!box;drawOverlay();
  }

  async function readVerifiedImage(frame,record,signal){
    const response=await fetch(pixelAssetUrl(frame,record,document.baseURI),{signal,cache:'force-cache'});
    if(!response.ok||Number(response.headers.get('content-length'))>8*1024*1024)throw new Error('The saved image is unavailable or too large.');
    const blob=await response.blob();if(blob.size>8*1024*1024)throw new Error('The saved image exceeds its size limit.');
    const digest=await crypto.subtle.digest('SHA-256',await blob.arrayBuffer()),hash=Array.from(new Uint8Array(digest),byte=>byte.toString(16).padStart(2,'0')).join('');if(hash!==record.derivedImageSha256)throw new Error('The saved image did not match its recorded hash.');
    return await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('The image could not be decoded.'));reader.readAsDataURL(blob);});
  }

  async function selectInput(){
    imageController?.abort();const controller=new AbortController();imageController=controller;const sequence=++imageSequence;imageReady=false;$('pixel-overlay').replaceChildren();$('pixel-image').hidden=true;
    selectedFrame=run?.frames.find(row=>row.frameId===$('pixel-source').value);const selection=pixelLevelSelection(run?.records??[],selectedFrame?.frameId,$('pixel-level').value);for(const option of $('pixel-level').options)option.disabled=!selection.levels.find(level=>level.id===option.value)?.available;$('pixel-level').disabled=selection.selectedId===null;if(selection.selectedId)$('pixel-level').value=selection.selectedId;
    selectedRecord=run?.records.find(row=>row.frameId===selectedFrame?.frameId&&row.conditionId===selection.selectedId);$('pixel-viewer').hidden=!selectedRecord;renderTable();if(!selectedRecord){$('pixels-status').textContent='No saved input is available for this source and target width. Missing views are not treated as failures or zero scores.';return;}
    const frame=selectedFrame,record=selectedRecord;$('pixels-status').textContent=`${run.frames.length} original references and controls · ${run.records.length} saved image inputs. Human video and AI photo results remain separate.`;renderDetails();zoom();$('pixel-image-status').textContent='Checking the actual saved image bytes…';$('pixel-open-image').href=pixelAssetUrl(frame,record,document.baseURI);
    const timeout=setTimeout(()=>controller.abort(),20000);
    try{const dataUrl=await readVerifiedImage(frame,record,controller.signal);if(sequence!==imageSequence)return;const img=$('pixel-image');img.width=record.width;img.height=record.height;img.alt=`${record.conditionId==='native'?'Original':'Saved synthetic'} image for ${frame.source.title||frame.frameId}. Reference and detections are research observations.`;
      img.onload=()=>{if(sequence!==imageSequence)return;if(img.naturalWidth!==record.width||img.naturalHeight!==record.height){$('pixel-image-status').textContent='The image dimensions do not match the saved input. Outlines are hidden.';return;}img.hidden=false;imageReady=true;zoom();if($('pixel-zoom').value==='8')centerTarget();drawOverlay();$('pixel-image-status').textContent='Saved image hash and dimensions verified. Cyan boxes are detector output; gold is the reference. At 1×, one image pixel occupies one CSS pixel. Turn off both outlines to inspect tiny targets without obstruction.';};
      img.onerror=()=>{if(sequence===imageSequence)$('pixel-image-status').textContent='The verified image could not be decoded. Outlines are hidden.';};img.src=dataUrl;
    }catch(error){if(sequence===imageSequence)$('pixel-image-status').textContent=error.name==='AbortError'?'The image request timed out. Choose the source again to retry.':error.message;}
    finally{clearTimeout(timeout);}
  }

  async function load(){
    $('pixels-error').hidden=true;$('pixels-retry').disabled=true;
    try{const response=await fetch(new URL('./testing-pixels.json',document.baseURI),{cache:'no-store',headers:{Accept:'application/json'},signal:AbortSignal.timeout(20000)});if(!response.ok||Number(response.headers.get('content-length'))>8*1024*1024)throw new Error('The saved public comparison is not available.');const raw=await response.text();if(raw.length>8*1024*1024)throw new Error('The public comparison exceeds its size limit.');run=publicPixels(JSON.parse(raw));const date=new Date(run.completedAt);$('pixel-run-date').textContent=Number.isFinite(date.valueOf())?`Saved ${date.toLocaleDateString(undefined,{year:'numeric',month:'long',day:'numeric'})}`:'Saved run date unavailable';$('pixels-status').textContent=`${run.frames.length} original references and controls · ${run.records.length} saved image inputs. Human video and AI photo results remain separate.`;
      $('pixel-recovery').hidden=!run.protocol.recovery;$('pixel-recovery').textContent=pixelRecoveryNote(run.protocol.recovery);
      const select=$('pixel-level');select.replaceChildren();for(const level of PIXEL_LEVELS){const option=create('option','',level.label+(level.distance?` · illustrative ${level.distance} m`:''));option.value=level.id;select.append(option);}const view=pixelViewSelection(run,document.location.href);select.value=view.level;$('pixel-mode').value=view.mode;chooseCohort(view.cohortId,view.frameId);
    }catch(error){$('pixels-error').hidden=false;$('pixels-error-text').textContent=error.message+' No missing measurement has been replaced with zero.';$('pixels-status').textContent='The saved run is unavailable. Opening this page does not launch inference or training.';}
    finally{$('pixels-retry').disabled=false;}
  }
  for(const id of COHORTS){const button=$(`${id}-tab`);button.addEventListener('click',()=>chooseCohort(id));button.addEventListener('keydown',event=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(event.key))return;event.preventDefault();const next=event.key==='Home'?'human-video':event.key==='End'?'ai-photo':id==='human-video'?'ai-photo':'human-video';chooseCohort(next);$(`${next}-tab`).focus();});}
  $('pixel-source').addEventListener('change',selectInput);$('pixel-level').addEventListener('change',selectInput);$('pixel-mode').addEventListener('change',renderDetails);$('pixel-zoom').addEventListener('change',()=>{zoom();if($('pixel-zoom').value==='8')centerTarget();});$('pixel-reference-toggle').addEventListener('change',drawOverlay);$('pixel-predictions-toggle').addEventListener('change',drawOverlay);$('pixels-retry').addEventListener('click',load);
  $('pixel-center').addEventListener('click',centerTarget);
  new ResizeObserver(()=>{if($('pixel-zoom').value==='fit')zoom();}).observe($('pixel-viewport'));load();
}
