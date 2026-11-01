const SCHEMA = 'mushroom-vision.public-testing-round4.v1';
const FRAME_ID = /^[A-Za-z0-9_-]{1,160}$/;
const HASH = /^[a-f0-9]{64}$/;
const text = (value, length = 2000) => typeof value === 'string' ? value.slice(0, length) : '';
const count = value => Number.isSafeInteger(value) && value >= 0 && value <= 100000 ? value : null;
const seconds = value => Number.isFinite(value) && value >= 0 && value <= 86400 ? value : null;
const webUrl = value => { try { const url = new URL(value); return ['http:', 'https:'].includes(url.protocol) && !url.username && !url.password ? url.href : null; } catch { return null; } };

/** Only an explicit, bounded public view is retained; private report fields are ignored. */
export function publicRound4(input) {
  if (input?.schema !== SCHEMA || !Array.isArray(input.frames) || input.frames.length < 1 || input.frames.length > 32 || !Array.isArray(input.conditions) || input.conditions.length < 1 || input.conditions.length > 8 || input.summary?.referenceAuthority !== 'ai-incomplete' || input.summary?.weightsChanged !== false) throw new Error('The saved experiment has an unsupported format.');
  const conditions = input.conditions.map(row => {
    if (!row || !/^[a-z0-9-]{1,40}$/.test(row.id)) throw new Error('A saved comparison is invalid.');
    const hits=count(row.selectedTargetHits), targets=count(row.selectedTargetCount), negatives=count(row.negativeFramesWithAlerts), negativeFrames=count(row.negativeFrameCount);
    if (hits !== null && targets !== null && hits > targets || negatives !== null && negativeFrames !== null && negatives > negativeFrames) throw new Error('A saved comparison has inconsistent counts.');
    return {id:row.id,label:text(row.label,160)||row.id,selectedTargetHits:hits,selectedTargetCount:targets,negativeFramesWithAlerts:negatives,negativeFrameCount:negativeFrames,negativeAlertCount:count(row.negativeAlertCount),meanSecondsPerFrame:seconds(row.meanSecondsPerFrame),measurementFrames:count(row.measurementFrames)};
  });
  if (new Set(conditions.map(row=>row.id)).size !== conditions.length) throw new Error('Duplicate saved comparison.');
  const frames = input.frames.map(row => {
    if (!row || !FRAME_ID.test(row.frameId) || !HASH.test(row.imageSha256) || !Number.isInteger(row.width) || !Number.isInteger(row.height) || row.width < 1 || row.height < 1 || row.width * row.height > 20000000 || !['visible','absent','uncertain'].includes(row.status)) throw new Error('An image reference is invalid.');
    const box=row.selectedTargetPixelsXYXY;
    if (box != null && (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite) || box[0] < 0 || box[1] < 0 || box[2] <= box[0] || box[3] <= box[1] || box[2] > row.width || box[3] > row.height)) throw new Error('An image reference box is out of bounds.');
    const results=(Array.isArray(row.results)?row.results:[]).slice(0,8).filter(result=>conditions.some(condition=>condition.id===result?.conditionId)).map(result=>({conditionId:result.conditionId,selectedTargetHit:typeof result.selectedTargetHit==='boolean'?result.selectedTargetHit:null,alertCount:count(result.alertCount),elapsedSeconds:seconds(result.elapsedSeconds)}));
    const source=row.source??{};
    return {frameId:row.frameId,imageSha256:row.imageSha256,width:row.width,height:row.height,status:row.status,note:text(row.note,6000),groupHint:text(row.groupHint,160),selectedTargetPixelsXYXY:box?[...box]:null,results,source:{title:text(source.title,300),attribution:text(source.attribution,1200),license:text(source.license,160),sourceUrl:webUrl(source.sourceUrl),licenseUrl:webUrl(source.licenseUrl)}};
  });
  if (new Set(frames.map(row=>row.frameId)).size !== frames.length) throw new Error('Duplicate image reference.');
  const protocol=input.protocol??{};
  return {title:text(input.title,200),completedAt:text(input.completedAt,60),summary:{positiveReferences:count(input.summary.positiveReferences),negativeReferences:count(input.summary.negativeReferences),referenceAuthority:'ai-incomplete',weightsChanged:false},protocol:{detectorName:text(protocol.detectorName,200),iouThreshold:Number.isFinite(protocol.iouThreshold)&&protocol.iouThreshold>=0&&protocol.iouThreshold<=1?protocol.iouThreshold:null,scoreThreshold:Number.isFinite(protocol.scoreThreshold)&&protocol.scoreThreshold>=0&&protocol.scoreThreshold<=1?protocol.scoreThreshold:null,notes:text(protocol.notes,2000)},conditions,frames};
}

if (typeof document !== 'undefined') {
  const $=id=>document.getElementById(id), create=(tag,cls,value)=>{const node=document.createElement(tag);if(cls)node.className=cls;if(value!=null)node.textContent=value;return node;};
  let run=null,selected=null,sequence=0;
  const formatSeconds=value=>value===null?'Unavailable':`${value.toFixed(2)} s`;
  const ratio=(a,b)=>a===null||b===null?'Not tested':`${a} / ${b}`;
  const imageUrl=row=>new URL(`images/${encodeURIComponent(row.frameId)}?v=${row.imageSha256}`,document.baseURI).href;
  const statusLabel=value=>({visible:'AI sees mushrooms',absent:'Prior AI sees no mushrooms',uncertain:'AI is uncertain'})[value];

  function renderSummary() {
    const date=new Date(run.completedAt);$('run-date').textContent=Number.isFinite(date.valueOf())?`Saved ${date.toLocaleDateString(undefined,{year:'numeric',month:'long',day:'numeric'})}`:'Saved run date unavailable';
    $('results-status').textContent=`${run.frames.length} new original-image AI reviews · ${run.summary.positiveReferences??'unreported'} selected positive regions · ${run.summary.negativeReferences??'unreported'} older negative controls reported separately. Model weights were not changed in this round.`;
    const protocol=[];if(run.protocol.detectorName)protocol.push(run.protocol.detectorName);if(run.protocol.scoreThreshold!==null)protocol.push(`score threshold ${run.protocol.scoreThreshold}`);if(run.protocol.iouThreshold!==null)protocol.push(`reference overlap (IoU) at least ${run.protocol.iouThreshold}`);
    $('run-protocol').textContent=[protocol.join(' · '),run.protocol.notes].filter(Boolean).join('. ')||'Counts refer to the selected AI target in each positive image. This is a preliminary diagnostic comparison, not a field accuracy estimate.';
    $('condition-results').replaceChildren();
    for(const row of run.conditions){const tr=create('tr');tr.append(create('td','',row.label),create('td','',ratio(row.selectedTargetHits,row.selectedTargetCount)),create('td','',ratio(row.negativeFramesWithAlerts,row.negativeFrameCount)),create('td','',row.negativeAlertCount===null?'Not tested':String(row.negativeAlertCount)));const timing=create('td','',formatSeconds(row.meanSecondsPerFrame));if(row.measurementFrames!==null)timing.append(create('small','',`${row.measurementFrames} timed image${row.measurementFrames===1?'':'s'}`));tr.append(timing);$('condition-results').append(tr);}
    $('timing-scope').textContent='Timing describes this saved run on its test computer. Different views can have different numbers of measured images. These timings do not establish drone speed, camera frame rate or live flight performance.';
  }

  function renderFrames(){
    const filter=$('frame-filter').value,visible=run.frames.filter(row=>filter==='all'||row.status===filter);$('review-frames').replaceChildren();$('empty-filter').hidden=visible.length>0;
    for(const [index,row] of run.frames.entries()){
      if(!visible.includes(row))continue;
      const button=create('button','review-frame'),img=create('img');button.type='button';button.setAttribute('aria-pressed',String(selected?.frameId===row.frameId));button.setAttribute('aria-label',`Inspect original ${index+1}: ${statusLabel(row.status)}`);img.src=imageUrl(row);img.alt='';img.loading='lazy';button.append(img,create('span','',`Original ${index+1} · ${statusLabel(row.status)}`));button.addEventListener('click',()=>{selectFrame(row);$('selected-review').scrollIntoView({behavior:'auto',block:'start'});});$('review-frames').append(button);
    }
  }

  function renderOutline(){
    const overlay=$('reference-overlay');overlay.replaceChildren();if(!selected)return;overlay.setAttribute('viewBox',`0 0 ${selected.width} ${selected.height}`);if(!$('show-reference').checked||!selected.selectedTargetPixelsXYXY||!$('original-error').hidden)return;
    const [x1,y1,x2,y2]=selected.selectedTargetPixelsXYXY,rect=document.createElementNS('http://www.w3.org/2000/svg','rect');for(const [key,value] of Object.entries({x:x1,y:y1,width:x2-x1,height:y2-y1}))rect.setAttribute(key,String(value));overlay.append(rect);
  }

  function selectFrame(row){
    selected=row;const token=++sequence;$('selected-review').hidden=false;$('selected-title').textContent=row.source.title||`Original ${run.frames.indexOf(row)+1}`;$('reference-status').textContent=statusLabel(row.status);$('reference-note').textContent=row.note||'No additional visual note was saved.';$('group-hint').textContent=row.groupHint?`Source group hint: ${row.groupHint}. This is not a verified identification.`:'No species or group identification is established.';$('original-size').textContent=`${row.width} × ${row.height} original pixels`;
    $('open-original').href=imageUrl(row);$('review-in-lab').href=new URL(`./?frame=${encodeURIComponent(row.frameId)}#review-workspace`,document.baseURI).href;$('frame-identity').textContent=`Frame: ${row.frameId}\nImage SHA-256: ${row.imageSha256}`;
    const box=row.selectedTargetPixelsXYXY;$('show-reference').disabled=!box;$('reference-description').textContent=box?`Dashed outline: one selected AI reference region, ${Math.round(box[2]-box[0])} × ${Math.round(box[3]-box[1])} pixels in the original. It is not a detector output or an exhaustive annotation.`:'No selected target was scored for this image. Read the qualitative AI observation; an unmarked region is not a negative example.';
    $('original-error').hidden=true;$('reference-overlay').replaceChildren();const img=$('review-original');img.width=row.width;img.height=row.height;img.alt=`Original under review: ${row.source.title||row.frameId}. ${statusLabel(row.status)}; unverified.`;img.onload=()=>{if(token!==sequence)return;const valid=img.naturalWidth===row.width&&img.naturalHeight===row.height;$('original-error').hidden=valid;renderOutline();};img.onerror=()=>{if(token!==sequence)return;$('original-error').hidden=false;$('reference-overlay').replaceChildren();};img.src=imageUrl(row);
    const credit=$('image-credit');credit.replaceChildren();credit.append(create('p','',[row.source.attribution,row.source.license?`License: ${row.source.license}`:''].filter(Boolean).join(' · ')||'Source attribution is available on the linked lab review.'));for(const [label,url] of [['Original source ↗',row.source.sourceUrl],['License ↗',row.source.licenseUrl]]){if(!url)continue;const link=create('a','',label);link.href=url;link.target='_blank';link.rel='noopener noreferrer';credit.append(link);}
    $('frame-results').replaceChildren();for(const condition of run.conditions){const result=row.results.find(value=>value.conditionId===condition.id),item=create('div','frame-result');item.append(create('strong','',condition.label));if(!result)item.append(create('span','','Not tested'));else{const target=result.selectedTargetHit===null?'No positive target scored':result.selectedTargetHit?'Selected AI target matched':'Selected AI target not matched';item.append(create('span','',`${target} · ${result.alertCount===null?'alert count unavailable':`${result.alertCount} alert${result.alertCount===1?'':'s'}`} · ${formatSeconds(result.elapsedSeconds)}`));}$('frame-results').append(item);}renderFrames();
  }

  async function load(){
    $('results-error').hidden=true;$('results-status').textContent='Loading the saved experiment…';$('retry-results').disabled=true;
    try{const response=await fetch(new URL('./testing-round4.json',document.baseURI),{cache:'no-store',headers:{Accept:'application/json'},signal:AbortSignal.timeout(15000)});if(!response.ok||Number(response.headers.get('content-length'))>512000)throw new Error('Saved results unavailable.');const raw=await response.text();if(raw.length>512000)throw new Error('Saved results exceed the public size limit.');run=publicRound4(JSON.parse(raw));renderSummary();renderFrames();selectFrame(run.frames[0]);}
    catch{$('results-error').hidden=false;$('results-status').textContent='The saved run is unavailable. Reloading this page does not start model training or testing.';}
    finally{$('retry-results').disabled=false;}
  }
  $('retry-results').addEventListener('click',load);$('frame-filter').addEventListener('change',()=>{if(run)renderFrames();});$('show-reference').addEventListener('change',renderOutline);load();
}
