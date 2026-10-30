import { CandidateTracker,LatestFrameGate,closeLookProposal,syntheticFrame } from './tracking-logic.mjs';

const $=id=>document.getElementById(id),node=(tag,text,cls)=>{const element=document.createElement(tag);if(text!==undefined)element.textContent=text;if(cls)element.className=cls;return element;};
const video=$('video'),canvas=$('scene'),ctx=canvas.getContext('2d'),tracker=new CandidateTracker(),gate=new LatestFrameGate();
let station=localStorage.getItem('mycoscout-station')||'',token='',epoch=0,mode='loading',sourceId='',demo=null,replayIndex=0,videoURL=null,stream=null,ready=false;
let syntheticTime=0,syntheticPlaying=false,lastSyntheticSample=-1,lastTick=performance.now(),lastUi=0,cameraStarted=0;
let automatic=false,lastRequested=-Infinity,requestCount=0,inferenceSeconds=null,proposals=[],lastTracks=[],sourceWidth=640,sourceHeight=480;
const capture=document.createElement('canvas');
const time=()=>mode==='synthetic'?syntheticTime:mode==='camera'?(performance.now()-cameraStarted)/1000:video.currentTime||0;
const duration=()=>mode==='synthetic'?20:Number.isFinite(video.duration)?video.duration:0;
const fmt=value=>Number.isFinite(value)?value.toFixed(2):'—';
function notice(text){$('notice').textContent=text;$('notice').hidden=!text;}
function controls(){
  const liveSource=['file','camera'].includes(mode),holdBusy=gate.pending&&mode==='file'&&$('hold-video').checked;
  $('play').disabled=!ready||mode==='camera'||holdBusy;$('seek').disabled=!ready||mode==='camera'||holdBusy;
  $('sample-once').disabled=!ready||!liveSource||gate.pending||requestCount>=60;
  $('sample-auto').disabled=!ready||!liveSource||requestCount>=60;
  $('sample-auto').textContent=automatic?'Stop sampled analysis':'Start sampled analysis';
  $('hold-video').disabled=mode!=='file'||gate.pending;
  $('play').textContent=(mode==='synthetic'?syntheticPlaying:!video.paused)?'Ⅱ Pause':'▶ Play';
  $('export-proposals').disabled=!proposals.length;
}
async function request(path,{body,raw=false}={}){
  const requestEpoch=epoch,response=await fetch(station+path,{method:body===undefined?'GET':'POST',headers:{...(body===undefined?{}:{'Content-Type':'application/json'}),...(token?{Authorization:'Bearer '+token}:{})},...(body===undefined?{}:{body:JSON.stringify(body)}),cache:'no-store',signal:AbortSignal.timeout(240000)});
  if(requestEpoch!==epoch)throw new Error('The source or connection changed; the earlier response was discarded.');
  if(!response.ok){let error;try{error=await response.json();}catch{}throw new Error(typeof error?.error==='string'?error.error:error?.error?.message||`The station returned ${response.status}.`);}
  return raw?response:response.json();
}
function resetTracking(message='Tracking reset for this source.'){
  gate.reset();tracker.reset(sourceId);lastTracks=[];lastSyntheticSample=-1;lastRequested=-Infinity;
  replayIndex=demo?.frames.findIndex(frame=>frame.timestampSeconds>=time())??0;if(replayIndex<0)replayIndex=demo?.frames.length??0;
  $('tracking-note').textContent=message;renderTracks();
}
function clearSource(next){
  epoch++;gate.reset();automatic=false;ready=false;video.pause();syntheticPlaying=false;
  if(stream){for(const track of stream.getTracks())track.stop();stream=null;}video.srcObject=null;video.removeAttribute('src');video.load();
  if(videoURL){URL.revokeObjectURL(videoURL);videoURL=null;}
  mode=next;sourceId=next+'-'+epoch;demo=null;syntheticTime=0;requestCount=0;inferenceSeconds=null;proposals=[];
  $('proposal-count').textContent='0';$('proposals').replaceChildren();$('credits').replaceChildren();$('sample-caption').textContent='No frame analyzed in this session.';
  const sample=$('sample-canvas');sample.getContext('2d').clearRect(0,0,sample.width,sample.height);
  $('viewer-message').hidden=false;$('viewer-message').textContent='Opening this source…';resetTracking();controls();notice('');
}
function dimensions(width,height){sourceWidth=width;sourceHeight=height;canvas.width=width;canvas.height=height;$('viewer').style.aspectRatio=`${width}/${height}`;}
function credit(source){$('credits').replaceChildren(node('span',source.attribution));for(const [label,value]of [['Source recording',source.sourceUrl],['License',source.licenseUrl]]){try{const url=new URL(value);if(url.protocol!=='https:')continue;const link=node('a',label+' ↗');link.href=url.href;link.target='_blank';link.rel='noopener';$('credits').append(link);}catch{}}}
async function loadReplay(){
  clearSource('replay');const currentEpoch=epoch;$('source-mode').textContent='RECORDED VIDEO + PRECOMPUTED DETECTIONS';$('source-title').textContent='Opening the licensed example…';$('frame-badge').textContent='PRECOMPUTED REPLAY';
  try{const result=await request('/api/tracking-demo');if(currentEpoch!==epoch)return;
    if(!result.available){useSynthetic(result.notice);return;}
    demo=result;sourceId=result.source.id;tracker.reset(sourceId);const response=await request(result.videoUrl,{raw:true});
    if(Number(response.headers.get('content-length'))>16*1024*1024)throw new Error('The replay exceeds the preview size limit.');
    const blob=await response.blob();if(currentEpoch!==epoch)return;if(blob.size>16*1024*1024)throw new Error('The replay exceeds the preview size limit.');
    videoURL=URL.createObjectURL(blob);video.src=videoURL;video.load();$('source-title').textContent=result.source.title;
    $('source-note').textContent=result.notice+' The video plays continuously; only '+result.frames.length+' recorded frames had detector results.';credit(result.source);
    $('connection-state').textContent='Field station connected';$('sample-status').textContent='This example reuses saved detections. Choose your own video or camera to run new inference.';
  }catch(error){if(currentEpoch===epoch){$('connection-state').textContent='Replay unavailable';useSynthetic('The real replay could not load. '+error.message);}}
}
function useSynthetic(reason=''){
  clearSource('synthetic');sourceId='synthetic-'+epoch;tracker.reset(sourceId);dimensions(640,480);ready=true;
  $('source-mode').textContent='SYNTHETIC TRACKING EXAMPLE';$('source-title').textContent='Illustrated candidate motion and an occlusion';$('frame-badge').textContent='SYNTHETIC · NOT REAL FOOTAGE';
  $('source-note').textContent='Programmed shapes and boxes demonstrate association, prediction and loss. This is not a detector result, a real mushroom, or evidence of detection accuracy.';
  $('sample-status').textContent='Synthetic mode does not run inference or produce training labels.';$('viewer-message').hidden=true;controls();if(reason)notice(reason+' Showing the explicitly labeled synthetic example.');
}
video.addEventListener('loadedmetadata',()=>{if(!['replay','file','camera'].includes(mode))return;if(video.videoWidth<1)return;dimensions(video.videoWidth,video.videoHeight);ready=true;$('viewer-message').hidden=true;controls();});
video.addEventListener('error',()=>{if(!['replay','file'].includes(mode))return;ready=false;$('viewer-message').hidden=false;$('viewer-message').textContent='This video format could not play in this browser. Try an MP4/H.264 video.';controls();});
video.addEventListener('seeking',()=>{if(mode!=='camera')resetTracking('Seeking cleared the old tracks and discarded any pending result.');});
video.addEventListener('ended',()=>{automatic=false;controls();});
$('play').onclick=async()=>{if(mode==='synthetic'){if(syntheticTime>=20){syntheticTime=0;resetTracking();}syntheticPlaying=!syntheticPlaying;}else{if(video.ended)video.currentTime=0;if(video.paused)await video.play().catch(error=>notice(error.message));else video.pause();}controls();};
$('seek').oninput=()=>{if(mode==='synthetic'){syntheticTime=Number($('seek').value);resetTracking('Seeking starts a new candidate association.');}else video.currentTime=Number($('seek').value);};
$('load-replay').onclick=loadReplay;$('synthetic').onclick=()=>useSynthetic();
$('video-file').onchange=()=>{const file=$('video-file').files?.[0];if(!file)return;if(file.size>1024*1024*1024){notice('Choose a video smaller than 1 GB for this browser preview.');return;}
  clearSource('file');sourceId='local-video-'+epoch;tracker.reset(sourceId);videoURL=URL.createObjectURL(file);video.src=videoURL;video.load();
  $('source-mode').textContent='YOUR VIDEO · SAMPLED NEW INFERENCE';$('source-title').textContent=file.name;$('frame-badge').textContent='LOCAL VIDEO · SAMPLED VISION';
  $('source-note').textContent='The video stays in this browser. Analyze sends one resized JPEG frame to your field station; the tracking endpoint does not save it or add training labels.';
  $('sample-status').textContent='Play or seek to a view, then analyze a frame. Pause-during-analysis keeps slow CPU results aligned with the displayed frame.';
};
$('start-camera').onclick=async()=>{
  if(!navigator.mediaDevices?.getUserMedia){notice('Camera access requires a supported browser and trusted HTTPS, or localhost.');return;}
  clearSource('camera');const currentEpoch=epoch;$('source-mode').textContent='LIVE CAMERA · SAMPLED INFERENCE';$('source-title').textContent='Your camera';$('frame-badge').textContent='LIVE CAMERA · SAMPLED VISION';
  try{const acquired=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:720}},audio:false});if(currentEpoch!==epoch){for(const track of acquired.getTracks())track.stop();return;}
    stream=acquired;video.srcObject=stream;cameraStarted=performance.now();await video.play();$('source-note').textContent='Live camera preview. Only analyzed frames reach your field station. Old results are labeled stale and are never presented as fresh camera evidence.';
    $('sample-status').textContent='Choose Analyze or Start sampled analysis. Camera capture stops when you switch source or leave this page.';
  }catch(error){if(currentEpoch===epoch){notice('Camera could not open: '+error.message);$('viewer-message').textContent='Camera unavailable. Choose a video or the recording.';}}
};

function drawBoxes(context,tracks,width,height,{sample=false}={}){
  context.save();context.lineWidth=Math.max(2,width/400);context.font=`${Math.max(12,width/48)}px system-ui`;
  for(const track of tracks){if(track.status==='lost'&&!sample)continue;const b=sample?track.lastObservedBbox:track.bbox;const color=track.status==='observed'?'#b7e57c':track.stale?'#c3b39c':'#f2ce76';
    context.strokeStyle=color;context.setLineDash(track.status==='observed'?[]:track.stale?[2,6]:[9,5]);context.strokeRect(b[0],b[1],b[2]-b[0],b[3]-b[1]);context.setLineDash([]);
    const text=`${track.id} · ${track.status}${track.stale?' / stale':''}`,metrics=context.measureText(text),top=Math.max(0,b[1]-25);
    context.fillStyle='#10271fe8';context.fillRect(b[0],top,Math.min(metrics.width+12,width-b[0]),24);context.fillStyle=color;context.fillText(text,b[0]+5,top+17);
  }context.restore();
}
function syntheticScene(t){ctx.fillStyle='#273c2b';ctx.fillRect(0,0,640,480);for(let i=0;i<14;i++){ctx.fillStyle=i%2?'#526443':'#465738';ctx.beginPath();ctx.ellipse((i*89)%640,180+(i*47)%300,35,10,i*.7,0,7);ctx.fill();}const f=syntheticFrame(t);if(f.detections.length){const [x,y,x2,y2]=f.detections[0].bbox;ctx.fillStyle='#cfbd88';ctx.fillRect(x+(x2-x)*.43,y+(y2-y)*.25,(x2-x)*.16,(y2-y)*.75);ctx.fillStyle='#967d45';ctx.beginPath();ctx.ellipse((x+x2)/2,y+(y2-y)*.25,(x2-x)/2,(y2-y)*.25,0,0,7);ctx.fill();}ctx.fillStyle='#cad4bd';ctx.font='14px system-ui';ctx.fillText('SYNTHETIC ILLUSTRATION · no real detection',18,453);}
async function analyzeFrame(){
  if(!ready||!['file','camera'].includes(mode)||gate.pending||requestCount>=60)return;
  const currentEpoch=epoch,capturedAt=time(),currentMode=mode,resume=currentMode==='file'&&$('hold-video').checked&&!video.paused;
  const width=video.videoWidth,height=video.videoHeight,scale=Math.min(1,960/width,720/height);capture.width=Math.max(1,Math.round(width*scale));capture.height=Math.max(1,Math.round(height*scale));
  capture.getContext('2d').drawImage(video,0,0,capture.width,capture.height);const payload={imageBase64:capture.toDataURL('image/jpeg',.9).split(',')[1],mimeType:'image/jpeg',boxThreshold:.3};
  const sample=$('sample-canvas');sample.width=capture.width;sample.height=capture.height;sample.getContext('2d').drawImage(capture,0,0);
  $('sample-caption').textContent=`Captured at ${fmt(capturedAt)} s · ${width} × ${height} source pixels → ${capture.width} × ${capture.height} analysis pixels. Waiting for inference.`;
  if(resume)video.pause();lastRequested=capturedAt;requestCount++;$('sample-status').textContent=`Analyzing sample ${requestCount} / 60. One request active; later frames are not queued.`;
  try{await gate.run(async()=>{controls();return request('/api/tracking-demo/detect',{body:payload});},result=>{
    if(currentEpoch!==epoch)return;if(result.image?.width!==capture.width||result.image?.height!==capture.height||!Array.isArray(result.detections))throw new Error('The sample result dimensions do not match the captured frame.');
    const detections=result.detections.map(item=>({...item,bbox:item.bbox.map((value,index)=>value*(index%2?height/capture.height:width/capture.width))}));
    tracker.update({detections,timestampSeconds:capturedAt,width,height});inferenceSeconds=result.requestSeconds??result.processingSeconds;
    const sampleTracks=tracker.snapshot(capturedAt).filter(track=>track.status==='observed').map(track=>({...track,bbox:track.bbox.map((value,index)=>value*(index%2?capture.height/height:capture.width/width)),lastObservedBbox:track.lastObservedBbox.map((value,index)=>value*(index%2?capture.height/height:capture.width/width))}));
    drawBoxes(sample.getContext('2d'),sampleTracks,sample.width,sample.height,{sample:true});
    $('sample-caption').textContent=`Sample ${fmt(capturedAt)} s · ${result.detections.length} unverified candidates · ${fmt(inferenceSeconds)} s request time. Boxes refer to this captured frame, not later video.`;
    $('sample-status').textContent=`Sample ${requestCount} / 60 complete · ${fmt(inferenceSeconds)} s latency. ${currentMode==='camera'?'The camera kept moving; inspect evidence age before using a track.':'Pause-during-analysis can align the next result with its frame.'}`;
    $('connection-state').textContent='Field station connected';
  });}catch(error){if(currentEpoch===epoch){automatic=false;notice(error.message);$('sample-status').textContent='Analysis stopped. Your video remains available; no background request queue was created.';}}
  finally{if(currentEpoch===epoch){if(resume)await video.play().catch(()=>{});if(requestCount>=60){automatic=false;$('sample-status').textContent+=' Session limit reached; choose a new source to start another bounded run.';}controls();}}
}
$('sample-once').onclick=analyzeFrame;
$('sample-auto').onclick=()=>{automatic=!automatic;if(!automatic){gate.reset();$('sample-status').textContent='Sampling stopped. Any outstanding response will be discarded.';}else{lastRequested=-Infinity;analyzeFrame();}controls();};

function renderTracks(){
  const tracks=ready?tracker.snapshot(time()):[];lastTracks=tracks;$('track-count').textContent=String(tracks.length);
  for(const status of ['observed','predicted','lost'])$(status+'-count').textContent=String(tracks.filter(track=>track.status===status).length);
  $('tracks').replaceChildren();
  for(const track of tracks){const card=node('article',undefined,'track-card '+track.status),heading=node('div',undefined,'track-heading');heading.append(node('strong',track.id),node('span',track.status+(track.stale?' · stale':''),'track-state'));card.append(heading);
    card.append(node('p',`${Math.round(track.observedPixelWidth)} × ${Math.round(track.observedPixelHeight)} source px · ${Math.round(track.confidence*100)}% detector score`),node('p',`${track.observations} observations · last seen ${fmt(track.lastSeenSeconds)} s · age ${fmt(track.ageSeconds)} s`));
    const button=node('button',proposals.some(item=>item.trackId===track.id)?'Closer look proposed':'Propose a closer photo','quiet');button.type='button';button.dataset.trackId=track.id;
    button.disabled=track.stale||track.status==='lost'||track.observations<2||track.confidence<.3||proposals.some(item=>item.trackId===track.id);card.append(button);$('tracks').append(card);
  }
  if(!tracks.length)$('tracks').append(node('p',mode==='replay'?'Play until the next recorded detector sample.':'Analyze a frame to create candidate tracks.','empty'));
}
$('tracks').onclick=event=>{const button=event.target.closest('button[data-track-id]');if(!button||button.disabled)return;try{const track=tracker.snapshot(time()).find(item=>item.id===button.dataset.trackId),proposal=closeLookProposal(track,{sourceId,timestampSeconds:time()});proposals.push(proposal);
  const card=node('div',undefined,'proposal');card.append(node('strong',proposal.trackId+' · closer photo'),node('p','Proposal only · no depth / pose / GPS · human review required'));$('proposals').append(card);$('proposal-count').textContent=String(proposals.length);renderTracks();controls();}catch(error){notice(error.message);}};
$('export-proposals').onclick=()=>{const data={schema:'mushroom-vision.tracking-proposals.v1',exportedAt:new Date().toISOString(),sourceId,mode,synthetic:mode==='synthetic',actuationEnabled:false,proposals};const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),link=node('a');link.href=url;link.download='mushroom-vision-close-look-proposals.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
function tick(now){
  const dt=Math.min(.2,(now-lastTick)/1000);lastTick=now;
  if(ready){
    if(mode==='synthetic'){if(syntheticPlaying){syntheticTime=Math.min(20,syntheticTime+dt);if(syntheticTime>=20)syntheticPlaying=false;}syntheticScene(syntheticTime);const sampleTime=Math.floor(syntheticTime*2)/2;if(sampleTime!==lastSyntheticSample){tracker.update(syntheticFrame(sampleTime));lastSyntheticSample=sampleTime;}}
    else if(video.readyState>=2){ctx.drawImage(video,0,0,sourceWidth,sourceHeight);if(mode==='replay'&&demo){while(replayIndex<demo.frames.length&&demo.frames[replayIndex].timestampSeconds<=time()){const frame=demo.frames[replayIndex++];tracker.update({detections:frame.detections,timestampSeconds:frame.timestampSeconds,width:frame.width,height:frame.height});}}}
    const tracks=tracker.snapshot(time());drawBoxes(ctx,tracks,sourceWidth,sourceHeight);
    if(automatic&&!gate.pending&&requestCount<60&&time()-lastRequested>=2)analyzeFrame();
    $('seek').max=String(duration()||20);$('seek').value=String(time());$('time').textContent=mode==='camera'?`${fmt(time())} s live`:`${fmt(time())} / ${fmt(duration())} s`;
    if(now-lastUi>500){renderTracks();controls();lastUi=now;}
  }
  requestAnimationFrame(tick);
}
$('connection-open').onclick=()=>{$('station-url').value=station;$('station-token').value=token;$('connection-dialog').showModal();};$('connection-close').onclick=()=>$('connection-dialog').close();
$('connection-form').onsubmit=event=>{event.preventDefault();try{const raw=$('station-url').value.trim();let next='';if(raw){const url=new URL(raw);if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.pathname!=='/'||url.search||url.hash)throw new Error('Use the station origin without a path, query or credentials.');next=url.origin;}station=next;token=$('station-token').value;localStorage.setItem('mycoscout-station',station);$('connection-dialog').close();loadReplay();}catch(error){notice(error.message);}};
window.addEventListener('pagehide',()=>{automatic=false;epoch++;gate.reset();video.pause();if(stream)for(const track of stream.getTracks())track.stop();if(videoURL)URL.revokeObjectURL(videoURL);});
requestAnimationFrame(tick);loadReplay();
