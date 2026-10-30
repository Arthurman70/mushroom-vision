const byId=id=>document.getElementById(id);
const element=(tag,text,className)=>{const node=document.createElement(tag);node.textContent=text;if(className)node.className=className;return node;};
const number=value=>Number.isFinite(value)?value.toLocaleString():'—';
const percent=value=>Number.isFinite(value)?(value*100).toFixed(1)+'%':'—';
let loading=false;

async function refreshVideoLearning() {
  if(loading)return;
  loading=true;
  try {
    const response=await fetch('./api/research/video-learning',{cache:'no-store'});
    if(!response.ok)throw new Error('Saved video-learning summary is unavailable.');
    const report=await response.json();
    if(report.schema!=='mycoscout.video-learning-summary.v1')throw new Error('Saved video-learning summary is unavailable.');
    const pass=report.pass,training=report.training,rows=report.comparison.rows;
    byId('video-learning-status').textContent=pass.status==='completed'?'Video pass complete':pass.status==='running'||pass.status==='in-progress'?'Video pass running':'Report pending';
    byId('video-learning-summary').textContent=`${number(pass.completedFrames)} frames have saved AI results; ${number(pass.failedFrames)} failed. The pass covers ${number(pass.totalFrames)} sampled frames from ${number(pass.totalVideos)} videos. Proposals are guesses awaiting review.`;
    const metrics=byId('video-learning-metrics');metrics.replaceChildren();
    for(const [value,label] of [[number(pass.processedFrames),'of '+number(pass.totalFrames)+' sampled frames processed'],[number(training?.frames),'assistant-audited pilot frames'],[number(rows[0]?.frames),'human comparison frames']]) {
      const metric=element('div','','metric');metric.append(element('b',value),element('span',label));metrics.append(metric);
    }
    const body=byId('video-learning-rows');body.replaceChildren();
    for(const row of rows) {
      const tr=document.createElement('tr');
      for(const value of [row.label,number(row.matched),number(row.missed),number(row.falsePositives)])tr.append(element('td',value));
      body.append(tr);
    }
    byId('video-learning-table').hidden=rows.length===0;
    byId('video-learning-comparison').textContent=rows.length ?
      `Same ${number(rows[0].frames)} complete human-reviewed frames, ${number(rows[0].targets)} human annotation regions and ${number(rows[0].sourceGroups)} source group. Detection score threshold ${rows[0].scoreThreshold}; box-overlap threshold ${rows[0].iouThreshold}. These count labeled regions, not distinct mushrooms. Correlated frames are not independent trials. ${(report.comparison.labelCautions??[]).join(' ')}` :
      'A comparison is not available yet. Reports with different frames, image versions or thresholds are not combined.';
    const inactive=report.pilot?.activationEligible===false&&report.pilot?.automaticActivation===false&&report.verifier?.activationEligible===false&&report.verifier?.automaticActivation===false;
    const tradeoff=rows.slice(1).some(row=>row.falsePositives<rows[0].falsePositives&&row.missed>rows[0].missed);
    byId('video-learning-result').textContent=(inactive?'Both saved candidates remain inactive. ':'No model is activated from this report. ')+
      (tradeoff?'Fewer false positives came with more missed mushrooms in this small comparison. This does not demonstrate improved detection.':'This small comparison does not establish better field performance.');
    const scale=report.comparison.evidence;
    byId('video-learning-scale').textContent=scale.targetsBelow64===0 ?
      'No human-evaluated targets in this comparison are below 64 pixels wide. There is no low-pixel detection evidence here, and no measured flight distance.' :
      scale.targetsBelow64===null?'Low-pixel coverage is unavailable. These saved results do not establish a flying distance.':
      `${number(scale.targetsBelow64)} comparison targets are below 64 pixels wide. This selected sample does not establish a flying distance.`;
    const details=byId('video-learning-details');details.replaceChildren();
    details.append(element('li',`${number(report.audit.reviewedFrames)} distinct frames in the assistant audit; ${number(report.audit.incompleteOrUncertainFrames)} incomplete or uncertain. Eligible pilot: ${number(training?.positiveFrames)} positive and ${number(training?.negativeFrames)} explicitly negative frames from ${number(training?.sourceGroups)} source groups. These are assistant weak labels, not human ground truth.`));
    details.append(element('li',`Detector pilot: ${number(report.pilot?.epochs)} epoch and ${number(report.pilot?.optimizerSteps)} training steps. Crop filter: ${number(report.verifier?.positiveCrops)} positive and ${number(report.verifier?.negativeCrops)} negative crops. Neither result is a species identification.`));
    if(report.groups)details.append(element('li',`Group-classifier refit: ${number(report.groups.photos)} photos, ${number(report.groups.categories)} categories. On ${number(report.groups.testPhotos)} held-out photos: ${percent(report.groups.top1Accuracy)} first-category accuracy, ${percent(report.groups.top3Accuracy)} top-three accuracy. These are still-photo classification results, not gains measured on video.`));
    const unavailable=Object.values(report.availability).filter(state=>state!=='available').length;
    if(unavailable)details.append(element('li',`${unavailable} saved report(s) are missing, invalid or not comparable; unavailable values remain blank.`));
    byId('video-learning-updated').textContent='Read-only saved results · refreshed '+new Date(report.generatedAt).toLocaleTimeString()+'.';
  }catch(error){
    byId('video-learning-status').textContent='Report unavailable';
    byId('video-learning-summary').textContent='The local saved-results summary could not be loaded. Other research panels remain available.';
    byId('video-learning-updated').textContent='Use Refresh reports to retry.';
  }finally{loading=false;}
}

byId('refresh')?.addEventListener('click',refreshVideoLearning);
await refreshVideoLearning();
setInterval(()=>{if(!document.hidden)refreshVideoLearning();},15000);
