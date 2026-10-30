const byId=id=>document.getElementById(id);
const element=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=String(text);if(className)node.className=className;return node;};
const number=value=>Number.isFinite(value)?value.toLocaleString(undefined,{maximumFractionDigits:3}):'—';
const pixels=value=>Number.isFinite(value)?value.toLocaleString(undefined,{maximumSignificantDigits:6}):'—';
const pixelRange=value=>value.min===value.max?pixels(value.min):`${pixels(value.min)}–${pixels(value.max)}`;
const percent=value=>Number.isFinite(value)?(value*100).toFixed(1)+'%':'—';
const methods=['raw','color','tiles'];
let report=null,loading=false;

function renderRows(){
  const rows=byId('video-distance-rows'),negatives=byId('video-distance-negative-rows');rows.replaceChildren();negatives.replaceChildren();
  const filter=byId('video-distance-filter').value;
  const selected=(report?.rows??[]).filter(row=>filter==='all'||filter==='controls'&&(row.native||row.targetPixels===32||row.targetPixels===64)||filter==='small'&&!row.native&&row.targetPixels<32);
  for(const row of selected){
    const tr=element('tr',undefined,row.native?'native-condition':''),condition=element('td',row.label);
    const actual=row.realizedTargetWidthPixels;
    condition.append(element('small',actual?`Actual target widths: ${pixelRange(actual)} px`:'Actual target widths: unavailable / pending'));
    condition.append(element('small',row.native?'Original video frame':'Synthetic whole-frame resize; requested and actual widths can differ'));
    tr.append(condition);
    for(const method of methods){
      const metric=row[method],cell=element('td');
      if(!metric)cell.textContent=report.status==='completed'?'Not measured in this run':'Pending / unavailable';
      else if(!metric.targets)cell.textContent='No labeled targets';
      else{
        cell.append(element('strong',`${number(metric.matched)} / ${number(metric.targets)} · ${percent(metric.recall)}`));
        cell.append(element('small',`${number(metric.frames)} positive frames · ${number(metric.falsePositives)} unmatched detections`));
      }
      tr.append(cell);
    }
    rows.append(tr);
    if(methods.some(method=>row[method]?.negativeFrames>0)){
      const negative=element('tr');negative.append(element('td',row.native?'Native frames':`Requested ${pixels(row.targetPixels)} px scene scale`));
      for(const method of methods){const metric=row[method],cell=element('td');
        if(!metric||!metric.negativeFrames)cell.textContent='Not measured';
        else{
          cell.append(element('strong',`${number(metric.negativeFalsePositives)} false detections`));
          cell.append(element('small',`${number(metric.negativeFramesWithDetections)} / ${number(metric.negativeFrames)} negative frames triggered`));
        }
        negative.append(cell);
      }
      negatives.append(negative);
    }
  }
  if(!selected.length){const tr=element('tr'),cell=element('td','No completed conditions match this selection yet.');cell.colSpan=4;tr.append(cell);rows.append(tr);}
  if(!negatives.children.length){const tr=element('tr'),cell=element('td','No completed negative-frame control is available for these conditions.');cell.colSpan=4;tr.append(cell);negatives.append(tr);}
}

function renderClassification(value){
  let section=byId('video-distance-classification');
  if(!section){section=element('details');section.id='video-distance-classification';byId('video-distance-results').append(section);}
  section.replaceChildren();section.hidden=!value.classificationRows?.length;
  if(section.hidden)return;
  section.append(element('summary','Did the expanded training help category guesses?'));
  section.append(element('p','Known-box crop test: the location is supplied by a human annotation. The comparison category is an assistant’s provisional “morel group” opinion, not a human species identification. This does not measure finding a mushroom.', 'fine'));
  const wrap=element('div',undefined,'table-scroll'),table=element('table',undefined,'comparison-table'),head=element('thead'),heading=element('tr'),body=element('tbody');
  for(const label of ['Condition','Earlier category head','Expanded category head','Live crop guard'])heading.append(element('th',label));
  head.append(heading);table.append(head,body);wrap.append(table);section.append(wrap);
  for(const row of value.classificationRows){
    const tr=element('tr');tr.append(element('td',row.label));
    for(const name of ['baseline','expanded']){const metric=row[name],cell=element('td');
      cell.append(element('strong',`${metric.top1Agreement} / ${row.targets} top guesses agree`));
      cell.append(element('small',`${metric.retainedAgreement} / ${metric.retained} retained guesses agree · ${metric.abstained} abstained`));tr.append(cell);
    }
    tr.append(element('td',`${row.productionEligible} / ${row.targets} eligible at ≥32 native pixels per side`));body.append(tr);
  }
  section.append(element('p','Tiny crops below the live pixel guard are diagnostic only, even when the model gives a high score. Neither head was retrained or tuned on these test frames. Frames and resized variants are correlated.', 'fine'));
}

function renderCropContext(value){
  let section=byId('video-distance-context');
  if(!section){section=element('details');section.id='video-distance-context';byId('video-distance-results').append(section);}
  section.replaceChildren();section.hidden=!value.cropContextRows?.length;
  if(section.hidden)return;
  section.append(element('summary','Crop framing experiment: preserve the whole mushroom'));
  section.append(element('p','A tall or wide box can lose important parts when the category model takes its center crop. This exploratory comparison keeps the same pixels and compares tight cropping, neutral square padding, and a square view with 25% surrounding context. Locations are supplied, and the morel-group reference remains an assistant opinion.', 'fine'));
  const wrap=element('div',undefined,'table-scroll'),table=element('table',undefined,'comparison-table'),head=element('thead'),heading=element('tr'),body=element('tbody');
  for(const label of ['Condition','Tight box','Square padding','Square context'])heading.append(element('th',label));
  head.append(heading);table.append(head,body);wrap.append(table);section.append(wrap);
  for(const row of value.cropContextRows){
    const tr=element('tr'),label=element('td',row.label);label.append(element('small',`${row.productionEligible} / ${row.targets} pass the native-pixel guard`));tr.append(label);
    for(const policy of ['tight','fitted','context']){const metric=row.policies[policy].expanded,earlier=row.policies[policy].baseline,cell=element('td');
      cell.append(element('strong',`Expanded: ${metric.top1Agreement} / ${row.targets} agree`));
      cell.append(element('small',`${metric.retainedAgreement} / ${metric.retained} retained guesses agree`));
      cell.append(element('small',`Earlier head: ${earlier.top1Agreement} / ${row.targets} top guesses agree`));tr.append(cell);
    }
    body.append(tr);
  }
  section.append(element('p','This is an exploratory crop diagnostic on three correlated source views. It does not improve the detector, establish species accuracy or change the live crop policy. Tiny views can produce confident wrong guesses; no pixels are recovered by padding.', 'fine'));
}

function renderAiCandidate(value,corpus=false){
  const id=corpus?'video-distance-corpus-candidate':'video-distance-ai-candidate';
  let section=byId(id);
  if(!section){section=element('details');section.id=id;byId('video-distance-results').append(section);}
  section.replaceChildren();const candidate=corpus?value.corpusCandidateDistance:value.aiCandidateDistance;section.hidden=!candidate;
  if(!candidate)return;
  section.append(element('summary',corpus?'New experiment: training after the corpus review':'Separate experiment: one pass of AI-reviewed training'));
  section.append(element('p',`This inactive Faster R-CNN candidate learned from ${candidate.trainingFrames} AI-reviewed frames, ${candidate.trainingBoxes} boxes and ${candidate.trainingSourceGroups} source groups. Its training labels are provisional AI opinions. Evaluation uses the frozen human presence boxes below, at score ${candidate.scoreThreshold} and overlap ${candidate.iouThreshold}.`, 'fine'));
  if(corpus)section.append(element('p',`${candidate.epochs} training passes; ${candidate.optimizerSteps} updates. Selected pass ${candidate.selectedEpoch} using AI validation only. Explicit nonfungal crops: ${Object.entries(candidate.hardNegativeCrops).map(([kind,count])=>`${count} ${kind}`).join(', ')||'none'}. Thumbnail screening and uncertain originals are excluded from fitting.`, 'fine'));
  const wrap=element('div',undefined,'table-scroll'),table=element('table',undefined,'comparison-table'),head=element('thead'),heading=element('tr'),body=element('tbody');
  for(const label of ['Condition','Human regions matched','Unmatched detections','Negative control'])heading.append(element('th',label));
  head.append(heading);table.append(head,body);wrap.append(table);section.append(wrap);
  for(const row of candidate.rows){const tr=element('tr');
    tr.append(element('td',row.label),element('td',`${row.matched} / ${row.targets}`),element('td',row.unmatchedDetections),
      element('td',row.negativeFrames?`${row.negativeFalsePositives} false detections / ${row.negativeFrames} frames`:'Not measured'));body.append(tr);
  }
  section.append(element('p',corpus?'These are the same previously inspected regression frames, not new independent validation. Each row contains three correlated targets; resized pixels are not measured meters. Compare missed targets and false alerts together. Training opinions still need human review, and this candidate remains inactive.':'This is a separate detector and threshold from the GroundingDINO table. One pass on a tiny AI-labeled sample does not establish improvement. No fitting, threshold choice or activation follows from these test results.', 'fine'));
}

function render(value){
  report=value;
  const available=value.availability==='available',sample=value.sample??{},progress=value.progress??{};
  byId('video-distance-status').textContent=available?(value.status==='running'?'Video test running':value.status==='completed'?'Video test complete':'Partial video results'):value.availability==='missing'?'Experiment pending':'Report unavailable';
  const unit=progress.unit==='model-forwards'?'model passes':progress.unit==='method-results'?'method results':'experiment steps';
  const progressText=Number.isFinite(progress.completed)&&Number.isFinite(progress.total)?` ${number(progress.completed)} / ${number(progress.total)} ${unit} completed.`:'';
  const authority=sample.labelAuthority==='human'?'human-reviewed labels':sample.labelAuthority==='assistant'?'assistant visual labels, awaiting human confirmation':'the label authority stated in the experiment';
  byId('video-distance-summary').textContent=available?
    `${value.runVersion==='video-distance-v3'?'Fresh active-detector check · ':value.runVersion==='video-distance-v2'?'Fresh retest · ':''}${number(sample.positiveFrames)} positive and ${number(sample.negativeFrames)} explicitly negative video frames from ${number(sample.sourceVideos)} ${sample.sourceVideos===1?'video':'videos'}. These results use ${authority}.${progressText}`:
    value.availability==='missing'?'The new video pixel experiment has not saved a report yet. Refresh reports to check its progress.':'The saved video test is missing required evidence or is unreadable. Other research reports remain available.';
  const metrics=byId('video-distance-metrics');metrics.replaceChildren();
  if(available)for(const [count,label] of [[sample.targets,'marked targets across source frames'],[sample.sourceGroups,'source groups · frames remain correlated'],[progress.completed,`completed ${unit}`]]){
    const card=element('div',undefined,'metric');card.append(element('b',number(count)),element('span',label));metrics.append(card);
  }
  const nativeRows=byId('video-distance-native-rows');nativeRows.replaceChildren();
  for(const [evidence,label,partial] of [[value.nativeBaseline,'Complete-frame comparison',false],[value.nativeSmallPositiveOnly,'Small targets · partial human labels',true]])if(evidence){
    const tr=element('tr'),name=element('td',label);
    if(evidence.targetWidthPixels)name.append(element('small',`${pixelRange(evidence.targetWidthPixels)} px labeled target widths`));
    name.append(element('small',`Score ${evidence.scoreThreshold} · overlap ${evidence.iouThreshold} · ${number(evidence.sourceGroups)} source groups`));
    name.append(element('small',evidence.reused?'Previous cached inference':'Fresh inference in this run'));
    tr.append(name,element('td',number(evidence.frames)),element('td',`${number(evidence.matched)} / ${number(evidence.targets)} · ${percent(evidence.recall)}`),element('td',partial?'Not evaluated':number(evidence.falsePositives)));nativeRows.append(tr);
  }
  byId('video-distance-native').hidden=!nativeRows.children.length;
  byId('video-distance-native-note').textContent='These original-RGB results are separate samples, not a pooled detection rate. Results count human annotation regions, not distinct mushrooms. Small-target frames have partial labels: only recall on the marked targets is measured; other detections cannot be classified as false positives. Repeated frames may show the same mushroom, and this subset does not establish morel detection performance. '+(value.nativeBaseline?.labelCautions??[]).join(' ');
  byId('video-distance-results').hidden=!available;
  renderRows();
  renderClassification(value);
  renderCropContext(value);
  renderAiCandidate(value);
  renderAiCandidate(value,true);
  byId('video-distance-method').textContent=available?value.methodNote:'No comparable measurements are displayed.';
  byId('video-distance-negatives-note').textContent=value.negativeNote||'False detections are counted on explicit negative controls, separately from unmatched boxes in positive frames. Missing results are never shown as zero.';
  byId('video-distance-limitations').replaceChildren(...(value.limitations??[]).map(text=>element('li',text)));
  byId('video-distance-updated').textContent=(value.generatedAt&&Number.isFinite(Date.parse(value.generatedAt))?'Saved '+new Date(value.generatedAt).toLocaleString()+'. ':'')+'Read-only experiment; no training, model activation or aircraft commands.';
}

async function refreshVideoDistance(){
  if(loading)return;loading=true;
  try{
    const response=await fetch('./api/research/video-distance',{cache:'no-store',signal:AbortSignal.timeout(15000)});
    if(!response.ok)throw new Error('Video pixel report unavailable');
    const value=await response.json();
    if(value.schema!=='mycoscout.video-distance-summary.v1'||!Array.isArray(value.rows))throw new Error('Invalid video pixel report');
    render(value);
  }catch{
    render({availability:'unavailable',rows:[],limitations:[]});
    byId('video-distance-updated').textContent='Use Refresh reports to retry. A previous measurement is not presented as a current result.';
  }finally{loading=false;}
}

byId('video-distance-filter').addEventListener('change',renderRows);
byId('refresh')?.addEventListener('click',refreshVideoDistance);
await refreshVideoDistance();
setInterval(()=>{if(!document.hidden)refreshVideoDistance();},15000);
