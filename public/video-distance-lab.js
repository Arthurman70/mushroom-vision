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
      if(!metric)cell.textContent='Pending / unavailable';
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

function render(value){
  report=value;
  const available=value.availability==='available',sample=value.sample??{},progress=value.progress??{};
  byId('video-distance-status').textContent=available?(value.status==='running'?'Video test running':value.status==='completed'?'Video test complete':'Partial video results'):value.availability==='missing'?'Experiment pending':'Report unavailable';
  const unit=progress.unit==='model-forwards'?'model passes':progress.unit==='method-results'?'method results':'experiment steps';
  const progressText=Number.isFinite(progress.completed)&&Number.isFinite(progress.total)?` ${number(progress.completed)} / ${number(progress.total)} ${unit} completed.`:'';
  const authority=sample.labelAuthority==='human'?'human-reviewed labels':sample.labelAuthority==='assistant'?'assistant visual labels, awaiting human confirmation':'the label authority stated in the experiment';
  byId('video-distance-summary').textContent=available?
    `${number(sample.positiveFrames)} positive and ${number(sample.negativeFrames)} explicitly negative video frames from ${number(sample.sourceVideos)} ${sample.sourceVideos===1?'video':'videos'}. These results use ${authority}.${progressText}`:
    value.availability==='missing'?'The new video pixel experiment has not saved a report yet. Refresh reports to check its progress.':'The saved video test is missing required evidence or is unreadable. Other research reports remain available.';
  const metrics=byId('video-distance-metrics');metrics.replaceChildren();
  if(available)for(const [count,label] of [[sample.targets,'marked targets across source frames'],[sample.sourceGroups,'source groups · frames remain correlated'],[progress.completed,`completed ${unit}`]]){
    const card=element('div',undefined,'metric');card.append(element('b',number(count)),element('span',label));metrics.append(card);
  }
  const nativeRows=byId('video-distance-native-rows');nativeRows.replaceChildren();
  for(const [evidence,label,partial] of [[value.nativeBaseline,'Cached complete-frame comparison',false],[value.nativeSmallPositiveOnly,'Small targets · partial human labels',true]])if(evidence){
    const tr=element('tr'),name=element('td',label);
    if(evidence.targetWidthPixels)name.append(element('small',`${pixelRange(evidence.targetWidthPixels)} px labeled target widths`));
    name.append(element('small',`Score ${evidence.scoreThreshold} · overlap ${evidence.iouThreshold} · ${number(evidence.sourceGroups)} source groups`));
    tr.append(name,element('td',number(evidence.frames)),element('td',`${number(evidence.matched)} / ${number(evidence.targets)} · ${percent(evidence.recall)}`),element('td',partial?'Not evaluated':number(evidence.falsePositives)));nativeRows.append(tr);
  }
  byId('video-distance-native').hidden=!nativeRows.children.length;
  byId('video-distance-native-note').textContent='These original-RGB results are separate samples, not a pooled detection rate. The cached baseline counts human annotation regions, not distinct mushrooms. Small-target frames have partial labels: only recall on the marked targets is measured; other detections cannot be classified as false positives. Repeated frames may show the same mushroom, and this subset does not establish morel detection performance. '+(value.nativeBaseline?.labelCautions??[]).join(' ');
  byId('video-distance-results').hidden=!available;
  renderRows();
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
