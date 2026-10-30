const $=id=>document.getElementById(id);
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const num=(n,fallback='—')=>finite(n)?String(n):fallback;
const percent=n=>finite(n)?`${(n*100).toFixed(1)}%`:'Not measured';
const readable=s=>String(s??'').replaceAll('_',' ').toLowerCase().replace(/^./,s=>s.toUpperCase());
const el=(tag,text,className)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(className)n.className=className;return n;};
const svgEl=(tag,attributes={},text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [key,value]of Object.entries(attributes))n.setAttribute(key,String(value));if(text!==undefined)n.textContent=String(text);return n;};
let flight=null;

function cards(target,items){target.replaceChildren(...items.map(([value,label])=>{const card=el('div',undefined,'metric');card.append(el('b',value),el('span',label));return card;}));}
function notes(target,items){target.replaceChildren(...items.filter(s=>typeof s==='string').map(s=>el('li',s)));}
function renderVision(report){
  const summary=report.summary??{},runs=Array.isArray(report.runs)?report.runs:[];
  $('vision-status').textContent=summary.status||'Report available';
  cards($('vision-metrics'),[[num(summary.imageCount),'source photos'],[num(summary.sourceGroupCount),'source groups'],[num(summary.fullyAnnotatedCount),'fully labeled photos']]);
  const content=$('vision-runs');content.replaceChildren();
  if(!runs.length)content.append(el('p','Photo collection or annotation is in progress. No detector accuracy is claimed.', 'empty'));
  for(const run of runs){
    const row=el('div',undefined,'vision-run'),heading=el('div',undefined,'run-heading');
    heading.append(el('strong',`${run.model||'Vision experiment'}${run.stage?` · ${run.stage}`:''}`),el('span',run.synthetic?'Synthetic scale test':run.measured===true?'Measured experiment':'Evaluation pending'));row.append(heading);
    if(run.measured===true && run.metrics){const metrics=el('div',undefined,'metric-row');for(const [key,label]of [['precision','Precision'],['recall','Recall'],['f1','F1']]){const metric=el('span',label);metric.append(el('b',percent(run.metrics[key])));metrics.append(metric);}row.append(metrics);}
    else row.append(el('p',run.measured===true&&Array.isArray(run.sizeBins)?'Measured recall at each scale is shown below.':'No measured precision or recall for this run.','fine'));
    if(Array.isArray(run.notes)){const list=el('ul',undefined,'notes');notes(list,run.notes);row.append(list);}content.append(row);
  }
  const reportNotes=[...(Array.isArray(report.notes)?report.notes:[])];
  if(summary.verifierStatus)reportNotes.push(summary.verifierStatus);
  if(finite(summary.trainCount))reportNotes.push(`${summary.trainCount} training photos. Splits must keep derivatives of the same source photograph together.`);
  notes($('vision-notes'),reportNotes);
  const curveRun=[...runs].reverse().find(r=>r.measured===true&&Array.isArray(r.sizeBins)&&r.sizeBins.some(b=>finite(b.recall)));
  renderPhotoCurve(curveRun);
}
function renderPhotoCurve(run){
  const target=$('vision-chart');target.replaceChildren();target.hidden=!run;if(!run)return;
  const bins=run.sizeBins.filter(b=>finite(b.recall)&&b.recall>=0&&b.recall<=1);
  if(!bins.length){target.hidden=true;return;}
  target.append(el('div','Recall as the scene shrinks','chart-label'));
  const svg=svgEl('svg',{viewBox:'0 0 500 190',role:'img','aria-label':'Recall by synthetic scene scale. These are photo experiments, not validated flight distances.'});
  for(const y of [0,0.5,1]){const py=145-y*120;svg.append(svgEl('line',{x1:42,x2:480,y1:py,y2:py,stroke:'#35483c'}),svgEl('text',{x:32,y:py+4,fill:'#afbcab','font-size':10,'text-anchor':'end'},`${y*100}%`));}
  const coords=bins.map((b,i)=>({x:bins.length===1?260:55+i*410/(bins.length-1),y:145-b.recall*120,b}));
  svg.append(svgEl('polyline',{points:coords.map(p=>`${p.x},${p.y}`).join(' '),fill:'none',stroke:'#efb65b','stroke-width':2}));
  for(const p of coords){svg.append(svgEl('circle',{cx:p.x,cy:p.y,r:4,fill:'#efb65b'}));const label=String(p.b.label??(finite(p.b.targetPixels)?`${p.b.targetPixels}px`:'bin')).slice(0,20);svg.append(svgEl('text',{x:p.x,y:170,fill:'#afbcab','font-size':10,'text-anchor':'middle'},label));}
  target.append(svg,el('p',`${run.model||'Model'} · ${run.synthetic?'Resized or transformed photographs; not measured outdoor range.':'Use the saved experiment notes to interpret its evaluation conditions.'}`));
}
function renderFlight(report){
  flight=report;const summary=report.summary??{};
  cards($('flight-metrics'),[[`${num(summary.passed)}/${num(summary.scenarios)}`,'simulated cases passed'],[num(summary.captures),'synthetic capture events'],[num(summary.physicalFlights),'physical flights']]);
  const select=$('scenario-select'),scenarios=Array.isArray(report.scenarios)?report.scenarios:[];
  select.replaceChildren(...scenarios.map((s,i)=>{const option=el('option',s.title||s.id||`Scenario ${i+1}`);option.value=String(i);return option;}));select.disabled=!scenarios.length;
  select.onchange=()=>renderScenario(scenarios[Number(select.value)]);if(scenarios.length)renderScenario(scenarios[0]);
  renderGeometry(report);
}
function renderScenario(scenario){
  if(!scenario)return;
  const overview=$('scenario-summary');overview.replaceChildren(el('p',scenario.title||scenario.id,'scenario-title'));
  const meta=el('div',undefined,'scenario-meta');
  const result=el('b',scenario.passed===true?'Expected outcome passed':'Outcome needs review',scenario.passed===true?'':'failure');
  meta.append(result,el('span',`${num(scenario.captures)} captures`),el('span',finite(scenario.durationMs)?`${(scenario.durationMs/1000).toFixed(1)} seconds simulated`:''),el('span',`Final: ${readable(scenario.finalState)}`));overview.append(meta);
  const timeline=$('scenario-events');timeline.replaceChildren();
  for(const event of Array.isArray(scenario.events)?scenario.events:[]){
    const li=el('li',undefined,event.capture?'capture':'');li.append(el('time',finite(event.nowMs)?`${(event.nowMs/1000).toFixed(1)}s`:'—'),el('strong',readable(event.reason||event.state)));
    const details=[readable(event.state)];if(event.capture)details.push('Synthetic capture record');if(event.proposal?.type==='position')details.push(`${Number(event.proposal.speedMps).toFixed(3)} m/s proposed`);if(event.clearance?.reason)details.push(readable(event.clearance.reason));
    if(event.evidence?.distinctFrames!==undefined)details.push(`${event.evidence.distinctFrames} distinct frames`);
    li.append(el('p',details.join(' · ')));timeline.append(li);
  }
  renderRoute(scenario);
}
function renderRoute(scenario){
  const target=$('route-preview');target.replaceChildren();
  const route=(scenario.route??scenario.planner?.points??[]).filter(p=>finite(p.x)&&finite(p.y));
  if(route.length<2){const final=scenario.events?.at(-1);target.append(el('p',`This scenario tests ${readable(final?.reason||scenario.finalState).toLowerCase()}. No completed survey route is claimed.`));return;}
  const xs=route.map(p=>p.x),ys=route.map(p=>p.y),minX=Math.min(...xs)-0.75,maxX=Math.max(...xs)+0.75,minY=Math.min(...ys)-1,maxY=Math.max(...ys)+1;
  const scale=Math.min(400/(maxX-minX),190/(maxY-minY)),cx=(minX+maxX)/2,cy=(minY+maxY)/2;
  const project=p=>({x:230+(p.x-cx)*scale,y:125-(p.y-cy)*scale});
  const svg=svgEl('svg',{viewBox:'0 0 460 250',role:'img','aria-label':'Top projection of a simulated route. The planner checks three-dimensional clearance.'});
  for(let x=30;x<460;x+=40)svg.append(svgEl('line',{x1:x,x2:x,y1:15,y2:230,stroke:'#243d2d'}));
  for(let y=25;y<240;y+=40)svg.append(svgEl('line',{x1:20,x2:440,y1:y,y2:y,stroke:'#243d2d'}));
  if(scenario.id==='trunk-detour'){const a=project({x:1.5,y:-0.5}),b=project({x:2,y:0.5});svg.append(svgEl('rect',{x:a.x,y:b.y,width:b.x-a.x,height:a.y-b.y,fill:'#617c51',stroke:'#aabf82',rx:3}),svgEl('text',{x:(a.x+b.x)/2,y:b.y-9,fill:'#b9d4a7','font-size':10,'text-anchor':'middle'},'Trunk'));}
  const points=route.map(project);svg.append(svgEl('polyline',{points:points.map(p=>`${p.x},${p.y}`).join(' '),fill:'none',stroke:'#efb65b','stroke-width':2.5,'stroke-linejoin':'round'}));
  points.forEach((p,i)=>svg.append(svgEl('circle',{cx:p.x,cy:p.y,r:5,fill:'#14271e',stroke:'#efb65b','stroke-width':2}),svgEl('text',{x:p.x+9,y:p.y-8,fill:'#edf0dc','font-size':10},i===0?'Start':i===points.length-1?'Finish':`Look ${i}`)));
  svg.append(svgEl('text',{x:20,y:245,fill:'#afbcab','font-size':9},'Synthetic scene · metre-based local map · top projection'));target.append(svg);
}
function renderGeometry(report){
  const geometry=report.cameraGeometry??{},rows=Array.isArray(geometry.rows)?geometry.rows:[],target=$('geometry-bars');target.replaceChildren();
  if(rows.length&&finite(rows[0].targetDiameterM))$('geometry-description').textContent=`An illustrative ${Math.round(rows[0].targetDiameterM*100)} cm mushroom cap, viewed face-on. Image width falls as camera range increases; foliage and slanted views remove more detail.`;
  const max=Math.max(1,...rows.map(r=>finite(r.targetPixels)?r.targetPixels:0));
  for(const row of rows){if(!finite(row.targetPixels)||!finite(row.rangeM))continue;const line=el('div',undefined,'geometry-row'),track=el('div',undefined,'bar-track'),fill=el('div',undefined,'bar-fill');fill.style.width=`${Math.max(0,Math.min(100,row.targetPixels/max*100))}%`;track.append(fill);line.append(el('span',`${row.rangeM} m`),track,el('span',`${row.targetPixels.toFixed(1)} px`));target.append(line);}
  const comparisons=Array.isArray(report.exposureComparison)?report.exposureComparison:[];
  if(comparisons.length>=2&&comparisons.every(c=>finite(c.exposureMs)&&finite(c.blurLimitedSpeedMps))){const[a,b]=comparisons;$('exposure-description').textContent=`At 1 m, the example blur budget allows ${a.blurLimitedSpeedMps.toFixed(3)} m/s at ${a.exposureMs} ms exposure. At ${b.exposureMs} ms, it allows only ${b.blurLimitedSpeedMps.toFixed(3)} m/s.`;}
  $('calibration-note').textContent=`Calibration: ${readable(geometry.calibrationStatus||'not measured')}. These estimates do not claim mushroom detection performance.`;
}
async function getReport(url){const response=await fetch(url,{cache:'no-store'});if(!response.ok)throw new Error(`Report unavailable (${response.status})`);return response.json();}
async function load(){
  $('refresh').disabled=true;
  const results=await Promise.allSettled([getReport('./research-report.json'),getReport('./flight-report.json')]);
  const dates=[];let available=0;
  if(results[0].status==='fulfilled'){renderVision(results[0].value);available++;if(results[0].value.generatedAt)dates.push(`Photos: ${String(results[0].value.generatedAt).slice(0,10)}`);}else{$('vision-status').textContent='Report pending';cards($('vision-metrics'),[['—','source photos'],['—','source groups'],['—','fully labeled photos']]);}
  if(results[1].status==='fulfilled'){renderFlight(results[1].value);available++;if(results[1].value.generatedAt)dates.push(`Simulation: ${String(results[1].value.generatedAt).slice(0,10)}`);}else{cards($('flight-metrics'),[['—','simulated cases passed'],['—','synthetic capture events'],['—','physical flights']]);}
  $('load-status').textContent=available===2?'Saved reports loaded. Photo measurements and simulated flight outcomes are labeled separately.':available===1?'One saved report is available. The other is still being prepared.':'Reports are not available yet. Refresh when the experiments finish.';
  $('updated').textContent=dates.join(' · ')||'No saved reports loaded.';$('refresh').disabled=false;
}
$('refresh').addEventListener('click',load);load();

if(['localhost','127.0.0.1','[::1]'].includes(location.hostname) && $('private-review-link')) $('private-review-link').hidden=false;
