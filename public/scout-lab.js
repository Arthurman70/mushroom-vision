const $=id=>document.getElementById(id);
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const pct=n=>finite(n)?`${(n*100).toFixed(1)}%`:'—';
const txt=s=>String(s??'').replaceAll('_',' ');
const el=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=String(text);if(cls)n.className=cls;return n;};
const svg=(tag,attrs={},text)=>{const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(attrs))n.setAttribute(k,String(v));if(text!==undefined)n.textContent=String(text);return n;};
function cards(id,items){$(id).replaceChildren(...items.map(([v,label])=>{const n=el('div',undefined,'metric');n.append(el('b',v??'—'),el('span',label));return n;}));}
function metricTable(id,rows,columns){const table=el('table',undefined,'comparison-table'),head=el('tr');for(const c of columns)head.append(el('th',c[1]));const thead=el('thead');thead.append(head);table.append(thead);const body=el('tbody');for(const row of rows){const tr=el('tr');for(const[key,,format]of columns)tr.append(el('td',format?format(row[key]):row[key]));body.append(tr);}table.append(body);$(id).replaceChildren(table);}
function render(report){
  const dataset=report.dataset??{},ranking=report.ranking??{},habitat=report.habitat??{},discovery=report.discovery??{};
  $('scout-status').textContent='Saved experiments';
  cards('corpus-metrics',[[dataset.imageCount,'licensed photos'],[dataset.blackTrumpetCount,'black-trumpet photos'],[ranking.testCount,'held-out ranking photos']]);
  metricTable('ranker-comparison',ranking.rows??[],[['label','Model'],['precision','Precision',pct],['recall','Recall',pct],['averagePrecision','Average precision',pct]]);
  $('ranker-result').textContent=ranking.conclusion??'Training report pending.';
  metricTable('discovery-comparison',discovery.rows??[],[['label','Localization experiment'],['precision','Precision',pct],['recall','Recall',pct]]);
  $('discovery-note').textContent=discovery.notes??'No completed localization result is available.';
  cards('habitat-metrics',[[habitat.imageCount,'photos segmented'],[habitat.screenLeadCount,'unlocated search leads'],[0,'real map locations']]);
  $('habitat-evidence').textContent=habitat.notes??'Habitat inference report pending.';
  const simulations=report.search?.scenarios??[],select=$('inspection-select');
  select.replaceChildren(...simulations.map((s,i)=>{const n=el('option',s.title);n.value=i;return n;}));select.disabled=!simulations.length;
  select.onchange=()=>renderInspection(simulations[Number(select.value)]);if(simulations.length)renderInspection(simulations[0]);
  const stress=discovery.scaleStress;
  if(stress?.rows?.length){const target=$('trumpet-scale');target.hidden=false;target.replaceChildren(el('div','Black-trumpet scale stress test','chart-label'));
    for(const row of stress.rows){const line=el('div',undefined,'geometry-row'),track=el('div',undefined,'bar-track'),fill=el('div',undefined,'bar-fill');fill.style.width=`${Math.max(0,Math.min(100,(row.recall??0)*100))}%`;track.append(fill);line.append(el('span',`${row.scale}×`),track,el('span',pct(row.recall)));target.append(line);}target.append(el('p',stress.note));}
}
function renderInspection(s){
  const targets=(s.mapTargets??[]).filter(t=>finite(t.position?.x)&&finite(t.position?.y)),events=s.events??[],points=events.filter(e=>e.proposal?.type==='position'&&finite(e.proposal.position?.x)&&finite(e.proposal.position?.y)).map(e=>e.proposal.position);
  const all=[{x:0,y:0},...targets.map(t=>t.position),...points],minX=Math.min(...all.map(p=>p.x))-1,maxX=Math.max(...all.map(p=>p.x))+1,minY=Math.min(...all.map(p=>p.y))-1,maxY=Math.max(...all.map(p=>p.y))+1;
  const scale=Math.min(380/(maxX-minX),150/(maxY-minY));const project=p=>({x:220+(p.x-(minX+maxX)/2)*scale,y:102-(p.y-(minY+maxY)/2)*scale});
  const map=svg('svg',{viewBox:'0 0 440 235',role:'img','aria-label':'Synthetic inspection targets and proposed route, top view. No real photo has a map location.'});
  for(let x=20;x<440;x+=40)map.append(svg('line',{x1:x,x2:x,y1:10,y2:200,stroke:'#294332'}));for(let y=20;y<205;y+=40)map.append(svg('line',{x1:15,x2:425,y1:y,y2:y,stroke:'#294332'}));
  if(points.length)map.append(svg('polyline',{points:[{x:0,y:0},...points].map(project).map(p=>`${p.x},${p.y}`).join(' '),fill:'none',stroke:'#efb65b','stroke-width':2}));
  const start=project({x:0,y:0});map.append(svg('circle',{cx:start.x,cy:start.y,r:5,fill:'#efb65b'}),svg('text',{x:start.x,y:start.y+19,fill:'#afbcab','font-size':10,'text-anchor':'middle'},'Start'));
  targets.forEach((t,i)=>{const p=project(t.position),hab=t.kind==='habitat'||t.routingCategory==='habitat_followup';map.append(svg(hab?'rect':'circle',hab?{x:p.x-6,y:p.y-6,width:12,height:12,rx:2,fill:'#b9d4a7'}:{cx:p.x,cy:p.y,r:6,fill:'#efb65b'}),svg('text',{x:p.x,y:p.y-(i%2?28:13),fill:'#edf0dc','font-size':10,'text-anchor':'middle'},hab?'Habitat lead':'Visual clue'));});
  map.append(svg('text',{x:15,y:227,fill:'#afbcab','font-size':10},'Simulation · local metres · 3D corridor checks · aircraft control off'));
  $('inspection-map').replaceChildren(map);$('inspection-summary').textContent=`${s.passed?'Passed':'Needs review'} · ${s.expected??''} · ${(s.screenLeads??[]).length} unlocated lead(s).`;
  $('inspection-events').replaceChildren(...events.filter(e=>e.reason).map(e=>{const li=el('li');li.append(el('time',finite(e.nowMs)?`${(e.nowMs/1000).toFixed(1)}s`:''),el('strong',txt(e.reason)),el('p',txt(e.stage??e.state)));return li;}));
}
async function load(){try{const r=await fetch('./scout-report.json',{cache:'no-store'});if(!r.ok)throw Error();render(await r.json());}catch{$('scout-status').textContent='Results pending';$('ranker-result').textContent='The new experiment report is being prepared. Refresh after it completes.';}}
$('refresh').addEventListener('click',load);load();
if(['localhost','127.0.0.1','[::1]'].includes(location.hostname))$('trumpet-review-link').hidden=false;
