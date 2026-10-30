const byId=id=>document.getElementById(id);
const node=(tag,text,cls)=>{const item=document.createElement(tag);if(text!==undefined)item.textContent=String(text);if(cls)item.className=cls;return item;};
const percent=value=>Number.isFinite(value)?`${(value*100).toFixed(1)}%`:'—';
function render(report){
  byId('far-status').textContent=report.status==='completed'?'Measured pixel test':'Experiment pending';
  byId('far-summary').textContent=report.summary||'Results are being prepared.';
  const body=byId('far-rows');body.replaceChildren();
  for(const row of report.rows||[]){const tr=node('tr');for(const value of [row.label,`${row.targetPixels} px`,row.raw?.targets?`${row.raw.hits}/${row.raw.targets} (${percent(row.raw.recall)})`:'—',row.color?.targets?`${row.color.hits}/${row.color.targets} (${percent(row.color.recall)})`:'—'])tr.append(node('td',value));body.append(tr);}
  byId('far-controls').textContent=report.controlsNote||'Background-only controls and false alerts are reported alongside target recovery.';
  byId('far-color').textContent=report.colorNote||'Color clues request another look; they do not establish mushroom identity.';
  byId('far-limitations').replaceChildren(...(report.limitations||[]).map(text=>node('li',text)));
  const log=byId('far-log');log.replaceChildren();
  for(const row of report.behaviorChecks||[]){const line=node('li');line.append(node('strong',row.title),node('p',`${row.passed?'Passed':'Needs review'} · ${row.result}`));log.append(line);}
  byId('far-framing').textContent=report.framingNote||'Close-photo requests remain pending until a saved image proves the target framing. A crop cannot supply missing optical detail.';
}
async function load(){try{const response=await fetch('./far-report.json',{cache:'no-store'});if(!response.ok)throw Error();render(await response.json());}catch{byId('far-status').textContent='Experiment pending';byId('far-summary').textContent='The pixel-scale test is being prepared. Refresh reports after the run finishes.';}}
byId('refresh').addEventListener('click',load);load();
