import { RESEARCH_CATALOG } from './research-taxonomy.js';

export function visibleResearchGuesses(prediction){
  return (Array.isArray(prediction?.guesses)?prediction.guesses:[])
    .filter(guess=>guess&&typeof guess.label==='string'&&Number.isFinite(guess.modelScore)&&guess.modelScore>=0&&guess.modelScore<=1)
    .sort((a,b)=>b.modelScore-a.modelScore).slice(0,3);
}
export function filteredResearchTaxa(search='',groupId='all'){
  const query=String(search).trim().toLowerCase();
  return RESEARCH_CATALOG.taxa.filter(taxon=>(groupId==='all'||taxon.broadGroupId===groupId)&&
    (!query||[taxon.scientificName,RESEARCH_CATALOG.groups.find(group=>group.id===taxon.broadGroupId)?.label,...taxon.aliases.map(alias=>alias.name)].some(name=>String(name??'').toLowerCase().includes(query))));
}

if(typeof document!=='undefined'){
  const $=id=>document.getElementById(id),el=(tag,text,className)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;if(className)node.className=className;return node;};
  let station=localStorage.getItem('mycoscout-station')||'',token='',status=null,photo=null,photoUrl=null,busy=false,generation=0,photoGeneration=0;
  const groupLabel=id=>RESEARCH_CATALOG.groups.find(group=>group.id===id)?.label||id||'Research category';
  const message=text=>{$('message').textContent=text;$('message').hidden=!text;};
  function controls(){$('classify').disabled=busy||!photo||status?.available!==true;$('photo').disabled=busy;$('refresh-status').disabled=busy;}
  async function request(path,body){
    const mine=generation,response=await fetch(station+path,{method:body===undefined?'GET':'POST',headers:{Accept:'application/json',...(body===undefined?{}:{'Content-Type':'application/json'}),...(token?{Authorization:'Bearer '+token}:{})},...(body===undefined?{}:{body:JSON.stringify(body)}),cache:'no-store',signal:AbortSignal.timeout(180000)});
    if(mine!==generation)throw new Error('The station connection changed. Try again on the new station.');
    let data;try{data=await response.json();}catch{throw new Error('The station returned an unreadable response.');}
    if(!response.ok)throw new Error(typeof data.error==='string'?data.error:data.error?.message||`The station could not complete this request (${response.status}).`);
    return data;
  }
  async function refreshStatus(){
    if(busy)return;busy=true;status=null;controls();$('station-status').textContent='Connecting to your field station…';message('');
    try{status=await request('/api/vision/research-status');$('station-status').textContent='Field station connected';
      $('model-status').textContent=status.available?'Research model ready':'Research model not ready';
      const classes=Array.isArray(status.supportedClasses)?status.supportedClasses:[];
      $('supported-classes').replaceChildren(...classes.map(item=>el('span',item.label||item.classId,'tag')));
      $('supported-note').textContent=`${classes.length} supported research categories. ${status.notice||'This model is separate from the default detection and flight pipeline.'}`;
      if(!status.available)message(status.notice||'The expanded research model is not installed on this station yet. You can still browse the catalog below.');
    }catch(error){$('station-status').textContent='Station unavailable';$('model-status').textContent='Catalog remains available';message(error.message);}
    finally{busy=false;controls();}
  }
  function clearResults(){
    $('results').replaceChildren(el('p','Choose Compare to see this image’s research shortlist.','empty'));
    $('result-state').textContent=photo?'Ready to compare':'Waiting for a photo';$('result-state').classList.remove('uncertain');$('result-model').textContent='';
  }
  $('photo').onchange=async()=>{
    const file=$('photo').files?.[0],mine=++photoGeneration;photo=null;clearResults();message('');controls();
    if(photoUrl){URL.revokeObjectURL(photoUrl);photoUrl=null;}$('photo-preview').hidden=true;$('photo-placeholder').hidden=false;
    if(!file)return;
    if(!['image/jpeg','image/png'].includes(file.type)||file.size<1||file.size>8*1024*1024){message('Choose a JPEG or PNG image of at most 8 MB.');return;}
    try{
      const dataUrl=await new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=()=>reject(new Error('The photo could not be read.'));reader.readAsDataURL(file);});
      if(mine!==photoGeneration)return;
      photoUrl=URL.createObjectURL(file);$('photo-preview').src=photoUrl;await $('photo-preview').decode();if(mine!==photoGeneration)return;
      if($('photo-preview').naturalWidth*$('photo-preview').naturalHeight>20000000)throw new Error('Use an image with at most 20 megapixels. A crop around one mushroom is useful.');
      photo={imageBase64:String(dataUrl).split(',')[1],mimeType:file.type};$('photo-preview').hidden=false;$('photo-placeholder').hidden=true;
      $('photo-detail').textContent=`${file.name} · ${$('photo-preview').naturalWidth} × ${$('photo-preview').naturalHeight} pixels. Compare sends this image to your field station; it does not submit a training label.`;clearResults();
    }catch(error){if(mine===photoGeneration)message(error.message);}
    finally{if(mine===photoGeneration)controls();}
  };
  $('classify').onclick=async()=>{
    if(busy||!photo||!status?.available)return;busy=true;controls();message('');$('result-state').textContent='Comparing the research categories…';
    try{
      const result=await request('/api/vision/research-classify',photo),prediction=result.prediction;
      if(!prediction||typeof prediction.abstained!=='boolean')throw new Error('The station returned an incomplete prediction.');
      const guesses=visibleResearchGuesses(prediction);$('results').replaceChildren();
      for(const [index,guess]of guesses.entries()){
        const row=el('div',undefined,'guess-card'),name=el('div');name.append(el('div',guess.label,'guess-name'),el('div',groupLabel(guess.groupId),'guess-group'));
        const score=el('div',`${Math.round(guess.modelScore*100)}%`,'guess-score');score.append(el('small','model score'));
        row.append(el('span',String(index+1).padStart(2,'0'),'guess-rank'),name,score);$('results').append(row);
      }
      if(!guesses.length)$('results').append(el('p','No supported category guess was returned. Keep this identification unresolved.','empty'));
      $('result-state').textContent=prediction.abstained?'Uncertain · more evidence needed':'Tentative category shortlist · human review needed';$('result-state').classList.toggle('uncertain',prediction.abstained);
      const reasons=Array.isArray(prediction.abstentionReasons)?prediction.abstentionReasons.map(reason=>String(reason).replaceAll('_',' ')).join('; '):'';
      $('result-notice').textContent='Scores compare supported categories, not species certainty. '+(prediction.abstained&&reasons?`The model held back: ${reasons}. `:'')+'This whole-image test does not find mushroom locations or establish flight-distance performance.';
      $('result-model').textContent='Research model: '+(prediction.evidenceVersion||prediction.modelVersion||status.modelVersion||'version not reported');
    }catch(error){$('result-state').textContent='Comparison could not finish';message(error.message);}
    finally{busy=false;controls();}
  };
  function renderCatalog(){
    const taxa=filteredResearchTaxa($('catalog-search').value,$('catalog-group').value);$('catalog-count').textContent=`${taxa.length} names shown`;$('catalog-list').replaceChildren();
    for(const taxon of taxa){
      const card=el('details',undefined,'taxon-card'),summary=el('summary');summary.append(el('span',taxon.scientificName,'taxon-name'),el('span',`${taxon.rank} · ${groupLabel(taxon.broadGroupId)}`,'taxon-group'));card.append(summary);
      card.append(el('p','Research annotation name. Species recognition accuracy has not been measured. '+(taxon.nameResolution==='source_name_unresolved'?'This preserves a study-guide name; its current taxonomic application remains unresolved.':'Exact name resolved in the cited iNaturalist record; the identity of your specimen still needs review.')));
      if(taxon.aliases.length)card.append(el('p','Source aliases (not automatically equated): '+taxon.aliases.map(alias=>alias.name).join(', ')));
      for(const sourceId of taxon.sourceIds.filter(id=>id!=='mami-index').slice(0,3)){const source=RESEARCH_CATALOG.sources.find(item=>item.id===sourceId);if(!source?.url?.startsWith('https://'))continue;const link=el('a',source.title||'Source record');link.href=source.url;link.target='_blank';link.rel='noopener';card.append(link);}
      $('catalog-list').append(card);
    }
    if(!taxa.length)$('catalog-list').append(el('p','No matching research name. Try a genus or broader group.','fine'));
  }
  for(const group of RESEARCH_CATALOG.groups)$('catalog-group').append(new Option(group.label,group.id));
  $('catalog-total').textContent=`${RESEARCH_CATALOG.taxa.length} research names`;
  $('catalog-search').oninput=$('catalog-group').onchange=renderCatalog;renderCatalog();
  $('refresh-status').onclick=refreshStatus;
  $('open-connection').onclick=()=>{$('station-url').value=station;$('station-token').value=token;$('connection-dialog').showModal();};
  $('close-connection').onclick=()=>$('connection-dialog').close();
  $('connection-form').onsubmit=event=>{event.preventDefault();if(busy)return;try{const raw=$('station-url').value.trim();let next='';if(raw){const url=new URL(raw);if(!['http:','https:'].includes(url.protocol)||url.username||url.password||url.pathname!=='/'||url.search||url.hash)throw new Error('Use the station’s http or https address without a path or credentials.');next=url.origin;}station=next;token=$('station-token').value;generation++;localStorage.setItem('mycoscout-station',station);$('connection-dialog').close();clearResults();refreshStatus();}catch(error){message(error.message);}};
  window.addEventListener('pagehide',()=>{if(photoUrl)URL.revokeObjectURL(photoUrl);});
  refreshStatus();
}
