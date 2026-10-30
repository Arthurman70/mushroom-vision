import { RESEARCH_CATALOG, EVIDENCE_SOURCES, researchTaxon, broadGroupForReview } from './research-taxonomy.js';

/** Optional research metadata. Never converts a model guess into a verified ID. */
export function createTaxonEditor({candidate=null, groupId='unknown', onChange, label='Tentative identification'}={}) {
  const make=(tag,text)=>{const node=document.createElement(tag);if(text)node.textContent=text;return node;};
  const root=make('details'),summary=make('summary',label+' (optional)');root.className='taxon-editor';
  root.open=Boolean(candidate);root.append(summary);
  const help=make('p','Keep this blank when detail is missing. The catalog accepts research guesses; it does not mean these species have been trained or verified.');help.className='field-help';root.append(help);
  const select=make('select');select.setAttribute('aria-label',label);select.append(new Option('Unresolved / broad group only',''));
  const broad=broadGroupForReview(groupId);
  for(const taxon of RESEARCH_CATALOG.taxa)if(broad==='unknown'||taxon.broadGroupId===broad||candidate?.taxonId===taxon.id){
    select.append(new Option(taxon.scientificName+(taxon.rank==='genus'?' · genus':'')+(taxon.nameResolution==='source_name_unresolved'?' · source name':''),taxon.id));
  }
  select.value=candidate?.taxonId??'';
  const confidence=make('select');confidence.setAttribute('aria-label',label+' confidence');
  for(const [value,title] of [['0.25','Low confidence'],['0.5','Moderate confidence'],['0.8','High confidence']])confidence.append(new Option(title,value));
  const initial=candidate?.confidence??.25;
  if(![.25,.5,.8].includes(initial))confidence.append(new Option(`${Math.round(initial*100)}% reviewer confidence`,String(initial)));
  confidence.value=String(initial);
  const evidence=make('select');evidence.setAttribute('aria-label',label+' evidence source');
  for(const [id,title] of EVIDENCE_SOURCES)evidence.append(new Option(title,id));evidence.value=candidate?.evidenceSource??'image_only';
  const note=make('textarea');note.rows=2;note.maxLength=1000;note.placeholder='Features, specimen record, or reference';note.setAttribute('aria-label',label+' evidence note');note.value=candidate?.evidenceNote??'';
  const status=make('p');status.className='field-help';
  const fields=make('div');fields.className='taxon-fields';
  for(const [title,node] of [['Research name',select],['Your confidence (not a model score)',confidence],['Evidence',evidence],['Evidence note',note]]){
    const field=make('label',title);field.append(node);fields.append(field);
  }
  root.append(fields,status);
  const update=()=>{const taxon=researchTaxon(select.value);for(const control of [confidence,evidence,note]){control.disabled=!taxon;control.dataset.emptyTaxon=String(!taxon);}
    status.textContent=taxon?'Tentative '+taxon.rank+' · catalog only · species model not trained. '+(taxon.nameResolution==='source_name_unresolved'?'Historical source name; current taxonomy needs review.':'Exact name resolved in iNaturalist; image identity still needs review.'):'No species guess recorded.';
  };
  const changed=()=>{update();onChange?.(select.value?{taxonId:select.value,confidence:Number(confidence.value),evidenceSource:evidence.value,evidenceNote:note.value}:null);};
  select.onchange=confidence.onchange=evidence.onchange=changed;note.oninput=changed;update();return root;
}
