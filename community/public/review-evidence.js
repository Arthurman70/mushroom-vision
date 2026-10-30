/** Shared vocabulary; these are review observations, never automatic training truth. */
export const CONFOUNDERS = [['hat','Hat / clothing'],['stump','Tree stump'],['leaves','Leaves'],['tree_bark','Tree bark'],['tree','Tree / branch'],['person','Person'],['animal','Animal'],['grass','Grass'],['wood','Other wood'],['rock','Rock'],['shadow','Shadow'],['other_non_mushroom','Other non-mushroom']];
export const validConfounder = value => CONFOUNDERS.some(([id])=>id===value);
export const confounderLabel = value => CONFOUNDERS.find(([id])=>id===value)?.[1]??String(value??'Unknown');
export function matchesReviewFilter(frame,{source='',status='',confounder='',kind=''}={}) {
  return (!source||frame.sourceId===source)&&(!kind||frame.kind===kind)&&
    (!status||(status==='reviewed'?Boolean(frame.ai?.audit):status==='pending'?!frame.ai?.audit:frame.ai?.audit?.status===status))&&
    (!confounder||frame.ai?.audit?.confounderTags?.includes(confounder)||frame.ai?.audit?.confounders?.some(x=>x.kind===confounder));
}
export function auditScope(audit) {
  if(!audit)return 'No visual AI review yet';
  return audit.observation?.mode==='original-image'?'AI viewed the original image':audit.observation?.mode==='crop'?'AI viewed a crop; the rest remains unchecked':'AI viewed a reduced contact-sheet preview';
}
