export const LEGACY_COMMUNITY_GROUPS = [
  ['unknown','Unknown / just a mushroom'], ['morels','Morels'], ['chanterelles','Chanterelles'],
  ['black-trumpets','Black trumpets'], ['boletes','Boletes'], ['oysters','Oyster mushrooms'],
  ['hen-of-the-woods','Hen of the woods'], ['chicken-of-the-woods','Chicken of the woods'],
  ['hedgehogs','Hedgehogs'], ['puffballs','Puffballs'], ['lion-mane','Lion’s mane'], ['other','Another group'],
];
export const COMMUNITY_GROUPS = [...LEGACY_COMMUNITY_GROUPS, ...RESEARCH_CATALOG.groups
  .filter(group=>!LEGACY_COMMUNITY_GROUPS.some(([id])=>group.id===id)).map(group=>[group.id,group.label])];
export const EVIDENCE_SOURCES = [
  ['image_only','This image only'], ['multiple_views','Several views of this specimen'],
  ['field_notes','Field observations'], ['microscopy','Microscopy record'],
  ['dna','DNA record'], ['expert_reference','Expert identification reference'],
  ['imported_record','Source observation record'],
];
const byId = new Map(RESEARCH_CATALOG.taxa.map(taxon=>[taxon.id,taxon]));
const legacyGroups = {'chanterelles':'chanterelles_trumpets','black-trumpets':'chanterelles_trumpets','oysters':'gilled','lion-mane':'tooth_fungi','hen-of-the-woods':'polypores','chicken-of-the-woods':'polypores','hedgehogs':'tooth_fungi','other':'unknown'};
export function researchTaxon(id) { return byId.get(id)??null; }
export function broadGroupForReview(id) { return legacyGroups[id]??id; }
export function normalizeTaxonCandidate(value, {groupId='unknown'}={}) {
  if(value===undefined||value===null)return null;
  if(!value||typeof value!=='object'||Array.isArray(value))throw new Error('A tentative taxon must be an object.');
  const taxon=byId.get(value.taxonId);
  if(!taxon)throw new Error('Choose a research taxon from the catalog or leave identification unresolved.');
  if(typeof value.confidence!=='number'||!Number.isFinite(value.confidence)||value.confidence<0||value.confidence>1)throw new Error('Identification confidence must be between 0 and 1.');
  if(!EVIDENCE_SOURCES.some(([id])=>id===value.evidenceSource))throw new Error('Record what supports the tentative identification.');
  const note=value.evidenceNote??'';
  if(typeof note!=='string'||note.length>1000)throw new Error('Identification evidence must be text of at most 1,000 characters.');
  if(value.evidenceSource!=='image_only'&&!note.trim())throw new Error('Add a reference or note for the identification evidence.');
  const broad=broadGroupForReview(groupId);
  if(broad!=='unknown'&&broad!==taxon.broadGroupId)throw new Error('The tentative taxon must match the broad group.');
  return {taxonId:taxon.id,scientificName:taxon.scientificName,rank:taxon.rank,
    broadGroupId:taxon.broadGroupId,confidence:value.confidence,confidenceMeaning:'reviewer_reported_not_calibrated',
    evidenceSource:value.evidenceSource,evidenceNote:note.trim(),status:'tentative',
    modelSupport:'catalog_only',speciesVerified:false,trainingApproved:false,
    catalogVersion:RESEARCH_CATALOG.version};
}
