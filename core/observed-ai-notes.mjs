export const observedConfounderTag=value=>({trees:'tree',leaves:'leaf',branches:'wood',tree_bark:'tree',hand:'person',hat:'hat',stump:'stump',tree:'tree',leaf:'leaf',wood:'wood',rock:'rock',person:'person',animal:'animal'})[value]??'other';
export function observedConfounders(row){
  if(row.confounders!==undefined&&!Array.isArray(row.confounders)||row.negativeRegions!==undefined&&!Array.isArray(row.negativeRegions))throw new Error('Observed nonfungal regions must be arrays.');
  const regions=[];
  for(const [boxes,legacy]of [[row.confounders??[],false],[row.negativeRegions??[],true]])for(const box of boxes){
    if(box.explicitNonFungal===false||!legacy&&box.explicitNonFungal!==true)throw new Error('A confounder must explicitly be observed as nonfungal.');
    const region={kind:observedConfounderTag(legacy?box.category:box.kind),bboxPixelsXYXY:box.bboxPixelsXYXY,explicitNonFungal:true,confidence:box.confidence,evidence:box.evidence};
    if(!regions.some(value=>JSON.stringify(value)===JSON.stringify(region)))regions.push(region);
  }
  return regions;
}
export function sameObservedConfounders(saved,observed){const clean=rows=>rows.map(({kind,bboxPixelsXYXY,explicitNonFungal,confidence,evidence})=>({kind,bboxPixelsXYXY,explicitNonFungal,confidence,evidence}));return JSON.stringify(clean(saved??[]))===JSON.stringify(clean(observed));}
