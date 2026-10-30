import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { BulkAiReviewStore } from '../core/bulk-ai-review.mjs';
const root=resolve(fileURLToPath(new URL('..',import.meta.url))),runId=process.argv[2],file=process.argv[3];if(!runId||!file)throw new Error('Use RUN_ID OBSERVED_SHEET_NOTES_JSON');
const bytes=await readFile(resolve(file)),data=JSON.parse(bytes);if(data.reviewerKind!=='ai'||data.humanVerified!==false||!Array.isArray(data.viewedSheets)||!Array.isArray(data.records))throw new Error('Record actual viewedSheets and explicit AI observations.');
const store=await new BulkAiReviewStore(root).initialize(),noteHash=createHash('sha256').update(bytes).digest('hex');
try{const covered=new Set(store.reviews().map(r=>r.imageId));let skipped=0,saved=0,queue=[];
  for(const row of data.records){const image=store.image(row.index);if(!data.viewedSheets.includes(image.sheetId))throw new Error('Observation has no declared sheet view.');if(covered.has(image.id)){skipped++;continue;}
    queue.push({imageId:image.id,sha256:image.sha256,baseRevision:0,status:row.status,complete:false,boxes:[],broadGroupId:row.broadGroupId??'unknown',confidence:row.confidence??.6,evidence:row.evidence,
      confounderTags:row.confounderTags??[],confounders:[],observation:{mode:'contact-sheet',viewed:true,sheetId:image.sheetId,...(data.renderedSheet?{renderedSheetWidth:data.renderedSheet.width,renderedSheetHeight:data.renderedSheet.height}:{})},modelSuggestionsViewed:false,sourceNotesSha256:noteHash});covered.add(image.id);
    if(queue.length===128){saved+=(await store.saveBatch(runId,queue)).length;queue=[];}}
  if(queue.length)saved+=(await store.saveBatch(runId,queue)).length;
  console.log(JSON.stringify({saved,skippedAlreadyReviewed:skipped,currentPassReviewed:store.status().currentPassReviewed,coveredImages:store.status().coveredImages}));
}finally{store.close();}
