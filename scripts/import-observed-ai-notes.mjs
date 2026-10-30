import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { BulkAiReviewStore } from '../core/bulk-ai-review.mjs';
import { observedConfounderTag as tag, observedConfounders, sameObservedConfounders } from '../core/observed-ai-notes.mjs';
const root=resolve(fileURLToPath(new URL('..',import.meta.url))),runId=process.argv[2],files=process.argv.slice(3);if(!runId||!files.length)throw new Error('Use RUN_ID NOTES_JSON_FILE [MORE_FILES...]');
const store=await new BulkAiReviewStore(root).initialize();
try{
  let saved=0,already=0;
  for(const file of files){const bytes=await readFile(resolve(file)),data=JSON.parse(bytes),noteHash=createHash('sha256').update(bytes).digest('hex');if(data.schema!=='mushroom-vision.observed-review-notes.v1'||!Array.isArray(data.records))throw new Error('Unsupported observed-note file.');
    const rows=[];
    for(const row of data.records){if(row.reviewerKind!=='ai'||row.humanVerified!==false||row.observation?.mode!=='original-image')throw new Error('Observed originals must retain AI authority.');
      const matches=store.manifest.records.filter(image=>image.aliases.some(alias=>alias.id===row.frameId&&alias.collection==='video-discovery-v1'));if(matches.length!==1)throw new Error('Original main-collection frame is missing or ambiguous: '+row.frameId);const image=matches[0],history=store.get(image.id).reviews.filter(review=>review.runId===runId),latest=history.at(-1);
      const confounders=observedConfounders(row);
      if(latest?.sourceNotesSha256===noteHash&&sameObservedConfounders(latest.confounders,confounders)){already++;continue;}
      rows.push({imageId:image.id,sha256:image.sha256,baseRevision:latest?.revision??0,status:row.status,complete:row.complete,boxes:row.boxes,broadGroupId:row.boxes?.[0]?.groupId??'unknown',confidence:row.boxes?.[0]?.confidence??(row.status==='uncertain'?.4:.8),evidence:row.evidence,
        observation:{...row.observation,viewed:true},confounders,confounderTags:[...new Set((row.confounderTags??[]).map(tag))],modelSuggestionsViewed:row.modelSuggestionsViewed,sourceNotesSha256:noteHash});
    }
    if(rows.length){await store.saveBatch(runId,rows);saved+=rows.length;}
  }
  console.log(JSON.stringify({saved,alreadyImported:already,...await store.exportFiles()},null,2));
}finally{store.close();}
