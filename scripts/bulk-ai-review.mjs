import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { BulkAiReviewStore } from '../core/bulk-ai-review.mjs';
const root=resolve(fileURLToPath(new URL('..',import.meta.url))),command=process.argv[2]??'status',store=await new BulkAiReviewStore(root).initialize();
try{
  if(command==='init'){const prior=await store.importPrior(),run=store.createRun({title:'All-source independent AI visual pass',reviewer:{name:'Codex visual reviewer',model:'GPT-6 (Codex visual review)'}});console.log(JSON.stringify({prior,run,status:store.status()},null,2));}
  else if(command==='submit'){const runId=process.argv[3],file=process.argv[4];if(!runId||!file)throw new Error('Use submit RUN_ID REVIEW_JSON_FILE');const data=JSON.parse(await readFile(resolve(file),'utf8'));const reviews=await store.saveBatch(runId,Array.isArray(data)?data:data.reviews);console.log(JSON.stringify({saved:reviews.length,coverage:store.status()},null,2));}
  else if(command==='export')console.log(JSON.stringify(await store.exportFiles(),null,2));
  else if(command==='status')console.log(JSON.stringify(store.status(),null,2));
  else throw new Error('Use init, submit, export or status.');
}finally{store.close();}
