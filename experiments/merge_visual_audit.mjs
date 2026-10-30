import { readFile,writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
const dir=resolve('data/ai-video-review-v1'),path=resolve(dir,'assistant-audit.json');
const current=JSON.parse(await readFile(path,'utf8'));
const updates=JSON.parse(await readFile(resolve(dir,'audit-additions.json'),'utf8'));
const reviews=new Map(current.reviews.map(row=>[row.frameId,row]));
for(const row of updates){const prediction=JSON.parse(await readFile(resolve(dir,'frames','main--'+row.frameId+'.json'),'utf8'));
 reviews.set(row.frameId,{...row,collection:'main',imageSha256:prediction.imageSha256,reviewer:'assistant-visual-audit',humanVerified:false});}
await writeFile(path,JSON.stringify({schema:'mycoscout.assistant-audit.v1',reviews:[...reviews.values()]},null,2)+'\n');
console.log(JSON.stringify({audited:reviews.size,complete:[...reviews.values()].filter(row=>row.complete).length}));
