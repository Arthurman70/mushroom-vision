import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { CommunityStore } from './store.mjs';
const [directory,output]=process.argv.slice(2);
if(!directory||!output)throw new Error('Usage: node community/export-reviews.mjs DATA_DIRECTORY OUTPUT_JSON');
const store=new CommunityStore(resolve(directory));
try{const data=store.export();await writeFile(resolve(output),JSON.stringify(data,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({reviews:data.reviews.length,output:resolve(output),trainingApproved:false}));}
finally{store.close();}
