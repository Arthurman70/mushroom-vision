import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { AIContributionStore } from './ai-store.mjs';
const [directory,output]=process.argv.slice(2);
if(!directory||!output)throw new Error('Usage: node community/export-ai-contributions.mjs DATA_DIRECTORY NEW_OUTPUT_JSON');
const store=new AIContributionStore(resolve(directory));
try{const data=store.export();await writeFile(resolve(output),JSON.stringify(data,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({contributions:data.contributions.length,output:resolve(output),trainingApproved:false,humanVerified:false}));}
finally{store.close();}
