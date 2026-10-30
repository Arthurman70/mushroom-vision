import {mkdir,writeFile} from 'node:fs/promises';
import {join} from 'node:path';

/** Metadata-only synthetic collection: no copied photographs or real reviewer records. */
export async function createImportCollection(directory) {
  const collectionDir=join(directory,'synthetic-collection');await mkdir(collectionDir,{recursive:true});
  const source={id:'synthetic-video',title:'Generated test collection',sourceUrl:'https://example.org/synthetic-video',
    attribution:'Synthetic test fixture',license:'CC0 1.0',licenseUrl:'https://creativecommons.org/publicdomain/zero/1.0/',
    videoHash:'b'.repeat(64),sourceGroupId:'synthetic:video',licenseAllowsTraining:true,evaluationOnly:true};
  const frame={id:'synthetic-frame',sourceId:source.id,sourceGroupId:source.sourceGroupId,partitionGroup:source.sourceGroupId,
    path:'frames/synthetic-frame.png',sha256:'a'.repeat(64),width:320,height:240,timestampSeconds:1,blurScore:0,
    blurMetric:'Synthetic fixture metadata; no image processing',evaluationOnly:true};
  await writeFile(join(collectionDir,'manifest.json'),JSON.stringify({schema:'mycoscout.video-collection.v1',sources:[source],frames:[frame]}));
  return {collectionDir,frame};
}
