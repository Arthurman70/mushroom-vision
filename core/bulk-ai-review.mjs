import { DatabaseSync } from 'node:sqlite';
import { mkdir,readFile,writeFile,realpath,stat } from 'node:fs/promises';
import { resolve,sep } from 'node:path';
import { createHash,randomUUID } from 'node:crypto';
import { ApiError,shortText } from './validation.mjs';
import { COMMUNITY_GROUPS } from '../community/public/research-taxonomy.js';

const HASH=/^[a-f0-9]{64}$/,ID=/^[A-Za-z0-9_-]{1,160}$/;
const groups=new Set(COMMUNITY_GROUPS.map(([id])=>id)),confounderKinds=new Set(['hat','stump','tree','leaf','rock','wood','person','animal','other']);
const hash=data=>createHash('sha256').update(data).digest('hex');
export const stableAiReview=value=>Array.isArray(value)?value.map(stableAiReview):value&&typeof value==='object'?Object.fromEntries(Object.keys(value).sort().map(key=>[key,stableAiReview(value[key])])):value;
export const aiReviewRecordHash=row=>hash(JSON.stringify(stableAiReview(row)));
const clone=value=>structuredClone(value),text=(value,name,max=4000)=>shortText(value,name,undefined,max);
const bad=(message,code='BULK_AI_INVALID')=>{throw new ApiError(message,409,code);};
function bbox(value,image){if(!Array.isArray(value)||value.length!==4||value.some(n=>typeof n!=='number'||!Number.isFinite(n))||value[0]<0||value[1]<0||value[2]>image.width||value[3]>image.height||value[2]<=value[0]||value[3]<=value[1])throw new ApiError('Use a nonempty original-pixel box within the image.');return [...value];}
function confidence(value){if(typeof value!=='number'||!Number.isFinite(value)||value<0||value>1)throw new ApiError('Confidence must be between zero and one.');return value;}
const overlap=(a,b)=>Math.min(a[2],b[2])>Math.max(a[0],b[0])&&Math.min(a[3],b[3])>Math.max(a[1],b[1]);

export class BulkAiReviewStore {
  constructor(root,directory=resolve(root,'data/bulk-ai-review-v1')){this.root=resolve(root);this.directory=resolve(directory);this.db=null;}
  async initialize(){
    const bytes=await readFile(resolve(this.directory,'manifest.json')),manifest=JSON.parse(bytes),freeze=JSON.parse(await readFile(resolve(this.directory,'manifest-sha256.json'),'utf8'));
    if(bytes.length>64*1024*1024||manifest.schema!=='mycoscout.bulk-ai-inventory.v1'||!Array.isArray(manifest.records)||manifest.records.length>10000||freeze.sha256!==hash(bytes))bad('Bulk inventory integrity failed.');
    this.inventorySha256=freeze.sha256;this.manifest=manifest;this.images=new Map();
    for(const image of manifest.records){if(image.id!=='image-'+image.sha256||!HASH.test(image.sha256)||!HASH.test(image.pixelSha256)||this.images.has(image.id)||!['train','validation','test','unassigned'].includes(image.split)||!Array.isArray(image.aliases)||!Number.isInteger(image.width)||!Number.isInteger(image.height)||image.width<1||image.height<1||image.width*image.height>30000000||typeof image.path!=='string'||!image.path.startsWith('data/'))bad('Invalid bulk inventory image.');this.images.set(image.id,image);}
    let sheets={sheets:[]};try{sheets=JSON.parse(await readFile(resolve(this.directory,'sheets.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
    this.sheets=new Map(sheets.sheets.map(sheet=>[sheet.id,sheet]));
    await mkdir(this.directory,{recursive:true});this.db=new DatabaseSync(resolve(this.directory,'bulk-ai-review.sqlite'));
    this.db.exec(`PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL; PRAGMA busy_timeout=10000;
      CREATE TABLE IF NOT EXISTS bulk_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS bulk_runs(id TEXT PRIMARY KEY,data TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS bulk_reviews(id TEXT PRIMARY KEY,run_id TEXT NOT NULL,image_id TEXT NOT NULL,revision INTEGER NOT NULL,record_hash TEXT NOT NULL UNIQUE,data TEXT NOT NULL,UNIQUE(run_id,image_id,revision));
      CREATE TRIGGER IF NOT EXISTS bulk_runs_immutable_update BEFORE UPDATE ON bulk_runs BEGIN SELECT RAISE(ABORT,'AI run is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS bulk_runs_immutable_delete BEFORE DELETE ON bulk_runs BEGIN SELECT RAISE(ABORT,'AI run is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS bulk_reviews_immutable_update BEFORE UPDATE ON bulk_reviews BEGIN SELECT RAISE(ABORT,'AI review is immutable'); END;
      CREATE TRIGGER IF NOT EXISTS bulk_reviews_immutable_delete BEFORE DELETE ON bulk_reviews BEGIN SELECT RAISE(ABORT,'AI review is immutable'); END;`);
    const prior=this.db.prepare('SELECT value FROM bulk_meta WHERE key=?').get('inventorySha256');if(prior&&prior.value!==this.inventorySha256)bad('Inventory changed after this ledger was created. Freeze a new review collection.');
    this.db.prepare('INSERT OR IGNORE INTO bulk_meta(key,value) VALUES(?,?)').run('inventorySha256',this.inventorySha256);return this;
  }
  async original(image){const root=await realpath(this.root),path=await realpath(resolve(this.root,image.path));if(!path.startsWith(root+sep))bad('Image path escaped the workspace.');if((await stat(path)).size>20*1024*1024)bad('Image exceeds review limit.');const bytes=await readFile(path);if(hash(bytes)!==image.sha256)bad('Image pixels changed.','BULK_AI_INTEGRITY');return bytes;}
  image(id){const image=Number.isInteger(id)?this.manifest.records[id]:this.images.get(id);if(!image)throw new ApiError('Unknown inventory image.',404);return image;}
  createRun(input){
    if(input?.humanVerified===true||input?.reviewerKind&&input.reviewerKind!=='ai')throw new ApiError('This is an AI-only review ledger.');
    const run={schema:'mycoscout.bulk-ai-run.v1',id:'bulk-'+randomUUID(),inventorySha256:this.inventorySha256,title:text(input?.title??'All-source AI visual pass','title',200),
      reviewer:{kind:'ai',name:text(input?.reviewer?.name??'Codex visual reviewer','reviewer',120),model:text(input?.reviewer?.model,'model identity',160)},reviewerKind:'ai',humanVerified:false,createdAt:new Date().toISOString(),scope:'all-inventory-images',automaticTraining:false};
    this.db.prepare('INSERT INTO bulk_runs(id,data) VALUES(?,?)').run(run.id,JSON.stringify(run));return run;
  }
  run(id){if(!ID.test(id??''))throw new ApiError('Unknown bulk run.',404);const row=this.db.prepare('SELECT data FROM bulk_runs WHERE id=?').get(id);if(!row)throw new ApiError('Unknown bulk run.',404);return JSON.parse(row.data);}
  reviews(){return this.db.prepare('SELECT data,record_hash FROM bulk_reviews ORDER BY rowid').all().map(row=>({...JSON.parse(row.data),ledgerRecordHash:row.record_hash}));}
  get(imageId){const image=this.image(imageId);return {image:clone(image),reviews:this.reviews().filter(row=>row.imageId===image.id)};}
  selectedReviews(){const selected=new Map();for(const row of this.reviews()){const previous=selected.get(row.imageId);if(!previous||(row.viewedOriginal&&!previous.viewedOriginal)||(row.viewedOriginal===previous.viewedOriginal&&row.createdAt>=previous.createdAt))selected.set(row.imageId,row);}return selected;}
  page(query={}){
    const offset=Number(query.offset??0),limit=Number(query.limit??24),status=query.status??'all',mode=query.mode??'all',split=query.split??'all';
    if(!Number.isInteger(offset)||offset<0||!Number.isInteger(limit)||limit<1||limit>48||!['all','unreviewed','visible','absent','uncertain'].includes(status)||!['all','original-image','contact-sheet'].includes(mode)||!['all','train','validation','test','unassigned'].includes(split))throw new ApiError('Invalid review gallery filters.');
    const selected=this.selectedReviews(),rows=this.manifest.records.filter(image=>{const review=selected.get(image.id);return (status==='all'||status==='unreviewed'&&!review||review?.status===status)&&(mode==='all'||review?.observation.mode===mode)&&(split==='all'||image.split===split);});
    return {inventorySha256:this.inventorySha256,total:rows.length,offset,limit,records:rows.slice(offset,offset+limit).map(image=>({id:image.id,index:image.index,width:image.width,height:image.height,kind:image.kind,split:image.split,sourceId:image.sourceId,sourceGroupId:image.sourceGroupId,rights:clone(image.rights),sha256:image.sha256,review:clone(selected.get(image.id)??null),imageUrl:'/api/bulk-ai-review/images/'+image.id+'/original'}))};
  }
  async imageBytes(imageId){const bytes=await this.original(this.image(imageId));let mimeType;if(bytes[0]===255&&bytes[1]===216&&bytes[2]===255)mimeType='image/jpeg';else if(bytes.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))mimeType='image/png';else if(bytes.subarray(0,4).toString()==='RIFF'&&bytes.subarray(8,12).toString()==='WEBP')mimeType='image/webp';else if(['GIF87a','GIF89a'].includes(bytes.subarray(0,6).toString()))mimeType='image/gif';else throw new ApiError('Unsupported original image format.',415);return {bytes,mimeType};}
  normalize(run,image,input,{prior=false}={}){
    if(input.reviewerKind&&input.reviewerKind!=='ai'||input.humanVerified===true||input.sha256!==image.sha256)bad('AI authority or source hash mismatch.');
    const status=input.status;if(!['visible','absent','uncertain'].includes(status)||typeof input.complete!=='boolean')throw new ApiError('Choose visible, absent or uncertain and an explicit completeness flag.');
    const observation=input.observation;if(!observation||!['contact-sheet','original-image','crop'].includes(observation.mode)||observation.viewed!==true)throw new ApiError('Record what was actually viewed.');
    let viewed={mode:observation.mode,viewed:true,displayedWidth:observation.displayedWidth,displayedHeight:observation.displayedHeight};
    if(observation.mode==='contact-sheet'){
      const sheet=this.sheets.get(observation.sheetId),tile=sheet?.tiles.find(tile=>tile.id===image.id);if(!sheet||!tile)bad('Contact sheet does not contain this image.');
      const renderedWidth=observation.renderedSheetWidth,renderedHeight=observation.renderedSheetHeight,known=renderedWidth!==undefined||renderedHeight!==undefined;
      if(known&&(!Number.isInteger(renderedWidth)||!Number.isInteger(renderedHeight)||renderedWidth<1||renderedHeight<1||renderedWidth>(sheet.width??1600)||renderedHeight>(sheet.height??1680)))throw new ApiError('Use the actual reported rendered sheet dimensions.');
      viewed={...viewed,sheetId:sheet.id,sheetSha256:sheet.sha256,generatedTileWidth:tile.displayedWidth,generatedTileHeight:tile.displayedHeight,
        displayedWidth:known?Number((tile.displayedWidth*renderedWidth/(sheet.width??1600)).toFixed(2)):null,displayedHeight:known?Number((tile.displayedHeight*renderedHeight/(sheet.height??1680)).toFixed(2)):null,
        renderedSheetWidth:known?renderedWidth:null,renderedSheetHeight:known?renderedHeight:null,displayResolutionKnown:known,tileIndex:sheet.tiles.indexOf(tile),resolutionNote:known?'Effective tile resolution estimated from the tool-reported rendered sheet size.':'Generated tile dimensions recorded; actual display scaling was not reported.'};
    }else{
      if(!Number.isInteger(viewed.displayedWidth)||!Number.isInteger(viewed.displayedHeight)||viewed.displayedWidth<1||viewed.displayedHeight<1||viewed.displayedWidth>image.width||viewed.displayedHeight>image.height)throw new ApiError('Record the displayed image dimensions without inventing extra pixels.');
      if(observation.mode==='crop')viewed.cropPixelsXYXY=bbox(observation.cropPixelsXYXY,image);
    }
    if(input.complete&&(status==='uncertain'||observation.mode!=='original-image'))throw new ApiError('Only a complete original-image review can supply full-frame labels. Contact sheets and crops remain incomplete.');
    if(!Array.isArray(input.boxes)||input.boxes.length>128||status!=='visible'&&input.boxes.length||status==='visible'&&input.complete&&!input.boxes.length)throw new ApiError('Visible complete frames need positive boxes; nonvisible frames cannot have them.');
    const boxes=input.boxes.map((box,index)=>{if(!groups.has(box.groupId??'unknown'))throw new ApiError('Unsupported broad group.');return {id:'positive-'+index,bboxPixelsXYXY:bbox(box.bboxPixelsXYXY,image),groupId:box.groupId??'unknown',confidence:confidence(box.confidence),evidence:text(box.evidence,'positive box evidence'),reviewerKind:'ai',humanVerified:false};});
    const rawConfounders=input.confounders??[];if(!Array.isArray(rawConfounders)||rawConfounders.length>128)throw new ApiError('Use at most 128 confounder regions.');
    const confounders=rawConfounders.map((box,index)=>{if(!confounderKinds.has(box.kind)||box.explicitNonFungal!==true)throw new ApiError('A negative region needs an explicit nonfungal observation and supported kind.');const region=bbox(box.bboxPixelsXYXY,image);if(boxes.some(positive=>overlap(region,positive.bboxPixelsXYXY)))throw new ApiError('A nonfungal region overlaps a mushroom box. Keep tree or stump context as a tag instead.');return {id:'confounder-'+index,kind:box.kind,bboxPixelsXYXY:region,explicitNonFungal:true,confidence:confidence(box.confidence),evidence:text(box.evidence,'confounder evidence'),reviewerKind:'ai',humanVerified:false};});
    const tags=input.confounderTags??[];if(!Array.isArray(tags)||tags.length>confounderKinds.size||tags.some(tag=>!confounderKinds.has(tag)))throw new ApiError('Unsupported confounder tags.');
    if(typeof input.modelSuggestionsViewed!=='boolean')throw new ApiError('Record whether model suggestions were seen.');
    const broadGroupId=input.broadGroupId??'unknown';if(!groups.has(broadGroupId))throw new ApiError('Unsupported broad group.');
    if(!Number.isInteger(input.baseRevision)||input.baseRevision<0)throw new ApiError('Supply the current review revision.');
    return {schema:'mycoscout.bulk-ai-review.v1',id:'bair-'+randomUUID(),runId:run.id,imageId:image.id,index:image.index,sha256:image.sha256,pixelSha256:image.pixelSha256,width:image.width,height:image.height,
      sourceGroupId:image.sourceGroupId,split:image.split,reviewer:run.reviewer,reviewerKind:'ai',humanVerified:false,reviewOrigin:prior?'prior-reviewed-original':'current-pass',
      status,complete:input.complete,broadGroupId,confidence:confidence(input.confidence),evidence:text(input.evidence,'image evidence'),boxes,confounders,confounderTags:[...new Set(tags)],observation:viewed,
      viewedOriginal:observation.mode==='original-image',modelSuggestionsViewed:input.modelSuggestionsViewed,automaticPropagation:false,trainingAllowed:false,
      experimentalTrainingAllowed:input.complete&&status!=='uncertain'&&observation.mode==='original-image'&&image.experimentalTrainingEligible===true,
      priorEvidence:prior?clone(input.priorEvidence):null,sourceNotesSha256:HASH.test(input.sourceNotesSha256??'')?input.sourceNotesSha256:null,revision:input.baseRevision+1,createdAt:new Date().toISOString()};
  }
  async saveBatch(runId,inputs){
    const run=this.run(runId);if(!Array.isArray(inputs)||!inputs.length||inputs.length>256)throw new ApiError('Submit 1–256 actually observed image reviews.');
    const seen=new Set(),checkedSheets=new Set(),reviews=[];for(const input of inputs){const image=this.image(input.imageId??input.index);if(seen.has(image.id))throw new ApiError('Repeated image in one batch.');seen.add(image.id);await this.original(image);const review=this.normalize(run,image,input);
      if(review.observation.mode==='contact-sheet'&&!checkedSheets.has(review.observation.sheetId)){const sheet=this.sheets.get(review.observation.sheetId),directory=await realpath(this.directory),path=await realpath(resolve(this.directory,sheet.path));if(!path.startsWith(directory+sep)||(await stat(path)).size>20*1024*1024||hash(await readFile(path))!==sheet.sha256)bad('Contact-sheet pixels changed.','BULK_AI_INTEGRITY');checkedSheets.add(sheet.id);}
      reviews.push(review);}
    return this.insert(reviews);
  }
  insert(reviews){this.db.exec('BEGIN IMMEDIATE');try{for(const review of reviews){const previous=this.db.prepare('SELECT MAX(revision) AS n FROM bulk_reviews WHERE run_id=? AND image_id=?').get(review.runId,review.imageId).n??0;if(previous!==review.revision-1)bad('A newer review exists; reload this image.','BULK_AI_CONFLICT');const recordHash=aiReviewRecordHash(review);this.db.prepare('INSERT INTO bulk_reviews(id,run_id,image_id,revision,record_hash,data) VALUES(?,?,?,?,?,?)').run(review.id,review.runId,review.imageId,review.revision,recordHash,JSON.stringify(review));review.ledgerRecordHash=recordHash;}this.db.exec('COMMIT');return reviews;}catch(error){this.db.exec('ROLLBACK');throw error;}}
  async importPrior(){
    const previous=this.db.prepare("SELECT value FROM bulk_meta WHERE key='priorImported'").get();if(previous)return {imported:0,alreadyImported:true};
    const run=this.createRun({title:'Preserved original-frame evidence from the previous AI pass',reviewer:{name:'Codex visual reviewer',model:'GPT-6 (Codex visual review)'}}),reviews=[];
    for(const image of this.manifest.records){const prior=image.priorAiReviews?.at(-1);if(!prior||prior.viewedOriginal!==true||prior.reviewerKind!=='ai'||prior.humanVerified!==false||prior.sha256!==image.sha256)continue;
      await this.original(image);const normalized=this.normalize({...run,reviewer:prior.reviewer},image,{baseRevision:0,sha256:image.sha256,status:prior.status,complete:prior.complete,boxes:prior.boxes,confounders:[],confounderTags:[],broadGroupId:prior.boxes[0]?.groupId??'unknown',confidence:prior.boxes[0]?.confidence??.5,evidence:prior.evidence,
        observation:{mode:'original-image',viewed:true,displayedWidth:image.width,displayedHeight:image.height},modelSuggestionsViewed:prior.modelSuggestionsViewed,
        priorEvidence:{runId:prior.runId,reviewId:prior.id,reviewSha256:aiReviewRecordHash(prior),createdAt:prior.createdAt,resolutionNote:'Previous record claimed original view; exact display resolution was not recorded.'}},{prior:true});reviews.push(normalized);}
    this.insert(reviews);this.db.prepare('INSERT INTO bulk_meta(key,value) VALUES(?,?)').run('priorImported',String(reviews.length));return {imported:reviews.length,runId:run.id};
  }
  status(){const reviews=this.reviews(),current=new Set(reviews.filter(r=>r.reviewOrigin==='current-pass').map(r=>r.imageId)),prior=new Set(reviews.filter(r=>r.reviewOrigin==='prior-reviewed-original').map(r=>r.imageId)),all=new Set([...current,...prior]);return {schema:'mycoscout.bulk-ai-status.v1',inventorySha256:this.inventorySha256,totalImages:this.images.size,currentPassReviewed:current.size,priorOriginalReviewed:prior.size,coveredImages:all.size,unreviewedImages:this.images.size-all.size,
    originalViewedImages:new Set(reviews.filter(r=>r.viewedOriginal).map(r=>r.imageId)).size,contactSheetReviewedImages:new Set(reviews.filter(r=>r.observation.mode==='contact-sheet').map(r=>r.imageId)).size,
    humanVerified:false,automaticTraining:false,runs:this.db.prepare('SELECT data FROM bulk_runs ORDER BY rowid DESC').all().map(row=>JSON.parse(row.data))};}
  async exportFiles(){
    const records=this.reviews(),ledger={schema:'mycoscout.bulk-ai-review-ledger.v1',inventorySha256:this.inventorySha256,reviewerKind:'ai',humanVerified:false,automaticActivation:false,coverage:this.status(),records};
    const ledgerBytes=JSON.stringify(ledger,null,2)+'\n',ledgerSha256=hash(ledgerBytes);await writeFile(resolve(this.directory,'ledger.json'),ledgerBytes);
    const choices=new Map();for(const row of records){if(!row.viewedOriginal||row.observation.mode!=='original-image')continue;const current=choices.get(row.imageId);if(!current||row.createdAt>=current.createdAt)choices.set(row.imageId,row);}
    const training=[];for(const row of choices.values()){const image=this.image(row.imageId);if(!row.complete||row.status==='uncertain'||!image.rights.experimentalTrainingPermission||!['train','validation'].includes(image.split))continue;await this.original(image);
      training.push({...clone(row),id:image.id,imagePath:image.path,imageSha256:image.sha256,sourceGroupId:image.sourceGroupId,groupKeys:image.groupKeys,split:image.split,
        rights:image.rights,experimentalTrainingEligible:image.experimentalTrainingEligible,activationEligible:false});}
    const exported={schema:'mushroom-vision.ai-corpus-training.v1',createdAt:new Date().toISOString(),reviewerKind:'ai',humanVerified:false,activationEligible:false,originalInventorySha256:this.inventorySha256,ledgerSha256,records:training,
      notice:'Only complete original-image AI evidence is eligible. Contact-sheet screening is incomplete; tags alone are never negative boxes. Reserved tests are excluded.'};
    await writeFile(resolve(this.directory,'training-export.json'),JSON.stringify(exported,null,2)+'\n');return {ledgerSha256,inventorySha256:this.inventorySha256,coverage:ledger.coverage,trainingRecords:training.length,trainingBoxes:training.reduce((sum,row)=>sum+row.boxes.length,0)};
  }
  async dispatch(method,path,input){
    if(method==='GET'&&path==='/api/bulk-ai-review/status')return {status:200,data:this.status()};
    if(method==='GET'&&path==='/api/bulk-ai-review/images')return {status:200,data:this.page(input)};
    if(method==='POST'&&path==='/api/bulk-ai-review/runs')return {status:201,data:this.createRun(input)};
    const review=/^\/api\/bulk-ai-review\/runs\/([A-Za-z0-9_-]+)\/reviews$/.exec(path);if(method==='POST'&&review)return {status:201,data:{reviews:await this.saveBatch(review[1],input.reviews)}};
    const image=/^\/api\/bulk-ai-review\/images\/(image-[a-f0-9]{64})$/.exec(path);if(method==='GET'&&image)return {status:200,data:this.get(image[1])};
    if(method==='POST'&&path==='/api/bulk-ai-review/export')return {status:200,data:await this.exportFiles()};return null;
  }
  close(){this.db?.close();this.db=null;}
}
