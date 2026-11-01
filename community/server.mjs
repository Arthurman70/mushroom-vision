import { createServer } from 'node:http';
import { readFile, realpath, stat, open } from 'node:fs/promises';
import { resolve, dirname, sep, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash, timingSafeEqual, randomBytes } from 'node:crypto';
import { CommunityStore, ReviewError, PARTICIPATION_TERMS } from './store.mjs';
import { AIContributionStore, AI_TERMS, AI_LIMITS, aiSourceAllowed, aiEligibleFrames } from './ai-store.mjs';
import { RESEARCH_CATALOG } from './public/research-taxonomy.js';
import { publicAudit } from './public/ai-review-data.js';

const ROOT=dirname(fileURLToPath(import.meta.url));
const MIME={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.jpg':'image/jpeg','.png':'image/png','.svg':'image/svg+xml','.txt':'text/plain; charset=utf-8','.json':'application/json; charset=utf-8'};
const ID=/^[A-Za-z0-9_-]{1,160}$/;
const secureEquals=(a,b)=>typeof a==='string'&&typeof b==='string'&&Buffer.byteLength(a)===Buffer.byteLength(b)&&timingSafeEqual(Buffer.from(a),Buffer.from(b));
const WORKER_STATES=new Set(['disabled','paused','waiting_data','waiting_daily_limit','waiting_resource','waiting_sync','exporting','training','comparing','retry_wait','candidate_ready','failed']);
const CANDIDATE_STATES=new Set(['candidate_better','candidate_not_better','insufficient_evidence','comparison_failed']);
export const STATUS_MAX_AGE_MS=15*60*1000;
export async function readTrainingStatus(dataDirectory,now=Date.now()) {
  const empty=availability=>({schema:'mycoscout.community-learning-status.v1',availability,updatedAt:null,
    worker:{state:'offline',enabled:false,paused:false},approvedFrames:null,newFrames:null,newSessions:null,thresholds:null,latestCandidate:null,automaticActivation:false});
  let handle;
  try {
    handle=await open(resolve(dataDirectory,'training-status.json'),'r');
    if((await handle.stat()).size>65536)return empty('invalid');
    const bytes=Buffer.alloc(65537),{bytesRead}=await handle.read(bytes,0,bytes.length,0);
    if(bytesRead>65536)return empty('invalid');
    const data=JSON.parse(bytes.subarray(0,bytesRead).toString('utf8'));
    const count=n=>Number.isSafeInteger(n)&&n>=0&&n<=10000000;
    const timestamp=value=>typeof value==='string'&&Number.isFinite(Date.parse(value))&&new Date(value).toISOString()===value&&Date.parse(value)<=now+300000;
    const metric=(value,max)=>value===null||(typeof value==='number'&&Number.isFinite(value)&&value>=0&&value<=max);
    if(data.schema!=='mycoscout.community-learning-status.v1'||!timestamp(data.updatedAt)||!WORKER_STATES.has(data.worker?.state)||
      typeof data.worker.enabled!=='boolean'||typeof data.worker.paused!=='boolean'||data.automaticActivation!==false||
      ![data.approvedFrames,data.newFrames,data.newSessions].every(n=>n===null||count(n))||
      ![data.thresholds?.newFrames,data.thresholds?.newSessions,data.thresholds?.maxRunsPerDay].every(count))return empty('invalid');
    let candidate=null;
    if(data.latestCandidate!==null){
      const item=data.latestCandidate,c=item?.comparison;
      if(!CANDIDATE_STATES.has(item?.status)||!timestamp(item.completedAt)||!c||!metric(c.baselineRecall,1)||!metric(c.candidateRecall,1)||
        !metric(c.baselineFalsePositivesPerFrame,1000000)||!metric(c.candidateFalsePositivesPerFrame,1000000))return empty('invalid');
      candidate={status:item.status,completedAt:item.completedAt,comparison:{baselineRecall:c.baselineRecall,candidateRecall:c.candidateRecall,
        baselineFalsePositivesPerFrame:c.baselineFalsePositivesPerFrame,candidateFalsePositivesPerFrame:c.candidateFalsePositivesPerFrame}};
    }
    const stale=now-Date.parse(data.updatedAt)>STATUS_MAX_AGE_MS;
    return {schema:'mycoscout.community-learning-status.v1',availability:stale?'stale':'available',updatedAt:data.updatedAt,
      worker:stale?{state:'offline',enabled:false,paused:false}:{state:data.worker.state,enabled:data.worker.enabled,paused:data.worker.paused},
      approvedFrames:data.approvedFrames,newFrames:data.newFrames,newSessions:data.newSessions,
      thresholds:{newFrames:data.thresholds.newFrames,newSessions:data.thresholds.newSessions,maxRunsPerDay:data.thresholds.maxRunsPerDay},
      latestCandidate:candidate,automaticActivation:false};
  }catch(error){return empty(error.code==='ENOENT'?'missing':'invalid');}
  finally{await handle?.close();}
}
export async function createCommunityServer(options={}) {
  const directory=resolve(options.collectionDir??process.env.COLLECTION_DIR??resolve(ROOT,'collection'));
  const base=options.basePath??process.env.BASE_PATH??'/mushroom-lab/';
  if(!/^\/[a-z0-9-]+\/$/.test(base)) throw new Error('BASE_PATH must be one absolute directory.');
  const origin=new URL(options.origin??process.env.PUBLIC_ORIGIN??'http://127.0.0.1:8791').origin;
  const collection=JSON.parse(await readFile(resolve(directory,'manifest.json'),'utf8'));
  if(collection.schema!=='mycoscout.community-collection.v1'||!Array.isArray(collection.frames)||collection.frames.length>10000||!Array.isArray(collection.sources)||collection.sources.length>10000) throw new Error('Invalid public collection');
  const sourceIds=new Set();
  for(const source of collection.sources){if(!source||typeof source.id!=='string'||!ID.test(source.id)||sourceIds.has(source.id))throw new Error('Invalid public source');sourceIds.add(source.id);}
  const frameMap=new Map();
  for(const frame of collection.frames) {
    if(!frame||typeof frame.id!=='string'||!ID.test(frame.id)||frameMap.has(frame.id)||!sourceIds.has(frame.sourceId)||
      !/^[a-f0-9]{64}$/.test(frame.sha256)||!/^images\/[a-f0-9]{64}\.(jpg|png)$/.test(frame.path)||frame.path.split('/')[1].split('.')[0]!==frame.sha256||
      !Number.isInteger(frame.width)||!Number.isInteger(frame.height)||frame.width<1||frame.height<1||frame.width*frame.height>20000000||
      !Number.isFinite(frame.timestampSeconds)||frame.timestampSeconds<0||frame.timestampSeconds>86400) throw new Error('Invalid public frame');
    frameMap.set(frame.id,frame);
  }
  const dataDirectory=resolve(options.dataDir??process.env.DATA_DIR??resolve(ROOT,'data'));
  const store=new CommunityStore(dataDirectory);
  const aiStore=new AIContributionStore(dataDirectory),sourceMap=new Map(collection.sources.map(source=>[source.id,source]));
  const rates=new Map();
  function rate(request,type,max) {
    // The reverse proxy overwrites this header; standalone local connections use socket IP.
    const address=options.trustProxy??process.env.TRUST_PROXY==='1'?request.headers['x-forwarded-for']??request.socket.remoteAddress:request.socket.remoteAddress;
    const key=createHash('sha256').update(type+String(address)).digest('hex'), now=Date.now();
    for(const [k,v] of rates) if(v.until<now) rates.delete(k);
    const bucket=rates.get(key)??{count:0,until:now+60000};
    if(++bucket.count>max||rates.size>10000) throw new ReviewError('Please wait a minute before trying again.',429);
    rates.set(key,bucket);
  }
  const json=(res,status,data,headers={})=>{res.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store',...headers});res.end(JSON.stringify(data));};
  function session(request) {
    const token=(request.headers.cookie??'').split(';').map(x=>x.trim()).find(x=>x.startsWith('mushroom_session='))?.slice(17);
    return store.session(token);
  }
  async function body(request) {
    if(!/^application\/json(?:;|$)/i.test(request.headers['content-type']??'')) throw new ReviewError('Use JSON.',415);
    let total=0;const chunks=[];
    for await(const chunk of request){total+=chunk.length;if(total>32768) throw new ReviewError('Review is too large.',413);chunks.push(chunk);}
    try{return JSON.parse(Buffer.concat(chunks).toString());}catch{throw new ReviewError('Invalid review JSON.');}
  }
  const summary=()=>({frames:collection.frames.length,videos:collection.sources.length,...store.summary()});
  const server=createServer(async(request,response)=>{
    const requestId=randomBytes(16).toString('hex');response.setHeader('X-Request-Id',requestId);
    response.setHeader('X-Content-Type-Options','nosniff');response.setHeader('Referrer-Policy','strict-origin-when-cross-origin');
    response.setHeader('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'");
    response.setHeader('Permissions-Policy','camera=(), microphone=(), geolocation=()');
    try {
      const url=new URL(request.url,'http://localhost');
      if(url.pathname===base.slice(0,-1)){response.writeHead(308,{Location:base});response.end();return;}
      if(!url.pathname.startsWith(base)) throw new ReviewError('Page not found.',404);
      const path=url.pathname.slice(base.length);
      let aiImageTask=null,aiImageAgent=null;
      if(path.startsWith('api/ai/')){
        // Bearer authentication is independent of browser sessions/CSRF. No CORS is granted.
        if(request.headers.origin&&request.headers.origin!==origin)throw new ReviewError('Cross-origin browser AI requests are not enabled.',403);
        rate(request,'ai-read',120);
        const authenticate=()=>aiStore.authenticate(/^Bearer ([a-f0-9]{64})$/.exec(request.headers.authorization??'')?.[1]);
        if(path==='api/ai/info'&&request.method==='GET')return json(response,200,{schema:'mushroom-vision.ai-api.v1',requestId,
          participationTerms:AI_TERMS,limits:AI_LIMITS,reviewerKind:'ai',labelAuthority:'ai-unverified',trainingApproved:false,humanVerified:false,
          eligibleFrames:aiEligibleFrames(collection.frames,sourceMap).length,
          schemaUrl:base+'ai-openapi.json',guideUrl:base+'ai-contribute.html',summary:aiStore.summary(),
          observationNotice:'Image delivery is recorded; actual visual inspection is self-attested. AI opinions remain quarantined until curator review.'});
        if(path==='api/ai/agents'&&request.method==='POST'){
          rate(request,'ai-register',4);return json(response,201,{requestId,...aiStore.register(await body(request))});
        }
        if(path==='api/ai/tasks'&&request.method==='GET'){
          if(url.searchParams.has('limit')&&url.searchParams.get('limit')!=='1')throw new ReviewError('Request one task at a time with limit=1.');
          rate(request,'ai-task',30);const task=aiStore.lease(authenticate(),collection.frames,sourceMap,base);
          return json(response,200,{requestId,tasks:task?[task]:[],participationTerms:AI_TERMS});
        }
        if(path==='api/ai/contributions'&&request.method==='POST'){
          rate(request,'ai-submit',30);const agent=authenticate(),input=await body(request),task=aiStore.task(agent,input?.taskId,Date.now(),{allowExpired:true});
          const frame=frameMap.get(task.frameId),result=aiStore.submit(agent,input,request.headers['idempotency-key'],frame,sourceMap.get(frame?.sourceId),requestId);
          return json(response,result.duplicate?200:201,{requestId,...result});
        }
        const receipt=/^api\/ai\/contributions\/([A-Za-z0-9_-]{16,128})$/.exec(path);
        if(receipt&&request.method==='GET')return json(response,200,{requestId,contribution:aiStore.get(authenticate(),receipt[1])});
        if(path==='api/ai/reviews'&&request.method==='GET'){
          const frame=frameMap.get(url.searchParams.get('frameId'));if(!frame)throw new ReviewError('Frame not found.',404);
          return json(response,200,{requestId,frameId:frame.id,imageSha256:frame.sha256,reviews:aiStore.publicReviews(frame.id,frame.sha256),
            notice:'External AI opinions, not human confirmation or approved training labels.'});
        }
        const aiImage=/^api\/ai\/tasks\/([A-Za-z0-9_-]{16,128})\/image$/.exec(path);
        if(aiImage&&request.method==='GET'){
          aiImageAgent=authenticate();aiImageTask=aiStore.task(aiImageAgent,aiImage[1]);
          const frame=frameMap.get(aiImageTask.frameId);
          if(!frame||frame.sha256!==aiImageTask.imageSha256||!aiSourceAllowed(frame,sourceMap.get(frame.sourceId)))throw new ReviewError('Task image or publication rights changed.',409);
        }else throw new ReviewError('AI endpoint not found.',404);
      }
      if(request.method==='POST') {
        if(request.headers.origin!==origin) throw new ReviewError('Please open this page on the Great Lakes Gleaner website.',403);
        rate(request,'write',45);
        if(path==='api/session') {
          await body(request);const existing=session(request);if(existing)return json(response,200,{csrfToken:existing.csrfToken,participationTerms:PARTICIPATION_TERMS});
          rate(request,'session',8);const created=store.createSession();
          return json(response,201,{csrfToken:created.csrfToken,participationTerms:PARTICIPATION_TERMS},{'Set-Cookie':`mushroom_session=${created.token}; Path=${base}; HttpOnly; SameSite=Strict; Max-Age=7776000${origin.startsWith('https:')?'; Secure':''}`});
        }
        const match=/^api\/reviews\/([A-Za-z0-9_-]{1,160})(\/withdraw)?$/.exec(path);
        if(!match||!frameMap.has(match[1])) throw new ReviewError('Frame not found.',404);
        const auth=session(request);if(!auth||!secureEquals(auth.csrfToken,request.headers['x-csrf-token']))throw new ReviewError('Your review session expired. Reload the page.',403);
        const input=await body(request),review=match[2]?store.withdraw(auth.reviewer,frameMap.get(match[1]),input):store.save(auth.reviewer,frameMap.get(match[1]),input);
        return json(response,201,{review,summary:summary()});
      }
      if(!['GET','HEAD'].includes(request.method)) throw new ReviewError('Method not available.',405);
      if(path==='api/health')return json(response,200,{ok:true,service:'gleaner-mushroom-lab',frames:frameMap.size});
      if(path==='api/training-status')return json(response,200,await readTrainingStatus(dataDirectory));
      if(path==='api/research-catalog')return json(response,200,RESEARCH_CATALOG);
      if(path==='api/collection') {
        let ai={frames:[],status:'pending'};
        try{ai=JSON.parse(await readFile(resolve(directory,'ai-suggestions.json'),'utf8'));}catch(error){if(error.code!=='ENOENT')throw error;}
        const suggestions=new Map((ai.frames??[]).filter(row=>row&&typeof row.frameId==='string').map(row=>[row.frameId,row]));
        const priorities=store.priorityFrames();
        const frames=collection.frames.map(frame=>{
          const row=suggestions.get(frame.id),matched=row?.imageSha256===frame.sha256&&row?.sourceId===frame.sourceId;
          const audit=matched?publicAudit(row.audit,frame):null;
          const boxes=matched&&Array.isArray(row.boxes)?row.boxes.slice(0,64).filter(box=>Array.isArray(box?.bbox)&&box.bbox.length===4&&box.bbox.every(Number.isFinite)&&box.bbox[0]>=0&&box.bbox[1]>=0&&box.bbox[2]>box.bbox[0]&&box.bbox[3]>box.bbox[1]&&box.bbox[2]<=frame.width&&box.bbox[3]<=frame.height&&Number.isFinite(box.score)&&box.score>=0&&box.score<=1).map(box=>({bbox:box.bbox,score:box.score,label:String(box.label??'Possible mushroom').slice(0,120)})):[];
          const visualBoxes=audit?.boxes.map(box=>({bbox:box.bbox,score:box.confidence,label:box.groupId,origin:'visual-ai-review'}))??[];
          return {id:frame.id,sourceId:frame.sourceId,kind:frame.kind==='photo'?'photo':'video-frame',timestampSeconds:frame.timestampSeconds,width:frame.width,height:frame.height,reviewPriority:Math.max(priorities.get(frame.id)??0,boxes.some(box=>box.bbox[2]-box.bbox[0]<64)?2:0,audit?.status==='uncertain'?3:0),
            imageUrl:base+'images/'+frame.id+'?v='+frame.sha256,ai:matched?{status:row.status,model:typeof row.model==='string'?row.model.slice(0,160):row.model?.name,boxes:[...visualBoxes,...boxes].slice(0,64),audit}:{status:'pending',boxes:[],audit:null}};
        });
        const restrictedRecords=(collection.restrictedRecords??[]).filter(row=>/^image-[a-f0-9]{64}$/.test(row?.id)).slice(0,10000).map(row=>({id:row.id,aiReviewed:row.aiReviewed===true,reason:'publication-rights-unresolved'}));
        const reviewed=frames.filter(frame=>frame.ai.audit),original=reviewed.filter(frame=>frame.ai.audit.observation.mode==='original-image');
        return json(response,200,{sources:collection.sources.map(({id,title,attribution,license,licenseUrl,sourceUrl,changes})=>({id,title,attribution,license,licenseUrl,sourceUrl,changes})),frames,summary:summary(),restrictedRecords,
          coverage:{totalRecords:frames.length+restrictedRecords.length,publicImages:frames.length,rightsLimited:restrictedRecords.length,visualAiReviewed:reviewed.length,originalAiReviewed:original.length,previewAiReviewed:reviewed.length-original.length},
          participationTerms:PARTICIPATION_TERMS,aiRun:ai.run??{status:ai.status??'pending'},notice:'Community labels await curator approval before training. Candidate results are evaluated separately; species and edibility are not verified here.'});
      }
      const reviewRoute=/^api\/reviews\/([A-Za-z0-9_-]{1,160})$/.exec(path);
      if(reviewRoute){if(!frameMap.has(reviewRoute[1])) throw new ReviewError('Frame not found.',404);const auth=session(request);return json(response,200,{review:auth?store.get(auth.reviewer,reviewRoute[1],frameMap.get(reviewRoute[1]).sha256):null});}
      const imageRoute=aiImageTask?[null,aiImageTask.frameId]:/^images\/([A-Za-z0-9_-]{1,160})$/.exec(path);
      let file,mime,expectedHash;
      if(imageRoute) {
        const frame=frameMap.get(imageRoute[1]);if(!frame)throw new ReviewError('Frame not found.',404);
        const root=await realpath(directory);file=await realpath(resolve(directory,frame.path));if(!file.startsWith(root+sep))throw new ReviewError('Frame not found.',404);
        if(url.searchParams.has('v')&&url.searchParams.get('v')!==frame.sha256)throw new ReviewError('This image version is no longer in the collection.',409);
        expectedHash=frame.sha256;mime=MIME[extname(file)];response.setHeader('Cache-Control',aiImageTask?'private, no-store':'public, max-age=3600');
      } else {
        const name=path===''?'index.html':path;
        const pixelEvidence=/^pixel-view-[A-Za-z0-9_-]{1,180}--(?:64px|32px|16px|6p9px|3p45px|1p38px|0p69px)\.png$/.test(name);
        if(!pixelEvidence&&!['index.html','styles.css','app.js','research-taxonomy.js','research-taxon-editor.js','review-evidence.js','ai-contribute.html','ai-contribute.css','llms.txt','ai-openapi.json','testing-round4.html','testing-round4.js','testing-round4.css','testing-round4.json','testing-pixels.html','testing-pixels.js','testing-pixels.css','testing-pixels.json'].includes(name))throw new ReviewError('Page not found.',404);
        file=name==='ai-openapi.json'?resolve(ROOT,name):resolve(ROOT,'public',name);mime=MIME[extname(file)];response.setHeader('Cache-Control','no-cache');
      }
      const size=(await stat(file)).size;if(size>8*1024*1024)throw new ReviewError('File is too large.',413);
      const bytes=await readFile(file);
      if(expectedHash&&createHash('sha256').update(bytes).digest('hex')!==expectedHash)throw new ReviewError('The original image failed its integrity check. Please report this frame.',409);
      if(aiImageTask){aiStore.markDelivered(aiImageAgent,aiImageTask.taskId,expectedHash);response.setHeader('X-Image-Sha256',expectedHash);}
      response.writeHead(200,{'Content-Type':mime,'Content-Length':bytes.length});response.end(request.method==='HEAD'?undefined:bytes);
    }catch(error){if(!response.headersSent)json(response,error.status??500,{requestId,error:error.status?error.message:'The review service is temporarily unavailable.'},error.status===429?{'Retry-After':'60'}:{});else response.end();}
  });
  server.requestTimeout=15000;server.headersTimeout=10000;
  return {server,store,aiStore,base,close:()=>new Promise(resolveClose=>server.close(()=>{store.close();aiStore.close();resolveClose();}))};
}
if(process.argv[1]&&resolve(process.argv[1])===fileURLToPath(import.meta.url)) {
  const app=await createCommunityServer();app.server.listen(Number(process.env.PORT??8791),process.env.HOST??'127.0.0.1',()=>console.log('Mushroom community review service ready'));
  for(const signal of ['SIGINT','SIGTERM'])process.on(signal,()=>app.close().then(()=>process.exit(0)));
}
