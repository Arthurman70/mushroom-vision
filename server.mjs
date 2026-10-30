import http from 'node:http';
import { readFile, realpath, stat } from 'node:fs/promises';
import { resolve, dirname, extname, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { randomUUID, timingSafeEqual } from 'node:crypto';
import { ApiError, shortText, validateMap, validateObservation } from './core/validation.mjs';
import { planWalk } from './core/planner.mjs';
import { StateStore, demoState, emptyState, deriveHazards, exportLocalGeoJSON } from './core/state.mjs';
import { FindsStore, normalizePrediction } from './core/finds.mjs';
import { FindsClassifier } from './core/finds-worker.mjs';
import { MAX_IMAGE_BODY, validateImage } from './core/finds-images.mjs';
import { TestbenchWorker } from './core/testbench-worker.mjs';
import { TestbenchStore } from './core/testbench-store.mjs';
import { VideoCollection } from './core/video-collection.mjs';
import { VideoGradesStore } from './core/video-grades.mjs';
import { videoDetectionMetrics } from './core/video-detection-metrics.mjs';
import { LiveMissionStore } from './core/live-mission.mjs';
import { LiveStation } from './core/live-station.mjs';
import { TrainingPipeline } from './core/training-pipeline.mjs';
import { LearningLoop } from './core/learning-loop.mjs';
import { CommunityLearningBridge } from './core/community-learning-bridge.mjs';
import { CandidateComparisons } from './core/candidate-comparison.mjs';
import { CommunitySuggestionsRefresh } from './core/community-suggestions-refresh.mjs';
import { ReviewAssistance } from './core/review-assistance.mjs';
import { readVideoLearningReport } from './core/video-learning-report.mjs';
import { readVideoDistanceReport } from './core/video-distance-report.mjs';
import { validateCaptureAck } from './flight/live-executor.mjs';
import { DeviceComputeHub, handleDeviceCompute } from './core/device-compute.mjs';
import { RESEARCH_CATALOG } from './community/public/research-taxonomy.js';

const ROOT=dirname(fileURLToPath(import.meta.url));
const LOOPBACK=new Set(['127.0.0.1','localhost','::1']);
const MIME={'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8',
  '.mjs':'text/javascript; charset=utf-8','.json':'application/json; charset=utf-8','.webmanifest':'application/manifest+json',
  '.svg':'image/svg+xml','.png':'image/png','.jpg':'image/jpeg','.ico':'image/x-icon','.woff2':'font/woff2'};
const MAX_BODY=1024*1024;

function json(response,status,data,headers={}) {
  response.writeHead(status,{'Content-Type':'application/json; charset=utf-8','Cache-Control':'no-store',...headers});
  response.end(JSON.stringify(data));
}
async function body(request,maxBody=MAX_BODY) {
  if(!/^application\/json(?:;|$)/i.test(request.headers['content-type']??'')) {
    throw new ApiError('Use Content-Type: application/json.',415,'CONTENT_TYPE_REQUIRED');
  }
  const declared=Number(request.headers['content-length']??0);
  if(declared>maxBody) throw new ApiError('Request body exceeds the allowed size.',413,'BODY_TOO_LARGE');
  const chunks=[]; let size=0;
  for await(const chunk of request) {
    size+=chunk.length;
    if(size>maxBody) throw new ApiError('Request body exceeds the allowed size.',413,'BODY_TOO_LARGE');
    chunks.push(chunk);
  }
  let value;
  try { value=JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch { throw new ApiError('Request body is not valid JSON.'); }
  if(!value || typeof value!=='object' || Array.isArray(value)) throw new ApiError('Request JSON must be an object.');
  return value;
}
function tokenMatches(actual,expected) {
  if(typeof actual!=='string') return false;
  const a=Buffer.from(actual), b=Buffer.from(expected);
  return a.length===b.length && timingSafeEqual(a,b);
}

export async function createApp(options={}) {
  const host=options.host??process.env.MYCOSCOUT_HOST??'127.0.0.1';
  const token=options.token??process.env.MYCOSCOUT_TOKEN??'';
  if(!LOOPBACK.has(host) && token.length<24) throw new Error('LAN binding requires MYCOSCOUT_TOKEN with at least 24 characters.');
  const publicDir=resolve(options.publicDir??resolve(ROOT,'public'));
  const photoReviewFile=resolve(options.photoReviewFile??resolve(ROOT,'data','facebook-michigan-fungus-hunters','photo-review.html'));
  const trumpetReviewFile=resolve(options.trumpetReviewFile??resolve(ROOT,'data','black-trumpet-discovery-v2','photo-review.html'));
  const allowedOrigins=new Set(['https://appassets.androidplatform.net']);
  const configured=options.allowedOrigin??process.env.MYCOSCOUT_ALLOWED_ORIGIN;
  if(configured) {
    const parsed=new URL(configured);
    if(!['https:','http:'].includes(parsed.protocol) || parsed.origin!==configured) throw new Error('MYCOSCOUT_ALLOWED_ORIGIN must be one exact HTTP(S) origin without a trailing slash.');
    allowedOrigins.add(configured);
  }
  const stateFile=options.stateFile??process.env.MYCOSCOUT_STATE_FILE??resolve(ROOT,'data','state.json');
  const store=await new StateStore(stateFile).initialize();
  const deviceProfiles=JSON.parse(await readFile(resolve(ROOT,'config/device-profiles.json'),'utf8')).profiles;
  const deviceCompute=options.deviceCompute??new DeviceComputeHub({profiles:deviceProfiles});
  const finds=await new FindsStore(options.findsDir??process.env.MYCOSCOUT_FINDS_DIR??resolve(dirname(stateFile),'finds'),
    options.catalogFile??resolve(ROOT,'config','michigan-mushrooms-v1.json')).initialize();
  const classifier=options.classifier??new FindsClassifier(ROOT);
  const researchClassifier=options.researchClassifier??new FindsClassifier(ROOT,{checkpoint:'vision/checkpoints/groups-expanded-v1.json'});
  const testbenchWorker=options.testbenchWorker??new TestbenchWorker(ROOT);
  const testbench=await new TestbenchStore(options.testbenchDir??process.env.MYCOSCOUT_TESTBENCH_DIR??resolve(dirname(stateFile),'testbench'),
    ROOT,testbenchWorker,{...(options.testbenchSample?{sample:options.testbenchSample}:{}),catalogFile:options.catalogFile??resolve(ROOT,'config','michigan-mushrooms-v1.json')}).initialize();
  const videos=new VideoCollection(options.videoCollectionDir??resolve(ROOT,'data','video-discovery-v1'));
  const videoGrades=await new VideoGradesStore(options.videoGradesDir??resolve(dirname(stateFile),'video-grades')).initialize();
  const videoCollectionId=options.videoCollectionId??'video-discovery-v1';
  const liveDir=options.liveDir??resolve(dirname(stateFile),'live');
  const learningDir=options.learningDir??resolve(dirname(stateFile),'learning');
  const mission=await new LiveMissionStore(liveDir).initialize();
  const live=await new LiveStation(ROOT,liveDir,mission,{python:options.livePython??process.env.MYCOSCOUT_PYTHON,
    collectionDir:options.videoCollectionDir,testbenchDb:resolve(testbench.directory,'testbench.sqlite'),learningDir}).initialize();
  const communityInboxDir=options.communityInboxDir??resolve(ROOT,'data/community-inbox-v1');
  const learning=await new TrainingPipeline(ROOT,learningDir,{testbench,videos,videoGrades,collectionId:videoCollectionId,communityInboxFile:resolve(communityInboxDir,'reviews.json')}).initialize();
  const communityBridge=await new CommunityLearningBridge(ROOT,resolve(learningDir,'community-sync'),{...options.communityBridgeOptions,inboxDir:communityInboxDir}).initialize();
  const comparisons=await new CandidateComparisons(resolve(learningDir,'comparisons'),{learning,beforeWork:()=>communityBridge.beforeWork()}).initialize();
  learning.releaseCheck=job=>comparisons.releaseCheck(job);
  const learningLoop=await new LearningLoop(resolve(learningDir,'automatic'),{learning,compareCandidate:job=>comparisons.compare(job),beforeWork:()=>communityBridge.beforeWork()}).initialize();
  const suggestionRefresh=await new CommunitySuggestionsRefresh(ROOT,resolve(learningDir,'suggestion-refresh'),{learning,bridge:communityBridge}).initialize();
  let releaseRequest=null,backgroundStarted=false;
  const backgroundTimers=[];
  const backgroundError=error=>console.error('Learning worker:',error.message);
  const publishProgress=()=>{
    const summary={...learningLoop.publicSummary(),updatedAt:new Date().toISOString()};
    if(communityBridge.snapshot().error){summary.worker.state='waiting_sync';summary.approvedFrames=null;summary.newFrames=null;summary.newSessions=null;}
    return communityBridge.publish(summary).catch(backgroundError);
  };
  const startBackground=()=>{
    if(backgroundStarted||options.backgroundLearning===false)return;backgroundStarted=true;
    learningLoop.start(15000);
    const sync=()=>communityBridge.sync().catch(backgroundError);
    const refreshSuggestions=()=>{if(!learningLoop.snapshot().active)suggestionRefresh.tick().catch(backgroundError);};
    for(const [fn,interval] of [[sync,300000],[publishProgress,60000],[refreshSuggestions,60000]]){const timer=setInterval(fn,interval);timer.unref?.();backgroundTimers.push(timer);}
    sync();if(learningLoop.snapshot().config.enabled)learningLoop.tick().then(publishProgress).catch(backgroundError);
  };
  const reviewAssistance=new ReviewAssistance(ROOT,videos);
  let activeImageRequests=0;
  const server=http.createServer(async(request,response)=>{
    let imageRequest=false;
    response.setHeader('X-Content-Type-Options','nosniff');
    response.setHeader('Referrer-Policy','same-origin');
    response.setHeader('Cross-Origin-Resource-Policy','same-site');
    try {
      const url=new URL(request.url,'http://localhost');
      const origin=request.headers.origin;
      if(LOOPBACK.has(host)) {
        let requestHost;
        try { requestHost=new URL('http://'+request.headers.host).hostname.replace(/^\[|\]$/g,''); }
        catch { throw new ApiError('Invalid Host header.',400); }
        if(!LOOPBACK.has(requestHost)) throw new ApiError('Host is not allowed.',403,'HOST_NOT_ALLOWED');
      }
      if(origin) {
        const sameOrigin=origin==='http://'+request.headers.host || origin==='https://'+request.headers.host;
        if(!sameOrigin && !allowedOrigins.has(origin)) throw new ApiError('Origin is not allowed.',403,'ORIGIN_NOT_ALLOWED');
        response.setHeader('Access-Control-Allow-Origin',origin);
        response.setHeader('Vary','Origin');
      }
      if(request.method==='OPTIONS' && url.pathname.startsWith('/api/')) {
        response.writeHead(204,{'Access-Control-Allow-Methods':'GET, POST, OPTIONS',
          'Access-Control-Allow-Headers':'Authorization, Content-Type','Access-Control-Max-Age':'600'});
        return response.end();
      }
      if(url.pathname.startsWith('/api/')) {
        // Restricted workers authenticate independently and cannot use station-owner routes.
        if(url.pathname==='/api/device-compute/worker'&&request.method==='POST') {
          if(await handleDeviceCompute({hub:deviceCompute,request,response,url,readBody:body,sendJson:json}))return;
        }
        if(token && !tokenMatches(request.headers.authorization,'Bearer '+token)) throw new ApiError('A valid Bearer token is required.',401,'UNAUTHORIZED');
        if(await handleDeviceCompute({hub:deviceCompute,request,response,url,readBody:body,sendJson:json}))return;
        if(request.method==='GET'&&url.pathname==='/api/research-catalog')return json(response,200,RESEARCH_CATALOG);
        if(request.method==='GET'&&url.pathname==='/api/vision/research-status') {
          const summary={available:false,modelVersion:null,supportedClasses:[],humanVerificationRequired:true,
            notice:'Experimental photo/crop category guesses, not verified species or woodland detection. The default reviewed detector is unchanged.'};
          try {
            const saved=JSON.parse(await readFile(resolve(ROOT,'vision/checkpoints/groups-expanded-v1.json'),'utf8'));
            if(saved.schema==='mycoscout.group-classifier.v1'&&Array.isArray(saved.classes)) {
              summary.available=true;summary.modelVersion=saved.evidenceVersion??'groups-expanded-v1';
              summary.supportedClasses=saved.classes.filter(c=>c.supported).map(({classId,label,groupId})=>({classId,label,groupId}));
            }
          } catch(error) { if(error.code!=='ENOENT')summary.notice='The optional research checkpoint could not be read.'; }
          return json(response,200,summary);
        }
        if(request.method==='POST'&&url.pathname==='/api/vision/research-classify') {
          if(activeImageRequests>=2)throw new ApiError('Image processing is busy. Retry shortly.',429,'IMAGE_REQUEST_BUSY');
          activeImageRequests++;imageRequest=true;
          const input=await body(request,MAX_IMAGE_BODY);
          const image=validateImage(input);
          return json(response,200,{prediction:await researchClassifier.classify(image),experimental:true,defaultModelChanged:false});
        }
        if(request.method==='POST' && (['/api/finds','/api/vision/classify','/api/testbench/runs','/api/testbench/sample','/api/live/frames','/api/live/captures'].includes(url.pathname)||/^\/api\/testbench\/videos\/frames\/[A-Za-z0-9_-]{1,160}\/run$/.test(url.pathname))) {
          if(activeImageRequests>=2) throw new ApiError('Image processing is busy. Retry shortly.',429,'IMAGE_REQUEST_BUSY');
          activeImageRequests++; imageRequest=true;
        }
        if(request.method==='GET' && url.pathname==='/api/health') return json(response,200,{ok:true,application:'MycoScout',version:'0.9.0-alpha.1',actuation:false});
        if(request.method==='GET' && url.pathname==='/api/live/state')return json(response,200,live.snapshot());
        if(request.method==='GET' && url.pathname==='/api/learning/status'){const loop=learningLoop.snapshot();return json(response,200,{...await learning.status(),automaticTraining:loop.config.enabled&&!loop.paused,automaticState:loop.workerState});}
        if(request.method==='GET' && url.pathname==='/api/learning/loop')return json(response,200,{loop:learningLoop.snapshot(),community:communityBridge.snapshot(),comparisons:comparisons.status(),releaseRequest,suggestions:suggestionRefresh.snapshot(),activeModel:learning.active?{jobId:learning.active.jobId,activatedAt:learning.active.activatedAt}:null});
        if(request.method==='GET' && url.pathname==='/api/review-assistance/status')return json(response,200,await reviewAssistance.status());
        if(request.method==='GET' && url.pathname==='/api/research/video-learning')return json(response,200,await readVideoLearningReport(ROOT));
        if(request.method==='GET' && url.pathname==='/api/research/video-distance')return json(response,200,await readVideoDistanceReport(ROOT));
        const assistanceRoute=/^\/api\/review-assistance\/frames\/([A-Za-z0-9_-]{1,160})$/.exec(url.pathname);
        if(request.method==='GET'&&assistanceRoute)return json(response,200,await reviewAssistance.frame(assistanceRoute[1]));
        const learningDatasetRoute=/^\/api\/learning\/datasets\/(dataset-[a-f0-9]{24})\/export$/.exec(url.pathname);
        if(request.method==='GET'&&learningDatasetRoute)return json(response,200,learning.manifest(learningDatasetRoute[1]),{'Content-Disposition':'attachment; filename="mycoscout-detection-dataset.json"'});
        const liveImageRoute=/^\/api\/live\/images\/([a-f0-9]{64})$/.exec(url.pathname);
        if(request.method==='GET'&&liveImageRoute){const image=await live.image(liveImageRoute[1]);response.writeHead(200,{'Content-Type':image.mimeType,'Cache-Control':'no-store','Content-Length':image.bytes.length,'Content-Security-Policy':"default-src 'none'; sandbox"});return response.end(image.bytes);}
        const liveTaskRoute=/^\/api\/live\/tasks\/([a-f0-9-]{36})$/.exec(url.pathname);
        const liveReviewRoute=/^\/api\/live\/landmarks\/([A-Za-z0-9_.:-]{1,160})\/review$/.exec(url.pathname);
        const learningJobRoute=/^\/api\/learning\/jobs\/(job-[a-f0-9-]{36})(?:\/(cancel))?$/.exec(url.pathname);
        if(request.method==='GET'&&learningJobRoute&&!learningJobRoute[2])return json(response,200,learning.job(learningJobRoute[1]));
        if(request.method==='GET' && url.pathname==='/api/testbench/status') return json(response,200,await testbench.status());
        if(request.method==='GET' && url.pathname==='/api/testbench/runs') return json(response,200,testbench.list(url.searchParams.get('limit'),url.searchParams.get('offset')));
        if(request.method==='GET' && url.pathname==='/api/testbench/sample') return json(response,200,await testbench.sampleInfo());
        const videoGradeRoute=/^\/api\/video-grades\/frames\/([A-Za-z0-9_-]{1,160})$/.exec(url.pathname);
        if(request.method==='GET'&&url.pathname==='/api/video-grades'){
          const frames=(await videos.read()).frames,snapshot=videoGrades.snapshot(videoCollectionId,frames);
          snapshot.detectorEvaluation=videoDetectionMetrics(frames,snapshot,testbench.completedVideoRuns());return json(response,200,snapshot);
        }
        if(request.method==='GET'&&videoGradeRoute){
          const {metadata}=await videos.frame(videoGradeRoute[1]);return json(response,200,videoGrades.get(videoCollectionId,metadata,{history:true}));
        }
        if(request.method==='GET' && url.pathname==='/api/testbench/videos') return json(response,200,await videos.list({
          offset:url.searchParams.get('offset')??0,limit:url.searchParams.get('limit')??12,sourceId:url.searchParams.get('sourceId')??'',sort:url.searchParams.get('sort')??'time'}));
        const videoRoute=/^\/api\/testbench\/videos\/frames\/([A-Za-z0-9_-]{1,160})\/(image|run)$/.exec(url.pathname);
        if(request.method==='GET'&&videoRoute?.[2]==='image'){
          const {image}=await videos.frame(videoRoute[1]);response.writeHead(200,{'Content-Type':image.mimeType,'Cache-Control':'no-store','Content-Length':image.bytes.length,
            'Content-Security-Policy':"default-src 'none'; sandbox"});return response.end(image.bytes);
        }
        if(request.method==='GET' && url.pathname==='/api/testbench/export') return json(response,200,await testbench.exportDataset(),{'Content-Disposition':'attachment; filename="mycoscout-testbench-dataset.json"'});
        const testbenchRoute=/^\/api\/testbench\/runs\/([a-f0-9-]{36})(?:\/(image|annotations))?$/.exec(url.pathname);
        if(request.method==='GET' && testbenchRoute){
          if(testbenchRoute[2]==='image'){
            const image=await testbench.image(testbenchRoute[1]);response.writeHead(200,{'Content-Type':image.mimeType,'Cache-Control':'no-store','Content-Length':image.bytes.length,
              'Content-Security-Policy':"default-src 'none'; sandbox"});return response.end(image.bytes);
          }
          if(!testbenchRoute[2])return json(response,200,{run:testbench.get(testbenchRoute[1])});
        }
        if(request.method==='GET' && url.pathname==='/api/catalog') return json(response,200,await finds.catalog());
        if(request.method==='GET' && url.pathname==='/api/finds') return json(response,200,finds.list(url.searchParams.get('since'),url.searchParams.get('limit')));
        if(request.method==='GET' && url.pathname==='/api/events') return json(response,200,finds.events(url.searchParams.get('since'),url.searchParams.get('limit')));
        if(request.method==='GET' && url.pathname==='/api/training/export') return json(response,200,await finds.exportTraining(),{'Content-Disposition':'attachment; filename="mycoscout-reviewed-training.json"'});
        if(request.method==='GET' && url.pathname==='/api/vision/status') {
          let checkpoint=null;
          try { const saved=JSON.parse(await readFile(resolve(ROOT,'vision','checkpoints','groups-v1.json'),'utf8'));
            checkpoint={available:true,schema:saved.schema??null,evidenceVersion:saved.evidenceVersion??null,
              classCount:Array.isArray(saved.classes)?saved.classes.length:Array.isArray(saved.classIds)?saved.classIds.length:null};
          } catch { checkpoint={available:false}; }
          return json(response,200,{schema:'mycoscout.vision-status.v1',workerStarted:!!classifier.process,workerLoaded:classifier.loaded===true,busy:!!classifier.pending,liveCameraConnected:false,
            checkpoint,humanVerificationRequired:true,speciesConfirmed:false,salesApproval:false,flightGateEnabled:false});
        }
        const findRoute=/^\/api\/finds\/([a-f0-9-]{36})(?:\/(image|reviews))?$/.exec(url.pathname);
        if(request.method==='GET' && findRoute) {
          if(findRoute[2]==='image') {
            const photo=await finds.image(findRoute[1]);
            response.writeHead(200,{'Content-Type':photo.mimeType,'Cache-Control':'no-store','Content-Length':photo.bytes.length,
              'Content-Security-Policy':"default-src 'none'; sandbox"});
            return response.end(photo.bytes);
          }
          if(!findRoute[2]) return json(response,200,{find:finds.get(findRoute[1])});
        }
        if(request.method==='GET' && ['/api/research/photo-review','/api/research/trumpet-review'].includes(url.pathname)) {
          if(!LOOPBACK.has(host)) throw new ApiError('The private photo review is available on the local computer only.',403,'LOCAL_REVIEW_ONLY');
          let content;
          try { content=await readFile(url.pathname==='/api/research/trumpet-review'?trumpetReviewFile:photoReviewFile); }
          catch(error) { if(error.code==='ENOENT') throw new ApiError('Generate the local photo review before opening it.',404,'REVIEW_NOT_GENERATED'); throw error; }
          response.writeHead(200,{'Content-Type':'text/html; charset=utf-8','Cache-Control':'no-store',
            'Content-Security-Policy':"default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'",
            'Content-Length':content.length});
          return response.end(content);
        }
        if(request.method==='GET' && url.pathname==='/api/state') return json(response,200,store.snapshot());
        if(request.method==='GET' && url.pathname==='/api/export') {
          const format=url.searchParams.get('format')??'json';
          if(format==='geojson') exportLocalGeoJSON();
          if(format!=='json') throw new ApiError('Export format must be json or geojson.');
          return json(response,200,store.snapshot(),{'Content-Disposition':'attachment; filename="mycoscout-survey.json"'});
        }
        if(request.method!=='POST') throw new ApiError('API route not found.',404,'NOT_FOUND');
        const input=await body(request,['/api/finds','/api/vision/classify','/api/testbench/runs','/api/live/frames','/api/live/captures','/api/live/navigation'].includes(url.pathname)?MAX_IMAGE_BODY:MAX_BODY);
        if(url.pathname==='/api/live/session'){
          if(live.child)throw new ApiError('Stop the replay before changing sessions.',409,'REPLAY_ACTIVE');
          return json(response,201,mission.startSession(input));
        }
        if(url.pathname==='/api/live/frames')return json(response,201,await live.ingest(input));
        if(url.pathname==='/api/live/navigation')return json(response,201,mission.updateNavigation(input));
        if(url.pathname==='/api/live/captures'){
          const state=mission.snapshot(),ack=input.ack,image=validateImage(input);
          if(state.mode!=='live'||ack?.synthetic!==false)throw new ApiError('This endpoint accepts saved real camera captures in a live session.',409,'LIVE_CAPTURE_REQUIRED');
          const task=state.tasks.find(t=>t.id===ack?.taskId);
          if(!task||task.status!=='running'||!task.captureRequest)throw new ApiError('No active task is waiting for this capture.',409,'CAPTURE_TASK_REQUIRED');
          if(image.sha256!==ack.photoSha256||image.width!==ack.width||image.height!==ack.height)throw new ApiError('The original photo bytes must match the capture acknowledgment.',409,'CAPTURE_IMAGE_MISMATCH');
          const result=validateCaptureAck(ack,task,state,Date.now());
          if(!result.accepted)throw new ApiError('Capture rejected: '+result.reason,409,'CAPTURE_REJECTED');
          const find=await finds.create({imageBase64:input.imageBase64,mimeType:input.mimeType,source:'live_mission_capture',sessionId:state.sessionId,
            captureTimestamp:ack.capturedMs,rights:input.rights??{trainingOptIn:false},
            metadata:{landmarkId:task.landmarkId,taskId:task.id,requestId:ack.requestId,mapRevision:state.mapRevision,
              positionEnuM:ack.positionEnuM,originalPhotoSha256:image.sha256,classificationEligible:true},
            camera:{nativeUncropped:true,width:image.width,height:image.height,yawRad:ack.yawRad,pitchRad:ack.pitchRad}});
          // File bytes are now persisted in the review inbox before the task sees its acknowledgement.
          let outcome;
          try{outcome=mission.tick({captureAck:{...ack,photoRef:'/api/finds/'+find.id+'/image'}});}
          catch(error){
            if(error instanceof ApiError)return json(response,error.status,{error:error.message,code:error.code,savedFindId:find.id});
            throw error;
          }
          if(outcome.action.type!=='completed'||outcome.action.taskId!==ack.taskId||outcome.state.sessionId!==ack.sessionId||outcome.state.tasks.find(t=>t.id===ack.taskId)?.captureRequest?.requestId!==ack.requestId)
            return json(response,409,{error:'Photo saved for review, but the capture task could not complete: '+(outcome.action.reason??outcome.action.type),code:'CAPTURE_NOT_ACCEPTED',savedFindId:find.id});
          return json(response,201,{find,...outcome});
        }
        if(url.pathname==='/api/live/tasks')return json(response,201,mission.enqueue(input));
        if(liveTaskRoute)return json(response,200,mission.updateTask(liveTaskRoute[1],input));
        if(liveReviewRoute)return json(response,201,mission.review(liveReviewRoute[1],input));
        if(url.pathname==='/api/live/tick'){
          if(input.captureAck)throw new ApiError('Capture acknowledgements require a saved photo from the capture adapter.',409,'CAPTURE_IMAGE_REQUIRED');
          return json(response,200,mission.tick({baseRevision:input.baseRevision,baseControlRevision:input.baseControlRevision,sessionId:input.sessionId}));
        }
        if(url.pathname==='/api/live/replay/start')return json(response,202,await live.startReplay(input));
        if(url.pathname==='/api/live/replay/stop')return json(response,200,{replay:live.stopReplay()});
        if(url.pathname==='/api/live/simulation/start'){
          if(live.child)throw new ApiError('Stop the replay before starting a simulation.',409,'REPLAY_ACTIVE');
          return json(response,201,mission.seedSimulation(input));
        }
        if(url.pathname==='/api/live/simulation/step')return json(response,200,mission.stepSimulation(input));
        if(url.pathname==='/api/learning/loop/config')return json(response,200,await learningLoop.configure(input));
        if(url.pathname==='/api/learning/loop/pause')return json(response,200,await learningLoop.pause());
        if(url.pathname==='/api/learning/loop/resume')return json(response,200,await learningLoop.resume());
        if(url.pathname==='/api/learning/loop/cancel')return json(response,200,await learningLoop.cancel());
        if(url.pathname==='/api/learning/loop/tick'){learningLoop.tick().then(publishProgress).catch(backgroundError);return json(response,202,{accepted:true});}
        if(url.pathname==='/api/learning/community-sync'){
          if(input.enabled!==undefined)await communityBridge.configure({enabled:input.enabled});
          communityBridge.sync({force:true}).then(publishProgress).catch(backgroundError);return json(response,202,communityBridge.snapshot());
        }
        if(url.pathname==='/api/learning/release-check'){
          if(releaseRequest?.status==='running')throw new ApiError('A release comparison is already running.',409,'LEARNING_BUSY');
          learning.checkIdle();comparisons.context(input.jobId,'test');
          releaseRequest={jobId:input.jobId,status:'running',startedAt:new Date().toISOString()};
          comparisons.compare(input.jobId,{split:'test'}).then(result=>{releaseRequest={...releaseRequest,status:'completed',result};}).catch(error=>{releaseRequest={...releaseRequest,status:'failed',error:error.message};});
          return json(response,202,releaseRequest);
        }
        if(url.pathname==='/api/learning/export')return json(response,201,await learning.export(input));
        if(url.pathname==='/api/learning/jobs')return json(response,202,await learning.startJob(input));
        if(learningJobRoute?.[2]==='cancel')return json(response,200,learning.cancel(learningJobRoute[1]));
        if(url.pathname==='/api/learning/activate')return json(response,200,await learning.activate(input));
        if(url.pathname==='/api/video-grades/tracks'){
          const collection=await videos.read(),source=collection.sources.find(source=>source.id===input.sourceId);
          if(!source)throw new ApiError('Video source not found.',404,'NOT_FOUND');
          return json(response,201,{track:videoGrades.createTrack(videoCollectionId,source,input)});
        }
        if(videoGradeRoute){const {metadata}=await videos.frame(videoGradeRoute[1]);return json(response,201,videoGrades.save(videoCollectionId,metadata,input));}
        if(url.pathname==='/api/testbench/runs')return json(response,202,{run:await testbench.create(input)});
        if(url.pathname==='/api/testbench/sample')return json(response,202,{run:await testbench.createSample(input)});
        if(videoRoute?.[2]==='run'){
          const {image,metadata}=await videos.frame(videoRoute[1]);
          return json(response,202,{run:await testbench.create({imageBase64:image.bytes.toString('base64'),settings:input.settings,
            name:metadata.source.title.slice(0,120)+' · '+metadata.timestampSeconds.toFixed(2)+' s'},metadata)});
        }
        if(testbenchRoute?.[2]==='annotations')return json(response,201,testbench.annotate(testbenchRoute[1],input));
        if(url.pathname==='/api/finds') return json(response,201,{find:await finds.create(input)});
        if(findRoute?.[2]==='reviews') return json(response,201,await finds.review(findRoute[1],input));
        if(url.pathname==='/api/vision/classify') {
          let image,find=null;
          if(input.findId!==undefined) {
            find=finds.get(input.findId);
            if(find.metadata?.cueType==='color_salience'||find.metadata?.classificationEligible===false)
              throw new ApiError('This color-only search cue needs a closer original photo before mushroom classification.',409,'CLOSER_VIEW_REQUIRED');
            image=await finds.image(find.id);
          }
          else image=validateImage(input);
          const raw=await classifier.classify(image,{findId:find?.id,capturedAt:find?.capturedAt});
          if(find) return json(response,200,await finds.prediction(find.id,raw));
          return json(response,200,{result:normalizePrediction(raw,await finds.catalog()),find:null});
        }
        if(url.pathname==='/api/reset') return json(response,200,await store.mutate(state=>{Object.assign(state,demoState());}));
        if(url.pathname==='/api/mode') {
          if(input.mode!=='survey') throw new ApiError('Only mode:"survey" is supported here. Use /api/reset for the demo.');
          return json(response,200,await store.mutate(state=>{Object.assign(state,emptyState());}));
        }
        if(url.pathname==='/api/map') {
          const map=validateMap(input);
          return json(response,200,await store.mutate(state=>{
            Object.assign(state,emptyState(),map);
            state.hazards=deriveHazards(state.grid,'survey');
          }));
        }
        if(url.pathname==='/api/observations') {
          const observation=await store.mutate(state=>{
            if(!state.grid) throw new ApiError('Import a map before adding an observation.');
            if(state.observations.length>=10000) throw new ApiError('Observation limit reached. Export and begin a new survey.',409);
            const observation={id:randomUUID(),...validateObservation(input,state.grid)};
            state.observations.push(observation); return observation;
          });
          return json(response,201,{observation});
        }
        if(url.pathname==='/api/route') {
          if(input.mode==='flight') throw new ApiError('Autonomous flight is not implemented. This server has no arming, takeoff, waypoint, velocity or other drone control endpoint.',409,'FLIGHT_NOT_IMPLEMENTED');
          if(input.mode!=='walk') throw new ApiError('Route mode must be "walk" or "flight".');
          const goalId=shortText(input.goalId,'goalId',undefined,80);
          const route=await store.mutate(state=>{
            if(!state.grid) throw new ApiError('Import a map before requesting a route.');
            if(!state.groundTraversabilityVerified) throw new ApiError('Walking routes are disabled: occupancy-free space does not establish walkable ground. A reviewed terrain map with groundTraversabilityVerified:true is required.',409,'GROUND_NOT_VERIFIED');
            const goal=state.observations.find(o=>o.id===goalId);
            if(!goal) throw new ApiError('Observation was not found.',404,'NOT_FOUND');
            const start=input.start??(state.mode==='simulation'?state.telemetry.position:null);
            if(!start) throw new ApiError('A survey walking route needs an explicit map-frame start:{x,y}; phone and drone positions are not connected.');
            state.route={...planWalk(state.grid,start,goal,{goalId,clearance:input.clearance??0.5}),createdAt:new Date().toISOString()};
            return state.route;
          });
          return json(response,200,{route});
        }
        throw new ApiError('API route not found.',404,'NOT_FOUND');
      }
      if(!['GET','HEAD'].includes(request.method)) throw new ApiError('Method not allowed.',405,'METHOD_NOT_ALLOWED');
      let pathname;
      try { pathname=decodeURIComponent(url.pathname); } catch { throw new ApiError('Invalid URL encoding.'); }
      if(pathname.includes('\0') || pathname.includes('\\')) throw new ApiError('Invalid path.');
      if(pathname==='/') pathname='/index.html';
      const target=resolve(publicDir,'.'+pathname);
      if(target!==publicDir && !target.startsWith(publicDir+sep)) throw new ApiError('Path is not allowed.',403);
      let actual;
      try { actual=await realpath(target); }
      catch(error) { if(error.code==='ENOENT'||error.code==='ENOTDIR') throw new ApiError('File not found.',404); throw error; }
      const actualPublic=await realpath(publicDir);
      if(!actual.startsWith(actualPublic+sep)) throw new ApiError('Path is not allowed.',403);
      if(!(await stat(actual)).isFile()) throw new ApiError('File not found.',404);
      const content=await readFile(actual);
      response.writeHead(200,{'Content-Type':MIME[extname(actual)]??'application/octet-stream',
        'Cache-Control':'no-cache','Content-Length':content.length});
      response.end(request.method==='HEAD'?undefined:content);
    } catch(error) {
      if(response.headersSent) return response.destroy();
      const status=error instanceof ApiError?error.status:500;
      if(status===500) console.error('MycoScout request failed:',error.message);
      json(response,status,{error:status===500?'Internal server error.':error.message,code:error.code??'INTERNAL_ERROR'});
    } finally {
      if(imageRequest) activeImageRequests--;
    }
  });
  server.requestTimeout=15000; server.headersTimeout=10000;
  server.keepAliveTimeout=5000;
  server.on('close',()=>{for(const timer of backgroundTimers)clearInterval(timer);classifier.stop?.();researchClassifier.stop?.();finds.pending.finally(()=>finds.close());testbench.stop().catch(error=>console.error('Testbench shutdown:',error.message));communityBridge.close().catch(backgroundError);suggestionRefresh.close().catch(backgroundError);learning.stop().then(()=>learningLoop.close()).finally(()=>videoGrades.close()).catch(backgroundError);live.close().finally(()=>mission.close());});
  return {server,store,finds,classifier,testbench,testbenchWorker,videos,videoGrades,live,mission,learning,learningLoop,communityBridge,comparisons,suggestionRefresh,deviceCompute,host,listen:(port=8787)=>new Promise((resolveListen,reject)=>{
    server.once('error',reject);
    server.listen(port,host,()=>{server.off('error',reject);startBackground();resolveListen(server.address());});
  })};
}

if(process.argv[1] && import.meta.url===pathToFileURL(resolve(process.argv[1])).href) {
  try {
    const app=await createApp();
    const port=Number(process.env.MYCOSCOUT_PORT??8787);
    if(!Number.isInteger(port)||port<1||port>65535) throw new Error('MYCOSCOUT_PORT must be an integer from 1 to 65535.');
    await app.listen(port);
    console.log('MycoScout is ready at http://'+app.host+':'+port+' (no drone control).');
    if(!LOOPBACK.has(app.host)) console.log('Use a trusted HTTPS proxy or VPN for remote clients; never expose the HTTP port to the public internet.');
  } catch(error) { console.error(error.message); process.exitCode=1; }
}
