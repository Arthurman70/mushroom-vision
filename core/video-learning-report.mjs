import { open } from 'node:fs/promises';
import { resolve } from 'node:path';

// Fixed local artifacts only. The API never returns source records, reviews,
// identifiers, model weights, local paths, or contributor information.
const FILES = {
  inputs: ['data/ai-video-review-v1/inputs.json','mycoscout.ai-video-pass-inputs.v1'],
  pass: ['data/ai-video-review-v1/status.json','mycoscout.ai-video-pass-status.v1'],
  audit: ['data/ai-video-review-v1/assistant-audit.json','mycoscout.assistant-audit.v1'],
  dataset: ['data/assistant-video-pilot-dataset-v1/manifest.json','mycoscout.assistant-video-training.v1'],
  pilot: ['experiments/results/assistant-video-pilot-v1/model-card.json','mycoscout.assistant-pilot-model-card.v1'],
  comparison: ['experiments/results/assistant-video-pilot-comparison-v1/comparison.json','mycoscout.assistant-pilot-comparison.v1'],
  verifier: ['experiments/results/proposal-verifier-pilot-v1/model-card.json','mycoscout.assistant-verifier-model-card.v1'],
  groups: ['experiments/results/groups-video-cycle-v2/study.json','mycoscout.group-study.v1'],
};
const MAX_BYTES = 4 * 1024 * 1024;
const count = value => Number.isSafeInteger(value) && value >= 0 ? value : null;
const fraction = value => Number.isFinite(value) && value >= 0 && value <= 1 ? value : null;
const flag = value => typeof value === 'boolean' ? value : null;
const states = new Set(['running','completed','failed','pending','in-progress']);

async function saved(root, [path,schema]) {
  let handle;
  try {
    handle = await open(resolve(root,path),'r');
    const info = await handle.stat();
    if(!info.isFile() || info.size > MAX_BYTES) return {state:'unavailable'};
    const bytes = Buffer.alloc(MAX_BYTES + 1);
    let length = 0;
    while(length < bytes.length) {
      const {bytesRead} = await handle.read(bytes,length,bytes.length-length,null);
      if(!bytesRead) break;
      length += bytesRead;
    }
    if(length > MAX_BYTES) return {state:'unavailable'};
    const value = JSON.parse(bytes.subarray(0,length).toString('utf8'));
    if(!value || Array.isArray(value) || value.schema !== schema) return {state:'invalid'};
    return {state:'available',value};
  } catch(error) {
    return {state:error.code === 'ENOENT' ? 'missing' : 'invalid'};
  } finally {
    await handle?.close();
  }
}

function metrics(evaluation) {
  const m = evaluation?.metrics;
  if(!m || ['frames','truePositives','falsePositives','falseNegatives'].some(key => count(m[key]) === null) ||
     fraction(m.scoreThreshold) === null || fraction(m.iouThreshold) === null) return null;
  return {frames:m.frames,matched:m.truePositives,missed:m.falseNegatives,falsePositives:m.falsePositives,
    targets:m.truePositives + m.falseNegatives,scoreThreshold:m.scoreThreshold,iouThreshold:m.iouThreshold,
    sourceGroups:m.sourceGroups && typeof m.sourceGroups === 'object' && !Array.isArray(m.sourceGroups) ? Object.keys(m.sourceGroups).length : null};
}

function sameFrames(left,right) {
  const a = metrics(left), b = metrics(right);
  if(!a || !b || !a.frames || a.frames !== b.frames || a.targets !== b.targets ||
     a.scoreThreshold !== b.scoreThreshold || a.iouThreshold !== b.iouThreshold) return false;
  const identities = evaluation => {
    if(!Array.isArray(evaluation.frames) || evaluation.frames.length !== a.frames) return null;
    const keys = evaluation.frames.map(frame => {
      if(typeof frame?.frameId !== 'string' || typeof frame.collection !== 'string' ||
         !/^[a-f0-9]{64}$/.test(frame.imageSha256 ?? '') || count(frame.humanReviewRevision) === null) return null;
      return JSON.stringify([frame.collection,frame.frameId,frame.imageSha256,frame.humanReviewRevision]);
    });
    return keys.some(key => key === null) || new Set(keys).size !== keys.length ? null : keys.sort();
  };
  const x = identities(left), y = identities(right);
  return x !== null && y !== null && x.every((key,index) => key === y[index]);
}

function scaleEvidence(evaluation) {
  const bins = evaluation?.metrics?.pixelWidthBins;
  const names = ['under4','4to8','8to16','16to32','32to64'];
  const totals = names.map(name => count(bins?.[name]?.total));
  return {thresholdPixels:64,targetsBelow64:totals.every(value => value !== null) ? totals.reduce((a,b)=>a+b,0) : null,
    actualDistanceMeasured:evaluation?.metrics?.actualDistanceMeasured === true};
}

function labelCautions(evaluation){
  // These caveats apply to the exact saved human revisions inspected during
  // the video-distance audit. They never rewrite those human annotations.
  const checks=[
    ['youtube-m9aGEh33w0I-024','3181fbf365d79f4716dd2e0bd42b349bfd1cd8e6cf75ca5537a5554f1ea45ef7',1,'Saved negative frame 024 is disputed by a later visual audit; its human grade remains unchanged.'],
    ['youtube-m9aGEh33w0I-038','3f9ad8cc170fe70a30856c457abcc1e87be68e033397dd867afc58966c916e37',2,'Saved frame 038 has one human annotation region around two fruiting bodies. The denominator counts labeled regions, not distinct mushrooms.'],
  ];
  return checks.filter(([frameId,hash,revision])=>evaluation?.frames?.some(row=>row.frameId===frameId&&row.imageSha256===hash&&row.humanReviewRevision===revision)).map(row=>row[3]);
}

export async function readVideoLearningReport(root) {
  const entries = await Promise.all(Object.entries(FILES).map(async ([key,file]) => [key,await saved(root,file)]));
  const artifacts = Object.fromEntries(entries), get = key => artifacts[key].value;
  const inputs = get('inputs'), status = get('pass'), audit = get('audit'), dataset = get('dataset');
  const pilot = get('pilot'), comparison = get('comparison'), verifier = get('verifier'), groups = get('groups');
  const collections = Array.isArray(inputs?.collections) && inputs.collections.length <= 10 &&
    inputs.collections.every(row => count(row?.frames) !== null && count(row?.sources) !== null) ? inputs.collections : null;
  const totalFrames = collections ? collections.reduce((sum,row) => sum + row.frames,0) : count(status?.totalFrames);
  const totalVideos = collections ? collections.reduce((sum,row) => sum + row.sources,0) : null;
  const validPass = states.has(status?.status) && ['processedFrames','completedFrames','failedFrames','candidateBoxes','totalFrames'].every(key => count(status[key]) !== null) &&
    status.totalFrames === totalFrames && status.processedFrames <= totalFrames && status.completedFrames + status.failedFrames === status.processedFrames;
  if(status && !validPass) artifacts.pass.state = 'invalid';
  const auditValid = Array.isArray(audit?.reviews);
  if(audit && !auditValid) artifacts.audit.state = 'invalid';
  const reviews = auditValid ? audit.reviews : [];
  const unique = new Map();
  for(const row of reviews) if(typeof row?.collection === 'string' && typeof row.frameId === 'string') unique.set(JSON.stringify([row.collection,row.frameId]),row);
  const audited = [...unique.values()];
  const rows = [], baseline = comparison?.baseline, candidate = comparison?.candidate;
  let evidence = {thresholdPixels:64,targetsBelow64:null,actualDistanceMeasured:false};
  let comparisonStatus = artifacts.comparison.state;
  if(comparison && sameFrames(baseline,candidate) && comparison.humanFrames === metrics(baseline).frames) {
    rows.push({id:'baseline',label:'Existing detector',...metrics(baseline)},
      {id:'pilot',label:'Detector training pilot',...metrics(candidate)});
    evidence = scaleEvidence(baseline);
    comparisonStatus = 'available';
    if(verifier && /^[a-f0-9]{64}$/.test(comparison.snapshotSha256 ?? '') && verifier.snapshotSha256 === comparison.snapshotSha256 && sameFrames(baseline,verifier.baseline) &&
       sameFrames(baseline,verifier.candidate) && JSON.stringify(metrics(baseline)) === JSON.stringify(metrics(verifier.baseline))) {
      rows.push({id:'verifier',label:'Existing detector + crop filter',...metrics(verifier.candidate)});
    } else if(verifier) artifacts.verifier.state = 'incomparable';
  } else if(comparison) comparisonStatus = 'incomparable';
  artifacts.comparison.state = comparisonStatus;
  const training = dataset && count(dataset.images) !== null && count(dataset.positiveImages) !== null && count(dataset.explicitNegativeImages) !== null &&
    dataset.positiveImages + dataset.explicitNegativeImages === dataset.images ? {
      frames:dataset.images,positiveFrames:dataset.positiveImages,negativeFrames:dataset.explicitNegativeImages,
      sourceGroups:Array.isArray(dataset.sourceGroups) ? dataset.sourceGroups.length : null,humanVerified:flag(dataset.humanVerified),
    } : null;
  return {
    schema:'mycoscout.video-learning-summary.v1',generatedAt:new Date().toISOString(),readOnly:true,
    availability:Object.fromEntries(Object.entries(artifacts).map(([key,value]) => [key,value.state])),
    pass:{status:validPass ? status.status : 'unavailable',totalFrames,totalVideos,
      processedFrames:validPass ? status.processedFrames : null,completedFrames:validPass ? status.completedFrames : null,
      failedFrames:validPass ? status.failedFrames : null,candidateBoxes:validPass ? status.candidateBoxes : null,
      elapsedSeconds:validPass && Number.isFinite(status.elapsedSeconds) && status.elapsedSeconds >= 0 ? Math.round(status.elapsedSeconds) : null},
    audit:{reviewedFrames:auditValid ? audited.length : null,completePositiveFrames:auditValid ? audited.filter(row => row.complete === true && row.status === 'visible').length : null,
      completeNegativeFrames:auditValid ? audited.filter(row => row.complete === true && row.status === 'absent').length : null,
      incompleteOrUncertainFrames:auditValid ? audited.filter(row => row.complete !== true || !['visible','absent'].includes(row.status)).length : null},
    training,
    pilot:pilot ? {status:states.has(pilot.status) ? pilot.status : 'unavailable',epochs:count(pilot.epochs),optimizerSteps:count(pilot.optimizerSteps),
      trainingImages:count(pilot.trainingImages),activationEligible:flag(pilot.activationEligible),automaticActivation:flag(pilot.automaticActivation),humanVerified:flag(pilot.humanVerified)} : null,
    comparison:{status:comparisonStatus,rows,evidence,labelCautions:rows.length?labelCautions(baseline):[]},
    verifier:verifier ? {status:states.has(verifier.status) ? verifier.status : 'unavailable',positiveCrops:count(verifier.positiveCrops),negativeCrops:count(verifier.negativeCrops),
      sourceGroups:Array.isArray(verifier.trainingGroups) ? verifier.trainingGroups.length : null,
      activationEligible:flag(verifier.activationEligible),automaticActivation:flag(verifier.automaticActivation),humanVerified:flag(verifier.humanVerified)} : null,
    groups:groups ? {photos:count(groups.extraction?.photos),categories:Array.isArray(groups.classes) ? groups.classes.length : null,
      testPhotos:count(groups.metrics?.test?.photos),top1Accuracy:fraction(groups.metrics?.test?.top1Accuracy),top3Accuracy:fraction(groups.metrics?.test?.top3Accuracy)} : null,
    notice:'Assistant labels remain separate from human labels. These selected samples do not establish field performance, species identity or a flight distance.',
  };
}
