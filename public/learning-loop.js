const $=id=>document.getElementById(id);
let current=null,loading=false,mutation=false;
const labels={disabled:'Automatic learning is off',paused:'Learning paused',waiting_data:'Waiting for approved examples',waiting_daily_limit:'Daily limit reached',waiting_resource:'Waiting for the training worker',waiting_sync:'Waiting for current permissions',exporting:'Preparing reviewed data',training:'Training a candidate',comparing:'Comparing on unseen validation videos',retry_wait:'Retry scheduled',candidate_ready:'Candidate ready for review',failed:'Worker needs attention'};
const when=value=>value?new Date(value).toLocaleString():'Not yet';
const percent=value=>typeof value==='number'?(value*100).toFixed(1)+'%':'Not measured';
async function api(path,body){const headers={};if($('token').value)headers.Authorization='Bearer '+$('token').value;if(body!==undefined)headers['Content-Type']='application/json';const response=await fetch('/api/learning/'+path,{method:body===undefined?'GET':'POST',headers,...(body!==undefined?{body:JSON.stringify(body)}:{})});const value=await response.json();if(!response.ok)throw new Error(value.error??'Request failed.');return value;}
function text(id,value){$(id).textContent=value;}
function detail(parent,tag,value){const element=document.createElement(tag);element.textContent=value;parent.append(element);return element;}
function metricsTable(parent,title,report){const table=detail(parent,'table','');detail(table,'caption',title);const header=detail(table,'tr','');for(const value of ['Measure','Current detector','Candidate'])detail(header,'th',value);
 for(const [label,key,fmt] of [['Mushrooms found','recall',percent],['Mushrooms below 64 px found','smallTargets',v=>percent(v?.recall)],['False detections / frame','falsePositivesPerFrame',v=>typeof v==='number'?v.toFixed(3):'Not measured']]){const row=detail(table,'tr','');detail(row,'td',label);for(const model of [report.baseline,report.candidate])detail(row,'td',fmt((model?.metrics??model)?.[key]));}
 if(report.support)detail(parent,'p',`${report.support.frames} frames · ${report.support.targets} marked mushrooms · ${report.support.targetGroups} independent mushroom-bearing source groups · ${report.support.negativeFrames} negative frames.`);}
function render(data){
 current=data;const {loop,community,comparisons,releaseRequest}=data,c=loop.config,latest=loop.latestCandidate;
 text('worker-state',labels[loop.workerState]??loop.workerState);text('worker-message',loop.reason?.message??'Your reviews stay separate from machine suggestions.');text('last-check','Last checked '+when(loop.updatedAt));
 text('new-frames',loop.counts.newFrames??'—');text('new-sessions',loop.counts.newSessions??'—');text('community-count',community.lastTotal??'—');text('active-model',data.activeModel?'Reviewed model':'Original');
 text('frame-threshold',`of ${c.minimumNewFrames} new approved frames`);text('session-threshold',`of ${c.minimumNewSessions} new source sessions`);
 for(const [id,value] of [['minimum-frames',c.minimumNewFrames],['minimum-sessions',c.minimumNewSessions],['daily-limit',c.maxRunsPerDay]])if(document.activeElement!==$(id))$(id).value=value;
 $('enable').disabled=c.enabled&&!loop.paused;$('pause').disabled=!c.enabled||loop.paused;$('cancel').disabled=!loop.active;
 text('readiness',loop.lastReadiness?.reasons?.join(' · ')??'Defaults: 50 new approved frames from 3 new source sessions; at most one candidate per day.');
 const sync=$('sync-status');sync.replaceChildren();for(const [a,b] of [['Sync',community.enabled?'Connected':'Off'],['Last imported',when(community.lastSyncAt)],['Last public update',when(community.lastPublishAt)],['Status',community.error??'Current permissions are rechecked before each training stage.']]){detail(sync,'dt',a);detail(sync,'dd',b);}
 if(data.suggestions){detail(sync,'dt','AI suggestions');detail(sync,'dd',data.suggestions.error??(!data.suggestions.activeModelAvailable?'Original suggestions remain until you approve a new model.':data.suggestions.running?`Refreshing suggestions${data.suggestions.progress?` · ${data.suggestions.progress.completed}/${data.suggestions.progress.total}`:''}`:`${data.suggestions.status} · last refresh ${when(data.suggestions.lastCompletedAt)}`));}
 text('candidate-title',latest?({candidate_better:'Candidate improved on validation',candidate_not_better:'Current detector still leads',insufficient_evidence:'More evidence needed',comparison_failed:'Comparison needs attention'}[latest.status]??latest.status):'No automatic candidate yet.');
 text('candidate-detail',latest?(latest.result?.reasons??['Review the saved comparison before release.']).join(' · '):'Training needs enough approved examples from independent videos, plus separate validation and test data.');
 const metrics=$('candidate-metrics');metrics.replaceChildren();if(latest?.result)metricsTable(metrics,'Validation comparison',latest.result);
 const release=comparisons.find(item=>item.jobId===latest?.jobId&&item.split==='test'&&item.status==='completed');
 if(release?.summary){metricsTable(metrics,'Final release test',release.summary);if(release.summary.reasons?.length)detail(metrics,'p',release.summary.reasons.join(' · '));}
 $('release-check').disabled=latest?.status!=='candidate_better'||releaseRequest?.status==='running'||!!release;
 $('activate').disabled=release?.decision!=='candidate_better'||data.activeModel?.jobId===latest?.jobId;
 text('release-status',releaseRequest?.error??(releaseRequest?.status==='running'?'Running the explicitly requested final release test…':release?`Release result: ${release.decision.replaceAll('_',' ')}. Review date: ${when(release.finishedAt)}.`:'A passing validation comparison unlocks an explicit release test. Review that result before selecting the model.'));
 const history=$('history');history.replaceChildren();if(!loop.history.length)detail(history,'p','No completed cycles.');for(const row of loop.history.slice(-8).reverse())detail(history,'p',`${when(row.at)} · ${row.outcome} · ${row.message}`);
}
async function refresh(){if(loading)return;loading=true;try{render(await api('loop'));}catch(error){text('notice',error.message);text('worker-state','Worker unavailable');}finally{loading=false;}}
async function action(path,body={}){if(mutation)return;mutation=true;try{await api(path,body);text('notice','Saved. The worker will update its progress here.');await refresh();}catch(error){text('notice',error.message);}finally{mutation=false;}}
$('enable').onclick=()=>action('loop/resume');$('pause').onclick=()=>action('loop/pause');$('cancel').onclick=()=>action('loop/cancel');$('check').onclick=()=>action('loop/tick');$('sync').onclick=()=>action('community-sync',{enabled:true});$('reconnect').onclick=refresh;
$('settings').onsubmit=event=>{event.preventDefault();action('loop/config',{minimumNewFrames:Number($('minimum-frames').value),minimumNewSessions:Number($('minimum-sessions').value),maxRunsPerDay:Number($('daily-limit').value)});};
$('release-check').onclick=()=>action('release-check',{jobId:current?.loop.latestCandidate?.jobId});
$('activate').onclick=()=>{const reviewer=$('release-reviewer').value.trim();if(!reviewer){text('notice','Enter the name of the person reviewing this release.');return;}action('activate',{jobId:current?.loop.latestCandidate?.jobId,reviewer});};
refresh();setInterval(refresh,10000);
