import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { resolve, join, relative, isAbsolute } from 'node:path';
import { tmpdir } from 'node:os';
import { LiveMissionStore } from '../core/live-mission.mjs';

const tempRoot=resolve(tmpdir()),directory=await mkdtemp(join(tempRoot,'mycoscout-live-simulation-'));
const store=await new LiveMissionStore(directory,{now:()=>1800000000000}).initialize();
try {
  store.seedSimulation();const clear=[];
  for(let i=0;i<5;i++)clear.push(store.stepSimulation().action);
  const completed=store.snapshot().tasks[0];store.seedSimulation({blocked:true});
  const blocked=store.stepSimulation().action;
  const report={schema:'mycoscout.live-executor-report.v1',createdAt:new Date().toISOString(),simulationOnly:true,realFlightTested:false,actuationEnabled:false,
    clearRoute:{actions:clear,task:completed},blockedRoute:{action:blocked},passed:completed.status==='completed'&&completed.capture.synthetic===true&&blocked.type==='hold'};
  const output=resolve('flight/reports/live-executor-latest.json');await mkdir(resolve('flight/reports'),{recursive:true});await writeFile(output,JSON.stringify(report,null,2)+'\n');
  console.log(JSON.stringify({passed:report.passed,clearRoute:clear.map(x=>x.type),blockedReason:blocked.reason,report:output,simulationOnly:true},null,2));
  if(!report.passed)process.exitCode=1;
}finally{
  store.close();const child=relative(tempRoot,directory);
  if(child&&!child.startsWith('..')&&!isAbsolute(child)&&child.startsWith('mycoscout-live-simulation-'))await rm(directory,{recursive:true,force:true});
}
