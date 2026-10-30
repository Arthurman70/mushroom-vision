import {readFile,writeFile} from 'node:fs/promises';
import {simulateOffload} from '../core/device-compute.mjs';
const [input,output]=process.argv.slice(2);if(!input||!output)throw new Error('Usage: node experiments/device_compute_simulation.mjs measured.json output.json');
const report=JSON.parse(await readFile(input,'utf8')),m=report.summary;
const scenarios=[
 {name:'PC detection every ten seconds over a good local connection',bandwidthMbps:30,rttMs:30,lossRate:.01,fps:.1,computeMs:m.detectorMedianMs,frameBytes:m.meanFrameBytes,deadlineMs:15000},
 {name:'PC detection at one frame per second (overloaded)',bandwidthMbps:30,rttMs:30,lossRate:.01,fps:1,computeMs:m.detectorMedianMs,frameBytes:m.meanFrameBytes,deadlineMs:15000},
 {name:'PC detection every ten seconds over a weak connection',bandwidthMbps:1,rttMs:250,lossRate:.1,fps:.1,computeMs:m.detectorMedianMs,frameBytes:m.meanFrameBytes,deadlineMs:15000},
 {name:'Native crop over a weak connection (same conservative detection cost)',bandwidthMbps:1,rttMs:250,lossRate:.1,fps:.1,computeMs:m.detectorMedianMs,frameBytes:m.meanNativeCropBytes,deadlineMs:15000},
 {name:'Quality work at this PC speed; hypothetical remote connection',bandwidthMbps:10,rttMs:50,lossRate:.02,fps:2,computeMs:m.qualityMedianMs,frameBytes:m.meanFrameBytes,deadlineMs:1000},
 {name:'Disconnected helper',bandwidthMbps:1,rttMs:250,lossRate:1,fps:1,computeMs:m.detectorMedianMs,frameBytes:m.meanFrameBytes,deadlineMs:5000}
];
const result={schema:'mycoscout.device-scenarios.v1',createdAt:new Date().toISOString(),benchmarkSchema:report.schema,benchmarkCreatedAt:report.createdAt,networkMeasured:false,deviceComputeMeasuredOnlyOnThisPC:true,scenarios:scenarios.map(({name,...params})=>{const {outcomes,...summary}=simulateOffload({...params,frames:100,seed:42});return {name,...summary};}),recommendations:['Keep stabilization, collision avoidance and failsafes local.','Use the measured computer for low-rate candidate analysis; drop expired work.','Preserve source-pixel crops when bandwidth is constrained; shrinking whole frames can erase tiny mushrooms.','Use phone/Quest/Pi quality workers only after measuring the actual device; background suspension and heat were not simulated.','No results become verified labels, coordinates, or aircraft commands automatically.']};
await writeFile(output,JSON.stringify(result,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify(result.scenarios.map(s=>({name:s.name,completed:s.completed,missedDeadlines:s.missedDeadlines,latency:s.latencyMs})),null,2));
