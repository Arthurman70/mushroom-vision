import { readFileSync } from 'node:fs';
import { key } from './geometry.mjs';

export function configFixture() { return JSON.parse(readFileSync(new URL('../config/x500-simulation.json',import.meta.url),'utf8')); }
export function sceneFixture(nowMs=0) {
  const map={frame:'map_enu',unit:'m',origin:{x:-2,y:-3,z:-1},size:{x:28,y:20,z:14},resolutionM:0.5,updatedMs:nowMs,voxels:{},synthetic:true};
  for(let z=0;z<map.size.z;z++) for(let y=0;y<map.size.y;y++) for(let x=0;x<map.size.x;x++) map.voxels[key(x,y,z)]={state:z<2?'occupied':'free',observedMs:nowMs};
  return map;
}
export function box(map,lo,hi,state='occupied',nowMs=map.updatedMs) {
  for(let z=0;z<map.size.z;z++) for(let y=0;y<map.size.y;y++) for(let x=0;x<map.size.x;x++) {
    const center={x:map.origin.x+(x+0.5)*map.resolutionM,y:map.origin.y+(y+0.5)*map.resolutionM,z:map.origin.z+(z+0.5)*map.resolutionM};
    if(['x','y','z'].every(a=>center[a]>=lo[a] && center[a]<=hi[a])) {
      if(state==='unknown') delete map.voxels[key(x,y,z)]; else map.voxels[key(x,y,z)]={state,observedMs:nowMs};
    }
  }
  return map;
}
export function refresh(map,nowMs) {
  map.updatedMs=nowMs;
  for(const cell of Object.values(map.voxels)) cell.observedMs=nowMs;
  return map;
}
export function inputFixture(nowMs=0,position={x:0,y:0,z:2},map=sceneFixture(nowMs)) {
  return {nowMs,map,telemetry:{position:{...position},yawRad:0,yawRateRadS:0,speedMps:0,batteryFraction:0.8,manualLinkAvailable:true,localization:{valid:true,varianceM2:0.0025,updatedMs:nowMs},depth:{valid:true,updatedMs:nowMs}}};
}
export function candidateFixture(nowMs=0,approachViewpoint={x:1,y:0,z:2}) {
  return {id:'synthetic-mushroom-1',frame:'map_enu',position:{x:2.5,y:1,z:0.3},approachViewpoint,synthetic:true,observations:[0,1,2].map(n=>({frameId:`camera-${nowMs}-${n}`,capturedMs:nowMs-n*20,capturePose:{frame:'map_enu',position:{x:0,y:0,z:2},yawRad:0,capturedMs:nowMs-n*20},depthKnown:true,qualityAccepted:true,rangeM:3.2}))};
}
