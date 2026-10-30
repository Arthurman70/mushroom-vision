// Pure conservative geometry. No transport, autopilot or flight-control imports.
export function positive(value, name, allowZero = false) {
  if (!Number.isFinite(value) || (allowZero ? value < 0 : value <= 0)) throw new Error(`${name} must be ${allowZero ? 'nonnegative' : 'positive'} and finite`);
  return value;
}
export function point(p, name = 'point') {
  if (!p || !['x','y','z'].every(k => Number.isFinite(p[k]))) throw new Error(`${name} requires finite x/y/z`);
  return p;
}
export const distance = (a, b) => Math.hypot(a.x-b.x, a.y-b.y, a.z-b.z);
export const key = (x,y,z) => `${x},${y},${z}`;

export function validateVoxelMap(map) {
  if (!map || map.frame !== 'map_enu' || map.unit !== 'm') throw new Error('3D voxel map requires map_enu frame and metre units');
  point(map.origin, 'origin'); positive(map.resolutionM, 'resolutionM');
  if (!map.size || !['x','y','z'].every(k => Number.isInteger(map.size[k]) && map.size[k] > 0 && map.size[k] <= 256)) throw new Error('size requires integer x/y/z dimensions in 1..256');
  if (!map.voxels || typeof map.voxels !== 'object' || Array.isArray(map.voxels)) throw new Error('voxels must be a sparse object; absent cells are unknown');
  if (!Number.isFinite(map.updatedMs)) throw new Error('map updatedMs is required');
  return map;
}

function segmentIntersectsAabb(a,b,lo,hi) {
  let t0=0, t1=1;
  for (const axis of ['x','y','z']) {
    const d=b[axis]-a[axis];
    if (Math.abs(d)<1e-12) { if (a[axis]<lo[axis] || a[axis]>hi[axis]) return false; continue; }
    let l=(lo[axis]-a[axis])/d, h=(hi[axis]-a[axis])/d;
    if(l>h) [l,h]=[h,l];
    t0=Math.max(t0,l); t1=Math.min(t1,h); if(t0>t1) return false;
  }
  return true;
}

/** Checks the whole segment against inflated voxel boxes, never just endpoints.
 * Inflating boxes in all axes is conservative relative to a swept sphere.
 * Unknown, expired or malformed free evidence blocks the corridor. */
export function sweptClearance(map,from,to,{radiusM,nowMs,maxVoxelAgeMs=1000,maxClockSkewMs=20}={}) {
  try { validateVoxelMap(map); point(from); point(to); positive(radiusM,'radiusM'); positive(maxVoxelAgeMs,'maxVoxelAgeMs'); positive(maxClockSkewMs,'maxClockSkewMs',true); if(!Number.isFinite(nowMs)) throw new Error('nowMs is required'); }
  catch(error) { return {clear:false,reason:'invalid_geometry',detail:error.message}; }
  const axes=['x','y','z'], r=map.resolutionM, lower={},upper={};
  for(const axis of axes) {
    const low=Math.min(from[axis],to[axis])-radiusM, high=Math.max(from[axis],to[axis])+radiusM;
    const mapMax=map.origin[axis]+map.size[axis]*r;
    if(low<=map.origin[axis] || high>=mapMax) return {clear:false,reason:'outside_known_volume',axis};
    // Include the lower neighbour when the expanded body exactly touches a cell face.
    lower[axis]=Math.max(0,Math.ceil((low-map.origin[axis])/r)-1);
    upper[axis]=Math.min(map.size[axis]-1,Math.floor((high-map.origin[axis])/r));
  }
  let checked=0;
  for(let z=lower.z;z<=upper.z;z++) for(let y=lower.y;y<=upper.y;y++) for(let x=lower.x;x<=upper.x;x++) {
    const index={x,y,z}, lo={},hi={};
    for(const axis of axes) { lo[axis]=map.origin[axis]+index[axis]*r-radiusM; hi[axis]=lo[axis]+r+2*radiusM; }
    if(!segmentIntersectsAabb(from,to,lo,hi)) continue;
    checked++;
    const cell=map.voxels[key(x,y,z)];
    if(!cell || cell.state!=='free') return {clear:false,reason:cell?.state==='occupied'?'occupied_voxel':'unknown_voxel',voxel:index,checked};
    const age=nowMs-cell.observedMs;
    if(!Number.isFinite(age) || age>maxVoxelAgeMs || age < -maxClockSkewMs) return {clear:false,reason:'stale_voxel',voxel:index,checked};
  }
  return {clear:true,checked,radiusM,method:'segment-versus-inflated-voxel-boxes'};
}

export function projection(camera,rangeM,diameterM=camera.minTargetDiameterM) {
  positive(camera.fxPx,'fxPx'); positive(rangeM,'rangeM'); positive(diameterM,'diameterM');
  return camera.fxPx*diameterM/rangeM;
}
export function blurPixels(camera,{speedMps,rangeM,yawRateRadS=0}) {
  positive(camera.fxPx,'fxPx'); positive(camera.exposureS,'exposureS'); positive(rangeM,'rangeM'); positive(speedMps,'speedMps',true);
  if(!Number.isFinite(yawRateRadS)) throw new Error('yawRateRadS must be finite');
  return camera.fxPx*camera.exposureS*(speedMps/rangeM+Math.abs(yawRateRadS));
}
export function blurLimitedSpeed(camera,rangeM,yawRateRadS=0) {
  positive(camera.fxPx,'fxPx'); positive(camera.exposureS,'exposureS'); positive(camera.maxBlurPx,'maxBlurPx'); positive(rangeM,'rangeM');
  if(!Number.isFinite(yawRateRadS)) throw new Error('yawRateRadS must be finite');
  return Math.max(0,(camera.maxBlurPx/(camera.fxPx*camera.exposureS)-Math.abs(yawRateRadS))*rangeM);
}
export function geometryReport(camera) {
  const rows=[0.5,1,2,4,8].map(rangeM=>({rangeM,targetDiameterM:camera.minTargetDiameterM,targetPixels:projection(camera,rangeM),blurLimitedSpeedMps:blurLimitedSpeed(camera,rangeM)}));
  return {calibrationStatus:camera.calibrationStatus,detectorPerformanceMeasured:false,validatedDetectorMinPixels:camera.validatedDetectorMinPixels,designTargetPixels:camera.designTargetPixels,designRangeM:camera.fxPx*camera.minTargetDiameterM/camera.designTargetPixels,rows,notes:['Pinhole fronto-parallel target geometry, not a recall guarantee. Slant, occlusion and motion reduce useful detail.','Speed formula assumes transverse image motion at the specified range; rotation consumes the same blur budget.','RGB rolling-shutter distortion, vibration and focus require separate measured tests.']};
}

/** Bounded 6-neighbour A* in genuine 3D. Every edge checks a swept volume.
 * Vertical moves are opt-in; z bounds must describe an already observed band.
 * This is a conservative reference planner, not a real-time motion controller. */
export function planCorridor3D(map,start,goal,options={}) {
  const {radiusM,nowMs,maxExpanded=12000,allowVertical=false,minZM=-Infinity,maxZM=Infinity,...freshness}=options;
  const check=(a,b)=>sweptClearance(map,a,b,{radiusM,nowMs,...freshness});
  const first=check(start,start),last=check(goal,goal);
  if(!first.clear || !last.clear) return {status:'blocked',reason:!first.clear?first.reason:last.reason,points:[],expanded:0};
  if(!Number.isInteger(maxExpanded) || maxExpanded<=0 || maxExpanded>100000) throw new Error('maxExpanded must be an integer in 1..100000');
  if(!allowVertical && Math.abs(start.z-goal.z)>1e-9) return {status:'blocked',reason:'altitude_change_not_authorized',points:[],expanded:0};
  if(start.z<minZM || start.z>maxZM || goal.z<minZM || goal.z>maxZM) return {status:'blocked',reason:'outside_altitude_band',points:[],expanded:0};
  const direct=check(start,goal);
  if(direct.clear) return {status:'ready',points:[{...start},{...goal}],expanded:0,clearance:direct,simulationOnly:true};
  const r=map.resolutionM;
  // An anchored lattice preserves initial altitude exactly when vertical moves are disabled.
  const pos=(idx)=>({x:start.x+idx.x*r,y:start.y+idx.y*r,z:start.z+idx.z*r});
  const root={x:0,y:0,z:0}, open=[{idx:root,g:0,f:distance(start,goal)}],cost=new Map([[key(0,0,0),0]]),parents=new Map(),closed=new Set();
  const moves=[{x:1,y:0,z:0},{x:-1,y:0,z:0},{x:0,y:1,z:0},{x:0,y:-1,z:0},...(allowVertical?[{x:0,y:0,z:1},{x:0,y:0,z:-1}]:[])];
  let expanded=0;
  while(open.length && expanded<maxExpanded) {
    open.sort((a,b)=>a.f-b.f || a.g-b.g || key(a.idx.x,a.idx.y,a.idx.z).localeCompare(key(b.idx.x,b.idx.y,b.idx.z)));
    const current=open.shift(),ck=key(current.idx.x,current.idx.y,current.idx.z);
    if(closed.has(ck)) continue; closed.add(ck); expanded++;
    const cp=pos(current.idx);
    if(distance(cp,goal)<=r*1.75 && check(cp,goal).clear) {
      const points=[{...goal},cp]; let k=ck;
      while(parents.has(k)) { const prev=parents.get(k); points.push(pos(prev)); k=key(prev.x,prev.y,prev.z); }
      points.reverse();
      // Greedy simplification also validates every longer edge against the full volume.
      const smooth=[points[0]]; let i=0;
      while(i<points.length-1) { let j=points.length-1; while(j>i+1 && !check(points[i],points[j]).clear) j--; smooth.push(points[j]); i=j; }
      return {status:'ready',points:smooth,expanded,simulationOnly:true};
    }
    for(const d of moves) {
      const next={x:current.idx.x+d.x,y:current.idx.y+d.y,z:current.idx.z+d.z},nk=key(next.x,next.y,next.z),np=pos(next),g=current.g+r;
      if(np.z<minZM || np.z>maxZM || closed.has(nk) || g>=(cost.get(nk)??Infinity) || !check(cp,np).clear) continue;
      cost.set(nk,g); parents.set(nk,current.idx); open.push({idx:next,g,f:g+distance(np,goal)});
    }
  }
  return {status:'blocked',reason:expanded>=maxExpanded?'search_budget_exhausted':'no_verified_3d_corridor',points:[],expanded,simulationOnly:true};
}
