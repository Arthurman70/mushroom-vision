import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { randomUUID } from 'node:crypto';
import { ApiError, shortText, validateMap, validateObservation } from './validation.mjs';

export const CELL_TYPES = { '-1':'unknown', 0:'free', 1:'tree', 2:'water', 3:'cliff', 4:'muck' };
export const CAPABILITIES = Object.freeze({
  simulation: true, mapImport: true, qualifiedObservationInput: true, sharedMap: true, walkingRoute: true,
  droneConnected: false, autonomousFlight: false, liveVision: false, speciesIdentification: false,
  terrainClassification: false, georeferencedAR: false,
  integrations: { rtabmap:'occupancy-grid import only; converter provided', px4:'not connected',
    mavsdk:'not connected', fastPlanner:'not integrated', fungalVision:'not installed' },
});

export function emptyState() {
  return {
    schemaVersion:1, mode:'survey',
    frame:{ id:'map', unit:'m', axes:'x east/right, y north/forward, z up; local coordinates only' },
    grid:null, groundTraversabilityVerified:false, observations:[], hazards:[], route:null,
    telemetry:{ connected:false, position:null, source:'none', battery:null },
    capabilities:structuredClone(CAPABILITIES), updatedAt:new Date().toISOString(),
  };
}

export function deriveHazards(grid, mode = 'survey') {
  if (!grid) return [];
  const hazards = [], visited = new Uint8Array(grid.cells.length);
  for (let index = 0; index < grid.cells.length; index++) {
    const code = grid.cells[index];
    if (code <= 0 || visited[index]) continue;
    const queue = [index]; visited[index] = 1;
    let sumX=0, sumY=0, minX=Infinity, maxX=-Infinity, minY=Infinity, maxY=-Infinity;
    for (let i=0; i<queue.length; i++) {
      const current=queue[i], col=current%grid.width, row=Math.floor(current/grid.width);
      sumX+=col+0.5; sumY+=row+0.5;
      minX=Math.min(minX,col); maxX=Math.max(maxX,col+1);
      minY=Math.min(minY,row); maxY=Math.max(maxY,row+1);
      for (const [dx,dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
        const x=col+dx, y=row+dy, next=y*grid.width+x;
        if (x>=0 && y>=0 && x<grid.width && y<grid.height && !visited[next] && grid.cells[next]===code) {
          visited[next]=1; queue.push(next);
        }
      }
    }
    const type = code===1 && mode!=='simulation' ? 'obstacle' : CELL_TYPES[code];
    hazards.push({
      id:'hazard-'+index, type, label:type,
      x:grid.origin.x+sumX/queue.length*grid.resolution,
      y:grid.origin.y+sumY/queue.length*grid.resolution,
      radius:Math.max(maxX-minX,maxY-minY)*grid.resolution/2,
      bounds:{ minX:grid.origin.x+minX*grid.resolution, maxX:grid.origin.x+maxX*grid.resolution,
        minY:grid.origin.y+minY*grid.resolution, maxY:grid.origin.y+maxY*grid.resolution },
      source:mode==='simulation'?'synthetic-demo':'supplied-grid', verified:false,
    });
  }
  return hazards;
}

export function demoState() {
  const state = emptyState();
  state.mode = 'simulation'; state.groundTraversabilityVerified = true;
  const grid = { width:50, height:40, resolution:0.5, origin:{x:0,y:0}, cells:Array(2000).fill(0) };
  const rectangle = (x0,y0,x1,y1,code) => {
    for (let y=y0; y<=y1; y++) for (let x=x0; x<=x1; x++) grid.cells[y*grid.width+x]=code;
  };
  rectangle(0,0,49,0,-1); rectangle(0,39,49,39,-1); rectangle(0,0,0,39,-1); rectangle(49,0,49,39,-1);
  rectangle(25,11,30,26,2); rectangle(35,29,43,33,4); rectangle(40,6,45,10,3); rectangle(3,30,11,37,-1);
  for (const [x,y] of [[8,8],[13,6],[18,9],[7,16],[13,20],[18,17],[10,26],[19,29],[23,34],[34,7],[36,17],[41,22],[32,33],[45,35]]) rectangle(x,y,x+1,y+1,1);
  state.grid=grid; state.hazards=deriveHazards(grid,'simulation');
  state.telemetry={connected:false,position:{x:2.25,y:2.25,z:0},source:'simulation',battery:null};
  state.observations=[
    {id:'demo-1',x:8.25,y:6.75,z:0,label:'Morel-like sample',confidence:0.82},
    {id:'demo-2',x:17.25,y:12.25,z:0,label:'Chanterelle-like sample',confidence:0.68},
    {id:'demo-3',x:9.75,y:16.75,z:0,label:'Unidentified cluster',confidence:0.43},
    {id:'demo-4',x:22.25,y:12.75,z:0,label:'Bracket-like sample',confidence:0.76},
  ].map(o=>({...o,source:'synthetic-demo',reviewed:false,frameId:'map',poseKnown:true,depthKnown:true,identification:'unverified',edible:null}));
  state.updatedAt='2026-10-26T00:00:00.000Z';
  return state;
}

export function validateStoredState(state) {
  if (!state || state.schemaVersion!==1 || !['simulation','survey'].includes(state.mode) ||
      !Array.isArray(state.observations) || state.observations.length>10000) {
    throw new Error('Saved state schema is invalid. Preserve the file and repair it before restarting.');
  }
  if (state.grid) {
    Object.assign(state, validateMap(state));
    const ids = new Set();
    state.observations=state.observations.map(o=>{
      const id=shortText(o.id,'id',undefined,80);
      if(ids.has(id)) throw new Error('Duplicate saved observation id.');
      ids.add(id); return {...validateObservation(o,state.grid),id};
    });
  } else if (state.observations.length) throw new Error('Saved observations require a map.');
  state.route=null; state.capabilities=structuredClone(CAPABILITIES);
  state.hazards=deriveHazards(state.grid,state.mode);
  state.telemetry = state.mode==='simulation' ? {connected:false,position:{x:2.25,y:2.25,z:0},source:'simulation',battery:null}
    : {connected:false,position:null,source:'none',battery:null};
  return state;
}

export class StateStore {
  constructor(file) { this.file=file; this.pending=Promise.resolve(); }
  async initialize() {
    try { this.state=validateStoredState(JSON.parse(await readFile(this.file,'utf8'))); }
    catch(error) {
      if(error.code!=='ENOENT') throw error;
      this.state=demoState(); await this.persist(this.state);
    }
    return this;
  }
  snapshot() { return structuredClone(this.state); }
  async persist(state) {
    await mkdir(dirname(this.file),{recursive:true});
    const temporary=this.file+'.'+randomUUID()+'.tmp';
    await writeFile(temporary,JSON.stringify(state,null,2),{encoding:'utf8',mode:0o600});
    await rename(temporary,this.file);
  }
  mutate(operation) {
    const mutation=this.pending.then(async()=>{
      const next=structuredClone(this.state), result=await operation(next);
      next.updatedAt=new Date().toISOString();
      await this.persist(next); this.state=next;
      return result===undefined?this.snapshot():result;
    });
    this.pending=mutation.catch(()=>{});
    return mutation;
  }
}

export function exportLocalGeoJSON() {
  throw new ApiError('GeoJSON export is unavailable: this map contains local meters, not WGS84 coordinates. Export JSON to preserve the local frame.',409,'GEOREFERENCE_REQUIRED');
}
