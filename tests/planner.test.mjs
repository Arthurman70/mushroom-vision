import test from 'node:test';
import assert from 'node:assert/strict';
import { planWalk, inflatedOccupancy } from '../core/planner.mjs';
import { validateMap, validateObservation, cellAt } from '../core/validation.mjs';
import { demoState } from '../core/state.mjs';

function grid(width=8,height=8) { return {width,height,resolution:1,origin:{x:0,y:0},cells:Array(width*height).fill(0)}; }
function input(g) { return {frame:{id:'map',unit:'m'},grid:g}; }

test('A* finds short diagonals on known ground with metric distance',()=>{
  const g=grid(), route=planWalk(g,{x:1.5,y:1.5},{x:5.5,y:5.5},{clearance:0});
  assert.equal(route.status,'ready'); assert.equal(route.points.length,5);
  assert.ok(Math.abs(route.distance-4*Math.SQRT2)<1e-10);
});

test('unknown space and each terrain hazard block routes',()=>{
  for(const code of [-1,1,2,3,4]) {
    const g=grid();
    for(let y=0;y<g.height;y++) g.cells[y*g.width+4]=code;
    const route=planWalk(g,{x:1.5,y:3.5},{x:6.5,y:3.5},{clearance:0});
    assert.equal(route.status,'unavailable','code '+code+' must block');
    assert.deepEqual(route.points,[]);
  }
});

test('diagonal route cannot squeeze between touching blocked cells',()=>{
  const g=grid(2,2); g.cells=[0,1,1,0];
  assert.equal(planWalk(g,{x:0.5,y:0.5},{x:1.5,y:1.5},{clearance:0}).status,'unavailable');
});

test('clearance reserves hazards and map edges, closing narrow gaps',()=>{
  const g=grid(9,9);
  for(let y=0;y<9;y++) if(y!==4) g.cells[y*9+4]=1;
  assert.equal(planWalk(g,{x:2.5,y:4.5},{x:6.5,y:4.5},{clearance:0}).status,'ready');
  assert.equal(planWalk(g,{x:2.5,y:4.5},{x:6.5,y:4.5},{clearance:0.5}).status,'unavailable');
  assert.equal(inflatedOccupancy(g,1)[0],1);
  assert.throws(()=>inflatedOccupancy(g,-1),/clearance/);
});

test('an isolated occupied cell inflates conservatively in every direction',()=>{
  const g=grid(9,9); g.cells[4*9+4]=1;
  const blocked=inflatedOccupancy(g,1);
  for(let y=3;y<=5;y++) for(let x=3;x<=5;x++) assert.equal(blocked[y*9+x],1);
  assert.equal(blocked[4*9+2],0);
});

test('world coordinates respect offset, resolution and strict bounds',()=>{
  const g=grid(4,4); g.origin={x:-2,y:7}; g.resolution=0.5;
  assert.deepEqual(cellAt(g,{x:-1.75,y:7.25}),{col:0,row:0,index:0});
  assert.throws(()=>cellAt(g,{x:0,y:7.25}),/outside/);
  assert.throws(()=>cellAt(g,{x:NaN,y:7.25}),/finite/);
  assert.throws(()=>cellAt(g,{x:-1.75,y:Infinity}),/finite/);
});

test('import rejects malformed grids and does not infer ground traversability',()=>{
  const g=grid(4,4);
  assert.equal(validateMap(input(g)).groundTraversabilityVerified,false);
  assert.throws(()=>validateMap({...input(g),frame:{id:'world',unit:'m'}}),/map/);
  assert.throws(()=>validateMap(input({...g,cells:[0]})),/cells/);
  assert.throws(()=>validateMap(input({...g,resolution:0})),/resolution/);
  assert.throws(()=>validateMap(input({...g,cells:Array(16).fill(100)})),/cells/);
  assert.throws(()=>validateMap({...input(g),groundTraversabilityVerified:'true'}),/boolean/);
});

test('observations require measured depth, shared frame and valid free location',()=>{
  const g=grid(4,4);
  const o={x:1.5,y:1.5,z:0,confidence:0.7,frameId:'map',poseKnown:true,depthKnown:true};
  assert.equal(validateObservation(o,g).identification,'unverified');
  assert.equal(validateObservation(o,g).edible,null);
  assert.throws(()=>validateObservation({...o,poseKnown:false},g),/poseKnown/);
  assert.throws(()=>validateObservation({...o,depthKnown:false},g),/depthKnown/);
  assert.throws(()=>validateObservation({...o,confidence:1.1},g),/confidence/);
  assert.throws(()=>validateObservation({...o,x:Infinity},g),/finite/);
  g.cells[5]=-1;
  assert.throws(()=>validateObservation(o,g),/known free/);
});

test('all demo observations are reachable and path segments avoid inflated hazards',()=>{
  const state=demoState(), blocked=inflatedOccupancy(state.grid,0.5);
  for(const goal of state.observations) {
    const route=planWalk(state.grid,state.telemetry.position,goal,{goalId:goal.id});
    assert.equal(route.status,'ready',goal.id);
    for(let i=0;i<route.points.length;i++) {
      const p=cellAt(state.grid,route.points[i]); assert.equal(blocked[p.index],0);
      if(i) {
        const prior=cellAt(state.grid,route.points[i-1]);
        if(p.col!==prior.col&&p.row!==prior.row) {
          assert.equal(blocked[p.row*state.grid.width+prior.col],0);
          assert.equal(blocked[prior.row*state.grid.width+p.col],0);
        }
      }
    }
  }
});
