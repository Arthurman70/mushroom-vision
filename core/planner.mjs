import { ApiError, cellAt, finite } from './validation.mjs';
const NEIGHBORS = [[1,0],[-1,0],[0,1],[0,-1],[1,1],[1,-1],[-1,1],[-1,-1]];

// Square clearance is conservative and includes unknown cells and map boundaries.
export function inflatedOccupancy(grid, clearance = 0.5) {
  finite(clearance, 'clearance', 0, 3);
  const { width, height, cells, resolution } = grid;
  const radius = Math.ceil(clearance / resolution);
  const distances = new Int32Array(cells.length).fill(0x7fffffff);
  const queue = new Int32Array(cells.length);
  let head = 0, tail = 0;
  for (let i = 0; i < cells.length; i++) {
    if (cells[i] !== 0) { distances[i] = 0; queue[tail++] = i; }
  }
  while (head < tail) {
    const i = queue[head++];
    if (distances[i] >= radius) continue;
    const col = i % width, row = Math.floor(i / width);
    for (const [dx, dy] of NEIGHBORS) {
      const x = col + dx, y = row + dy, j = y * width + x;
      if (x >= 0 && y >= 0 && x < width && y < height && distances[j] > distances[i] + 1) {
        distances[j] = distances[i] + 1; queue[tail++] = j;
      }
    }
  }
  return Uint8Array.from(distances, (distance, i) => {
    const x = i % width, y = Math.floor(i / width);
    return distance <= radius || x < radius || y < radius || x >= width-radius || y >= height-radius ? 1 : 0;
  });
}

class MinHeap {
  items = [];
  push(item) {
    let i = this.items.length;
    this.items.push(item);
    while (i > 0) {
      const p = (i-1) >> 1;
      if (this.items[p].priority <= item.priority) break;
      this.items[i] = this.items[p]; i = p;
    }
    this.items[i] = item;
  }
  pop() {
    const first = this.items[0], last = this.items.pop();
    if (this.items.length) {
      let i = 0;
      while (true) {
        let child = i*2+1;
        if (child >= this.items.length) break;
        if (child+1 < this.items.length && this.items[child+1].priority < this.items[child].priority) child++;
        if (last.priority <= this.items[child].priority) break;
        this.items[i] = this.items[child]; i = child;
      }
      this.items[i] = last;
    }
    return first;
  }
  get length() { return this.items.length; }
}

export function planWalk(grid, start, goal, { clearance = 0.5, goalId = null } = {}) {
  if (!grid) throw new ApiError('Import a map before requesting a route.');
  const startCell = cellAt(grid, start), goalCell = cellAt(grid, goal);
  const blocked = inflatedOccupancy(grid, clearance);
  const base = {
    mode: 'walk', goalId, clearance, frameId: 'map',
    advisory: 'Ground route suggestion from supplied 2D data. Slopes, overhead hazards and field conditions are not verified. Never use for drone control.',
  };
  const unavailable = reason => ({ ...base, status: 'unavailable', reason, points: [], distance: 0 });
  if (blocked[startCell.index]) return unavailable('Start lies inside a hazard, unknown area or required clearance.');
  if (blocked[goalCell.index]) return unavailable('Goal lies inside a hazard, unknown area or required clearance.');
  const distance = new Float64Array(grid.cells.length).fill(Infinity);
  const parents = new Int32Array(grid.cells.length).fill(-1);
  const closed = new Uint8Array(grid.cells.length);
  const heap = new MinHeap();
  distance[startCell.index] = 0;
  heap.push({ index: startCell.index, priority: 0 });
  while (heap.length) {
    const { index } = heap.pop();
    if (closed[index]) continue;
    if (index === goalCell.index) {
      const indices = [];
      for (let i = index; i !== -1; i = parents[i]) indices.push(i);
      indices.reverse();
      const points = indices.map(i => ({
        x: grid.origin.x + (i % grid.width + 0.5)*grid.resolution,
        y: grid.origin.y + (Math.floor(i / grid.width) + 0.5)*grid.resolution, z: 0,
      }));
      // Cell-center endpoints avoid adding unchecked subcell segments.
      return { ...base, status: 'ready', points, distance: distance[index]*grid.resolution,
        requestedStart: { x:start.x, y:start.y }, requestedGoal: { x:goal.x, y:goal.y } };
    }
    closed[index] = 1;
    const col = index % grid.width, row = Math.floor(index / grid.width);
    for (const [dx,dy] of NEIGHBORS) {
      const x = col+dx, y = row+dy;
      if (x < 0 || y < 0 || x >= grid.width || y >= grid.height) continue;
      const j = y*grid.width+x;
      if (closed[j] || blocked[j]) continue;
      if (dx && dy && (blocked[row*grid.width+x] || blocked[y*grid.width+col])) continue;
      const next = distance[index] + (dx && dy ? Math.SQRT2 : 1);
      if (next >= distance[j]) continue;
      distance[j] = next; parents[j] = index;
      heap.push({ index:j, priority:next+Math.hypot(x-goalCell.col, y-goalCell.row) });
    }
  }
  return unavailable('No continuous route exists through known free cells at the requested clearance.');
}
