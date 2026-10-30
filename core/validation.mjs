export class ApiError extends Error {
  constructor(message, status = 400, code = 'INVALID_INPUT') {
    super(message); this.status = status; this.code = code;
  }
}

export function finite(value, name, min = -1e6, max = 1e6) {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw new ApiError(name + ' must be a finite number from ' + min + ' to ' + max + '.');
  }
  return value;
}

export function shortText(value, name, fallback, limit = 160) {
  if (value === undefined && fallback !== undefined) return fallback;
  if (typeof value !== 'string' || !value.trim() || value.length > limit || /[\u0000-\u001f]/u.test(value)) {
    throw new ApiError(name + ' must be nonempty text of at most ' + limit + ' characters.');
  }
  return value.trim();
}

export function validateMap(input) {
  if (!input || input.frame?.id !== 'map' || input.frame?.unit !== 'm') {
    throw new ApiError('A map must use frame {id:"map",unit:"m"}; no geographic position is inferred.');
  }
  const g = input.grid;
  if (!g || !Number.isInteger(g.width) || !Number.isInteger(g.height) ||
      g.width < 2 || g.height < 2 || g.width > 256 || g.height > 256) {
    throw new ApiError('Grid width and height must be integers from 2 to 256.');
  }
  const resolution = finite(g.resolution, 'resolution', 0.1, 100);
  const origin = { x: finite(g.origin?.x, 'origin.x'), y: finite(g.origin?.y, 'origin.y') };
  if (!Array.isArray(g.cells) || g.cells.length !== g.width * g.height ||
      g.cells.some(cell => !Number.isInteger(cell) || cell < -1 || cell > 4)) {
    throw new ApiError('cells must be row-major width*height integers: -1 unknown, 0 free, 1 tree, 2 water, 3 cliff, 4 muck.');
  }
  if (input.groundTraversabilityVerified !== undefined && typeof input.groundTraversabilityVerified !== 'boolean') {
    throw new ApiError('groundTraversabilityVerified must be a boolean.');
  }
  return {
    frame: { id: 'map', unit: 'm', axes: 'x east/right, y north/forward, z up; local coordinates only' },
    grid: { width: g.width, height: g.height, resolution, origin, cells: [...g.cells] },
    groundTraversabilityVerified: input.groundTraversabilityVerified === true,
  };
}

export function cellAt(grid, point) {
  finite(point?.x, 'x'); finite(point?.y, 'y');
  const col = Math.floor((point.x - grid.origin.x) / grid.resolution);
  const row = Math.floor((point.y - grid.origin.y) / grid.resolution);
  if (col < 0 || row < 0 || col >= grid.width || row >= grid.height) throw new ApiError('Position lies outside the current map.');
  return { col, row, index: row * grid.width + col };
}

export function validateObservation(input, grid) {
  if (!input || input.poseKnown !== true || input.depthKnown !== true || input.frameId !== 'map') {
    throw new ApiError('Observation requires poseKnown:true, depthKnown:true and frameId:"map" from a calibrated map transform. A photograph alone does not locate a mushroom.');
  }
  const point = { x: finite(input.x, 'x'), y: finite(input.y, 'y'), z: finite(input.z, 'z', -1000, 1000) };
  const cell = cellAt(grid, point);
  if (grid.cells[cell.index] !== 0) throw new ApiError('Observation must be in known free map space; unknown and hazard cells are rejected.');
  if (input.reviewed !== undefined && typeof input.reviewed !== 'boolean') throw new ApiError('reviewed must be a boolean.');
  return {
    ...point, label: shortText(input.label, 'label', 'Unidentified fungus'),
    confidence: finite(input.confidence ?? 0, 'confidence', 0, 1),
    source: shortText(input.source, 'source', 'manual-qualified-point', 120),
    reviewed: input.reviewed === true, poseKnown: true, depthKnown: true, frameId: 'map',
    identification: 'unverified', edible: null,
  };
}
