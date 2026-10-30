// Local, synthetic image-retry UI fixture. All mutations are rejected.
import http from 'node:http';
import { readFile, writeFile, stat } from 'node:fs/promises';
import { resolve, sep, extname } from 'node:path';
import { deflateSync } from 'node:zlib';
import { createHash } from 'node:crypto';

const project = resolve(import.meta.dirname, '..');
const publicDir = resolve(project, 'public');
const qaDir = resolve(project, process.argv[2] || 'data/grader-retry-qa-v1');
if (!qaDir.startsWith(resolve(project, 'data') + sep)) throw new Error('QA output must stay inside project data');
const controlPath = resolve(qaDir, 'control.json');
const infoPath = resolve(qaDir, 'server.json');
const width = 320, height = 240;

function chunk(kind, bytes) {
  const type = Buffer.from(kind), payload = Buffer.concat([type, bytes]);
  const size = Buffer.alloc(4), checksum = Buffer.alloc(4);
  size.writeUInt32BE(bytes.length);
  let crc = 0xffffffff;
  for (const byte of payload) {
    crc ^= byte;
    for (let i = 0; i < 8; i++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  checksum.writeUInt32BE((crc ^ 0xffffffff) >>> 0);
  return Buffer.concat([size, payload, checksum]);
}
const header = Buffer.alloc(13);
header.writeUInt32BE(width); header.writeUInt32BE(height, 4); header[8] = 8; header[9] = 2;
const pixels = Buffer.alloc(height * (1 + width * 3));
for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
  const at = y * (1 + width * 3) + 1 + x * 3;
  const checker = (Math.floor(x / 40) + Math.floor(y / 40)) % 2;
  pixels[at] = checker ? 220 : 35;
  pixels[at + 1] = checker ? 145 : 95;
  pixels[at + 2] = checker ? 70 : 65;
}
const image = Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  chunk('IHDR', header), chunk('IDAT', deflateSync(pixels)), chunk('IEND', Buffer.alloc(0))]);
const sha256 = createHash('sha256').update(image).digest('hex');
const source = { id: 'qa-synthetic-video', title: 'SYNTHETIC QA · image loading and retry',
  sourceUrl: 'https://example.org/qa-fixture', attribution: 'Generated software fixture; not mushroom footage',
  license: 'Synthetic software fixture', licenseUrl: '', sourceGroupId: 'qa:retry',
  videoHash: 'a'.repeat(64), evaluationOnly: true, licenseAllowsTraining: false };
const frames = ['qa-valid-frame', 'qa-retry-frame'].map((id, i) => ({
  id, sourceId: source.id, sourceGroupId: source.sourceGroupId, partitionGroup: source.sourceGroupId,
  sha256, width, height, timestampSeconds: i * 10, blurScore: i + 5,
  blurMetric: 'synthetic fixture', evaluationOnly: true, actualDistanceM: null, source,
  imageUrl: '/api/testbench/videos/frames/' + id + '/image'
}));
const grades = { schema: 'mycoscout.video-grades.v1', collectionId: 'qa-retry-only',
  reviews: Object.fromEntries(frames.map(frame => [frame.id, { frameId: frame.id, revision: 0, review: null, historyCount: 0 }])),
  tracks: [{ id: 'qa-individual-1', sourceId: source.id, videoHash: source.videoHash, label: 'QA individual 1' }],
  report: { totalFrames: 2, reviewedFrames: 0, unreviewedFrames: 2, verifiedBoxes: 0,
    statusCounts: { visible: 0, absent: 0, uncertain: 0 }, tracks: [] } };
const mime = { '.html': 'text/html; charset=utf-8', '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8', '.mjs': 'text/javascript; charset=utf-8',
  '.svg': 'image/svg+xml', '.json': 'application/json; charset=utf-8', '.png': 'image/png' };
function json(res, status, data) { res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(data)); }

await writeFile(controlPath, JSON.stringify({ imageReady: false }, null, 2), { flag: 'wx' });
const server = http.createServer(async (req, res) => {
  try {
    if (req.method !== 'GET') return json(res, 405, { error: 'Read-only synthetic QA station. Saving is disabled.' });
    const url = new URL(req.url, 'http://127.0.0.1');
    if (url.pathname === '/api/testbench/videos') {
      const offset = Math.max(0, Math.min(2, Number(url.searchParams.get('offset')) || 0));
      return json(res, 200, { schema: 'mycoscout.video-gallery.v1', available: true, sources: [source],
        frames: frames.slice(offset), total: 2, offset, limit: 24, labelsVerified: false, automaticTraining: false });
    }
    if (url.pathname === '/api/video-grades') return json(res, 200, grades);
    const match = /^\/api\/testbench\/videos\/frames\/(qa-valid-frame|qa-retry-frame)\/image$/.exec(url.pathname);
    if (match) {
      const control = JSON.parse(await readFile(controlPath, 'utf8'));
      if (match[1] === 'qa-retry-frame' && control.imageReady !== true)
        return json(res, 503, { error: 'Synthetic image failure. Enable imageReady in the QA control file, then Retry image.' });
      res.writeHead(200, { 'Content-Type': 'image/png', 'Content-Length': image.length, 'Cache-Control': 'no-store' });
      return res.end(image);
    }
    if (url.pathname.startsWith('/api/')) return json(res, 404, { error: 'No fixture for this API' });
    const path = resolve(publicDir, '.' + decodeURIComponent(url.pathname === '/' ? '/video-grader.html' : url.pathname));
    if (!path.startsWith(publicDir + sep) || !mime[extname(path)]) return json(res, 404, { error: 'Not found' });
    if (!(await stat(path)).isFile()) return json(res, 404, { error: 'Not found' });
    res.writeHead(200, { 'Content-Type': mime[extname(path)], 'Cache-Control': 'no-store' });
    res.end(await readFile(path));
  } catch (error) { if (!res.headersSent) json(res, 500, { error: 'QA fixture error: ' + error.message }); else res.destroy(); }
});
await new Promise(done => server.listen(0, '127.0.0.1', done));
const info = { pid: process.pid, port: server.address().port,
  url: 'http://127.0.0.1:' + server.address().port + '/video-grader.html?source=qa-synthetic-video',
  controlPath, synthetic: true, productionWrites: false };
await writeFile(infoPath, JSON.stringify(info, null, 2), { flag: 'wx' });
console.log(JSON.stringify(info));
