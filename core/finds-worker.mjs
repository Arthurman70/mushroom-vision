import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import { ApiError } from './validation.mjs';

/** Fixed local executable and checkpoint; API callers cannot choose programs or paths. */
export class FindsClassifier {
  constructor(root, { timeoutMs = 90000, spawnWorker = spawn, checkpoint = 'vision/checkpoints/groups-v1.json' } = {}) {
    if(!/^vision\/checkpoints\/[a-zA-Z0-9_-]+\.json$/.test(checkpoint))throw new Error('Choose a fixed local classifier checkpoint.');
    this.checkpoint = checkpoint;
    this.root = root; this.timeoutMs = timeoutMs; this.spawnWorker = spawnWorker;
    this.process = null; this.pending = null; this.buffer = ''; this.closed = false; this.loaded = false;
  }
  stop() { this.closed = true; this.fail(new ApiError('Classifier stopped.', 503, 'CLASSIFIER_UNAVAILABLE')); }
  fail(error) {
    const pending = this.pending; this.pending = null;
    if (pending) { clearTimeout(pending.timer); pending.reject(error); }
    const child = this.process; this.process = null; this.buffer = ''; this.loaded = false;
    if (child) child.kill();
  }
  start() {
    if (this.process) return;
    const python = process.env.MYCOSCOUT_PYTHON ?? (process.platform === 'win32'
      ? resolve(this.root, 'runtime-vision', 'Scripts', 'python.exe')
      : resolve(this.root, 'runtime-vision', 'bin', 'python'));
    const child = this.spawnWorker(python, ['-m', 'vision.multiclass', 'worker', '--checkpoint', this.checkpoint], {
      cwd: this.root, stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true,
      env: { ...process.env, PYTHONUNBUFFERED: '1', HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1' },
    });
    this.process = child;
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', chunk => {
      if (this.process !== child) return;
      this.buffer += chunk;
      if (this.buffer.length > 256 * 1024) return this.fail(new ApiError('Classifier response exceeded its limit.', 502, 'CLASSIFIER_PROTOCOL'));
      let newline;
      while ((newline = this.buffer.indexOf('\n')) >= 0) {
        const line = this.buffer.slice(0, newline); this.buffer = this.buffer.slice(newline + 1);
        if (!line.trim()) continue;
        let message;
        try { message = JSON.parse(line); } catch { return this.fail(new ApiError('Classifier returned invalid JSON.', 502, 'CLASSIFIER_PROTOCOL')); }
        if (!this.pending || message.id !== this.pending.id) return this.fail(new ApiError('Classifier response ID did not match.', 502, 'CLASSIFIER_PROTOCOL'));
        const pending = this.pending; this.pending = null; clearTimeout(pending.timer);
        if (message.status === 'ok' && message.result && typeof message.result === 'object') { this.loaded = true; pending.resolve(message.result); }
        else pending.reject(new ApiError('Local classifier could not process this image.', 422, 'CLASSIFIER_FAILED'));
      }
    });
    // Drain logs; never mix local paths or diagnostic data into a client response.
    child.stderr.resume();
    child.on('error', () => { if (this.process === child) this.fail(new ApiError('Local classifier is unavailable; configure its runtime and checkpoint.', 503, 'CLASSIFIER_UNAVAILABLE')); });
    child.on('exit', () => { if (this.process === child) this.fail(new ApiError('Local classifier exited.', 503, 'CLASSIFIER_UNAVAILABLE')); });
    child.stdin.on('error', () => { if (this.process === child) this.fail(new ApiError('Classifier input failed.', 503, 'CLASSIFIER_UNAVAILABLE')); });
  }
  classify(image, context = {}) {
    if (this.closed) return Promise.reject(new ApiError('Classifier stopped.', 503, 'CLASSIFIER_UNAVAILABLE'));
    if (this.pending) return Promise.reject(new ApiError('Classifier is processing another image. Retry shortly.', 429, 'CLASSIFIER_BUSY'));
    return new Promise((resolveRequest, reject) => {
      const id = randomUUID();
      this.pending = { id, resolve: resolveRequest, reject,
        timer: setTimeout(() => this.fail(new ApiError('Classifier timed out.', 504, 'CLASSIFIER_TIMEOUT')), this.timeoutMs) };
      try {
        this.start();
        this.process.stdin.write(JSON.stringify({ id, imageBase64: image.bytes.toString('base64'), mimeType: image.mimeType,
          candidateId: context.findId ?? null, captureTimestamp: context.capturedAt ?? null }) + '\n');
      } catch { this.fail(new ApiError('Local classifier is unavailable.', 503, 'CLASSIFIER_UNAVAILABLE')); }
    });
  }
}
