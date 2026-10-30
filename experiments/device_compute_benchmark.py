"""Measure local optional vision tasks on real saved frames; never a detection accuracy test."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import os
import platform
import statistics
import time
from pathlib import Path
from PIL import Image
from adapters.device_worker import quality_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=Path('data/video-discovery-v1'))
    parser.add_argument('--model', type=Path, default=Path('models/grounding-dino-tiny'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-name', default=platform.processor())
    parser.add_argument('--memory-gib', type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Preserve previous benchmarks; choose a new output path')
    manifest = json.loads((args.collection / 'manifest.json').read_text(encoding='utf8'))
    ids = ['youtube-1Z9FgVpYris-003', 'youtube-1Z9FgVpYris-019', 'archive-freestone-huge-mushroom-000']
    selected = [next(f for f in manifest['frames'] if f['id'] == identifier) for identifier in ids]
    import torch
    from adapters.scout_frame import LocalDetector
    started = time.perf_counter()
    detector = LocalDetector(args.model, device='cpu', threads=4)
    load_ms = (time.perf_counter() - started) * 1000
    rows = []
    for frame in selected:
        path = args.collection / frame['path']
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != frame['sha256']:
            raise ValueError('Saved frame hash changed')
        image = Image.open(io.BytesIO(raw)).convert('RGB')
        quality_times = []
        for i in range(12):
            start = time.perf_counter()
            decoded = Image.open(io.BytesIO(raw)).convert('RGB')
            quality = quality_metrics(decoded)
            if i >= 2:
                quality_times.append((time.perf_counter() - start) * 1000)
        # Warm once; record separately so a warm-only result is never presented as startup cost.
        if not rows:
            start = time.perf_counter()
            detector.predict(image)
            warmup_ms = (time.perf_counter() - start) * 1000
        start = time.perf_counter()
        candidates = detector.predict(image)
        detector_ms = (time.perf_counter() - start) * 1000
        cx, cy = image.width // 2, image.height // 2
        crop = image.crop((max(0, cx - 128), max(0, cy - 128), min(image.width, cx + 128), min(image.height, cy + 128)))
        encoded_crop = io.BytesIO()
        crop.save(encoded_crop, format='JPEG', quality=93)
        rows.append({'frameId': frame['id'], 'sourceGroupId': frame['sourceGroupId'], 'sha256': frame['sha256'],
                     'width': image.width, 'height': image.height, 'jpegBytes': len(raw),
                     'qualityMs': {'median': statistics.median(quality_times), 'min': min(quality_times), 'max': max(quality_times), 'runs': len(quality_times), 'includesDecode': True},
                     'detectorMs': detector_ms, 'unverifiedCandidateCount': len(candidates),
                     'quality': quality, 'nativeCenterCrop': {'width': crop.width, 'height': crop.height, 'jpegBytes': len(encoded_crop.getvalue()), 'containsMushroom': 'not evaluated'}})
        print(json.dumps({'frame': frame['id'], 'qualityMs': rows[-1]['qualityMs']['median'], 'detectorMs': detector_ms}), flush=True)
    report = {'schema': 'mycoscout.device-local-benchmark.v1', 'createdAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'measuredHere': True, 'physicalOtherDevicesTested': [], 'hardware': {'cpu': args.cpu_name, 'logicalProcessors': os.cpu_count(), 'memoryGiB': args.memory_gib, 'platform': platform.platform(), 'python': platform.python_version(), 'torch': torch.__version__, 'cudaAvailable': torch.cuda.is_available(), 'inferenceDevice': 'cpu', 'threads': 4},
              'model': detector.info, 'modelLoadMs': load_ms, 'firstInferenceWarmupMs': warmup_ms, 'frames': rows,
              'summary': {'qualityMedianMs': statistics.median(r['qualityMs']['median'] for r in rows), 'detectorMedianMs': statistics.median(r['detectorMs'] for r in rows), 'meanFrameBytes': round(statistics.mean(r['jpegBytes'] for r in rows)), 'meanNativeCropBytes': round(statistics.mean(r['nativeCenterCrop']['jpegBytes'] for r in rows))},
              'limits': ['Three saved frames, one measured detector pass each; not a sustained thermal or power benchmark.', 'Candidate counts are unverified model outputs, not detection recall, precision, species IDs or distance success rates.', 'Measured on this PC only. Browser, Python, CPU and GPU pipelines may differ.', 'A 256-pixel crop preserves source pixel scale but needs a useful local region proposal; this benchmark did not evaluate crop selection.', 'No physical phone, headset, Pi, network, drone or flight testing occurred.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
