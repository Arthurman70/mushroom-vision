"""Sample completed camera-folder frames into optional device analysis; never controls a drone.

A camera writer should atomically rename finished JPEG/PNG files into this folder.
This is a portable sampled-frame producer, not a camera or ROS driver.
"""
from __future__ import annotations
import argparse
import base64
import json
import math
import os
from pathlib import Path
import time
from adapters.review_stream import Station, read_frame


def enqueue_frame(station, path, *, session_id, kind='image-quality-v1', ttl_ms=30000, target_device_id=None):
    raw, mime, digest = read_frame(path)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError('Optional device frames must be no more than 4 MiB')
    payload = {'kind': kind, 'sessionId': session_id, 'frameId': path.stem[:50] + '-' + digest[:32],
               'imageBase64': base64.b64encode(raw).decode('ascii'), 'mimeType': mime, 'ttlMs': ttl_ms}
    if target_device_id:
        payload['targetDeviceId'] = target_device_id
    result = station.request('/api/device-compute/jobs', payload)
    return {'jobId': result['id'], 'sessionId': session_id, 'frameId': result['frameId'], 'imageSha256': digest,
            'status': result['status'], 'flightAuthority': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='A JPEG/PNG or folder of atomically completed frames')
    parser.add_argument('--station', default='http://127.0.0.1:8787')
    parser.add_argument('--token-env', default='MYCOSCOUT_TOKEN', help='Station owner token environment variable')
    parser.add_argument('--session-id', required=True, help='Stable recording session identity, at most 120 characters')
    parser.add_argument('--kind', choices=['image-quality-v1', 'fungi-candidates-v1'], default='image-quality-v1')
    parser.add_argument('--target-device-id')
    parser.add_argument('--interval-seconds', type=float, default=2, help='At most one selected frame per interval, minimum 0.5 s')
    parser.add_argument('--max-frames', type=int, default=30)
    parser.add_argument('--ttl-ms', type=int, default=30000)
    parser.add_argument('--watch', action='store_true', help='Sample the newest new frame per interval, intentionally skipping backlog')
    parser.add_argument('--max-duration-seconds', type=int, default=600)
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval_seconds) or not .5 <= args.interval_seconds <= 60:
        parser.error('interval-seconds must be 0.5..60')
    if not 1 <= args.max_frames <= 1000 or not 1 <= args.max_duration_seconds <= 3600 or not 1000 <= args.ttl_ms <= 120000:
        parser.error('Frame, duration or lifetime bound is invalid')
    if not args.session_id.strip() or len(args.session_id) > 120 or any(ord(c) < 32 for c in args.session_id):
        parser.error('Provide a short stable session ID')
    if not args.input.exists() or args.watch and not args.input.is_dir():
        parser.error('Input must exist; watch mode requires a folder')
    station = Station(args.station, os.environ.get(args.token_env, ''))
    seen = set()
    sent = 0
    started = time.monotonic()
    while sent < args.max_frames and time.monotonic() - started < args.max_duration_seconds:
        paths = [args.input] if args.input.is_file() else [p for p in args.input.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
        paths = [p for p in paths if (str(p), p.stat().st_mtime_ns) not in seen]
        paths.sort(key=lambda p: (p.stat().st_mtime_ns, p.name), reverse=args.watch)
        if not paths:
            if not args.watch:
                break
        else:
            selected = paths[0]
            # Watch mode intentionally sheds older frames to avoid a stale analysis backlog.
            for path in paths if args.watch else [selected]:
                seen.add((str(path), path.stat().st_mtime_ns))
            try:
                result = enqueue_frame(station, selected, session_id=args.session_id, kind=args.kind,
                                       ttl_ms=args.ttl_ms, target_device_id=args.target_device_id)
                sent += 1
                print(json.dumps(result), flush=True)
            except (ValueError, RuntimeError, OSError) as error:
                print(json.dumps({'frame': selected.name, 'status': 'skipped', 'error': str(error)[:240]}), flush=True)
                if args.input.is_file():
                    return 2
        if args.input.is_file():
            break
        time.sleep(min(args.interval_seconds, max(0, args.max_duration_seconds - (time.monotonic() - started))))
    print(json.dumps({'state': 'finished', 'enqueued': sent, 'flightAuthority': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
