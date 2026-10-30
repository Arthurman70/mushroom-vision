"""Package one existing licensed recording and its hash-matched detector results.

No new labels are produced. The rendered video is a silent H.264 derivative of
the CC BY-SA 2.0 recording; its attribution and license remain in the manifest.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ID = 'archive-freestone-huge-mushroom'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    collection = json.loads((ROOT / 'data/video-discovery-v1/manifest.json').read_text(encoding='utf-8'))
    source = next(item for item in collection['sources'] if item['id'] == SOURCE_ID)
    original = ROOT / 'data/archive-video-v1/originals/Freestonew-theHugeMushroom948.flv'
    if digest(original) != source['videoHash']:
        raise ValueError('The original recording no longer matches its recorded provenance.')
    directory = ROOT / 'data/tracking-demo-v1'
    directory.mkdir(exist_ok=True)
    video = directory / 'replay.mp4'
    subprocess.run([shutil.which('ffmpeg') or 'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y',
                    '-copyts', '-i', str(original), '-map', '0:v:0', '-an', '-sn', '-dn', '-c:v', 'libx264', '-preset', 'fast',
                    '-crf', '20', '-profile:v', 'baseline', '-pix_fmt', 'yuv420p', '-fps_mode', 'passthrough',
                    '-enc_time_base', '1/1000', '-video_track_timescale', '1000', '-movflags', '+faststart', str(video)], check=True)
    frames = []
    for frame in sorted((f for f in collection['frames'] if f['sourceId'] == SOURCE_ID), key=lambda f: f['timestampSeconds']):
        prediction_path = ROOT / 'data/ai-video-review-v1/frames' / ('main--' + frame['id'] + '.json')
        result = json.loads(prediction_path.read_text(encoding='utf-8'))
        if (result['imageSha256'] != frame['sha256'] or result['videoHash'] != source['videoHash']
                or result['status'] != 'completed' or result['sourceId'] != SOURCE_ID):
            raise ValueError('Detector output is not tied to the original frame.')
        image = ROOT / 'data/video-discovery-v1' / frame['path']
        if digest(image) != frame['sha256']:
            raise ValueError('Recorded frame integrity check failed.')
        frames.append({'id': frame['id'], 'timestampSeconds': frame['timestampSeconds'], 'imageSha256': frame['sha256'],
                       'width': frame['width'], 'height': frame['height'], 'processingSeconds': result['seconds'],
                       'detectorVersion': result['model']['resolvedRevision'], 'detectorCompletedAt': result['completedAt'],
                       'detections': [{'bbox': box['bbox'], 'confidence': box['score'], 'label': 'Unverified mushroom candidate'} for box in result['boxes']]})
    manifest = {'schema': 'mushroom-vision.tracking-demo.v1', 'source': {key: source[key] for key in
        ['id', 'title', 'attribution', 'license', 'licenseUrl', 'sourceUrl', 'videoHash']},
        'videoSha256': digest(video), 'videoBytes': video.stat().st_size,
        'width': source['video']['width'], 'height': source['video']['height'], 'durationSeconds': source['video']['durationSeconds'],
        'changes': 'Silent H.264 transcode for browser playback. Detector overlays and temporal association are separate, unverified research results.',
        'inferenceMode': 'precomputed_replay', 'humanVerified': False, 'trainingAllowed': False, 'actuationEnabled': False,
        'frames': frames, 'gps': None, 'depthAvailable': False,
        'notice': 'Recorded licensed video with precomputed GroundingDINO detections. Track IDs are heuristic associations, not verified specimen identities. No measured distance, depth, pose or GPS is available.'}
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'frames': len(frames), 'videoBytes': video.stat().st_size, 'detections': sum(len(f['detections']) for f in frames), 'source': SOURCE_ID}))


if __name__ == '__main__':
    main()
