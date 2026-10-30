"""Publish only aggregate far-pixel results and simulated behavior checks."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-report', type=Path, required=True)
    parser.add_argument('--behavior-report', type=Path, required=True)
    args = parser.parse_args()
    vision = json.loads(args.vision_report.read_text(encoding='utf-8'))
    behavior = json.loads(args.behavior_report.read_text(encoding='utf-8'))
    if vision.get('schema')!='mycoscout.far-pixel-report.v1' or vision.get('status') != 'completed' or vision.get('actualDistanceDetectionRates') is not None:
        raise ValueError('A completed pixel-only report is required')
    for filename,key in [('protocol.json','protocolSha256'),('inputs.json','inputManifestSha256')]:
        if hashlib.sha256((args.vision_report.parent/filename).read_bytes()).hexdigest()!=vision[key]:
            raise ValueError('Report no longer matches its frozen protocol and inputs')
    if behavior.get('schema')!='mycoscout.range-inspection-report.v1' or behavior.get('simulationOnly') is not True or behavior.get('physicalFlights')!=0 or behavior.get('physicalPhotos')!=0:
        raise ValueError('Expected the explicitly simulated range behavior report')
    controls = vision['negativeControls']
    color_total = sum(row['colorOnly']['focalCueHits'] for row in vision['rows'] if row['illustrativeDistanceM'] is not None)
    color_targets = sum(row['colorOnly']['targets'] for row in vision['rows'] if row['illustrativeDistanceM'] is not None)
    report = {
        'schema': 'mycoscout.far-lab.v1', 'status': 'completed',
        'generatedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'summary': f"{vision['modelInferences']} actual model passes on {vision['images']} synthetic images from {vision['sourceScenes']} source scenes. Original RGB and color-enhanced RGB use the same fixed detection rules. Results below count the focal target in each scene.",
        'rows': vision['rows'], 'protocolSha256': vision['protocolSha256'],
        'inputManifestSha256': vision['inputManifestSha256'],
        'frameDimensions': vision['frameDimensions'], 'negativeControls': controls,
        'controlsNote': f"On {controls['raw']['frames']} background-only controls: original RGB produced {controls['raw']['falseDetections']} false detections; color enhancement produced {controls['color']['falseDetections']}. The separate color-cue finder produced {controls['colorOnly']['cues']} alerts. Controls are selected background crops, not flight footage.",
        'colorNote': f"The separate color-contrast finder placed a cue near {color_total}/{color_targets} tiny focal targets. This is proximity within two pixels, not a mushroom detection or ID. Color clues remain habitat follow-ups, with the original image preserved.",
        'limitations': vision['limitations'],
        'behaviorChecks': behavior.get('checks', []),
        'framingNote': 'The close-photo goal is 70% of the original frame width. With the illustrative 1080p camera, a 5 cm cap would require about 5 cm range, inside the 1 m minimum standoff. The request remains pending for optical zoom or a manual close-up; a digital crop does not complete it.',
        'hardwareTested': False, 'physicalFlights': 0,
    }
    (ROOT / 'public/far-report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'rows': len(report['rows']), 'behaviorChecks': len(report['behaviorChecks']), 'sourceScenes': vision['sourceScenes']}))


if __name__ == '__main__':
    main()
