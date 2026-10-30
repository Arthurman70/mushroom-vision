"""Publish aggregate research evidence; never copy photos, names or coordinates."""
import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def main():
    dataset = read('data/public-fungi-v2/summary.json')
    study = read('experiments/results/v2-transfer-01/study.json')
    discovery = read('experiments/results/v2-discovery-comparison.json')
    flight = read('flight/reports/habitat-latest.json')
    cue = flight.get('archiveInference', {})
    ranking_rows = [{'label': label, **study['models'][key]['metrics']['test']}
                    for key, label in [('handcrafted', 'Color + texture'), ('resnet18', 'Learned visual features')]]
    labels = {'generic': 'Generic · 12 photos', 'black-aware': 'Black prompt · 12 photos',
              'generic-context-subset': 'Generic · 4 context photos',
              'generic-tiled': 'Generic + tiles · same 4',
              'black-aware-context-subset': 'Black prompt · same 4',
              'black-aware-tiled': 'Black + tiles · same 4'}
    rows = [{'label': label, **discovery['comparisons'][key]['evaluated']['all']}
            for key, label in labels.items() if key in discovery['comparisons']]
    report = {
        'schema': 'mycoscout.scout-lab.v2', 'generatedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'dataset': {'imageCount': dataset['images'], 'blackTrumpetCount': dataset['weakLabels']['black_trumpet'],
                    'observerCount': dataset['observers'], 'bytes': dataset['bytes'], 'photoLicenses': dataset['photoLicenses']},
        'ranking': {'rows': ranking_rows, 'testCount': study['splitCounts']['test']['photos'],
                    'selectedByValidation': study['selectedFamilyByValidation'],
                    'testUncertainty': study['models']['resnet18']['testUncertainty'],
                    'conclusion': 'The learned ranker recalled 94.9% of black-trumpet-labeled test photos, versus 57.6% for color and texture. Precision was 77.7%, below the 80% validation target. It ranks photos or candidate crops for review; it does not establish discovery range or species identity.',
                    'limitations': study['limitations']},
        'discovery': {'rows': rows,
                      'notes': '12 selected photos / 30 approximate boxes, frozen before predictions; overlap threshold 0.5. All photos contain fungi. Compare tiling only within the same four context photos. More detections can bring more false boxes; generic detection remains the default. No physical range has been measured.',
                      'benchmarkManifestSha256': discovery['benchmarkManifestSha256']},
        'habitat': {'imageCount': cue.get('imageCount'), 'screenLeadCount': cue.get('screenLeadCount'),
                    'mapTargetCount': cue.get('mapTargetCount'), 'meanInferenceSeconds': cue.get('meanInferenceSeconds'),
                    'notes': 'Actual scene segmentation on 12 archive photos proposed nine unlocated inspection leads; three close-up controls were withheld. Ecological accuracy is unmeasured. The optional SegFormer model is licensed for research/evaluation only and is excluded from deployment packages.'},
        'search': {key: flight[key] for key in ['simulationOnly','actuationEnabled','summary','mapTargets','screenLeads','schedule','scenarios','targetPixelEstimates']},
    }
    stress_path = ROOT / 'experiments/results/v2-discovery-scale-stress.json'
    if stress_path.exists():
        stress = json.loads(stress_path.read_text(encoding='utf-8'))
        report['discovery']['scaleStress'] = {
            'rows': [{'scale': row['relativeSceneScale'], **row['metrics']} for row in stress['levels']],
            'sourceOriginalPhotos': stress['sourceOriginalPhotos'],
            'note': 'Recall on four repeated scenes shrunk within a fixed canvas. Blur, JPEG and padding are artificial; these are not measured distances. At ⅛ scale, all eight targets smaller than 16 pixels were missed.'}
    (ROOT / 'public/scout-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'publicReport': 'public/scout-report.json', 'photos': dataset['images'],
                      'realMapLocations': cue.get('mapTargetCount'), 'scenarios': flight['summary']}))


if __name__ == '__main__':
    main()
