"""Connect saved vision candidates to the flight computer's unlocated lead queue.

Archive photographs have no synchronized depth, capture time or calibrated pose.
This bridge deliberately emits no metric targets, even if input metadata claims
otherwise. Rankings annotate review order; no detector candidate is filtered.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from .dataset import valid_box

DETECTOR_SCHEMA = 'mycoscout.unlocated-candidates.v1'
COLOR_SCHEMA = 'mycoscout.color-proposals.v1'
COLOR_MEANING = 'Local color contrast, not mushroom/species probability'


def probability(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def archive_flags(document):
    for key in ('evaluationOnly', 'synthetic'):
        if key in document and not isinstance(document[key], bool):
            raise ValueError(f'{key} must be boolean when supplied')
    synthetic = document.get('synthetic', False)
    return {'synthetic': synthetic, 'evaluationOnly': document.get('evaluationOnly', False) or synthetic}


def archive_cue_evidence(candidate, schema):
    """Whitelist research evidence; saved photos can never supply flight range.

    Both the review inbox and scout bridge use this contract. Synthetic distance,
    arbitrary caller routing, metric pose and capture times are deliberately not
    accepted. A larger derived crop also never counts as a close field capture.
    """
    if schema not in {DETECTOR_SCHEMA, COLOR_SCHEMA} or not isinstance(candidate, dict):
        raise ValueError('Unsupported candidate evidence schema')
    cue_type = 'color_salience' if schema == COLOR_SCHEMA else candidate.get('cueType', 'detector')
    if cue_type not in {'detector', 'color_salience', 'habitat'}:
        raise ValueError('Unsupported cueType')
    is_color = cue_type == 'color_salience'
    kind = 'fungus' if cue_type == 'detector' else 'habitat'
    result = {'cueType': cue_type, 'classificationEligible': cue_type == 'detector',
        'investigation': {
            'routingCategory': 'habitat_followup' if kind == 'habitat' else 'unlocated_followup',
            'originalKind': kind, 'rangeEvidence': {'status': 'unknown', 'source': 'archival_photo',
                'reason': 'No synchronized measured range'},
            'humanConfirmationRequired': True, 'photoStatus': 'closer_photo_requested',
            'reason': 'color_cue' if is_color else 'habitat_cue' if kind == 'habitat' else 'range_unknown',
            'actuationEnabled': False}}
    evidence = candidate.get('colorEvidence')
    if evidence is None and is_color:
        evidence = {'method': 'lab-local-contrast-v1', 'maxDeltaE': candidate.get('deltaE'),
                    'meanDeltaE': candidate.get('deltaE'), 'contrastScore': candidate.get('contrastScore'),
                    'enhancedViewUsed': False}
    if evidence is not None:
        if not isinstance(evidence, dict) or evidence.get('method') != 'lab-local-contrast-v1':
            raise ValueError('Unsupported color evidence method')
        if not probability(evidence.get('contrastScore')) or not isinstance(evidence.get('enhancedViewUsed'), bool):
            raise ValueError('Color evidence requires a bounded contrast score and enhancement flag')
        for key in ('maxDeltaE', 'meanDeltaE'):
            value = evidence.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 400:
                raise ValueError('Color DeltaE values must be finite and between 0 and 400')
        if evidence['meanDeltaE'] > evidence['maxDeltaE']:
            raise ValueError('Mean color contrast cannot exceed maximum contrast')
        result['colorEvidence'] = {key: evidence[key] for key in ('method', 'maxDeltaE', 'meanDeltaE', 'contrastScore', 'enhancedViewUsed')}
        result['colorEvidence']['scoreMeaning'] = COLOR_MEANING
    return result


def candidate_leads(prediction, sources):
    schema = prediction.get('schema')
    if schema not in {DETECTOR_SCHEMA, COLOR_SCHEMA}:
        return []
    if schema == COLOR_SCHEMA and prediction.get('colorOnly') is not True:
        raise ValueError('Color proposals must explicitly declare colorOnly:true')
    flags = archive_flags(prediction)
    meta = prediction.get('image', {})
    sha = meta.get('sha256')
    source = sources.get(sha)
    if source is None or (meta.get('width'), meta.get('height')) != (source['width'], source['height']):
        raise ValueError('Candidate source hash/dimensions do not match the frozen source manifest')
    if not isinstance(prediction.get('candidates'), list) or len(prediction['candidates']) > 1000:
        raise ValueError('Explicit candidate list of at most 1000 entries required')
    result = []
    for index, candidate in enumerate(prediction['candidates']):
        if not isinstance(candidate, dict):
            raise ValueError('Candidate must be an object')
        score = candidate.get('contrastScore') if schema == COLOR_SCHEMA else candidate.get('confidence')
        if not probability(score) or not valid_box(candidate.get('bboxPixelsXYXY'), source['width'], source['height']):
            raise ValueError('Invalid candidate score or image-space box')
        evidence = archive_cue_evidence(candidate, schema)
        color_only = evidence['cueType'] == 'color_salience'
        row = {'id': f'{"color" if color_only else "visual"}-{sha[:16]}-{index}',
               'kind': evidence['investigation']['originalKind'], 'frameId': 'archive-' + sha,
               'capturedMs': None, 'archivalPhoto': True, 'confidence': score,
               **flags,
               'source': 'Saved color-contrast cue; fungal presence unknown' if color_only else 'Saved detector candidate; not species confirmed',
               **evidence,
               'bbox': {'xyxy': candidate['bboxPixelsXYXY'], 'imageWidth': source['width'], 'imageHeight': source['height']}}
        ranking = candidate.get('reviewRanking')
        if isinstance(ranking, dict):
            score = ranking.get('blackTrumpetPatternScore')
            if not probability(score):
                raise ValueError('Invalid review ranker score')
            row['reviewMetadata'] = {'blackTrumpetPatternScore': score,
                                     'prioritizeHumanReview': ranking.get('prioritizeHumanReview') is True,
                                     'scoreCalibrated': False, 'speciesConfirmed': False,
                                     'candidateFilteringEnabled': False, 'flightGateEnabled': False}
        result.append(row)
    return result


def habitat_leads(report):
    if report.get('schema') != 'mycoscout.habitat-cues.v2':
        raise ValueError('Expected offline habitat cue report')
    rows = []
    for index, obs in enumerate(report.get('observations', [])):
        if not probability(obs.get('confidence')):
            raise ValueError('Invalid habitat cue score')
        signals = {key: value for key, value in obs.get('signals', {}).items()
                   if key in {'forest_context', 'vegetation_ground', 'exposed_ground'} and probability(value)}
        rows.append({'id': f'archive-habitat-{index}', 'kind': 'habitat', 'frameId': f'archive-habitat-frame-{index}',
                     'capturedMs': None, 'archivalPhoto': True, 'confidence': obs['confidence'],
                     'source': 'Offline semantic research cue; ecological accuracy unmeasured', 'signals': signals})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--detections', type=Path, required=True)
    parser.add_argument('--habitat-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new output file to preserve prior evidence')
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    sources = {row['sha256']: row for row in manifest['images']}
    root = args.manifest.parent.resolve()
    for row in sources.values():
        image = (root / row['path']).resolve()
        if not image.is_relative_to(root) or hashlib.sha256(image.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Source image missing, escaped dataset or changed checksum')
    paths = sorted(args.detections.glob('*.json')) if args.detections.is_dir() else [args.detections]
    observations = []
    for path in paths:
        observations.extend(candidate_leads(json.loads(path.read_text(encoding='utf-8')), sources))
    candidate_count = len(observations)
    visual_count = sum(row['kind'] == 'fungus' for row in observations)
    color_count = sum(row.get('cueType') == 'color_salience' for row in observations)
    if args.habitat_report:
        observations.extend(habitat_leads(json.loads(args.habitat_report.read_text(encoding='utf-8'))))
    if len({o['id'] for o in observations}) != len(observations):
        raise ValueError('Repeated candidate IDs; provide one detector configuration per photo')
    if not observations or len(observations) > 10000:
        raise ValueError('Expected 1..10000 archival leads')
    result = {'schema': 'mycoscout.archival-scout-leads.v2', 'simulationOnly': True, 'actuationEnabled': False,
              'observations': observations, 'summary': {'visualCandidates': visual_count,
              'habitatContexts': len(observations) - visual_count, 'colorOnlyCues': color_count,
              'imageCandidates': candidate_count, 'metricTargets': 0, 'candidatesFiltered': 0},
              'notes': ['Capture time, synchronized depth and calibrated pose are absent.',
                        'All observations remain screen-only; none can start an aircraft approach.',
                        'Review ranker metadata does not alter detector confidence or flight eligibility.',
                        'Color-only contrast proposals are habitat follow-ups, never confirmed mushrooms.',
                        'Synthetic pixel-scale distances and caller-supplied range claims are not actual observation ranges.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
