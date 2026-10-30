"""One bounded, inactive detector experiment from explicitly AI-reviewed frames.

This never writes human reviews, production training records or active models.
The checkpoint uses a separate schema rejected by the operational loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
HASH = re.compile(r'^[a-f0-9]{64}$')
RANK = {'unassigned': 0, 'train': 1, 'validation': 2, 'test': 3}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path, maximum=16 * 1024 * 1024):
    if path.stat().st_size > maximum:
        raise ValueError('Experiment metadata exceeds its bounded size')
    raw = path.read_bytes()
    def invalid_constant(value):
        raise ValueError('Non-finite JSON number: ' + value)
    return raw, json.loads(raw, parse_constant=invalid_constant)


def freeze_reservations(manifest, policy, learning_dir):
    """The strictest known reservation follows source, video and pixel links."""
    if not (learning_dir / 'datasets').is_dir():
        raise ValueError('Learning dataset directory is missing; cannot check frozen holdouts')
    reserved, reserved_hashes, reserved_pixels = {}, {}, {}
    def reserve(mapping, key, split):
        if isinstance(key, str) and RANK[split] > RANK.get(mapping.get(key), -1):
            mapping[key] = split
    for group, split in policy['sourceGroups'].items():
        reserve(reserved, 'source:' + group, split)
    for path in sorted(learning_dir.glob('datasets/*/manifest.json')):
        _, frozen = read_json(path)
        if frozen.get('schema') != 'mycoscout.reviewed-detection-dataset.v1' or not isinstance(frozen.get('records'), list):
            raise ValueError('Frozen human dataset is unreadable; cannot lower its reservations')
        for row in frozen['records']:
            split = row.get('split')
            if split not in ('train', 'validation', 'test'):
                raise ValueError('Invalid frozen partition')
            if (not isinstance(row.get('groupKeys'), list) or any(not isinstance(key, str) for key in row['groupKeys'])
                    or not isinstance(row.get('originalSha256'), str) or not HASH.fullmatch(row['originalSha256'])
                    or row.get('pixelSha256') is not None and not HASH.fullmatch(row['pixelSha256'])):
                raise ValueError('Invalid frozen provenance; cannot check its holdouts')
            for key in row.get('groupKeys', []):
                reserve(reserved, key, split)
            reserve(reserved_hashes, row.get('originalSha256'), split)
            reserve(reserved_pixels, row.get('pixelSha256'), split)
    sources = {source['id']: source for source in manifest['sources']}
    parents, components, owners = {}, {}, {}
    def find(key):
        parents.setdefault(key, key)
        if parents[key] != key:
            parents[key] = find(parents[key])
        return parents[key]
    for frame in manifest['frames']:
        source = sources[frame['sourceId']]
        keys = ['source:' + source['sourceGroupId'], 'video:' + source.get('videoHash', source.get('sha256', '')), 'pixels:' + frame['sha256']]
        for key in keys[1:]:
            parents[find(key)] = find(keys[0])
        owners[frame['id']] = keys[0]
        if source.get('evaluationOnly') is not False or frame.get('evaluationOnly') is True:
            reserve(reserved, keys[0], 'test')
        reserve(reserved, keys[2], reserved_hashes.get(frame['sha256'], 'unassigned'))
    for key in parents:
        root = find(key)
        reserve(components, root, reserved.get(key, 'unassigned'))
    return {frame_id: components[find(key)] for frame_id, key in owners.items()}, reserved_pixels


def load_records(exports, collection_dir, partitions_file, learning_dir):
    _, manifest = read_json(collection_dir / 'manifest.json')
    if manifest.get('schema') != 'mycoscout.video-collection.v1' or not isinstance(manifest.get('frames'), list) or not isinstance(manifest.get('sources'), list):
        raise ValueError('Invalid original video collection')
    frames = {f['id']: f for f in manifest['frames']}
    sources = {s['id']: s for s in manifest['sources']}
    if len(frames) != len(manifest['frames']) or len(sources) != len(manifest['sources']) or len(frames) > 2000:
        raise ValueError('Duplicate or excessive source/frame identities')
    _, policy = read_json(partitions_file)
    if (policy.get('schema') != 'mycoscout.ai-review-partitions.v1' or not isinstance(policy.get('sourceGroups'), dict)
            or any(split not in ('train', 'validation', 'test') for split in policy['sourceGroups'].values())):
        raise ValueError('Unknown partition policy')
    reservations, reserved_pixels = freeze_reservations(manifest, policy, learning_dir)
    rows, seen, pixels, evidence = [], set(), {}, []
    if not 1 <= len(exports) <= 20:
        raise ValueError('Use one to twenty finite AI review exports')
    for path in exports:
        raw, data = read_json(path)
        if (data.get('schema') != 'mycoscout.ai-visual-review-export.v1'
                or data.get('reviewerKind') != 'ai' or data.get('humanVerified') is not False
                or data.get('productionTrainingAllowed') is not False or data.get('automaticActivation') is not False
                or data.get('labelAuthority') != 'ai-visual-review' or not isinstance(data.get('records'), list)
                or len(data['records']) > 100):
            raise ValueError('Use an explicit AI-only review export')
        evidence.append({'runId': data['runId'], 'sha256': digest(raw)})
        for row in data['records']:
            if (row.get('schema') != 'mycoscout.ai-visual-review.v1' or row.get('reviewerKind') != 'ai'
                    or row.get('humanVerified') is not False or row.get('trainingAllowed') is not False
                    or row.get('automaticActivation') is not False or row.get('runId') != data['runId']
                    or row.get('labelAuthority') != 'ai-visual-review' or row.get('reviewer') != data.get('reviewer')
                    or not isinstance(row.get('reviewer'), dict) or row['reviewer'].get('kind') != 'ai'
                    or not isinstance(row['reviewer'].get('model'), str) or not row['reviewer']['model'].strip()):
                raise ValueError('AI authority fields changed')
            if (row.get('complete') is not True or row.get('status') not in ('visible', 'absent')
                    or row.get('viewedOriginal') is not True or not isinstance(row.get('modelSuggestionsViewed'), bool)
                    or not isinstance(row.get('evidence'), str) or not row['evidence'].strip()):
                raise ValueError('Incomplete or uncertain records cannot fit this experiment')
            frame = frames[row['frameId']]
            source = sources[frame['sourceId']]
            group = source['sourceGroupId']
            split = row['split']
            keys = {'source:' + group, 'video:' + source.get('videoHash', source.get('sha256', ''))}
            if (split not in ('train', 'validation') or policy['sourceGroups'].get(group) != split
                    or row['sourceGroupId'] != group or row['sourceId'] != source['id']
                    or row['sha256'] != frame['sha256'] or row['videoHash'] != source.get('videoHash', source.get('sha256'))
                    or source.get('evaluationOnly') is not False or frame.get('evaluationOnly') is True
                    or source.get('licenseAllowsTraining') is not True
                    or reservations[frame['id']] != split):
                raise ValueError('A source, reservation or license is incompatible with this experiment')
            if row.get('experimentalTrainingAllowed') is not (split == 'train'):
                raise ValueError('Missing explicit experimental training eligibility')
            if (any(type(row.get(key)) is not int or row[key] != frame.get(key) for key in ('width', 'height'))
                    or row['width'] < 1 or row['height'] < 1 or row['width'] * row['height'] > 20000000):
                raise ValueError('Original dimensions disagree with the collection')
            image_path = (collection_dir / frame['path']).resolve()
            if not image_path.is_relative_to(collection_dir.resolve()):
                raise ValueError('Frame path escaped its collection')
            if image_path.stat().st_size > 8 * 1024 * 1024 or not HASH.fullmatch(row['sha256']) or digest(image_path.read_bytes()) != row['sha256']:
                raise ValueError('Frame bytes changed')
            with Image.open(image_path) as image:
                image = image.convert('RGB')
                if image.size != (row['width'], row['height']):
                    raise ValueError('Frame dimensions changed')
                pixel_hash = digest(json.dumps(list(image.size), separators=(',', ':')).encode() + image.tobytes())
            if RANK[reserved_pixels.get(pixel_hash, 'unassigned')] > RANK[split]:
                raise ValueError('Decoded pixels were already reserved for a stricter partition')
            if pixel_hash in pixels:
                raise ValueError('Duplicate decoded pixels in AI experiment')
            pixels[pixel_hash] = split
            if row['frameId'] in seen:
                raise ValueError('Repeated frame review')
            seen.add(row['frameId'])
            if not isinstance(row.get('boxes'), list) or len(row['boxes']) > 128 or (row['status'] == 'visible') != bool(row['boxes']):
                raise ValueError('Presence and boxes disagree')
            for box in row['boxes']:
                bbox = box.get('bboxPixelsXYXY')
                if (box.get('reviewerKind') != 'ai' or box.get('humanVerified') is not False
                        or not isinstance(bbox, list) or len(bbox) != 4
                        or any(type(n) not in (int, float) or not math.isfinite(n) for n in bbox)
                        or type(box.get('confidence')) not in (int, float) or not math.isfinite(box['confidence'])
                        or not 0 <= box['confidence'] <= 1 or not isinstance(box.get('evidence'), str)
                        or not box['evidence'].strip()):
                    raise ValueError('Invalid AI box authority, confidence or evidence')
                x1, y1, x2, y2 = bbox
                if not (0 <= x1 < x2 <= row['width'] and 0 <= y1 < y2 <= row['height']):
                    raise ValueError('Invalid original pixel box')
            rows.append({**row, 'groupKeys': sorted(keys), 'imagePath': frame['path'], 'leakageGroup': group, 'id': row['frameId']})
    train = [r for r in rows if r['split'] == 'train']
    if len(train) < 4 or len({r['leakageGroup'] for r in train}) < 2 or sum(bool(r['boxes']) for r in train) < 2:
        raise ValueError('Need at least four frames, two sources and two positive frames for the bounded experiment')
    return rows, evidence


def train_candidate(exports, collection_dir, partitions_file, learning_dir, output):
    from vision.training_pipeline import build_model, training_example, evaluate, MODEL_NAME, WEIGHT_SHA256
    import torch
    rows, evidence = load_records(exports, collection_dir, partitions_file, learning_dir)
    output.mkdir(parents=True, exist_ok=False)
    random.seed(4127)
    torch.manual_seed(4127)
    torch.set_num_threads(4)
    model = build_model(pretrained=True)
    train = [r for r in rows if r['split'] == 'train']
    validation = [r for r in rows if r['split'] == 'validation']
    before = evaluate(model, collection_dir, validation, .3)[0] if validation else None
    weights_before = model.roi_heads.box_predictor.cls_score.weight.detach().clone()
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.005, momentum=.9, weight_decay=.0005)
    random.shuffle(train)
    model.train()
    model.backbone.body.eval()
    started = time.time()
    losses = []
    for row in train:
        image, target = training_example(collection_dir, row, augment=True)
        loss = sum(model([image], [target]).values())
        if not torch.isfinite(loss):
            raise ValueError('Non-finite loss; no candidate saved')
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 10)
        optimizer.step()
        losses.append({'frameId': row['frameId'], 'loss': float(loss.detach())})
        print(json.dumps({'step': len(losses), 'steps': len(train), 'loss': losses[-1]['loss']}), flush=True)
    after = evaluate(model, collection_dir, validation, .3)[0] if validation else None
    checkpoint = {'schema': 'mushroom-vision.ai-candidate-checkpoint.v1', 'architecture': MODEL_NAME,
        'state_dict': model.state_dict(), 'reviewerKind': 'ai', 'humanVerified': False,
        'activationEligible': False, 'epochs': 1, 'scoreThreshold': .3, 'sourceExports': evidence}
    torch.save(checkpoint, output / 'ai-candidate.pt')
    restored = torch.load(output / 'ai-candidate.pt', map_location='cpu', weights_only=True)
    probe = build_model(pretrained=False)
    probe.load_state_dict(restored['state_dict'])
    change = float((weights_before - model.roi_heads.box_predictor.cls_score.weight.detach()).abs().max())
    report = {'schema': 'mushroom-vision.ai-candidate-run.v1', 'completed': True, 'epochs': 1,
        'optimizerSteps': len(train), 'trainFrames': len(train), 'trainSources': len({r['sourceGroupId'] for r in train}),
        'trainPositiveFrames': sum(bool(r['boxes']) for r in train), 'trainBoxes': sum(len(r['boxes']) for r in train),
        'validationFrames': len(validation), 'validationBefore': before, 'validationAfter': after,
        'labelAuthority': 'ai-visual-review', 'humanVerified': False, 'activationEligible': False,
        'productionModelChanged': False, 'testEvaluationPerformed': False, 'sourceExports': evidence,
        'pretrainedInitializationSha256': WEIGHT_SHA256, 'seed': 4127, 'learningRate': .005,
        'device': 'cpu', 'threads': 4, 'elapsedSeconds': time.time() - started, 'losses': losses,
        'maxClassifierWeightChange': change, 'checkpointReloadVerified': True,
        'checkpointSha256': digest((output / 'ai-candidate.pt').read_bytes()),
        'limitations': ['Tiny AI-labeled development sample; not human verified.',
            'One optimizer pass demonstrates the experiment workflow, not improved recognition.',
            'Validation agreement, if present, uses AI labels and is not independent human accuracy.',
            'No reserved test footage was used to fit or choose this candidate.',
            'This separate checkpoint schema cannot be activated through the station model loader.']}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', type=Path, action='append', required=True, dest='exports')
    parser.add_argument('--collection', type=Path, default=ROOT / 'data/video-discovery-v1')
    parser.add_argument('--partitions', type=Path, default=ROOT / 'config/ai-review-partitions-v1.json')
    parser.add_argument('--learning', type=Path, default=ROOT / 'data/learning')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = train_candidate(args.exports, args.collection, args.partitions, args.learning, args.output)
    print(json.dumps({key: result[key] for key in ('completed', 'optimizerSteps', 'trainFrames', 'validationFrames', 'maxClassifierWeightChange', 'productionModelChanged')}))
