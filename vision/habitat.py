"""Semantic scene cues for offline habitat research; never a flight-space map.

An ADE20K model can propose generic vegetation/earth/tree regions. It cannot
establish moisture, moss, tree species, fungal presence, or safe access. Archive
photos deliberately have no metric pose and cannot become navigation targets.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageOps

CATEGORIES = {
    'trees': {'tree'}, 'vegetation': {'plant', 'grass'},
    'ground': {'earth', 'dirt track', 'field'}, 'water': {'water', 'river', 'lake', 'sea'},
    'rock': {'rock', 'mountain', 'hill'}, 'path': {'path', 'road', 'sidewalk'},
    'sky': {'sky'}, 'person': {'person'},
}


def summarize_semantics(labels, confidence, id_to_label, context='unknown', minimum_confidence=0.55):
    """Summarize confident labels; broad-scene context must be reviewed separately.

    Confidence is a model score, not a calibrated ecological probability.
    Lower image regions are only an image-space proxy, never measured ground.
    """
    labels = np.asarray(labels)
    confidence = np.asarray(confidence, dtype=float)
    if labels.ndim != 2 or confidence.shape != labels.shape or labels.size == 0:
        raise ValueError('Expected matching nonempty 2D label/confidence arrays')
    if (isinstance(minimum_confidence, (bool, np.bool_)) or not isinstance(minimum_confidence, Real)
            or not np.isfinite(minimum_confidence) or not 0 <= minimum_confidence <= 1):
        raise ValueError('Minimum confidence must be a finite number within 0..1')
    if not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0):
        raise ValueError('Labels must be nonnegative integer class IDs')
    if not isinstance(id_to_label, Mapping) or not id_to_label:
        raise ValueError('A nonempty class-ID mapping is required')
    known_ids = set()
    for key, name in id_to_label.items():
        valid_key = ((isinstance(key, Integral) and not isinstance(key, (bool, np.bool_)) and key >= 0)
                     or (isinstance(key, str) and key.isascii() and key.isdigit()))
        if not valid_key or not isinstance(name, str) or not name.strip():
            raise ValueError('Class mapping requires nonnegative integer IDs and nonempty names')
        class_id = int(key)
        if class_id in known_ids:
            raise ValueError('Class mapping contains duplicate numeric IDs')
        known_ids.add(class_id)
    if not set(np.unique(labels)).issubset(known_ids):
        raise ValueError('Label array contains IDs absent from the model class mapping')
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError('Confidence must be finite and within 0..1')
    valid = confidence >= minimum_confidence
    lower = np.zeros_like(valid)
    lower[int(labels.shape[0] * 0.4):] = True
    fractions = {}
    lower_fractions = {}
    for category, names in CATEGORIES.items():
        ids = [int(i) for i, name in id_to_label.items() if str(name).split(',')[0].strip().lower() in names]
        mask = np.isin(labels, ids) & valid
        fractions[category] = float(mask.mean())
        lower_fractions[category] = float((mask & lower).sum() / lower.sum())
    signals = {
        'forest_context': min(1.0, fractions['trees'] / 0.3),
        'vegetation_ground': min(1.0, lower_fractions['vegetation'] / 0.5),
        'exposed_ground': min(1.0, lower_fractions['ground'] / 0.5),
    }
    eligible = context == 'context' and (lower_fractions['vegetation'] + lower_fractions['ground'] >= 0.10)
    return {'fractions': fractions, 'lowerImageFractions': lower_fractions,
            'uncertainFraction': float((~valid).mean()), 'signals': signals,
            'inspectionLeadEligible': bool(eligible),
            'eligibilityReason': 'reviewed broad scene with generic ground/vegetation cues' if eligible else
            'needs broad-scene review or enough visible ground context',
            'modelScore': float(confidence[valid].mean()) if valid.any() else 0.0,
            'pixelThreshold': float(minimum_confidence),
            'limitations': ['No moisture, moss, tree-species or fungal-presence inference',
                            'No clearance, depth, traversability or metric location inference']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path('models/segformer-habitat-v2'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results/habitat-v2'))
    parser.add_argument('--research-evaluation', action='store_true', required=True)
    args = parser.parse_args()
    import torch
    from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    torch.set_num_threads(2)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    processor = SegformerImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model, local_files_only=True,
                                                            use_safetensors=True, trust_remote_code=False).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    observations = []
    root = args.manifest.parent.resolve()
    for item in manifest['images']:
        source = (root / item['path']).resolve()
        if not source.is_relative_to(root):
            raise ValueError('Image path outside dataset')
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Image checksum changed')
        image = ImageOps.exif_transpose(Image.open(source)).convert('RGB')
        inputs = processor(images=image, return_tensors='pt')
        started = time.perf_counter()
        with torch.inference_mode():
            logits = model(**inputs).logits
            probabilities = logits.softmax(dim=1)
            confidence, labels = probabilities.max(dim=1)
        elapsed = time.perf_counter() - started
        labels = labels[0].numpy(); confidence = confidence[0].numpy()
        summary = summarize_semantics(labels, confidence, model.config.id2label, item.get('context', 'unknown'))
        row = {'imageId': item['id'], 'imageSha256': item['sha256'], 'context': item.get('context', 'unknown'),
               'inferenceSeconds': elapsed, **summary}
        rows.append(row)
        if summary['inspectionLeadEligible']:
            observations.append({'id': 'habitat-' + item['id'], 'kind': 'habitat', 'frameId': item['id'],
                                 'capturedMs': None, 'confidence': min(0.65, summary['modelScore']),
                                 'source': 'SegFormer ADE20K offline research replay', 'signals': summary['signals'],
                                 'simulationOnly': True, 'archivalPhoto': True,
                                 'context': {'semanticFractions': summary['fractions'], 'captureTimeKnown': False}})
        print(json.dumps({'image': item['id'], 'eligible': summary['inspectionLeadEligible'],
                          'seconds': round(elapsed, 3)}), flush=True)
    report = {'schema': 'mycoscout.habitat-cues.v2',
              'generatedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'model': json.loads((args.model / 'mycoscout-model-provenance.json').read_text(encoding='utf-8')),
              'measuredInference': True, 'ecologicalAccuracyMeasured': False,
              'sourceManifestSha256': hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              'images': rows, 'observations': observations,
              'notes': ['Offline research only; inference is measured but ecological accuracy is unvalidated.',
                        'Capture time is unknown for archive photos; receipt time cannot replace it.',
                        'All leads lack measured depth and pose, so remain unlocated screen leads.',
                        'ADE20K has no mushroom habitat, soil moisture, moss or oak ground-truth classes.']}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
