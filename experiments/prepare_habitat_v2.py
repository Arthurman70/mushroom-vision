"""Fetch official SegFormer assets for non-commercial research/evaluation only.

Model weights are optional local assets, excluded from deployment packages.
NVIDIA's model license restricts use to research/evaluation. This adapter is not
a production habitat model. No remote code or pickle checkpoint is loaded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-evaluation', action='store_true', required=True)
    parser.add_argument('--output', type=Path, default=Path('models/segformer-habitat-v2'))
    args = parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    model_id = 'nvidia/segformer-b0-finetuned-ade-512-512'
    revision = HfApi().model_info(model_id).sha
    snapshot_download(model_id, revision=revision, local_dir=args.output,
                      allow_patterns=['config.json', 'preprocessor_config.json', 'model.safetensors', 'README.md'])
    license_url = 'https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE'
    with urllib.request.urlopen(license_url, timeout=60) as response:
        license_bytes = response.read(100000)
    (args.output / 'LICENSE-NVIDIA.txt').write_bytes(license_bytes)
    files = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
             for p in args.output.iterdir() if p.is_file() and p.name != 'mycoscout-model-provenance.json']
    provenance = {'modelId': model_id, 'resolvedRevision': revision,
                  'source': f'https://huggingface.co/{model_id}/tree/{revision}',
                  'licenseSource': license_url, 'useRestriction': 'Non-commercial research/evaluation only',
                  'remoteCodeAllowed': False, 'weightsFormat': 'safetensors', 'files': files}
    (args.output / 'mycoscout-model-provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    print(json.dumps(provenance))


if __name__ == '__main__':
    main()
