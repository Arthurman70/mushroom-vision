"""Replay a fixed licensed evaluation photo into the review inbox, without human labels."""
import argparse
import json
from pathlib import Path
from adapters.review_stream import Station, send_frame

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--station',default='http://127.0.0.1:8787')
    p.add_argument('--photo-id',default='inat-photo-749288059')
    p.add_argument('--classify',action='store_true')
    args=p.parse_args()
    manifest_path=ROOT/'data/public-fungi-v2/manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    row=next(r for r in manifest['images'] if r['id']==args.photo_id)
    if row['split']!='test':
        raise ValueError('This replay requires a held-out test photo')
    result=send_frame(Station(args.station),manifest_path.parent/row['path'],session_id='archive-replay-v4',source='archive_replay',
        evaluation_only=True,classify=args.classify,metadata={'dataset':'public-fungi-v2','originalSplit':row['split'],
            'photoId':row['photoId'],'attribution':row['photoLicense']['attribution'],'licenseUrl':row['photoLicense']['url'],
            'sourceUrl':row['sourceUrl'],'archivalPhoto':True})
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
