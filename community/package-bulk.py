"""Publishable originals + separate AI opinions; never publish the private inventory."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re

ROOT = Path(__file__).resolve().parents[1]

def strip_jpeg_metadata(raw):
    """Remove EXIF/IPTC/comments without decoding or changing JPEG image pixels."""
    if not raw.startswith(b'\xff\xd8'):
        raise ValueError('Only verified JPEG originals are supported')
    output = bytearray(raw[:2]); position = 2
    while position < len(raw):
        start = position
        if raw[position] != 255: raise ValueError('Malformed JPEG marker')
        while raw[position] == 255: position += 1
        marker = raw[position]; position += 1
        if marker in (0xDA, 0xD9):
            output.extend(raw[start:]); return bytes(output)
        if marker in range(0xD0, 0xD8) or marker == 1:
            output.extend(raw[start:position]); continue
        length = int.from_bytes(raw[position:position+2], 'big')
        if length < 2 or position+length > len(raw): raise ValueError('Truncated JPEG')
        if marker not in (0xE1, 0xED, 0xFE): output.extend(raw[start:position+length])
        position += length
    raise ValueError('JPEG has no image data')

def allowed_credit(credit):
    license = str(credit.get('license', '')).lower().replace('-', ' ')
    allowed = bool(re.fullmatch(r'cc by(?: sa)?(?: \d+(?:\.\d+)?)?(?: \(youtube\))?|cc0|cdla permissive 1.0', license))
    return credit.get('publicDisplayAllowed') is True and allowed and all(credit.get(k) for k in ('licenseUrl','sourceUrl','attribution'))

def build(inventory_path, ledger_path, output, root=ROOT):
    root=Path(root).resolve(); output=Path(output).resolve()
    inventory=json.loads(Path(inventory_path).read_text(encoding='utf-8'))
    if inventory.get('schema')!='mycoscout.bulk-ai-inventory.v1' or len(inventory.get('records',[]))>10000: raise ValueError('Invalid bounded inventory')
    ledger=json.loads(Path(ledger_path).read_text(encoding='utf-8')) if ledger_path and Path(ledger_path).exists() else {'records':[]}
    audits={}
    for row in sorted(ledger.get('records',[]),key=lambda r:({'contact-sheet':0,'crop':1,'original-image':2}.get(r.get('observation',{}).get('mode'),-1),r.get('createdAt',''))):
        audits[row['imageId']]=row
    existing={}; video_sources={}
    old=root/'data/community-publish-v1/manifest.json'
    if old.exists():
        old_data=json.loads(old.read_text(encoding='utf-8'));existing={f['id']:f for f in old_data['frames']};video_sources={s['id']:s for s in old_data['sources']}
    old_ai=root/'data/community-publish-v1/ai-suggestions.json'
    detectors={r['frameId']:r for r in json.loads(old_ai.read_text(encoding='utf-8')).get('frames',[])} if old_ai.exists() else {}
    output.mkdir(parents=True,exist_ok=True);(output/'images').mkdir(exist_ok=True)
    frames=[];sources={};suggestions=[];restricted=[];metadata_stripped=0
    for record in inventory['records']:
        audit=audits.get(record['id'])
        if audit and audit.get('sha256')!=record['sha256']: raise ValueError('AI ledger hash mismatch')
        if not record['rights']['publicDisplayAllowed']:
            restricted.append({'id':record['id'],'aiReviewed':bool(audit),'reason':'publication-rights-unresolved'});continue
        credits=[c for c in record['rights']['credits'] if allowed_credit(c)]
        if not credits: raise ValueError('Public original lacks explicit reusable credit: '+record['id'])
        credit=credits[0]
        path=(root/record['path']).resolve(strict=True)
        if not path.is_relative_to(root/'data') or path.is_symlink(): raise ValueError('Image path outside data')
        original=path.read_bytes()
        if len(original)>8*1024*1024 or hashlib.sha256(original).hexdigest()!=record['sha256']: raise ValueError('Image changed or oversized')
        clean=strip_jpeg_metadata(original); digest=hashlib.sha256(clean).hexdigest();metadata_stripped+=clean!=original
        alias=next((a for a in record['aliases'] if a['collection']=='video-discovery-v1'),None)
        identifier=alias['id'] if alias else record['id']
        # Keep already-published originals byte-for-byte so human reviews keep their identity.
        if identifier in existing and digest!=existing[identifier]['sha256']: raise ValueError('Metadata removal would change an existing human-review image')
        is_video=record['kind']=='video-frame'
        source_id=(alias or record)['sourceId'] if is_video else 'photo-'+digest[:24]
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',source_id): source_id='source-'+hashlib.sha256(source_id.encode()).hexdigest()[:24]
        prior=video_sources.get(source_id,{})
        source={'id':source_id,'title':prior.get('title') or ('Licensed video sequence' if is_video else 'Licensed research photo'),
                'attribution':credit['attribution'],'license':credit['license'],'licenseUrl':credit['licenseUrl'],'sourceUrl':credit['sourceUrl'],
                'changes':'Original dimensions and JPEG image pixels retained; EXIF/IPTC/comments removed when present.'}
        if source_id in sources and sources[source_id]!=source: raise ValueError('Source attribution collision')
        sources[source_id]=source
        target=output/'images'/(digest+'.jpg')
        if not target.exists():
            if clean==original:
                try:os.link(path,target)
                except OSError:target.write_bytes(clean)
            else:target.write_bytes(clean)
        elif hashlib.sha256(target.read_bytes()).hexdigest()!=digest: raise ValueError('Existing published file changed')
        frame={'id':identifier,'bulkImageId':record['id'],'sourceId':source_id,'kind':'video-frame' if is_video else 'photo',
               'width':record['width'],'height':record['height'],'timestampSeconds':record.get('timestampSeconds') or 0,'sha256':digest,
               'originalSha256':record['sha256'],'path':'images/'+digest+'.jpg','sourceGroupId':record['sourceGroupId'],
               'videoHash':record.get('videoHash'),'evaluationOnly':record['split']!='train',
               'sourceLicenseAllowsTraining':record['rights']['experimentalTrainingPermission'] is True,'actualDistanceM':None}
        frames.append(frame)
        old_row=detectors.get(identifier,{})
        boxes=old_row.get('boxes',[]) if old_row.get('imageSha256')==digest else []
        if audit:
            review_id=audit['id']
            audit={k:audit[k] for k in ('imageId','reviewerKind','humanVerified','status','complete','boxes','confounders','confounderTags','observation','evidence','revision') if k in audit}
            audit['reviewId']=review_id;audit['humanVerified']=False;audit['trainingAllowed']=False
            aliases={'leaf':'leaves','other':'other_non_mushroom','bark':'tree_bark','clothing':'hat'}
            audit['confounderTags']=[aliases.get(kind,kind) for kind in audit.get('confounderTags',[])]
            audit['confounders']=[{**item,'kind':aliases.get(item['kind'],item['kind'])} for item in audit.get('confounders',[])]
        suggestions.append({'frameId':identifier,'sourceId':source_id,'imageSha256':digest,'status':'completed' if audit else 'pending',
                            'model':'AI visual review; unverified','boxes':boxes,'audit':audit})
    manifest={'schema':'mycoscout.community-collection.v1','sources':list(sources.values()),'frames':frames,'restrictedRecords':restricted,
              'privacy':'Licensed image pixels only; no private source aliases, uploader identities, GPS or local file paths.',
              'training':'AI and community opinions remain unverified. Preview reviews do not become training ground truth.'}
    ai={'schema':'mycoscout.public-ai-suggestions.v1','frames':suggestions,'run':{'status':'completed' if len(audits)==len(inventory['records']) else 'in-progress',
        'reviewedFrames':sum(bool(s['audit']) for s in suggestions),'totalFrames':len(frames),'humanVerified':False}}
    for name,data in [('manifest.json',manifest),('ai-suggestions.json',ai)]:
        temporary=output/(name+'.tmp');temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(output/name)
    return {'publicImages':len(frames),'sources':len(sources),'rightsLimited':len(restricted),'visualAiReviews':sum(bool(s['audit']) for s in suggestions),'metadataStripped':metadata_stripped,'output':str(output)}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--inventory',type=Path,default=ROOT/'data/bulk-ai-review-v1/manifest.json');parser.add_argument('--ledger',type=Path,default=ROOT/'data/bulk-ai-review-v1/ledger.json');parser.add_argument('--output',type=Path,default=ROOT/'data/community-publish-all-v1');args=parser.parse_args()
    print(json.dumps(build(args.inventory,args.ledger,args.output),indent=2))
