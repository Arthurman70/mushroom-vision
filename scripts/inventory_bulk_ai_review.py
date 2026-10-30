"""Freeze a deduplicated inventory of real acquisition and submitted images."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
from datetime import datetime,timezone
from PIL import Image, ImageOps,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
COLLECTIONS=['biodiversity-fungi-v1','public-fungi-v3','public-fungi-v2','black-trumpet-discovery-v2','facebook-michigan-fungus-hunters','wildlife-context-v1','video-discovery-v1','morel-research-frames-v1','morel-video-feed-v1','morel-video-frames-v1-batch-1','morel-video-frames-v1-batch-1b','morel-video-frames-v1-batch-2','morel-video-frames-v1-batch-3','morel-video-frames-v1-batch-4']
RANK={'unassigned':0,'train':1,'validation':2,'test':3}
def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def inspect(root):
    result=[]
    for name in COLLECTIONS:
        path=root/'data'/name/'manifest.json'
        if path.exists():
            data=read(path);result.append({'collection':name,'schema':data.get('schema'),'images':len(data.get('images',[])),'frames':len(data.get('frames',[]))})
    for name in ['finds','testbench']:
        path=root/'data'/name/(name+'.sqlite')
        if path.exists():
            with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
                tables={row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                result.append({'database':name,'tables':sorted(tables),'counts':{table:db.execute('SELECT COUNT(*) FROM '+table).fetchone()[0] for table in ['photos','runs','finds','observations'] if table in tables}})
    return result

def collect(root):
    items=[];excluded=[];policy=read(root/'config/ai-review-partitions-v1.json')['sourceGroups']
    for name in COLLECTIONS:
        path=root/'data'/name/'manifest.json'
        if not path.exists():continue
        data=read(path);manifest_hash=digest(path.read_bytes());sources={row['id']:row for row in data.get('sources',[])}
        for row in data.get('images',[])+data.get('frames',[]):
            if row.get('synthetic') is True:
                excluded.append({'collection':name,'id':row['id'],'reason':'synthetic'});continue
            video=bool(data.get('frames'));source=sources.get(row.get('sourceId'),{}) if video else row
            license_info=row.get('photoLicense',row.get('license',data.get('license',{})))
            if isinstance(license_info,str):license_info={'code':license_info}
            license_code=source.get('license','') if video else license_info.get('code','')
            public=source.get('licenseAllowsTraining') is True if video else license_code.lower() in ['cc0','cc-by','cc-by-sa','cdla-permissive-1.0']
            group=source.get('sourceGroupId') if video else row.get('splitGroupId',row.get('groupId','image:'+row['sha256']))
            split=policy.get(group,'unassigned') if video else row.get('split','unassigned')
            evaluation=source.get('evaluationOnly') is True or row.get('evaluationOnly') is True or 'facebook' in name
            if evaluation:split='test'
            items.append({'id':row['id'],'collection':name,'path':(path.parent/row['path']).relative_to(root).as_posix(),'sha256':row['sha256'],
                'kind':'video-frame' if video else 'wildlife-photo' if 'wildlife' in name else 'fungi-photo','sourceId':source.get('id',row['id']),
                'sourceGroupId':group,'videoHash':source.get('videoHash',source.get('sha256')) if video else None,'timestampSeconds':row.get('timestampSeconds'),
                'split':split,'evaluationOnly':evaluation,'groupKeys':['source:'+group]+(['video:'+source.get('videoHash',source.get('sha256'))] if video else []),
                'rights':{'license':license_code,'licenseUrl':source.get('licenseUrl','') if video else license_info.get('url',''),'attribution':source.get('attribution','') if video else license_info.get('attribution',row.get('rightsHolder','')),
                    'sourceUrl':source.get('sourceUrl',''),'publicDisplayAllowed':public,'experimentalTrainingPermission':public,'restriction':row.get('rightsStatus','')},
                'sourceManifest':path.relative_to(root).as_posix(),'sourceManifestSha256':manifest_hash})
    # Originals explicitly submitted to the two live local databases. Never scan
    # temporary exports, generated crops, fixtures or screenshots of dashboards.
    for name in ['finds','testbench']:
        path=root/'data'/name/(name+'.sqlite')
        if not path.exists():continue
        with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
            db.row_factory=sqlite3.Row;photos={p['hash']:dict(p) for p in db.execute('SELECT * FROM photos')}
            rows=[dict(row) for row in db.execute('SELECT * FROM '+('finds' if name=='finds' else 'runs'))]
        for row in rows:
            metadata=json.loads(row['data']) if name=='finds' else {}
            if metadata.get('metadata',{}).get('parentImageSha256'):
                excluded.append({'collection':name,'id':row['id'],'reason':'derived-detector-crop'});continue
            photo=photos[row['photo_hash']];credits=json.loads(photo.get('credits','{}'));provenance=json.loads(row['video_provenance']) if name=='testbench' and row.get('video_provenance') else None
            if provenance:
                source=provenance['source'];group=source['sourceGroupId'];keys=['source:'+group,'video:'+source['videoHash']];split=policy.get(group,'unassigned')
            else:
                group=metadata.get('sessionId',row.get('scene_id','submitted:'+row['photo_hash']));keys=['source:'+group];split=metadata.get('initialPartition','unassigned')
            evaluation=metadata.get('evaluationOnly') is True or photo.get('evaluation_only')==1
            if evaluation:split='test'
            # Unknown submitted-photo rights stay local, even if the submitter
            # allowed model training. Existing licensed aliases can authorize use.
            items.append({'id':row['id'],'collection':name,'path':'data/'+name+'/photos/'+photo['hash']+'.'+photo['extension'],'sha256':photo['hash'],
                'kind':'video-frame' if provenance else 'submitted-photo','sourceId':provenance['sourceId'] if provenance else row['id'],
                'sourceGroupId':group,'videoHash':provenance['source']['videoHash'] if provenance else None,'timestampSeconds':provenance.get('timestampSeconds') if provenance else None,
                'split':split,'evaluationOnly':evaluation,'groupKeys':keys,
                'rights':{'license':credits.get('license','Unknown submitted-image rights'),'licenseUrl':credits.get('licenseUrl',''),'attribution':credits.get('attribution',''),
                    'sourceUrl':credits.get('sourceUrl',''),'publicDisplayAllowed':False,'experimentalTrainingPermission':metadata.get('rights',{}).get('trainingOptIn') is True,'restriction':'Local submitted image; redistribution permission not inferred'},
                'sourceManifest':'data/'+name+'/'+name+'.sqlite','sourceManifestSha256':None})
    return items,excluded

def freeze(root,output,sheets=False):
    output.mkdir(parents=True,exist_ok=True);items,excluded=collect(root);by_hash={};pixels={};records=[]
    for item in items:
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root.resolve()):raise ValueError('Outside source path')
        if not path.exists():excluded.append({'collection':item['collection'],'id':item['id'],'reason':'missing-original'});continue
        raw=path.read_bytes()
        if digest(raw)!=item['sha256']:raise ValueError('Source hash changed: '+item['path'])
        if item['sha256'] in by_hash:
            by_hash[item['sha256']]['aliases'].append(item);continue
        with Image.open(path) as image:
            orientation=image.getexif().get(274,1);image=ImageOps.exif_transpose(image).convert('RGB');width,height=image.size
            pixel_hash=digest(json.dumps([width,height],separators=(',',':')).encode()+image.tobytes())
        if pixel_hash in pixels:
            pixels[pixel_hash]['aliases'].append(item);by_hash[item['sha256']]=pixels[pixel_hash];continue
        record={'id':'image-'+item['sha256'],'sha256':item['sha256'],'pixelSha256':pixel_hash,'path':item['path'],'width':width,'height':height,
            'orientationApplied':orientation!=1,'kind':item['kind'],'sourceId':item['sourceId'],'sourceGroupId':item['sourceGroupId'],'videoHash':item['videoHash'],
            'timestampSeconds':item['timestampSeconds'],'aliases':[item]}
        records.append(record);by_hash[item['sha256']]=record;pixels[pixel_hash]=record
        if len(records)%500==0:print(json.dumps({'verifiedImages':len(records)}),flush=True)
    # All recorded datasets and source manifests reserve their strictest groups.
    protected={};protected_pixels={};protected_hashes={}
    def reserve(mapping,key,split):
        if key is not None and RANK.get(split,0)>RANK.get(mapping.get(key),-1):mapping[key]=split
    for path in sorted((root/'data/learning/datasets').glob('*/manifest.json')):
        for row in read(path).get('records',[]):
            for key in row.get('groupKeys',[]):reserve(protected,key,row['split'])
            reserve(protected_hashes,row.get('originalSha256'),row['split']);reserve(protected_pixels,row.get('pixelSha256'),row['split'])
    parents={}
    def find(key):
        parents.setdefault(key,key)
        if parents[key]!=key:parents[key]=find(parents[key])
        return parents[key]
    for record in records:
        keys=set()
        for alias in record['aliases']:
            keys.update(alias['groupKeys']);keys.add('hash:'+alias['sha256'])
            for key in alias['groupKeys']:reserve(protected,key,alias['split'])
            reserve(protected,'hash:'+alias['sha256'],protected_hashes.get(alias['sha256'],'unassigned'))
        keys.add('pixels:'+record['pixelSha256']);reserve(protected,'pixels:'+record['pixelSha256'],protected_pixels.get(record['pixelSha256'],'unassigned'))
        record['groupKeys']=sorted(keys)
        first=record['groupKeys'][0]
        for key in record['groupKeys'][1:]:parents[find(key)]=find(first)
    component={}
    for key in list(parents):reserve(component,find(key),protected.get(key,'unassigned'))
    prior={}
    prior_db=root/'data/ai-visual-review/ai-visual-review.sqlite'
    if prior_db.exists():
        with sqlite3.connect(prior_db.as_uri()+'?mode=ro',uri=True) as db:
            for raw, in db.execute('SELECT data FROM ai_reviews ORDER BY revision'):
                review=json.loads(raw);prior[review['sha256']]=review
    for record in records:
        record['split']=component[find(record['groupKeys'][0])]
        record['rights']={'publicDisplayAllowed':any(a['rights']['publicDisplayAllowed'] for a in record['aliases']),
            'experimentalTrainingPermission':any(a['rights']['experimentalTrainingPermission'] for a in record['aliases']),
            'credits':list({json.dumps(a['rights'],sort_keys=True):a['rights'] for a in record['aliases'] if a['rights']['publicDisplayAllowed']}.values())}
        record['experimentalTrainingEligible']=record['split']=='train' and record['rights']['experimentalTrainingPermission']
        matches=[prior[a['sha256']] for a in record['aliases'] if a['sha256'] in prior]
        record['priorAiReviews']=matches;record['humanVerified']=False;record['reviewStatus']='prior-ai-original-review' if matches else 'unreviewed'
    priority={'video-frame':0,'submitted-photo':1,'fungi-photo':2,'wildlife-photo':3}
    records.sort(key=lambda r:(priority[r['kind']],r['sourceId'] if r['kind']=='video-frame' else r['aliases'][0]['collection'],r['timestampSeconds'] or 0,r['id']))
    for index,record in enumerate(records):record['index']=index;record['sheetId']='sheet-'+str(index//16).zfill(4);record['tileIndex']=index%16
    manifest={'schema':'mycoscout.bulk-ai-inventory.v1','createdAt':datetime.now(timezone.utc).isoformat(),'records':records,'excluded':excluded,
        'summary':{'uniqueImages':len(records),'sourceReferences':sum(len(r['aliases']) for r in records),'duplicateReferences':sum(len(r['aliases'])-1 for r in records),
            'kinds':dict(Counter(r['kind'] for r in records)),'splits':dict(Counter(r['split'] for r in records)),'publicDisplayAllowed':sum(r['rights']['publicDisplayAllowed'] for r in records),
            'priorAiOriginalReviews':sum(bool(r['priorAiReviews']) for r in records),'excludedReferences':len(excluded)},
        'scope':'Manifest-backed original photos and extracted video frames plus live local finds/testbench submissions. Synthetic sources, derived crops, QA fixtures and temporary exports are excluded.',
        'reviewProtocol':{'sheetTileMaximumSide':384,'sheetImages':16,'sheetReviewMode':'contact-sheet','sheetReviewsAreComplete':False,'sheetReviewsAreTrainingEligible':False,
            'positiveIdentificationFromSourceLabels':False,'confounders':['hat','stump','tree','leaf','rock','wood','person','animal','other']}}
    raw=(json.dumps(manifest,indent=2)+'\n').encode();(output/'manifest.json').write_bytes(raw);(output/'manifest-sha256.json').write_text(json.dumps({'sha256':digest(raw)})+'\n')
    print(json.dumps(manifest['summary']),flush=True)
    if sheets:make_sheets(root,output,manifest)
    return manifest

def make_sheets(root,output,manifest):
    (output/'sheets').mkdir(exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18) if Path('C:/Windows/Fonts/arial.ttf').exists() else ImageFont.load_default()
    metadata=[]
    for start in range(0,len(manifest['records']),16):
        records=manifest['records'][start:start+16];canvas=Image.new('RGB',(4*400,4*420),'#e9e6db');draw=ImageDraw.Draw(canvas);tiles=[]
        for i,row in enumerate(records):
            x,y=(i%4)*400,(i//4)*420
            with Image.open(root/row['path']) as image:
                image=ImageOps.exif_transpose(image).convert('RGB');image.thumbnail((384,384));left=x+(400-image.width)//2;top=y+28+(384-image.height)//2;canvas.paste(image,(left,top))
                tiles.append({'index':row['index'],'id':row['id'],'sha256':row['sha256'],'displayedWidth':image.width,'displayedHeight':image.height,'bboxSheetPixelsXYXY':[left,top,left+image.width,top+image.height]})
            draw.text((x+9,y+5),str(row['index']).zfill(4)+'  '+row['sha256'][:12],font=font,fill='#18382c')
        sheet_id='sheet-'+str(start//16).zfill(4);file=output/'sheets'/(sheet_id+'.jpg');canvas.save(file,quality=95)
        metadata.append({'id':sheet_id,'path':file.relative_to(output).as_posix(),'sha256':digest(file.read_bytes()),'width':1600,'height':1680,'tiles':tiles})
    (output/'sheets.json').write_text(json.dumps({'schema':'mycoscout.bulk-ai-sheets.v1','sheets':metadata},indent=2)+'\n',encoding='utf-8');print(json.dumps({'sheets':len(metadata)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--freeze',action='store_true');parser.add_argument('--sheets',action='store_true');parser.add_argument('--output',type=Path,default=ROOT/'data/bulk-ai-review-v1');args=parser.parse_args()
    if args.freeze:freeze(args.root.resolve(),args.output.resolve(),args.sheets)
    else:print(json.dumps(inspect(args.root.resolve()),indent=2))
