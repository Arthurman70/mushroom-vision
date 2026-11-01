"""Append licensed images to an unpublished clone without changing old review identities."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import io
import json
import re
import tarfile
from copy import deepcopy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('community_deploy', HERE/'deploy.py')
deploy = importlib.util.module_from_spec(spec); spec.loader.exec_module(deploy)
spec = importlib.util.spec_from_file_location('public_packager', ROOT/'community/package-bulk.py')
public = importlib.util.module_from_spec(spec); spec.loader.exec_module(public)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2)+'\n').encode()


APPEND_VALIDATOR = r'''
import re,hashlib
def validate_append(old,old_ai,new,new_ai,files,added_ids):
 if new.get('schema')!='mycoscout.community-collection.v1' or new_ai.get('schema')!='mycoscout.public-ai-suggestions.v1':raise ValueError('Unknown append schema')
 for key in set(old)|set(new):
  if key not in ('sources','frames') and old.get(key)!=new.get(key):raise ValueError('Existing collection metadata changed')
 for key in ('sources','frames'):
  if not isinstance(new.get(key),list) or new[key][:len(old[key])]!=old[key]:raise ValueError('Existing review identity changed')
 for key in set(old_ai)|set(new_ai):
  if key not in ('frames','run') and old_ai.get(key)!=new_ai.get(key):raise ValueError('Existing AI metadata changed')
 if not isinstance(new_ai.get('frames'),list) or new_ai['frames'][:len(old_ai['frames'])]!=old_ai['frames']:raise ValueError('Existing AI opinions changed')
 added=new['frames'][len(old['frames']):];new_sources=new['sources'][len(old['sources']):]
 if not 1<=len(added)<=500 or len(new['frames'])>10000 or len(new['sources'])>10000 or added_ids!=[r.get('id') for r in added]:raise ValueError('Unexpected new frames')
 if len({r.get('id') for r in new['frames']})!=len(new['frames']) or len({r.get('id') for r in new['sources']})!=len(new['sources']):raise ValueError('Repeated frame or source identity')
 if len(new_sources)!=len(added):raise ValueError('Every added photo needs one distinct source')
 sources={s['id']:s for s in new_sources};expected={'collection/manifest.json','collection/ai-suggestions.json'}
 known_hashes={r['sha256'] for r in old['frames']};groups={}
 for frame in new['frames']:
  group=frame.get('sourceGroupId',frame.get('sourceId'));groups.setdefault(group,set()).add(frame.get('evaluationOnly'))
 for frame in added:
  keys={'id','sourceId','kind','width','height','timestampSeconds','sha256','originalSha256','path','sourceGroupId','videoHash','evaluationOnly','sourceLicenseAllowsTraining','actualDistanceM'}
  if set(frame)!=keys or not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',str(frame['id'])) or frame['kind']!='photo':raise ValueError('Invalid appended image identity')
  if type(frame['width']) is not int or type(frame['height']) is not int or min(frame['width'],frame['height'])<1 or frame['width']*frame['height']>20000000:raise ValueError('Invalid appended dimensions')
  if not re.fullmatch(r'[a-f0-9]{64}',str(frame['sha256'])) or not re.fullmatch(r'[a-f0-9]{64}',str(frame['originalSha256'])) or frame['path']!='images/'+frame['sha256']+'.jpg':raise ValueError('Invalid appended image path')
  if frame['sha256'] in known_hashes:raise ValueError('Repeated published image bytes')
  known_hashes.add(frame['sha256'])
  if frame['timestampSeconds']!=0 or frame['videoHash'] is not None or frame['actualDistanceM'] is not None or type(frame['evaluationOnly']) is not bool or frame['sourceLicenseAllowsTraining'] is not True:raise ValueError('Invalid photo partition')
  if not isinstance(frame['sourceGroupId'],str) or not frame['sourceGroupId'] or len(frame['sourceGroupId'])>240 or re.search(r'[\x00-\x1f\x7f]',frame['sourceGroupId']) or len(groups[frame['sourceGroupId']])!=1:raise ValueError('Invalid or conflicting source partition')
  if frame['sourceId']!='photo-'+frame['sha256'][:24] or frame['sourceId'] not in sources:raise ValueError('Invalid added source identity')
  source=sources[frame['sourceId']]
  if set(source)!={'id','title','attribution','license','licenseUrl','sourceUrl','changes'} or source['license'] not in ('CC-BY','CC-BY-SA','CC0'):raise ValueError('Publication rights missing')
  if not all(isinstance(source.get(k),str) and 0<len(source[k])<=4000 and not re.search(r'[\x00-\x1f\x7f]',source[k]) for k in ('title','attribution','licenseUrl','sourceUrl','changes')):raise ValueError('Invalid source attribution')
  if not re.fullmatch(r'https://www\.inaturalist\.org/observations/[0-9]+',source['sourceUrl']):raise ValueError('Unexpected source URL')
  license_path={'CC-BY':r'licenses/by/[234]\.0/','CC-BY-SA':r'licenses/by-sa/[234]\.0/','CC0':r'publicdomain/zero/1\.0/'}[source['license']]
  if not re.fullmatch('https://creativecommons\\.org/'+license_path,source['licenseUrl']):raise ValueError('License URL does not match declared rights')
  name='collection/'+frame['path'];expected.add(name)
  if name not in files or len(files[name])>8*1024*1024 or hashlib.sha256(files[name]).hexdigest()!=frame['sha256']:raise ValueError('Image hash mismatch')
 if set(files)!=expected:raise ValueError('Unreferenced files')
 pending=[{'frameId':f['id'],'sourceId':f['sourceId'],'imageSha256':f['sha256'],'status':'pending','model':'Not yet reviewed','boxes':[],'audit':None} for f in added]
 if new_ai['frames'][len(old_ai['frames']):]!=pending:raise ValueError('New images must start with exact pending AI entries')
 before=old_ai.get('run',{});after=new_ai.get('run',{})
 if not isinstance(after,dict) or after.get('totalFrames')!=len(new['frames']) or after.get('status')!='in-progress' or after.get('humanVerified') is not False:raise ValueError('Invalid append run state')
 for key in set(before)|set(after):
  if key not in ('totalFrames','status','humanVerified') and before.get(key)!=after.get(key):raise ValueError('Existing AI run metadata changed')
 return added
'''
validator_namespace = {}; exec(APPEND_VALIDATOR, validator_namespace)
validate_append = validator_namespace['validate_append']


def build(snapshot, source, output):
    from PIL import Image, ImageOps
    snapshot = json.loads(Path(snapshot).read_text(encoding='utf8'))
    source = Path(source).resolve(strict=True)
    dataset = json.loads(source.read_text(encoding='utf8'))
    manifest, suggestions = snapshot['manifest'], snapshot['suggestions']
    old_manifest, old_suggestions = deepcopy(manifest), deepcopy(suggestions)
    before = {'manifest': digest(encoded(manifest)), 'suggestions': digest(encoded(suggestions))}
    if manifest.get('schema') != 'mycoscout.community-collection.v1' or suggestions.get('schema') != 'mycoscout.public-ai-suggestions.v1':
        raise ValueError('Unknown public snapshot schema')
    old_ids = {r['id'] for r in manifest['frames']}
    old_hashes = {r['sha256'] for r in manifest['frames']}
    old_sources = {r['id'] for r in manifest['sources']}
    additions = {}; added = []; image_bytes = 0
    if not isinstance(dataset.get('images'),list) or not 1<=len(dataset['images'])<=500:raise ValueError('Append batch outside bounds')
    for row in dataset['images']:
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,160}',str(row.get('id',''))) or row.get('split') not in ('train','validation','test'):
            raise ValueError('Invalid image ID or training split')
        if not isinstance(row.get('sourceGroupId'),str) or not row['sourceGroupId'] or len(row['sourceGroupId'])>240:
            raise ValueError('Missing source group')
        if row['id'] in old_ids: raise ValueError('Existing image ID cannot be replaced')
        license = row.get('photoLicense', {})
        if license.get('code') not in {'cc-by','cc-by-sa','cc0'} or not all(license.get(k) for k in ('url','attribution')):
            raise ValueError('Photo lacks a permitted publication license')
        if not re.fullmatch(r'https://www\.inaturalist\.org/observations/\d+',row['sourceUrl']):
            raise ValueError('Expected observation provenance')
        declared = source.parent/row['path']
        path = declared.resolve(strict=True)
        if not path.is_relative_to(source.parent) or declared.is_symlink(): raise ValueError('Unexpected image path')
        raw = path.read_bytes()
        if digest(raw) != row['sha256'] or len(raw)>8*1024*1024: raise ValueError('Image changed or oversized')
        with Image.open(io.BytesIO(raw)) as image:
            oriented = ImageOps.exif_transpose(image).convert('RGB')
            if oriented.size != (row['width'],row['height']): raise ValueError('Oriented dimensions changed')
            # Most source JPEGs are already upright. Nontrivial EXIF orientation
            # needs baked pixels before stripping metadata, or boxes would rotate.
            if image.getexif().get(274,1) not in (None,1):
                buffer=io.BytesIO();oriented.save(buffer,format='JPEG',quality=95)
                clean=buffer.getvalue();changes='EXIF orientation baked; JPEG re-encoded at quality 95; metadata removed.'
            else:
                clean=public.strip_jpeg_metadata(raw);changes='Original dimensions and JPEG pixels retained; EXIF/IPTC/comments removed.'
        sha = digest(clean)
        image_bytes += len(clean)
        if image_bytes > 740*1024*1024: raise ValueError('Append image payload exceeds bounded staging capacity')
        if sha in old_hashes: raise ValueError('Duplicate published image bytes')
        sid = 'photo-'+sha[:24]
        if sid in old_sources: raise ValueError('Source collision')
        frame = {'id':row['id'],'sourceId':sid,'kind':'photo','width':row['width'],'height':row['height'],
            'timestampSeconds':0,'sha256':sha,'originalSha256':row['sha256'],'path':'images/'+sha+'.jpg',
            'sourceGroupId':row['sourceGroupId'],'videoHash':None,'evaluationOnly':row['split']!='train',
            'sourceLicenseAllowsTraining':True,'actualDistanceM':None}
        manifest['frames'].append(frame)
        manifest['sources'].append({'id':sid,'title':'Licensed research photo', 'attribution':license['attribution'],
            'license':license['code'].upper(),'licenseUrl':license['url'],'sourceUrl':row['sourceUrl'],'changes':changes})
        suggestions['frames'].append({'frameId':frame['id'],'sourceId':sid,'imageSha256':sha,'status':'pending',
            'model':'Not yet reviewed','boxes':[],'audit':None})
        additions['collection/'+frame['path']]=clean;added.append(frame['id'])
        old_ids.add(frame['id']);old_hashes.add(sha);old_sources.add(sid)
    if not 1<=len(added)<=500 or len(manifest['frames'])>10000: raise ValueError('Append batch outside bounds')
    suggestions.setdefault('run',{})['totalFrames']=len(manifest['frames'])
    suggestions['run']['status']='in-progress';suggestions['run']['humanVerified']=False
    additions['collection/manifest.json']=encoded(manifest)
    additions['collection/ai-suggestions.json']=encoded(suggestions)
    if sum(len(raw) for raw in additions.values())>768*1024*1024:raise ValueError('Append payload exceeds bounded staging capacity')
    validate_append(old_manifest,old_suggestions,manifest,suggestions,additions,added)
    receipt={'schema':'mycoscout.community-append.v1','base':before,'addedIds':added,
        'files':{name:{'bytes':len(raw),'sha256':digest(raw)} for name,raw in additions.items()}}
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'x:gz',compresslevel=1) as archive:
        for name,raw in {**additions,'append.json':encoded(receipt)}.items():
            info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;archive.addfile(info,io.BytesIO(raw))
    output.with_suffix('.receipt.json').write_bytes(encoded(receipt))
    return {'bundle':str(output),'added':len(added),'totalFrames':len(manifest['frames']),'bytes':output.stat().st_size}


REMOTE = APPEND_VALIDATOR + r'''
import pathlib,sys,re,json,tarfile,hashlib,os,io
base=pathlib.Path('/root/gleaner-mushroom-lab');name=sys.argv[1]
if not re.fullmatch(r'[a-z0-9-]{6,80}',name):raise ValueError('Invalid release')
release=(base/'releases'/name).resolve(strict=True)
state=json.loads((base/'current.json').read_text())
if release.parent!=base/'releases' or state['release']==str(release):raise ValueError('Only unpublished managed clones may be extended')
if (release/'UPLOAD-INCOMPLETE').exists():raise ValueError('Incomplete release')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def enc(v):return (json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode()
raw=sys.stdin.buffer.read(768*1024*1024+1)
if len(raw)>768*1024*1024:raise ValueError('Append archive too large')
files={};size=0
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
 for member in archive:
  if not member.isfile() or member.name in files or member.size>20*1024*1024:raise ValueError('Invalid member')
  if member.name not in ('append.json','collection/manifest.json','collection/ai-suggestions.json') and not re.fullmatch(r'collection/images/[a-f0-9]{64}\.jpg',member.name):raise ValueError('Unexpected member')
  size+=member.size
  if size>768*1024*1024 or len(files)>502:raise ValueError('Append exceeds bounds')
  files[member.name]=archive.extractfile(member).read()
receipt=json.loads(files.pop('append.json'))
if receipt.get('schema')!='mycoscout.community-append.v1' or set(files)!=set(receipt['files']):raise ValueError('Invalid append receipt')
for name,blob in files.items():
 if receipt['files'][name]!={'bytes':len(blob),'sha256':sha(blob)}:raise ValueError('Append content mismatch')
old=json.loads((release/'collection/manifest.json').read_text());old_ai=json.loads((release/'collection/ai-suggestions.json').read_text())
new=json.loads(files['collection/manifest.json']);new_ai=json.loads(files['collection/ai-suggestions.json'])
if receipt['base']!={'manifest':sha(enc(old)),'suggestions':sha(enc(old_ai))}:raise ValueError('Stale base; snapshot and rebuild')
added=validate_append(old,old_ai,new,new_ai,files,receipt['addedIds'])
integrity_path=release/'release-manifest.json';integrity=json.loads(integrity_path.read_text())
if integrity_path.is_symlink():raise ValueError('Unexpected release manifest symlink')
for name in files:
 target=release/name
 if target.is_symlink() or not target.parent.resolve(strict=True).is_relative_to(release):raise ValueError('Append target escapes managed release')
# Publish is a separate operation. Leave a fail-closed marker if staging is interrupted.
marker=release/'UPLOAD-INCOMPLETE';marker.write_text('Appending verified licensed originals')
for name,blob in files.items():
 target=release/name
 if target.is_symlink():raise ValueError('Unexpected symlink')
 if name.startswith('collection/images/') and target.exists():raise ValueError('Image already exists')
 temporary=target.with_suffix(target.suffix+'.append')
 with temporary.open('xb') as out:out.write(blob);out.flush();os.fsync(out.fileno())
 temporary.replace(target);integrity['files'][name]={'bytes':len(blob),'sha256':sha(blob)}
integrity['frames']=len(new['frames']);integrity_path.write_bytes(enc(integrity));marker.unlink()
print(json.dumps({'staged':True,'added':len(added),'frames':len(new['frames']),'published':False}))
'''


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='action',required=True)
    build_parser=sub.add_parser('build');build_parser.add_argument('--snapshot',required=True,type=Path)
    build_parser.add_argument('--source',required=True,type=Path);build_parser.add_argument('--output',required=True,type=Path)
    upload=sub.add_parser('stage');upload.add_argument('--bundle',required=True,type=Path);upload.add_argument('--release',required=True)
    args=parser.parse_args()
    if args.action=='build':print(json.dumps(build(args.snapshot,args.source,args.output)))
    else:
        with args.bundle.open('rb') as stream:deploy.ssh_command(['python3','-c',REMOTE,deploy.release_name(args.release)],stdin=stream)
