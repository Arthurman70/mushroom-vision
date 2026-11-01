"""Narrow, backed-up links to the public AI contribution guide."""
from pathlib import Path
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import uuid
import xml.etree.ElementTree as ET
from deploy import ssh_command

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / 'data' / 'ai-contribution-publication'
NAMES = ('index.html', 'articles/mushroom-vision.html', 'sitemap.xml')
EXISTING = NAMES[:3]
LIMIT = 4 * 1024 * 1024
SCHEMA = 'gleaner.ai-contribution-publication.v1'
ID = re.compile(r'ai-contribution-\d{8}T\d{6}Z-[a-f0-9]{32}')

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf8')

def read_bounded(path, limit=LIMIT):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError('Missing, linked or oversized file: '+path.name)
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError('File exceeded its byte limit')
    return raw

def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temp.open('xb') as output:
        output.write(json.dumps(value, indent=2).encode('utf8'))
        output.flush(); os.fsync(output.fileno())
    temp.replace(path)

def prepare(staged=WORK/'staged', original=WORK/'original.json'):
    staged = Path(staged).absolute()
    if staged.is_symlink() or staged.resolve() != staged:
        raise ValueError('Staging directory must not contain a symlink')
    snapshot_data = json.loads(read_bounded(Path(original), 12*1024*1024))
    actual = {path.relative_to(staged).as_posix() for path in staged.rglob('*') if path.is_file() or path.is_symlink()}
    if actual != set(NAMES):
        raise ValueError('Staging must contain exactly the three allowed files')
    files = []
    for name in NAMES:
        path = staged/name
        if path.resolve() != path:
            raise ValueError('Staged path traverses a symlink')
        raw = read_bounded(path)
        if not raw or b'\0' in raw:
            raise ValueError('Staged files must be nonempty UTF-8 text')
        raw.decode('utf8')
        item = snapshot_data.get(name, {})
        before = None
        if name in EXISTING:
            text = item.get('text')
            if not isinstance(text, str) or digest(text.encode('utf8')) != item.get('sha256'):
                raise ValueError('Original snapshot byte/hash mismatch: '+name)
            before = item['sha256']
        elif item.get('exists', False) is not False:
            raise ValueError('A new publication path already existed in the snapshot')
        files.append({'name': name, 'beforeSha256': before, 'afterSha256': digest(raw),
                      'bytes': len(raw), 'data': base64.b64encode(raw).decode('ascii')})
        if name == 'sitemap.xml':
            if b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
                raise ValueError('Sitemap declarations are not allowed')
            if ET.fromstring(raw).tag != '{http://www.sitemaps.org/schemas/sitemap/0.9}urlset':
                raise ValueError('Sitemap must be a standard urlset')
    return {'schema': SCHEMA, 'id': 'ai-contribution-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex,
            'files': files}

# A fixed script and fixed public allowlist; payload text never becomes shell
# code. A receipt is written before any replacement, including exact backup
# bytes and modes. Retrying a saved request resumes only its before/after states.
REMOTE = r'''
from pathlib import Path
import base64,datetime,fcntl,hashlib,json,os,re,stat,subprocess,sys,uuid
import xml.etree.ElementTree as ET
BASE=Path('/root/gleaner')
BACKUPS=Path('/root/gleaner-mushroom-lab/backups')
NAMES=('index.html','articles/mushroom-vision.html','sitemap.xml')
SCHEMA='gleaner.ai-contribution-publication.v1'
LIMIT=4*1024*1024
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf8')
def safe(path):
 if path.is_symlink() or path.resolve()!=path:raise ValueError('Unexpected linked path')
 if path.exists() and not path.is_file():raise ValueError('Public path is not a file')
 return path
def read(path):
 safe(path)
 if path.stat().st_size>LIMIT:raise ValueError('Oversized public file')
 raw=path.read_bytes()
 if len(raw)>LIMIT:raise ValueError('Oversized public file')
 return raw
def current(path):return sha(read(path)) if safe(path).exists() else None
def atomic(path,raw,mode=0o600,uid=None,gid=None,new=False):
 safe(path)
 if path.parent.resolve()!=path.parent or not path.parent.is_dir():raise ValueError('Unexpected parent directory')
 temp=path.parent/('.mushroom-vision-'+uuid.uuid4().hex+'.tmp')
 try:
  with temp.open('xb') as output:
   output.write(raw);output.flush();os.fsync(output.fileno())
  if uid is not None:os.chown(temp,uid,gid)
  os.chmod(temp,mode)
  if new:os.link(temp,path);temp.unlink()
  else:os.replace(temp,path)
  fd=os.open(path.parent,os.O_RDONLY)
  try:os.fsync(fd)
  finally:os.close(fd)
 finally:
  if temp.exists():temp.unlink()
def save(path,value):atomic(path,json.dumps(value,indent=2).encode('utf8'))
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
action=sys.argv[1]
if action not in ('publish','rollback','status'):raise ValueError('Unknown publication action')
raw=sys.stdin.buffer.read(30*1024*1024+1)
if len(raw)>30*1024*1024:raise ValueError('Oversized request')
request=json.loads(raw)
if request.get('schema')!=SCHEMA or not re.fullmatch(r'ai-contribution-\d{8}T\d{6}Z-[a-f0-9]{32}',str(request.get('id',''))):raise ValueError('Invalid publication request')
files=request.get('files')
if not isinstance(files,list) or [row.get('name') for row in files]!=list(NAMES):raise ValueError('Unexpected publication files')
contents={}
for row in files:
 name=row['name'];content=base64.b64decode(row['data'],validate=True)
 if not content or len(content)>LIMIT or len(content)!=row['bytes'] or sha(content)!=row['afterSha256'] or b'\0' in content:raise ValueError('Invalid staged bytes')
 content.decode('utf8')
 if name in NAMES[:3]:
  if not re.fullmatch('[a-f0-9]{64}',str(row.get('beforeSha256',''))):raise ValueError('Missing original hash')
 elif row.get('beforeSha256') is not None:raise ValueError('New page already existed')
 contents[name]=content
xml=contents['sitemap.xml']
if b'<!DOCTYPE' in xml.upper() or b'<!ENTITY' in xml.upper() or ET.fromstring(xml).tag!='{http://www.sitemaps.org/schemas/sitemap/0.9}urlset':raise ValueError('Invalid sitemap')
if BASE.resolve()!=BASE or not BASE.is_dir() or BACKUPS.resolve()!=BACKUPS:raise ValueError('Unexpected site or backup directory')
BACKUPS.mkdir(parents=True,exist_ok=True)
request_sha=sha(canonical(request));backup=BACKUPS/request['id'];receipt_path=backup/'receipt.json'
lock_path=BACKUPS/'.mushroom-vision.lock'
if lock_path.is_symlink():raise ValueError('Linked lock file')
with lock_path.open('a+b') as lock:
 fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
 for row in files:safe(BASE/row['name'])
 if receipt_path.exists():
  receipt=json.loads(read(receipt_path))
  if receipt.get('schema')!=SCHEMA or receipt.get('requestSha256')!=request_sha or receipt.get('id')!=request['id']:raise ValueError('Backup receipt does not match request')
 elif action!='publish':raise ValueError('No saved remote publication receipt; nothing to roll back')
 else:
  # All original hashes and all new-path absence checks happen before backup or publication.
  for row in files:
   if current(BASE/row['name'])!=row['beforeSha256']:raise ValueError('Site changed since snapshot: '+row['name'])
  if backup.exists():raise ValueError('Incomplete backup directory; use a new request only after inspection')
  backup.mkdir(mode=0o700)
  receipt={'schema':SCHEMA,'id':request['id'],'requestSha256':request_sha,'backup':str(backup),'status':'preparing','files':[],'createdAt':now()}
  # Save a recoverable journal before the first backup is written.
  for row in files:
   target=BASE/row['name'];info=target.stat() if target.exists() else None
   receipt['files'].append({**{k:row[k] for k in ('name','beforeSha256','afterSha256')},'existed':info is not None,
    'mode':stat.S_IMODE(info.st_mode) if info else 0o644,'uid':info.st_uid if info else os.getuid(),'gid':info.st_gid if info else os.getgid()})
  save(receipt_path,receipt)
 if len(receipt.get('files',[]))!=len(files) or [r.get('name') for r in receipt['files']]!=list(NAMES):raise ValueError('Incomplete backup receipt')
 for row,meta in zip(files,receipt['files']):
  if any(meta.get(k)!=row.get(k) for k in ('name','beforeSha256','afterSha256')) or meta.get('existed')!=(row['beforeSha256'] is not None):raise ValueError('Receipt file identity changed')
  if not isinstance(meta.get('mode'),int) or not 0<=meta['mode']<=0o7777 or not all(isinstance(meta.get(k),int) and meta[k]>=0 for k in ('uid','gid')):raise ValueError('Invalid original modes')
 if action=='status':
  result={**receipt,'currentFiles':[{'name':r['name'],'sha256':current(BASE/r['name'])} for r in files]}
  print(json.dumps(result));sys.exit(0)
 # A finished operation is immutable: a later deliberate reversion must not be
 # mistaken for an interrupted write and reapplied by a retry.
 if action=='publish' and receipt['status']=='published':
  if any(current(BASE/r['name'])!=r['afterSha256'] for r in files):raise ValueError('Published site was edited later; refusing to reapply')
  print(json.dumps(receipt));sys.exit(0)
 if action=='rollback' and receipt['status']=='rolled_back':
  if any(current(BASE/r['name'])!=r['beforeSha256'] for r in files):raise ValueError('Rolled-back site was edited later; refusing to overwrite')
  print(json.dumps(receipt));sys.exit(0)
 if action=='publish':
  if receipt['status']=='rolled_back':raise ValueError('Rolled-back operations cannot be republished; prepare a new request')
  # Finish interrupted backups before allowing any public file replacement.
  for row,meta in zip(files,receipt['files']):
   target=BASE/row['name'];copy=backup/'original'/row['name']
   if meta['existed']:
    if copy.exists():
     if sha(read(copy))!=row['beforeSha256']:raise ValueError('Backup bytes changed')
    else:
     if receipt['status']!='preparing' or current(target)!=row['beforeSha256']:raise ValueError('Missing original backup')
     copy.parent.mkdir(parents=True,exist_ok=True)
     original=read(target)
     if sha(original)!=row['beforeSha256']:raise ValueError('Original changed while backing up')
     atomic(copy,original,meta['mode'],meta['uid'],meta['gid'],new=True)
  # No mutation if any path has been edited outside this exact operation.
  for row in files:
   if current(BASE/row['name']) not in (row['beforeSha256'],row['afterSha256']):raise ValueError('Concurrent site edit: '+row['name'])
  receipt['status']='publishing';save(receipt_path,receipt)
  # Install the page and its stylesheet before exposing navigation links.
  paired={row['name']:(row,meta) for row,meta in zip(files,receipt['files'])}
  for name in ('articles/mushroom-vision.html','index.html','sitemap.xml'):
   row,meta=paired[name]
   target=BASE/row['name'];found=current(target)
   if found==row['afterSha256']:continue
   if found!=row['beforeSha256']:raise ValueError('Concurrent site edit: '+row['name'])
   atomic(target,contents[row['name']],meta['mode'],meta['uid'],meta['gid'],new=not meta['existed'])
  if any(current(BASE/r['name'])!=r['afterSha256'] for r in files):raise ValueError('Published bytes changed during verification')
  receipt.update(status='published',publishedAt=now());save(receipt_path,receipt)
 else:
  # Precheck the whole rollback before touching any file; partial retries accept
  # already restored before states but refuse subsequent unrelated edits.
  for row,meta in zip(files,receipt['files']):
   found=current(BASE/row['name'])
   if found not in (row['beforeSha256'],row['afterSha256']):raise ValueError('Rollback would overwrite subsequent edits: '+row['name'])
   if meta['existed'] and found==row['afterSha256']:
    if sha(read(backup/'original'/row['name']))!=row['beforeSha256']:raise ValueError('Original backup is invalid')
  receipt['status']='rolling_back';save(receipt_path,receipt)
  for row,meta in zip(files,receipt['files']):
   target=BASE/row['name'];found=current(target)
   if found==row['beforeSha256']:continue
   if found!=row['afterSha256']:raise ValueError('Concurrent edit during rollback')
   if meta['existed']:atomic(target,read(backup/'original'/row['name']),meta['mode'],meta['uid'],meta['gid'])
   else:target.unlink()
  if any(current(BASE/r['name'])!=r['beforeSha256'] for r in files):raise ValueError('Rollback verification changed')
  receipt.update(status='rolled_back',rolledBackAt=now());save(receipt_path,receipt)
 print(json.dumps(receipt))
'''

def load_request(receipt_path):
    receipt = json.loads(read_bounded(receipt_path, 2*1024*1024))
    if receipt.get('schema') != SCHEMA or not ID.fullmatch(str(receipt.get('id', ''))):
        raise ValueError('Invalid local publication receipt')
    request_path = receipt_path.with_name(receipt['id']+'.request.json')
    request = json.loads(read_bounded(request_path, 30*1024*1024))
    if request.get('id') != receipt['id'] or digest(canonical(request)) != receipt.get('requestSha256'):
        raise ValueError('Saved publication request changed')
    return receipt, request

def operate(action, receipt_path=None):
    if action == 'publish' and receipt_path is None:
        request = prepare()
        receipt_path = WORK/'receipts'/(request['id']+'.json')
        receipt = {'schema': SCHEMA, 'id': request['id'], 'requestSha256': digest(canonical(request)), 'status': 'prepared',
                   'files': [{k: row[k] for k in ('name', 'beforeSha256', 'afterSha256', 'bytes')} for row in request['files']]}
        atomic_json(receipt_path.with_name(request['id']+'.request.json'), request)
        atomic_json(receipt_path, receipt)
    else:
        if receipt_path is None:
            raise ValueError('This command requires --receipt from the original publication')
        receipt_path = Path(receipt_path).resolve()
        receipt, request = load_request(receipt_path)
    # The local prepared receipt exists before SSH. Retry the same receipt after
    # an unknown network outcome; never start a replacement operation blindly.
    print(json.dumps({'receipt': str(receipt_path), 'operation': action, 'id': request['id']}), flush=True)
    try:
        result = json.loads(ssh_command(['python3', '-c', REMOTE, action], input=canonical(request), capture=True, timeout=180))
        if result.get('schema') != SCHEMA or result.get('id') != request['id'] or result.get('requestSha256') != receipt['requestSha256']:
            raise ValueError('Remote receipt identity mismatch')
    except Exception:
        receipt['lastAttempt'] = action; receipt['lastOutcome'] = 'unknown_or_failed'
        atomic_json(receipt_path, receipt)
        raise
    receipt.update(status=result['status'], lastAttempt=action, lastOutcome='verified', remote=result)
    atomic_json(receipt_path, receipt)
    print(json.dumps({'receipt': str(receipt_path), 'status': result['status'], 'backup': result['backup']}))
    return receipt
SNAPSHOT = r'''
from pathlib import Path
import json,hashlib
base=Path('/root/gleaner')
names=('index.html','articles/mushroom-vision.html','sitemap.xml','EDITORIAL-GUIDELINES.md')
result={}
for name in names:
 path=base/name
 if path.is_symlink() or path.resolve()!=path or not path.is_file() or path.stat().st_size>4*1024*1024:raise ValueError('Unexpected or oversized source file')
 raw=path.read_bytes()
 if len(raw)>4*1024*1024:raise ValueError('Public source exceeded bound')
 result[name]={'sha256':hashlib.sha256(raw).hexdigest(),'text':raw.decode('utf8')}
print(json.dumps(result))
'''


def snapshot():
    destination=WORK/'original.json'
    if destination.exists():
        raise ValueError('Snapshot already exists; preserve it and its receipts. Use the saved operation.')
    data=json.loads(ssh_command(['python3','-c',SNAPSHOT],capture=True))
    atomic_json(destination,data)
    print(json.dumps({'saved':str(destination),'files':len(data),'editorialGuidelinesCaptured':True}))


def stage(snapshot_path=WORK/'original.json', staged=WORK/'staged'):
    snapshot_data=json.loads(read_bounded(Path(snapshot_path),16*1024*1024))
    originals={}
    for name in NAMES:
        item=snapshot_data.get(name,{})
        text=item.get('text')
        if not isinstance(text,str) or digest(text.encode('utf8'))!=item.get('sha256'):
            raise ValueError('Invalid snapshot bytes: '+name)
        originals[name]=text
    def once(text,old,new):
        if text.count(old)!=1:raise ValueError('Expected one unchanged insertion point')
        return text.replace(old,new,1)
    guide='https://thegreatlakesgleaner.press/mushroom-lab/ai-contribute.html'
    if any('ai-contribute.html' in originals[name] for name in NAMES):
        raise ValueError('Contribution link already exists; inspect current site rather than duplicate it')
    home_anchor='<a class="btn btn-light" href="articles/mushroom-vision.html">Read about Mushroom Vision</a>'
    homepage=once(originals['index.html'],home_anchor,home_anchor+'<a class="btn btn-light" href="mushroom-lab/ai-contribute.html">Contribute with your AI</a>')
    article_anchor='<section class="article-toc" aria-labelledby="sources-title">'
    invitation='''<section class="mv-invitation" aria-labelledby="ai-contribution-title">
      <div class="eyebrow">Open contributions / People and their AI tools</div>
      <h2 id="ai-contribution-title">Bring your AI to the mushroom lab.</h2>
      <p>Mushroom Vision now has a dedicated contribution API for people who want their image models or AI assistants to help review mushroom photographs and woodland frames. The source code is public, and the project is still unfinished. Tools can retrieve an assigned image and submit mushroom regions, confusing objects and uncertain observations. These stay explicitly labeled as unverified AI evidence for human review; a successful submission does not approve a training label or activate a model.</p>
      <a class="btn btn-primary" href="../mushroom-lab/ai-contribute.html">Read the AI contribution guide <span aria-hidden="true">↗</span></a>
    </section>
    '''
    article=once(originals['articles/mushroom-vision.html'],article_anchor,invitation+article_anchor)
    sitemap=originals['sitemap.xml']
    if '<!DOCTYPE' in sitemap.upper() or '<!ENTITY' in sitemap.upper():raise ValueError('Sitemap declarations forbidden')
    xml=ET.fromstring(sitemap)
    if xml.tag!='{http://www.sitemaps.org/schemas/sitemap/0.9}urlset':raise ValueError('Invalid sitemap')
    sitemap=once(sitemap,'</urlset>',f'  <url><loc>{guide}</loc><lastmod>2026-10-31</lastmod></url>\n</urlset>')
    ET.fromstring(sitemap)
    staged=Path(staged)
    for name,text in zip(NAMES,(homepage,article,sitemap)):
        destination=staged/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.is_symlink() or destination.resolve()!=destination.absolute():raise ValueError('Linked staging target')
        destination.write_text(text,encoding='utf8',newline='')
    print(json.dumps({'staged':str(staged),'files':len(NAMES),'guide':guide,'preservedExistingProse':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['snapshot','stage','validate','publish','rollback','status'])
    parser.add_argument('--receipt',type=Path)
    args=parser.parse_args()
    if args.action=='snapshot':snapshot()
    elif args.action=='stage':stage()
    elif args.action=='validate':
        request=prepare()
        print(json.dumps({'valid':True,'files':[{k:row[k] for k in ('name','beforeSha256','afterSha256','bytes')} for row in request['files']]}))
    else:operate(args.action,args.receipt)
