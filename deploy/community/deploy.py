"""Explicit SSH operations for the standalone public contributor service."""
from pathlib import Path
import argparse
import base64
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import uuid

# Operator credentials/hosts belong in an ignored local file or environment,
# never in the public source distribution.
def deployment_target():
    target = os.environ.get("MYCOSCOUT_DEPLOY_TARGET", "")
    if not target:
        local = Path(__file__).resolve().parents[2] / "data" / "deployment-target.json"
        if local.exists():
            target = json.loads(local.read_text(encoding="utf-8"))["sshTarget"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+", target):
        raise ValueError("Set MYCOSCOUT_DEPLOY_TARGET to your authorized SSH user@host")
    return target
BASE = "/root/gleaner-mushroom-lab"
SERVICE = "gleaner-mushroom-lab"


def ssh_command(args, *, stdin=None, input=None, capture=False, timeout=600):
    ssh = shutil.which("ssh") or "C:/Windows/System32/OpenSSH/ssh.exe"
    return subprocess.run([ssh, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", deployment_target(), shlex.join(args)],
                          stdin=stdin, input=input, stdout=subprocess.PIPE if capture else None,
                          check=True, timeout=timeout).stdout


def release_name(value):
    if not re.fullmatch(r"[a-z0-9-]{6,80}", value or ""):
        raise ValueError("Use a lower-case release ID containing letters, digits and hyphens")
    return value


UPLOAD = r'''
import sys,tarfile,json,pathlib,hashlib,re,os
name=sys.argv[1]
if not re.fullmatch(r'[a-z0-9-]{6,80}',name): raise ValueError('Invalid release ID')
base=pathlib.Path('/root/gleaner-mushroom-lab/releases')
base.mkdir(parents=True,exist_ok=True)
release=base/name
release.mkdir(exist_ok=False)
marker=release/'UPLOAD-INCOMPLETE'
marker.write_text('Upload must pass integrity verification before publication')
count=total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as archive:
 for member in archive:
  count+=1;total+=member.size
  path=pathlib.PurePosixPath(member.name)
  if not member.isfile() or path.is_absolute() or '..' in path.parts or count>5000 or total>512*1024*1024:
   raise ValueError('Unsafe or oversized release archive')
  target=release.joinpath(*path.parts)
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open('xb') as output:
   stream=archive.extractfile(member)
   while True:
    chunk=stream.read(1024*1024)
    if not chunk:break
    output.write(chunk)
  os.chmod(target,0o644)
m=json.loads((release/'release-manifest.json').read_text())
if m.get('schema')!='mycoscout.community-release.v1':raise ValueError('Invalid release manifest')
for name,expected in m['files'].items():
 p=(release/name).resolve(strict=True)
 if not p.is_relative_to(release) or p.is_symlink():raise ValueError('Invalid manifest path')
 data=p.read_bytes()
 if len(data)!=expected['bytes'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:raise ValueError('Integrity mismatch')
if count!=len(m['files'])+1:raise ValueError('Unexpected archive files')
marker.unlink()
print(json.dumps({'uploaded':str(release),'files':count,'frames':m['frames'],'published':False}))
'''

UPDATE_AI = r'''
import pathlib,json,sys,hashlib,os,datetime
base=pathlib.Path('/root/gleaner-mushroom-lab')
state=json.loads((base/'current.json').read_text())
release=(base/'releases'/sys.argv[1]).resolve(strict=True) if len(sys.argv)>1 else pathlib.Path(state['release']).resolve(strict=True)
if release.parent!=base/'releases':raise ValueError('Unmanaged release')
if len(sys.argv)>1 and str(release)==state['release']:raise ValueError('Use the ordinary update-ai command for the current release')
raw=sys.stdin.buffer.read(20*1024*1024+1)
if len(raw)>20*1024*1024:raise ValueError('Oversized AI file')
ai=json.loads(raw)
if ai.get('schema')!='mycoscout.public-ai-suggestions.v1':raise ValueError('Unexpected AI schema')
collection=json.loads((release/'collection/manifest.json').read_text())
allowed={f['id']:f for f in collection['frames']}
ids=[r['frameId'] for r in ai.get('frames',[])]
if len(set(ids))!=len(ids) or not set(ids)<=set(allowed):raise ValueError('AI frames differ from published images')
for row in ai['frames']:
 frame=allowed[row['frameId']]
 if row.get('imageSha256')!=frame['sha256'] or row.get('sourceId')!=frame['sourceId']:raise ValueError('AI suggestion identity does not match published image bytes')
 for box in row.get('boxes',[]):
  if box.get('humanVerified') is not False or box.get('trainingAllowed') is not False:raise ValueError('AI suggestion claims authority')
target=release/'collection/ai-suggestions.json'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup=base/'ai-history'
backup.mkdir(exist_ok=True)
if target.exists():
 old=target.read_bytes()
 copy=backup/(stamp+'-'+hashlib.sha256(old).hexdigest()[:12]+'.json')
 if not copy.exists():copy.write_bytes(old)
temporary=target.with_suffix('.json.new')
with temporary.open('xb') as output:output.write(raw);output.flush();os.fsync(output.fileno())
temporary.replace(target)
integrity_path=release/'release-manifest.json'
integrity=json.loads(integrity_path.read_text())
integrity['files']['collection/ai-suggestions.json']={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
integrity_path.write_text(json.dumps(integrity,indent=2))
print(json.dumps({'updated':True,'frames':len(ids),'run':ai.get('run'),'imagesCopied':0,'serviceRestarted':False}))
'''

CLONE_CURRENT = r'''
import pathlib,json,sys,re,os,shutil
base=pathlib.Path('/root/gleaner-mushroom-lab')
name=sys.argv[1]
if not re.fullmatch(r'[a-z0-9-]{6,80}',name):raise ValueError('Invalid release ID')
state=json.loads((base/'current.json').read_text())
current=pathlib.Path(state['release']).resolve(strict=True)
if current.parent!=base/'releases':raise ValueError('Unmanaged current release')
destination=base/'releases'/name
destination.mkdir(exist_ok=False)
marker=destination/'UPLOAD-INCOMPLETE'
marker.write_text('Staged clone has not completed')
manifest=json.loads((current/'release-manifest.json').read_text())
linked=0
for name in [*manifest['files'],'release-manifest.json']:
 relative=pathlib.PurePosixPath(name)
 if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid release member')
 source=current/name
 if source.is_symlink() or not source.is_file():raise ValueError('Unexpected release file')
 target=destination/name
 target.parent.mkdir(parents=True,exist_ok=True)
 if re.fullmatch(r'collection/images/[a-f0-9]{64}\.(jpg|png)',name):
  os.link(source,target);linked+=1
 else:shutil.copy2(source,target)
marker.unlink()
print(json.dumps({'staged':str(destination),'fromRelease':str(current),'immutableImagesLinked':linked,'imagesUploaded':0,'published':False}))
'''

REFRESH_STAGED_APP = r'''
import pathlib,json,sys,hashlib,base64,re
base=pathlib.Path('/root/gleaner-mushroom-lab')
name=sys.argv[1]
if not re.fullmatch(r'[a-z0-9-]{6,80}',name):raise ValueError('Invalid release ID')
release=base/'releases'/name
state=json.loads((base/'current.json').read_text()) if (base/'current.json').exists() else {}
if state.get('release')==str(release):raise ValueError('Cannot refresh the published release')
if (release/'UPLOAD-INCOMPLETE').exists():raise ValueError('Upload is incomplete')
payload=json.loads(sys.stdin.buffer.read(5*1024*1024+1))
integrity_path=release/'release-manifest.json'
integrity=json.loads(integrity_path.read_text())
for name,encoded in payload.items():
 allowed=name in {'community/server.mjs','community/store.mjs','community/export-reviews.mjs','Dockerfile','compose.yaml','caddy-route.txt','deploy-server.py'} or re.fullmatch(r'community/public/[A-Za-z0-9_./-]+\.(html|js|css|svg|png|jpg|ico|webp)',name)
 if not allowed or '..' in pathlib.PurePosixPath(name).parts:raise ValueError('Unexpected staged app file')
 data=base64.b64decode(encoded,validate=True)
 target=release/name
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(data)
 integrity['files'][name]={'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
integrity_path.write_text(json.dumps(integrity,indent=2))
print(json.dumps({'refreshedStagedFiles':len(payload),'imagesCopied':0,'published':False}))
'''


def refresh_staged_app(release):
    here = Path(__file__).resolve().parent
    root = here.parents[1]
    paths = [(root / 'community' / name, 'community/' + name) for name in ('server.mjs', 'store.mjs', 'export-reviews.mjs')]
    paths += [(path, 'community/public/' + path.relative_to(root / 'community/public').as_posix()) for path in (root / 'community/public').rglob('*') if path.is_file()]
    paths += [(here / name, name) for name in ('Dockerfile', 'compose.yaml', 'caddy-route.txt', 'deploy-server.py')]
    if any(path.is_symlink() for path, _ in paths):
        raise ValueError('Symlink cannot enter public app release')
    payload = json.dumps({name: base64.b64encode(path.read_bytes()).decode() for path, name in paths}).encode()
    if len(payload) > 5 * 1024 * 1024:
        raise ValueError('App refresh exceeds 5 MiB')
    ssh_command(['python3', '-c', REFRESH_STAGED_APP, release_name(release)], input=payload)


def export_reviews(output):
    remote = "/tmp/community-reviews-" + uuid.uuid4().hex + ".json"
    try:
        ssh_command(["docker", "exec", SERVICE, "node", "/app/export-reviews.mjs", "/data", remote], capture=True)
        raw = ssh_command(["docker", "exec", SERVICE, "cat", remote], capture=True)
        payload = json.loads(raw)
        if payload.get("schema") != "mycoscout.community-reviews.v1" or payload.get("trainingApproved") is not False:
            raise ValueError("Unexpected community export")
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as stream:
            stream.write(raw)
        print(json.dumps({"file": str(output), "reviews": len(payload["reviews"]), "sha256": hashlib.sha256(raw).hexdigest(), "trainingApproved": False}))
    finally:
        ssh_command(["docker", "exec", SERVICE, "node", "-e", "require('node:fs').rmSync(" + json.dumps(remote) + ",{force:true})"], capture=True)


STATUS_VALIDATOR = r'''
import json, math, datetime
def clean_training_status(value):
 states={'disabled','paused','waiting_data','waiting_daily_limit','waiting_resource','waiting_sync','exporting','training','comparing','retry_wait','candidate_ready','failed'}
 messages={'disabled':'Automatic training is off.','paused':'Automatic training is paused.','waiting_data':'Waiting for more approved examples from independent recording sessions.','waiting_daily_limit':'The next training run is waiting for its daily limit.','waiting_resource':'Waiting for the training computer to become available.','waiting_sync':'Waiting for a fresh community permissions check.','exporting':'Preparing a frozen reviewed dataset.','training':'Training a candidate model.','comparing':'Comparing the candidate with the current detector.','retry_wait':'Waiting before a bounded retry.','candidate_ready':'Candidate comparison is ready for curator review.','failed':'The worker needs attention; the current model remains unchanged.'}
 def stamp(x):
  if not isinstance(x,str) or len(x)>64:raise ValueError('Invalid status timestamp')
  parsed=datetime.datetime.fromisoformat(x.replace('Z','+00:00'))
  if parsed.tzinfo is None:raise ValueError('Timestamp requires timezone')
  if parsed>datetime.datetime.now(datetime.timezone.utc)+datetime.timedelta(minutes=5):raise ValueError('Future status timestamp')
  return parsed.astimezone(datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')
 def count(x,nullable=False):
  if x is None and nullable:return None
  if type(x) is not int or not 0<=x<=10000000:raise ValueError('Invalid aggregate count')
  return x
 if not isinstance(value,dict) or value.get('schema')!='mycoscout.community-learning-status.v1' or value.get('automaticActivation') is not False:raise ValueError('Invalid training status schema')
 w=value.get('worker',{})
 if not isinstance(w,dict) or not isinstance(w.get('state'),str) or w.get('state') not in states or type(w.get('enabled')) is not bool or type(w.get('paused')) is not bool:raise ValueError('Invalid worker status')
 thresholds=value.get('thresholds',{})
 if not isinstance(thresholds,dict):raise ValueError('Invalid training thresholds')
 result={'schema':value['schema'],'updatedAt':stamp(value.get('updatedAt')),'worker':{k:w[k] for k in ('state','enabled','paused')},'message':messages[w['state']],
  'approvedFrames':count(value.get('approvedFrames'),True),'newFrames':count(value.get('newFrames'),True),'newSessions':count(value.get('newSessions'),True),
  'thresholds':{k:count(thresholds.get(k)) for k in ('newFrames','newSessions','maxRunsPerDay')},'latestCandidate':None,'automaticActivation':False}
 candidate=value.get('latestCandidate')
 if candidate is not None:
  if not isinstance(candidate,dict) or not isinstance(candidate.get('status'),str) or candidate.get('status') not in {'candidate_better','candidate_not_better','insufficient_evidence','comparison_failed'}:raise ValueError('Invalid candidate status')
  metrics=candidate.get('comparison',{})
  if not isinstance(metrics,dict):raise ValueError('Invalid candidate comparison')
  comparison={}
  for key in ('baselineRecall','candidateRecall','baselineFalsePositivesPerFrame','candidateFalsePositivesPerFrame'):
   x=metrics.get(key)
   if x is not None and (type(x) not in (int,float) or not math.isfinite(x) or x<0 or x>(1 if key.endswith('Recall') else 100000)):raise ValueError('Invalid candidate metric')
   comparison[key]=x
  result['latestCandidate']={'status':candidate['status'],'completedAt':stamp(candidate.get('completedAt')),'comparison':comparison}
 return result
'''

UPDATE_TRAINING_STATUS = STATUS_VALIDATOR + r'''
import pathlib,sys,os,uuid
raw=sys.stdin.buffer.read(65537)
if len(raw)>65536:raise ValueError('Training summary exceeds64KiB')
data=clean_training_status(json.loads(raw))
base=pathlib.Path('/root/gleaner-mushroom-lab/data')
if base.resolve()!=base or not base.is_dir():raise ValueError('Managed public data directory missing')
temporary=base/('.training-status-'+uuid.uuid4().hex+'.json')
try:
 with temporary.open('x') as output:
  json.dump(data,output,indent=2);output.write('\n');output.flush();os.fsync(output.fileno())
 os.chown(temporary,1000,1000);os.chmod(temporary,0o644)
 temporary.replace(base/'training-status.json')
finally:
 if temporary.exists():temporary.unlink()
print(json.dumps({'updated':True,'workerState':data['worker']['state'],'automaticActivation':False}))
'''


def update_training_status(file):
    raw=Path(file).read_bytes()
    if len(raw)>65536:raise ValueError('Training summary exceeds64KiB')
    namespace={}
    exec(STATUS_VALIDATOR,namespace)
    clean=namespace['clean_training_status'](json.loads(raw))
    ssh_command(['python3','-c',UPDATE_TRAINING_STATUS],input=json.dumps(clean).encode())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    upload = sub.add_parser("upload")
    upload.add_argument("--bundle", required=True, type=Path)
    upload.add_argument("--release", required=True)
    publish = sub.add_parser("publish")
    publish.add_argument("--release", required=True)
    refresh = sub.add_parser("refresh-staged-app")
    refresh.add_argument("--release", required=True)
    clone = sub.add_parser("clone-current")
    clone.add_argument("--release", required=True)
    ai = sub.add_parser("update-ai")
    ai.add_argument("--file", required=True, type=Path)
    ai.add_argument("--release", help="Update an unpublished staged release instead of the current release")
    export = sub.add_parser("export-reviews")
    export.add_argument("--output", required=True, type=Path)
    training_status = sub.add_parser("update-training-status")
    training_status.add_argument("--file", required=True, type=Path)
    rollback = sub.add_parser("rollback")
    rollback.add_argument("--release", required=True)
    rollback.add_argument("--backup", required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.action == "upload":
        with args.bundle.open("rb") as stream:
            ssh_command(["python3", "-c", UPLOAD, release_name(args.release)], stdin=stream)
    elif args.action == "publish":
        ssh_command(["python3", BASE + "/releases/" + release_name(args.release) + "/deploy-server.py", "publish"])
    elif args.action == "refresh-staged-app":
        refresh_staged_app(args.release)
    elif args.action == "clone-current":
        ssh_command(["python3", "-c", CLONE_CURRENT, release_name(args.release)])
    elif args.action == "update-ai":
        command=["python3", "-c", UPDATE_AI]
        if args.release:
            command.append(release_name(args.release))
        ssh_command(command, input=args.file.read_bytes())
    elif args.action == "export-reviews":
        export_reviews(args.output)
    elif args.action == "update-training-status":
        update_training_status(args.file)
    elif args.action == "rollback":
        if not re.fullmatch(re.escape(BASE) + r"/backups/[A-Za-z0-9-]+", args.backup):
            raise ValueError("Invalid managed backup path")
        ssh_command(["python3", BASE + "/releases/" + release_name(args.release) + "/deploy-server.py", "rollback", "--backup", args.backup])
    else:
        print(ssh_command(["python3", "-c", "from pathlib import Path; p=Path('/root/gleaner-mushroom-lab/current.json'); print(p.read_text() if p.exists() else '{\"deployed\":false}')"], capture=True).decode())
