"""Resume only a marked, unpublished release; verify every byte before readiness."""
from pathlib import Path, PurePosixPath
import argparse, contextlib, hashlib, io, json, tarfile, uuid
from deploy import ssh_command, release_name

PREFIX = r'''
import pathlib,sys,json,hashlib,tarfile,os
base=pathlib.Path('/root/gleaner-mushroom-lab')
requested=base/'releases'/sys.argv[1]
if requested.is_symlink() or (base/'releases').is_symlink():raise ValueError('Linked release directory')
release=requested.resolve(strict=True)
if release.parent!=base/'releases':raise ValueError('Unmanaged release')
current=json.loads((base/'current.json').read_text())
if pathlib.Path(current['release']).resolve(strict=True)==release:raise ValueError('Cannot modify current release')
marker=release/'UPLOAD-INCOMPLETE'
if marker.is_symlink() or not marker.is_file():raise ValueError('Only incomplete staged releases may resume')
if (release/'release-manifest.json').is_symlink():raise ValueError('Linked release manifest')
def safe(name):
 p=pathlib.PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or not p.parts or str(p)!=name or name in {'release-manifest.json','UPLOAD-INCOMPLETE'}:raise ValueError('Unsafe member')
 target=release.joinpath(*p.parts)
 if any(release.joinpath(*p.parts[:i]).is_symlink() for i in range(1,len(p.parts)+1)) or not target.resolve().is_relative_to(release):raise ValueError('Escaping member')
 return target
'''
SCAN = PREFIX + r'''
m=json.loads(sys.stdin.buffer.read(5*1024*1024+1))
if m.get('schema')!='mycoscout.community-release.v1' or len(m['files'])>11000:raise ValueError('Invalid manifest')
if sum(x['bytes'] for x in m['files'].values())>4*1024**3:raise ValueError('Oversized release')
missing=[]
for name,want in m['files'].items():
 p=safe(name)
 if not p.is_file() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:missing.append(name)
(release/'release-manifest.json').write_text(json.dumps(m))
print(json.dumps({'missing':missing,'verified':len(m['files'])-len(missing)}))
'''
COPY = PREFIX + r'''
m=json.loads((release/'release-manifest.json').read_text())
count=total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as archive:
 for member in archive:
  count+=1;total+=member.size
  if not member.isfile() or member.name not in m['files'] or total>130*1024**2 or count>11000:raise ValueError('Unexpected resume member')
  want=m['files'][member.name]
  if member.size!=want['bytes']:raise ValueError('Size mismatch')
  data=archive.extractfile(member).read()
  if hashlib.sha256(data).hexdigest()!=want['sha256']:raise ValueError('Hash mismatch')
  target=safe(member.name);target.parent.mkdir(parents=True,exist_ok=True)
  temporary=target.with_name(target.name+'.resume-new')
  if temporary.is_symlink():raise ValueError('Unsafe temporary')
  with temporary.open('wb') as output:output.write(data);output.flush();os.fsync(output.fileno())
  temporary.replace(target);os.chmod(target,0o644)
print(json.dumps({'copied':count,'bytes':total,'published':False}))
'''
VERIFY = PREFIX + r'''
m=json.loads((release/'release-manifest.json').read_text())
for name,want in m['files'].items():
 p=safe(name)
 if not p.is_file() or p.stat().st_size!=want['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=want['sha256']:raise ValueError('Final integrity mismatch')
expected=set(m['files'])|{'release-manifest.json','UPLOAD-INCOMPLETE'}
actual={p.relative_to(release).as_posix() for p in release.rglob('*') if p.is_file()}
if actual!=expected:raise ValueError('Unexpected staged files')
marker.unlink()
print(json.dumps({'verifiedFiles':len(m['files']),'frames':m['frames'],'ready':True,'published':False}))
'''

@contextlib.contextmanager
def chunk_workspace(parent):
    # Avoid tempfile's restrictive Windows ACL on this bundled interpreter.
    directory=Path(parent).resolve()/('community-resume-'+uuid.uuid4().hex)
    directory.mkdir()
    try:yield directory
    finally:
        chunk=directory/'chunk.tar'
        if chunk.is_file():chunk.unlink()
        directory.rmdir()

def resume(archive_path, release):
    release=release_name(release)
    with tarfile.open(archive_path,'r:gz') as source:
        member=source.getmember('release-manifest.json')
        if not member.isfile() or member.size>5*1024**2:raise ValueError('Invalid release manifest')
        raw=source.extractfile(member).read()
    manifest=json.loads(raw)
    if manifest.get('schema')!='mycoscout.community-release.v1':raise ValueError('Invalid schema')
    for name in manifest['files']:
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or not path.parts or str(path)!=name or name in {'release-manifest.json','UPLOAD-INCOMPLETE'}:raise ValueError('Unsafe member')
    state=json.loads(ssh_command(['python3','-c',SCAN,release],input=raw,capture=True))
    missing=set(state['missing'])
    print(json.dumps({'alreadyVerified':state['verified'],'missingFiles':len(missing)}),flush=True)
    with chunk_workspace(Path(archive_path).parent) as directory:
        chunk=Path(directory)/'chunk.tar'
        out=tarfile.open(chunk,'w'); size=0; count=0
        def send():
            with chunk.open('rb') as stream:
                result=ssh_command(['python3','-c',COPY,release],stdin=stream,capture=True,timeout=600)
            print(result.decode().strip(),flush=True)
        with tarfile.open(archive_path,'r|gz') as source:
            for member in source:
                if member.name not in missing:continue
                want=manifest['files'][member.name]
                if not member.isfile() or member.size!=want['bytes'] or member.size>128*1024**2:raise ValueError('Unexpected file')
                data=source.extractfile(member).read()
                if hashlib.sha256(data).hexdigest()!=want['sha256']:raise ValueError('Archive hash mismatch')
                if size and size+len(data)>128*1024**2:
                    out.close();send();out=tarfile.open(chunk,'w');size=0
                out.addfile(member,io.BytesIO(data));size+=len(data);count+=1
        out.close()
        if size:send()
        if count!=len(missing):raise ValueError('Missing archive member')
    print(ssh_command(['python3','-c',VERIFY,release],capture=True).decode().strip(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--release',required=True)
    a=p.parse_args()
    try:resume(a.archive,a.release)
    except Exception as e:
        print(json.dumps({'error':type(e).__name__,'message':'Resume incomplete; staged marker retained and live release unchanged.'}),flush=True)
        raise SystemExit(1)
