"""Add or roll back the one Gleaner navigation link, with its own backup."""
from pathlib import Path
import argparse
import json
from deploy import ssh_command

SCRIPT = r'''
from pathlib import Path
import hashlib,json,datetime,sys,subprocess,os
base=Path('/root/gleaner-mushroom-lab')
target=Path('/root/gleaner/js/components.js')
digest=lambda raw:hashlib.sha256(raw).hexdigest()
statefile=base/'site-link.json'
original=target.read_bytes()
if sys.argv[1]=='rollback':
 state=json.loads(statefile.read_text())
 if digest(original)!=state['afterSha256']:raise ValueError('Navigation changed since this link; preserve newer edits')
 raw=Path(state['backup']).read_bytes()
 if digest(raw)!=state['beforeSha256']:raise ValueError('Backup integrity mismatch')
 target.write_bytes(raw)
 statefile.unlink()
 print(json.dumps({'navigationRestored':True}))
else:
 before=original.decode()
 if 'key: "mushroom-lab"' in before:raise ValueError('Navigation link already exists')
 anchor='    { href: "about.html", label: "About", key: "about" },'
 if before.count(anchor)!=1:raise ValueError('Expected unique navigation anchor missing')
 addition='    { href: "mushroom-lab/", label: "Mushroom Lab", key: "mushroom-lab" },\n'
 updated=before.replace(anchor,addition+anchor).encode()
 subprocess.run(['docker','exec','-i','gleaner-mushroom-lab','node','--check','-'],input=updated,check=True,timeout=15)
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 backup=base/'backups'/('navigation-'+stamp)
 backup.mkdir(mode=0o700)
 path=backup/'components.js'
 path.write_bytes(original)
 if target.read_bytes()!=original:raise ValueError('Navigation changed concurrently')
 target.write_bytes(updated)
 state={'target':str(target),'backup':str(path),'beforeSha256':digest(original),'afterSha256':digest(updated),'publishedAt':stamp,'link':'/mushroom-lab/'}
 statefile.write_text(json.dumps(state,indent=2))
 print(json.dumps(state))
'''

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['add', 'rollback'])
    args = parser.parse_args()
    ssh_command(['python3', '-c', SCRIPT, args.action])
