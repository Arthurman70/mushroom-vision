"""Interactive production smoke test. Credentials exist in memory only.

Inspect the downloaded original before entering one AI review JSON on stdin.
The script will never invent a visual review or mark one as human verified.
"""
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid

BASE='https://thegreatlakesgleaner.press/mushroom-lab/'
OUT=Path(__file__).resolve().parents[1]/'data/community-round4/webhook-smoke'
OUT.mkdir(exist_ok=False)

def request(path, payload=None, headers=None):
    url=urllib.parse.urljoin(BASE,path)
    if not url.startswith(BASE):raise ValueError('Unexpected API origin')
    req=urllib.request.Request(url,data=json.dumps(payload).encode() if payload is not None else None,
        headers={**({'Content-Type':'application/json'} if payload is not None else {}),**(headers or {})})
    with urllib.request.urlopen(req,timeout=30) as response:
        return response.status,response.headers,response.read()

def call(path,payload=None,headers=None):
    status,_,raw=request(path,payload,headers)
    return status,json.loads(raw)

checks=[]
for path in ('api/health','api/ai/info','ai-openapi.json','testing-round4.json'):
    status,value=call(path);checks.append({'path':path,'status':status})
    if path=='api/health':assert value['frames']==4660
    if path=='api/ai/info':info=value
_,registration=call('api/ai/agents',{'reviewerKind':'ai','model':{'provider':'OpenAI','name':'Codex assistant','version':'not exposed by runtime'},'participationTerms':info['participationTerms']})
auth={'Authorization':'Bearer '+registration['token']}
_,queue=call('api/ai/tasks?limit=1',headers=auth);task=queue['tasks'][0]
status,headers,raw=request(task['imageUrl'],headers=auth)
assert hashlib.sha256(raw).hexdigest()==task['imageSha256']==headers['X-Image-Sha256']
(OUT/'original.jpg').write_bytes(raw)
(OUT/'task.json').write_text(json.dumps(task,indent=2)+'\n',encoding='utf8')
print(json.dumps({'waitingForActualVisualInspection':True,'original':str(OUT/'original.jpg'),'task':task}),flush=True)
review=json.loads(input('Enter the actual AI review JSON after viewing the original: '))
payload={'taskId':task['taskId'],'imageSha256':task['imageSha256'],'reviewerKind':'ai',
    **review,'participationTerms':info['participationTerms']}
payload['observation']={**payload['observation'],'imageSha256':task['imageSha256']}
key='smoke-'+uuid.uuid4().hex
status,receipt=call('api/ai/contributions',payload,{**auth,'Idempotency-Key':key})
retry_status,retry=call('api/ai/contributions',payload,{**auth,'Idempotency-Key':key})
identifier=receipt['contribution']['id']
_,own=call('api/ai/contributions/'+identifier,headers=auth)
_,public=call('api/ai/reviews?frameId='+task['frameId'])
result={'schema':'mushroom-vision.webhook-smoke.v1','checks':checks,'firstStatus':status,'retryStatus':retry_status,
    'receipt':receipt,'retry':retry,'ownReadback':own,'publicReadback':public,'imageSha256':task['imageSha256'],
    'tokenSaved':False}
assert status==201 and retry_status==200 and retry['duplicate'] is True
assert receipt['contribution']['humanVerified'] is False and receipt['contribution']['trainingApproved'] is False
assert any(row['id']==identifier for row in public['reviews'])
(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps({'submitted':identifier,'frameId':task['frameId'],'idempotentRetryVerified':True,'publicReadbackVerified':True,'trainingApproved':False,'result':str(OUT/'result.json')}),flush=True)
