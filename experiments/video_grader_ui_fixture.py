"""Create isolated UI fixtures; not video or recognition evaluation evidence."""
from pathlib import Path
import hashlib,json,shutil
from PIL import Image
root=Path(__file__).resolve().parents[1]
out=root/'experiments/results/video-grader-ui-v1'
(out/'collection/frames').mkdir(parents=True,exist_ok=True)
image=root/'data/black-trumpet-discovery-v2/images/inat-photo-745660170.jpg'
payload=image.read_bytes(); digest=hashlib.sha256(payload).hexdigest()
with Image.open(image) as im: w,h=im.size
sources=[]; frames=[]
for s in range(2):
    sid='ui-fixture-'+str(s)
    sources.append({'id':sid,'title':'UI test fixture — not collected video '+str(s),'sourceUrl':'https://www.inaturalist.org/observations/405910182','attribution':'CC0 photo reused for isolated interface tests. Not video evidence.','license':'CC0','licenseUrl':'https://creativecommons.org/publicdomain/zero/1.0/','videoHash':hashlib.sha256(sid.encode()).hexdigest(),'sourceGroupId':sid,'evaluationOnly':True,'licenseAllowsTraining':True})
    for t in [0,2,4]:
        fid=sid+'-'+str(t); path='frames/'+fid+'.jpg';shutil.copyfile(image,out/'collection'/path)
        frames.append({'id':fid,'sourceId':sid,'sha256':digest,'path':path,'width':w,'height':h,'timestampSeconds':t,'blurScore':100+t,'blurMetric':'UI fixture only','blurStratum':'middle','evaluationOnly':True})
(out/'collection/manifest.json').write_text(json.dumps({'schema':'mycoscout.video-collection.v1','sources':sources,'frames':frames}),encoding='utf-8')
(out/'station.mjs').write_text("import {createApp} from '../../../server.mjs';\nconst app=await createApp({stateFile:new URL('./state.json',import.meta.url).pathname.replace(/^\\/([A-Za-z]:)/,'$1'),videoCollectionDir:new URL('./collection',import.meta.url).pathname.replace(/^\\/([A-Za-z]:)/,'$1')});app.server.listen(8792,'127.0.0.1');\n",encoding='utf-8')
print(out)
