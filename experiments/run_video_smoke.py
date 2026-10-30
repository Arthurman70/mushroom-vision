"""Run bounded original-RGB tests on installed real video frames; never grade them."""
import argparse,json,time
from pathlib import Path
from adapters.review_stream import Station

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--station',default='http://127.0.0.1:8787')
    p.add_argument('--manifest',type=Path,default=Path('data/video-discovery-v1/manifest.json'))
    p.add_argument('--output',type=Path,default=Path('experiments/results/video-smoke-v1.json'))
    p.add_argument('--limit',type=int,default=12)
    a=p.parse_args()
    if not 1<=a.limit<=24: p.error('Limit must be 1–24')
    if a.output.exists(): p.error('Choose a new report path')
    frames=json.loads(a.manifest.read_text(encoding='utf-8'))['frames'][:a.limit]
    station=Station(a.station);reports=[]
    for frame in frames:
        started=station.request('/api/testbench/videos/frames/'+frame['id']+'/run',{'settings':{'methods':['original'],'tileGrid':2,'scale':1,'boxThreshold':.25}})['run']
        stop=time.monotonic()+260
        while True:
            run=station.request('/api/testbench/runs/'+started['id'])['run']
            if run['status'] in ['completed','failed']:break
            if time.monotonic()>stop:raise TimeoutError('Video smoke deadline exceeded')
            time.sleep(2)
        row={'frameId':frame['id'],'timestampSeconds':frame['timestampSeconds'],'runId':run['id'],'status':run['status'],'videoProvenance':run.get('videoProvenance'),'candidates':len(run['result']['methods'][0]['candidates']) if run['result'] else None,'humanGrade':None,'detectionAccuracy':None}
        reports.append(row);print(json.dumps(row),flush=True)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps({'schema':'mycoscout.video-smoke.v1','frames':reports,'notice':'Actual inference only; no human labels or accuracy claims.','automaticTraining':False},indent=2),encoding='utf-8')

if __name__=='__main__':main()
