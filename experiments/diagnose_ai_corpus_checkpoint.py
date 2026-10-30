"""Training-set diagnostics for the selected AI candidate; never reads test data."""
from pathlib import Path
import json
import sys
from collections import Counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from experiments.train_ai_corpus_candidate import load_records, read, digest, evaluate, negative_regions, tensor_example, CHECKPOINT_SCHEMA
from vision.training_pipeline import build_model, predictions_for
from vision.video_ai_pass import atomic_json


def run(directory):
    directory=Path(directory)
    report=read(directory/'report.json')
    if digest(directory/'candidate.pt')!=report['checkpointSha256']:
        raise ValueError('Candidate changed')
    inputs=directory/'inputs'
    rows,_=load_records(inputs/'training-export.json',inputs/'inventory.json',inputs/'ledger.json')
    rows=[row for row in rows if row['split']=='train']
    torch.set_num_threads(4)
    checkpoint=torch.load(directory/'candidate.pt',map_location='cpu',weights_only=True)
    if checkpoint.get('schema')!=CHECKPOINT_SCHEMA or checkpoint.get('activationEligible') is not False:
        raise ValueError('Unsupported candidate')
    model=build_model(pretrained=False);model.load_state_dict(checkpoint['state_dict'])
    metrics,frames=evaluate(model,rows)
    crop_frames,crop_alerts=Counter(),Counter()
    for row in rows:
        for region in negative_regions(row)[0][:2]:
            image,_=tensor_example(ROOT,{'row':row,'crop':region['bounds']})
            predictions=predictions_for(model,image,.3)
            crop_frames[region['kind']]+=1
            crop_alerts[region['kind']]+=len(predictions)
    result={'schema':'mushroom-vision.ai-corpus-fit-diagnostic.v1','status':'completed',
        'labelAuthority':'ai-visual-review','humanVerified':False,'trainingSetDiagnosticOnly':True,
        'generalizationEvaluated':False,'scoreThreshold':.3,'iouThreshold':.5,
        'selectedEpoch':report['selectedEpoch'],'selectedCheckpointOptimizerSteps':report['selectedEpoch']*report['protocol']['trainingExamples'],
        'totalRunOptimizerSteps':report['optimizerSteps'],'metrics':metrics,
        'explicitNegativeCropCounts':dict(crop_frames),'explicitNegativeCropFalseAlerts':dict(crop_alerts),
        'records':frames,'activationEligible':False,'productionModelChanged':False}
    atomic_json(directory/'fit-only-diagnostic.json',result)
    print(json.dumps({key:value for key,value in result.items() if key!='records'}),flush=True)
    return result


if __name__=='__main__':
    run(ROOT/'experiments/results/ai-corpus-candidate-v2')
