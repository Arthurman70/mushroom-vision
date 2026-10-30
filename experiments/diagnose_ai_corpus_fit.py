"""Train-input-only numerical diagnostic; never reads the test benchmark."""
import random
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from experiments.train_ai_corpus_candidate import read, tensor_example, balanced_order, negative_regions, ensure_background_proposals
from vision.training_pipeline import build_model

random.seed(8251); torch.manual_seed(8251); torch.set_num_threads(4)
rows=read(ROOT/'experiments/results/ai-corpus-candidate-v1/accepted-records.json')
examples=[]
for row in rows:
    if row['split']!='train': continue
    examples.append({'id':row['id'],'row':row,'positive':bool(row['boxes'])})
    for region in negative_regions(row)[0][:2]:
        examples.append({'id':region['id'],'row':row,'positive':False,'crop':region['bounds']})
model=build_model(pretrained=True)
def background_hook(module, inputs, result):
    if not module.training: return result
    proposals, count=ensure_background_proposals(result[0],inputs[0].image_sizes,inputs[2])
    if count: print('Confirmed-negative background proposal fallback:',count,flush=True)
    return proposals,result[1]
model.rpn.register_forward_hook(background_hook)
optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.001,momentum=.9,weight_decay=.0005)
model.train();model.backbone.body.eval()
for index,example in enumerate(balanced_order(examples,8252)):
    image,target=tensor_example(ROOT,example,augment=True)
    losses=model([image],[target]);loss=sum(losses.values())
    print(index,example['id'],example['row']['imagePath'],example.get('crop'),image.shape,target,{k:float(v.detach()) for k,v in losses.items()},flush=True)
    if not torch.isfinite(loss):break
    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],10);optimizer.step()
    if index>=12:break
