import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import torch
root=Path('/home/pshuai/bvi-research/assets/data/scene_datasets/replica_cad_dataset/rearrange')
x=torch.load(root/'spawn_data/tidy_house/pick/val/spawn_data.pt',map_location='cpu',weights_only=True)
def describe(x,depth=0):
    if hasattr(x,'shape'):return {'shape':list(x.shape),'dtype':str(x.dtype),'sample':x.reshape(-1)[:10].tolist()}
    if isinstance(x,dict):return {'count':len(x),'keys':list(x)[:15],'first':{str(k):describe(v,depth+1) for k,v in list(x.items())[:(1 if depth==0 else 10)]}}
    if isinstance(x,(list,tuple)):return {'length':len(x),'first':[describe(v,depth+1) for v in x[:2]]}
    return str(x)[:200]
print(json.dumps(describe(x)))
e=root/'val/tidy_house/episode_10.json'
y=json.loads(e.read_text());print(json.dumps(y)[:9000])
print('CUDA_INITIALIZED',torch.cuda.is_initialized())
