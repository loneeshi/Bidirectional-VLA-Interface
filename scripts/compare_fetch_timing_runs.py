"""CPU comparison of paired timing controls, including exact pre-branch inputs."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--batch',required=True,type=Path)
args=parser.parse_args()
out=args.batch

def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

arms={}
for name in ['legacy-pre-action','post-action']:
    folder=out/name
    result=json.loads((folder/'result.json').read_text())
    events=[json.loads(x) for x in (folder/'events.jsonl').read_text().splitlines()]
    steps={e['step']:e for e in events if e['event']=='step'}
    switches=[]
    for e in events:
        if e['event']=='learned_progress' and e['trigger']:
            state=steps[e['step']]['extra']
            distance=float(np.linalg.norm(np.array(state['tcp_pose_wrt_base'][0][:3])-np.array(state['obj_pose_wrt_base'][0][:3])))
            switches.append(dict(step=e['step'],prediction_step=e['prediction_step'],family=e['family'],
                progress=e['values'][0],reason=e['trigger'],tcp_object_distance=distance,
                is_grasped=bool(np.asarray(state['is_grasped']).reshape(-1)[0]),
                executed_since_prediction=e['executed_since_prediction']))
    manifests=[json.loads(p.read_text()) for p in sorted((folder/'decisions').glob('*-manifest.json'))]
    arms[name]=dict(result=result,switches=switches,decision_count=len(manifests),manifests=manifests)

left=arms['legacy-pre-action']['manifests'];right=arms['post-action']['manifests']
pairs=[]
for lm in left:
    if lm['metadata']['family']!='reach':continue
    candidates=[m for m in right if m['metadata']['family']=='reach' and m['metadata']['prediction_step']==lm['metadata']['prediction_step']]
    if not candidates:continue
    rm=candidates[0]
    def load(arm,manifest,kind):
        return torch.load(out/arm/'decisions'/manifest['artifacts'][kind]['path'],map_location='cpu',weights_only=False)
    li,ri=load('legacy-pre-action',lm,'inputs'),load('post-action',rm,'inputs')
    lo,ro=load('legacy-pre-action',lm,'outputs'),load('post-action',rm,'outputs')
    lr,rr=load('legacy-pre-action',lm,'raw'),load('post-action',rm,'raw')
    pairs.append(dict(prediction_step=lm['metadata']['prediction_step'],
        model_batch_bitwise_equal=equal(li['batch'],ri['batch']),
        model_rng_equal=equal(li['rng_before_model'],ri['rng_before_model']),
        raw_values_equal=equal(lr['raw'],rr['raw']),
        actions_equal=equal(lo['actions'],ro['actions']),progress_equal=equal(lo['progress'],ro['progress'])))

report=dict(arms={k:{f:v for f,v in d.items() if f!='manifests'} for k,d in arms.items()},paired_reach_decisions=pairs,
    interpretation='One episode per mode; exact pre-branch comparison, not a success-rate estimate.')
original=out.parent/'fetch-tapt-online-pick-2026-09-16-run01/events.jsonl'
if original.exists():
    old=[json.loads(x) for x in original.read_text().splitlines()]
    new=[json.loads(x) for x in (out/'legacy-pre-action/events.jsonl').read_text().splitlines()]
    oldsteps=[e for e in old if e['event']=='step'];newsteps=[e for e in new if e['event']=='step']
    fields=['step','action','qpos','qvel','extra','info']
    report['legacy_reproduces_historical_steps']=len(oldsteps)==len(newsteps) and all(
        all(equal(a[k],b[k]) for k in fields) for a,b in zip(oldsteps,newsteps))
    oldp=[(e['step'],e['family'],e['values'],e['trigger']) for e in old if e['event']=='learned_progress']
    newp=[(e['step'],e['family'],e['values'],e['trigger']) for e in new if e['event']=='learned_progress']
    report['legacy_reproduces_historical_progress']=oldp==newp
(out/'comparison.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2))
print(json.dumps({k:{'steps':v['result']['steps'],'success':v['result']['success'],'switches':v['switches']} for k,v in arms.items()},indent=2))
