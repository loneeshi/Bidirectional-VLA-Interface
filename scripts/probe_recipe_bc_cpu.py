"""One bounded CPU-only official BC architecture learnability probe; never a VLA attempt."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import time
import hashlib

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['audit','source','official-agent','output']:
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    import numpy as np
    import h5py
    import torch
    started=time.monotonic(); deadline=started+570
    torch.set_num_threads(2); torch.manual_seed(20260919)
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.audit.read_text()); spec=importlib.util.spec_from_file_location('official_bc_agent',a.official_agent)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    data={}
    with h5py.File(a.source,'r') as handle:
        for split in ['train','validation']:
            rows=[]
            for e in manifest['episodes']:
                if e.get('split')!=split: continue
                # Fixed start, middle and last pre-success frames, no model-based selection.
                for t in sorted({0,e['exported_steps']//2,e['exported_steps']-1}): rows.append((e['parent_id'],t))
            if split=='validation': rows=rows[::max(1,len(rows)//96)][:96]
            pixels={c:[] for c in ['fetch_head_depth','fetch_hand_depth']}; states=[]; actions=[]
            for ident,t in rows:
                if time.monotonic()>deadline-120: raise TimeoutError('CPU data load consumed probe budget')
                g=handle[f'traj_{ident}']; extra=g['obs/extra']
                state=np.concatenate([g['obs/agent/qpos'][t],g['obs/agent/qvel'][t],extra['tcp_pose_wrt_base'][t],extra['obj_pose_wrt_base'][t],extra['goal_pos_wrt_base'][t],np.asarray([extra['is_grasped'][t]],np.float32)])
                states.append(state);actions.append(g['actions'][t])
                for c in pixels: pixels[c].append(g['obs/sensor_data/'+c.removesuffix('_depth')+'/depth'][t].transpose(2,0,1))
            data[split]={'state':torch.tensor(np.asarray(states),dtype=torch.float32),
                         'pixels':{c:torch.tensor(np.asarray(v),dtype=torch.float32) for c,v in pixels.items()},
                         'actions':torch.tensor(np.asarray(actions),dtype=torch.float32),'rows':rows}
    def batch(split,ids):
        d=data[split]
        return {'state':d['state'][ids], 'pixels':{k:v[ids] for k,v in d['pixels'].items()}}
    model=module.Agent(batch('train',[0]),(13,))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    def metrics(split):
        ds=data[split]; preds=[]
        with torch.no_grad():
            for i in range(0,len(ds['rows']),16): preds.append(model(batch(split,slice(i,i+16))))
        pred=torch.cat(preds); target=ds['actions']; rmse=((pred-target).square().mean(0)).sqrt()
        zero=(target.square().mean(0)).sqrt()
        return {'rmse':rmse.tolist(),'zero_rmse':zero.tolist(),'rmse_ratio_to_zero':(rmse/zero.clamp_min(1e-8)).tolist(),'n':len(target)}
    initial=metrics('validation'); history=[]; updates=0
    for update in range(500):
        if time.monotonic()>deadline-35: break
        ids=torch.randint(0,len(data['train']['rows']),(16,))
        pred=model(batch('train',ids));loss=(pred-data['train']['actions'][ids]).square().mean()
        if not torch.isfinite(loss): raise ValueError('Nonfinite CPU BC loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
        updates=update+1
        if updates%25==0:history.append({'update':updates,'loss':float(loss.detach())})
    final=metrics('validation')
    report={'status':'completed_bounded_cpu_probe','updates':updates,'max_updates':500,'outer_seconds':600,'wall_seconds':time.monotonic()-started,
            'batch_size':16,'lr':3e-4,'optimizer':'AdamW','seed':20260919,'initialization':'random; no pretrained BC weights',
            'source_agent':str(a.official_agent),'source_sha256':hashlib.sha256(a.official_agent.read_bytes()).hexdigest(),
            'train_rows':len(data['train']['rows']),'validation_rows':len(data['validation']['rows']),
            'observation':'Official two depth128 cameras, native qpos12/qvel12 + current privileged18; official model transform unchanged',
            'before':initial,'after':final,'history':history,'gpu_operations':0,'vla_updates':0,'api_calls':0,
            'limitations':['Bounded CPU probe is not a full BC reproduction or native task result.',
                           'A negative result cannot establish unlearnability, missing information or uselessness of additional data.',
                           'BC depth/state42 differs from the proposed pi05 input; differences cannot be solely attributed to architecture.',
                           'Marginal target multimodality is not conditional ambiguity.']}
    (a.output/'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'status':report['status'],'updates':updates,'seconds':report['wall_seconds'],
                      'torso_yaw_ratios':[final['rmse_ratio_to_zero'][10],final['rmse_ratio_to_zero'][12]]}))


if __name__=='__main__': main()
