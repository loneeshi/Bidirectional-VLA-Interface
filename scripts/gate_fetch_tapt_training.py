"""Twenty real AC-DiT TAPT updates on fixed recorded examples, not full SFT.

Requires completed 20/5 trajectory collection and all four annotated families.
Small gate samples do not replace training over the complete split. No API.
"""
import os,sys,json,time,hashlib,subprocess,argparse,random
from pathlib import Path
root=Path.home()/'bvi-research'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--full',action='store_true',help='Bounded native-start SFT pilot over all labeled trajectories')
args=parser.parse_args()
if args.full:
    gate=json.loads((root/'runs/fetch-tapt-gate-2026-09-16/result.json').read_text())
    assert gate['status']=='20_updates_passed_not_full_sft'
source=root/'src/AC-DiT'
sys.path[:0]=[str(source),str(root/'src/Bidirectional-VLA-Interface/src')]
data=root/'runs/fetch-tapt-teacher-2026-09-16'
segments=json.loads((data/'segments.json').read_text())
assert segments['collection_status']=='collected_not_segmented_not_training_ready'
assert segments['all_families_present'], 'Missing family/split; cannot start training'
used=int(subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
assert used<1024, 'GPU1 occupied; do not overlap workloads'
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3',
                  HF_HOME=str(root/'hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2')
out=root/('runs/fetch-tapt-sft-2026-09-16-run01' if args.full else 'runs/fetch-tapt-gate-2026-09-16')
out.mkdir(exist_ok=False)
report={'status':'loading','updates':0,'api_calls':0,'scope':'20-update real-data integration gate, not full SFT',
        'source_scope':'SAC native starts; actual navigation handoff not yet covered',
        'rank':8,'alpha':8,'progress_weight':.1,'micro_batch':1,'effective_batch':1,
        'full_training_effective_batch':8,'checkpoint_selection':'none: gate only'}
def save():(out/'result.json').write_text(json.dumps(report,indent=2))
save()
try:
    import torch,numpy as np,h5py,yaml
    from PIL import Image
    from transformers import AutoTokenizer,SiglipTextModel
    from model_wrappers.mshab_model import create_model
    from bvi.acdit_contract import to_unified
    from bvi.acdit_tapt import ACDiTFamilyTool,FAMILIES
    torch.set_num_threads(2);torch.manual_seed(4017);np.random.seed(4017)
    os.chdir(source)
    encoder=json.loads((root/'encoder-lock.json').read_text())
    wrapper=create_model(args=yaml.safe_load((source/'configs/config.yaml').read_text()),
        pretrained=str(root/'checkpoints/acdit/stage2_all7_baseline/checkpoint-25000.pt'),
        device='cuda:0',dtype=torch.float32,method_name='AC-DiT',combine_flag=False,
        pretrained_vision_encoder_name_or_path=encoder['snapshot'],
        mobility_head_ckpt_path=str(root/'checkpoints/acdit/stage1_mobility_head/checkpoint-30000.pt'))
    tokenizer=AutoTokenizer.from_pretrained(encoder['snapshot'])
    text_model=SiglipTextModel.from_pretrained(encoder['snapshot'],torch_dtype=torch.float32).eval()
    selected=[next(w for w in segments['windows'] if w['split']==split and w['family']==family)
              for split in ['train','validation'] for family in FAMILIES]
    if args.full:
        selected=[{**w,'sample_t':t} for w in segments['windows']
                  for t in sorted({w['start'],(w['start']+w['end']-1)//2,w['end']-1})]
        report.update(scope='bounded native-start TAPT SFT pilot; not full paper IL or handoff adaptation',
                      effective_batch=8,max_updates=2000,max_training_seconds=6900,
                      checkpoint_selection='minimum held-out joint action+0.1progress loss',
                      sampling='uniform family per update, uniform labeled window/time per microbatch; start/middle/end support')
    report['gate_examples']=selected
    cache=[]
    for window in selected:
        tokens=tokenizer(window['instruction'],return_tensors='pt',padding=False)
        with torch.no_grad():embedding=text_model(input_ids=tokens['input_ids']).last_hidden_state[0]
        t=window.get('sample_t',window['start']);end=window['end']
        with h5py.File(window['trajectory']) as h:
            def obs(i):
                g=h[f'observations/{i:04d}']
                extra={k:torch.from_numpy(g['extra'][k][:]) for k in g['extra']}
                return g,extra
            g,extra=obs(t);qpos=g['agent/qpos'][0]
            native=np.r_[qpos[[2,4,5,6,7,8,9,10,1,3,0]],extra['base_linear_vel'][0,0],extra['base_angular_vel'][0,2]].astype('float32')
            images=[];clouds=[]
            for i in [t-1,t]:
                if i<0:images.extend([None,None,None]);clouds.append(None)
                else:
                    gi,_=obs(i)
                    images.extend([Image.fromarray(gi[f'sensor_data/{c}/rgb'][0]) for c in ['fetch_head','fetch_hand']]+[None])
                    cloud=gi['pointcloud/xyzrgb'][:]
                    clouds.append(cloud.reshape(-1,6))
            captured={}
            original=wrapper.policy.predict_action
            def capture(**kwargs):
                captured.update({k:v.detach().cpu().clone() for k,v in kwargs.items()})
                return torch.zeros(1,2,128,device='cuda')
            wrapper.policy.predict_action=capture
            try:
                with torch.no_grad():wrapper.step(torch.from_numpy(native)[None],images,embedding,clouds,extra)
            finally:wrapper.policy.predict_action=original
            # Native preprocessing is reused; dummy action is never sent to an environment.
            indices=[min(t+j,end-1) for j in range(2)]
            actions=np.stack([to_unified(h[f'actions/{i:04d}'][0]) for i in indices])
            assert np.isfinite(actions).all() and np.abs(actions).max()<=1, 'Invalid normalized teacher labels'
            target=np.array([min((t+j+1-window['start'])/(end-window['start']),1) for j in range(2)])
            valid=np.array([t+j<end for j in range(2)])
            cache.append((window,captured,torch.tensor(actions)[None],
                          torch.tensor(target,dtype=torch.float32)[None],torch.tensor(valid)[None]))
        report.update(status='caching_real_features',cached_examples=len(cache));save()
    del text_model
    paths=[name for name,module in wrapper.policy.model.named_modules()
           if isinstance(module,torch.nn.Linear) and name.endswith(('attn.qkv','cross_attn.q','cross_attn.kv'))]
    assert paths, 'No native insertion targets'
    report['projection_paths']=paths
    tool=ACDiTFamilyTool(wrapper.policy,paths)
    params=[p for p in tool.parameters() if p.requires_grad]
    report['trainable_parameters']=sum(p.numel() for p in params)
    trainable_before={n:p.detach().cpu().clone() for n,p in tool.named_parameters() if p.requires_grad}
    # Verify all native tensors, not just parameter-count assertions.
    def frozen_digest():
        sha=hashlib.sha256()
        for name,p in tool.named_parameters():
            if not p.requires_grad:sha.update(name.encode());sha.update(p.detach().cpu().contiguous().numpy().tobytes())
        return sha.hexdigest()
    before=frozen_digest()
    optimizer=torch.optim.AdamW(params,lr=1e-4)
    rng=random.Random(4017)
    training={f:[row for row in cache if row[0]['split']=='train' and row[0]['family']==f] for f in FAMILIES}
    validation=[row for row in cache if row[0]['split']=='validation']
    def run_loss(row):
        w,batch,actions,target,valid=row
        return tool.training_loss(w['family'],{**{k:v.to('cuda') for k,v in batch.items()},
            'action_gt':actions.to('cuda')},target.to('cuda'),valid.to('cuda'))
    def checkpoint(name):
        torch.save({'trainable':{n:p.detach().cpu() for n,p in tool.named_parameters() if p.requires_grad},
                    'optimizer':optimizer.state_dict(),'updates':report['updates'],'report':report,
                    'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                    'sampler_rng':rng.getstate()},out/name)
    def validate():
        rows=[]
        # Identical held-out diffusion randomness at each checkpoint.
        with torch.random.fork_rng(devices=[0]),torch.no_grad():
            torch.manual_seed(8181)
            for row in validation:
                loss=run_loss(row)
                rows.append({'family':row[0]['family'],'loss':float(loss['loss'])})
        return rows
    best=float('inf');training_start=time.monotonic()
    report['status']='training';save()
    with (out/'updates.jsonl').open('w') as log:
        for step in range(2000 if args.full else 20):
            if args.full and time.monotonic()-training_start>=6900:break
            family=FAMILIES[step%4]
            optimizer.zero_grad(set_to_none=True)
            totals={k:0. for k in ['loss','action_loss','progress_loss']}
            accumulation=8 if args.full else 1
            for micro in range(accumulation):
                result=run_loss(rng.choice(training[family]) if args.full else training[family][0])
                if not torch.isfinite(result['loss']):raise ValueError('nonfinite training loss')
                (result['loss']/accumulation).backward()
                for k in totals:totals[k]+=float(result[k].detach())/accumulation
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):raise ValueError('nonfinite gradients')
            for layer in tool.layers:
                assert all(layer.a[f].grad is None and layer.b[f].grad is None for f in FAMILIES if f!=family)
            torch.nn.utils.clip_grad_norm_(params,1.)
            optimizer.step()
            report['updates']=step+1
            log.write(json.dumps({'step':step+1,'family':family,**totals})+'\n');log.flush();save()
            if args.full and (step+1)%50==0:checkpoint('latest.pt')
            if args.full and (step+1)%200==0:
                values=validate();score=float(np.mean([r['loss'] for r in values]))
                report['validation']=values;report['validation_step']=step+1
                checkpoint(f'checkpoint-{step+1}.pt')
                if score<best:best=score;checkpoint('best.pt');report['best_step']=step+1
    report['frozen_native_unchanged']=before==frozen_digest()
    assert report['frozen_native_unchanged']
    changed=[n for n,p in tool.named_parameters() if p.requires_grad and not torch.equal(p.detach().cpu(),trainable_before[n])]
    report['changed_parameters_by_family']={f:sum(n.endswith((f'.a.{f}',f'.b.{f}')) for n in changed) for f in FAMILIES}
    report['progress_head_changed']=any(n.startswith('progress.') for n in changed)
    assert all(report['changed_parameters_by_family'].values()) and report['progress_head_changed'], 'Training did not update every component'
    report['validation']=validate();report['validation_step']=report['updates']
    score=float(np.mean([r['loss'] for r in report['validation']]))
    if args.full and score<best:report['best_step']=report['updates'];checkpoint('best.pt')
    checkpoint('final.pt' if args.full else 'gate-checkpoint.pt')
    report.update(status='bounded_sft_completed_not_evaluated' if args.full else '20_updates_passed_not_full_sft',
                  max_allocated_bytes=torch.cuda.max_memory_allocated())
except Exception as exc:
    report.update(status='failed',error=repr(exc));raise
finally:save()
