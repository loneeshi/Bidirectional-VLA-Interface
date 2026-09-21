"""One bounded stage of original AC-DiT; upstream architecture, dataset transforms and loss.

Single GPU FP32 and gradient accumulation adapt the published multi-GPU recipe.
The source directory and all inputs are fixed by the pipeline contract.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,choices=[1,2],required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--mobility',type=Path);p.add_argument('--preflight-only',action='store_true')
    a=p.parse_args();root=Path.home()/'bvi-research';source=root/'src/AC-DiT-rdt-20260919'
    assert os.environ['CUDA_VISIBLE_DEVICES']=='GPU-b7ebba23-7824-7601-df32-be55628936c3'
    sys.path.insert(0,str(source));os.chdir(source)
    import numpy as np
    import torch
    import yaml
    from models.dit_runner import DiTRunner
    from models.acdit_runner import ACDiTRunner
    from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
    from train.dataset import VLAConsumerDataset,DataCollatorForVLAConsumerDataset
    from data.hdf5_mshab_dataset import HDF5MSHABDataset
    torch.set_num_threads(2);random.seed(20260919);np.random.seed(20260919);torch.manual_seed(20260919)
    a.output.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load((source/'configs'/('config_s.yaml' if a.stage==1 else 'config.yaml')).read_text())
    roster=json.loads((a.data/'training-manifest.json').read_text())
    config_model=config['model']
    stage1=a.stage==1;accumulation=24 if stage1 else 16;lr=1e-4 if stage1 else 1e-5
    max_steps=30000 if stage1 else 20000;channels=[100,102] if stage1 else [0,1,2,3,4,5,6,10,125,126,127,100,102]
    report={'status':'initializing','stage':a.stage,'updates':0,'microbatches':0,'learning_rate':lr,
        'gradient_accumulation':accumulation,'microbatch':1,'max_updates':max_steps,'precision':'float32',
        'api_calls':0,'deadline_unix':a.deadline_unix,'best_score':None,'best_checkpoint':None,'preflight_only':a.preflight_only,'ema':False}
    started=time.monotonic()
    stop=False
    def on_signal(signum,frame):
        nonlocal stop
        stop=True;report['signal']=signum
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)
    def save_status():
        report['wall_seconds']=time.monotonic()-started;report['updated_unix']=time.time()
        tmp=a.output/'status.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.output/'status.json')
    save_status()
    try:
        vision=SiglipVisionTower(vision_tower='google/siglip-so400m-patch14-384',args=None)
        kwargs=dict(action_dim=128,pred_horizon=2,config=config_model,
            lang_token_dim=1152,img_token_dim=1152,pc_token_dim=1152,state_token_dim=128,
            max_lang_cond_len=1024,img_cond_len=4374,pc_cond_len=256,in_context_cond_dim=18,dtype=torch.float32)
        if not stage1:
            if a.mobility is None: raise ValueError('Stage 2 requires this run stage 1 checkpoint')
            kwargs['mobility_head_ckpt_path']=str(a.mobility)
        model=(DiTRunner if stage1 else ACDiTRunner)(**kwargs)
        weights=torch.load(root/'checkpoints/acdit-rdt-init-20260919'/('rdt-170m' if stage1 else 'rdt-1b')/'pytorch_model.bin',map_location='cpu',weights_only=True)
        for key in list(weights):
            if key.startswith('lang_adaptor'): del weights[key]
        load=model.load_state_dict(weights,strict=False)
        if any(k.startswith('model.blocks.') for k in load.missing_keys): raise ValueError('Missing pretrained transformer block')
        (a.output/'initialization.json').write_text(json.dumps({'missing':load.missing_keys,'unexpected':load.unexpected_keys},indent=2))
        del weights;gc.collect()
        if not stage1:
            mobility=torch.load(a.mobility,map_location='cpu',weights_only=True)['module']
            for name in ['lang_adaptor','img_adaptor','state_adaptor','lift3d_adaptor','in_context_conditions_adaptor']:
                module=getattr(model,name+'_mobility_head')
                module.load_state_dict({k[len(name)+1:]:v for k,v in mobility.items() if k.startswith(name+'.')},strict=True)
                module.requires_grad_(False)
            model.lift3d.load_state_dict({k[7:]:v for k,v in mobility.items() if k.startswith('lift3d.')},strict=True)
            report['stage1_transfer']='model, all mobility input adaptors, lift3d; explicit upstream initialization fix'
            with a.mobility.open('rb') as f:report['mobility_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
            del mobility;gc.collect()
        frozen={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()
                for n,p in model.named_parameters() if not p.requires_grad}
        (a.output/'frozen-parameters.json').write_text(json.dumps(frozen,indent=2))
        # Non-EMA evaluation follows author evaluator; omit unused EMA to save time.
        trainable_parameters=[p for p in model.parameters() if p.requires_grad]
        model=model.float().to('cuda');vision.vision_tower.to(device='cuda',dtype=torch.float32).eval().requires_grad_(False)
        optimizer=torch.optim.AdamW(trainable_parameters,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
        trainable={n for n,p in model.named_parameters() if p.requires_grad}
        report['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
        class RosterDataset(HDF5MSHABDataset):
            def __init__(self,ids):
                super().__init__(str(a.data),config,mode='base_only' if stage1 else 'full')
                self.task_subtask_obj=[('set_table','pick','013_apple')]
                self.num_episode_per_task=roster['episodes'];self.ids=ids;self.last_meta=None
            def __len__(self): return len(self.ids)
            def get_item(self,index=None,state_only=False):
                parent=int(np.random.choice(self.ids)) if index is None else self.ids[index]
                valid,result=self.parse_hdf5_file(parent)
                if not valid: raise ValueError('Invalid frozen trajectory')
                self.last_meta={'parent':parent,'step':result['meta']['step_id']}
                return result
        def make_dataset(role):
            dataset=VLAConsumerDataset(config=config,tokenizer=None,image_processor=vision.image_processor,
                num_cameras=3,img_history_size=2,dataset_type='finetune',image_aug=role=='train',
                cond_mask_prob=.1 if role=='train' else 0,cam_ext_mask_prob=-1,
                state_noise_snr=40 if role=='train' else None,use_hdf5=True,hdf5_dataset_name='mshab',
                use_precomp_lang_embed=True,base_only=stage1,data_dir=str(a.data))
            dataset.hdf5_dataset=RosterDataset(roster[role+'_parents'])
            dataset.dataset_stat['mshab']['state_mean']=roster['train_universal_state_mean']
            return dataset
        train=make_dataset('train');dev=make_dataset('validation');collate=DataCollatorForVLAConsumerDataset(None)
        def inputs(dataset):
            batch=collate([dataset[0]])
            batch={k:(v.to('cuda',dtype=torch.float32) if v.is_floating_point() else v.to('cuda'))
                   if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
            with torch.no_grad():
                images=batch['images'];b,_,c,h,w=images.shape
                img=vision(images.reshape(-1,c,h,w)).detach().reshape(b,-1,vision.hidden_size)
            pc=model.encode_pointcloud(batch['pointclouds'],weight_dtype=torch.float32)
            args=dict(lang_tokens=batch['lang_embeds'],lang_attn_mask=batch['lang_attn_mask'],img_tokens=img,
                pc_tokens=pc,state_tokens=batch['states'][:,-1:,:],action_mask=batch['state_elem_mask'].unsqueeze(1),
                ctrl_freqs=batch['ctrl_freqs'],in_context_conditions=batch['in_context_conditions'])
            return args,batch['actions']
        def evaluate():
            states=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state())
            random.seed(92026);np.random.seed(92026);torch.manual_seed(92026)
            model.eval();pred=[];target=[];identities=[]
            try:
                with torch.no_grad():
                    for _ in range(16):
                        inp,gt=inputs(dev);pred.append(model.predict_action(**inp)[0,0,channels].cpu().numpy())
                        target.append(gt[0,0,channels].cpu().numpy());identities.append(dev.hdf5_dataset.last_meta)
                predictions=np.stack(pred);targets=np.stack(target)
                if not np.isfinite(predictions).all(): raise ValueError('Nonfinite development prediction')
                rmse=np.sqrt(np.mean((predictions-targets)**2,axis=0));zero=np.sqrt(np.mean(targets**2,axis=0))
                ratios=np.divide(rmse,zero,out=np.full_like(rmse,np.nan),where=zero>1e-8)
                score=float(np.sqrt(np.mean((predictions-targets)**2)))
                entry={'update':report['updates'],'rmse':rmse.tolist(),'zero_rmse':zero.tolist(),
                    'ratios':[None if not np.isfinite(v) else float(v) for v in ratios],
                    'score':score,'universal_channels':channels,'identities':identities}
                np.savez_compressed(a.output/f'dev-{report["updates"]:06d}.npz',prediction=predictions,target=targets)
                with (a.output/'validation.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
                return score
            finally:
                random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2]);torch.cuda.set_rng_state(states[3]);model.train()
        def checkpoint(label,verify=False):
            path=a.output/f'{label}.pt';tmp=path.with_suffix('.tmp')
            state={k:v.detach().cpu() for k,v in model.state_dict().items()}
            torch.save({'module':state,'updates':report['updates'],'stage':a.stage,'precision':'float32'},tmp)
            tmp.replace(path)
            if verify:
                restored=torch.load(path,map_location='cpu',weights_only=True)['module']
                if set(restored)!=set(state) or any(not torch.equal(v,restored[k]) for k,v in state.items()):
                    raise RuntimeError('Checkpoint tensor round-trip mismatch')
                report['checkpoint_roundtrip_verified']=True
                del restored
            del state;gc.collect()
            return str(path)
        def save_latest():
            slot=report.get('checkpoint_generation',0)%2
            label=f'latest-{slot}'
            report['latest_checkpoint']=checkpoint(label,verify=report['updates']==1)
            tmp=a.output/f'{label}-state.tmp';state_path=a.output/f'{label}-state.pt'
            torch.save({'optimizer':optimizer.state_dict(),'updates':report['updates'],
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},tmp)
            tmp.replace(state_path)
            if report['updates']==1:
                restored=torch.load(state_path,map_location='cpu',weights_only=False)
                if restored['updates']!=1 or not restored['optimizer']['state']:
                    raise RuntimeError('Optimizer checkpoint restore failed')
                for entry in restored['optimizer']['state'].values():
                    for key in ['exp_avg','exp_avg_sq']:
                        if not torch.isfinite(entry[key]).all():raise RuntimeError('Nonfinite saved optimizer state')
                report['optimizer_checkpoint_verified']=True
                del restored;gc.collect()
            manifest={'updates':report['updates'],'weights':report['latest_checkpoint'],
                      'training_state':str(state_path),'generation':report.get('checkpoint_generation',0)}
            tmp=a.output/'latest-checkpoint.tmp';tmp.write_text(json.dumps(manifest,indent=2));tmp.replace(a.output/'latest-checkpoint.json')
            report['checkpoint_generation']=report.get('checkpoint_generation',0)+1
        if a.preflight_only:
            model.train();inp,gt=inputs(train);loss=model(**inp,action_gt=gt)['loss']
            if not torch.isfinite(loss):raise ValueError('Nonfinite preflight loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(trainable_parameters,1.0,error_if_nonfinite=True)
            if not any(p.grad is not None and torch.count_nonzero(p.grad).item()>0 for p in model.model.parameters()):
                raise ValueError('Main transformer has no gradient')
            for param in trainable_parameters:
                if param.grad is not None:
                    optimizer.state[param].update(step=torch.tensor(0.),exp_avg=torch.zeros_like(param),exp_avg_sq=torch.zeros_like(param))
            torch.cuda.synchronize()
            report.update(preflight_loss=float(loss.detach()),grad_norm=float(norm),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),optimizer_state_allocated=True,optimizer_updates=0)
            optimizer.zero_grad(set_to_none=True)
            report['final_checkpoint']=checkpoint('preflight',verify=True)
            report['status']='preflight_passed';save_status();print(json.dumps(report),flush=True);return
        initial=evaluate();report['initial_dev_score']=initial;save_status()
        model.train();report['status']='training';save_status();last_save=time.monotonic()
        while report['updates']<max_steps and time.time()<a.deadline_unix-600 and not stop:
            optimizer.zero_grad(set_to_none=True);loss_sum=0;tick=time.monotonic();completed_micro=0
            for micro in range(accumulation):
                if stop or time.time()>=a.deadline_unix-300:break
                inp,gt=inputs(train);loss=model(**inp,action_gt=gt)['loss']
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                (loss/accumulation).backward();loss_sum+=float(loss.detach());report['microbatches']+=1;completed_micro+=1
                if micro%4==0:save_status()
            if completed_micro!=accumulation:break
            norm=torch.nn.utils.clip_grad_norm_(trainable_parameters,1.0,error_if_nonfinite=True)
            optimizer.step();report['updates']+=1
            report.update(last_loss=loss_sum/accumulation,last_grad_norm=float(norm),peak_gpu_bytes=torch.cuda.max_memory_allocated(),last_update_seconds=time.monotonic()-tick)
            with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps({'update':report['updates'],'loss':report['last_loss'],'grad_norm':float(norm),'seconds':report['last_update_seconds'],'unix':time.time()})+'\n')
            if report['updates']%1==0:
                save_status();print(json.dumps(report),flush=True)
            if report['updates']==1 or report['updates']%100==0 or time.monotonic()-last_save>=900:
                score=evaluate()
                if report['best_score'] is None or score<report['best_score']:
                    report['best_score']=score;report['best_checkpoint']=checkpoint('best')
                save_latest();last_save=time.monotonic();save_status()
        if report['updates']==0: raise RuntimeError('No optimizer update within stage bound')
        score=evaluate()
        save_latest();report['final_checkpoint']=report['latest_checkpoint']
        if report['best_score'] is None or score<report['best_score']:
            report['best_score']=score;report['best_checkpoint']=checkpoint('best')
        changed=[n for n,v in model.named_parameters() if n in frozen and hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()!=frozen[n]]
        if changed:raise RuntimeError('Frozen parameters changed: '+str(changed[:5]))
        report['frozen_parameters_verified']=True
        report['status']='completed_bounded_stage';save_status()
    except Exception as exc:
        report.update(status='error',error=repr(exc));save_status();raise


if __name__=='__main__': main()
