"""One-time deterministic construction of the bounded single-task runner."""
from pathlib import Path

p=Path('scripts/train_acdit_rdt_bounded.py')
s=p.read_text()
def replace(old,new):
    global s
    if old not in s: raise ValueError('Missing source anchor: '+old[:90])
    s=s.replace(old,new)
replace('import random\n','import random\nimport signal\n')
replace("p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--mobility',type=Path)","p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--mobility',type=Path);p.add_argument('--preflight-only',action='store_true')")
replace("'best_score':None,'best_checkpoint':None}","'best_score':None,'best_checkpoint':None,'preflight_only':a.preflight_only,'ema':False}")
replace('    started=time.monotonic()','''    started=time.monotonic()
    stop=False
    def on_signal(signum,frame):
        nonlocal stop
        stop=True;report['signal']=signum
    signal.signal(signal.SIGTERM,on_signal)
    signal.signal(signal.SIGINT,on_signal)''')
replace("report['wall_seconds']=time.monotonic()-started", "report['wall_seconds']=time.monotonic()-started;report['updated_unix']=time.time()")
replace('        del weights;gc.collect()','''        del weights;gc.collect()
        if not stage1:
            mobility=torch.load(a.mobility,map_location='cpu',weights_only=True)['module']
            for name in ['lang_adaptor','img_adaptor','state_adaptor','lift3d_adaptor','in_context_conditions_adaptor']:
                module=getattr(model,name+'_mobility_head')
                module.load_state_dict({k[len(name)+1:]:v for k,v in mobility.items() if k.startswith(name+'.')},strict=True)
                module.requires_grad_(False)
            model.lift3d.load_state_dict({k[7:]:v for k,v in mobility.items() if k.startswith('lift3d.')},strict=True)
            report['stage1_transfer']='model, all mobility input adaptors, lift3d; explicit upstream initialization fix'
            with a.mobility.open('rb') as f:report['mobility_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
            del mobility;gc.collect()''')
replace('''        # EMA is retained on CPU to fit the full original network on this 48 GiB GPU.
        ema={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}''','''        # Non-EMA evaluation follows author evaluator; omit unused EMA to save time.
        trainable_parameters=[p for p in model.parameters() if p.requires_grad]''')
replace("vision.vision_tower.to(device='cuda',dtype=torch.float32).eval()","vision.vision_tower.to(device='cuda',dtype=torch.float32).eval().requires_grad_(False)")
replace("torch.optim.AdamW(model.parameters(),lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01)","torch.optim.AdamW(trainable_parameters,lr=lr,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)")
replace('self.num_episode_per_task=1000',"self.num_episode_per_task=roster['episodes']")
replace('for _ in range(32):','for _ in range(16):')
replace("predictions=np.stack(pred);targets=np.stack(target)","predictions=np.stack(pred);targets=np.stack(target)\n                if not np.isfinite(predictions).all(): raise ValueError('Nonfinite development prediction')")
start=s.index('        def checkpoint(label):')
end=s.index('        initial=evaluate()',start)
s=s[:start]+'''        def checkpoint(label,verify=False):
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
            report['latest_checkpoint']=checkpoint('latest',verify=report['updates']==1)
            tmp=a.output/'latest-state.tmp'
            torch.save({'optimizer':optimizer.state_dict(),'updates':report['updates'],
                'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                'numpy_rng':np.random.get_state(),'python_rng':random.getstate()},tmp)
            tmp.replace(a.output/'latest-state.pt')
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
''' +s[end:]
replace("model.train();report['status']='training'","model.train();report['status']='training';save_status();last_save=time.monotonic()")
replace("while report['updates']<max_steps and time.time()<a.deadline_unix-600:","while report['updates']<max_steps and time.time()<a.deadline_unix-600 and not stop:")
replace('optimizer.zero_grad(set_to_none=True);loss_sum=0','optimizer.zero_grad(set_to_none=True);loss_sum=0;tick=time.monotonic();completed_micro=0')
replace('''            for _ in range(accumulation):
                inp,gt=inputs(train)''','''            for micro in range(accumulation):
                if stop or time.time()>=a.deadline_unix-300:break
                inp,gt=inputs(train)''')
replace("report['microbatches']+=1", "report['microbatches']+=1;completed_micro+=1\n                if micro%4==0:save_status()")
replace("            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)","            if completed_micro!=accumulation:break\n            norm=torch.nn.utils.clip_grad_norm_(trainable_parameters,1.0,error_if_nonfinite=True)")
start=s.index('            # Same EMA warmup equation')
end=s.index('            report.update(last_loss=',start)
s=s[:start]+s[end:]
replace("peak_gpu_bytes=torch.cuda.max_memory_allocated())\n            if report['updates']%10==0 or report['updates']==1:","peak_gpu_bytes=torch.cuda.max_memory_allocated(),last_update_seconds=time.monotonic()-tick)\n            with (a.output/'train.jsonl').open('a') as f:f.write(json.dumps({'update':report['updates'],'loss':report['last_loss'],'grad_norm':float(norm),'seconds':report['last_update_seconds'],'unix':time.time()})+'\\n')\n            if report['updates']%1==0:")
replace("if report['updates']%100==0:","if report['updates']==1 or report['updates']%100==0 or time.monotonic()-last_save>=900:")
replace("checkpoint(f'best-{report[\"updates\"]:06d}')","checkpoint('best')")
replace("                save_status()\n        if report['updates']==0", "                save_latest();last_save=time.monotonic();save_status()\n        if report['updates']==0")
replace("report['final_checkpoint']=checkpoint(f'final-{report[\"updates\"]:06d}')","save_latest();report['final_checkpoint']=report['latest_checkpoint']")
replace("report['best_checkpoint']=report['final_checkpoint']","report['best_checkpoint']=checkpoint('best')")
replace("        report['status']='completed_bounded_stage';save_status()",'''        changed=[n for n,v in model.named_parameters() if n in frozen and hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()!=frozen[n]]
        if changed:raise RuntimeError('Frozen parameters changed: '+str(changed[:5]))
        report['frozen_parameters_verified']=True
        report['status']='completed_bounded_stage';save_status()''')
p.write_text(s)
print('Runner upgraded; review generated diff before execution.')
