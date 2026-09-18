"""One bounded invocation-aligned S1 epoch, explicit held-out validation and best/latest recovery.

Uses author initialization, action objective and optimizer. No TAPT progress.
Launch on lab GPU1 with an additional external timeout. Resume preserves the
fixed permutation and random key by restoring step and optimizer state.
"""
import argparse
import dataclasses
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','normalizer','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=18000)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args()
    if not 1<=a.seconds<=18000:raise ValueError('One-epoch wall bound <=18000s')
    gpu='GPU-b7ebba23-7824-7601-df32-be55628936c3'
    actual=subprocess.check_output(['nvidia-smi','-i','1','--query-gpu=uuid,memory.used',
                                   '--format=csv,noheader,nounits'],text=True).strip().split(',')
    if actual[0].strip()!=gpu or int(actual[1])>=1024:
        raise RuntimeError('Physical GPU1 identity/idle gate failed')
    os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3',
                      JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2')
    import numpy as np
    import jax
    import openpi
    from openpi.models import model as models
    from openpi.training import checkpoints,sharding
    from fetch_native_s1_config import config
    from fetch_ia_s1_dataset import InvocationDataset
    import ia_s1_steps
    sys.path.insert(0,str(Path(openpi.__file__).resolve().parents[2]))
    from scripts import train
    a.output.mkdir(parents=True,exist_ok=a.resume)
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if len(manifest['parent_split']['train'])!=150 or len(manifest['parent_split']['validation'])!=50:
        raise ValueError('Throughput-selected medium150/50 data required')
    total=sum(e['exported_steps'] for e in manifest['episodes'] if e['split']=='train')
    cfg=config('bvi/s1-official-pick-medium-train',str(a.output),str(a.checkpoint),a.normalizer,steps=total,batch=1)
    cfg=dataclasses.replace(cfg,exp_name='ia-one-epoch',seed=7,lr_schedule=ia_s1_steps.SampleSchedule(cfg.lr_schedule))
    metadata=dict(cfg.policy_metadata,training_stage='S1_IA_single_bank_no_progress',invocation_aligned=True)
    cfg=dataclasses.replace(cfg,policy_metadata=metadata)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    datasets={s:InvocationDataset(a.source,a.dataset/'manifest.json',a.normalizer,dc,cfg.model,s) for s in ('train','validation')}
    if len(datasets['train'])!=total:raise ValueError('IA sample exposure differs from frozen epoch')
    total_updates=(total+7)//8
    identity=dict(dataset_manifest_sha256=hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest(),
                  norm_sha256=cfg.policy_metadata['normalizer_sha256'],checkpoint=str(a.checkpoint),
                  total_samples=total,total_updates=total_updates,accumulation=8,seed=7,validation='first_current_frame_each_heldout_invocation_fixed',source_sha256=datasets['train'].provenance['source_sha256'],lr_clock='processed_samples_at_group_start',objective='masked_time_mean_then_sample_mean_action32')
    identity_path=a.output/'identity.json'
    if a.resume:
        if json.loads(identity_path.read_text())!=identity:raise ValueError('Resume identity mismatch')
    else:identity_path.write_text(json.dumps(identity,indent=2))
    report=dict(status='initializing',stage='S1-IA',updates=0,total_updates=total_updates,processed_samples=0,total_samples=total,api_calls=0,
                max_total_seconds=a.seconds,progress_training=False)
    report_path=a.output/'status.json'
    prior=json.loads(report_path.read_text()) if a.resume else {}
    prior_seconds=prior.get('elapsed_seconds',0.)
    started=time.monotonic()
    def save_report():
        report['elapsed_seconds']=prior_seconds+time.monotonic()-started
        tmp=report_path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(report_path)
    save_report()
    mesh=sharding.make_mesh(cfg.fsdp_devices)
    ds_sharding=jax.sharding.NamedSharding(mesh,jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    latest,resuming=checkpoints.initialize_checkpoint_dir(a.output/'latest',keep_period=None,overwrite=False,resume=a.resume)
    best_manager,_=checkpoints.initialize_checkpoint_dir(a.output/'best',keep_period=None,overwrite=False,resume=a.resume)
    class Assets:
        def data_config(self):return dc
    assets=Assets()
    state,_=train.init_train_state(cfg,jax.random.key(7),mesh,resume=resuming)
    jax.block_until_ready(state)
    if resuming:state=checkpoints.restore_state(latest,state,assets)
    step=int(state.step)
    best_path=a.output/'best.json'
    best_record=json.loads(best_path.read_text()) if a.resume and best_path.exists() else None
    best_loss=float('inf') if best_record is None else best_record['validation_action_loss']
    permutation=np.random.default_rng(7).permutation(total)
    validation=[]; seen=set()
    for i,row in enumerate(datasets['validation'].rows):
        key=(row['parent_id'],row['call_index'])
        if key not in seen:validation.append(i);seen.add(key)
    (a.output/'validation-indices.json').write_text(json.dumps(validation))
    def batch(split,index):
        row=datasets[split][int(index)]
        valid=~row.pop('actions_is_pad')
        if not valid.any():raise ValueError('No valid action target')
        x=jax.tree.map(lambda v:np.asarray(v)[None],row)
        return jax.device_put((models.Observation.from_dict(x),x['actions'],valid[None]),ds_sharding)
    grad=jax.jit(functools.partial(ia_s1_steps.micro_gradient,cfg))
    apply=jax.jit(functools.partial(ia_s1_steps.apply_average,cfg))
    evaluate=jax.jit(functools.partial(ia_s1_steps.evaluate,cfg))
    train_key=jax.random.key(7001)
    def checkpoint(manager):
        checkpoints.save_state(manager,state,assets,step);manager.wait_until_finished()
        path=Path(manager.directory)/str(step)/'assets'/dc.asset_id/'bvi-state-contract.json'
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(cfg.policy_metadata,indent=2))
        return str(Path(manager.directory)/str(step))
    def validate():
        nonlocal best_loss,best_record
        losses=[]
        for index in validation:
            if prior_seconds+time.monotonic()-started>=a.seconds:raise TimeoutError('S1 wall limit in validation')
            with sharding.set_mesh(mesh):value=evaluate(state,jax.random.fold_in(jax.random.key(123),index),batch('validation',index))
            loss=float(value)
            if not np.isfinite(loss):raise ValueError('Nonfinite validation loss')
            losses.append(loss)
        mean=float(np.mean(losses))
        with (a.output/'validation.jsonl').open('a') as f:f.write(json.dumps(dict(step=step,loss=mean,n=len(losses)))+'\n')
        report['validation_action_loss']=mean
        if mean<best_loss:
            path=checkpoint(best_manager);best_loss=mean
            best_record=dict(step=step,validation_action_loss=mean,checkpoint=path,selection='heldout_action_loss_only')
            best_path.write_text(json.dumps(best_record,indent=2))
        report['best']=best_record
    try:
        report['status']='training';save_report()
        for begin in range(min(step*8,total),total,8):
            summed=None; losses=[]
            count=min(8,total-begin)
            for i in range(begin,begin+count):
                if prior_seconds+time.monotonic()-started>=a.seconds:raise TimeoutError('S1-IA wall limit')
                with sharding.set_mesh(mesh):
                    loss,g=grad(state,jax.random.fold_in(train_key,i),batch('train',permutation[i]))
                value=float(loss)
                if not np.isfinite(value):raise ValueError('Nonfinite masked action loss')
                summed=g if summed is None else jax.tree.map(lambda x,y:x+y,summed,g)
                losses.append(value)
            with sharding.set_mesh(mesh):state,gn=apply(state,summed,count)
            step=int(state.step)
            if not np.isfinite(float(gn)):raise ValueError('Nonfinite accumulated gradient')
            report.update(updates=step,processed_samples=begin+count,effective_batch=count,
                          loss=float(np.mean(losses)),grad_norm=float(gn))
            if step==1 or step%10==0:
                save_report();print(json.dumps(report),flush=True)
            # First 20 updates are an in-run correctness gate, retained in the same epoch.
            if step==20:
                if not float(gn)>0:raise ValueError('Zero accumulated gradient at 20-update gate')
                report['gradient_gate']='20_finite_optimizer_updates_passed'
                report['latest_checkpoint']=checkpoint(latest)
                save_report()
            if step%200==0 or begin+count==total:
                report['latest_checkpoint']=checkpoint(latest)
                validate();save_report()
        report['status']='completed_one_epoch'
    except Exception as exc:
        report.update(status='failed',error=repr(exc),updates=step)
        if step>0:report['latest_checkpoint']=checkpoint(latest)
        raise
    finally:
        save_report();latest.wait_until_finished();best_manager.wait_until_finished()
        for dataset in datasets.values():dataset.close()


if __name__=='__main__':main()
