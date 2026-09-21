"""One bounded full-privileged42 diagnostic S1 epoch, explicit held-out validation and best/latest recovery.

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


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset','normalizer','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seconds',type=int,default=18000)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not 1<=a.seconds<=18000:raise ValueError('One-epoch wall bound <=18000s')
    os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3',
                      JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2')
    import numpy as np
    import jax
    import openpi
    from openpi.models import model as models
    from openpi.training import checkpoints,sharding
    from fetch_privileged_s1_config import config,native_dataset
    sys.path.insert(0,str(Path(openpi.__file__).resolve().parents[2]))
    from scripts import train
    a.output.mkdir(parents=True,exist_ok=a.resume)
    manifest=json.loads((a.dataset/'manifest.json').read_text())
    if len(manifest['parent_split']['train'])!=150 or len(manifest['parent_split']['validation'])!=50:
        raise ValueError('Throughput-selected medium150/50 data required')
    total=sum(e['exported_steps'] for e in manifest['episodes'] if e['split']=='train')
    if total != 6835:
        raise ValueError('Frozen diagnostic training budget requires6835 frames/updates')
    cfg=config('bvi/s1-privileged42-pick-medium-train',str(a.output),str(a.checkpoint),a.normalizer,steps=total,batch=1)
    cfg=dataclasses.replace(cfg,exp_name='privileged42-medium-one-epoch',seed=7)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    datasets={s:native_dataset(dc,cfg.model,a.dataset/s) for s in ('train','validation')}
    identity=dict(dataset_manifest_sha256=hashlib.sha256((a.dataset/'manifest.json').read_bytes()).hexdigest(),
                  norm_sha256=cfg.policy_metadata['normalizer_sha256'],checkpoint=str(a.checkpoint),
                  total_steps=total,seed=7,validation='all50parents_start_mid_last_preaction_fixed')
    identity_path=a.output/'identity.json'
    if a.resume:
        if json.loads(identity_path.read_text())!=identity:raise ValueError('Resume identity mismatch')
    else:identity_path.write_text(json.dumps(identity,indent=2))
    report=dict(status='initializing',stage='S1_privileged_diagnostic',privileged_policy_inputs=True,native_gate_eligible=False,updates=0,total_updates=total,api_calls=0,
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
    validation=[];offset=0
    for episode in (e for e in manifest['episodes'] if e['split']=='validation'):
        n=episode['exported_steps']
        validation.extend(offset+t for t in sorted({0,n//2,n-1}));offset+=n
    (a.output/'validation-indices.json').write_text(json.dumps(validation))
    def batch(split,index):
        row=datasets[split][int(index)]
        x=jax.tree.map(lambda v:np.asarray(v)[None],row)
        return jax.device_put((models.Observation.from_dict(x),x['actions'],
                               np.zeros((1,10),np.float32),np.zeros(1,np.int32)),ds_sharding)
    update=jax.jit(functools.partial(train.train_step,cfg),donate_argnums=(1,))
    evaluate=jax.jit(functools.partial(train.eval_step,cfg))
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
            with sharding.set_mesh(mesh):metrics=evaluate(state,batch('validation',index))
            loss=float(metrics['val/action_loss'])
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
        for i in range(step,total):
            if prior_seconds+time.monotonic()-started>=a.seconds:raise TimeoutError('S1 wall limit')
            with sharding.set_mesh(mesh):state,metrics=update(train_key,state,batch('train',permutation[i]))
            metrics=jax.device_get(metrics);step=int(state.step)
            if not all(np.isfinite(float(v)) for v in metrics.values()):raise ValueError('Nonfinite training metrics')
            report.update(updates=step,loss=float(metrics['loss']))
            if step==1 or step%10==0:
                save_report();print(json.dumps(report),flush=True)
            if step%1000==0 or step==total:
                report['latest_checkpoint']=checkpoint(latest)
                validate();save_report()
        report['status']='completed_one_epoch'
    except Exception as exc:
        report.update(status='failed',error=repr(exc),updates=step)
        if step>0:report['latest_checkpoint']=checkpoint(latest)
        raise
    finally:
        save_report();latest.wait_until_finished();best_manager.wait_until_finished()


if __name__=='__main__':main()
