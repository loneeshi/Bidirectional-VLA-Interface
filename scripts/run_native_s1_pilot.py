"""Bounded ordinary S1 pilot using author train functions and explicit data roots.

Default is CPU data preflight. --train is explicit, GPU1 only; caller must also
apply an outer timeout. No S2 family/progress updates and no automatic long run.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--normalizer',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--train',action='store_true')
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--seconds',type=int,default=3600)
    a=p.parse_args()
    if not 1<=a.steps<=100 or not 1<=a.seconds<=3600:
        p.error('Pilot bounds: <=100 updates, <=3600 seconds')
    os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3' if a.train else '',
                      JAX_PLATFORMS='cuda' if a.train else 'cpu',
                      XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2')
    import numpy as np
    import jax
    import openpi
    from openpi.models import model as models
    from openpi.training import data_loader
    from fetch_native_s1_config import config,native_dataset
    sys.path.insert(0,str(Path(openpi.__file__).resolve().parents[2]))
    from scripts import train
    a.output.mkdir(parents=True,exist_ok=False)
    if not (a.checkpoint/'_METADATA').is_file():
        raise ValueError('Actual parameter checkpoint required')
    cfg=config('bvi/s1-official-pick-pilot-train',str(a.output),str(a.checkpoint),a.normalizer,steps=a.steps,batch=1)
    cfg=dataclasses.replace(cfg,log_interval=1,save_interval=a.steps,seed=7)
    dc=cfg.data.create(cfg.assets_dirs,cfg.model)
    train_data=native_dataset(dc,cfg.model,a.dataset/'train')
    val_data=native_dataset(dc,cfg.model,a.dataset/'validation')
    report=dict(stage='S1_ordinary_SFT',status='data_preflight_passed',training_updates=0,
                train_frames=len(train_data),validation_frames=len(val_data),
                progress_training=False,heldout_selection=False,
                checkpoint=str(a.checkpoint),microbatch=1,accumulation=1,
                max_updates=a.steps,max_seconds=a.seconds)
    def save():
        (a.output/'pilot-status.json').write_text(json.dumps(report,indent=2)+'\n')
    # Validate both split model inputs before any model allocation.
    for dataset in (train_data,val_data):
        row=dataset[0]
        assert row['state'].shape==(32,) and row['actions'].shape==(10,32)
    save()
    if not a.train:return
    started=time.monotonic()
    durations=[]

    class Loader:
        def data_config(self):return dc
        def __iter__(self):
            rng=np.random.default_rng(7)
            while True:
                for i in rng.permutation(len(train_data)):
                    if time.monotonic()-started>a.seconds:
                        raise TimeoutError('S1 pilot wall limit')
                    before=time.monotonic()
                    row=train_data[int(i)]
                    batch=jax.tree.map(lambda x:np.asarray(x)[None],row)
                    obs=models.Observation.from_dict(batch)
                    # Author action-only train_step requires two extra fields.
                    # Progress head/loss are disabled; these zeros are not labels.
                    value=(obs,batch['actions'],np.zeros((1,10),np.float32),np.zeros(1,np.int32))
                    value=jax.device_put(value,self.sharding)
                    yield value
                    durations.append(time.monotonic()-before)

    original=data_loader.create_data_loader
    def factory(config,*,sharding=None,**kwargs):
        if config.use_val_set or config.progress_loss_weight!=0:
            raise ValueError('S1 runner cannot train progress or hash-repartition parents')
        loader=Loader();loader.sharding=sharding;return loader
    data_loader.create_data_loader=factory
    try:
        report['status']='initializing_and_training';save()
        train.main(cfg)
        report.update(status='pilot_complete',training_updates=a.steps,
                      elapsed_seconds=time.monotonic()-started,
                      steady_samples_per_second=(len(durations[10:])/sum(durations[10:])) if durations[10:] else None,
                      throughput_note='microbatch1; sample preparation through completed step, excluding first10; includes preprocessing/device transfer/logging; excludes initial model load/compilation warmup')
    except Exception as exc:
        report.update(status='failed',error=repr(exc),elapsed_seconds=time.monotonic()-started,
                      training_updates=None,updates_note='Consult step logs; failure may follow partial updates')
        raise
    finally:
        data_loader.create_data_loader=original;save()


if __name__=='__main__':main()
