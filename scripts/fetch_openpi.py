"""Fetch-specific pi05 LoRA pipeline. Run inside the pinned openpi environment.

Only successful, contiguous Pick/Place segments enter this diagnostic dataset.
Controller-normalized actions are learned directly (no second delta transform).
Training on seed 1 does not establish held-out benchmark performance.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np


def convert(runs, repo_id, include_velocity=False):
    from PIL import Image
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    features = {
        'image': {'dtype': 'image', 'shape': (128,128,3), 'names': ['height','width','channel']},
        'wrist_image': {'dtype': 'image', 'shape': (128,128,3), 'names': ['height','width','channel']},
        'state': {'dtype': 'float32', 'shape': (30 if include_velocity else 15,), 'names': ['state']},
        'actions': {'dtype': 'float32', 'shape': (13,), 'names': ['actions']},
    }
    dataset = LeRobotDataset.create(repo_id=repo_id, robot_type='fetch', fps=20,
        features=features, image_writer_threads=2, image_writer_processes=0)
    manifest = {'repo_id': repo_id, 'split': 'seed1_training_diagnostic', 'segments': [],
                'action_convention': 'Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity',
                'state_components':['qpos','qvel'] if include_velocity else ['qpos']}
    joint_order = None
    for directory in map(Path, runs):
        events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
        applied={e['frame_id']:e['controller_action'][0] for e in events if e['event']=='mshab_step'}
        rows = [e for e in events if e['event']=='demonstration_step' and e['skill'] in ('pick','place')]
        for index in sorted({e['subtask_index'] for e in rows}):
            segment = [e for e in rows if e['subtask_index']==index]
            if segment[-1]['feedback']['adapter_subtask_after'] <= index:
                continue  # Never train failed Place tails as demonstrations of success.
            for row in segment:
                if joint_order is None: joint_order = row['joint_names']
                if row['joint_names'] != joint_order: raise ValueError('Joint order changed')
                state = np.asarray(row['qpos'][0], np.float32)
                if include_velocity: state=np.concatenate([state,np.asarray(row['qvel'][0],np.float32)])
                # Match post-wrapper actions, including official stationary-head masking.
                action = np.asarray(applied[row['next_frame_id']], np.float32)
                if state.shape != (30 if include_velocity else 15,) or action.shape != (13,): raise ValueError('Contract mismatch')
                if not np.isfinite(state).all() or not np.isfinite(action).all(): raise ValueError('Nonfinite data')
                if (abs(action)>1.00001).any(): raise ValueError('Unnormalized controller actions')
                def frame(camera):
                    path=directory/'frames'/f"{row['frame_id']}-{camera}.png"
                    arr=np.asarray(Image.open(path).convert('RGB'))
                    if arr.shape != (128,128,3): raise ValueError(f'Unexpected image shape {arr.shape}')
                    return arr
                dataset.add_frame({'image':frame('fetch_head'), 'wrist_image':frame('fetch_hand'),
                    'state':state, 'actions':action, 'task':row['target_description']})
            dataset.save_episode()
            manifest['segments'].append({'run':str(directory),'subtask_index':index,
                'skill':segment[0]['skill'],'frames':len(segment),
                'events_sha256':hashlib.sha256((directory/'events.jsonl').read_bytes()).hexdigest()})
    if not manifest['segments']: raise ValueError('No successful manipulation segments')
    manifest['joint_names']=joint_order
    (Path(dataset.root)/'bvi-manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


def config(repo_id, work, steps=1000, batch=4,state_input=True,include_velocity=False,init_checkpoint=None):
    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.policies import libero_policy
    from openpi.training import config as c, optimizer, weight_loaders

    @dataclasses.dataclass(frozen=True)
    class FetchOutputs(transforms.DataTransformFn):
        def __call__(self, data):
            return {'actions':np.asarray(data['actions'][...,:13])}

    @dataclasses.dataclass(frozen=True)
    class FetchData(c.LeRobotLiberoDataConfig):
        def create(self, assets_dirs, model_config):
            result=super().create(assets_dirs,model_config)
            return dataclasses.replace(result, data_transforms=transforms.Group(
                inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
                outputs=[FetchOutputs()]))

    model=pi0_config.Pi0Config(pi05=True,action_horizon=10,discrete_state_input=state_input,
        paligemma_variant='gemma_2b_lora', action_expert_variant='gemma_300m_lora')
    name='pi05_fetch_lora_velocity' if include_velocity else ('pi05_fetch_lora_state' if state_input else 'pi05_fetch_lora')
    return c.TrainConfig(name=name,exp_name='seed1-diagnostic',model=model,
        data=FetchData(repo_id=repo_id,base_config=c.DataConfig(prompt_from_task=True),
                       extra_delta_transform=False),
        weight_loader=weight_loaders.CheckpointWeightLoader(init_checkpoint or 'gs://openpi-assets/checkpoints/pi05_base/params'),
        freeze_filter=model.get_freeze_filter(),ema_decay=None,batch_size=batch,num_workers=0,
        num_train_steps=steps,log_interval=10,save_interval=min(500,max(100,steps//4)),keep_period=None,
        assets_base_dir=str(Path(work)/'assets'), checkpoint_base_dir=str(Path(work)/'checkpoints'),
        lr_schedule=optimizer.CosineDecaySchedule(warmup_steps=20,peak_lr=1e-4,
                                                  decay_steps=steps,decay_lr=1e-5),
        wandb_enabled=False,policy_metadata={'robot':'fetch','state_dim':30 if include_velocity else 15,'action_dim':13,
            'state_components':['qpos','qvel'] if include_velocity else ['qpos'],
            'state_conditioning':state_input,
            'action_convention':'Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity',
            'training_repo':repo_id,'evaluation_scope':'training-scene-diagnostic'})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['convert','norm','train','serve'])
    p.add_argument('--runs',nargs='+')
    p.add_argument('--repo-id',default='bvi/fetch-seed1-diagnostic')
    p.add_argument('--work',default='/workspace/fetch-pi')
    p.add_argument('--openpi-root',default='/workspace/probe/openpi')
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--checkpoint')
    p.add_argument('--init-checkpoint',help='Explicit parameter directory for a warm-start experiment')
    p.add_argument('--include-velocity',action='store_true',help='Use named qpos and qvel (30 robot state values)')
    p.add_argument('--no-state-input',action='store_true',help='Historical vision-only pilot reproduction, not recommended for Fetch')
    p.add_argument('--port',type=int,default=8051)
    p.add_argument('--denoising-steps',type=int,default=10,
                   help='Native pi05 flow-matching inference steps (1..100); recorded in server metadata')
    a=p.parse_args()
    if not 1<=a.denoising_steps<=100: p.error('--denoising-steps must be in 1..100')
    if a.include_velocity and a.no_state_input: p.error('Velocity experiment requires state conditioning')
    if a.mode=='convert': return convert(a.runs,a.repo_id,a.include_velocity)
    sys.path.insert(0,a.openpi_root)
    cfg=config(a.repo_id,a.work,a.steps,a.batch,state_input=not a.no_state_input,
        include_velocity=a.include_velocity,init_checkpoint=a.init_checkpoint)
    if a.mode=='norm':
        from scripts.compute_norm_stats import create_torch_dataloader
        from openpi.shared import normalize
        dc=cfg.data.create(cfg.assets_dirs,cfg.model)
        loader,_=create_torch_dataloader(dc, cfg.model.action_horizon,cfg.batch_size,cfg.model,0)
        stats={key:normalize.RunningStats() for key in ('state','actions')}
        for batch in loader:
            for key in stats: stats[key].update(np.asarray(batch[key]))
        normalize.save(cfg.assets_dirs/dc.repo_id,{k:v.get_statistics() for k,v in stats.items()})
    elif a.mode=='train':
        from scripts.train import main as train, init_logging
        init_logging()
        train(cfg)
    elif a.mode=='serve':
        from openpi.policies.policy_config import create_trained_policy
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        if not a.checkpoint: p.error('--checkpoint required; never substitute a DROID checkpoint')
        cfg=dataclasses.replace(cfg,policy_metadata={**cfg.policy_metadata,
            'checkpoint':str(Path(a.checkpoint).resolve()),
            'denoising_steps':a.denoising_steps,
            'action_horizon':cfg.model.action_horizon,
            'server_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        policy=create_trained_policy(cfg,a.checkpoint,
                                    sample_kwargs={'num_steps':a.denoising_steps})
        WebsocketPolicyServer(policy,host='127.0.0.1',port=a.port,metadata=policy.metadata).serve_forever()


if __name__=='__main__': main()
