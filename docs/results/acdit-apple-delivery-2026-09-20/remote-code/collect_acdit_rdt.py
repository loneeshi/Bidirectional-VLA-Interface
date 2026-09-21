"""Collect native train-split SAC demonstrations with the original AC-DiT observations.

Runs only inside the once-only bounded pipeline. Failed episodes are retained.
All successful trajectories stop at first success, before native failure.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

UUID = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'


def batch_scenes(scene_ids, batch_id, num_envs, seed=20260919):
    """Cycle all available scenes; upstream otherwise always builds the first N."""
    ordered=sorted(scene_ids)
    if not ordered:raise ValueError('No training scenes')
    random.Random(seed).shuffle(ordered)
    return [ordered[(batch_id*num_envs+i)%len(ordered)] for i in range(num_envs)]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-unix',type=float,required=True);p.add_argument('--count',type=int,default=1000)
    p.add_argument('--min-successes',type=int,default=300);p.add_argument('--max-attempts',type=int,default=1500);p.add_argument('--num-envs',type=int,default=8)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=Path.home()/'bvi-research'
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    assert os.environ.get('PYTHONHASHSEED')=='20260919'
    os.environ['MS_ASSET_DIR']=str(root/'assets')
    import h5py
    import numpy as np
    import torch
    import yaml
    import gymnasium as gym
    from gymnasium import spaces
    import mshab.envs
    from mshab.envs.planner import plan_data_from_file
    from mshab.agents.sac import Agent
    from mani_skill.utils.common import flatten_state_dict
    torch.set_num_threads(2);random.seed(20260919);np.random.seed(20260919);torch.manual_seed(20260919)
    a.output.mkdir(parents=True,exist_ok=a.resume)
    started=time.monotonic(); report={'status':'collecting','target_successes':a.count,'attempts':0,
        'successes':0,'simulator_actions':0,'policy_calls':0,'api_calls':0,'training_updates':0,
        'episodes':[],'source':'new native official SAC demonstrations in pinned AC-DiT fork',
        'split':'train','gpu_uuid':UUID,'first_native_stop':True}
    if a.resume:
        previous=json.loads((a.output/'collection.json').read_text())
        report.update(previous)
        report.setdefault('prior_segments',[]).append({'wall_seconds':previous['wall_seconds'],'status':previous['status'],'error':previous.get('error')})
        report.update(status='collecting',error=None)
    def save():
        tmp=a.output/'collection.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(a.output/'collection.json')
    rearrange=root/'assets/data/scene_datasets/replica_cad_dataset/rearrange'
    plans=plan_data_from_file(rearrange/'task_plans/set_table/pick/train/013_apple.json')
    env=None
    try:
        env=gym.make('PickSubtaskTrain-v0',num_envs=a.num_envs,robot_uids='fetch',obs_mode='rgbdp',
            control_mode='pd_joint_delta_pos',render_mode='rgb_array',reward_mode='normalized_dense',
            sensor_configs={'shader_pack':'default'},human_render_camera_configs={'shader_pack':'default'},
            viewer_camera_configs={'shader_pack':'default'},sim_backend='gpu',render_backend='gpu',
            max_episode_steps=200,reconfiguration_freq=1,task_plans=plans.plans,scene_builder_cls=plans.dataset,
            spawn_data_fp=rearrange/'spawn_data/set_table/pick/train/spawn_data.pt',
            require_build_configs_repeated_equally_across_envs=False)
        raw=env.unwrapped
        pci=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=pci.bus_id','--format=csv,noheader'],text=True).strip()
        assert str(raw._render_device.pci_string).lower()[-7:]==pci.lower()[-7:]
        ck=root/'checkpoints/mshab/rl/set_table/pick/013_apple'
        cfg=yaml.safe_load((ck/'config.yml').read_text())['algo']
        keys=['actor_hidden_dims','critic_hidden_dims','critic_layer_norm','critic_dropout',
              'encoder_pixels_feature_dim','encoder_state_feature_dim','cnn_features','cnn_filters','cnn_strides','cnn_padding']
        policy=Agent(spaces.Dict({c+'_depth':spaces.Box(0,32767,(3,128,128),np.int16) for c in ['fetch_head','fetch_hand']}),
            (42,),(13,),**{k:cfg[k] for k in keys},log_std_min=cfg['actor_log_std_min'],
            log_std_max=cfg['actor_log_std_max'],device='cuda')
        policy.load_state_dict(torch.load(ck/'policy.pt',map_location='cpu',weights_only=True)['agent'],strict=True)
        policy.to('cuda').eval()
        report['teacher_sha256']=hashlib.sha256((ck/'policy.pt').read_bytes()).hexdigest()
        dataset_dir=a.output/'set_table/pick/train/013_apple';dataset_dir.mkdir(parents=True,exist_ok=a.resume)
        accepted=h5py.File(dataset_dir/'demonstrations.h5','r+' if a.resume else 'x')
        failed=h5py.File(a.output/'failed-demonstrations.h5','r+' if a.resume else 'x')
        try:
            first_batch=report.get('last_started_batch',report['attempts']//a.num_envs if a.resume else -1)+1
            for batch_id in range(first_batch,(a.max_attempts+a.num_envs-1)//a.num_envs):
                if time.time()>=a.deadline_unix-60: break
                report['last_started_batch']=batch_id;save()
                scenes=batch_scenes(raw.build_config_idx_to_task_plans,batch_id,a.num_envs)
                obs,info=env.reset(seed=20260919+batch_id,options={'reconfigure':True,'build_config_idxs':scenes})
                active=np.ones(a.num_envs,dtype=bool)
                trajectories=[{'obs':{},'actions':[],'success':[],'fail':[]} for _ in active]
                meta=[{'seed':20260919+batch_id,'env_index':i,'build_config_idx':int(raw.build_config_idxs[i]),
                    'init_config_idx':int(raw.init_config_idxs[i]),'task_plan_idx':int(raw.task_plan_idxs[i])} for i in range(a.num_envs)]
                frames={c:deque(maxlen=3) for c in ['fetch_head','fetch_hand']}
                def record(observation, mask):
                    flattened={}
                    for section in ['agent','extra']:
                        for key,value in observation[section].items(): flattened[f'{section}/{key}']=value
                    for c in frames:
                        for key in ['rgb','depth']: flattened[f'sensor_data/{c}/{key}']=observation['sensor_data'][c][key]
                        for key,value in observation['sensor_param'][c].items(): flattened[f'sensor_param/{c}/{key}']=value
                    flattened['pointcloud/xyzrgb']=observation['pointcloud']['xyzrgb']
                    for name,value in flattened.items():
                        value=value.detach().cpu().numpy()
                        for i in np.flatnonzero(mask): trajectories[i]['obs'].setdefault(name,[]).append(value[i].copy())
                record(obs,active)
                for step in range(200):
                    if time.time()>=a.deadline_unix-60: break
                    pixels={}
                    for c,q in frames.items():
                        frame=obs['sensor_data'][c]['depth'].permute(0,3,1,2)
                        for _ in range(3 if step==0 else 1): q.append(frame)
                        pixels[c+'_depth']=torch.cat(list(q),dim=1).float().contiguous()
                    extra={k:obs['extra'][k] for k in ['tcp_pose_wrt_base','obj_pose_wrt_base','goal_pos_wrt_base','is_grasped']}
                    state=torch.cat([flatten_state_dict(obs['agent'],use_torch=True),flatten_state_dict(extra,use_torch=True)],dim=1)
                    assert state.shape==(a.num_envs,42)
                    with torch.no_grad(): action=policy.actor(pixels,state.float(),compute_pi=False,compute_log_pi=False)[0]
                    finite=torch.isfinite(action).all(dim=1).cpu().numpy()
                    for i in np.flatnonzero(active & ~finite):
                        meta[i]['infrastructure_failure']='nonfinite_teacher_action_or_observation'
                    # Inactive environments can diverge after their first stop;
                    # never send their NaNs to physics or count them as successes.
                    active &= finite
                    if not active.any():break
                    action=action.clamp(-1,1);action[:,8:10]=0
                    action[torch.tensor(~active,device=action.device)]=0
                    nxt,reward,terminated,truncated,info=env.step(action)
                    acts=action.detach().cpu().numpy();success=info['success'].cpu().numpy().astype(bool)
                    failure=info['fail'].cpu().numpy().astype(bool)
                    done=(terminated|truncated).cpu().numpy().astype(bool)|success|failure
                    record(nxt,active)
                    for i in np.flatnonzero(active):
                        trajectories[i]['actions'].append(acts[i].copy())
                        trajectories[i]['success'].append(bool(success[i]))
                        trajectories[i]['fail'].append(bool(failure[i]))
                    report['simulator_actions']+=a.num_envs;report['policy_calls']+=a.num_envs
                    active &= ~done; obs=nxt
                    if not active.any(): break
                for i,traj in enumerate(trajectories):
                    if report['attempts']>=a.max_attempts:continue
                    if not traj['actions']:
                        report.setdefault('zero_action_rejections',[]).append(meta[i])
                        report['attempts']+=1
                        continue
                    passed=bool(traj['success'][-1] and not traj['fail'][-1] and 'infrastructure_failure' not in meta[i])
                    if passed and report['successes']>=a.count: break
                    destination=accepted if passed else failed
                    idx=report['successes'] if passed else len(failed)
                    group=destination.create_group(f'traj_{idx}')
                    for name,values in traj['obs'].items(): group.create_dataset('obs/'+name,data=np.stack(values),compression='lzf')
                    for name in ['actions','success','fail']: group.create_dataset(name,data=np.asarray(traj[name]),compression='lzf')
                    episode={**meta[i],'episode_id':idx,'success':passed,'steps':len(traj['actions']),
                             'file':'accepted' if passed else 'failed','ended_by_deadline':bool(active[i])}
                    group.attrs['provenance']=json.dumps(episode);report['episodes'].append(episode)
                    report['attempts']+=1;report['successes']+=int(passed)
                accepted.flush();failed.flush();save()
                print(json.dumps({k:report[k] for k in ['attempts','successes','simulator_actions']}),flush=True)
                if report['successes']>=a.count: break
        finally:
            accepted.close();failed.close()
        report['status']='complete' if report['successes']>=a.min_successes else 'insufficient_data_no_training'
    except Exception as exc:
        report.update(status='error',error=repr(exc));raise
    finally:
        if env is not None: env.close()
        report['wall_seconds']=time.monotonic()-started;save()
    if report['status']!='complete': raise RuntimeError(report['status'])


if __name__=='__main__': main()
