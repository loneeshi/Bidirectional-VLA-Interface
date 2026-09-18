"""Official deterministic SAC reference, same native task; not a VLA result.

Historical comparison is observation-matched only; full old state was not saved.
New simulator/controller snapshots support subsequent controlled comparisons.
"""
import os, sys, json, random, hashlib, subprocess, argparse
from pathlib import Path
from collections import deque
from run_lab_acdit_native import jsonable

root = Path.home() / 'bvi-research'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('seed',type=int)
parser.add_argument('--collect',action='store_true')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--reference-state',type=Path,required=True)
parser.add_argument('--task',choices=['pick','place'],default='pick')
args=parser.parse_args();seed=args.seed;task=args.task
split='train' if args.collect else 'val'
out=args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
used = int(subprocess.check_output(['nvidia-smi', '-i', '1', '--query-gpu=memory.used',
                                  '--format=csv,noheader,nounits'], text=True).strip())
if used > 1024:
    raise RuntimeError('GPU1 occupied')
os.environ.update(CUDA_VISIBLE_DEVICES='GPU-b7ebba23-7824-7601-df32-be55628936c3',
                  MS_ASSET_DIR=str(root/'assets'), OMP_NUM_THREADS='2')
assert os.environ.get('PYTHONHASHSEED') == str(seed)
import torch
import numpy as np
import yaml
import gymnasium as gym
import imageio.v2 as imageio
from gymnasium import spaces
import mshab.envs
from mshab.envs.planner import plan_data_from_file
from mshab.agents.sac import Agent
from mani_skill.utils.common import flatten_state_dict
torch.set_num_threads(2)
torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
source = root/'src/AC-DiT'; os.chdir(source)
rearrange = root/'assets/data/scene_datasets/replica_cad_dataset/rearrange'
plans = plan_data_from_file(rearrange/f'task_plans/set_table/{task}/{split}/013_apple.json')
env = gym.make(f'{task.capitalize()}SubtaskTrain-v0', num_envs=1, robot_uids='fetch', obs_mode='rgbdp',
    control_mode='pd_joint_delta_pos', render_mode='rgb_array', reward_mode='dense',
    sensor_configs={'shader_pack':'minimal'}, human_render_camera_configs={'shader_pack':'minimal'},
    viewer_camera_configs={'shader_pack':'minimal'}, sim_backend='gpu', max_episode_steps=200,
    task_plans=plans.plans, scene_builder_cls=plans.dataset,
    spawn_data_fp=rearrange/f'spawn_data/set_table/{task}/{split}/spawn_data.pt',
    require_build_configs_repeated_equally_across_envs=False)
writer = None
dataset = None
report = {'seed':seed, 'status':'started', 'success':False, 'steps':0, 'api_calls':0,
          'training_updates':0, 'historical_pairing':'observation-only, not full state',
          'task':f'set_table/{task}/013_apple', 'split':split, 'max_steps':200,
          'policy':'official per-object SAC, deterministic, stationary_head=true'}
try:
    reference=torch.load(args.reference_state,map_location='cpu',weights_only=False)
    saved=reference['simulator']
    obs,info=env.reset(seed=seed,options=dict(reconfigure=True,build_config_idxs=jsonable(saved['build_config_idxs']),task_plan_idxs=jsonable(saved['task_plan_idxs'])))
    u=env.unwrapped
    def device(x):
        if isinstance(x,torch.Tensor):return x.to(u.device)
        if isinstance(x,dict):return {k:device(v) for k,v in x.items()}
        return x
    u.set_state_dict(device(saved));u.agent.controller.set_state(device(reference['controller']))
    info=device(reference['reset_info']);obs=u.get_obs(info=info)
    from bvi.native_pick_audit import state_max_errors
    error=state_max_errors(jsonable(saved),jsonable(u.get_state_dict()))
    if max(error.values(),default=0)>1e-5:raise ValueError('Teacher initial restore mismatch')
    report['initial_restore_max_error']=max(error.values(),default=0)
    from bvi.ia_call_predicates import distance
    def snapshot(family):
        path=out/(family+'-state.pt')
        if path.exists():return
        torch.save(dict(simulator=u.get_state_dict(),controller=u.agent.controller.get_state(),reset_info=info),path)
    snapshot('reach')
    streak=0
    state = {'simulator':u.get_state_dict(), 'controller':u.agent.controller.get_state(),
             'reset_info':info, 'python_rng':random.getstate(), 'numpy_rng':np.random.get_state(),
             'torch_rng':torch.get_rng_state()}
    torch.save(state, out/'initial-state.pt')
    (out/'initial-state.json').write_text(json.dumps(jsonable(state),indent=2))
    report['historical_pairing']='saved native validation state restored; SAC creates diagnostic call starts'
    def record_observation(index, observation):
        if dataset is None:return
        def write(group,data):
            for key,value in data.items():
                if isinstance(value,dict):write(group.create_group(key),value)
                elif hasattr(value,'cpu'):group.create_dataset(key,data=value.cpu().numpy(),compression='gzip')
        write(dataset.create_group(f'observations/{index:04d}'),observation)
    record_observation(0,obs)
    frames={cam:deque(maxlen=3) for cam in ['fetch_head','fetch_hand']}
    def encode(observation, first=False):
        pixels={}
        for cam,queue in frames.items():
            depth=observation['sensor_data'][cam]['depth'].permute(0,3,1,2)
            for _ in range(3 if first else 1):queue.append(depth)
            # Official evaluate.py applies to_tensor(dtype="float") before actor.
            pixels[cam+'_depth']=torch.cat(list(queue),dim=1).to(device='cuda',dtype=torch.float32).contiguous()
        # Official e9ff3d2 _get_obs_extra contains exactly these four fields.
        # AC-DiT fork adds nine base velocity/position coordinates for its own model.
        official_extra={k:observation['extra'][k] for k in
                        ['tcp_pose_wrt_base','obj_pose_wrt_base','goal_pos_wrt_base','is_grasped']}
        state=torch.cat([flatten_state_dict(observation['agent'],use_torch=True),
                         flatten_state_dict(official_extra,use_torch=True)],dim=1).to('cuda')
        assert state.shape==(1,42), state.shape
        return pixels,state
    pixels,state=encode(obs,True)
    ck=root/f'checkpoints/mshab/rl/set_table/{task}/013_apple';cfg=yaml.safe_load((ck/'config.yml').read_text())['algo']
    keys=['actor_hidden_dims','critic_hidden_dims','critic_layer_norm','critic_dropout',
          'encoder_pixels_feature_dim','encoder_state_feature_dim','cnn_features','cnn_filters','cnn_strides','cnn_padding']
    policy=Agent(spaces.Dict({k:spaces.Box(0,32767,tuple(v.shape[1:]),np.int16) for k,v in pixels.items()}),
        tuple(state.shape[1:]),(13,),**{k:cfg[k] for k in keys},
        log_std_min=cfg['actor_log_std_min'],log_std_max=cfg['actor_log_std_max'],device='cuda')
    policy.load_state_dict(torch.load(ck/'policy.pt',map_location='cuda',weights_only=False)['agent'],strict=True)
    policy.to('cuda').eval()
    report['checkpoint_sha256']=hashlib.sha256((ck/'policy.pt').read_bytes()).hexdigest()
    report['state_shape']=list(state.shape)
    def frame():return np.concatenate([obs['sensor_data'][c]['rgb'][0].cpu().numpy() for c in frames],axis=1)
    recording=out/f'sac-reference-seed{seed}-incomplete.mp4'
    writer=imageio.get_writer(recording,fps=20,codec='libx264',quality=8);writer.append_data(frame())
    with (out/'events.jsonl').open('w') as log:
        for step in range(200):
            with torch.no_grad():action=policy.actor(pixels,state,compute_pi=False,compute_log_pi=False)[0].cpu()
            if not torch.isfinite(action).all():raise ValueError('nonfinite SAC action')
            action[...,8:10]=0 # Official FetchActionWrapper stationary_head.
            action=action.clamp(-1,1)
            obs,reward,terminated,truncated,info=env.step(action.to(u.device))
            held=bool(obs['extra']['is_grasped'].item())
            streak=streak+1 if held else 0
            if distance(jsonable(obs['extra']))<=.08 and not held:snapshot('grasp')
            if streak>=3:snapshot('move')
            report.update(steps=step+1,success=bool(info['success'].item()),final_info=jsonable(info))
            record_observation(step+1,obs)
            if dataset is not None:
                dataset.create_dataset(f'actions/{step:04d}',data=action.numpy())
            log.write(json.dumps(jsonable({'step':step+1,'action':action,'info':info,'qpos':u.agent.robot.qpos}))+'\n')
            writer.append_data(frame())
            if bool(terminated.item()) or bool(truncated.item()):break
            pixels,state=encode(obs)
    report['status']='completed'
except Exception as exc:
    report.update(status='error',error=repr(exc));raise
finally:
    if dataset is not None:
        dataset.attrs['native_success']=report['success']
        dataset.attrs['status']=report['status']
        dataset.close()
    if writer:
        writer.close()
        suffix='' if report['success'] else '-failed' if report['status']=='completed' else '-incomplete'
        final=out/f'sac-reference-seed{seed}{suffix}.mp4';recording.rename(final)
        report.update(video=final.name,video_sha256=hashlib.sha256(final.read_bytes()).hexdigest())
    env.close()
    (out/'result.json').write_text(json.dumps(report,indent=2))
