"""T0a diagnostic: pinned official BC, official wrappers vs project-style input/action path.

This is a single-env diagnostic using official wrappers, not the unmodified
upstream full-suite evaluator or a benchmark reproduction. Both paths stop on
native termination; therefore success-once is measured before native stopping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

GPU = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'
REVISION = '91e96be85128df43728a7511355c3fa999bd2c94'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--path', choices=['official', 'project'], required=True)
    p.add_argument('--environment', choices=['official','acdit'], default='official')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    root = Path.home() / 'bvi-research'
    used = int(subprocess.check_output(['nvidia-smi', '-i', GPU,
        '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).strip())
    if used > 1024:
        raise RuntimeError('GPU1 occupied; refusing to launch')
    os.environ.update(CUDA_VISIBLE_DEVICES=GPU, MS_ASSET_DIR=str(root/'assets'), OMP_NUM_THREADS='2')
    if os.environ.get('PYTHONHASHSEED') != str(a.seed):
        raise RuntimeError('PYTHONHASHSEED must be fixed before interpreter startup')
    source = root/'src/official-mshab-runtime' if a.environment=='official' else root/'src/AC-DiT/third_party'
    sys.path[:0] = [str(source/'mshab'), str(source/'ManiSkill')]
    import numpy as np
    import torch
    import gymnasium as gym
    import mshab.envs
    from mshab.agents.bc import Agent
    from mshab.envs.planner import plan_data_from_file
    from mshab.envs.wrappers import FetchDepthObservationWrapper, FrameStack, FetchActionWrapper
    from mani_skill.utils.common import flatten_state_dict
    from mshab.utils.array import to_tensor

    def plain(x):
        if isinstance(x, dict): return {str(k): plain(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)): return [plain(v) for v in x]
        if hasattr(x, 'detach'): return x.detach().cpu().numpy().tolist()
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        return x

    def digest(x):
        return hashlib.sha256(json.dumps(plain(x), sort_keys=True).encode()).hexdigest()

    def encode(raw):
        extra = {k:raw['extra'][k] for k in
                 ['tcp_pose_wrt_base','obj_pose_wrt_base','goal_pos_wrt_base','is_grasped']}
        return {'state':torch.cat([flatten_state_dict(raw['agent'],use_torch=True),
                flatten_state_dict(extra,use_torch=True)],dim=1),
                'pixels':{c+'_depth':raw['sensor_data'][c]['depth'].permute(0,3,1,2)[:,None]
                          for c in ['fetch_head','fetch_hand']}}

    a.output.mkdir(parents=True, exist_ok=False)
    report = dict(status='initializing', seed=a.seed, path=a.path, environment=a.environment, steps=0,
        success_once=False, native_success=False, fail_once=False, training_updates=0,
        api_calls=0, checkpoint_revision=REVISION, gpu_before_mib=used,
        protocol='native termination, max200; not continuous-task paper success-once',
        scope='official wrappers vs independently assembled project-style BC path; not full VLA harness',
        max_seconds=180, source_revision=subprocess.check_output(
            ['git','-C',str(source/'mshab'),'rev-parse','HEAD'],text=True).strip())
    start=time.monotonic(); env=None
    def save():
        report['wall_seconds']=time.monotonic()-start
        (a.output/'result.json').write_text(json.dumps(plain(report),indent=2)+'\n')
    save()
    try:
        torch.set_num_threads(2)
        torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
        torch.backends.cudnn.deterministic=True
        base=root/'assets/data/scene_datasets/replica_cad_dataset/rearrange'
        plan=base/'task_plans/set_table/pick/val/013_apple.json'
        plans=plan_data_from_file(plan)
        kwargs=dict(num_envs=1, robot_uids='fetch', obs_mode='depth',
            control_mode='pd_joint_delta_pos', reward_mode='normalized_dense',
            render_mode='rgb_array', shader_dir='minimal', sim_backend='cpu',
            max_episode_steps=200, task_plans=plans.plans, scene_builder_cls=plans.dataset,
            spawn_data_fp=base/'spawn_data/set_table/pick/val/spawn_data.pt',
            require_build_configs_repeated_equally_across_envs=False,
            target_randomization=False,robot_force_mult=.001,robot_force_penalty_min=.2)
        env=gym.make('PickSubtaskTrain-v0',**kwargs)
        u=env.unwrapped
        # Capture native flags before wrappers; no safety condition is changed.
        if a.path=='official':
            env=FetchDepthObservationWrapper(env,cat_state=True,cat_pixels=False)
            env=FrameStack(env,num_stack=1)
            env=FetchActionWrapper(env,stationary_head=True)
        obs,info=env.reset(seed=a.seed,options={'reconfigure':True})
        pci=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=pci.bus_id',
                                    '--format=csv,noheader'],text=True).strip()
        if str(u._render_device.pci_string).lower()[-7:]!=pci.lower()[-7:]:
            raise RuntimeError('Renderer not on designated GPU1')
        state={'simulator':u.get_state_dict(),'controller':u.agent.controller.get_state()}
        report['initial_state_sha256']=digest(state)
        report['renderer_pci']=str(u._render_device.pci_string)
        report['plan_sha256']=hashlib.sha256(plan.read_bytes()).hexdigest()
        torch.save(state,a.output/'initial-state.pt')
        model_obs=to_tensor(obs if a.path=='official' else encode(obs),device='cuda',dtype='float')
        report['initial_policy_obs_sha256']=digest(model_obs)
        report['state_shape']=list(model_obs['state'].shape)
        ck=root/'checkpoints/mshab/bc/set_table/pick/all/policy.pt'
        report['checkpoint_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest()
        policy=Agent(model_obs,(13,)).to('cuda').eval()
        policy.load_state_dict(torch.load(ck,map_location='cuda',weights_only=False)['agent'],strict=True)
        report['strict_load']=True; save()
        with (a.output/'events.jsonl').open('w') as log:
            for step in range(200):
                if time.monotonic()-start>180: raise TimeoutError('Episode time limit')
                with torch.inference_mode(): raw_action=policy(model_obs)
                if raw_action.shape!=(1,13) or not torch.isfinite(raw_action).all():
                    raise ValueError('Invalid BC action')
                action=raw_action.detach().cpu().clone()
                if a.path=='project':
                    action=action.clamp(-1,1); action[...,8:10]=0
                input_hash=digest(model_obs)
                obs,_,term,trunc,info=env.step(action)
                success=bool(info['success'].item()); fail=bool(info.get('fail',torch.tensor(False)).item())
                report.update(steps=step+1,success_once=report['success_once'] or success,
                    fail_once=report['fail_once'] or fail,native_success=success,
                    terminated=bool(term.item()),truncated=bool(trunc.item()),final_info=plain(info))
                log.write(json.dumps(plain(dict(step=step+1,input_sha256=input_hash,
                    raw_action=raw_action,submitted_action=action,info=info,
                    qpos=u.agent.robot.qpos)))+'\n'); log.flush()
                if bool(term.item()) or bool(trunc.item()): break
                model_obs=to_tensor(obs if a.path=='official' else encode(obs),device='cuda',dtype='float')
        report['status']='completed'
    except Exception as exc:
        report.update(status='infrastructure_failure',error=repr(exc)); raise
    finally:
        if env is not None: env.close()
        save()
    print(json.dumps(plain(report)))


if __name__=='__main__': main()
