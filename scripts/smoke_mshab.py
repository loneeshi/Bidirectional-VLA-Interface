"""G1: real MS-HAB SequentialTask reset/step and RGB video, no learned policy."""
import argparse
import json
import os
from pathlib import Path
import time

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', type=Path, default=Path('runs/g1'))
    args = p.parse_args()
    if args.steps < 1: p.error('--steps must be positive')
    import torch
    import numpy as np
    import imageio.v2 as imageio
    from mani_skill import ASSET_DIR
    from mshab.envs.make import EnvConfig, make_env
    args.output.mkdir(parents=True, exist_ok=False)
    plan = ASSET_DIR/'scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json'
    assert plan.is_file(), f'Missing {plan}; run download_assets.py'
    cfg = EnvConfig(env_id='SequentialTask-v0', num_envs=1,
                    max_episode_steps=max(7000,args.steps+1), task_plan_fp=str(plan),
                    obs_mode='rgbd', render_mode='rgb_array', record_video=False,
                    env_kwargs={'require_build_configs_repeated_equally_across_envs':False,
                                'task_cfgs':{'navigate':{'ignore_arm_checkers':True}},
                                'human_render_camera_configs':{'width':512,'height':512}})
    env = make_env(cfg)
    def shape(x):
        if isinstance(x,dict): return {k:shape(v) for k,v in x.items()}
        return {'shape':list(x.shape),'dtype':str(x.dtype)} if hasattr(x,'shape') else str(x)
    try:
        obs, info = env.reset(seed=args.seed)
        u = env.unwrapped
        assert u.single_action_space.shape == (13,)
        metadata={'stage':'G1','benchmark_result':False,'task':'tidy_house','split':'val',
                  'seed':args.seed,'steps':args.steps,'observation':shape(obs),
                  'action_space':str(u.single_action_space),'control_mode':u.control_mode,
                  'control_freq':u.control_freq,'sim_config':str(u.sim_config),
                  'build_config_idxs':[int(x) for x in u.build_config_idxs],
                  'task_plan_idxs':[int(x) for x in u.task_plan_idxs],
                  'info_keys':list(info)}
        t0=time.monotonic()
        with imageio.get_writer(args.output/'scene-smoke.mp4',fps=20) as writer:
            for i in range(args.steps):
                action=torch.zeros((1,13),device=u.device)
                obs, reward, term, trunc, info=env.step(action)
                assert torch.isfinite(reward).all()
                frame=u.render().cpu().numpy()
                if frame.ndim==4: frame=frame[0]
                assert frame.ndim==3 and np.isfinite(frame).all()
                if i==0:
                    assert frame.std()>1, 'blank render'
                    imageio.imwrite(args.output/'scene-smoke.png',frame)
                writer.append_data(frame)
        metadata['wall_seconds']=time.monotonic()-t0
        metadata['passed']=True
        (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata,indent=2))
        print('G1_REAL_SCENE_OK')
    finally:
        env.close()
if __name__=='__main__': main()
