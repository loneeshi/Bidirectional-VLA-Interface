"""Run pinned upstream MS-HAB evaluator with passive step tracing (oracle dispatcher)."""
import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path
import time

def jsonable(x):
    if isinstance(x,dict): return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [jsonable(v) for v in x]
    if hasattr(x,'detach'): return x.detach().cpu().tolist()
    if hasattr(x,'tolist'): return x.tolist()
    if dataclasses.is_dataclass(x): return jsonable(dataclasses.asdict(x))
    if isinstance(x,(str,int,float,bool)) or x is None: return x
    return str(x)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--max-steps',type=int,default=7000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--episodes',type=int,default=1)
    p.add_argument('--policy-type',choices=['rl_all_obj','rl_per_obj'],default='rl_all_obj')
    p.add_argument('--checkpoint-root',type=Path,default=os.getenv('MSHAB_CHECKPOINT_DIR'))
    p.add_argument('--output',type=Path,default=Path('runs/g3'))
    args=p.parse_args()
    if args.max_steps<1 or args.episodes<1: p.error('positive steps/episodes required')
    if args.checkpoint_root is None: p.error('set --checkpoint-root or MSHAB_CHECKPOINT_DIR')
    root=args.checkpoint_root.resolve()
    assert (root/'rl/tidy_house/navigate/all/policy.pt').is_file(), root
    args.output=args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    os.symlink(root,args.output/'mshab_checkpoints',target_is_directory=True)
    os.chdir(args.output)
    import torch
    from mani_skill import ASSET_DIR
    import mshab.evaluate as official
    from mshab.envs.make import EnvConfig
    from mshab.utils.logger import LoggerConfig
    trace=(args.output/'steps.jsonl').open('w',buffering=1)
    original_make=official.make_env
    class TracedEnv:
        def __init__(self,env): self.env=env; self.steps=0
        def __getattr__(self,name): return getattr(self.env,name)
        def reset(self,*a,**kw):
            result=self.env.reset(*a,**kw)
            u=self.env.unwrapped
            trace.write(json.dumps({'event':'reset','seed':args.seed,
                'build_config_idxs':jsonable(u.build_config_idxs),
                'task_plan_idxs':jsonable(u.task_plan_idxs),
                'task_plan':jsonable(u.task_plan),'info':jsonable(result[1])})+'\n')
            return result
        def step(self,action):
            u=self.env.unwrapped
            before=u.subtask_pointer.clone()
            raw=action.clone()
            if raw.shape[-1]!=13 or not torch.isfinite(raw).all():
                trace.write(json.dumps({'event':'invalid_action','step':self.steps})+'\n')
                raise ValueError('nonfinite or wrong dimension action')
            result=self.env.step(action)
            trace.write(json.dumps({'event':'step','step':self.steps,
                'subtask_before':jsonable(before),'subtask_after':jsonable(u.subtask_pointer),
                'raw_action':jsonable(raw),'raw_outside_unit_box':bool((raw.abs()>1).any()),
                'terminated':jsonable(result[2]),'truncated':jsonable(result[3]),
                'info':jsonable(result[4])})+'\n')
            self.steps+=1
            return result
    official.make_env=lambda *a,**kw: TracedEnv(original_make(*a,**kw))
    plan=ASSET_DIR/'scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json'
    cfg=official.EvalConfig(seed=args.seed,task='tidy_house',policy_type=args.policy_type,
        max_trajectories=args.episodes,
        eval_env=EnvConfig(env_id='SequentialTask-v0',num_envs=1,
            max_episode_steps=args.max_steps,task_plan_fp=str(plan),
            obs_mode='depth',render_mode='rgb_array',record_video=True,info_on_video=True,
            env_kwargs={'require_build_configs_repeated_equally_across_envs':False,
                        'add_event_tracker_info':True,
                        'human_render_camera_configs':{'width':512,'height':512},
                        'task_cfgs':{'navigate':{'ignore_arm_checkers':True}}}),
        logger=LoggerConfig(workspace=str(args.output),exp_name='official',clear_out=False,
                            tensorboard=False,wandb=False))
    (args.output/'run-metadata.json').write_text(json.dumps({
        'dispatcher':'official_oracle','vlm':False,'benchmark_result':False,
        'coverage':'single validation scene sampled plans; not full validation set',
        'config':jsonable(cfg)},indent=2))
    saved_argv=sys.argv
    sys.argv=[sys.argv[0]]  # upstream parse_cfg must not reinterpret this wrapper's flags
    try: official.eval(cfg)
    finally:
        trace.close()
        sys.argv=saved_argv
if __name__=='__main__': main()
