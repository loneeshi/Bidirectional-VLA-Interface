"""Verify the pinned Pick checker/configuration and Gym clocks without physics."""
import ast
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path('/home/pshuai/bvi-research')
RUNTIME=ROOT/'src/official-mshab-runtime/mshab'


def main():
    import torch
    import gymnasium as gym
    source=(RUNTIME/'mshab/envs/sequential_task.py').read_text()
    tree=ast.parse(source)
    checker=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_pick_check_success')
    checker_source=ast.get_source_segment(source,checker)
    checker=copy.deepcopy(checker)
    for arg in checker.args.args:arg.annotation=None
    checker.returns=None
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[checker],type_ignores=[])),'official_pick_checker','exec'),namespace)
    fn=namespace['_pick_check_success']
    outcomes={}
    for horizon in (200,601):
        flags=[]
        for grasp,ee,rest,static,force in itertools.product((False,True),repeat=5):
            cfg=SimpleNamespace(horizon=horizon,ee_rest_thresh=.05,robot_resting_qpos_tolerance_grasping=.05,robot_cumulative_force_limit=5000)
            agent=SimpleNamespace(is_grasping=lambda obj,max_angle:torch.tensor([grasp]),
                tcp_pose=SimpleNamespace(p=torch.tensor([[0.,0.,0. if ee else .06]])),
                robot=SimpleNamespace(qpos=torch.zeros((1,15)) if rest else torch.ones((1,15))),
                is_static=lambda threshold,base_threshold:torch.tensor([static]))
            fake=SimpleNamespace(agent=agent,pick_cfg=cfg,ee_rest_world_pose=SimpleNamespace(p=torch.zeros((1,3))),
                resting_qpos=torch.zeros(10),robot_cumulative_force=torch.tensor([4999. if force else 5000.]),_add_event_tracker_info=False)
            success,checkers=fn(fake,None,torch.tensor([0]))
            assert bool(success[0])==all((grasp,ee,rest,static,force))
            flags.append(bool(success[0]))
        outcomes[str(horizon)]=flags
    assert outcomes['200']==outcomes['601']
    assert 'horizon' not in checker_source
    assert 'self.task_cfgs[k].update(v)' in source
    assert '].horizon' in source
    assert 'self.subtask_steps_left -= 1' in source
    assert '((self.subtask_steps_left <= 0) & ~success)' in source
    make_source=(RUNTIME/'mshab/envs/make.py').read_text()
    assert 'max_episode_steps=env_cfg.max_episode_steps' in make_source
    base_source=(RUNTIME.parent/'ManiSkill/mani_skill/envs/sapien_env.py').read_text()
    base_tree=ast.parse(base_source)
    reset=next(n for n in ast.walk(base_tree) if isinstance(n,ast.FunctionDef) and n.name=='reset')
    get_info=next(n for n in ast.walk(base_tree) if isinstance(n,ast.FunctionDef) and n.name=='get_info')
    assert 'self.get_info()' in ast.get_source_segment(base_source,reset)
    assert 'self.evaluate()' in ast.get_source_segment(base_source,get_info)
    class CounterEnv(gym.Env):
        action_space=gym.spaces.Discrete(1)
        observation_space=gym.spaces.Discrete(1)
        def reset(self,*,seed=None,options=None):return 0,{}
        def step(self,action):return 0,0.,False,False,{}
    clock_checks={}
    for horizon in (200,600):
        env=gym.wrappers.TimeLimit(CounterEnv(),max_episode_steps=horizon);env.reset()
        for step in range(1,horizon+1):
            _,_,_,truncated,_=env.step(0)
            assert truncated==(step==horizon)
        clock_checks[str(horizon)]=True
    files=['pick.py','subtask.py','sequential_task.py','make.py','planner.py']
    result={'mshab_commit':subprocess.check_output(['git','-C',str(RUNTIME),'rev-parse','HEAD'],text=True).strip(),
        'source_sha256':{f:hashlib.sha256((RUNTIME/'mshab/envs'/f).read_bytes()).hexdigest() for f in files},
        'configuration':{'max_episode_steps':600,'env_kwargs':{'task_cfgs':{'pick':{'horizon':601}},'target_randomization':False}},
        'reset_clock_offset':{'internal_horizon':601,'reset_evaluate_decrements':1,'physical_action_budget':600,
            'post_reset_counter_or_force_mutation':False,'sapien_env_sha256':hashlib.sha256(base_source.encode()).hexdigest()},
        'sac_configuration':{'max_episode_steps':200,'env_kwargs':{'task_cfgs':{'pick':{'horizon':200}},'target_randomization':False}},
        'source_route_passed':True,'actual_checker_truth_table_cases':64,'checker_identical_at_both_horizons':True,
        'gym_clock_checks':clock_checks,'official_pick_checker_source':checker_source,
        'fresh_process_per_condition_required':'task_cfgs updates class-held config objects; never construct SAC after a 600-step env in the same process',
        'runtime_physics_verified':False,'gpu_runs':0,'api_calls':0,'simulator_steps':0,'cuda_initialized':torch.cuda.is_initialized()}
    assert not result['cuda_initialized']
    assert result['mshab_commit']=='e9ff3d23496d38e4431c8d913e147ffa007f7f72'
    Path('/tmp/bvi-arm-horizon-20261001.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('official_pick_checker_source','source_sha256')}))


if __name__=='__main__':main()
