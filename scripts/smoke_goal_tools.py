"""No-API native compatibility gate on train seed100, never a validation score."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import numpy as np
import torch
from mani_skill import ASSET_DIR
from mshab.envs.make import EnvConfig
import bvi.nav_camera_env
from bvi.logging import JsonlLogger
from bvi.mshab_adapter import make_mshab_adapter,jsonable,scalar
from bvi.goal_tools import GoalToolAdapter,GoalRLSkill
from bvi.protocol import SkillRequest,Requirement
from bvi.runtime import SerialRuntime

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
p.add_argument('--checkpoint-root',type=Path,required=True)
p.add_argument('--bridge-dir',type=Path)
p.add_argument('--authorization-id')
a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
random.seed(100);np.random.seed(100);torch.manual_seed(100)
torch.backends.cudnn.deterministic=True
log=JsonlLogger(a.output/'events.jsonl','goal-tools-smoke')
cfg=EnvConfig(env_id='BVISequentialWorkspaceCamera-v0',num_envs=1,max_episode_steps=7000,
    task_plan_fp=str(ASSET_DIR/'scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/train/all.json'),
    obs_mode='rgbd',render_mode='rgb_array',record_video=False,frame_stack=3,
    env_kwargs=dict(require_build_configs_repeated_equally_across_envs=False,add_event_tracker_info=True,
                    task_cfgs={'navigate':{'ignore_arm_checkers':True}}))
base=make_mshab_adapter(cfg,log,a.output,seed=100)
try:
    adapter=GoalToolAdapter(base);obs=adapter.observe()
    state_before=jsonable(base.uenv.get_state_dict())
    original=base.observe().policy['state']
    matching=adapter.conditioned_policy(obs,0).policy['state']
    assert torch.allclose(original,matching,atol=1e-6,rtol=0), (original,matching)
    states=[adapter.conditioned_policy(obs,i).policy['state'] for i in (0,4,8)]
    assert any(not torch.equal(states[0],s) for s in states[1:])
    for i in range(20):adapter.predicate(i)
    assert jsonable(base.uenv.get_state_dict())==state_before,'Observation/predicate changed native state'
    skills={name:GoalRLSkill(name,adapter,a.checkpoint_root,'rl_per_obj') for name in ('navigate','pick','place')}
    runtime=SerialRuntime(adapter,skills,adapter.skill_specs(),log)
    # Execute a different object's navigation: actual target must NOT be pointer0.
    g=adapter.catalog.goals[1]
    req=SkillRequest('other-object','navigate',g['object_id'],obs.frame_id,(Requirement('done','benchmark_success'),),1,180)
    result=runtime.execute(req)
    assert result.steps==1 and not (result.feedback.reason or '').startswith('adapter_error'),result
    assert skills['navigate'].selected_index==4
    from bvi.protocol import SkillStatus
    for call in range(13):
        o=adapter.observe()
        r=SkillRequest(f'nav-success-{call}','navigate',adapter.catalog.goals[0]['object_id'],
            o.frame_id,(Requirement('done','benchmark_success'),),40,180)
        nav_result=runtime.execute(r)
        if nav_result.feedback.status is SkillStatus.SUCCEEDED:break
        assert nav_result.feedback.status is SkillStatus.TIMED_OUT,nav_result
    assert nav_result.feedback.status is SkillStatus.SUCCEEDED,'Native success-return gate not reached'
    o=adapter.observe()
    r=SkillRequest('handoff-pick','pick',adapter.catalog.goals[0]['object_id'],o.frame_id,
        (Requirement('done','benchmark_success'),),1,180)
    pick_result=runtime.execute(r)
    assert pick_result.steps==1 and not (pick_result.feedback.reason or '').startswith('adapter_error'),pick_result
    # Check both object-specific manipulation actors load and emit finite actions.
    for name,target in [('pick',g['object_id']),('place',g['destination_id'])]:
        o=adapter.observe();r=SkillRequest(name,name,target,o.frame_id,(Requirement('done','benchmark_success'),),1,180)
        skills[name].start(r,o);assert len(skills[name].act(o))==13
    report=dict(passed=True,train_seed=100,api_requests=0,physical_steps=base.steps,
        navigation_success_return=True,pick_handoff=True,
        parity_max_abs=float((original-matching).abs().max()),different_target_index=4,
        pointer_after=int(scalar(base.uenv.subtask_pointer)),state_nonmutating=True)
    if a.bridge_dir:
        from bvi.bridge import FileBridgeTransport
        from bvi.coordinator import VLMCoordinator,APIBudget
        from dataclasses import replace
        specs={k:replace(v,max_steps=40) for k,v in adapter.skill_specs().items()}
        transport=FileBridgeTransport(a.bridge_dir,'openai','gpt-5.6-luna',
            a.authorization_id,timeout_seconds=120,
            image_detail='low',reasoning_effort='none')
        coordinator=VLMCoordinator(transport,specs,log,
            APIBudget(a.authorization_id,1,2048,.00125,.00125,512000))
        o=adapter.observe()
        o=replace(o,images=tuple(i for i in o.images if i.camera in ('fetch_nav','fetch_workspace')))
        selected=coordinator.decide(o)
        executed=runtime.execute(selected)
        assert executed.steps > 0 and not (executed.feedback.reason or '').startswith('adapter_error'),executed
        report.update(api_requests=1,real_bridge_passed=True,physical_steps=base.steps,
                      selected_skill=selected.skill)
    (a.output/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
finally:base.close()
