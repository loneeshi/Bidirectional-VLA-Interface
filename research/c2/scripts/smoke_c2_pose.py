"""API-free Fetch base preparation smoke; diagnostic only, never benchmark SR."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from bvi.logging import JsonlLogger
from bvi.mshab_adapter import make_mshab_adapter, jsonable
from bvi.continuation import ContinuationGoalAdapter
from bvi.pose_goal import GoalFetchPreparation, StaticFloorValidator, GoalRepositionSkill
from bvi.protocol import SkillRequest, SkillSpec, Requirement, SkillStatus
from bvi.runtime import SerialRuntime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=100)
    p.add_argument('--held',action='store_true')
    p.add_argument('--checkpoint-root',type=Path)
    p.add_argument('--base-evidence',type=Path)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    from mani_skill import ASSET_DIR
    from mshab.envs.make import EnvConfig
    import bvi.nav_camera_env
    cfg=EnvConfig(env_id='BVISequentialWorkspaceCamera-v0',num_envs=1,max_episode_steps=1000,
        task_plan_fp=str(ASSET_DIR/'scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/train/all.json'),
        obs_mode='rgbd',render_mode='rgb_array',record_video=False,continuous_task=True,frame_stack=3,
        stationary_base=False,stationary_torso=False,stationary_head=True,
        env_kwargs={'require_build_configs_repeated_equally_across_envs':False,'add_event_tracker_info':True,
                    'task_cfgs':{'navigate':{'ignore_arm_checkers':True}}})
    log=JsonlLogger(a.output/'events.jsonl','manual-pose-smoke')
    base=None; results=[]; started=time.monotonic(); status='failed'
    try:
        base=make_mshab_adapter(cfg,log,a.output,seed=a.seed)
        adapter=ContinuationGoalAdapter(base,'C2',structured_recovery=True)
        adapter.recovery.backend=GoalFetchPreparation(adapter,StaticFloorValidator(adapter,ASSET_DIR))
        target=adapter.catalog.goals[0]['object_id']
        held_verified=False
        if a.held:
            from bvi.continuation import ContinuationGoalRLSkill
            from bvi.mshab_adapter import scalar
            if not a.checkpoint_root or not a.base_evidence:raise ValueError('Held smoke requires checkpoints and base evidence')
            base_evidence=json.loads(a.base_evidence.read_text())
            if base_evidence['status']!='passed_base_only':raise ValueError('Base smoke not passed')
            specs=adapter.skill_specs(90.)
            tools={name:ContinuationGoalRLSkill(name,adapter,a.checkpoint_root,'rl_per_obj') for name in specs}
            runtime=SerialRuntime(adapter,tools,specs,log)
            for skill in ('navigate','pick'):
                req=SkillRequest('held-'+skill,skill,target,adapter.observe().frame_id,
                    (Requirement('completion','benchmark_success'),),specs[skill].max_steps,90.)
                result=runtime.execute(req);adapter.finish_request(req,result.feedback.status is SkillStatus.SUCCEEDED)
                results.append(dict(stage='acquire_held_object',skill=skill,status=result.feedback.status.value,
                                    reason=result.feedback.reason,steps=result.steps))
                if result.feedback.status is SkillStatus.FAILED:raise RuntimeError('Low-level smoke failed')
            pick_index=adapter.catalog.bindings[('pick',target)]
            if not bool(scalar(adapter.uenv.agent.is_grasping(adapter.uenv.subtask_objs[pick_index],max_angle=30))):
                raise RuntimeError('No held object after bounded SAC; Place preparation not validated')
            target=adapter.catalog.goals[0]['destination_id']
        # Synthetic eligibility only: no claim that SAC was attempted or failed.
        log.emit('synthetic_recovery_eligibility',reason='manual_controller_test_no_SAC')
        adapter.retry_ledger.attempted.add(('place' if a.held else 'pick',target))
        spec=SkillSpec('reposition',('pose_arrived',),('pose_arrived',),200,90.)
        runtime=SerialRuntime(adapter,{'reposition':GoalRepositionSkill(adapter)}, {'reposition':spec},log)
        goals=[(-.1,0.,0.),(0.,0.,.3)] if a.held else [(.15,0.,0.),(0.,0.,.3),(0.,.15,0.)]
        for i,g in enumerate(goals):
            request=SkillRequest(f'manual-{i}','reposition',target,adapter.observe().frame_id,
                (Requirement('arrival','pose_arrived'),),200,90.,recovery_goal=dict(
                    frame='base_at_request',x_m=g[0],y_m=g[1],yaw_rad=g[2]))
            before=adapter.recovery.backend.snapshot()
            result=runtime.execute(request)
            adapter.finish_request(request,result.feedback.status is SkillStatus.SUCCEEDED)
            results.append(dict(request=jsonable(request),status=result.feedback.status.value,
                reason=result.feedback.reason,steps=result.steps,before=before,
                after=adapter.recovery.backend.snapshot()))
            print(json.dumps(results[-1]),flush=True)
        if a.held:
            held_verified=all(r['status']=='succeeded' for r in results if 'request' in r)
            status='passed' if held_verified else 'not_passed'
        else:
            status='passed_base_only' if all(r['status']=='succeeded' for r in results) else 'not_passed'
    finally:
        if base is not None:base.close()
        (a.output/'summary.json').write_text(json.dumps(dict(status=status,api_calls=0,
            physical_steps=sum(r['steps'] for r in results),wall_seconds=time.monotonic()-started,
            held_place_verified=locals().get('held_verified',False),results=results,
            controller_sources={name:hashlib.sha256((Path(__file__).resolve().parents[1]/'src/bvi'/name).read_bytes()).hexdigest()
                                for name in ('pose_goal.py','pose_recovery.py','lightnav_skill.py')}
            ),indent=2),encoding='utf-8')

if __name__=='__main__':main()
