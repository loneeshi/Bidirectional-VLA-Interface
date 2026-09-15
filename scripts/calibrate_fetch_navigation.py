"""Bounded physical sign and grasp-hold checks; no learned navigation calls."""
import argparse
import json
import math
from pathlib import Path

from bvi import JsonlLogger, Requirement, SerialRuntime, SkillRequest, SkillStatus
from bvi.lightnav_skill import FetchNavigationControl, wrap
from bvi.mshab_adapter import OfficialRLSkill, make_mshab_adapter, jsonable, scalar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    args = parser.parse_args()
    from mani_skill import ASSET_DIR
    from mshab.envs.make import EnvConfig
    args.output.mkdir(parents=True, exist_ok=False)
    logger = JsonlLogger(args.output/'events.jsonl', args.output.name)
    cfg = EnvConfig(env_id='SequentialTask-v0', num_envs=1, max_episode_steps=7000,
        task_plan_fp=str(ASSET_DIR/'scene_datasets/replica_cad_dataset/rearrange/task_plans/tidy_house/sequential/val/all.json'),
        obs_mode='rgbd', render_mode='rgb_array', record_video=True,
        info_on_video=False, continuous_task=True, frame_stack=3,
        stationary_base=False, stationary_torso=False, stationary_head=True,
        env_kwargs={'require_build_configs_repeated_equally_across_envs':False,
          'invisible_goals_in_human_render':True,
          'add_event_tracker_info':True,
          'human_render_camera_configs':{'width':512,'height':512},
          'task_cfgs':{'navigate':{'ignore_arm_checkers':True}}})
    adapter = make_mshab_adapter(cfg, logger, args.output, seed=1)
    report = {'benchmark_result':False, 'checks':[]}
    try:
        control = FetchNavigationControl(adapter)
        control.capture_hold()
        for label, linear, angular, steps in [('zero',0,0,10),('forward',.15,0,20),('yaw',0,.3,20)]:
            before = control.pose()
            for _ in range(steps):
                adapter.step(control.action(linear,angular))
            after = control.pose()
            forward = (after[0]-before[0])*math.cos(before[2])+(after[1]-before[1])*math.sin(before[2])
            yaw = wrap(after[2]-before[2])
            passed = (math.hypot(after[0]-before[0],after[1]-before[1]) < .02 if label=='zero'
                      else forward > .03 if label=='forward' else yaw > .05)
            report['checks'].append(dict(check=label,before=before,after=after,forward_m=forward,yaw_rad=yaw,passed=passed))
            adapter.save_observation_images()
            assert passed, f'{label} calibration failed'
        # New episode explicitly separates sign calibration from grasp-hold check.
        adapter.reset(1)
        specs = adapter.skill_specs(180)
        runtime = SerialRuntime(adapter,{s:OfficialRLSkill(s,adapter,args.checkpoint_root,'rl_per_obj') for s in specs},specs,logger)
        for index in range(2):
            obs=adapter.observe(); call=obs.allowed_calls[0]; spec=specs[call.skill]
            req=SkillRequest(f'calibration-official-{index}',call.skill,call.target_id,obs.frame_id,
                (Requirement('completion','benchmark_success'),),spec.max_steps,180)
            result=runtime.execute(req)
            report['checks'].append(dict(check=call.skill,steps=result.steps,feedback=jsonable(result.feedback)))
            assert result.feedback.status is SkillStatus.SUCCEEDED, result.feedback
        control.capture_hold()
        report['before_hold_info']=adapter.last_info
        grasp_checks=[]
        for _ in range(40):
            adapter.step(control.action(0,.15))
            grasp_checks.append(bool(scalar(adapter.last_info['is_grasped'])))
        report['after_hold_info']=adapter.last_info
        report['hold_grasp_checks']=grasp_checks
        report['hold_targets']=control.hold
        adapter.save_observation_images()
        assert all(grasp_checks), 'Grasp was lost during navigation hold'
    finally:
        (args.output/'calibration.json').write_text(json.dumps(report,indent=2))
        adapter.close()
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
