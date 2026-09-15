"""Replay a recorded failed-policy prefix, then collect SAC recovery labels.

Training data only. No live VLA prediction and no C/D task-success claim.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from bvi import JsonlLogger,ProtocolError,Requirement,SerialRuntime,SkillRequest,SkillStatus
from bvi.mshab_adapter import OfficialRLSkill,make_mshab_adapter,scalar,jsonable


def prefix_rows(events,pick_steps):
    if not 0<=pick_steps<=40:raise ProtocolError('Pick prefix must be in0..40')
    steps=[e for e in events if e['event']=='mshab_step']
    nav=[e for e in steps if e['subtask_before']==0]
    pick=[e for e in steps if e['subtask_before']==1]
    if not nav or len(pick)<pick_steps:raise ProtocolError('Insufficient recorded prefix')
    return nav+pick[:pick_steps]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--pick-prefix',type=int,required=True)
    p.add_argument('--checkpoint-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=a.source_run/'events.jsonl';events=[json.loads(x) for x in source.read_text().splitlines()]
    resets=[e for e in events if e['event']=='mshab_reset']
    if len(resets)!=1:raise ProtocolError('Source must have exactly one reset')
    reset=resets[0];rows=prefix_rows(events,a.pick_prefix)
    source_meta=json.loads((a.source_run/'run-metadata.json').read_text())
    from mshab.envs.make import EnvConfig
    if source_meta['config']['env_id']=='BVISequentialNavCamera-v0':import bvi.nav_camera_env
    cfg=EnvConfig(**source_meta['config']);cfg.record_video=True;cfg.info_on_video=False
    metadata=dict(stage='training_recovery_collection',benchmark_result=False,vlm=False,
        mixed_teacher_collection=True,manipulation_policy='recorded_pi_prefix_then_sac_teacher',
        source_run=str(a.source_run),source_events_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        replay_pick_steps=a.pick_prefix,config=jsonable(cfg),seed=reset['seed'])
    (a.output/'run-metadata.json').write_text(json.dumps(metadata,indent=2))
    logger=JsonlLogger(a.output/'events.jsonl',a.output.name);adapter=None;result=None;started=time.monotonic()
    try:
        adapter=make_mshab_adapter(cfg,logger,a.output,seed=reset['seed'])
        if adapter.original_plan.subtasks[0].uid!=reset['task_plan']['subtasks'][0]['uid']:
            raise ProtocolError('Source task plan differs from replay environment')
        adapter.save_observation_images()
        for actual in (a.output/'frames').glob(f'seed-{reset["seed"]}-step-0-*.png'):
            expected=a.source_run/'frames'/actual.name
            if not expected.is_file() or hashlib.sha256(actual.read_bytes()).digest()!=hashlib.sha256(expected.read_bytes()).digest():
                raise ProtocolError('Initial rendered observation differs from source')
        adapter.record_demonstrations=False
        for e in rows:
            if time.monotonic()-started>180 or adapter.ended:raise ProtocolError('Replay prefix exceeded its limit')
            if adapter.observe().metadata['subtask_index']!=e['subtask_before']:
                raise ProtocolError('Replay subtask differs from source')
            logger.emit('recorded_prefix_action',source_frame=e['frame_id'],evaluation_eligible=False)
            adapter.step(e['controller_action'][0])
        if adapter.ended or adapter.observe().metadata['subtask_index']!=1:
            raise ProtocolError('Recovery must start during active Pick')
        adapter.record_demonstrations=True;adapter.demonstration_source='official_sac_recovery'
        obs=adapter.observe();logger.emit('recovery_teacher_takeover',frame_id=obs.frame_id,
            learner_steps=a.pick_prefix,teacher='official_sac',evaluation_eligible=False)
        specs=adapter.skill_specs(180)
        skills={n:OfficialRLSkill(n,adapter,a.checkpoint_root,'rl_per_obj') for n in specs}
        allowed=obs.allowed_calls[0]
        remaining_wall=min(180,280-(time.monotonic()-started))
        if remaining_wall<=0:raise ProtocolError('No remaining recovery time')
        request=SkillRequest('teacher-pick','pick',allowed.target_id,obs.frame_id,
            (Requirement('benchmark-completion','benchmark_success'),),
            min(specs['pick'].max_steps,int(scalar(adapter.last_info['subtasks_steps_left']))),remaining_wall)
        result=SerialRuntime(adapter,skills,specs,logger).execute(request,obs)
    finally:
        if adapter is not None:
            summary=dict(benchmark_result=False,first_object_chain_success=False,
                mixed_teacher_collection=True,replay_only_prefix=True,api_requests=0,live_vla_predictions=0,
                recovery_pick_success=result is not None and result.feedback.status is SkillStatus.SUCCEEDED,
                recovery_result=None,
                steps=adapter.steps,wall_seconds=time.monotonic()-started)
            # Avoid serializing terminal RGB/policy tensors from SkillResult.
            if result is not None:summary['recovery_result']={'steps':result.steps,'feedback':jsonable(result.feedback)}
            (a.output/'summary.json').write_text(json.dumps(summary,indent=2));logger.emit('collection_finished',**summary)
            adapter.close();print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
