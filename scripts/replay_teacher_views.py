"""Re-render recorded teacher controls with an additional egocentric RGB sensor.

Training data only. Each replay starts at the native reset, executes every prefix
action, and records current state/images. No replay is counted as VLA success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-wall-seconds',type=float,default=180)
    p.add_argument('--max-steps',type=int,default=600)
    p.add_argument('--max-boundary-holds',type=int,default=10)
    a=p.parse_args()
    from bvi import JsonlLogger
    from bvi.mshab_adapter import make_mshab_adapter,jsonable
    from mshab.envs.make import EnvConfig
    import bvi.nav_camera_env
    raw=(a.source/'events.jsonl').read_bytes()
    events=[json.loads(x) for x in raw.decode().splitlines()]
    demos=[e for e in events if e['event']=='demonstration_step' and e['skill'] in ('pick','place')]
    successful={i for i in {e['subtask_index'] for e in demos}
                if [e for e in demos if e['subtask_index']==i][-1]['feedback']['adapter_subtask_after']>i}
    selected={e['frame_id']:e for e in demos if e['subtask_index'] in successful}
    if not selected:raise ValueError('Source has no successful teacher segment')
    last=max(int(e['next_frame_id'].rsplit('-',1)[1]) for e in selected.values())
    if last>a.max_steps:raise ValueError('Source exceeds declared step limit')
    rows=[e for e in events if e['event']=='mshab_step'][:last]
    source_meta=json.loads((a.source/'run-metadata.json').read_text())
    cfg=EnvConfig(**source_meta['config'])
    cfg.env_id='BVISequentialWorkspaceCamera-v0';cfg.record_video=False;cfg.info_on_video=False
    a.output.mkdir(parents=True,exist_ok=False)
    meta=dict(stage='teacher_sensor_rerender',replay_only=True,replayed_teacher=True,
        benchmark_result=False,first_object_chain_success=False,vlm=False,
        mixed_teacher_collection=True,live_model_predictions=0,api_requests=0,
        source_run=str(a.source),source_events_sha256=hashlib.sha256(raw).hexdigest(),
        config=jsonable(cfg),seed=source_meta.get('seed',1),replayed_skill_origins={})
    logger=JsonlLogger(a.output/'events.jsonl',a.output.name)
    started=time.monotonic();adapter=None;frames=0;max_drift=0.;completed=set();holds=0
    try:
        adapter=make_mshab_adapter(cfg,logger,a.output,seed=meta['seed'])
        for row in rows:
            if time.monotonic()-started>a.max_wall_seconds:raise TimeoutError('Replay wall limit')
            if adapter.ended:raise RuntimeError('Environment ended before the source teacher segment')
            obs=adapter.observe();index=int(obs.metadata['subtask_index'])
            if index!=row['subtask_before']:raise RuntimeError('Replay task pointer differs from source')
            qpos=np.asarray(jsonable(adapter.uenv.agent.robot.qpos)[0])
            meta['replayed_skill_origins'].setdefault(str(index),qpos[:2].tolist())
            source_frame=f"seed-{meta['seed']}-step-{int(row['frame_id'].rsplit('-',1)[1])-1}"
            source_row=selected.get(source_frame)
            adapter.record_demonstrations=source_row is not None
            if source_row is not None:
                max_drift=max(max_drift,float(np.max(np.abs(qpos-np.asarray(source_row['qpos'][0])))))
                frames+=1
            transition=adapter.step(row['controller_action'][0])
            # GPU physics can reach a boundary one or more steps later than the
            # source. Hold only the last recorded control, with explicit caps;
            # never change state, force completion, or query a teacher model.
            boundary_holds=0
            while row['subtask_after']>index and transition.info['adapter_subtask_after']==index:
                if (boundary_holds>=a.max_boundary_holds or adapter.ended or adapter.steps>=a.max_steps
                        or time.monotonic()-started>a.max_wall_seconds):break
                logger.emit('recorded_boundary_action_hold',source_frame=row['frame_id'],
                            subtask_index=index,hold=boundary_holds+1)
                transition=adapter.step(row['controller_action'][0])
                boundary_holds+=1;holds+=1
                if source_row is not None:frames+=1
            if source_row is not None and transition.info['adapter_subtask_after']>index:completed.add(index)
        if completed!=successful:raise RuntimeError('Replayed teacher did not complete every selected segment')
        summary={**meta,'steps':adapter.steps,'frames':frames,'completed_teacher_segments':sorted(completed),
            'max_qpos_difference_from_source':max_drift,'wall_seconds':time.monotonic()-started,
            'training_replay_complete':True,'recorded_boundary_action_holds':holds}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        logger.emit('experiment_finished',**summary)
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        summary={**meta,'training_replay_complete':False,'error':str(exc),
            'steps':None if adapter is None else adapter.steps,'frames':frames,
            'completed_teacher_segments':sorted(completed),'recorded_boundary_action_holds':holds,
            'max_qpos_difference_from_source':max_drift,'wall_seconds':time.monotonic()-started}
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
        logger.emit('experiment_failed',**summary)
        raise
    finally:
        (a.output/'run-metadata.json').write_text(json.dumps(meta,indent=2))
        if adapter is not None:adapter.close()


if __name__=='__main__':main()
