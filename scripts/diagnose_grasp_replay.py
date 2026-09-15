"""Recorded-action contact interventions. Never a live VLA evaluation.

Rebuild every prefix from native reset; no forced task transitions or teleports.
Teacher action substitutions use recorded controls, not a teacher queried at the
learner state. Replay drift and equal-prefix checks must precede causal claims.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np

def load(path):return [json.loads(x) for x in (path/'events.jsonl').read_text().splitlines()]

def diagnostic(env,obj,jsonable):
    agent=env.agent
    l=env.scene.get_pairwise_contact_forces(agent.finger1_link,obj)
    r=env.scene.get_pairwise_contact_forces(agent.finger2_link,obj)
    return dict(qpos=jsonable(agent.robot.qpos),qvel=jsonable(agent.robot.qvel),
        tcp_pose=jsonable(agent.tcp_pose.raw_pose),object_pose=jsonable(obj.pose.raw_pose),
        object_in_tcp=jsonable((agent.tcp_pose.inv()*obj.pose).raw_pose),
        left_finger_pose=jsonable(agent.finger1_link.pose.raw_pose),
        right_finger_pose=jsonable(agent.finger2_link.pose.raw_pose),
        left_force=jsonable(l),right_force=jsonable(r),
        left_inward=jsonable(-agent.finger1_link.pose.to_transformation_matrix()[...,:3,1]),
        right_inward=jsonable(agent.finger2_link.pose.to_transformation_matrix()[...,:3,1]),
        grasp30=jsonable(agent.is_grasping(obj,max_angle=30)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--learner',type=Path,required=True);p.add_argument('--teacher',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['learner','teacher','base','arm','gripper','all'],required=True)
    p.add_argument('--window-start',type=int,default=23);p.add_argument('--window-end',type=int,default=26)
    p.add_argument('--max-steps',type=int,default=75);p.add_argument('--max-wall-seconds',type=float,default=180)
    a=p.parse_args()
    if not 1<=a.window_start<=a.window_end<=45 or not 1<=a.max_steps<=100:raise ValueError('Unbounded diagnostic')
    from bvi import JsonlLogger
    from bvi.mshab_adapter import make_mshab_adapter,jsonable
    from mshab.envs.make import EnvConfig
    import bvi.nav_camera_env
    ce,te=load(a.learner),load(a.teacher)
    cr=[e for e in ce if e['event']=='mshab_step'];tr=[e for e in te if e['event']=='mshab_step']
    tp=[e for e in tr if e['subtask_before']==1]
    source=tr if a.case=='teacher' else cr
    meta=json.loads(((a.teacher if a.case=='teacher' else a.learner)/'run-metadata.json').read_text())
    cfg=EnvConfig(**meta['config']);cfg.record_video=True;cfg.info_on_video=False
    a.output.mkdir(parents=True,exist_ok=False);logger=JsonlLogger(a.output/'events.jsonl',a.output.name)
    description=dict(stage='causal_recorded_action_diagnostic',replay_only=True,mixed_teacher_collection=True,training_collection=True,benchmark_result=False,first_object_chain_success=False,live_model_predictions=0,api_requests=0,case=a.case,window=[a.window_start,a.window_end],config=jsonable(cfg),source_sha256={str(x):hashlib.sha256((x/'events.jsonl').read_bytes()).hexdigest() for x in [a.learner,a.teacher]})
    (a.output/'run-metadata.json').write_text(json.dumps(description,indent=2))
    adapter=None;started=time.monotonic();grasp=False;boundary_drift=[];pickstep=0;summary={}
    try:
        adapter=make_mshab_adapter(cfg,logger,a.output,seed=1)
        reset=next(e for e in ce if e['event']=='mshab_reset')
        if adapter.original_plan.subtasks[0].uid!=reset['task_plan']['subtasks'][0]['uid']:raise ValueError('Wrong task plan')
        obj=adapter.uenv.subtask_objs[1]
        for row in source[:a.max_steps]:
            if adapter.ended or time.monotonic()-started>a.max_wall_seconds:break
            actual=int(adapter.observe().metadata['subtask_index'])
            if actual!=row['subtask_before']:boundary_drift.append(dict(step=adapter.steps,actual=actual,source=row['subtask_before']))
            if row['subtask_before']==1:pickstep+=1
            action=np.array(row['controller_action'][0],dtype=float)
            channels=[]
            if a.case in ('base','arm','gripper','all') and row['subtask_before']==1 and a.window_start<=pickstep<=a.window_end:
                channels={'base':[11,12],'arm':list(range(7)),'gripper':[7],'all':list(range(13))}[a.case]
                action[channels]=np.array(tp[pickstep-1]['controller_action'][0])[channels]
            before=diagnostic(adapter.uenv,obj,jsonable)
            adapter.step(tuple(action))
            after=diagnostic(adapter.uenv,obj,jsonable);grasp|=bool(after['grasp30'][0])
            logger.emit('contact_diagnostic',step=adapter.steps,source_pick_step=pickstep if row['subtask_before']==1 else None,substituted_channels=channels,before=before,after=after)
            if row['subtask_before']==1 and 15<=pickstep<=35:adapter.save_observation_images()
        summary=dict(steps=adapter.steps,ever_grasped=grasp,boundary_drift=boundary_drift,wall_seconds=time.monotonic()-started,completed=True)
    except Exception as exc:
        summary=dict(completed=False,error=repr(exc),wall_seconds=time.monotonic()-started)
        raise
    finally:
        (a.output/'summary.json').write_text(json.dumps({**description,**summary},indent=2))
        if adapter is not None:adapter.close()
    print(json.dumps(summary))

if __name__=='__main__':main()
