"""Authorized lab worker: official spawn, one condition, no retries or tuning."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import time

from eef_arm_server_broker import validate_scope,save

GPU='GPU-b7ebba23-7824-7601-df32-be55628936c3'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind_initial(path, binding):
    """All four conditions must start from the same frozen physical state."""
    import numpy as np
    if path.exists():
        prior=json.loads(path.read_text())
        for key in ('uid','spawn_index','seed'):
            if prior[key]!=binding[key]:raise ValueError('paired spawn metadata mismatch')
        for key in ('qpos','base_world','target_world'):
            if not np.allclose(prior[key],binding[key],rtol=0,atol=1e-6):
                raise ValueError('paired physical initial state mismatch: '+key)
    else:save(path,binding)


def validate(auth,args):
    validate_scope(auth)
    if (os.name!='posix' or os.environ.get('CUDA_VISIBLE_DEVICES')!=GPU
        or args.case_id not in auth['case_ids'] or args.condition not in ('V','SAC','script')
        or not auth.get('scheduling_clearance_confirmed')):
        raise ValueError('lab worker authorization/device/case gate')
    if sha(args.roster)!=auth['roster_sha256']:raise ValueError('roster changed')
    if not args.output.resolve().is_relative_to(Path(auth['output_root']).resolve()):
        raise ValueError('output outside approved root')
    for name,digest in auth['source_sha256'].items():
        if sha(args.code/name)!=digest:raise ValueError('source changed: '+name)


def run(args):
    started=time.monotonic();auth=json.loads(args.authorization.read_text());validate(auth,args)
    roster=json.loads(args.roster.read_text());row=next(r for r in roster['dev'] if r['case_id']==args.case_id)
    args.output.mkdir(parents=True,exist_ok=False)
    report={'case_id':args.case_id,'position':int(args.case_id.rsplit('-',1)[1]),'condition':args.condition,
            'status':'running','recorded_at_utc':datetime.now(timezone.utc).isoformat(),
            'strict_pick_success':False,'frames':[],'commands':[],'simulator_actions':0,'api_requests':0}
    cap={'V':1800,'SAC':300,'script':600}[args.condition]
    env=None;recording=None;ex=None
    def deadline():
        if time.monotonic()-started>=cap-10:raise TimeoutError('authorized worker wall limit')
    try:
        from arm_runtime_paths import configure_arm_runtime
        configure_arm_runtime(args.lab_root)
        import numpy as np
        import torch
        from dacite import from_dict
        from omegaconf import OmegaConf
        from mshab.envs.make import EnvConfig,make_env
        from bvi.mshab_adapter import jsonable,scalar,load_rl_policy
        from bvi.eef_tools import RobotFK,validate_official_controller
        from bvi.eef_arm_executor import ArmCoordinatedV2
        from bvi.eef_arm_contract import dispatch
        from bvi.eef_arm_reference import run_reference
        from bvi.eef_arm_scoring import tcp_box_distance,funnel,horizon_results
        from bvi.eef_arm_sensing import observe
        from bvi.eef_compact_tools import ObservationDepth
        from bvi.eef_visual_history import state_snapshot
        from bvi.offline_spatial_wire import target_semantics
        from bvi.eef_g0 import strict_pick
        from bvi.eef_attempt_recording import AttemptRecording
        from bvi.logging import JsonlLogger
        from types import SimpleNamespace
        from mani_skill.utils.geometry.trimesh_utils import get_component_meshes
        from export_c2_collision_bundle import _component_and_shapes
        import subprocess
        runtime=args.lab_root/'src/official-mshab-runtime/mshab'
        if subprocess.check_output(['git','-C',str(runtime),'rev-parse','HEAD'],text=True).strip()!='e9ff3d23496d38e4431c8d913e147ffa007f7f72':raise ValueError('runtime pin')
        for name,digest in auth['asset_sha256'].items():
            if sha(args.lab_root/name)!=digest:raise ValueError('frozen asset changed: '+name)
        data=args.lab_root/'assets/data/scene_datasets/replica_cad_dataset/rearrange'
        plan_path=data/f"task_plans/tidy_house/pick/val/{row['object_category']}.json"
        source=json.loads(plan_path.read_text());plans=[p for p in source['plans'] if p['subtasks'][0]['composite_subtask_uids'][0]==row['uid']]
        if len(plans)!=1:raise ValueError('spawn UID ambiguity')
        save(args.output/'single-plan.json',{'dataset':source['dataset'],'plans':plans})
        # Actual checkpoint location is frozen by the staging operator, not guessed.
        checkpoint=Path(auth['checkpoint_root'])/'rl/tidy_house/pick'/row['object_category']
        raw=OmegaConf.to_container(OmegaConf.load(checkpoint/'config.yml'),resolve=True)
        cfg=dict(raw['eval_env']);horizon=200 if args.condition=='SAC' else 600
        cfg.update(num_envs=1,max_episode_steps=horizon,continuous_task=True,record_video=False,
                   task_plan_fp=str(args.output/'single-plan.json'),spawn_data_fp=str(data/'spawn_data/tidy_house/pick/val/spawn_data.pt'))
        kwargs=dict(cfg.get('env_kwargs',{}));task_cfgs=dict(kwargs.get('task_cfgs',{}))
        # reset calls evaluate once. 601 is only a config offset so reset leaves
        # 600 physical control steps; no counter/force mutation after reset.
        task_cfgs['pick']={'horizon':200 if args.condition=='SAC' else 601}
        kwargs.update(require_build_configs_repeated_equally_across_envs=False,target_randomization=False,
                      add_event_tracker_info=True,task_cfgs=task_cfgs,human_render_camera_configs={'width':512,'height':512})
        if args.condition!='SAC':
            kwargs['sensor_configs']={'width':640,'height':640}
            cfg.update(stationary_base=False,stationary_torso=False,stationary_head=False,obs_mode='rgbd')
        cfg['env_kwargs']=kwargs
        if cfg['env_id']!='PickSubtaskTrain-v0':raise ValueError('official Pick checkpoint environment mismatch')
        env=make_env(from_dict(EnvConfig,cfg));env.env.auto_reset=False
        seed=20260930+int(args.case_id.rsplit('-',1)[1])
        obs,info=env.reset(seed=seed,options={'reconfigure':True,'spawn_selection_idxs':[row['spawn_index']]})
        u=env.unwrapped;target=u.subtask_objs[0]
        if str(u.task_plan[0].composite_subtask_uids[0])!=row['uid'] or u.spawn_selection_idxs[0]!=row['spawn_index']:raise ValueError('runtime spawn binding')
        if args.condition!='SAC' and int(scalar(info['subtasks_steps_left']))!=600:raise ValueError('reset must leave exactly 600 steps; no counter repair')
        if float(u.pick_cfg.robot_cumulative_force_limit)!=5000 or float(u.pick_cfg.ee_rest_thresh)!=.05:raise ValueError('official criteria changed')
        validate_official_controller(u.agent.controller)
        fk=RobotFK(args.code/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/fetch.urdf',[j.name for j in u.agent.robot.active_joints])
        def qpos():return u.agent.robot.qpos[0].detach().cpu().numpy().astype(float)
        def qvel():return u.agent.robot.qvel[0].detach().cpu().numpy().astype(float)
        def matrix(pose):return np.asarray(pose[0].sp.to_transformation_matrix())
        def world_base():return matrix(u.agent.base_link.pose)
        if np.max(abs(fk.tcp(qpos())-np.linalg.inv(world_base())@matrix(u.agent.tcp.pose)))>1e-5:raise ValueError('FK identity')
        component,shapes=_component_and_shapes(target);meshes=get_component_meshes(component)
        if not meshes or len(meshes)!=len(shapes):raise ValueError('target collision coverage')
        vertices=np.concatenate([m.vertices for m in meshes]);lo=vertices.min(0);hi=vertices.max(0)
        base_from_object=np.linalg.inv(world_base())@matrix(target.pose)
        surface_triangles=np.concatenate([m.triangles for m in meshes])
        corners=np.array([[x,y,z,1] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        box_base=(corners@base_from_object.T)[:,:3]
        initial_z=float(matrix(target.pose)[2,3]);stopped=False
        report.update(control_dt=1/float(u.control_freq),initial_robot_state={'qpos':qpos().tolist(),'tcp_base':fk.tcp(qpos()).tolist(),
            'base_world_eval_only':world_base().tolist(),'finger_distance_m':float(qpos()[-2:].sum())},
            initial_binding={'uid':row['uid'],'spawn_index':row['spawn_index'],'seed':seed,'qpos':qpos().tolist(),
                             'base_world':world_base().tolist(),'target_world':matrix(target.pose).tolist()},
            configured_horizon=horizon,internal_pick_horizon=int(u.pick_cfg.horizon),initial_steps_left=int(scalar(info['subtasks_steps_left'])))
        save(args.output/'initial-binding.json',report['initial_binding'])
        bind_initial(args.output.parent/'paired-initial-binding.json',report['initial_binding'])
        recording=AttemptRecording(args.output/'recording-frames',float(u.control_freq));recording.capture_environment(u)
        initial_info=jsonable(info);accepted_checks=[]
        report['initial_evaluation']={'step':0,'strict_success':strict_pick(initial_info),
            'is_grasped':bool(scalar(info['is_grasped'])),'object_z_m':initial_z,
            'tcp_box_distance_m':tcp_box_distance((np.linalg.inv(matrix(target.pose))@matrix(u.agent.tcp.pose))[:3,3],lo,hi)}
        def step(action):
            nonlocal info,obs,stopped
            deadline()
            if stopped:raise RuntimeError('step after official terminal')
            old_left=int(scalar(info['subtasks_steps_left']));old_force=float(scalar(info['robot_cumulative_force']))
            obs,_,terminated,truncated,info=env.step(torch.as_tensor(action,device=u.device,dtype=torch.float32).reshape(1,-1))
            report['simulator_actions']+=1;values=jsonable(info);success=strict_pick(values)
            if int(scalar(info['subtasks_steps_left']))!=old_left-1 or float(scalar(info['robot_cumulative_force']))+1e-5<old_force:raise ValueError('clock/force discontinuity')
            tcp_object=np.linalg.inv(matrix(target.pose))@matrix(u.agent.tcp.pose)
            frame={'step':report['simulator_actions'],'qpos':qpos().tolist(),'qvel':qvel().tolist(),
                'base_world_eval_only':world_base().tolist(),'tcp_base':fk.tcp(qpos()).tolist(),
                'finger_distance_m':float(qpos()[-2:].sum()),'action':list(map(float,action)),
                'official_info_eval_only':values,'strict_success':success,'is_grasped':bool(scalar(info['is_grasped'])),
                'object_z_m':float(matrix(target.pose)[2,3]),'tcp_box_distance_m':tcp_box_distance(tcp_object[:3,3],lo,hi)}
            report['frames'].append(frame);report['strict_pick_success']|=success
            stopped=success or bool(scalar(info.get('fail',False))) or bool(scalar(terminated)) or bool(scalar(truncated)) or report['simulator_actions']>=horizon
            report['official_final_info']=values;save(args.output/'result.json',report);recording.capture_environment(u)
            return {'terminated':stopped,'truncated':bool(scalar(truncated))}
        if args.condition=='SAC':
            from mshab.utils.array import to_tensor
            obs=to_tensor(obs,device=u.device,dtype='float')
            adapter=SimpleNamespace(uenv=u,logger=JsonlLogger(args.output/'events.jsonl',args.case_id),observe=lambda:SimpleNamespace(policy=obs))
            policy=load_rl_policy(checkpoint/'config.yml',checkpoint/'policy.pt',adapter)
            while not stopped and report['simulator_actions']<200:
                with torch.no_grad():action=policy(obs)
                if not torch.isfinite(action).all():raise ValueError('nonfinite SAC action')
                step(action[0].detach().cpu().tolist());obs=to_tensor(obs,device=u.device,dtype='float')
        else:
            model=json.loads((args.code/'research/c2/diagnostics/2026-09-30-eef-r3-offline-followup/response-model.json').read_text())['model']
            ex=ArmCoordinatedV2(fk,qpos,step,lambda:float(qpos()[-2:].sum()),
                lambda:{k:bool(scalar(info.get(k,False))) for k in ('ee_rest','robot_rest','is_static')},model,
                read_qvel=qvel,control_dt=report['control_dt'],initial_delta=qvel()*report['control_dt'],gripper=float(np.clip((qpos()[-2:].mean()+.01)/.06,0,1)))
            def record(call,result):
                report['commands'].append({'turn':len(report['commands']),'call':call,'result':result,
                    'start_step':ex.steps-result.get('steps_used',0)})
                if call and call['tool']=='check_path':accepted_checks.append(result.get('accepted') is True)
                save(args.output/'result.json',report)
            if args.condition=='script':
                report['script_result']=run_reference(ex,box_base.min(0).tolist(),box_base.max(0).tolist(),record,lambda:stopped)
            else:
                from eef_arm_server_broker import process,runtime_sender
                archive=[];history=[];previous=None;ids=[];depth=ObservationDepth()
                for turn in range(25):
                    if stopped:break
                    observation,manifest=observe(u,fk,qpos,target_semantics(row['object_id']),history,600-ex.steps,
                        args.output/'sensors'/f'turn-{turn:03d}',ex.odometry(),depth,previous)
                    bundle={'position':args.case_id,'turn':turn,'kind':'tool','authorization_sha256':sha(args.authorization),
                            'observation':observation,'sensor_manifest':manifest}
                    request_dir=args.output/'provider'/f'turn-{turn:03d}';request_dir.mkdir(parents=True)
                    response=process(bundle,archive,args.condition,request_dir,args.ledger,auth,sha(args.authorization),runtime_sender(request_dir,lambda:cap-(time.monotonic()-started)-10))
                    report['api_requests']+=1
                    if response['status']=='censored':raise RuntimeError('provider censored')
                    call=response.get('call')
                    report['pending_command']={'turn':turn,'call':call,'start_step':ex.steps,
                        'reasoning_summaries':response.get('reasoning_summaries',[])}
                    save(args.output/'result.json',report)
                    if call is None:actual=None;result={'accepted':False,'reason':'invalid_tool_call','steps_used':0}
                    else:actual,result=dispatch(ex,call,depth,ids)
                    if actual and actual['tool']=='locate_point':
                        from bvi.eef_arm_localization import diagnose_location
                        diagnostic=diagnose_location(result,surface_triangles,matrix(target.pose),world_base())
                        report.setdefault('localization_eval_only',[]).append(dict(diagnostic,turn=turn,step=ex.steps))
                    record(actual,result);report['commands'][-1]['reasoning_summaries']=response.get('reasoning_summaries',[])
                    report.pop('pending_command',None)
                    archive.append(copy.deepcopy({'bundle':bundle,'call':call}))
                    history.append(copy.deepcopy({'call':actual,'result':result}))
                    previous=copy.deepcopy(state_snapshot(observation));ids.append(observation['observation_id'])
                    if actual and actual['tool'] in ('done','give_up'):
                        report['termination_category']=actual['tool'];break
                report['reflection_status']='not_requested; terminal hindsight and per-call notes retained'
            report.update(executor_trace=ex.trace,ik_records=ex.ik_records)
        scored=[report['initial_evaluation']]+report['frames']
        report.update(status='completed',**horizon_results(scored))
        if args.condition=='SAC':report['horizon_description']='official SAC 200-step configuration'
        report['termination_category']=report.get('termination_category') or ('official_success' if report['strict_pick_success'] else
            'cumulative_force_limit' if float(scalar(info['robot_cumulative_force']))>=5000 else 'time_limit' if stopped else
            'script_sequence_end' if args.condition=='script' else 'tool_call_limit')
        report['funnel']=funnel(scored,accepted_checks,initial_z)
        if args.condition=='SAC':report['funnel']['check_path_applicability']='N/A'
    except Exception as exc:
        report.update(status='infrastructure_censored',error_type=type(exc).__name__)
        # Do not stringify provider exceptions (may contain request material).
    finally:
        if env is not None:
            try:env.close()
            except Exception as exc:report['close_error_type']=type(exc).__name__
        report.update(wall_seconds=time.monotonic()-started,recording={'kind':'online official-spawn '+args.condition,
            'frames':0 if recording is None else recording.count,'directory':'recording-frames'})
        save(args.output/'result.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('authorization','roster','code','lab-root','output','ledger'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--case-id',required=True);p.add_argument('--condition',choices=['V','P','SAC','script'],required=True)
    args=p.parse_args();result=run(args)
    print(json.dumps({k:result.get(k) for k in ('case_id','condition','status','strict_pick_success','wall_seconds')}))
    raise SystemExit(0 if result['status']=='completed' else 1)
