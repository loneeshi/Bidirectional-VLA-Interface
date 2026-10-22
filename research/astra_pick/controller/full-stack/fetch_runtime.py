"""Lazy official-SDK Fetch profile; importing this file creates no environment.
Uses a private agent lookup while retaining the scene's fetch string identity; never mutates registry or
legacy Fetch profiles. Deployment/physical execution require separate approval.
"""
import copy
FLOW_CONTROL_MODE='flowsde_full'

def flow_config(original,velocity_config_cls):
    base=copy.deepcopy(original['pd_joint_delta_pos_vel'])
    if list(base['body'].joint_names)!=['head_pan_joint','head_tilt_joint','torso_lift_joint']:
        raise ValueError('Unexpected Fetch body joint order')
    head=base.pop('body');head.joint_names=['head_pan_joint','head_tilt_joint']
    # True velocity actuator for torso, matching native Command.vz semantics.
    # This intentionally changes torso stiffness from position PD to zero.
    torso=velocity_config_cls(joint_names=['torso_lift_joint'],lower=-.1,upper=.1,damping=100.,force_limit=100.,friction=0.,normalize_action=True)
    base['head']=head;base['torso']=torso
    return base

def agent_class():
    from mani_skill.agents.robots.fetch.fetch import Fetch
    from mani_skill.agents.controllers import PDJointVelControllerConfig
    class FetchFlowCore(Fetch):
        @property
        def _controller_configs(self):
            configs=super()._controller_configs
            configs[FLOW_CONTROL_MODE]=flow_config(configs,PDJointVelControllerConfig)
            return configs
    # Snapshot coverage identifies non-serializable SDK objects by class name.
    # Preserve that transport identity without changing the global Fetch class.
    FetchFlowCore.__name__=Fetch.__name__
    FetchFlowCore.flowsde_private_controller=True
    return FetchFlowCore


def inspect_sdk_config_cpu():
    """Evaluate actual SDK dataclasses without constructing any agent/environment."""
    import ast,inspect,textwrap
    from types import SimpleNamespace
    from mani_skill.agents.robots.fetch.fetch import Fetch
    from mani_skill.agents.controllers import PDJointVelControllerConfig
    attrs={}
    for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(Fetch.__init__)))):
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            target=node.targets[0]
            if isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self':
                try:attrs[target.attr]=ast.literal_eval(node.value)
                except (ValueError,TypeError):pass
    attrs['urdf_path']=Fetch.urdf_path
    original=Fetch._controller_configs.fget(SimpleNamespace(**attrs))
    candidate=flow_config(original,PDJointVelControllerConfig)
    assert original['pd_joint_delta_pos_vel']['body'].joint_names==['head_pan_joint','head_tilt_joint','torso_lift_joint']
    assert candidate['torso'].controller_cls.__name__=='PDJointVelController'
    assert candidate['torso'].joint_names==['torso_lift_joint']
    assert candidate['head'].joint_names==['head_pan_joint','head_tilt_joint']
    assert issubclass(agent_class(),Fetch)
    groups={k:dict(class_name=type(v).__name__,controller=v.controller_cls.__name__,joints=list(v.joint_names),lower=v.lower,upper=v.upper,damping=v.damping,force_limit=v.force_limit,friction=v.friction) for k,v in candidate.items()}
    return dict(status='SDK config CPU PASS',groups=groups,legacy_body_unchanged=True,agent_instances=0,task_environments=0,physics_steps=0,live_driver_readback=False)


def robot_spec_from_sources(urdf_path,fetch_source_path,driver_receipt,*,physics_dt,n_sub):
    """Build robot facts from archived sources; rates remain explicit runtime inputs.

    URDF speed limits are planner bounds, not velocity-feedforward action scales.
    Drive torque limits come from SDK configuration, not URDF effort metadata.
    This factory does not certify the dynamics model or an active driver.
    """
    import ast,math
    import xml.etree.ElementTree as ET
    from flowsde_ctrl.robot import RobotSpec,InnerLoopSpec,BaseSpec
    if not math.isfinite(physics_dt) or physics_dt<=0 or type(n_sub) is not int or n_sub<=0:
        raise ValueError('Invalid explicit control rates')
    attrs={}
    for node in ast.walk(ast.parse(fetch_source_path.read_text())):
        if isinstance(node,ast.Assign) and len(node.targets)==1:
            t=node.targets[0]
            if isinstance(t,ast.Attribute) and isinstance(t.value,ast.Name) and t.value.id=='self':
                try:attrs[t.attr]=ast.literal_eval(node.value)
                except (ValueError,TypeError):pass
    groups=driver_receipt['groups']; arm=groups['arm']; base=groups['base']
    names=list(attrs['arm_joint_names'])
    if arm['joints']!=names or arm['controller']!='PDJointPosVelController':
        raise ValueError('Unexpected source/driver arm contract')
    if arm['lower']!=-.1 or arm['upper']!=.1:
        raise ValueError('Unexpected Fetch arm position scale')
    if arm['damping']!=attrs['arm_damping'] or arm['force_limit']!=attrs['arm_force_limit']:
        raise ValueError('Source and config drive gains disagree')
    joints={j.attrib['name']:j for j in ET.parse(urdf_path).getroot().findall('joint')}
    velocities=[float(joints[n].find('limit').attrib['velocity']) for n in names]
    if any(not math.isfinite(v) or v<=0 for v in velocities):raise ValueError('Invalid URDF speed limit')
    links=[joints[n].find('child').attrib['link'] for n in names]
    inner=InnerLoopSpec(kp=[attrs['arm_stiffness']]*7,kd=[arm['damping']]*7,
        tau_max=[arm['force_limit']]*7,physics_dt=physics_dt,n_sub=n_sub,
        sp_scale=.1,vff_scale=1.,sp_window=.1,pose_delay_steps=0)
    return RobotSpec(name='Fetch_flowsde_candidate',arm_joints=names,ee_site='fetch_tcp',inner=inner,
        base=BaseSpec(type='differential',joints=base['joints'],torso_joint='torso_lift_joint',
            body='base_link',vmax=.5,wmax=.6,torso_vmax=.1,fric_ff=(0,0,0),yaw_action_scale=base['upper'][1]),
        qd_max=velocities,shoulder_body=links[0],wrist_body=links[-1],hand_body='gripper_link',link_bodies=links,
        notes='URDF speed limits; SDK gains. Base planning caps .5m/s/.6rad/s are candidate policy, not identified hardware. Explicit caller rates require runtime verification. Nominal dynamics not accepted.')


def run_episode_ports(episode,*,observe,step_once,recorder,initial_info,save_report,
                      case_seconds,batch_deadline,clock=None):
    """Execute an already restored candidate through explicit ports.

    Never creates/restores a task or launches a process. A separately authorized
    worker must supply the actual ports and an external hard watchdog (blocking
    SDK/capture calls cannot be interrupted by these cooperative deadline checks).
    Recorder is the existing paired-template Recorder with action_size=20.
    """
    import copy,time,math
    import numpy as np
    clock=time.monotonic if clock is None else clock
    if not math.isfinite(case_seconds) or case_seconds<=0 or not math.isfinite(batch_deadline):
        raise ValueError('Explicit finite case and batch deadlines required')
    if getattr(recorder,'action_size',None)!=20:raise ValueError('Full controller requires20-action recorder')
    start=clock();deadline=min(start+case_seconds,batch_deadline)
    report=dict(status='running',confirmed_actions=0,pending_action=None,frames=0,frame_timings=[],trace=[],api_requests=0)
    info=copy.deepcopy(initial_info);status=dict(terminated=False,truncated=False)
    def check_time():
        if clock()>=deadline:raise TimeoutError('case_or_batch_wall_limit')
    def capture(action,phase):
        report['stage_events']=copy.deepcopy(episode.events)
        begin=clock();recorder.record(report['confirmed_actions'],action,info,phase=phase)
        elapsed=clock()-begin;step=report['confirmed_actions']
        report['frames']+=1
        report['frame_timings'].append(dict(step=step,wall_seconds=elapsed,slow_frame_warning=elapsed>(10 if step==0 else 2.5),policy='warn_only'))
        save_report(copy.deepcopy(report));check_time()
    try:
        check_time();capture(None,'restored_entry')
        while True:
            check_time()
            measured=observe()
            # Control receives only selected measurements/official flags. Full
            # info and third-person recorder state never enter the controller.
            packet=episode.advance(sequence=report['confirmed_actions'],
                qpos=measured['qpos'],qvel=measured['qvel'],local_from_base=measured['local_from_base'],
                base_velocity=measured['base_velocity'],rest_flags=measured.get('rest_flags'),
                terminated=status['terminated'],truncated=status['truncated'])
            if packet is None:
                report['stage_events']=copy.deepcopy(episode.events)
                report['status']=episode.status;break
            action=np.asarray(packet['action'],np.float32)
            if action.shape!=(20,) or not np.isfinite(action).all() or np.max(abs(action))>1:
                raise ValueError('Invalid full-controller action')
            check_time()
            report['pending_action']=report['confirmed_actions']+1
            from bvi.pick_system.executor_diagnostics import state_plain
            report['pending_command']=dict(action=action.tolist(),phase=packet['phase'],
                controller_info=state_plain(copy.deepcopy(packet.get('controller_info'))),
                clipped_groups=state_plain(copy.deepcopy(packet.get('clipped_groups'))),
                measured_input=state_plain(copy.deepcopy(measured)))
            save_report(copy.deepcopy(report))
            status,info=step_once(action)
            report['confirmed_actions']+=1;report['pending_action']=None
            report['trace'].append(dict(step=report['confirmed_actions'],**report.pop('pending_command')))
            save_report(copy.deepcopy(report))
            # Retain the last executed frame even when official invariants fail.
            capture(action,packet['phase'])
            if status.get('clock_valid') is not True or status.get('force_valid') is not True:
                raise ValueError('Official clock/force discontinuity')
        recorder.finish(report['status'])
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        # pending_action remains if dispatch outcome is unknown. Never retry it.
        try:recorder.finish('failed')
        except BaseException as finish_error:report['recorder_finish_error']=str(finish_error)
        save_report(copy.deepcopy(report))
        raise
    report['wall_seconds']=clock()-start
    report['official_info_eval_only']=copy.deepcopy(info)
    save_report(copy.deepcopy(report))
    return report


class FetchMeasuredPorts:
    """Existing C0 root-joint relative odometry plus robot-only camera FK.

    Root joints are the existing simulator odometry proxy, not a calibrated
    wheel/IMU sensor. No absolute base_link pose, object pose or scene mesh is
    read. Absolute root coordinates are stripped before returning measurements.
    SDK robot sensor depth uses millimetres; map input uses metres/OpenCV axes.
    """
    def __init__(self,read_qpos,read_qvel,read_rest_status,robot_fk):
        import numpy as np
        self.read_qpos=read_qpos;self.read_qvel=read_qvel;self.read_rest_status=read_rest_status;self.fk=robot_fk
        initial=np.asarray(read_qpos(),float)
        if initial.shape!=(15,) or not np.isfinite(initial).all():raise ValueError('Invalid initial encoders')
        self.origin=initial[:3].copy()

    def observe(self):
        import numpy as np
        q=np.asarray(self.read_qpos(),float).copy();v=np.asarray(self.read_qvel(),float).copy()
        if q.shape!=(15,) or v.shape!=(15,) or not np.isfinite(np.r_[q,v]).all():raise ValueError('Invalid encoders')
        c,s=np.cos(self.origin[2]),np.sin(self.origin[2]);dx,dy=q[:2]-self.origin[:2]
        yaw=q[2]-self.origin[2];cy,sy=np.cos(yaw),np.sin(yaw)
        local=np.eye(4);local[:2,:2]=[[cy,-sy],[sy,cy]];local[:2,3]=[c*dx+s*dy,-s*dx+c*dy]
        c,s=np.cos(q[2]),np.sin(q[2]);vx,vy=v[:2]
        body_velocity=np.array([c*vx+s*vy,-s*vx+c*vy,v[2]])
        q[:3]=0.;v[:3]=body_velocity  # absolute root pose never escapes adapter
        flags=self.read_rest_status()
        if set(flags)!=set(('ee_rest','robot_rest','is_static')) or any(type(x) is not bool for x in flags.values()):
            raise ValueError('Explicit official rest flags required')
        return dict(qpos=q,qvel=v,local_from_base=local,base_velocity=body_velocity,rest_flags=dict(flags))

    def local_from_camera(self,measured,camera):
        import numpy as np
        if camera not in ('fetch_head','fetch_hand'):raise ValueError('Robot sensors only')
        links=self.fk.links(measured['qpos'])
        mount=np.eye(4)
        if camera=='fetch_head':base_camera=links['head_camera_link']
        else:
            mount[:3,3]=[-.1,0,.1];base_camera=links['gripper_link']@mount
        # SDK mount uses forward/left/up; measured depth uses right/down/forward.
        flu_from_cv=np.eye(4);flu_from_cv[:3,:3]=[[0,0,1],[-1,0,0],[0,-1,0]]
        return measured['local_from_base']@base_camera@flu_from_cv

    def fuse_depth(self,depth_map,measured,*,camera,depth_mm,intrinsics,timestamp,robot_mask):
        import numpy as np
        depth=np.asarray(depth_mm)
        if depth.ndim!=2 or depth.dtype.kind not in 'uif' or robot_mask is None:
            raise ValueError('Need raw2D millimetre depth and explicit robot self-mask')
        return depth_map.push_depth(depth.astype(float)*.001,intrinsics,self.local_from_camera(measured,camera),
            timestamp=timestamp,robot_mask=robot_mask)


def private_fetch_spec(gym_module,env_id,fetch_class):
    """Private Gym/MS/agent specs; preserve native scene initialization and wrappers."""
    import types,functools
    from mani_skill.envs.sapien_env import BaseEnv
    from mani_skill.utils import registration
    if env_id!='PickSubtaskTrain-v0':raise ValueError('Only the pinned Fetch Pick task is supported')
    native=registration.REGISTERED_ENVS[env_id]
    loader=BaseEnv._load_agent
    agents=dict(loader.__globals__['REGISTERED_AGENTS'])
    agent_spec=copy.copy(agents['fetch']);agent_spec.agent_cls=fetch_class;agents['fetch']=agent_spec
    scope=dict(loader.__globals__,REGISTERED_AGENTS=agents)
    private_loader=types.FunctionType(loader.__code__,scope,loader.__name__,loader.__defaults__,loader.__closure__)
    private_loader.__kwdefaults__=loader.__kwdefaults__
    # Insert just before BaseEnv: native SceneManipulationEnv._load_agent still
    # forwards its initial pose through super(), then reaches the private lookup.
    private_base=type('PrivateFetchLookup',(BaseEnv,),{'_load_agent':private_loader})
    # Snapshot transport keys the task layer by type.__name__. Preserve that
    # archival identity; the distinct class/MRO and explicit marker identify the
    # candidate implementation without rewriting archived state or RNG keys.
    candidate=type(native.cls.__name__,(native.cls,private_base),
                   {'__module__':__name__,'flowsde_private_agent_lookup':True})
    if candidate.__mro__.index(private_base)+1!=candidate.__mro__.index(BaseEnv):
        raise ValueError('Unexpected task inheritance; cannot preserve native loader')
    env_spec=copy.copy(native);env_spec.cls=candidate
    envs=dict(registration.REGISTERED_ENVS);envs[env_id]=env_spec
    make=registration.make
    private_make=types.FunctionType(make.__code__,dict(make.__globals__,REGISTERED_ENVS=envs),make.__name__,make.__defaults__,make.__closure__)
    spec=copy.deepcopy(gym_module.spec(env_id))
    spec.entry_point=functools.partial(private_make,env_id=env_id)
    return spec


def bind_fetch_factory(make_env,fetch_class):
    """Reuse native factory code with a private gym.make binding.

    MS-HAB hardcodes robot_uids='fetch'; an env_kwargs duplicate is invalid.
    Clone only function/spec lookups. Keep robot_uids='fetch' for scene builders.
    Never mutates gym.make, SDK registry or the original factory globals.
    """
    import types
    gym_module=make_env.__globals__['gym']
    def create(*args,**kwargs):
        if kwargs.get('robot_uids')!='fetch' or kwargs.get('control_mode')!='pd_joint_delta_pos':
            raise ValueError('Native factory robot/control contract changed')
        if len(args)!=1:raise ValueError('Expected one explicit native environment id')
        spec=private_fetch_spec(gym_module,args[0],fetch_class)
        return gym_module.make(spec,**kwargs)
    scope=dict(make_env.__globals__)
    proxy=types.SimpleNamespace(**vars(gym_module));proxy.make=create;scope['gym']=proxy
    result=types.FunctionType(make_env.__code__,scope,make_env.__name__,make_env.__defaults__,make_env.__closure__)
    result.__kwdefaults__=make_env.__kwdefaults__
    return result


def activate_full_controller(agent,*,capture_plant_state,states_equal):
    """After strict legacy snapshot restore, initialize new control state only.

    capture_plant_state must cover robot/scene/task/RNG state while excluding
    controller state and drive target/configuration, which intentionally change.
    Caller must separately verify official task limits and stop flags.
    """
    if agent.control_mode!='pd_joint_delta_pos' or FLOW_CONTROL_MODE in agent.controllers:
        raise ValueError('Expected fresh legacy-restored candidate agent')
    if FLOW_CONTROL_MODE not in agent.supported_control_modes:
        raise ValueError('Candidate Fetch subclass required at construction')
    before=capture_plant_state()
    agent.set_control_mode(FLOW_CONTROL_MODE)
    agent.controller.reset()
    if not states_equal(before,capture_plant_state()):
        raise ValueError('Controller activation changed physical/task/RNG state; do not step')
    if agent.controller.single_action_space.shape!=(20,):raise ValueError('Full action space mismatch')
    return dict(control_mode=agent.control_mode,action_mapping=dict(agent.controller.action_mapping),
        physical_state_unchanged=True,controller_reinitialized=True,physics_steps=0)


def run_serial_workers(jobs,*,dispatch,save_report,total_steps,batch_deadline,clock=None):
    """Budget/receipt orchestration only; authorized caller supplies dispatch.

    Reserve each complete case cap before dispatch. Unknown outcomes retain that
    reservation and stop. No retry/resume or implicit physical authorization.
    """
    import time,math,copy
    clock=time.monotonic if clock is None else clock
    if type(total_steps) is not int or total_steps<=0 or not math.isfinite(batch_deadline):raise ValueError('Explicit batch limits required')
    if not jobs or len({j['case_id'] for j in jobs})!=len(jobs):raise ValueError('Unique frozen case order required')
    for job in jobs:
        if type(job['steps']) is not int or job['steps']<=0 or not math.isfinite(job['seconds']) or job['seconds']<=0:raise ValueError('Invalid case budget')
    if sum(j['steps'] for j in jobs)>total_steps:raise ValueError('Case reservations exceed batch cap')
    report=dict(status='running',reserved_steps=0,confirmed_steps=0,attempts=[],retries=0,api_requests=0)
    for job in jobs:
        remaining=batch_deadline-clock()
        if remaining<=0:
            report['status']='batch_deadline';save_report(copy.deepcopy(report));return report
        entry=dict(case_id=job['case_id'],reserved_steps=job['steps'],wall_seconds=min(job['seconds'],remaining),status='pending')
        report['attempts'].append(entry);report['reserved_steps']+=job['steps']
        # Persist reservation before any side effect. Persistence failure stops.
        save_report(copy.deepcopy(report))
        try:
            result=dispatch(copy.deepcopy(job),wall_seconds=entry['wall_seconds'])
            count=result['confirmed_steps']
            if type(count) is not int or not 0<=count<=job['steps'] or type(result['accepted']) is not bool:raise ValueError('Invalid case receipt')
            entry.update(status='accepted' if result['accepted'] else 'rejected',receipt=copy.deepcopy(result))
            report['confirmed_steps']+=count
        except BaseException as error:
            entry.update(status='unknown_or_failed',error_type=type(error).__name__,error=str(error))
            report['status']='stopped_on_error'
            try:save_report(copy.deepcopy(report))
            except BaseException as receipt_error:
                # Retain the dispatch failure even when its receipt cannot persist.
                import sys
                print('Serial receipt persistence failed: '+repr(receipt_error),file=sys.stderr)
            raise
        # Check after every return, including the final case. The parent watchdog
        # enforces termination; this check prevents late completion acceptance.
        if clock()>=batch_deadline:
            report['status']='batch_deadline';save_report(copy.deepcopy(report));return report
        if not result['accepted']:
            report['status']='stopped_on_case';save_report(copy.deepcopy(report));return report
        save_report(copy.deepcopy(report))
    report['status']='completed';save_report(copy.deepcopy(report));return report


def run_worker_bounded(argv,*,cwd,environment,log_path,wall_seconds,grace_seconds=5.):
    """POSIX parent watchdog for one explicitly supplied worker; never retries.

    TERM starts before the hard deadline; KILL at the deadline includes all
    descendants still in the new process group. Callers enforce authorization,
    source hashes, shared lock and aggregate budgets before calling this helper.
    A zero exit code alone is not scientific/task success.
    """
    import os,signal,subprocess,time,math
    from pathlib import Path
    if os.name!='posix':raise RuntimeError('Lab POSIX watchdog required')
    if not argv or not all(isinstance(x,str) and x for x in argv):raise ValueError('Explicit argv required')
    if not math.isfinite(wall_seconds) or wall_seconds<=0 or not 0<=grace_seconds<wall_seconds:
        raise ValueError('Invalid hard deadline/grace')
    started=time.monotonic();deadline=started+wall_seconds;process=None
    def members(pgid):
        live=[]
        for entry in Path('/proc').iterdir():
            if not entry.name.isdigit():continue
            try:fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            except (FileNotFoundError,ProcessLookupError):continue
            if int(fields[2])==pgid and fields[0]!='Z':live.append(int(entry.name))
        return live
    def send(sig):
        try:os.killpg(process.pid,sig)
        except ProcessLookupError:pass
    receipt=dict(status='starting',hard_timeout=False,soft_timeout=False,retries=0)
    with Path(log_path).open('xb') as stream:
        try:
            process=subprocess.Popen(argv,cwd=cwd,env=dict(environment,FLOW_WATCHDOG_PARENT=str(os.getpid())),stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            receipt.update(pid=process.pid,pgid=process.pid)
            try:process.wait(timeout=max(0,deadline-grace_seconds-time.monotonic()))
            except subprocess.TimeoutExpired:
                receipt['soft_timeout']=True;send(signal.SIGTERM)
                try:process.wait(timeout=max(0,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    receipt['hard_timeout']=True;send(signal.SIGKILL);process.wait(timeout=5)
            # Parent exit can leave a child alive. Do not equate leader exit
            # with cleanup or permit that child to continue the experiment.
            remaining=members(process.pid)
            receipt['descendants_after_leader_exit']=remaining
            if remaining:send(signal.SIGKILL)
            until=time.monotonic()+2.
            while members(process.pid) and time.monotonic()<until:time.sleep(.02)
            receipt['remaining_live_members']=members(process.pid)
            if receipt['remaining_live_members']:raise RuntimeError('Owned worker group failed to exit')
            receipt.update(exit_code=process.returncode,status='exited' if process.returncode==0 and not receipt['soft_timeout'] and not remaining else 'stopped')
        except BaseException:
            if process is not None:
                send(signal.SIGKILL)
                process.wait(timeout=5)
            raise
    receipt['wall_seconds']=time.monotonic()-started
    return receipt


def plant_transition_view(payload):
    """Exact captured-state view for legacy->full control activation only.

    Preserve all physics/task/RNG/coverage; exclude controller-owned state and
    declared action-space caches. Only Fetch's15 drive target slots may change;
    other articulations, padding and physical q/qvel/qf remain checked.
    Does NOT weaken the strict snapshot restore check before activation.
    """
    from bvi.pick_system.executor_diagnostics import validate_snapshot
    if set(payload)!=set(('contract','layout','state')) or payload['contract']!='bvi-eval-snapshot-transport-v2':
        raise ValueError('Complete native transport required')
    state=payload['state'];validate_snapshot(state)
    result=dict(state);cache_fields={'action_space','single_action_space','_orig_single_action_space','_action_space'}
    is_control=lambda owner:owner=='controller' or owner.startswith('controller:')
    for key in ('attributes','skipped'):
        result[key]={owner:{name:value for name,value in fields.items() if name not in cache_fields}
            for owner,fields in state[key].items() if not is_control(owner)}
    # Controller RNG is not expected. Do not silently exclude it if introduced.
    result['gpu_buffers']=dict(state['gpu_buffers'])
    for key in ('cuda_articulation_target_qpos','cuda_articulation_target_qvel'):
        ids=payload['layout'][key];identity='[0,"articulation","fetch"]'
        if ids.count(identity)!=1:raise ValueError('Unambiguous Fetch target row required')
        values=state['gpu_buffers'][key]
        if values.ndim!=2 or values.shape[0]!=len(ids) or values.shape[1]<15:raise ValueError('Incomplete drive targets')
        view=values.clone() if hasattr(values,'clone') else values.copy()
        view[ids.index(identity),:15]=0
        result['gpu_buffers'][key]=view
    return dict(contract=payload['contract'],layout=payload['layout'],state=result)


def refresh_full_action_spaces(env):
    """Refresh declared caches on the existing wrapper chain, without stepping."""
    import copy
    u=env.unwrapped
    if u.agent.control_mode!=FLOW_CONTROL_MODE:raise ValueError('Activate full controller first')
    single=u.agent.single_action_space;batched=u.agent.action_space
    if single.shape!=(20,):raise ValueError('Expected full action space')
    node=env;seen=set();changed=[]
    while id(node) not in seen:
        seen.add(id(node));fields=vars(node)
        if type(node).__name__=='FetchActionWrapper':
            if any(fields.get(k) is not False for k in ('_stationary_base','_stationary_torso','_stationary_head')):
                raise ValueError('Legacy positional stationary mask would corrupt full actions')
        for name,value in (('action_space',batched),('single_action_space',single),('_orig_single_action_space',single),('_action_space',batched)):
            if name in fields:setattr(node,name,copy.deepcopy(value));changed.append(type(node).__name__+'.'+name)
        if node is u:break
        node=fields.get('env',fields.get('_env'))
        if node is None:raise ValueError('Unknown action wrapper chain')
    else:raise ValueError('Cyclic wrapper chain')
    return changed


def make_controller_session(full_root,qpos,*,sim_freq,control_freq,variant='cc_obs_v5'):
    """Construct the full core from actual restored encoders, never a test fixture."""
    import json,hashlib
    import numpy as np
    import mujoco
    from pathlib import Path
    from astra_pick.robot.eef_tools import FETCH_ORDER
    from fetch_tracker import CurrentMujocoModel,FetchTrackerSession,restore_loader_inertias
    root=Path(full_root);repo=root.parents[4]
    source=repo/'research/c2/diagnostics/2026-10-01-arm-reference-wp1-cpu/fetch.urdf'
    q=np.asarray(qpos,float)
    if q.shape!=(15,) or not np.isfinite(q).all():raise ValueError('Actual15-joint state required')
    if sim_freq!=100 or control_freq!=20:raise ValueError('Verified Fetch100/20Hz profile required')
    readback=json.loads((root/'robot-loader-readback.json').read_text())
    if hashlib.sha256(source.read_bytes()).hexdigest()!=readback['source_sha256']:raise ValueError('Robot source changed')
    spec=robot_spec_from_sources(source,root.parent/'followup/driver-source/fetch.py',
        json.loads((root/'driver-config-cpu.json').read_text()),physics_dt=1/sim_freq,n_sub=sim_freq//control_freq)
    m=mujoco.MjModel.from_xml_path(str(root/'fetch_kinematics_only.xml'))
    if m.ngeom!=0:raise ValueError('Controller model must be robot-only without scene geometry')
    restore_loader_inertias(m,readback)
    initial=m.qpos0.copy()
    for name,value in zip(FETCH_ORDER,q):initial[m.joint(name).qposadr[0]]=value
    model=CurrentMujocoModel(spec,m,initial)
    return FetchTrackerSession(spec,model,variant=variant)


def validate_full_driver(u):
    """Actual mode/config/drive readback before any full action; single Fetch only."""
    import numpy as np
    from astra_pick.robot.eef_tools import FETCH_ORDER
    if u.agent.control_mode!=FLOW_CONTROL_MODE or u.sim_freq!=100 or u.control_freq!=20:
        raise ValueError('Full mode/rate mismatch')
    if tuple(j.name for j in u.agent.robot.active_joints)!=FETCH_ORDER:raise ValueError('Robot joint order changed')
    if u.pick_cfg.robot_cumulative_force_limit!=5000 or u.pick_cfg.ee_rest_thresh!=.05:
        raise ValueError('Official task criteria changed')
    ctrl=u.agent.controller
    expected_names=dict(arm=[FETCH_ORDER[i] for i in [5,7,8,9,10,11,12]],
        gripper=['l_gripper_finger_joint','r_gripper_finger_joint'],base=list(FETCH_ORDER[:3]),
        head=['head_pan_joint','head_tilt_joint'],torso=['torso_lift_joint'])
    if set(ctrl.controllers)!=set(expected_names):raise ValueError('Unexpected action groups')
    receipt=dict(control_mode=FLOW_CONTROL_MODE,sim_freq=100,control_freq=20,groups={})
    for name,sub in ctrl.controllers.items():
        classes=dict(arm='PDJointPosVelController',head='PDJointPosController',gripper='PDJointPosMimicController',base='PDBaseForwardVelController',torso='PDJointVelController')
        if type(sub).__name__!=classes[name]:raise ValueError('Unexpected driver implementation: '+name)
        cfg=sub.config
        expected=dict(damping=1000 if name=='base' else 100,force_limit=500 if name=='base' else 100,friction=0.,normalize_action=True,drive_mode='force')
        if name in ('arm','head','gripper'):
            expected.update(stiffness=1000,interpolate=False,use_target=False,use_delta=name!='gripper')
        if name=='arm':expected.update(vel_lower=-1.,vel_upper=1.)
        bounds={'arm':(-.1,.1),'head':(-.1,.1),'gripper':(-.01,.05),'base':([-1.,-3.14],[1.,3.14]),'torso':(-.1,.1)}
        expected.update(lower=bounds[name][0],upper=bounds[name][1])
        if list(cfg.joint_names)!=expected_names[name] or sub._sim_steps!=5:raise ValueError('Joint/rate mapping changed')
        for key,value in expected.items():
            if not hasattr(cfg,key) or not np.all(np.asarray(getattr(cfg,key))==value):raise ValueError(name+'.'+key+' changed')
        for joint in sub.joints:
            if len(joint._objs)!=1:raise ValueError('Single environment required')
            drive=joint._objs[0]
            for key,value in dict(stiffness=0 if name in ('base','torso') else 1000,damping=expected['damping'],force_limit=expected['force_limit'],friction=0).items():
                if not np.all(np.asarray(getattr(drive,key))==value):raise ValueError('Actual drive differs: '+name+'.'+key)
        receipt['groups'][name]=dict(controller=type(sub).__name__,joints=list(cfg.joint_names),checked=True)
    return receipt


def run_restored_candidate(base,loaded,row,*,full_root,output,case_seconds,batch_deadline,phase=None):
    """Concrete worker body AFTER hash-bound legacy restore, not a launch command.

    The separately authorized supervisor owns construction, input/source binding,
    lock, reservation and watchdog. No default authorization or automatic retry.
    """
    import numpy as np
    from pathlib import Path
    import preflight
    from astra_pick.robot.eef_tools import RobotFK
    from bvi.pick_system.snapshot_transport import capture_transport
    from fetch_episode import FetchEpisode,stages_from_frozen_commands
    if loaded.get('restore',{}).get('passed') is not True:raise ValueError('Strict legacy restore receipt required')
    cap=row['score_steps']
    if type(cap) is not int or not 0<cap<=600:raise ValueError('Bounded official score steps required')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    phase=phase or (lambda name,operation:operation())
    captures=[]
    def capture():
        value=capture_transport(base)
        preflight.write_diagnostic_bundle(output/'diagnostics','activation_'+str(len(captures)),value)
        captures.append(value)
        return preflight.state_fingerprint(plant_transition_view(value))
    before=phase('activation_capture',capture)
    def activate():
        try:return activate_full_controller(base.uenv.agent,capture_plant_state=capture,states_equal=lambda a,b:a==b)
        except BaseException as error:
            try:
                if len(captures)>1:error.receipt=preflight.diagnostic_residuals(plant_transition_view(captures[0]),plant_transition_view(captures[-1]))
            except BaseException as secondary:error.diagnostic_error=repr(secondary)
            raise
    transition=phase('activation',activate)
    transition['action_caches']=phase('action_caches',lambda:refresh_full_action_spaces(base.env))
    def preserved():
        if capture()!=before:
            error=ValueError('Activation changed preserved task/scene/RNG state')
            error.receipt=preflight.diagnostic_residuals(plant_transition_view(captures[0]),plant_transition_view(captures[-1]))
            raise error
    phase('activation_preserved',preserved)
    preflight.save(output/'activation.json',transition)
    preflight.save(output/'driver.json',phase('driver',lambda:validate_full_driver(base.uenv)))
    ports=phase('recording_ports',lambda:preflight.native_ports(base,output/'frames',row['case_id'],'FULL',loaded['initial_info'],action_size=20))
    from video_template import full_action_labels
    ports['recorder'].action_labels=phase('action_labels',lambda:full_action_labels(base.uenv.agent.controller.action_mapping))
    session=phase('controller_session',lambda:make_controller_session(full_root,ports['read_qpos'](),sim_freq=base.uenv.sim_freq,control_freq=base.uenv.control_freq))
    fk=RobotFK(Path(full_root).parents[4]/'research/c2/diagnostics/2026-10-01-arm-reference-wp1-cpu/fetch.urdf')
    measured=FetchMeasuredPorts(ports['read_qpos'],ports['read_qvel'],ports['read_rest_status'],fk)
    episode=FetchEpisode(session,stages_from_frozen_commands(row['commands']),base.uenv.agent.controller.action_mapping,max_steps=cap)
    def persist(report):
        temporary=output/'progress.tmp';preflight.save(temporary,report);temporary.replace(output/'progress.json')
    with preflight.RecordingGCScope():
        result=phase('episode_first_frame_and_actions',lambda:run_episode_ports(episode,observe=measured.observe,step_once=ports['step_once'],recorder=ports['recorder'],
            initial_info=loaded['initial_info'],save_report=persist,case_seconds=case_seconds,batch_deadline=batch_deadline))
    preflight.save(output/'result.json',result)
    return result


class FetchNavCycle:
    """Compose measured perception/world/native QP with explicit action ack.

    Acquisition and dispatch are caller ports. Exceptions latch the cycle;
    stateful perception/QP updates must never be blindly retried.
    """
    def __init__(self,world,perception,session,*,control_dt=.05):
        if control_dt!=.05:raise ValueError('Native reference/control contract requires20Hz')
        if session.tracker.w is not world or session.tracker.measured_map is not perception.depth_map:
            raise ValueError('Navigation components do not share world/map')
        self.world=world;self.perception=perception;self.session=session;self.dt=control_dt
        self.pending=None;self.last_ack=None;self.stopped=False;self.failed=False
    def stop(self):
        self.stopped=True;self.session.official_stop()
    def prepare(self,measured,frames,*,timestamp,sequence,head_target,grip_closed,official_terminal=False):
        import numpy as np
        if official_terminal:self.stop()
        if self.stopped:raise StopIteration('Official navigation stop')
        if self.failed:raise RuntimeError('Navigation cycle failed; no automatic retry')
        if self.pending is not None:raise RuntimeError('Unacknowledged navigation action')
        if type(sequence) is not int or sequence<0 or (self.last_ack is not None and sequence!=self.last_ack+1):
            raise ValueError('Missing/duplicate navigation observation')
        if not np.isclose(timestamp,sequence*self.dt,atol=1e-9,rtol=0):raise ValueError('Navigation timestamp/tick mismatch')
        try:
            perception=self.perception.ingest(measured,frames,timestamp=timestamp)
            self.world.update(measured,timestamp)
            inp=self.world.track_input(measured,self.session.tracker.mod,grip_closed=grip_closed)
            packet=self.session.tick(inp,sequence=sequence,head_q=np.asarray(measured['qpos'])[[4,6]],head_target=head_target,grip_closed=grip_closed)
        except Exception:
            self.failed=True
            raise
        self.pending=sequence
        return dict(packet,perception=perception,timestamp=timestamp)
    def acknowledge(self,sequence,*,official_terminal=False):
        if self.failed:raise RuntimeError('Navigation cycle failed')
        if self.pending is None or sequence!=self.pending:raise ValueError('Unexpected navigation acknowledgement')
        self.last_ack=sequence;self.pending=None
        if official_terminal:self.stop()


class FetchNavWorld:
    """Native nav world interface restricted to commanded reference and odometry.

    Reference poses are in episode-local coordinates, sampled at upstream DT_REF.
    Endpoint hold matches the upstream planner's clamped reference indexing.
    """
    def __init__(self,positions,rotations,carry_rotation):
        import numpy as np
        from scipy.spatial.transform import Rotation,Slerp
        self.ref_pos=np.asarray(positions,float).copy();self.ref_rot=np.asarray(rotations,float).copy()
        if self.ref_pos.ndim!=2 or self.ref_pos.shape[1]!=3 or len(self.ref_pos)<2 or self.ref_rot.shape!=(len(self.ref_pos),3,3):
            raise ValueError('Expected sampled nav reference poses')
        self.carry_R=np.asarray(carry_rotation,float).copy()
        mats=np.concatenate([self.ref_rot,self.carry_R[None]],axis=0)
        if not np.isfinite(self.ref_pos).all() or not np.isfinite(mats).all() or not np.allclose(mats.transpose(0,2,1)@mats,np.eye(3),atol=1e-6) or not np.allclose(np.linalg.det(mats),1,atol=1e-6):
            raise ValueError('Invalid nav reference rotation')
        self.times=np.arange(len(self.ref_pos))*.05;self.slerp=Slerp(self.times,Rotation.from_matrix(self.ref_rot))
        self.yaw_anchor_b=np.zeros(3);self.base_z=0.;self.seed=0;self._time=None;self.B=None
    def update(self,measured,timestamp):
        import numpy as np
        from astra_pick.robot.eef_tools import transform
        B=transform(measured['local_from_base'])
        if not np.isfinite(timestamp) or timestamp<0 or (self._time is not None and timestamp<=self._time):raise ValueError('Nav clock must advance')
        if not np.allclose(B[2],[0,0,1,0],atol=1e-6):raise ValueError('Planar odometry required')
        self.B=B.copy();self._time=float(timestamp)
    def time(self):
        if self._time is None:raise ValueError('Nav odometry not initialized')
        return self._time
    def base_state(self):
        import numpy as np
        self.time();return float(self.B[0,3]),float(self.B[1,3]),float(np.arctan2(self.B[1,0],self.B[0,0]))
    def base_origin(self,x,y,yaw):
        import numpy as np
        return np.array([x,y])  # Fetch yaw joint is at the base frame origin.
    def track_input(self,measured,model,*,grip_closed):
        import numpy as np
        from flowsde_ctrl.inputs import TrackInput
        if not np.array_equal(self.B,measured['local_from_base']):raise ValueError('Nav observation frame changed')
        q=np.asarray(measured['qpos']);v=np.asarray(measured['qvel']);arm=[5,7,8,9,10,11,12]
        relative=np.arange(13)*.05;t=np.clip(self.time()+relative,self.times[0],self.times[-1])
        P=np.stack([np.interp(t,self.times,self.ref_pos[:,k]) for k in range(3)],axis=1);R=self.slerp(t).as_matrix()
        inv=np.linalg.inv(self.B);P=P@inv[:3,:3].T+inv[:3,3];R=np.einsum('ij,njk->nik',inv[:3,:3],R)
        p,rotation=model.fk(q[arm],q[3],jac=False)
        return TrackInput(False,q[arm],v[arm],relative,P-p,np.einsum('ij,njk->nik',rotation.T,R),h=q[3],
            base_vel=np.asarray(measured['base_velocity']),grip_cmd=np.full(2,1. if grip_closed else -1.))


def capture_nav_depth(u,ports,*,read_tick,control_dt):
    """Explicit native sensor capture port; never called by CPU validation.

    Reads intrinsic calibration only. Extrinsics remain measured FK/odometry.
    No get_info/evaluate, third-person cameras, scene poses or segmentation.
    """
    import numpy as np
    def array(x):
        if hasattr(x,'detach'):x=x.detach().cpu().numpy()
        return np.asarray(x).copy()
    if not np.isfinite(control_dt) or control_dt<=0:raise ValueError('Invalid control period')
    tick=read_tick()
    if type(tick) is not int or tick<0:raise ValueError('Invalid control tick')
    before=ports.observe();timestamp=tick*float(control_dt)
    names=('fetch_head','fetch_hand')
    sensors={name:u.scene.sensors[name] for name in names}
    for obj in u._hidden_objects:obj.hide_visual()
    u.scene.update_render(update_sensors=True,update_human_render_cameras=False)
    for sensor in sensors.values():sensor.capture()
    frames={}
    for name,sensor in sensors.items():
        data=sensor.get_obs(rgb=False,depth=True,position=False,segmentation=False,normal=False,albedo=False,apply_texture_transforms=True)
        depth=array(data['depth']);K=array(sensor.camera.get_intrinsic_matrix())
        if depth.ndim==4 and depth.shape[0]==1 and depth.shape[-1]==1:depth=depth[0,:,:,0]
        if K.shape==(1,3,3):K=K[0]
        if depth.ndim!=2 or K.shape!=(3,3):raise ValueError('Single-environment depth required')
        frames[name]=dict(depth_mm=depth,K=K,timestamp=timestamp,pixel_offset=(.5,.5))
    after=ports.observe()
    if read_tick()!=tick or any(not np.array_equal(before[k],after[k]) for k in ('qpos','qvel','local_from_base')):
        raise ValueError('Robot advanced during sensor capture')
    return before,frames,timestamp


class MeasuredNavPerception:
    """Fuse a synchronous robot-sensor bundle into the original navigation map.

    Caller supplies calibrated K, measured encoders and captured raw depth at
    the same timestamp. No sensor acquisition, simulator pose or scene IDs.
    """
    def __init__(self,ports,depth_map,self_filter,*,pixel_stride=10):
        if type(pixel_stride) is not int or pixel_stride<1:raise ValueError('Invalid pixel stride')
        self.ports=ports;self.depth_map=depth_map;self.self_filter=self_filter;self.stride=pixel_stride
    def ingest(self,measured,frames,*,timestamp):
        import numpy as np
        if set(frames)!= {'fetch_head','fetch_hand'}:raise ValueError('Synchronous head/hand bundle required')
        local=np.asarray(measured['local_from_base'],float)
        # Robot FK is base-local; self rays and robot meshes must share the
        # same local odometry frame after the base has moved.
        links={name:local@T for name,T in self.ports.fk.links(measured['qpos']).items()}
        prepared=[];counts={}
        for camera in ('fetch_head','fetch_hand'):
            frame=frames[camera]
            if frame['timestamp']!=timestamp:raise ValueError('Unsynchronized sensor bundle')
            raw=np.asarray(frame['depth_mm']);K=np.asarray(frame['K'],float).copy()
            if raw.ndim!=2 or K.shape!=(3,3):raise ValueError('Invalid raw sensor shape')
            offset=np.asarray(frame.get('pixel_offset',(0.,0.)),float)
            if offset.shape!=(2,) or not np.isfinite(offset).all():raise ValueError('Invalid pixel-center convention')
            # Convert declared pixel centers to integer array indices BEFORE
            # subsampling. Both mesh rays and map backprojection share this K.
            K[:2,2]-=offset
            depth=raw[::self.stride,::self.stride];K[:2]/=self.stride
            T=self.ports.local_from_camera(measured,camera)
            mask,_=self.self_filter.mask(depth,K,T,links)
            prepared.append((depth.astype(float)*.001,K,T,mask))
            counts[camera]=dict(sampled_pixels=int(depth.size),self_pixels=int(mask.sum()))
        points=self.depth_map.push_batch(prepared,timestamp=timestamp)
        return dict(timestamp=float(timestamp),cameras=counts,fused_points=int(len(points)))


class RobotVisualSelfFilter:
    """Predict robot-only visible depth from calibrated meshes and measured FK.

    Keep closer external occluders and mismatching farther returns. Tolerance is
    an explicit sensor/model policy, not proof of perfect visual registration.
    Per-link ray acceleration is cached; no scene model or segmentation input.
    """
    def __init__(self,archive,metadata,*,tolerance_m=.005,near_m=.01,backend='triangle',surface_fallback=False):
        import hashlib,numpy as np,trimesh
        from pathlib import Path
        if tolerance_m<=0 or near_m<=0 or not np.isfinite([tolerance_m,near_m]).all():raise ValueError('Invalid self-filter tolerance')
        path=Path(archive)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['archive_sha256']:raise ValueError('Robot visual archive changed')
        if backend=='triangle':
            from trimesh.ray.ray_triangle import RayMeshIntersector
        elif backend=='embree':
            from trimesh.ray.ray_pyembree import RayMeshIntersector
        else:raise ValueError('Unknown robot ray backend')
        if type(surface_fallback) is not bool:raise ValueError('Explicit surface fallback flag required')
        self.surface_fallback=surface_fallback
        self.backend=backend;self.intersectors={}
        self.tolerance=float(tolerance_m);self.near=float(near_m);self.meshes=[]
        with np.load(path,allow_pickle=False) as arrays:
            for row in metadata['meshes']:
                vertices=arrays[row['key']+'_vertices'];faces=arrays[row['key']+'_faces']
                mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
                self.meshes.append((row['link'],mesh))
                self.intersectors[id(mesh)]=RayMeshIntersector(mesh)
        if not self.meshes:raise ValueError('Robot visuals missing')

    def predict_depth(self,shape,K,base_from_camera,links):
        import numpy as np
        from astra_pick.robot.eef_tools import transform as rigid
        K=np.asarray(K,float);camera=rigid(base_from_camera)
        if len(shape)!=2 or any(type(v) is not int or v<=0 for v in shape):raise ValueError('Invalid image shape')
        if K.shape!=(3,3) or not np.isfinite(K).all() or K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]) or K[0,1]!=0 or K[1,0]!=0:
            raise ValueError('Calibrated pinhole required')
        y,x=np.indices(shape);rays=np.stack([(x-K[0,2])/K[0,0],(y-K[1,2])/K[1,1],np.ones(shape)],axis=-1).reshape(-1,3)
        depth=np.full(len(rays),np.inf)
        for name,mesh in self.meshes:
            if name not in links:raise ValueError('Missing robot FK link: '+name)
            transform=np.linalg.inv(rigid(links[name]))@camera;R=transform[:3,:3];t=transform[:3,3]
            directions=rays@R.T
            # Conservative link-local slab test before expensive triangle BVH.
            # Parameter t remains OpenCV z because each original ray has z=1.
            parallel=abs(directions)<1e-15
            with np.errstate(divide='ignore',invalid='ignore'):
                a=(mesh.bounds[0]-t)/directions;b=(mesh.bounds[1]-t)/directions
            low=np.minimum(a,b);high=np.maximum(a,b)
            inside=(t>=mesh.bounds[0])&(t<=mesh.bounds[1])
            low=np.where(parallel,np.where(inside,-np.inf,np.inf),low)
            high=np.where(parallel,np.where(inside,np.inf,-np.inf),high)
            selected=np.flatnonzero(high.min(axis=1)>=np.maximum(low.max(axis=1),self.near))
            if not len(selected):continue
            origins=np.broadcast_to(t,(len(selected),3))
            locations,index,_=self.intersectors[id(mesh)].intersects_location(origins,directions[selected],multiple_hits=True)
            if len(index):
                z=((locations-t)@R)[:,2];valid=np.isfinite(z)&(z>=self.near)
                np.minimum.at(depth,selected[index[valid]],z[valid])
        return depth.reshape(shape)

    def mask(self,depth_mm,K,base_from_camera,links):
        import numpy as np
        measured=np.asarray(depth_mm,float)*.001
        if measured.ndim!=2:raise ValueError('Expected2D millimetre depth')
        predicted=self.predict_depth(measured.shape,K,base_from_camera,links)
        valid=np.isfinite(measured)&(measured>0)&np.isfinite(predicted)
        mask=np.zeros(measured.shape,bool)
        mask[valid]=abs(measured[valid]-predicted[valid])<=self.tolerance
        if self.surface_fallback:
            import trimesh
            K=np.asarray(K,float);T=np.asarray(base_from_camera,float)
            ys,xs=np.nonzero(np.isfinite(measured)&(measured>=self.near)&~mask)
            z=measured[ys,xs]
            points=np.column_stack([(xs-K[0,2])*z/K[0,0],(ys-K[1,2])*z/K[1,1],z])@T[:3,:3].T+T[:3,3]
            on_surface=np.zeros(len(points),bool)
            for name,mesh in self.meshes:
                inv=np.linalg.inv(links[name]);local=points@inv[:3,:3].T+inv[:3,3]
                outside=np.maximum(np.maximum(mesh.bounds[0]-local,local-mesh.bounds[1]),0.)
                candidates=np.flatnonzero((np.linalg.norm(outside,axis=1)<=self.tolerance)&~on_surface)
                if not len(candidates):continue
                _,distances,_=trimesh.proximity.closest_point(mesh,local[candidates])
                on_surface[candidates]=distances<=self.tolerance
            mask[ys[on_surface],xs[on_surface]]=True
        return mask,predicted
