"""Frozen legacy executors plus an explicitly selected full-B online adapter."""

# ========================================================================
# Frozen source: eef_visual_config.py
"""Frozen first visual pilot. Changing any constant requires a new version."""
EEF_VISUAL_CONFIG_VERSION = 'eef-visual-pilot-v3-active-head'
RESOLUTION = 2048
POSITIONS = (5, 55, 58)
MODEL = 'gpt-6-astra'
OUTPUT_CAP = 4000
TOOL_CAP = 25
STEP_CAP = 200
LOOK_CAP = 40
HEAD_LIMITS = ((-1.57, 1.57), (-.76, 1.45))
HEAD_ERROR = .02
HEAD_SPEED = .03
HEAD_STABLE_READINGS = 2
INPUT_BYTE_CAP = 14336
CONTEXT_LIMIT = 1050000
GPU_UUID = 'GPU-b7ebba23-7824-7601-df32-be55628936c3'


# ========================================================================
# Frozen source: eef_visual_tools.py
"""Visual additions around immutable v13 motion. No scene or simulator imports."""
import math
import numpy as np
from .eef_tools import vector, locate_point
from .motion import FetchToolsV13



class VisualTools(FetchToolsV13):
    def look(self, pan_rad, tilt_rad):
        values = (pan_rad, tilt_rad)
        for value, (lo, hi) in zip(values, HEAD_LIMITS):
            if type(value) not in (int, float) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError('head angle outside frozen URDF limits')
        # Capture targets before settling, just as a v13 base instruction does.
        hold = vector(self.read_qpos(), 15).copy()
        target = hold.copy(); target[[4, 6]] = values
        def motion(left):
            previous = np.zeros(13); stable = 0
            while self._available():
                action = self.settling_action(target, previous)
                # Active head motion uses the official normalized range. Other
                # channels retain v13's conservative holding controller.
                q = vector(self.read_qpos(), 15)
                velocity = vector(self.read_qvel(), 15)
                desired = np.clip((np.asarray(values)-q[[4, 6]]
                    -.5*self.control_dt*velocity[[4, 6]])/.1, -1., 1.)
                action[[8, 9]] = previous[[8, 9]] + np.clip(
                    desired-previous[[8, 9]], -.25, .25)
                action[11:] = 0.
                self._step(action, 'look', reference_qpos=target.tolist())
                previous = action
                error = float(np.max(abs(vector(self.read_qpos(), 15)[[4, 6]] - values)))
                speed = float(np.max(abs(vector(self.read_qvel(), 15)[[4, 6]])))
                stable = stable + 1 if error <= HEAD_ERROR and speed < HEAD_SPEED else 0
                if self.stopped or stable >= HEAD_STABLE_READINGS: break
            error = float(np.max(abs(vector(self.read_qpos(), 15)[[4, 6]] - values)))
            arrived = stable >= HEAD_STABLE_READINGS and not self.stopped
            return {'accepted': True, 'arrived': arrived, 'head_error_rad': error,
                    'reason': 'official_terminated' if self.stopped else 'ok' if arrived else 'tool_step_limit'}
        return self._guarded('look', LOOK_CAP, motion)


class ObservationDepth:
    """Only the latest sensor frame; root changes also invalidate saved pixels."""
    def __init__(self):
        self.frame = None

    def set(self, observation_id, qpos, cameras):
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError('observation_id required')
        if set(cameras) != {'head', 'hand'}: raise ValueError('two robot cameras required')
        copied = {}
        for name, camera in cameras.items():
            depth = np.asarray(camera['depth_mm'])
            if depth.shape != (RESOLUTION, RESOLUTION) or not np.issubdtype(depth.dtype, np.integer):
                raise ValueError('raw millimeter depth resolution/type')
            copied[name] = {k: np.asarray(v).copy() for k, v in camera.items()}
        self.frame = (observation_id, vector(qpos, 15).copy(), copied)

    def locate(self, camera, pixel, observation_id, measured):
        if self.frame is None or observation_id != self.frame[0]: raise ValueError('use latest observation_id')
        if not np.array_equal(vector(measured, 15), self.frame[1]): raise ValueError('observation is stale after movement')
        if camera not in ('head', 'hand'): raise ValueError('camera must be head or hand')
        if (not isinstance(pixel, list) or len(pixel) != 2 or
                any(type(p) is not int or not 0 <= p < RESOLUTION for p in pixel)):
            raise ValueError('pixel must be integer [x,y] within 0..2047')
        frame = self.frame[2][camera]; x, y = pixel
        raw = frame['depth_mm']; depth = float(raw[y, x]) / 1000.
        result = locate_point(pixel, depth, frame['intrinsics'], frame['camera_in_base'], convention='sapien')
        area = raw[max(0, y-1):y+2, max(0, x-1):x+2]
        valid = area[area > 0].astype(float) / 1000.
        return {'accepted': True, 'readonly': True, 'reason': 'ok' if result['valid'] else 'invalid_depth',
                'valid': result['valid'], 'point_base_m': result['point_base'].tolist() if result['valid'] else None,
                'depth_m': depth if depth > 0 else None,
                'depth_spread_m': float(np.ptp(valid)) if len(valid) else None,
                'observation_id': observation_id}


# ========================================================================
# Frozen source: eef_coordinated_tools_v2.py
"""Joint-path time parameterization; base/look control remains inherited v13.
No scene geometry. FK references share the very same arm/torso joint curve.
"""
import numpy as np
PreviousVisualTools = VisualTools
from .arm import ARM_IDX, periodic_delta
EEF_COORDINATED_TOOLS_V2_VERSION = 'eef-coordinated-arm-v2'

class CoordinatedVisualToolsV2(PreviousVisualTools):
    def coupled_schedule(self, plan):
        refs=np.asarray(plan['refs'],dtype=float).copy()
        for i in range(1,len(refs)):
            refs[i]=refs[i-1]+periodic_delta(refs[i],refs[i-1])
        speeds=np.asarray(self.model['reference_speed_limits'])[:7]
        lengths=np.maximum(np.max(abs(np.diff(refs[:,ARM_IDX],axis=0))/speeds,axis=1),1e-8)
        times=np.r_[0,np.cumsum(lengths)]
        initial=float(self.previous_delta[3])
        for count in range(max(1,int(np.ceil(times[-1]))),201):
            t=np.linspace(0.,1.,count+1)
            # Shared linear path clock; A1 inverse limits below determine duration.
            ticks=t*times[-1]
            q=np.column_stack([np.interp(ticks,times,refs[:,j]) for j in range(15)])
            dq=np.diff(q[:,3]);u=(dq-self.rho[-1]*np.r_[initial,dq[:-1]])/(.1*self.beta[-1])
            arm_speed=np.max(abs(np.diff(q[:,ARM_IDX],axis=0))/speeds)
            if np.max(abs(u))<=1+1e-9 and arm_speed<=1+1e-9:break
        else:raise ValueError('joint reference exceeds frozen 200-step horizon')
        self.schedule_joint_references=q.copy()  # Diagnostic; check_path never calls this.
        return [self.fk.tcp(x) for x in q],q[:,3].copy()


# ========================================================================
# Frozen source: eef_arm_executor.py
"""Arm-only 600-step budget and one candidate change: joint-space approach.

No scene geometry. Existing coordinated-v2 control gains and execution remain
inherited. The candidate changes only reference path generation.
"""
from dataclasses import replace
import numpy as np

from .eef_tools import REST, densify, transform, vector
from .arm import ARM_IDX, CONTINUOUS, nearest_continuous, periodic_delta, pose_error
from .dynamics import QIND, grip_band
from .arm import ParametersV3

PLANNING_CAP = 600
MOVE_CAP = 150
STRAIGHT_TAIL_M = .12


class ArmCoordinatedV2(CoordinatedVisualToolsV2):
    """Fallback: v2, with only the user-authorized reference horizon extended."""
    VERSION = 'arm-coordinated-v2-600'

    def __init__(self, *args, **kwargs):
        self.schedule_precheck = kwargs.pop('schedule_precheck', False)
        self.robot_diagnostics = kwargs.pop('robot_diagnostics', False)
        if kwargs.pop('extended_attempt', False):
            if kwargs.get('max_steps') != 1000:
                raise ValueError('extended attempt requires exactly 1000 steps')
            kwargs['parameters'] = replace(kwargs.get('parameters', ParametersV3()), rest_cap=150)
            super().__init__(*args, **kwargs)
            return
        kwargs.setdefault('max_steps', 600)
        if kwargs['max_steps'] > 600:
            raise ValueError('arm attempt budget exceeds 600')
        kwargs['parameters'] = replace(kwargs.get('parameters', ParametersV3()), rest_cap=150)
        super().__init__(*args, **kwargs)

    def check_path(self, targets):
        if not self.schedule_precheck:
            return super().check_path(targets)
        import hashlib
        import json
        state = dict(qpos=vector(self.read_qpos(),15).tolist(),
            qvel=vector(self.read_qvel(),15).tolist(),
            previous_delta=vector(self.previous_delta,15).tolist(),
            steps=self.steps, max_steps=self.max_steps, control_dt=self.control_dt,
            stopped=self.stopped, last_plan_endpoint=vector(self.last_plan_endpoint,15).tolist())
        stamp = hashlib.sha256(json.dumps(state,sort_keys=True,separators=(',',':'),
                                          allow_nan=False).encode()).hexdigest()
        plan = self._plan(targets)
        result = {k:v for k,v in plan.items() if k not in ('points','refs')}
        result.update(schedule_check_state_sha256=stamp,
            schedule_check_scope='robot_state_only_no_scene_collision_check',
            schedule_reference_steps_are_physical_steps=False)
        if not plan['accepted']:
            result['schedule_feasible'] = None
            return result
        sentinel = object()
        old = getattr(self,'schedule_joint_references',sentinel)
        try:
            points, _ = self.coupled_schedule(plan)
        except ValueError as error:
            result.update(accepted=False,reason='reference_timing_infeasible',
                          schedule_feasible=False,execution_detail=str(error)[:150])
        else:
            result.update(schedule_feasible=True,schedule_reference_steps=len(points)-1,
                          schedule_horizon_steps=PLANNING_CAP)
        finally:
            if old is sentinel:
                if hasattr(self,'schedule_joint_references'):
                    del self.schedule_joint_references
            else:
                self.schedule_joint_references = old
        return result

    def set_gripper(self, command):
        if not self.robot_diagnostics:
            return super().set_gripper(command)
        before = self.steps
        tool_deadline = min(self.max_steps, before+15)
        requested_enter_closed = grip_band(command)==0 and grip_band(self.gripper)!=0
        entry = vector(self.read_qpos(),15).copy()
        result = super().set_gripper(command)
        alignment = result.get('alignment')
        if alignment is not None:
            pl = self.p.endpoint_position if self.p.close_uses_endpoint_tolerance else self.p.position_tolerance
            rl = self.p.endpoint_rotation if self.p.close_uses_endpoint_tolerance else self.p.rotation_tolerance
            result.update(close_position_error_m=float(alignment['position_error_m']),
                close_rotation_error_rad=float(alignment['rotation_error_rad']),
                close_torso_delta_m=float(alignment['torso_delta_m']),
                close_position_limit_m=float(pl),close_rotation_limit_rad=float(rl),
                close_torso_delta_limit_m=float(self.p.torso_stable),
                close_position_failed=bool(alignment['position_error_m']>pl),
                close_rotation_failed=bool(alignment['rotation_error_rad']>rl),
                close_torso_failed=bool(alignment['torso_delta_m']>=self.p.torso_stable))
        samples = [(i,t) for i,t in enumerate(self.trace)
                   if t['step']>before and t['kind']=='gripper']
        readings=0; last_delta=None; individual_delta=None; masked=False
        start_q=None; end_q=None
        for i,t in samples:
            previous = vector(self.trace[i-1]['qpos'],15) if i else entry
            current = vector(t['qpos'],15)
            if start_q is None:
                start_q=previous.copy()
            end_q=current
            last_delta=float(abs(current[-2:].sum()-previous[-2:].sum()))
            individual_delta=float(np.max(abs(current[-2:]-previous[-2:])))
            readings = readings+1 if last_delta<self.p.finger_stable else 0
            masked = masked or (last_delta<self.p.finger_stable and individual_delta>=self.p.finger_stable)
        if self.stopped:
            end_reason='official_terminated'
        elif self.steps>=self.max_steps:
            end_reason='step_budget_exhausted'
        elif self.steps>=tool_deadline:
            end_reason='tool_step_limit'
        elif not samples:
            end_reason='no_finger_motion'
        elif not result.get('enter_closed'):
            end_reason='single_step_command'
        elif readings>=2:
            end_reason='opening_stable'
        else:
            end_reason='finger_wait_limit'
        result.update(finger_motion_steps=len(samples),finger_stable_readings=readings,
            finger_wait_cap_steps=self.p.close_wait_cap if requested_enter_closed else 1,
            gripper_tool_cap_steps=15,
            finger_delta_m=last_delta,finger_individual_delta_m=individual_delta,
            finger_delta_limit_m=float(self.p.finger_stable),
            finger_stability_basis='opening_changes_only_not_object_hold',
            finger_sum_masks_individual_motion=masked,
            finger_wait_end_reason=end_reason,
            finger_hold_tcp_drift_m=None if start_q is None else float(np.linalg.norm(
                self.fk.tcp(end_q)[:3,3]-self.fk.tcp(start_q)[:3,3])),
            finger_hold_torso_drift_m=None if start_q is None else float(end_q[3]-start_q[3]))
        return result

    def coupled_schedule(self, plan):
        refs=np.asarray(plan['refs'],dtype=float).copy()
        for i in range(1,len(refs)):
            refs[i]=refs[i-1]+periodic_delta(refs[i],refs[i-1])
        speeds=np.asarray(self.model['reference_speed_limits'])[:7]
        lengths=np.maximum(np.max(abs(np.diff(refs[:,ARM_IDX],axis=0))/speeds,axis=1),1e-8)
        times=np.r_[0,np.cumsum(lengths)]
        initial=float(self.previous_delta[3])
        for count in range(max(1,int(np.ceil(times[-1]))),PLANNING_CAP+1):
            ticks=np.linspace(0.,1.,count+1)*times[-1]
            q=np.column_stack([np.interp(ticks,times,refs[:,j]) for j in range(15)])
            dq=np.diff(q[:,3]);u=(dq-self.rho[-1]*np.r_[initial,dq[:-1]])/(.1*self.beta[-1])
            arm_speed=np.max(abs(np.diff(q[:,ARM_IDX],axis=0))/speeds)
            if np.max(abs(u))<=1+1e-9 and arm_speed<=1+1e-9:break
        else:raise ValueError('joint reference exceeds 600-step horizon')
        self.schedule_joint_references=q.copy()
        return [self.fk.tcp(x) for x in q],q[:,3].copy()


def synchronized_joint_path(start, end, fk, rotation_step=.1, torso_step=.02):
    """Only continuous joints wrap. Bounded joints retain their legal interval."""
    start=vector(start,15);end=nearest_continuous(vector(end,15),start)
    delta=end-start
    count=max(1,int(np.ceil(np.max(abs(delta[ARM_IDX]))/rotation_step)),int(np.ceil(abs(delta[3])/torso_step)))
    if count>PLANNING_CAP:raise ValueError('joint path reference cap')
    refs=start+np.linspace(0.,1.,count+1)[:,None]*delta
    for name,index in zip(fk.order,range(15)):
        if name not in fk.limits:continue
        lo,hi,kind=fk.limits[name]
        if kind!='continuous' and (np.any(refs[:,index]<lo-1e-3) or np.any(refs[:,index]>hi+1e-3)):
            raise ValueError('joint path exceeds limits')
    return refs


class ArmJointApproach(ArmCoordinatedV2):
    VERSION = 'arm-joint-approach-candidate-v1'

    def endpoint_candidates(self, goal, previous):
        solutions=[];records=[]
        rest=previous.copy();rest[self.ik8.indices]=REST[self.ik8.indices]
        for seed in (previous,nearest_continuous(rest,previous)):
            raw=self.ik8.raw.solve(goal,seed)
            q=nearest_continuous(raw.qpos,previous)
            pe,re=pose_error(self.fk,q,goal)
            records.append({'accepted':bool(raw.accepted),'position_error_m':pe,'rotation_error_rad':re})
            if raw.accepted and pe<=self.p.position_tolerance and re<=self.p.rotation_tolerance:
                solutions.append(q)
        if not solutions:return None,records
        # A distant endpoint is traversed by a continuous joint path. Its total
        # displacement is not a one-step IK jump; every adjacent reference is.
        return min(solutions,key=lambda q:float(np.sum((q[QIND]-previous[QIND])**2))),records

    def _plan(self, targets):
        q=vector(self.read_qpos(),15).copy();refs=[q.copy()];points=[];records=[];tails=[]
        for raw_goal in targets:
            goal=transform(raw_goal);start=self.fk.tcp(q)
            distance=float(np.linalg.norm(goal[:3,3]-start[:3,3]))
            if distance>STRAIGHT_TAIL_M:
                endpoint,info=self.endpoint_candidates(goal,q)
                records.append({'phase':'goal_ik','candidates':info})
                if endpoint is None:return {'accepted':False,'reason':'endpoint_ik_not_found','records':records}
                entry=goal.copy();entry[:3,3]-=(goal[:3,3]-start[:3,3])*STRAIGHT_TAIL_M/distance
                pre,info=self.endpoint_candidates(entry,endpoint)
                records.append({'phase':'tail_entry_ik','candidates':info})
                if pre is None:return {'accepted':False,'reason':'tail_entry_ik_not_found','records':records}
                joint=synchronized_joint_path(q,pre,self.fk)
                for following in joint[1:]:
                    if self.ik8.jump(following,q)[0]:
                        return {'accepted':False,'reason':'joint_reference_jump','records':records}
                    q=following.copy();refs.append(q);points.append(self.fk.tcp(q))
                start=self.fk.tcp(q)
            tail_points=densify(start,[goal],self.p.translation_sample,self.p.rotation_sample)
            tails.append({'start':start[:3,3].tolist(),'goal':goal[:3,3].tolist(),
                          'length_m':float(np.linalg.norm(goal[:3,3]-start[:3,3]))})
            for point in tail_points:
                solved=self.ik8.solve(point,q);records.append(solved.summary())
                if not solved.accepted:return {'accepted':False,'reason':solved.reason,'records':records,'tails':tails}
                q=solved.qpos.copy();refs.append(q);points.append(point)
        if not points:raise ValueError('empty path')
        lower=int(np.ceil(sum(np.max(abs(b[QIND]-a[QIND]))/.1 for a,b in zip(refs,refs[1:]))))
        accepted=lower<=min(PLANNING_CAP,self.max_steps-self.steps) and not self.stopped
        return {'accepted':accepted,'reason':'ok' if accepted else 'step_budget_or_terminal',
                'records':records,'refs':np.asarray(refs),'points':points,'minimum_steps':lower,'tails':tails}



def smooth_localization_duration(start, goal, dt=.05):
    """Quintic timing with unchanged .2 m/s and .5 rad/s peak bounds."""
    from scipy.spatial.transform import Rotation
    distance=float(np.linalg.norm(goal[:3,3]-start[:3,3]))
    angle=float(Rotation.from_matrix(start[:3,:3].T@goal[:3,:3]).magnitude())
    return 1.875*max(distance/.2,angle/.5,dt)


class _SmoothLocalizationSession:
    """Task-reference adapter; delegates every tick to the unchanged Full B.

    The episode owns the extended duration, so preview and stage completion use
    one clock. No extra tick, physical step, controller reset or changed goal.
    """
    def __init__(self, session, episode):
        self.session=session;self.episode=episode;self.timed_stages=set()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self,'session'),name)

    def tick(self, inp, *, sequence, phase):
        from copy import copy
        from scipy.spatial.transform import Rotation,Slerp
        e=self.episode
        if phase!='pregrasp':return self.session.tick(inp,sequence=sequence,phase=phase)
        if e.index not in self.timed_stages:
            e.duration*=1.875
            self.timed_stages.add(e.index)
        times=np.asarray(inp.ref_t)
        u=np.clip((e.stage_steps*e.dt+times)/e.duration,0.,1.)
        alpha=u**3*(10.+u*(-15.+6.*u))
        P=(1-alpha[:,None])*e.start_local[:3,3]+alpha[:,None]*e.goal_local[:3,3]
        Rs=Slerp([0.,1.],Rotation.from_matrix(np.stack([
            e.start_local[:3,:3],e.goal_local[:3,:3]])))(alpha).as_matrix()
        # The original input carries the same current base transform implicitly.
        # Save that transform on the episode at the executor's measured boundary.
        inv=np.linalg.inv(e.reference_base)
        P=P@inv[:3,:3].T+inv[:3,3];Rs=np.einsum('ij,njk->nik',inv[:3,:3],Rs)
        pp,RR=self.model.fk(inp.q-self.ctrl.dl*inp.qd,inp.h,jac=False)
        x=copy(inp);x.ref_rel_p=P-pp;x.ref_rel_R=np.einsum('ij,njk->nik',RR.T,Rs)
        cmd,arm=self.session.tick(x,sequence=sequence,phase=phase)
        cmd.info['task_reference']=dict(timing='quintic-localization-v1',
            duration_s=e.duration,elapsed_s=e.stage_steps*e.dt)
        return cmd,arm


class _RestJointIK:
    """Nominal joint target plus the unchanged Cartesian compliance correction.

    The nominal branch is explicit, not a weak null-space preference. The old
    solver still handles the controller's offset. Bound the combined increment
    by the same robot speed limit; do not wrap the raw REST representation.
    """
    def __init__(self, solver, references):
        self.solver=solver;self.references=iter(references)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self,'solver'),name)

    def solve(self, q0, h, pt, Rt):
        nominal=next(self.references)
        corrected=self.solver.solve(nominal,h,pt,Rt)
        delta=corrected-np.asarray(q0)
        ratio=max(1.,float(np.max(abs(delta)/(self.velocity_limits*self.dt))))
        return np.asarray(q0)+delta/ratio


class _RestJointSession:
    """Rest-only reference adapter; servo, observer and contact law stay live.

    Preview and execution use this same adapter. Each tick starts from the
    integrated reference (or the original tracker's measured resync condition),
    advances toward exact REST, then adds the existing IK compliance response.
    No scene truth, threshold relaxation, controller reset or extra tick.
    """
    def __init__(self, session):self.session=session

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self,'session'),name)

    def tick(self, inp, *, sequence, phase):
        if phase!='rest':return self.session.tick(inp,sequence=sequence,phase=phase)
        from copy import copy
        from fetch_episode import ARM,REST
        ctrl=self.ctrl;solver=ctrl.ik
        q0=np.asarray(ctrl.q_ref,float).copy()
        if np.max(abs(q0-inp.q))>ctrl.cfg.ik.resync:q0=np.asarray(inp.q,float).copy()
        target=np.asarray(REST[ARM],float)
        # Half the existing reference-speed bound leaves room for compliance.
        limits=.5*np.asarray(solver.velocity_limits,float)
        delta=target-q0;duration=max(float(np.max(abs(delta)/limits)),ctrl.dtp)
        hd=(inp.h-ctrl.prev_h)/ctrl.dtp if ctrl.k>0 else 0.
        def reference(t):return q0+delta*min(1.,max(0.,t)/duration)
        times=np.asarray(inp.ref_t,float);poses=[]
        for t in times:
            # Original TrackInput's pose-delay convention is retained. The
            # arm solver receives the same predicted torso height at D+dl.
            poses.append(self.model.fk(reference(t),inp.h+hd*t,jac=False))
        pp,RR=self.model.fk(inp.q-ctrl.dl*inp.qd,inp.h,jac=False)
        x=copy(inp);x.ref_rel_p=np.asarray([v[0] for v in poses])-pp
        x.ref_rel_R=np.einsum('ij,njk->nik',RR.T,np.asarray([v[1] for v in poses]))
        references=[reference(ctrl.dtp+ctrl.dl),reference(2*ctrl.dtp+ctrl.dl)]
        ctrl.ik=_RestJointIK(solver,references)
        try:cmd,arm=self.session.tick(x,sequence=sequence,phase=phase)
        finally:ctrl.ik=solver
        cmd.info['rest_joint_reference']=dict(target=target.tolist(),nominal_next=references[0].tolist(),
            speed_limits_rad_s=limits.tolist(),remaining_nominal_s=duration,
            basis='robot_rest_raw_joint_target_with_existing_compliance')
        return cmd,arm


class ArmToolsLockedBasePlanner:
    """Transactional planar constraint, before arm/base planning; torso stays free.

    The deployed cc_obs_v5 planner uses Mink. Freeze its planar QP DOFs rather
    than merely discarding a whole-body solution's base command. ReachBand's
    optional whole-body allocation is disabled too. Preserve planner state and
    restore only the temporary constraints/configuration at the tool boundary.
    """
    def __init__(self, planner):
        from copy import deepcopy
        self.planner=planner
        self.original_config=getattr(planner,'c',None)
        config=deepcopy(self.original_config)
        if config is not None:
            if hasattr(config,'wb_gain'):config.wb_gain=0.
            # A previous planar command must not leak through a smoothing tail.
            for name,value in (('base_tau',0.),('base_acc',None),('yaw_acc',None)):
                if hasattr(config,name):setattr(config,name,value)
        self.qp=getattr(planner,'qp',None)
        self.original_constraints=None
        if self.qp is not None:
            self.original_constraints=self.qp.constraints
            frozen=self.qp.mink.DofFreezingTask(self.qp.m,dof_indices=list(self.qp.bv[:3]))
            self.qp.constraints=list(self.original_constraints)+[frozen]
        if config is not None:planner.c=config

    def restore(self):
        if self.original_config is not None:self.planner.c=self.original_config
        if self.original_constraints is not None:self.qp.constraints=self.original_constraints

    def reset(self, inp):
        reset=getattr(self.planner,'reset',None)
        if reset is not None:reset(inp)

    def plan(self, *args, **kwargs):
        _,_,vz=self.planner.plan(*args,**kwargs)
        if hasattr(self.planner,'vb_f'):self.planner.vb_f=np.zeros(2)
        if hasattr(self.planner,'w_f'):self.planner.w_f=0.
        return np.zeros(2),0.,vz


class RestProgressGuard:
    """Robot-only lack-of-response detector, not a scene-collision classifier.

    Compare an acknowledged position with the preceding command reference.
    A disturbance estimate alone is insufficient: the SAME joint must retain
    a signed tracking error while making almost no motion over five ACKs.
    These diagnostic thresholds do not change any arrival/grasp tolerance.
    """
    def __init__(self):
        self.samples=[]

    def update(self, sequence, measured_arm, diagnostic, pending=True):
        ci=diagnostic.get('controller_info') or {}
        fd=ci.get('fetch_diagnostics') or {}
        try:
            q=np.asarray(measured_arm,dtype=float)
            ref=np.asarray(fd.get('q_ref'),dtype=float)
            offset=np.asarray(ci.get('offset'),dtype=float)
        except (ValueError,TypeError):
            self.samples=[];return False
        valid=(pending and diagnostic.get('sequence')==sequence-1
            and diagnostic.get('phase')=='rest' and fd.get('sequence')==sequence-1
            and fd.get('phase')=='rest' and q.shape==ref.shape==(7,)
            and offset.shape==(3,) and np.isfinite(q).all()
            and np.isfinite(ref).all() and np.isfinite(offset).all())
        if (not valid or ci.get('contact') is not True or np.linalg.norm(offset)<=.002):
            self.samples=[];return False
        if self.samples and sequence!=self.samples[-1][0]+1:self.samples=[]
        self.samples.append((sequence,q.copy(),ref-q))
        self.samples=self.samples[-5:]
        if len(self.samples)<5:return False
        positions=np.stack([x[1] for x in self.samples])
        errors=np.stack([x[2] for x in self.samples])
        # Range, rather than endpoint displacement, rejects oscillation as a stall.
        stationary=np.ptp(positions,axis=0)<=.002
        persistent=(np.all(errors>.02,axis=0)|np.all(errors<-.02,axis=0))
        return bool(np.any(stationary & persistent))


class MotionProgressGuard:
    """Robot-only stationary window, after nominal motion; never declares arrival."""
    window_steps = 20  # one second at the existing 20 Hz clock

    def __init__(self):
        self.key=None;self.samples=[]

    def update(self,key,step,current,target,*,nominal_done,position_tolerance,rotation_tolerance):
        from scipy.spatial.transform import Rotation
        if key!=self.key:
            self.key=key;self.samples=[]
        pe=float(np.linalg.norm(current[:3,3]-target[:3,3]))
        re=float(Rotation.from_matrix(current[:3,:3].T@target[:3,:3]).magnitude())
        if not nominal_done or (pe<=position_tolerance and re<=rotation_tolerance):
            self.samples=[];return None
        self.samples.append((step,current.copy(),pe,re))
        self.samples=[s for s in self.samples if step-s[0]<=self.window_steps]
        if step-self.samples[0][0]<self.window_steps:return None
        travel=sum(float(np.linalg.norm(b[1][:3,3]-a[1][:3,3])) for a,b in zip(self.samples,self.samples[1:]))
        rotation=sum(float(Rotation.from_matrix(a[1][:3,:3].T@b[1][:3,:3]).magnitude()) for a,b in zip(self.samples,self.samples[1:]))
        # Count cumulative travel: moving/oscillating or rotating in place is not
        # this stationary failure, even if net displacement happens to be zero.
        if travel>.0005 or rotation>.002:return None
        if self.samples[0][2]-pe>.0005 or self.samples[0][3]-re>.002:return None
        return dict(arrived=False,position_error_m=pe,rotation_error_rad=re,
                    stalled_steps=self.window_steps,waypoint_index=key+1)


class MotionErrorProgressGuard:
    """Default-off target-error progress exit; no contact or evaluation inputs.

    Thresholds were selected on archived successful actions before implementation.
    A window's best error must improve over its entry error. This preserves slow,
    non-monotonic progress that an endpoint slope can incorrectly call a stall.
    These are stop thresholds, never relaxed arrival thresholds.
    """
    motion_window_steps = 60
    rest_window_steps = 20

    def __init__(self):
        self.key=None;self.samples=[]

    def update(self,key,step,current,target,*,nominal_done,position_tolerance,
               rotation_tolerance,phase='pregrasp',rest_report=None):
        from scipy.spatial.transform import Rotation
        identity=(key,phase)
        if identity!=self.key:
            self.key=identity;self.samples=[]
        if rest_report is None:
            errors=dict(position_m=float(np.linalg.norm(current[:3,3]-target[:3,3])),
                rotation_rad=float(Rotation.from_matrix(current[:3,:3].T@target[:3,:3]).magnitude()))
            limits=dict(position_m=.0001,rotation_rad=.0004)
            window=self.motion_window_steps;basis='task_pose'
            within=(errors['position_m']<=position_tolerance
                    and errors['rotation_rad']<=rotation_tolerance)
        else:
            arm=rest_report.get('arm') or {};torso=rest_report.get('torso') or {}
            errors=dict(arm_l2_rad=arm.get('end_l2_rad'),arm_linf_rad=arm.get('end_linf_rad'),
                        torso_abs_m=abs(torso['end_error_m']) if torso.get('end_error_m') is not None else None)
            limits=dict(arm_l2_rad=.001,arm_linf_rad=.001)
            window=self.rest_window_steps;basis='rest_joint_residual';within=False
        if (not nominal_done or within or any(v is None or not np.isfinite(v) for v in errors.values())):
            self.samples=[];return None
        if rest_report is not None and errors['arm_linf_rad']<=.01:
            # Reuse the existing robot-only rest-preview joint band. Once the
            # arm is there, normal torso/head homing must not look like an arm
            # stall. This does not declare official rest or alter its gates.
            limits=({'torso_abs_m':.0001} if errors['torso_abs_m']>.01 else {})
            if not limits:self.samples=[];return None
        if self.samples and step!=self.samples[-1]['step']+1:self.samples=[]
        self.samples.append(dict(step=step,errors=errors))
        self.samples=[s for s in self.samples if step-s['step']<=window]
        if step-self.samples[0]['step']<window:return None
        improvements={k:self.samples[0]['errors'][k]-min(s['errors'][k] for s in self.samples)
                      for k in errors}
        # Torso progress is reported, but cannot conceal arm residual stagnation.
        if any(improvements[k]>threshold for k,threshold in limits.items()):return None
        report=dict(version='target-error-progress-v1',phase=phase,basis=basis,
            window_start_step=self.samples[0]['step'],window_end_step=step,window_steps=window,
            errors=errors,improvements=improvements,samples=[dict(step=s['step'],errors=dict(s['errors'])) for s in self.samples],
            requires_replan=True,fresh_observation_required=True)
        result=dict(arrived=False,completed=False,requires_replan=True,reason='motion_stalled',
                    stalled_steps=window,waypoint_index=key+1,error_progress=report)
        if rest_report is None:
            result.update(position_error_m=errors['position_m'],rotation_error_rad=errors['rotation_rad'])
        return result


def fast_approach_speed(remaining, heading_error, previous, *, dt=.05):
    """Encoder-only speed schedule; no scene-clearance assertion or physics change.

    0.4 m/s maximum, 0.4 m/s2 ramp, conservative stopping-distance envelope.
    Heading loss and overshoot command zero immediately rather than continuing
    acceleration-limited travel in an unverified direction.
    """
    if not np.isfinite([remaining, heading_error, previous, dt]).all() or dt != .05:
        raise ValueError('Finite 20 Hz approach state required')
    if abs(heading_error) > .04 or remaining <= 0:
        return 0.
    desired = min(.4, 1.5 * remaining, math.sqrt(2 * .4 * max(0., remaining - .015)))
    return float(np.clip(desired, max(0., previous - .4 * dt), min(.4, previous + .4 * dt)))


class HeadArmClearance:
    """Conservative calibrated robot-link OBBs; no scene or simulator access.

    Envelopes are bounds of the installed Fetch collision STLs (URDF origins zero).
    SAT separation is a lower bound, not exact mesh distance. Only elbow/forearm/wrist versus head pairs
    are covered; proximal upper-arm boxes overlap the normal head envelope and
    are excluded rather than reported as confirmed collisions. A 15mm margin and 150ms encoder-velocity lookahead are guards,
    not a proof of collision-free dynamics or furniture clearance.
    """
    GEOMETRY = {'head_pan_link': {'bounds': [[-0.12489648163318634, -0.14300252497196198, -0.0030014384537935257], [0.19749999046325684, 0.14300252497196198, 0.11299861967563629]], 'sha256': '36362238ca8c68d5aec127130014aa5c7b0ae5613b52ebd82603f6d9b03cf426'}, 'head_tilt_link': {'bounds': [[-0.05495608597993851, -0.10499999672174454, -0.054999977350234985], [0.05496508628129959, 0.10499999672174454, 0.05500003695487976]], 'sha256': '3444f186e3faa628654ba7c9ee096d422c34d07948fd5f502918ac4781238919'}, 'upperarm_roll_link': {'bounds': [[-0.04800000041723251, -0.07050000131130219, -0.052000001072883606], [0.1850000023841858, 0.052000001072883606, 0.052000001072883606]], 'sha256': 'b52550449cfa148390cbbcb3879ff3ee67a484e98bbd2ad91046c09d49dae9d1'}, 'elbow_flex_link': {'bounds': [[-0.052000001072883606, -0.052000001072883606, -0.052000001072883606], [0.16599999368190765, 0.07050000131130219, 0.052000001072883606]], 'sha256': 'fa200b07d6ffe096eb7823b2c2154cc8ce8e22ea5516f557719b9fb308283cb1'}, 'forearm_roll_link': {'bounds': [[-0.03200000151991844, -0.08699999749660492, -0.052000001072883606], [0.17649999260902405, 0.052000001072883606, 0.052000001072883606]], 'sha256': 'a74b8d891845f56c58cdc9c4fb115b5ad1a89893cf9d5902310ea5bd7c70c81e'}, 'wrist_flex_link': {'bounds': [[-0.052000001072883606, -0.05400000140070915, -0.052000001072883606], [0.11649999767541885, 0.06700000166893005, 0.052000001072883606]], 'sha256': '435f181dea365137f63e74f718911b7dfd780af6fc92ce5c74908e7d0790413b'}, 'wrist_roll_link': {'bounds': [[-0.023000000044703484, -0.05194687098264694, -0.05196351185441017], [0.019500000402331352, 0.05194687098264694, 0.05196351185441017]], 'sha256': '942ca8736eecefd12c6d5397292c5725546b70c0826d9daf67b68d69255011d5'}}
    margin = .015

    def __init__(self, model):
        import mujoco
        self.m=model.m
        self.d=mujoco.MjData(self.m)
        self.ids={name:mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_BODY,name) for name in self.GEOMETRY}
        if min(self.ids.values())<0:raise ValueError('Missing calibrated self geometry body')
        names=['torso_lift_joint','head_pan_joint','head_tilt_joint']+list(model.spec.arm_joints)
        self.addresses=[int(self.m.jnt_qposadr[mujoco.mj_name2id(self.m,mujoco.mjtObj.mjOBJ_JOINT,n)]) for n in names]

    def separation(self,q):
        import mujoco
        q=np.asarray(q,float)
        if q.shape!=(15,) or not np.isfinite(q).all():raise ValueError('Invalid robot self-check joints')
        self.d.qpos[:]=self.m.qpos0
        self.d.qpos[self.addresses]=q[[3,4,6,5,7,8,9,10,11,12]]
        mujoco.mj_kinematics(self.m,self.d)
        boxes={}
        for name,row in self.GEOMETRY.items():
            b=np.asarray(row['bounds']);R=self.d.xmat[self.ids[name]].reshape(3,3)
            boxes[name]=(self.d.xpos[self.ids[name]]+R@b.mean(axis=0),R,(b[1]-b[0])/2)
        best=(float('inf'),None)
        for head in ('head_pan_link','head_tilt_link'):
            for arm in ('elbow_flex_link','forearm_roll_link','wrist_flex_link','wrist_roll_link'):
                c,A,a=boxes[head];d,B,b=boxes[arm]
                axes=np.concatenate((A.T,B.T,np.cross(A.T[:,None,:],B.T[None,:,:]).reshape(-1,3)))
                norms=np.linalg.norm(axes,axis=1);axes=axes[norms>1e-9]/norms[norms>1e-9,None]
                gap=float(np.max(np.abs(axes@(d-c))-np.abs(axes@A)@a-np.abs(axes@B)@b))
                if gap<best[0]:best=(gap,[head,arm])
        return best

    def segment(self,start,end):
        start=np.asarray(start);end=np.asarray(end)
        n=max(1,int(np.ceil(np.max(np.abs(end[3:13]-start[3:13]))/.02)))
        initial=self.separation(start)[0]
        for i in range(1,n+1):
            gap,pair=self.separation(start+(end-start)*i/n)
            # Permit a strictly separating escape from a conservative near envelope.
            if gap<self.margin and gap<initial-1e-6:
                return dict(accepted=False,arrived=False,reason='self_collision_risk',
                    self_collision_pair=pair,self_collision_clearance_m=gap)
        return None


# Existing base_tracking_exit lateral bound and base-arrival yaw bound.
# These are inherited robot tracking tolerances, not fitted to S1.2 outcomes.
ARM_BASE_TRANSLATION_LIMIT_M = .006
ARM_BASE_ROTATION_LIMIT_RAD = .02


def arm_base_push_feedback(entry, current, entry_step, step):
    delta=np.linalg.inv(entry)@current
    translation=float(np.linalg.norm(delta[:2,3]))
    yaw=float(math.atan2(delta[1,0],delta[0,0]))
    if translation<=ARM_BASE_TRANSLATION_LIMIT_M and abs(yaw)<=ARM_BASE_ROTATION_LIMIT_RAD:
        return None
    return dict(accepted=False,arrived=False,completed=False,reason='base_pushed',
        requires_replan=True,localization_invalidated=True,
        base_push=dict(entry_step=entry_step,first_exceeded_step=step,
            translation_m=translation,turn_rad=yaw,delta_xy_m=delta[:2,3].tolist(),
            translation_limit_m=ARM_BASE_TRANSLATION_LIMIT_M,rotation_limit_rad=ARM_BASE_ROTATION_LIMIT_RAD))


def arm_base_drift_sample(entry, current, entry_step, step):
    """Robot-odometry sample; it does not infer contact or command a correction."""
    delta=np.linalg.inv(entry)@current
    return dict(entry_step=int(entry_step),step=int(step),
        translation_m=float(np.linalg.norm(delta[:2,3])),
        turn_rad=float(math.atan2(delta[1,0],delta[0,0])),
        delta_xy_m=delta[:2,3].tolist())


class BaseMotionProgress:
    """Measured command-frame motion only; no scene/contact inputs or replanning."""
    def __init__(self, forward, turn, *, deviation_limit=.02):
        self.forward=float(forward);self.turn=float(turn);self.samples=[]
        self.max_deviation=0.;self.deviation_limit=deviation_limit

    def update(self, relative, step):
        x,y=map(float,relative[:2,3]);yaw=math.atan2(relative[1,0],relative[0,0])
        c,s=math.cos(self.turn),math.sin(self.turn)
        along=c*x+s*y;cross=-s*x+c*y
        deviation=math.hypot(x,y) if abs(self.forward)<1e-9 else abs(cross)
        self.max_deviation=max(self.max_deviation,deviation)
        angle=math.atan2(math.sin(self.turn-yaw),math.cos(self.turn-yaw))
        feedback=dict(requested_forward_m=self.forward,requested_turn_rad=self.turn,
            actual_forward_m=along,actual_turn_rad=yaw,actual_lateral_m=cross,
            base_translation_m=math.hypot(x,y),base_path_deviation_m=self.max_deviation,
            position_error_m=math.hypot(x-self.forward*c,y-self.forward*s),rotation_error_rad=abs(angle))
        reason=None
        if deviation>self.deviation_limit:reason='base_path_deviation'
        self.samples.append((step,x,y,yaw))
        self.samples=[v for v in self.samples if v[0]>=step-20]
        if self.samples and step-self.samples[0][0]>=20:
            _,a,b,t=self.samples[0]
            if (math.hypot(x-a,y-b)<.002 and abs(math.atan2(math.sin(yaw-t),math.cos(yaw-t)))<.002
                    and (abs(self.forward-along)>.015 or abs(angle)>.02)):
                reason=reason or 'base_motion_stalled'
        return feedback,reason


def clone_full_b_preview(episode):
    """Clone mutable controller state, sharing imported code modules only.

    The robot QP adapter retains its imported backend module. Modules are code
    dependencies, not controller state, and Python's plain deepcopy cannot
    pickle them. This uses the same module-identity boundary as cold restore;
    session, servo, observer, IK, solver instances and arrays remain deep copies.
    """
    import sys
    from copy import deepcopy
    from types import ModuleType
    modules={id(value):value for value in tuple(sys.modules.values()) if isinstance(value,ModuleType)}
    return deepcopy(episode,modules)


def clearance_codec_targets(qpos, action, mapping):
    """Decode the validated flowsde_full SDK contract, without changing actions.

    SDK gym_utils.clip_and_scale_action uses midpoint + half-range * action,
    with float32 bounds/actions; delta position addition is also float32.
    The raw gripper PD target is negative on close. Its geometry endpoint is
    separately clipped to the robot's own Fetch URDF finger limits [0,.05],
    represented in SDK float32. This is a kinematic envelope, not a dynamics
    prediction or a proof that actual joints cannot overshoot their limits.
    """
    from fetch_episode import ARM,HEAD
    q=np.asarray(qpos,np.float32);a=np.asarray(action,np.float32)
    if q.shape!=(15,) or a.shape!=(20,) or not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ValueError('Invalid clearance SDK codec input')
    def scaled(name,low,high):
        start,end=mapping[name]
        values=np.clip(a[start:end],np.float32(-1),np.float32(1))
        lo,hi=np.asarray(low,np.float32),np.asarray(high,np.float32)
        return np.float32(.5)*(hi+lo)+np.float32(.5)*(hi-lo)*values
    arm=scaled('arm',[-.1]*7+[-1.]*7,[.1]*7+[1.]*7)
    head=scaled('head',-.1,.1);grip=scaled('gripper',-.01,.05)
    torso=scaled('torso',-.1,.1);base=scaled('base',[-1.,-3.14],[1.,3.14])
    raw=q.copy();raw[ARM]=q[ARM]+arm[:7];raw[HEAD]=q[HEAD]+head;raw[13:15]=grip
    target=raw.copy();target[13:15]=np.clip(grip,np.float32(0),np.float32(.05))
    return dict(raw_position_target=raw.astype(float),geometry_position_target=target.astype(float),
        arm_velocity=arm[7:].astype(float),torso_velocity=float(torso[0]),base_velocity=base.astype(float),
        gripper_envelope_rule='fetch_urdf_kinematic_limits_float32_0_to_0.05')


class FullBOnlineExecutor:
    """Dynamic tools over one full B session; never constructs a simulator/provider.

    Base commands use bounded encoder odometry tracking, with full B arm hold.
    This is a declared metric-motion boundary, not native long-range navigation.
    recorder and persist are evaluation-only ports; no recorded truth is returned.
    """
    version = 'full-B-r4-task-rules-v1-online-v1'

    def __init__(self, session, ports, mapping, *, step_once, read_qpos, is_stopped,
                 recorder, persist, deadline, max_steps=600, timed=None, weekly_budget=False,
                 motion_guard=False, fast_base_approach=False, self_collision_guard=False, base_tracking_guard=False,
                 look_only_self_guard=False,
                 local_clearance_guard=False, clearance=None, refresh_clearance=None,
                 defer_clearance_binding=False, clearance_unknown_advisory=False,
                 carried_envelope_enabled=False):
        from fetch_episode import FetchEpisode, Stage
        from fetch_driver import flatten_groups, encode_flow_robot, encode_arm
        if (type(max_steps) is not int or max_steps not in ((600, 1200, 1600, 3400) if weekly_budget else (600,))
                or not np.isclose(session.ctrl.dtp, .05)):
            raise ValueError('Full B requires600 steps and20Hz')
        if recorder.action_size != 20:
            raise ValueError('Full B requires20D recording; no legacy fallback')
        # Validate the real runtime mapping before any action.
        probe=encode_flow_robot(encode_arm(np.zeros(7),np.zeros(7),np.zeros(7)),
            head_q=np.zeros(2),head_target=np.zeros(2),torso_velocity=0.,base_velocity=[0.,0.],grip_closed=False)
        flatten_groups(probe,mapping)
        self.Episode=FetchEpisode;self.Stage=Stage
        self.session=session;self.ports=ports;self.mapping=dict(mapping)
        self.step_once=step_once;self.read_qpos=read_qpos;self.is_stopped=is_stopped
        self.recorder=recorder;self.persist=persist;self.deadline=deadline
        self.timed=timed or (lambda name, fn, *a, **kw:fn(*a, **kw))
        self.max_steps=max_steps;self.steps=0;self.trace=[];self.ik_records=[]
        if type(motion_guard) is not bool:raise ValueError('Boolean motion guard required')
        self.motion_guard=motion_guard
        if look_only_self_guard and self_collision_guard:raise ValueError("Look-only guard cannot enable global self guard")
        self.look_only_self_guard=bool(look_only_self_guard)
        self.look_self_clearance=HeadArmClearance(session.model) if look_only_self_guard else None
        self.self_collision_guard=bool(self_collision_guard)
        self.self_clearance=HeadArmClearance(session.model) if self_collision_guard else None
        if type(base_tracking_guard) is not bool:raise ValueError('Boolean base tracking flag required')
        self.base_tracking_guard=base_tracking_guard
        if type(fast_base_approach) is not bool:raise ValueError("Boolean approach flag required")
        self.fast_base_approach=fast_base_approach
        self.weekly_budget=weekly_budget
        self.closed=False;self.pending=None;self.failed=False;self.ended=False
        self.last_goal_local=None;self.initial_recorded=False
        self.initial_base=np.asarray(read_qpos(),float)[:3].copy()
        if type(local_clearance_guard) is not bool:raise ValueError('Boolean local clearance flag required')
        if type(defer_clearance_binding) is not bool:raise ValueError('Boolean deferred clearance binding required')
        if defer_clearance_binding and not local_clearance_guard:
            raise ValueError('Deferred clearance binding requires the local clearance profile')
        if type(clearance_unknown_advisory) is not bool:
            raise ValueError('Boolean clearance advisory policy required')
        if clearance_unknown_advisory and not local_clearance_guard:
            raise ValueError('Clearance advisory policy requires local clearance')
        if type(carried_envelope_enabled) is not bool or (carried_envelope_enabled and not local_clearance_guard):
            raise ValueError('Estimated carried envelope requires local clearance')
        if carried_envelope_enabled:self.carried_envelope_enabled=True
        if local_clearance_guard:
            if clearance is None and not defer_clearance_binding:
                raise ValueError('Sensor clearance runtime port required')
            if refresh_clearance is not None and not callable(refresh_clearance):
                raise ValueError('Invalid sensor refresh port')
            # Runtime ports are explicitly excluded from cold snapshots. The
            # unique map owner is policy State; it is rebound after restoration.
            self.local_clearance_guard=True;self.clearance=clearance
            if clearance_unknown_advisory:
                self.clearance_unknown_advisory=True
                self.clearance_advisory_during_command=None
            self.refresh_clearance=refresh_clearance
            self.motion_goal_local=None;self.hold_reference_local=None
            self.motion_suspended=False;self.clearance_last_feedback={}
            # The opt-in factory may restore State before binding its sole
            # memory owner. This construction permission is not persisted and
            # never authorizes executing with a missing runtime checker.

    def odometry(self):
        from .motion import odometry
        return odometry(self.initial_base,self.read_qpos()[:3])

    def _state(self):
        state=dict(executor=self.version,steps=self.steps,pending_action=self.pending,
                    failed=self.failed,ended=self.ended,trace=self.trace)
        if getattr(self,'local_clearance_guard',False):
            state.update(motion_goal_local=None if self.motion_goal_local is None else self.motion_goal_local.tolist(),
                hold_reference_local=None if self.hold_reference_local is None else self.hold_reference_local.tolist(),
                motion_suspended=self.motion_suspended,clearance_last_feedback=self.clearance_last_feedback)
        if getattr(self,'rest_depth_guard',False):
            state.update(rest_depth_guard=True,rest_brake_in_progress=self.rest_brake_in_progress)
        return state

    def _flush(self):
        from copy import deepcopy
        state=self._state()
        snapshot=getattr(self.persist,"snapshot",deepcopy)
        self.persist(snapshot(state))

    def _available(self):
        if self.failed or self.pending is not None or getattr(self,'rest_brake_in_progress',False):
            raise RuntimeError('Unknown/failed full B action cannot be retried')
        if self.is_stopped():
            self.ended=True
            self.session.latch_official_end(terminated=True,truncated=False)
        return not self.ended and self.steps<self.max_steps

    def _capture(self,action,phase):
        self.recorder.record(self.steps,action,phase=phase)

    def _require_clearance_binding(self):
        checker=getattr(self,'clearance',None)
        if checker is None or any(not callable(getattr(checker,name,None))
                                  for name in ('check_base','check_joint_path')):
            raise ValueError('Sensor clearance runtime checker is not bound')
        return checker

    def _clearance_result(self, report, *, scope, allow_observation_unknown=False):
        """Bounded sensor-only public feedback; no map, contacts or scene IDs."""
        if not isinstance(report,dict) or type(report.get('accepted')) is not bool:
            raise ValueError('Invalid sensor clearance result')
        status=report.get('status',report.get('clearance_status'))
        if status=='stale':status='stale_evidence'
        if status not in ('observed_conflict','coverage_insufficient','stale_evidence','no_observed_conflict'):
            raise ValueError('Invalid sensor clearance status')
        result=dict(accepted=report['accepted'],arrived=False,
            reason='local_clearance_'+status,clearance_status=status,clearance_scope=str(scope)[:80])
        for source,target in (('body','clearance_body'),('clearance_m','clearance_distance_m'),
                              ('evidence_step','clearance_evidence_step')):
            value=report.get(source,report.get(target))
            if value is not None:
                if source=='body':
                    if not isinstance(value,str) or len(value)>80:raise ValueError('Invalid clearance body')
                elif source=='evidence_step':
                    if type(value) is not int or not 0<=value<=self.steps:raise ValueError('Future clearance evidence')
                elif not isinstance(value,(float,int)) or not np.isfinite(value):
                    raise ValueError('Invalid sensor clearance distance')
                result[target]=value
        exception=report.get('clearance_observation_exception',False)
        if type(exception) is not bool:raise ValueError('Invalid bounded observation flag')
        if exception:
            if not (allow_observation_unknown and result['accepted'] and status=='coverage_insufficient'):
                raise ValueError('Unscoped bounded observation exception')
            result['clearance_observation_exception']=True
        if not exception and result['accepted']!=(status=='no_observed_conflict'):
            raise ValueError('Inconsistent sensor clearance status')
        # Policy decision is separate from geometry classification. Keep UNKNOWN
        # visible. Missing/stale data and unavailable carried geometry are not
        # covered by this experiment; occupied checks run before coverage checks.
        if (getattr(self,'clearance_unknown_advisory',False)
                and status=='coverage_insufficient'
                and result.get('clearance_evidence_step') is not None
                and scope!='carried_geometry_unavailable'):
            result.update(accepted=True,clearance_unknown_advisory=True)
            self.clearance_advisory_during_command=result.copy()
        self.clearance_last_feedback=result.copy()
        return result

    def _clearance_refresh(self, measured):
        self._require_clearance_binding()
        callback=getattr(self,'refresh_clearance',None)
        if callback is not None:
            self.timed('clearance.sensor_refresh',callback,step=self.steps,measured=measured)

    def _attachment_candidate(self):
        """Physical attachment remains bound to its original candidate after re-identification."""
        if not getattr(self,'carried_envelope_enabled',False):return None
        self._require_clearance_binding()
        row=self.clearance.memory.attachment()
        if row is None or row['status'] not in ('estimated','possibly_attached'):return None
        return row['candidate_id']

    def _clearance_path(self, qpath, measured, *, scope, odometry=None, allow_contact=False,
                        allow_observation_unknown=False, carried_possible=None):
        self._require_clearance_binding()
        carried_possible=self.closed if carried_possible is None else carried_possible
        attachment_candidate=self._attachment_candidate() if carried_possible else None
        if carried_possible:
            from fetch_episode import ARM
            joints=np.asarray(qpath,float)
            bases=np.asarray(measured['local_from_base'] if odometry is None else odometry,float)
            if bases.shape==(4,4):bases=np.repeat(bases[None],len(joints),axis=0)
            if (joints.ndim!=2 or joints.shape[1]!=15 or not len(joints)
                    or bases.shape!=(len(joints),4,4) or not np.isfinite(joints).all()
                    or not np.isfinite(bases).all()):
                raise ValueError('Invalid carried attachment preview')
            # Missing estimates retain the historical rejection. With an
            # estimate, the checker sweeps the additional body, not bare links.
            # Certify stationary attachment only from unchanged generating
            # coordinates. Endpoint FK equality cannot rule out an intervening
            # loop; calling live controller FK would also mutate its workcache.
            moving=(any(not np.array_equal(B,bases[0]) for B in bases[1:])
                    or any(not np.array_equal(q[ARM],joints[0,ARM]) or q[3]!=joints[0,3]
                           for q in joints[1:]))
            if moving and attachment_candidate is None:
                return self._clearance_result(dict(accepted=False,clearance_status='coverage_insufficient',
                    clearance_body='carried_object'),scope='carried_geometry_unavailable')
        attachment_kw={} if attachment_candidate is None else dict(attachment_candidate=attachment_candidate)
        if allow_contact and attachment_candidate is None and getattr(self,'carried_envelope_enabled',False):
            contact=self.clearance.memory.contact_attachment()
            if contact is not None:attachment_kw['target_contact_candidate']=contact['candidate_id']
        report=self.timed('clearance.joint_path',self.clearance.check_joint_path,
            qpath,measured['local_from_base'] if odometry is None else odometry,self.steps,scope=scope,
            contact_region=getattr(self.clearance,'contact_region',None) if allow_contact else None,
            allow_observation_unknown=allow_observation_unknown,**attachment_kw)
        return self._clearance_result(report,scope=scope,allow_observation_unknown=allow_observation_unknown)

    def _clearance_preflight(self, stages, measured, *, motion=None, alignment_goal=None, local_goal=None,
                             allow_contact=False,allow_observation_unknown=False):
        """Nominal command path, followed by a separate check of each proposed tick."""
        self._require_clearance_binding()
        from copy import deepcopy
        from fetch_episode import ARM,HEAD,REST
        q=np.asarray(measured['qpos'],float).copy();B=measured['local_from_base']
        self._clearance_refresh(measured)
        transport=(motion is not None and motion['kind']=='base' and
                   (motion['turn']!=0 or motion['forward']!=0)) or any(
                       s.phase=='rest' or s.target is not None for s in stages) or (
                           alignment_goal is not None or local_goal is not None)
        attachment_candidate=self._attachment_candidate() if self.closed else None
        if self.closed and transport and attachment_candidate is None:
            return self._clearance_result(dict(accepted=False,clearance_status='coverage_insufficient',
                clearance_body='carried_object'),scope='carried_geometry_unavailable')
        if motion is not None and motion['kind']=='base':
            attachment_kw={} if attachment_candidate is None else dict(attachment_candidate=attachment_candidate)
            report=self.timed('clearance.base_path',self.clearance.check_base,q,B,
                motion['turn'],motion['forward'],self.steps,**attachment_kw)
            return self._clearance_result(report,scope='command_base')
        path=[q.copy()]
        if motion is not None and motion['kind']=='look':
            q[HEAD]=motion['target'];path.append(q.copy())
        else:
            ik=None
            for stage in stages:
                if stage.phase=='rest':
                    q[3:13]=REST[3:13];path.append(q.copy())
                if stage.phase in ('open','close'):
                    q[13:15]=float(np.float32(.05)) if stage.phase=='open' else 0.;path.append(q.copy())
                target=stage.target
                if target is not None or (stage.phase=='hold' and alignment_goal is not None):
                    if local_goal is not None:target=np.linalg.inv(B)@local_goal
                    elif target is None:target=np.linalg.inv(B)@alignment_goal
                    if ik is None:
                        ik=deepcopy(self.session.ctrl.ik)
                        if ik.q_post is None:ik.q_post=q[ARM].copy()
                    # Keep all intermediate private IK iterates. They are a
                    # conservative nominal path, never the executed trajectory.
                    for _ in range(80):
                        self.deadline();q[ARM]=ik.solve(q[ARM],q[3],target[:3,3],target[:3,:3]);path.append(q.copy())
        return self._clearance_path(path,measured,scope='command_joint',allow_contact=allow_contact,
                                    allow_observation_unknown=allow_observation_unknown)

    def suspend_motion_goal(self, measured, reason):
        """Re-anchor task references only; no action, observer reset or new grasp."""
        if not getattr(self,'local_clearance_guard',False):return
        from fetch_episode import ARM,pose
        q=np.asarray(measured['qpos'],float);B=measured['local_from_base']
        p,R=self.session.model.fk(q[ARM],q[3],jac=False)
        self.hold_reference_local=B@pose(p,R)
        self.last_goal_local=self.hold_reference_local.copy()
        self.motion_suspended=True
        self.clearance_last_feedback=dict(self.clearance_last_feedback,motion_suspended=True,
                                         clearance_stop_reason=str(reason)[:100])
        # q_ref and q_post are the remaining command references. Learned force,
        # predicted previous actuation, observer history and sequence stay intact.
        if hasattr(self.session.ctrl,'q_ref'):self.session.ctrl.q_ref=q[ARM].copy()
        ik=getattr(self.session.ctrl,'ik',None)
        if ik is not None:ik.q_post=q[ARM].copy()

    def _motion_action(self, packet, measured, start, motion):
        """Apply the existing metric command to a packet; no physical dispatch."""
        from fetch_episode import ARM,HEAD
        action=np.asarray(packet['action'],dtype=np.float32).copy();done=False
        q=measured['qpos'];B=measured['local_from_base']
        if (getattr(self,'carried_envelope_enabled',False) and self.closed
                and motion is None and packet.get('phase') in ('pregrasp','lift','rest')):
            # Diagnostic carried transport uses the Full B position correction
            # without an additional arm velocity drive. SDK position and
            # velocity are simultaneous PD targets, not consecutive waypoints.
            # Keep the actual packet and preview identical; all geometry and
            # measured-momentum checks still apply to this dispatched action.
            a,b=self.mapping['arm'];action[a+7:b]=0.
        if (getattr(self,'carried_envelope_enabled',False)
                and (packet.get('phase')=='close' or
                     (self.closed and packet.get('phase') in ('settling','hold')))):
            # Closure requests finger motion while the aligned arm holds pose.
            # Keep its position correction, but do not carry a fresh arm
            # velocity feedforward into the support surface. This adapter is
            # shared by clone preview and dispatch; contact checks stay active.
            a,b=self.mapping['arm'];action[a+7:b]=0.
            for part in ('base','head','torso'):
                a,b=self.mapping[part];action[a:b]=0.
        if motion is not None:
            if getattr(self,'local_clearance_guard',False) and not self.closed:
                # Look and base motion do not request a new gripper aperture.
                # The shared actuator holds the encoded measured mean; do not
                # inherit Full B's default fully-open command while travelling.
                lo,hi=np.float32(-.01),np.float32(.05)
                center=np.float32(.5)*(hi+lo);scale=np.float32(.5)*(hi-lo)
                aperture=np.float32(np.mean(np.asarray(q[13:15],np.float32)))
                a,b=self.mapping['gripper']
                action[a:b]=np.clip((aperture-center)/scale,-1.,1.)
            if motion['kind']=='look':
                target=np.asarray(motion['target'],float)
                if getattr(self,'s11_system_fixes',False):
                    hold=np.asarray(motion['hold_qpos'],float)
                    a,b=self.mapping['arm'];action[a:a+7]=np.clip((hold[ARM]-q[ARM])/.1,-1,1);action[a+7:b]=0.
                    a,b=self.mapping['torso'];action[a:b]=np.clip((hold[3]-q[3])/.1,-1,1)
                    if not (getattr(self,'look_only_self_guard',False) and self.closed):
                        a,b=self.mapping['gripper'];action[a:b]=np.clip((float(np.mean(hold[13:15]))-.02)/.03,-1,1)
                    # Closed S1.2 look retains the packet's closing drive; measured
                    # aperture under load is not a replacement grip command.
                    # Base zero velocity holds the measured pose; no route or old base command is inherited.

                if getattr(self,'local_clearance_guard',False):
                    # A sensing command owns the head only. Full B's default
                    # packet opens the fingers and carries arm feedforward even
                    # during hold; neither is part of this observation request.
                    # Apply the same override to preview and dispatched packets.
                    a,b=self.mapping['arm'];action[a:b]=0.
                    a,b=self.mapping['torso'];action[a:b]=0.
                a,b=self.mapping['head'];action[a:b]=np.clip((target-q[HEAD])/.1,-1,1)
                a,b=self.mapping['base'];action[a:b]=0
                done=bool(np.max(abs(target-q[HEAD]))<.02)
            else:
                relative=np.linalg.inv(start)@B
                yaw=math.atan2(relative[1,0],relative[0,0])
                error=math.atan2(math.sin(motion['turn']-yaw),math.cos(motion['turn']-yaw))
                forward=np.array([math.cos(motion['turn']),math.sin(motion['turn'])])
                distance=float(relative[:2,3]@forward);remaining=motion['forward']-distance
                turn=float(np.clip(2*error,-.6,.6))
                speed=0. if abs(error)>.04 else float(np.clip(1.5*remaining,-.25,.25))
                if motion.get('speed_mode')=='fast':
                    speed=fast_approach_speed(remaining,error,motion.get('previous_speed',0.))
                    motion['previous_speed']=speed
                a,b=self.mapping['base'];action[a:b]=[speed,turn/3.14]
                done=abs(error)<.02 and abs(remaining)<.015
        if getattr(self,'arm_tools_lock_base',False) and motion is None:
            a,b=self.mapping['base']
            if np.any(action[a:b]!=0):raise RuntimeError('Locked planner emitted base drive')
        return action,done

    def _reconcile_arm_action(self, episode, packet, measured, action):
        """Reconcile task-layer arm overrides after the frozen encoder tick.

        The frozen session already accounts for its encoded packet. Apply only
        the difference to the dispatched targets; do not reset learned forces.
        Preview callers pass their private episode clone, never the live session.
        """
        from fetch_episode import ARM
        from fetch_driver import decode_arm
        from copy import deepcopy
        a,b=self.mapping['arm']
        original=np.asarray(packet['action'],np.float32)[a:b]
        if np.array_equal(original,np.asarray(action,np.float32)[a:b]):return
        session=episode.session
        if isinstance(session,_SmoothLocalizationSession):session=session.session
        servo=session.ctrl.servo
        q=np.asarray(measured['qpos'],float)[ARM]
        qd=np.asarray(measured['qvel'],float)[ARM]
        old_p,old_v=decode_arm(q,original)
        decoded=clearance_codec_targets(measured['qpos'],action,self.mapping)
        new_p=decoded['raw_position_target'][ARM];new_v=decoded['arm_velocity']
        delta=servo.KP*(new_p-old_p)+servo.KD*(new_v-old_v)
        servo.pred+=servo.SG@delta
        servo.last_u+=delta
        torque=servo.drive_torque_preview(q,qd,new_p,new_v)
        obs=session.ctrl.obs
        fraction=obs.c.sat_frac if obs is not None else None
        fraction=.9 if fraction is None else fraction
        near=np.any(abs(torque)>fraction*np.asarray(session.spec.inner.tau_max,float),axis=0)
        session.drive_interval=dict(nominal_drive_torque=torque.tolist(),near_limit=near.tolist(),
            linear_dob_invalid=bool(np.any(near)),threshold_fraction=float(fraction),measured=False)
        if obs is not None:obs.sat_next=near.copy()
        diagnostics=packet.get('controller_info',{}).get('fetch_diagnostics')
        if diagnostics is not None:
            diagnostics.update(predicted_state=servo.pred.copy(),last_u=servo.last_u.copy(),
                drive_interval=deepcopy(session.drive_interval),
                task_adapter_applied_position=new_p.copy(),task_adapter_applied_velocity=new_v.copy())

    def _rest_depth_preview(self, episode, measured, start):
        """Private clone only; normal live ticks retain their original bytes."""
        from fetch_episode import ARM
        clone=clone_full_b_preview(episode)
        packet=clone.advance(sequence=self.steps,**measured,terminated=False,truncated=False)
        if packet is None or packet['phase']!='rest':return None,None
        action,_=self._motion_action(packet,measured,start,None)
        q=measured['qpos'];decoded=clearance_codec_targets(q,action,self.mapping)
        target=decoded['geometry_position_target'].copy()
        target[3]=q[3]+decoded['torso_velocity']*.15
        future=target.copy();future[ARM]+=.15*decoded['arm_velocity']
        report=self.rest_depth_checker.check_observed_arm_path(
            [q,target,future],measured['local_from_base'],self.steps)
        return report,action

    def _rest_brake_action(self, measured, hold, grip):
        from fetch_episode import ARM,HEAD
        action=np.zeros(20,dtype=np.float32);q=measured['qpos']
        a,b=self.mapping['arm'];action[a:a+7]=(hold[ARM]-q[ARM])/.1
        a,b=self.mapping['head'];action[a:b]=(hold[HEAD]-q[HEAD])/.1
        a,b=self.mapping['gripper'];action[a:b]=grip
        if not np.isfinite(action).all() or np.max(abs(action))>1:
            raise ValueError('Brake target outside unchanged codec bounds')
        return action

    def _clearance_tick(self, episode, measured, start, motion, *, allow_contact=False,
                        allow_observation_unknown=False):
        """Preview a clone, then check the actual codec targets before live tick."""
        self._require_clearance_binding()
        from copy import deepcopy
        from fetch_episode import ARM,HEAD
        clone=self.timed('clearance.clone_controller',clone_full_b_preview,episode)
        packet=self.timed('clearance.preview_controller',clone.advance,sequence=self.steps,**measured,
                          terminated=False,truncated=False)
        if packet is None:return None,None
        action,_=self._motion_action(packet,measured,start,deepcopy(motion))
        self._reconcile_arm_action(clone,packet,measured,action)
        q=np.asarray(measured['qpos'],float).copy();horizon=.15
        decoded=clearance_codec_targets(q,action,self.mapping)
        target=decoded['geometry_position_target'].copy();velocity=decoded['arm_velocity']
        target[3]=q[3]+decoded['torso_velocity']*horizon
        self.clearance_last_codec={key:(value.tolist() if isinstance(value,np.ndarray) else value)
                                   for key,value in decoded.items()}
        self.clearance_last_codec['phase']=packet.get('phase')
        future=target.copy();future[ARM]+=horizon*velocity
        speed,yaw=decoded['base_velocity']
        odometry=[]
        for seconds in (0.,horizon/2,horizon):
            theta=float(yaw*seconds);T=np.eye(4)
            T[:2,:2]=[[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]]
            T[0,3]=float(speed*seconds) if abs(yaw)<1e-9 else float(speed/yaw*math.sin(theta))
            T[1,3]=0. if abs(yaw)<1e-9 else float(speed/yaw*(1-math.cos(theta)))
            odometry.append(measured['local_from_base']@T)
        # Joint and base proposals are checked together, not only at independent
        # endpoints. These are codec target envelopes, not a physics prediction.
        report=self._clearance_path([q,target,future],measured,scope='proposed_tick',odometry=odometry,
                                    allow_contact=allow_contact,allow_observation_unknown=allow_observation_unknown,
                                    carried_possible=self.closed or episode.closed or
                                        bool(decoded['raw_position_target'][13]<=0))
        if not report['accepted']:return report,action
        return None,action

    def _arm_base_proposal(self, episode, measured, *, sequence=None):
        """Private planning transaction; never advance live observers on refusal."""
        preview=clone_full_b_preview(episode)
        packet=preview.advance(sequence=self.steps if sequence is None else sequence,**measured,terminated=False,truncated=False)
        if packet is None:return None,None
        action=np.asarray(packet['action'],dtype=np.float32)
        a,b=self.mapping['base'];base=action[a:b]
        if base.shape!=(2,) or not np.isfinite(base).all():raise ValueError('Invalid base proposal')
        # This is an actuator permission, not an arrival/no-progress tolerance.
        # Even a small nonzero drive must be explicitly planned by the caller.
        if np.any(base!=0):
            return dict(accepted=False,arrived=False,reason='base_motion_required',
                        base_motion_required=dict(proposed_action=base.tolist(),
                            phase=packet['phase'],waypoint_index=sum(s.target is not None
                                for s in preview.stages[:packet['stage']+1]),
                            action_dispatched=False)),action
        return None,action

    def _locked_base_reach(self, goals, *, strict=False):
        """Private fixed-base IK; suggestions are robot geometry, not a safe route."""
        report=self.check_motion_waypoints(goals,position_tolerance=.002 if strict else .01,
                                           rotation_tolerance=.02 if strict else .05)
        if report['accepted']:return None
        if report['reason']!='waypoint_ik_unresolved':return report
        T=np.asarray(goals[report['waypoint_index']-1],float)
        model=self.session.model
        wrist=T[:3,3]+T[:3,:3]@model.wrist_off
        rel=wrist[:2]-model.shoulder[:2];radius=float(np.linalg.norm(rel))
        bearing=float(math.atan2(rel[1],rel[0]))
        # A proposed heading and radial displacement only. The model must inspect
        # RGB-D and explicitly command move_base; no scene filtering or dispatch.
        forward=float(np.clip(radius-.575,-1.5,1.5))
        return dict(accepted=False,arrived=False,readonly=True,steps_used=0,
            reason='needs_base_reposition',base_reposition=dict(
                forward_m=forward,turn_rad=bearing,waypoint_index=report['waypoint_index'],
                position_error_m=report['position_error_m'],rotation_error_rad=report['rotation_error_rad'],
                basis='fixed_base_ik_unresolved_not_proof_of_unreachability',
                scene_clearance='unknown',action_dispatched=False))

    def _run(self, stages, *, motion=None, tool_cap=None, alignment_goal=None, local_goal=None):
        if not getattr(self,'arm_tools_lock_base',False) or motion is not None:
            return self._run_unlocked(stages,motion=motion,tool_cap=tool_cap,
                                      alignment_goal=alignment_goal,local_goal=local_goal)
        measured=self.ports.observe();B=np.asarray(measured['local_from_base'])
        goals=[np.asarray(s.target) for s in stages if s.target is not None]
        if alignment_goal is not None:goals=[np.linalg.inv(B)@alignment_goal]
        elif local_goal is not None:goals=[np.linalg.inv(B)@local_goal]
        if goals:
            refusal=self._locked_base_reach(goals,strict=alignment_goal is not None or local_goal is not None)
            if refusal is not None:return refusal
        planner=self.session.ctrl.base
        locked=ArmToolsLockedBasePlanner(planner)
        self.session.ctrl.base=locked
        try:
            return self._run_unlocked(stages,motion=motion,tool_cap=tool_cap,
                                      alignment_goal=alignment_goal,local_goal=local_goal)
        finally:
            locked.restore()
            self.session.ctrl.base=planner

    def _run_unlocked(self, stages, *, motion=None, tool_cap=None, alignment_goal=None, local_goal=None):
        from fetch_episode import pose, ARM, HEAD
        before=self.steps
        if not self._available():return dict(accepted=False,reason='official_end_or_step_limit',steps_used=0)
        cap=sum(stage.cap for stage in stages) if tool_cap is None else tool_cap
        if type(cap) is not int or cap<1:raise ValueError('Invalid tool budget')
        plan_ceiling=getattr(self,'_s2_plan_step_ceiling',None)
        if plan_ceiling is not None:
            if type(plan_ceiling) is not int:raise ValueError('Invalid S2 plan step ceiling')
            cap=min(cap,plan_ceiling-self.steps)
            if cap<=0:return dict(accepted=False,reason='plan_step_limit',steps_used=0,
                steps_remaining=self.max_steps-self.steps)
        local_guard=getattr(self,'local_clearance_guard',False)
        if getattr(self,'clearance_unknown_advisory',False):self.clearance_advisory_during_command=None
        if local_guard:self._require_clearance_binding()
        allow_contact=(motion is None and (local_goal is not None or alignment_goal is not None
                        or any(s.phase=='close' for s in stages)
                        or (self.closed and getattr(self,'grasp_goal_local',None) is not None
                            and all(s.phase=='hold' for s in stages))))
        first_measured=None
        allow_observation_unknown=False;observation_exception_used=False
        if local_guard:
            first_measured=self.timed('controller.observe',self.ports.observe)
            allow_observation_unknown=(motion is not None and motion['kind']=='look' and
                np.max(np.abs(np.asarray(motion['target'])-first_measured['qpos'][HEAD]))<=.1+1e-12)
            checked=self._clearance_preflight(stages,first_measured,motion=motion,
                alignment_goal=alignment_goal,local_goal=local_goal,allow_contact=allow_contact,
                allow_observation_unknown=allow_observation_unknown)
            observation_exception_used=checked.get('clearance_observation_exception',False)
            if not checked['accepted']:
                self.suspend_motion_goal(first_measured,checked['reason']);self._flush()
                return dict(checked,motion_suspended=True,steps_used=0,steps_remaining=self.max_steps-self.steps)
        episode=self.Episode(self.session,stages,self.mapping,max_steps=min(cap,self.max_steps-self.steps))
        if getattr(self,'rest_joint_target',False) and any(s.phase=='rest' for s in stages):
            episode.session=_RestJointSession(episode.session)
        if local_goal is not None and getattr(self,'smooth_localization',False):
            episode.session=_SmoothLocalizationSession(self.session,episode)
        episode.closed=self.closed
        if getattr(self,'_g1_previous_q',None) is not None:
            # Actual preceding ACK, never a synthetic stable observation.
            episode.last_q=self._g1_previous_q.copy()
        episode.goal_local=None if self.last_goal_local is None else self.last_goal_local.copy()
        if local_guard and self.motion_suspended and self.hold_reference_local is not None:
            episode.goal_local=self.hold_reference_local.copy()
        if alignment_goal is not None:
            episode.goal_local=alignment_goal.copy()
        if local_guard:
            intended=local_goal if local_goal is not None else alignment_goal
            if intended is not None:self.motion_goal_local=intended.copy()
            elif any(s.target is not None for s in stages):
                self.motion_goal_local=first_measured['local_from_base']@next(s.target for s in stages if s.target is not None)
        alignment_previous=None;alignment_feedback={};blocked_count=0;compliance_blocked=False
        error_progress_enabled=getattr(self,'error_progress_guard',False) and motion is None
        rest_progress=RestProgressGuard() if getattr(self,'rest_obstruction_guard',False) else None
        rest_feedback_enabled=(getattr(self,'rest_progress_feedback',False) or error_progress_enabled) and any(s.phase=='rest' for s in stages)
        rest_samples=[]
        # C1 observes acknowledged proprioception before another command. It
        # does not infer object identity or consume evaluation/contact state.
        grip_monitor = (getattr(self,'r2b_grip_monitor',False) and self.closed
                        and not any(s.phase in ('open','close') for s in stages))
        grip_samples=[]
        def sample_rest(measured):
            # Reuse observations already required by the executor. A dispatched
            # action with unknown outcome must never become a feedback endpoint.
            if rest_feedback_enabled and self.pending is None:
                sample=dict(step=self.steps,qpos=np.asarray(measured['qpos'],float).copy(),
                            qvel=np.asarray(measured.get('qvel'),float).copy())
                if not rest_samples:rest_samples.append(sample)
                elif len(rest_samples)==1:rest_samples.append(sample)
                else:rest_samples[-1]=sample
        start=None;terminal=False;reason=None
        progress_only = (getattr(self,"motion_progress_only",False) and motion is None
                         and alignment_goal is None and any(s.target is not None for s in stages))
        progress=(MotionErrorProgressGuard() if error_progress_enabled else
                  MotionProgressGuard() if self.motion_guard or progress_only else None)
        waypoint_stages=[i for i,s in enumerate(stages) if s.target is not None]
        active_waypoint_stage=None
        base_progress=(BaseMotionProgress(motion['forward'],motion['turn'],
                deviation_limit=.006 if getattr(self,'base_tracking_exit',False) else .02)
            if (self.base_tracking_guard or getattr(self,'base_tracking_exit',False)) and motion is not None and motion['kind']=='base' else None)
        base_feedback={}
        arm_entry=None
        if motion is None and getattr(self,'arm_base_push_exit',False) and getattr(self,'arm_tools_lock_base',False):
            arm_entry=getattr(self,'_arm_tool_entry',None)
            if arm_entry is None:
                arm_entry=(self.ports.observe()['local_from_base'].copy(),self.steps)
        report_rest_push=(getattr(self,'rest_base_push_report_only',False)
            and getattr(self,'_arm_tool_name',None)=='return_to_rest')
        reported_push=getattr(self,'_rest_base_push_report',None) if report_rest_push else None
        stalled=None;rest_brake=None;rest_brake_stable=0
        try:
            self.deadline()
            if not self.initial_recorded:
                self._capture(None,'tool_loop_initial');self.initial_recorded=True
            while True:
                boundary_expired=getattr(self,'action_budget_expired',None)
                if (boundary_expired is not None and boundary_expired()
                        and not terminal and not self.is_stopped()):
                    # Finish only the measured ACK of the last dispatched tick.
                    # The controller must not produce another action. Unknown
                    # dispatches still go through the fail-closed exception path.
                    if self.pending is not None:
                        raise RuntimeError('Cannot checkpoint an unknown action')
                    measured=self.timed('controller.observe',self.ports.observe)
                    sample_rest(measured)
                    episode.max_steps=episode.steps+int(episode.pending)
                    packet=self.timed('controller.advance',episode.advance,
                        sequence=self.steps,**measured,terminated=False,truncated=False)
                    if packet is not None:
                        raise RuntimeError('Deadline boundary produced an action')
                    self.closed=episode.closed
                    self.last_goal_local=episode.goal_local
                    reason='wall_time_limit'
                    break
                self.deadline()
                measured=first_measured if first_measured is not None else self.timed('controller.observe',self.ports.observe)
                first_measured=None
                sample_rest(measured)
                if local_guard and not terminal and not self.is_stopped():self._clearance_refresh(measured)
                if start is None:
                    start=measured['local_from_base'].copy()
                    if local_goal is not None:
                        episode.stages=[self.Stage(s.phase,s.cap,np.linalg.inv(start)@local_goal,'episode_base')
                                        if s.phase=='pregrasp' else s for s in episode.stages]
                q=measured['qpos'];B=measured['local_from_base']
                if arm_entry is not None and not terminal and not self.is_stopped():
                    if report_rest_push:
                        sample=arm_base_drift_sample(arm_entry[0],B,arm_entry[1],self.steps)
                        samples=self.rest_base_push_samples
                        if not samples or samples[-1]['step']!=sample['step']:samples.append(sample)
                    pushed=arm_base_push_feedback(arm_entry[0],B,arm_entry[1],self.steps)
                    if pushed is not None and report_rest_push:
                        row=pushed['base_push']
                        if reported_push is None:
                            reported_push=row.copy()
                        else:
                            score=lambda r:max(r['translation_m']/ARM_BASE_TRANSLATION_LIMIT_M,
                                abs(r['turn_rad'])/ARM_BASE_ROTATION_LIMIT_RAD)
                            if score(row)>score(reported_push):
                                first=reported_push['first_exceeded_step']
                                reported_push=row.copy();reported_push['first_exceeded_step']=first
                        self._rest_base_push_report=reported_push.copy()
                        pushed=None
                    if pushed is not None:
                        if episode.pending:
                            episode.max_steps=episode.steps+1
                            if episode.advance(sequence=self.steps,**measured,terminated=False,truncated=False) is not None:
                                raise RuntimeError('Base push stop produced an action')
                        self.closed=episode.closed
                        p_hold,R_hold=self.session.model.fk(q[ARM],q[3],jac=False)
                        self.last_goal_local=B@pose(p_hold,R_hold)
                        self.grasp_ready=False
                        self.r2a_blocked_grasp=None
                        stalled=pushed;reason='base_pushed'
                        break
                if grip_monitor:
                    gap=float(np.sum(q[13:15]))
                    if not np.isfinite(gap):raise ValueError('Nonfinite finger distance')
                    grip_samples.append(dict(step=self.steps,distance_m=gap))
                    if (gap < .003 or grip_samples[0]['distance_m']-gap > .015):
                        if not terminal and not self.is_stopped():
                            # Consume the last ACK, never dispatch an extra tick.
                            if episode.pending:
                                episode.max_steps=episode.steps+1
                                packet=episode.advance(sequence=self.steps,**measured,
                                                       terminated=False,truncated=False)
                                if packet is not None:raise RuntimeError('Grip stop produced an action')
                            self.closed=episode.closed
                            p_hold,R_hold=self.session.model.fk(q[ARM],q[3],jac=False)
                            self.last_goal_local=B@pose(p_hold,R_hold)
                            stalled=dict(accepted=False,arrived=False,reason='grip_lost')
                            reason='grip_lost'
                            break
                if base_progress is not None:
                    base_feedback,base_reason=base_progress.update(np.linalg.inv(start)@B,self.steps)
                    if base_reason and not terminal and not self.is_stopped():
                        stalled=dict(accepted=False,arrived=False,reason=base_reason)
                        episode.stages=[self.Stage('hold',episode.stage_steps+int(episode.pending))]
                        episode.index=0
                episode.reference_base=B
                if (local_goal is not None and episode.entered and
                        episode.stages[episode.index].phase=='pregrasp' and
                        (episode.stage_steps+int(episode.pending))*episode.dt>=episode.duration):
                    # End the nominal trajectory here; only the separate strict
                    # convergence stage may declare the grasp localization ready.
                    episode.stages[episode.index]=self.Stage('hold',episode.stage_steps+int(episode.pending))
                if alignment_goal is not None:
                    from scipy.spatial.transform import Rotation
                    p,R=self.session.model.fk(q[ARM],q[3],jac=False)
                    target=np.linalg.inv(B)@alignment_goal
                    pe=float(np.linalg.norm(p-target[:3,3]))
                    re=float(Rotation.from_matrix(R.T@target[:3,:3]).magnitude())
                    stable=alignment_previous is not None and abs(q[3]-alignment_previous)<.001
                    alignment_feedback=dict(position_error_m=pe,rotation_error_rad=re,stable=bool(stable))
                    if (getattr(self,'grasp_direction_feedback',False)
                            or getattr(self,'blocked_grasp_direction',False)):
                        # Command minus measured robot FK, not an object/contact estimate.
                        alignment_feedback['grasp_error_base_m']=(target[:3,3]-p).tolist()
                    self.alignment_samples.append(dict(step=self.steps,**alignment_feedback,
                        position_error_base_m=(target[:3,3]-p).tolist(),
                        target_base=target.tolist(),tcp_base=pose(p,R).tolist(),
                        measured_qpos=q.tolist()))
                    # Only the controller's own estimate is used here, never
                    # evaluation-side scene contact data. Count acknowledged holds.
                    history=getattr(self,'controller_diagnostics',[])
                    last=history[-1] if history else {}
                    ci=last.get('controller_info') or {}
                    offset=np.asarray(ci.get('offset',[0.,0.,0.]),float)
                    obstructed=(last.get('sequence')==self.steps-1 and last.get('phase')=='hold'
                        and ci.get('contact') is True and offset.shape==(3,)
                        and np.isfinite(offset).all() and np.linalg.norm(offset)>.002
                        and (pe>.002 or re>.02))
                    blocked_count=blocked_count+1 if obstructed else 0
                    compliance_blocked=blocked_count>=5
                    if compliance_blocked:
                        # Consume the pending acknowledgement without another tick.
                        episode.stages=[self.Stage('hold',self.steps-before)]
                    if pe<=.002 and re<=.02 and stable:
                        # Acknowledge the last action through the frozen episode before
                        # ending hold. No unused controller prediction/tick is generated.
                        episode.stages=[self.Stage('hold',self.steps-before)]
                    alignment_previous=float(q[3])
                if (getattr(self,'rest_obstruction_guard',False) and episode.entered
                        and episode.stages[episode.index].phase=='rest'
                        and not terminal and not self.is_stopped()):
                    # Compare this ACK with its preceding reference, not with the
                    # next tick's reference. Scene contact identities stay private.
                    history=getattr(self,'controller_diagnostics',[])
                    last=history[-1] if history else {}
                    if rest_progress.update(self.steps,q[ARM],last,pending=episode.pending):
                        stalled=dict(accepted=False,arrived=False,reason='compliance_blocked')
                        episode.stages=[self.Stage('hold',episode.stage_steps+int(episode.pending))]
                        episode.index=0
                if motion is not None and motion['kind']=='base':
                    # Base-only command holds the measured entry TCP in moving-base coordinates.
                    if 'tcp' not in motion:
                        p,R=self.session.model.fk(q[ARM],q[3],jac=False);motion['tcp']=pose(p,R)
                    episode.goal_local=B@motion['tcp']
                if (progress is not None and episode.entered and motion is None
                        and not terminal and not self.is_stopped()):
                    stage=episode.stages[episode.index]
                    if (stage.phase in ('pregrasp','descend','lift')
                            or (stage.phase=='hold' and alignment_goal is not None)
                            or (error_progress_enabled and stage.phase=='rest')):
                        p,R=self.session.model.fk(q[ARM],q[3],jac=False)
                        current=B@pose(p,R)
                        strict=alignment_goal is not None
                        progress_args={}
                        if error_progress_enabled:
                            progress_args['phase']=stage.phase
                            if stage.phase=='rest':
                                progress_args['rest_report']=self._rest_progress_summary([
                                    dict(step=self.steps,qpos=q,qvel=measured['qvel'])])
                        stalled=progress.update(episode.index,self.steps,current,episode.goal_local,
                            nominal_done=(episode.stage_steps+int(episode.pending))*episode.dt>=episode.duration,
                            position_tolerance=.002 if strict else .01,
                            rotation_tolerance=.02 if strict else .05,**progress_args)
                        if stalled:
                            stalled['waypoint_index']=sum(s.target is not None for s in episode.stages[:episode.index+1]) or 1
                            # Acknowledge the already-sent action through advance,
                            # with no extra controller tick or synthetic terminal.
                            episode.stages=[self.Stage('hold',episode.stage_steps+int(episode.pending))]
                            episode.index=0
                if (self.self_clearance is not None and not terminal and not self.is_stopped()
                        and (episode.pending or motion is None or motion['kind']!='look')):
                    # Check measured momentum before producing another controller tick.
                    # A new look has a checked head-joint path. Its first tick may
                    # reverse the preceding command's velocity; do not deadlock
                    # that reversal on the unchanged old velocity snapshot.
                    risk=self.self_clearance.segment(q,q+.15*np.asarray(measured['qvel']))
                    if risk:
                        stalled=risk
                        episode.stages=[self.Stage('hold',episode.stage_steps+int(episode.pending))]
                        episode.index=0
                arm_preview=None
                if (getattr(self,'arm_base_explicit',False) and motion is None
                        and not terminal and not self.is_stopped() and stalled is None):
                    base_stop,arm_preview=self._arm_base_proposal(episode,measured)
                    if base_stop is not None:
                        # Consume the last ACK only, including a zero-step refusal.
                        episode.max_steps=episode.steps+int(episode.pending)
                        packet=episode.advance(sequence=self.steps,**measured,terminated=False,truncated=False)
                        if packet is not None:raise RuntimeError('Base refusal produced an action')
                        self.closed=episode.closed
                        p_hold,R_hold=self.session.model.fk(q[ARM],q[3],jac=False)
                        self.last_goal_local=B@pose(p_hold,R_hold)
                        stalled=base_stop;reason='base_motion_required'
                        break
                rest_preview=None
                if (getattr(self,'rest_depth_guard',False) and motion is None
                        and any(s.phase=='rest' for s in stages) and rest_brake is None
                        and not terminal and not self.is_stopped() and not stalled):
                    report,rest_preview=self._rest_depth_preview(episode,measured,start)
                    if report is not None and report['clearance_stop_required']:
                        rest_brake=dict(start=self.steps,hold=q.copy(),grip=float(rest_preview[self.mapping['gripper'][0]]))
                        self.rest_brake_in_progress=True
                        # Keep the pending ACK in the original episode. The replacement
                        # hold consumes it once, then produces 60 real brake packets.
                        episode.stages=[self.Stage('hold',episode.stage_steps+int(episode.pending)+60)]
                        episode.index=0
                        p,R=self.session.model.fk(q[ARM],q[3],jac=False)
                        episode.goal_local=B@pose(p,R);episode.start_local=episode.goal_local.copy()
                        episode.duration=0.;episode.head_target=q[HEAD].copy()
                        self._flush()
                if rest_brake is not None:
                    velocity=np.asarray(measured['qvel'],float)
                    settled=(np.isfinite(velocity).all() and np.max(abs(velocity[ARM]))<.02
                        and np.max(abs(velocity[HEAD]))<.02 and abs(velocity[3])<.005
                        and np.max(abs(velocity[:3]))<.01)
                    rest_brake_stable=rest_brake_stable+1 if settled else 0
                preview_action=None
                if local_guard and not terminal and not self.is_stopped() and not stalled and not compliance_blocked:
                    risk,preview_action=self._clearance_tick(episode,measured,start,motion,allow_contact=allow_contact,
                                                            allow_observation_unknown=allow_observation_unknown)
                    observation_exception_used|=self.clearance_last_feedback.get('clearance_observation_exception',False)
                    if risk:
                        stalled=risk
                        # Consume only the already dispatched acknowledgement.
                        # No live tick was produced by the rejected clone.
                        episode.stages=[self.Stage('hold',episode.stage_steps+int(episode.pending))]
                        episode.index=0
                packet=self.timed('controller.advance',episode.advance,sequence=self.steps,**measured,
                    terminated=terminal or self.is_stopped(),truncated=False)
                self.closed=episode.closed
                self.last_goal_local=episode.goal_local
                if packet is None:
                    if progress is not None and episode.status in ('official_terminated','official_truncated'):
                        self.ended=True;stalled=None
                    reason='motion_stalled' if stalled else episode.status;break
                action,done=self._motion_action(packet,measured,start,motion)
                if arm_preview is not None and not np.array_equal(action,arm_preview):
                    raise RuntimeError('Arm/base planning differed from dispatch')
                if rest_brake is not None:
                    action=self._rest_brake_action(measured,rest_brake['hold'],rest_brake['grip'])
                elif rest_preview is not None and not np.array_equal(action,rest_preview):
                    raise RuntimeError('Rest preview differed from live command; no dispatch')
                if local_guard and (preview_action is None or not np.array_equal(action,preview_action)):
                    raise RuntimeError('Sensor preview differed from live command; no action dispatched')
                if rest_brake is not None or local_guard or getattr(self,'carried_envelope_enabled',False):
                    self._reconcile_arm_action(episode,packet,measured,action)
                # Every live servo tick corresponds to a sent action. Preview
                # ticks mutate only their temporary clone and are separately timed.
                if action.shape!=(20,) or not np.isfinite(action).all() or np.max(abs(action))>1:
                    raise ValueError('Invalid full B20D action')
                self.deadline()
                if (progress_only or error_progress_enabled) and packet['stage'] in waypoint_stages:
                    active_waypoint_stage=packet['stage']
                if hasattr(self,'_record_controller_diagnostic'):
                    self._record_controller_diagnostic(packet,measured,episode,action)
                self.pending=dict(sequence=self.steps,action=action.tolist(),phase=packet['phase'])
                if getattr(self,'carried_envelope_enabled',False) and packet['phase']=='close':
                    attachment=self.clearance.memory.attachment()
                    if attachment is not None and attachment['status']=='estimated':
                        # Record conservatively before dispatch. An unknown ACK
                        # cannot be interpreted as an empty or released gripper.
                        self.clearance.memory.attachment_event('close_started',
                            candidate_id=attachment['candidate_id'],step=self.steps)
                self._flush()  # durable before physical dispatch
                status=self.step_once(action)
                if not isinstance(status,dict) or any(type(status.get(k)) is not bool for k in ('terminated','truncated')):
                    raise RuntimeError('Action outcome/acknowledgement unknown')
                self.steps+=1
                self.trace.append(dict(step=self.steps,kind=packet['phase'],action=action.tolist()))
                self.pending=None;self._flush()
                self._capture(action,packet['phase'])
                terminal=status['terminated'] or status['truncated']
                if terminal:
                    self.ended=True
                    self.session.latch_official_end(terminated=status['terminated'],truncated=status['truncated'])
                if done and not terminal:
                    # Consume final measured acknowledgement without producing an extra tick.
                    episode.stages=[self.Stage('hold',episode.stage_steps+1)]
                    episode.index=0
                self.deadline()
            completed=reason=='plan_complete' and not stalled
            result=dict(accepted=True,reason='completed' if completed else reason,
                        steps_used=self.steps-before,steps_remaining=self.max_steps-self.steps)
            if reported_push is not None:
                result.update(base_push=reported_push,requires_replan=True,localization_invalidated=True)
                # Completion and localization are separate. This candidate does
                # not withdraw a real completion or allow a following stale goal.
                self.grasp_ready=False
            if rest_feedback_enabled:
                result['rest_progress_feedback']=self._rest_progress_summary(rest_samples)
            if grip_monitor:
                result['grip_distance_sequence']=grip_samples
            if rest_brake is not None:
                brake_steps=self.steps-rest_brake['start']
                brake_complete=bool(completed and brake_steps==60 and rest_brake_stable>=5 and not self.ended)
                self.rest_brake_in_progress=not brake_complete
                # A completed stop is NOT a completed return_to_rest or grasp.
                result.update(accepted=False,arrived=False,brake_completed=brake_complete,
                    brake_steps=brake_steps,reason=('rest_obstacle_braked' if brake_complete else
                    ('rest_brake_unsettled' if completed else reason)))
                self.last_goal_local=episode.goal_local.copy()
            if stalled:result.update(stalled)
            if (getattr(self,'blocked_grasp_direction',False) and not completed
                    and result['reason'] in ('motion_stalled','compliance_blocked','stage_step_limit')
                    and motion is None and not self.ended and episode.entered
                    and self.steps>before and episode.goal_local is not None):
                # A grip-loss or budget stop can occur before this command's
                # goal is bound, or while its final ACK enters the next stage.
                # Only an executed, blocked target has a meaningful direction.
                p_feedback,_=self.session.model.fk(q[ARM],q[3],jac=False)
                target_feedback=np.linalg.inv(B)@episode.goal_local
                result['grasp_error_base_m']=(target_feedback[:3,3]-p_feedback).tolist()
            if progress_only or (error_progress_enabled and (waypoint_stages or any(s.phase=='rest' for s in stages))):
                count=len(waypoint_stages) or 1
                index=(waypoint_stages.index(active_waypoint_stage)+1 if active_waypoint_stage is not None else 1)
                result['waypoint_progress']=dict(waypoint_index=None if completed else index,
                    waypoints_total=count,waypoints_completed=count if completed else index-1,
                    stalled_steps=int(stalled.get('stalled_steps',0)) if stalled else 0)
            if base_progress is not None:
                result.update(base_feedback)
                result['arrived']=bool(completed and base_feedback['position_error_m']<.025
                    and base_feedback['rotation_error_rad']<.02 and base_progress.max_deviation<=base_progress.deviation_limit)
                if getattr(self,'base_tracking_exit',False):
                    result.update(completed=result['arrived'],requires_replan=not result['arrived'])
                    if stalled:
                        result.update(localization_invalidated=True,clearance_invalidated=True,
                            recovery_guidance='Reassess RGB-D and restore furniture clearance before continuing; explicitly plan move_base. No automatic retreat.')
            if alignment_goal is not None:
                aligned=(alignment_feedback.get('position_error_m',float('inf'))<=.002 and
                         alignment_feedback.get('rotation_error_rad',float('inf'))<=.02 and
                         alignment_feedback.get('stable',False))
                result.update(alignment_feedback,arrived=bool(aligned and completed))
                if completed and not aligned:
                    result['reason']='compliance_blocked' if compliance_blocked else 'grasp_alignment_timeout'
            if any(s.phase=='close' for s in stages):
                result['closure_command_completed']=completed and any(e.get('closure_command_completed') is True for e in episode.events)
            if completed and any(s.phase=='open' for s in stages) and getattr(self,'carried_envelope_enabled',False):
                attachment=self.clearance.memory.attachment()
                if attachment is not None:
                    self.clearance.memory.attachment_event('open_completed',
                        candidate_id=attachment['candidate_id'],step=self.steps)
            if local_guard:
                if not self.ended and (not completed or (alignment_goal is not None and not result.get('arrived'))):
                    self.suspend_motion_goal(measured,result['reason'])
                elif completed and (motion is None or motion['kind']!='look'):
                    self.motion_suspended=False
                result['motion_suspended']=self.motion_suspended
                if stalled and str(stalled.get('reason','')).startswith('local_clearance_'):
                    result.update({k:v for k,v in self.clearance_last_feedback.items() if k.startswith('clearance_')})
                if observation_exception_used and not stalled:
                    result['clearance_observation_exception']=True
                    result.update(clearance_status='coverage_insufficient',clearance_scope='bounded_head_observation')
                advisory=getattr(self,'clearance_advisory_during_command',None)
                if advisory and not stalled:
                    result.update({k:v for k,v in advisory.items() if k.startswith('clearance_')
                                   and k!='clearance_observation_exception'})
            self._flush()
            return result
        except BaseException:
            self.failed=True;self._flush()
            try:self.recorder.finish('failed')
            except BaseException:pass
            raise

    def move_eef_chunk(self,targets,*,max_steps=150):
        from fetch_episode import rigid
        goals=[rigid(x) for x in targets]
        if not 1<=len(goals)<=8:raise ValueError('Expected1..8 tool-frame poses')
        remaining=min(max_steps,self.max_steps-self.steps)
        if remaining<3:return dict(accepted=False,reason='episode_step_limit',steps_used=0)
        if self.motion_guard:
            checked=self.check_motion_waypoints(goals)
            if not checked['accepted']:return checked
        # All points are expressed in the command-entry frame, not each new waypoint frame.
        stages=[self.Stage('settling',2)]+[self.Stage('pregrasp',remaining-2,g,'episode_base') for g in goals]
        result=self._run(stages,tool_cap=remaining)
        # One shared cap covers all waypoints and settling.
        return result

    def move_to(self,target,*,max_steps=150):return self.move_eef_chunk([target],max_steps=max_steps)

    def set_gripper(self,command):
        if command not in (0,1):raise ValueError('gripper is0 close or1 open')
        if command==0 and getattr(self,'carried_envelope_enabled',False) and self._attachment_candidate() is None:
            return dict(accepted=False,reason='carried_envelope_required',steps_used=0,
                        steps_remaining=self.max_steps-self.steps,closure_command_completed=False)
        return self._run([self.Stage('settling',2),self.Stage('open',1) if command else self.Stage('close',15)])

    def return_to_rest(self):
        from task_rules import REST_CAP
        return self._run([self.Stage('settling',2),self.Stage('rest',298 if getattr(self,'r15_protocol',False) else REST_CAP)])

    def look(self,pan_rad,tilt_rad):
        if not (-1.57<=pan_rad<=1.57 and -.76<=tilt_rad<=1.45):raise ValueError('Head joint limits')
        guard=getattr(self,'look_self_clearance',None) or self.self_clearance
        if guard is not None:
            measured=self.ports.observe();q=measured['qpos'];target=q.copy();target[[4,6]]=[pan_rad,tilt_rad]
            risk=guard.segment(q,target)
            if risk:
                if getattr(self,'local_clearance_guard',False):
                    self.suspend_motion_goal(measured,risk['reason']);self._flush()
                return dict(risk,steps_used=0,readonly=True)
        motion=dict(kind='look',target=[pan_rad,tilt_rad])
        if getattr(self,'s11_system_fixes',False):
            if guard is None:raise ValueError('S1.1 look requires robot self clearance')
            motion['hold_qpos']=np.asarray(measured['qpos'],float).copy()
        return self._run([self.Stage('hold',150 if getattr(self,'r15_protocol',False) else 40)],motion=motion)

    def move_base(self,turn_rad,forward_m,*,speed_mode="normal",max_steps=150):

        if type(max_steps) is not int or not 1 <= max_steps <= (250 if getattr(self,'r15_protocol',False) else 150):

            raise ValueError('Base step cap must be 1..250' if getattr(self,'r15_protocol',False) else 'Base step cap must be 1..150')

        if speed_mode not in ("normal","fast") or (speed_mode=="fast" and (not self.fast_base_approach or not .5<=forward_m<=1.5)):
            raise ValueError("Fast approach requires explicit profile and bounded forward command")
        if not np.isfinite([turn_rad,forward_m]).all() or abs(turn_rad)>math.pi or abs(forward_m)>2:
            raise ValueError('Bounded metric base command required')
        return self._run([self.Stage('hold',max_steps)],motion=dict(kind='base',turn=turn_rad,forward=forward_m,speed_mode=speed_mode))


    def check_motion_waypoints(self,targets,*,position_tolerance=.01,rotation_tolerance=.05,report_residuals=False):
        """All proposed waypoints, sequential IK seeds; no scene geometry/steps."""
        from copy import deepcopy
        from fetch_episode import rigid,ARM
        from scipy.spatial.transform import Rotation
        if not self._available():
            return dict(accepted=False,reason='official_end_or_step_limit',steps_used=0,arrived=False)
        goals=[rigid(t) for t in targets]
        if not goals:raise ValueError('Expected pose')
        measured=self.ports.observe();q=measured['qpos']
        ik=deepcopy(self.session.ctrl.ik);candidate=q[ARM].copy()
        if ik.q_post is None:ik.q_post=candidate.copy()
        failed=None;residuals=[]
        for index,T in enumerate(goals,1):
            for iteration in range(80):
                self.deadline()
                if not self._available():
                    return dict(accepted=False,reason='official_end_or_step_limit',steps_used=0,arrived=False)
                previous=candidate.copy()
                candidate=ik.solve(candidate,q[3],T[:3,3],T[:3,:3])
                if self.self_clearance is not None:
                    qa=q.copy();qb=q.copy();qa[ARM]=previous;qb[ARM]=candidate
                    risk=self.self_clearance.segment(qa,qb)
                    if risk:return dict(risk,readonly=True,steps_used=0,waypoint_index=index)
            p,R=self.session.model.fk(candidate,q[3],jac=False)
            pe=float(np.linalg.norm(p-T[:3,3]))
            re=float(Rotation.from_matrix(R.T@T[:3,:3]).magnitude())
            if not (np.isfinite(pe) and np.isfinite(re)):raise ValueError('Nonfinite waypoint IK')
            if report_residuals:residuals.append(dict(waypoint_index=index,position_error_m=pe,rotation_error_rad=re))
            if (pe>position_tolerance or re>rotation_tolerance) and failed is None:
                failed=dict(waypoint_index=index,position_error_m=pe,rotation_error_rad=re)
        result=dict(accepted=failed is None,readonly=True,steps_used=0,waypoints_checked=len(goals),
            reason='waypoint_ik_unresolved' if failed else 'robot_ik_only')
        if failed:result.update(failed,arrived=False)
        if report_residuals:result['waypoint_errors']=residuals
        return result

    def check_path(self,targets):
        if getattr(self,'dual_reachability',False):
            goals=list(targets)
            checked=self.check_motion_waypoints(goals,report_residuals=True)
            rows=checked.pop('waypoint_errors',[]);complete=len(rows)==len(goals) and bool(goals)
            def summary(position,rotation):
                failed=next((r for r in rows if r['position_error_m']>position or r['rotation_error_rad']>rotation),None)
                return dict(reachable=(failed is None) if complete else None,position_tolerance_m=position,
                    rotation_tolerance_rad=rotation,waypoints_checked=len(rows),
                    failed_waypoint=None if failed is None else failed['waypoint_index'],
                    position_error_m=None if failed is None else failed['position_error_m'],
                    rotation_error_rad=None if failed is None else failed['rotation_error_rad'])
            result=dict(accepted=bool(complete and checked['accepted']),
                reason='robot_ik_dual_report' if complete else checked['reason'],readonly=True,steps_used=0,
                reachability=dict(ordinary_move=summary(.01,.05),strict_grasp=summary(.002,.02),
                    complete=complete,basis='current_torso_robot_ik_no_scene_clearance'))
            if getattr(self,'arm_tools_lock_base',False):
                result['arm_base_policy']=dict(base_locked=True,wrist_radius_m=[.50,.65],bearing_limit_rad=.6,
                    reach_basis='preferred_band_not_ik_guarantee',clearance_validity='invalidate_after_pose_route_change_or_unexpected_base_motion')
            return result
        if getattr(self,'arm_tools_lock_base',False):
            goals=list(targets);refusal=self._locked_base_reach(goals)
            result=refusal if refusal is not None else dict(accepted=True,reason='fixed_base_robot_ik_only',readonly=True,steps_used=0)
            result['arm_base_policy']=dict(base_locked=True,wrist_radius_m=[.50,.65],bearing_limit_rad=.6,
                reach_basis='preferred_band_not_ik_guarantee',clearance_validity='invalidate_after_pose_route_change_or_unexpected_base_motion')
            return result
        from copy import deepcopy
        from fetch_episode import rigid, ARM
        goals=[rigid(x) for x in targets]
        if not goals:raise ValueError('Expected pose')
        measured=self.ports.observe();q=measured['qpos'];ik=deepcopy(self.session.ctrl.ik)
        candidate=q[ARM].copy();ok=True
        if ik.q_post is None:ik.q_post=candidate.copy()
        for T in goals:
            for _ in range(80):candidate=ik.solve(candidate,q[3],T[:3,3],T[:3,:3])
            p,R=self.session.model.fk(candidate,q[3],jac=False)
            ok &= bool(np.linalg.norm(p-T[:3,3])<=.01 and np.linalg.norm(R-T[:3,:3])<=.08)
        return dict(accepted=ok,reason='robot_ik_only' if ok else 'ik_unresolved',readonly=True,steps_used=0)

    def check_clearance(self, targets=(), *, rest=False, turn_rad=None, forward_m=None):
        """Read-only command preview using the same episode, IK and SDK codec.

        Later measurements do not exist yet. Preview uses the controller's
        nominal implicit-PD motion; feedback-dependent tracking and contact remain
        unknown. The live session is never advanced, reset or vetoed.
        """
        from fetch_episode import rigid, ARM, HEAD, REST
        from scipy.spatial.transform import Rotation
        from .clearance import odometry_transform
        if not getattr(self,'r2b_clearance',False):raise ValueError('Clearance report disabled')
        if turn_rad is not None or forward_m is not None:
            if (not getattr(self,'r2d_base_clearance',False) or targets or rest is not False
                    or any(type(v) not in (int,float) or not np.isfinite(v) for v in (turn_rad,forward_m))
                    or abs(turn_rad)>math.pi or abs(forward_m)>1.5):
                raise ValueError('Explicit bounded base clearance query required')
            measured=self.ports.observe()
            report=self._require_clearance_binding().check_base(measured['qpos'],
                measured['local_from_base'],turn_rad,forward_m,self.steps,report_only=True)
            return self._clearance_verification(report,base=True)
        goals=[rigid(t) for t in targets]
        if type(rest) is not bool or (rest and goals) or (not rest and not 1<=len(goals)<=8):
            raise ValueError('Use 1..8 poses or rest')
        measured=self.ports.observe()
        ceiling=self.usable_step_cap() if hasattr(self,'usable_step_cap') else self.max_steps
        if not rest:ceiling-=getattr(self,'rest_reserve',0)
        cap=min(300,ceiling-self.steps)
        if cap<3:
            result=self._require_clearance_binding().check_joint_path([measured['qpos']],
                measured['local_from_base'],self.steps,report_only=True)
            result['path_clearance'].update(path_model='no_executable_budget',preview_steps=0,path_complete=False)
            return self._clearance_verification(result)
        stages=[self.Stage('settling',2)]+([self.Stage('rest',cap-2)] if rest else
                 [self.Stage('pregrasp',cap-2,g,'episode_base') for g in goals])
        episode=self.Episode(self.session,stages,self.mapping,max_steps=cap)
        episode.closed=self.closed
        episode.goal_local=None if self.last_goal_local is None else self.last_goal_local.copy()
        if getattr(self,'rest_joint_target',False) and rest:episode.session=_RestJointSession(episode.session)
        preview=clone_full_b_preview(episode)
        if getattr(self,'arm_tools_lock_base',False):
            preview.session.ctrl.base=ArmToolsLockedBasePlanner(preview.session.ctrl.base)
        q=np.asarray(measured['qpos'],float).copy();v=np.asarray(measured['qvel'],float).copy()
        B=np.asarray(measured['local_from_base'],float).copy();bv=np.asarray(measured['base_velocity'],float).copy()
        path=[q.copy()];odoms=[B.copy()];base_stop=None
        for tick in range(cap+1):
            self.deadline()
            # Preview-only robot-kinematic predicate, never an official outcome.
            rp,rR=preview.session.model.fk(REST[ARM],REST[3],jac=False)
            qp,qR=preview.session.model.fk(q[ARM],q[3],jac=False)
            nominal_rest=dict(ee_rest=bool(np.linalg.norm(qp-rp)<=.01 and
                Rotation.from_matrix(qR.T@rR).magnitude()<=.05),
                robot_rest=bool(np.max(abs(q[np.r_[3,HEAD,ARM]]-REST[np.r_[3,HEAD,ARM]]))<=.01),
                is_static=bool(np.max(abs(v[3:13]))<=.01))
            if getattr(self,'arm_base_explicit',False):
                state=dict(qpos=q,qvel=v,local_from_base=B,base_velocity=bv,rest_flags=nominal_rest)
                # The shared gate uses the current observation sequence.
                base_stop,_=self._arm_base_proposal(preview,state,sequence=self.steps+tick)
                if base_stop is not None:
                    preview.max_steps=preview.steps+int(preview.pending)
                    ack=preview.advance(sequence=self.steps+tick,**state,terminated=False,truncated=False)
                    if ack is not None:raise RuntimeError('Preview base refusal emitted action')
                    break
            packet=preview.advance(sequence=self.steps+tick,qpos=q,qvel=v,local_from_base=B,
                base_velocity=bv,rest_flags=nominal_rest,terminated=False,truncated=False)
            if packet is None:break
            if tick==cap:raise AssertionError('Preview emitted beyond its ACK budget')
            decoded=clearance_codec_targets(q,packet['action'],self.mapping)
            next_q=decoded['geometry_position_target'];next_q[3]=q[3]+.05*decoded['torso_velocity']
            # A PD setpoint is not the next measured position. Reuse the
            # controller's own float32-codec-corrected nominal transition.
            predicted=np.asarray(preview.session.ctrl.servo.pred,float)
            if predicted.shape!=(14,) or not np.isfinite(predicted).all():
                raise ValueError('Invalid nominal arm prediction')
            next_q[ARM]=predicted[:7]
            # Closing-finger drive is not the object's future width.
            next_q[13:15]=q[13:15]
            next_q[:3]=path[0][:3]
            v=(next_q-q)/.05;v[ARM]=predicted[7:];q=next_q
            speed,yaw=decoded['base_velocity'];theta=float(yaw*.05)
            delta=odometry_transform(dict(dx_m=float(speed*.05) if abs(yaw)<1e-9 else float(speed/yaw*math.sin(theta)),
                dy_m=0. if abs(yaw)<1e-9 else float(speed/yaw*(1-math.cos(theta))),dyaw_rad=theta))
            B=B@delta;bv=np.array([speed,0.,yaw])
            path.append(q.copy());odoms.append(B.copy())
        result=self._require_clearance_binding().check_joint_path(path,np.asarray(odoms),self.steps,report_only=True)
        result['path_clearance']['path_model']='same_controller_implicit_pd_nominal_tracking'
        result['path_clearance']['preview_steps']=len(path)-1
        if base_stop is not None:result.update(base_stop,readonly=True,steps_used=0)
        result['path_clearance']['path_complete']=base_stop is None and preview.status=='plan_complete'
        result['path_clearance'].update(preview_status='base_motion_required' if base_stop else preview.status,
            acknowledged_steps=preview.steps,pending_ack=bool(preview.pending),
            completion_basis='nominal_robot_kinematics_not_official_success',
            stages_completed=preview.index)
        return self._clearance_verification(result)

    def _clearance_verification(self, result, *, base=False):
        """Advisory evidence quality, without changing geometry or admission."""
        if not getattr(self,'clearance_verification',False):return result
        report=result['path_clearance']
        # A base command carries every link. For an arm command, include torso
        # and the arm/fingers; an unobserved stationary wheel is not arm evidence.
        relevant=[r for r in report['links'] if base or any(token in r['link'] for token in
            ('torso','shoulder','upperarm','elbow','forearm','wrist','gripper'))]
        high=[r['link'] for r in relevant if r['unobserved_fraction']>=.5]
        reasons=[]
        if not relevant:reasons.append('missing_relevant_links')
        if high:reasons.append('majority_unobserved')
        if report['stale_evidence']:reasons.append('stale_evidence')
        if report.get('path_complete') is False:reasons.append('incomplete_preview')
        report['verification']=dict(status='unverified' if reasons else 'sensor_report_only',
            reasons=reasons,high_unobserved_links=high,unobserved_threshold=.5,
            guidance='Turn the head to observe the swept region, then refresh this report.' if reasons else
                'Sensor evidence only; future tracking and contact remain unknown.')
        return result

    def finish(self,status):
        if self.pending is not None:status='unknown_action'
        self.recorder.finish(status)
        self._flush()
        finalize=getattr(self.persist,"finish",None)
        if finalize is not None:finalize()


class AlignedFullBOnlineExecutor(FullBOnlineExecutor):
    """Opt-in task policy; frozen Full B dynamics, IK and close gate unchanged.

    The locked goal is a robot-command pose in local odometry, never an object
    pose. Explicit replans are auditable self-reports, not grasp certification.
    """
    version='full-B-r4-grasp-alignment-v1'
    rest_reserve=122  # unchanged rest cap 120 plus two settling steps
    convergence_cap=40

    def __init__(self,*args,smooth_localization=False,grasp_direction_feedback=False,rest_obstruction_guard=False,rest_depth_checker=None,rest_progress_feedback=False,r15_protocol=False,r15_completion=False,r15_pregrasp=False,r15_carry=False,**kwargs):
        if type(rest_progress_feedback) is not bool:raise ValueError('Boolean rest progress feedback required')
        if type(rest_obstruction_guard) is not bool:raise ValueError('Boolean rest obstruction guard required')
        if type(grasp_direction_feedback) is not bool:raise ValueError('Boolean grasp feedback required')
        if type(smooth_localization) is not bool:raise ValueError('Boolean localization timing required')
        super().__init__(*args,**kwargs)
        flags=dict(r15_protocol=r15_protocol,r15_completion=r15_completion,r15_pregrasp=r15_pregrasp,r15_carry=r15_carry)
        if any(type(v) is not bool for v in flags.values()):raise ValueError('Boolean R1.5 switches required')
        if (r15_pregrasp or r15_carry) and not (r15_protocol and r15_completion):raise ValueError('R1.5 motion requires protocol and completion')
        self.__dict__.update(flags)
        if r15_protocol:
            self.rest_reserve=300
            # Instance-local task protocol: estimated rest duration cannot deny
            # a plan. The unchanged stage/episode counters still stop at timeout.
            # Do not mutate the frozen module or another executor's globals.
            import types
            base_episode=self.Episode;original=base_episode.advance
            old_budget=original.__globals__['rest_budget']
            def timeout_budget(*args,**kw):
                value=old_budget(*args,**kw)
                return dict(value,feasible=True,admission_mode='timeout_only')
            advance=types.FunctionType(original.__code__,dict(original.__globals__,rest_budget=timeout_budget),
                original.__name__,original.__defaults__,original.__closure__)
            advance.__kwdefaults__=original.__kwdefaults__
            self.Episode=type('R15TimeoutEpisode',(base_episode,),dict(advance=advance))
        self.r15_rest_requires_replan=False;self._r15_rest_prelude=False
        self.smooth_localization=smooth_localization
        self.grasp_direction_feedback=grasp_direction_feedback
        self.rest_obstruction_guard=rest_obstruction_guard
        if rest_progress_feedback:self.rest_progress_feedback=True
        if rest_depth_checker is not None:
            if rest_obstruction_guard or getattr(self,'local_clearance_guard',False):
                raise ValueError('Rest braking must be isolated from other interruption guards')
            if not callable(getattr(rest_depth_checker,'check_observed_arm_path',None)):
                raise ValueError('Rest depth checker required')
            self.rest_depth_guard=True;self.rest_depth_checker=rest_depth_checker
            self.rest_brake_in_progress=False
        self.grasp_goal_local=None;self.grasp_revisions=[];self.grasp_ready=False
        self.alignment_samples=[];self.phase_step_ceiling=None
        self.controller_diagnostics=[]
        self.primary_step_cap=self.max_steps
        self.recovery_released=False
        self.search_continuation_released=False

    def configure_recovery(self, primary_steps=800):
        if (not self.weekly_budget or self.steps or self.recovery_released
                or (self.max_steps, primary_steps) not in ((1200, 800), (1600, 1200), (3400, 1800))):
            raise ValueError('Fresh weekly executor with a 400-step recovery reserve required')
        self.primary_step_cap=primary_steps

    def release_recovery(self):
        if (not self.weekly_budget or self.max_steps-self.primary_step_cap not in ((400,1600) if self.max_steps==3400 else (400,)) or self.recovery_released
                or not self._available()):
            raise ValueError('Recovery unavailable; no reset or replay')
        self.recovery_released=True
        self.primary_step_cap+=400
        self._flush()

    def release_search_continuation(self):
        if (not self.weekly_budget or self.max_steps!=3400 or self.search_continuation_released
                or not self._available()):
            raise ValueError('Search continuation unavailable')
        if self.primary_step_cap not in (1800,2200):raise ValueError('Unexpected current allocation')
        self.primary_step_cap+=1200
        self.search_continuation_released=True
        self._flush()

    def usable_step_cap(self):
        return self.primary_step_cap if self.weekly_budget else self.max_steps

    @staticmethod
    def _rest_progress_summary(samples):
        """Robot joint residuals only; no arrival, possession or safety verdict.

        Residual is target minus measured. Improvement is start norm minus end
        norm, so negative means farther from REST. This is not servo tracking
        error. Torso metres and arm radians are never combined.
        """
        from fetch_episode import ARM, REST
        report=dict(version='rest-progress-v1',status='unknown',entry_step=None,exit_step=None,
                    residual_convention='target_minus_measured',arm=None,torso=None)
        if not samples:return report
        first,last=samples[0],samples[-1]
        report.update(entry_step=first['step'],exit_step=last['step'])
        indices=np.asarray(ARM,dtype=int)
        def finite_slice(value,index):
            try:
                part=np.asarray(value,float)[index]
                return part if np.isfinite(part).all() else None
            except (IndexError,TypeError,ValueError):return None
        target=finite_slice(REST,indices)
        a=finite_slice(first['qpos'],indices);b=finite_slice(last['qpos'],indices)
        velocity=finite_slice(last['qvel'],indices)
        arm=dict(joint_indices=indices.tolist(),target_rad=None if target is None else target.tolist(),
                 start_error_rad=None,end_error_rad=None,start_l2_rad=None,end_l2_rad=None,
                 start_linf_rad=None,end_linf_rad=None,l2_improvement_rad=None,linf_improvement_rad=None,
                 exit_velocity_rad_s=None if velocity is None else velocity.tolist(),
                 exit_max_abs_velocity_rad_s=None if velocity is None else float(np.max(np.abs(velocity))))
        if target is not None and a is not None and b is not None:
            start_error=target-a;end_error=target-b
            arm.update(start_error_rad=start_error.tolist(),end_error_rad=end_error.tolist(),
                       start_l2_rad=float(np.linalg.norm(start_error)),end_l2_rad=float(np.linalg.norm(end_error)),
                       start_linf_rad=float(np.max(np.abs(start_error))),end_linf_rad=float(np.max(np.abs(end_error))))
            arm.update(l2_improvement_rad=arm['start_l2_rad']-arm['end_l2_rad'],
                       linf_improvement_rad=arm['start_linf_rad']-arm['end_linf_rad'])
            report['status']='measured' if velocity is not None else 'partial'
        torso_target=finite_slice(REST,3)
        ta=finite_slice(first['qpos'],3);tb=finite_slice(last['qpos'],3)
        torso=dict(joint_index=3,target_m=None if torso_target is None else float(torso_target),
                   start_error_m=None,end_error_m=None,absolute_error_improvement_m=None)
        if torso_target is not None and ta is not None and tb is not None:
            torso.update(start_error_m=float(torso_target-ta),end_error_m=float(torso_target-tb),
                         absolute_error_improvement_m=float(abs(torso_target-ta)-abs(torso_target-tb)))
        else:report['status']='partial' if report['status']=='measured' else report['status']
        report.update(arm=arm,torso=torso)
        return report

    def _record_controller_diagnostic(self,packet,measured,episode,action):
        """Evaluation-only snapshot before dispatch; never recompute a controller tick.

        sequence is the pre-action observation step. A physical acknowledgement
        is established only by trace step sequence+1; pending alone is not one.
        Missing controller fields remain missing, never synthesized as zero.
        """
        def plain(value):
            if isinstance(value,np.ndarray):return value.tolist()
            if isinstance(value,np.generic):return value.item()
            if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [plain(v) for v in value]
            return value
        self.controller_diagnostics.append(plain(dict(sequence=self.steps,
            phase=packet['phase'],evaluation_only=True,
            controller_info=packet.get('controller_info'),
            clipped_groups=packet.get('clipped_groups'),
            measured_qpos=measured['qpos'],measured_qvel=measured['qvel'],
            local_from_base=measured['local_from_base'],
            commanded_goal_local=episode.goal_local,
            locked_grasp_goal_local=self.grasp_goal_local,
            action_mapping=self.mapping,dispatched_action=action)))

    def _state(self):
        value = dict(super()._state(),grasp_goal_local=None if self.grasp_goal_local is None
                    else self.grasp_goal_local.tolist(),grasp_ready=self.grasp_ready,
                    grasp_revisions=self.grasp_revisions,rest_reserved_steps=self.rest_reserve,
                    alignment_samples=self.alignment_samples,
                    controller_diagnostics=self.controller_diagnostics)
        if self.r15_completion:value['r15_rest_requires_replan']=self.r15_rest_requires_replan
        if getattr(self,'rest_base_push_report_only',False):
            value.update(rest_base_push_report_only=True,
                rest_base_push_samples=getattr(self,'rest_base_push_samples',[]),
                rest_base_push_report=getattr(self,'_rest_base_push_report',None))
        if self.weekly_budget:
            value.update(primary_step_cap=self.primary_step_cap,recovery_released=self.recovery_released)
            if self.max_steps==3400:value["search_continuation_released"]=self.search_continuation_released
        return value

    def _reject(self,reason):
        return dict(accepted=False,reason=reason,steps_used=0,
                    steps_remaining=self.max_steps-self.steps)

    def _run(self,stages,*,motion=None,tool_cap=None,alignment_goal=None,local_goal=None):
        rest=any(s.phase=='rest' for s in stages) or getattr(self,'_r15_rest_prelude',False)
        if self.weekly_budget and rest:
            usable=self.usable_step_cap()-self.steps
            if usable<=0:return self._reject('episode_step_limit')
            tool_cap=min(sum(s.cap for s in stages) if tool_cap is None else tool_cap,usable)
        if not rest:
            ceiling=self.usable_step_cap()-self.rest_reserve
            if self.phase_step_ceiling is not None:ceiling=min(ceiling,self.phase_step_ceiling)
            usable=ceiling-self.steps
            if usable<=0:return self._reject('rest_budget_reserved')
            requested=sum(s.cap for s in stages) if tool_cap is None else tool_cap
            tool_cap=min(requested,usable)
            self.grasp_ready=False
        result=super()._run(stages,motion=motion,tool_cap=tool_cap,alignment_goal=alignment_goal,local_goal=local_goal)
        if result['reason'] in ('episode_step_limit','stage_step_limit'):
            # FetchEpisode owns one tool invocation, not the whole task.
            # Do not expose its local cap as an official environment end.
            if self.steps >= self.max_steps:
                result['reason']='episode_step_limit'
            elif not rest and self.steps>=self.usable_step_cap()-self.rest_reserve:
                result['reason']='rest_budget_reserved'
            elif alignment_goal is not None:
                result['reason']='grasp_alignment_timeout'
                result['arrived']=False
            else:
                result['reason']='rest_motion_limit' if rest else 'tool_motion_limit'
        return result

    def _align_grasp(self):
        if self.grasp_goal_local is None:return self._reject('prepare_grasp_required')
        result=self._run([self.Stage('hold',self.convergence_cap)],alignment_goal=self.grasp_goal_local)
        self.grasp_ready=result.get('arrived',False)
        self._flush()
        return result

    def _compound_stage(self, name, before, result):
        return dict(stage=name,start_step=before,end_step=self.steps,
                    steps_used=self.steps-before,reason=result['reason'],
                    action_dispatched=self.steps>before,ack_complete=self.pending is None)

    def _compound_result(self, result, before, stages):
        result=dict(result,steps_used=self.steps-before)
        if not getattr(self,'compound_feedback',False):return result
        # Only acknowledged execution may be summarized. Unknown dispatches
        # retain the existing exception/pending path and cannot become feedback.
        if self.pending is not None or self.failed:
            raise RuntimeError('Compound action acknowledgement unknown')
        dispatched=self.steps>before
        result.update(readonly=not dispatched,action_dispatched=dispatched,
            execution_summary=dict(start_step=before,end_step=self.steps,
                ack_complete=True,sub_stages=stages))
        if 'base_reposition' in result:
            result['base_reposition']=dict(result['base_reposition'],action_dispatched=dispatched)
        return result

    def prepare_grasp(self,target,*,reason,replan=False):
        from fetch_episode import rigid
        target=rigid(target)
        if not isinstance(reason,str) or not reason.strip() or len(reason)>900:
            raise ValueError('Grasp localization requires a short evidence-based reason')
        if self.closed:return self._reject('open_before_grasp_localization')
        if (self.grasp_goal_local is not None)!=bool(replan):
            return self._reject('replan_grasp_required' if self.grasp_goal_local is not None else 'prepare_grasp_required')
        if not self._available():return self._reject('official_end_or_step_limit')
        # Reserve both this localization's convergence and the next close's
        # recheck (40) + frozen closure (17), as well as the rest allocation.
        motion_cap=min(150,self.usable_step_cap()-self.steps-self.rest_reserve-2*self.convergence_cap-17)
        if motion_cap<3:return self._reject('insufficient_grasp_and_rest_budget')
        if self.motion_guard:
            checked=self.check_motion_waypoints([target],position_tolerance=.002,rotation_tolerance=.02)
            if not checked['accepted']:return checked
        B=self.ports.observe()['local_from_base']
        if self.smooth_localization:
            from fetch_episode import ARM,pose
            measured=self.ports.observe();q=measured['qpos'];B=measured['local_from_base']
            p,R=self.session.model.fk(q[ARM],q[3],jac=False)
            duration=smooth_localization_duration(pose(p,R),target,self.session.ctrl.dtp)
            if 2+int(math.ceil(duration/self.session.ctrl.dtp))>motion_cap:
                return dict(self._reject('insufficient_smooth_localization_budget'),
                    nominal_localization_seconds=duration,motion_steps_available=motion_cap)
        previous=self.grasp_goal_local
        goal=B@target
        self.grasp_revisions.append(dict(step=self.steps,replan=bool(replan),reason=reason,
            previous=None if previous is None else previous.tolist(),target=goal.tolist()))
        self.grasp_goal_local=goal.copy();self.grasp_ready=False;self._flush()
        before=self.steps
        # Bind the command once in local odometry. The first actual observation
        # re-expresses it in the new base frame without changing the target.
        result=self._run([self.Stage('settling',2),self.Stage('pregrasp',motion_cap-2,target,'episode_base')],
                         tool_cap=motion_cap,local_goal=goal)
        stages=[self._compound_stage('localization',before,result)]
        if result.get('reason')=='completed':
            alignment_start=self.steps
            result=self._align_grasp()
            stages.append(self._compound_stage('alignment',alignment_start,result))
        result=self._compound_result(result,before,stages)
        if getattr(self,'local_clearance_guard',False):
            # Settling/observation alone does not enter Pick. This is executed
            # command-phase evidence, not scene contact or grasp ground truth.
            result['localization_motion_executed']=any(row['step']>before and row['kind']=='pregrasp'
                                                      for row in self.trace)
        self.r2a_blocked_grasp = (self.steps if result.get('reason')=='compliance_blocked' else None)
        return result

    def close_after_blocked(self,note,observation_id,*,close_evidence=None,structured_evidence=False):
        import re
        if structured_evidence:
            evidence=(isinstance(close_evidence,dict)
                and set(close_evidence)=={'fingers_bracket_object','observation_id'}
                and close_evidence['fingers_bracket_object'] is True
                and isinstance(observation_id,str) and bool(observation_id)
                and close_evidence['observation_id']==observation_id)
        else:
            evidence=(isinstance(note,str) and isinstance(observation_id,str) and observation_id in note
                and (re.search(r'hand.*(between.*fingers|fingers.*around)',note,re.I) or ('手' in note and '两指之间' in note)))
        if not evidence:return dict(self._reject('hand_image_evidence_required'),closure_command_completed=False)
        if getattr(self,'r2a_blocked_grasp',None)!=self.steps or self.grasp_goal_local is None or self.closed:
            return dict(self._reject('blocked_grasp_required'),closure_command_completed=False)
        target,actual,pe,re=self._r15_pose_error(self.grasp_goal_local)
        residual=dict(position_error_m=pe,rotation_error_rad=re)
        if pe>.020 or re>.1:return dict(self._reject('grasp_alignment_required'),preclose_residual=residual,closure_command_completed=False)
        if self.usable_step_cap()-self.steps<self.rest_reserve+17:return dict(self._reject('insufficient_close_and_rest_budget'),preclose_residual=residual,closure_command_completed=False)
        self.r2a_blocked_grasp=None
        # G1 closes at the measured robot pose, rather than commanding another
        # convergence toward the blocked goal. Keep that original grasp goal
        # for residual provenance; only the servo hold reference changes.
        # This also satisfies the parent's unchanged strict close-at-hold gate.
        measured=self.ports.observe()
        self.last_goal_local=measured['local_from_base']@self._r15_pose(measured)
        if not getattr(self,'g1_dispatch_semantics',False):
            result=super().set_gripper(0)
            return dict(result,preclose_residual=residual)
        # The normal close gate is unchanged. Avoid two open settling ticks
        # moving away from this measured hold under the persistent compliance
        # offset. Seed its stability check with the real preceding ACK only.
        previous=next((x for x in reversed(self.controller_diagnostics)
                       if x.get('sequence')==self.steps-1),None)
        previous_q=None if previous is None else np.asarray(previous['measured_qpos'],float)
        torso_delta=None if previous_q is None else float(abs(measured['qpos'][3]-previous_q[3]))
        hold=dict(position_error_m=0.,rotation_error_rad=0.,torso_delta_m=torso_delta)
        if torso_delta is None or torso_delta>=.001:
            return dict(self._reject('close_hold_unstable'),preclose_residual=residual,
                        close_dispatched=False,hold_residual=hold,closure_command_completed=False)
        start=self.steps
        self._g1_previous_q=previous_q
        try:
            result=self._run([self.Stage('close',15)])
        finally:
            self._g1_previous_q=None
        dispatched=any(x['step']>start and x['action'][self.mapping['gripper'][0]]<0 for x in self.trace)
        return dict(result,preclose_residual=residual,close_dispatched=dispatched,hold_residual=hold)

    def set_gripper(self,command):
        if command not in (0,1):raise ValueError('gripper is0 close or1 open')
        if command==1:
            self.grasp_ready=False
            return super().set_gripper(command)
        if self.grasp_goal_local is None:return dict(self._reject('prepare_grasp_required'),closure_command_completed=False)
        if self.closed:return dict(self._reject('already_closed'),closure_command_completed=False)
        if self.usable_step_cap()-self.steps<self.rest_reserve+17:
            return dict(self._reject('insufficient_close_and_rest_budget'),closure_command_completed=False)
        before=self.steps
        # Re-check against the locked grasp goal, never last_goal_local from a
        # later generic move_to. Reserve the complete unchanged closure cap.
        available=self.usable_step_cap()-self.steps-self.rest_reserve-17
        if available<1:return dict(self._reject('insufficient_close_and_rest_budget'),closure_command_completed=False)
        result=self._run([self.Stage('hold',min(self.convergence_cap,available))],alignment_goal=self.grasp_goal_local)
        stages=[self._compound_stage('alignment',before,result)]
        if result.get('arrived'):
            residuals={k:result[k] for k in ('position_error_m','rotation_error_rad','stable')}
            self.last_goal_local=self.grasp_goal_local.copy()
            close_start=self.steps
            result=super().set_gripper(command)
            stages.append(self._compound_stage('closure',close_start,result))
            result.update(residuals)
        result.update(steps_used=self.steps-before,closure_command_completed=result.get('closure_command_completed',False))
        return self._compound_result(result,before,stages)


    def _r15_pose(self, measured):
        from fetch_episode import ARM,pose
        q=measured['qpos'];p,R=self.session.model.fk(q[ARM],q[3],jac=False)
        return pose(p,R)

    def _r15_pose_error(self, local_goal):
        from scipy.spatial.transform import Rotation
        measured=self.ports.observe();actual=self._r15_pose(measured)
        target=np.linalg.inv(measured['local_from_base'])@local_goal
        pe=float(np.linalg.norm(target[:3,3]-actual[:3,3]))
        re=float(Rotation.from_matrix(actual[:3,:3].T@target[:3,:3]).magnitude())
        return target,actual,pe,re

    def r15_execute(self,tool,call,operation):
        """Completion measured from robot sensors, never inferred from stage count.

        This wrapper reports after execution. It adds no automatic retry, scene
        check or change to controller dynamics. Lateral drift is reported, not
        secretly used to reject a base command.
        """
        from fetch_episode import ARM,REST
        from scipy.spatial.transform import Rotation
        before=self.ports.observe();start=self.steps
        previous_entry=getattr(self,'_arm_tool_entry',None)
        previous_tool=getattr(self,'_arm_tool_name',None)
        track_push=(getattr(self,'arm_base_push_exit',False) and getattr(self,'arm_tools_lock_base',False)
                    and tool in ('prepare_grasp','replan_grasp','move_to','move_eef_chunk','set_gripper','return_to_rest'))
        if track_push:
            self._arm_tool_entry=(before['local_from_base'].copy(),start)
            self._arm_tool_name=tool
            if tool=='return_to_rest' and getattr(self,'rest_base_push_report_only',False):
                self.rest_base_push_samples=[]
                self._rest_base_push_report=None
        try:
            result=operation()
        finally:
            if track_push:
                self._arm_tool_entry=previous_entry
                self._arm_tool_name=previous_tool
        after=self.ports.observe()
        if (tool=='return_to_rest' and getattr(self,'rest_base_push_report_only',False)
                and getattr(self,'_rest_base_push_report',None) is not None):
            result=dict(result,base_push=self._rest_base_push_report.copy(),
                requires_replan=True,localization_invalidated=True)
        requested={};actual={};residual={};arrived=False
        raw=result.get('reason','unknown')
        if tool=='look':
            target=np.array([call['pan_rad'],call['tilt_rad']]);q=after['qpos'][[4,6]]
            requested={'head_rad':target.tolist()};actual={'head_rad':q.tolist()}
            residual={'head_rad':(target-q).tolist()};arrived=bool(np.max(abs(target-q))<.02)
        elif tool=='move_base':
            delta=np.linalg.inv(before['local_from_base'])@after['local_from_base']
            turn=call['turn_rad'];forward=call['forward_m']
            yaw=math.atan2(delta[1,0],delta[0,0]);direction=np.array([math.cos(turn),math.sin(turn)])
            distance=float(direction@delta[:2,3]);lateral=float(np.array([-direction[1],direction[0]])@delta[:2,3])
            error=math.atan2(math.sin(turn-yaw),math.cos(turn-yaw))
            requested={'base_delta':[forward,0.,turn]};actual={'base_delta':[distance,lateral,yaw]}
            residual={'base_delta':[forward-distance,-lateral,error]}
            arrived=abs(forward-distance)<.015 and abs(error)<.02
        elif tool in ('move_to','move_eef_chunk','prepare_grasp','replan_grasp'):
            # Public call poses use xyzw; no scene pose enters this calculation.
            waypoints=result.get('waypoint_progress') if getattr(self,'motion_progress_only',False) else None
            index=waypoints.get('waypoint_index') if waypoints else None
            if getattr(self,'arm_tools_lock_base',False) and raw=='needs_base_reposition':
                index=result['base_reposition']['waypoint_index']
            item=call['poses'][index-1 if index is not None else -1];target=np.eye(4)
            target[:3,3]=item['position'];target[:3,:3]=Rotation.from_quat(item['quaternion_xyzw']).as_matrix()
            local_goal=before['local_from_base']@target
            target,measured,pe,re=self._r15_pose_error(local_goal)
            requested={'poses':[target.tolist()]};actual={'poses':[measured.tolist()]}
            residual={'position_error_m':pe,'rotation_error_rad':re}
            strict=tool in ('prepare_grasp','replan_grasp')
            arrived=pe<=(.002 if strict else .01) and re<=(.02 if strict else .05)
            if strict:arrived=arrived and result.get('arrived',False)
        elif tool=='return_to_rest':
            indices=[3]+list(ARM);q=after['qpos'][indices];target=np.asarray(REST)[indices]
            flags={k:bool(after.get('rest_flags',{}).get(k,False)) for k in ('ee_rest','robot_rest','is_static')}
            requested={'joints':target.tolist(),'rest_flags':dict.fromkeys(flags,True)}
            actual={'joints':q.tolist(),'rest_flags':flags}
            residual={'joints':(target-q).tolist(),'max_joint_error_rad':float(np.max(abs(target[1:]-q[1:])))}
            arrived=all(flags.values())
        elif tool=='set_gripper':
            requested={'gripper_command':call['gripper']}
            actual={'gripper_command':0 if self.closed else 1,'finger_distance_m':float(np.sum(after['qpos'][13:15])),
                    'closure_command_completed':bool(result.get('closure_command_completed',False))}
            residual={'finger_distance_m':None if call['gripper']==0 else .10-actual['finger_distance_m']}
            arrived=(self.closed and actual['closure_command_completed'] if call['gripper']==0 else not self.closed and abs(residual['finger_distance_m'])<=.002)
        else:raise ValueError('Unknown completion tool')
        if self.ended or self.is_stopped() or raw=='official_terminated':status='official_terminated'
        elif result.get('accepted') and raw in ('completed','arrived') and arrived:status='arrived'
        elif raw=='completed':
            cap={'look':150,'move_base':250,'move_to':300,'move_eef_chunk':300,'return_to_rest':300,'prepare_grasp':190,'replan_grasp':190,'set_gripper':57}[tool]
            status='step_cap_reached' if self.steps-start>=cap else 'blocked'
        elif raw in ('tool_motion_limit','rest_motion_limit','stage_step_limit','episode_step_limit','rest_budget_reserved','grasp_alignment_timeout','wall_time_limit'):
            status='step_cap_reached'
        else:status='blocked'
        if self.r15_carry:
            if tool=='return_to_rest' and status!='arrived':self.r15_rest_requires_replan=True
            elif tool in ('move_to','move_eef_chunk','prepare_grasp','replan_grasp') and status=='arrived':
                if (np.max(abs(after['qpos'][ARM]-before['qpos'][ARM]))>.002
                        or abs(after['qpos'][3]-before['qpos'][3])>.001):self.r15_rest_requires_replan=False
        feedback=dict(tool=tool,status=status,requested=requested,actual=actual,residual=residual,
            detail=raw,requires_replan=bool(self.r15_rest_requires_replan))
        if getattr(self,'motion_progress_only',False) and 'waypoint_progress' in result:
            feedback['waypoint_progress']=result['waypoint_progress']
            feedback['requires_replan']=status not in ('arrived','official_terminated')
        if raw in ('base_motion_required','needs_base_reposition','base_pushed'):feedback['requires_replan']=True
        if result.get('localization_invalidated'):feedback['requires_replan']=True
        if getattr(self,'base_tracking_exit',False) and tool=='move_base':
            feedback['requires_replan']=bool(result.get('requires_replan',False))
        result=dict(result,reason=status,arrived=status=='arrived',motion_completion=feedback,steps_used=self.steps-start)
        self._flush()
        return result

    def _r15_cartesian(self,local_goals,cap):
        before=self.steps;result=None;stages=[]
        for local_goal in local_goals:
            remaining=cap-(self.steps-before)
            if remaining<3:return self._compound_result(self._reject('tool_motion_limit'),before,stages)
            target=np.linalg.inv(self.ports.observe()['local_from_base'])@local_goal
            stage_start=self.steps
            result=FullBOnlineExecutor.move_eef_chunk(self,[target],max_steps=remaining)
            _,_,pe,re=self._r15_pose_error(local_goal)
            if result.get('reason')!='completed' or pe>.01 or re>.05:
                if result.get('reason')=='completed':result=dict(result,reason='cartesian_endpoint_not_arrived',arrived=False)
                stages.append(self._compound_stage('cartesian',stage_start,result))
                break
            stages.append(self._compound_stage('cartesian',stage_start,result))
        if result is None:raise ValueError('Empty Cartesian path')
        return self._compound_result(result,before,stages)

    def move_eef_chunk(self,targets,*,max_steps=150):
        if not self.r15_pregrasp or self.closed:
            return super().move_eef_chunk(targets,max_steps=max_steps)
        from fetch_episode import rigid
        if not 1<=len(targets)<=8:raise ValueError('Expected 1..8 poses')
        measured=self.ports.observe();B=measured['local_from_base'];current=self._r15_pose(measured)
        goals=[]
        for value in targets:
            target=rigid(value);height=max(float(current[2,3]),float(target[2,3])+.15)
            up=current.copy();up[2,3]=height
            horizontal=target.copy();horizontal[2,3]=height
            goals.extend([B@up,B@horizontal,B@target]);current=target
        return self._r15_cartesian(goals,max_steps)

    def return_to_rest(self):
        if not self.r15_carry:return super().return_to_rest()
        if self.r15_rest_requires_replan:return self._reject('rest_requires_pose_change')
        before=self.steps;stages=[]
        if self.closed:
            measured=self.ports.observe();B=measured['local_from_base'];up=self._r15_pose(measured)
            up[2,3]+=.10;back=up.copy();back[0,3]-=.15
            self._r15_rest_prelude=True
            try:result=self._r15_cartesian([B@up,B@back],300)
            finally:self._r15_rest_prelude=False
            stages.append(self._compound_stage('rest_prelude',before,result))
            if result.get('reason')!='completed':return result
        remaining=300-(self.steps-before)
        if remaining<3:return self._compound_result(self._reject('rest_motion_limit'),before,stages)
        rest_start=self.steps
        result=self._run([self.Stage('settling',2),self.Stage('rest',remaining-2)],tool_cap=remaining)
        stages.append(self._compound_stage('rest',rest_start,result))
        return self._compound_result(result,before,stages)
