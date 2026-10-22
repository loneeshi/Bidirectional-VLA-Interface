"""Static grouping of frozen robot code; algorithms and inheritance preserved."""


class BaseAbortGuard:
    """C2-r3 robot-only rules, shared by CPU archive checks and execution."""
    def __init__(self):
        self.phase = None
        self.count = 0
        self.rule = None

    def update(self, phase, command, measured):
        import math
        v, w = command
        vx, vy = measured[:2]
        if not all(math.isfinite(float(x)) for x in (v, w, vx, vy)):
            raise ValueError('nonfinite base proprioception')
        rule = ('A_spin' if phase == 'base_drive' and abs(w) >= .9 else
                'B_slip' if phase in ('base_turn', 'base_settle') and v == 0 and math.hypot(vx, vy) > .08 else None)
        self.count = self.count + 1 if rule and (phase, rule) == (self.phase, self.rule) else int(rule is not None)
        self.phase, self.rule = phase, rule
        return rule if self.count >= 3 else None


class BaseProgressMonitor:
    """CPU contract r32-v1. Robot odometry only; never infers contact.

    20 Hz; four startup samples, six eligible samples per window. Ten percent
    of integrated command is the minimum signed progress. These are synthetic
    engineering thresholds, not physical force/safety guarantees.
    """
    CONTRACT = dict(version='r32-followup-progress-v2', dt=.05, startup_samples=4, window_samples=6,
                    minimum_progress_ratio=.1, numeric_epsilon=1e-9,
                    braking_command_ratio=.8)

    def __init__(self, *, drive_heading=None):
        self.drive_heading = drive_heading
        self.last_step = None
        self.phase = None
        self.previous = None
        self.heading = 0.
        self.samples = []
        self.age = 0
        self.previous_command = None
        self.yaw = 0.

    def update(self, step, phase, command, measured, pose, *, goal_arrived=False):
        c = self.CONTRACT
        values = np.r_[command, measured[:3], pose[:3]]
        if not np.isfinite(values).all():
            self.samples.clear(); self.age = 0
            self.previous = None; self.previous_command = None
            return dict(status='nonfinite', rule=None)
        if self.last_step is not None and step <= self.last_step:
            self.samples.clear(); self.age = 0
            self.previous = None; self.previous_command = None
            return dict(status='stale_or_repeated', rule=None)
        gap = self.last_step is not None and step != self.last_step+1
        self.last_step = step
        axis = 1 if phase == 'base_turn' or (phase == 'base_drive' and abs(command[0])<=c['numeric_epsilon'] and abs(command[1])>c['numeric_epsilon']) else 0
        demand = float(command[axis])
        key = (phase, axis, int(np.sign(demand)))
        changed = key != self.phase or gap
        braking = (self.previous_command is not None and
                   abs(demand) < c['braking_command_ratio']*abs(self.previous_command))
        if changed or braking:
            self.samples.clear(); self.age = 0
            self.previous = None
            self.heading = float(pose[2])
            self.yaw = 0.
        self.phase = key; self.previous_command = demand
        current = np.asarray(pose[:3], dtype=float)
        prior = self.previous
        self.previous = current.copy()
        if goal_arrived or phase not in ('base_drive', 'base_turn') or abs(demand)<=c['numeric_epsilon']:
            self.samples.clear(); self.age = 0
            return dict(status='goal_arrived' if goal_arrived else 'zero_or_inactive', rule=None)
        self.age += 1
        yaw_delta = 0. if prior is None else wrap(current[2]-prior[2])
        self.yaw += yaw_delta
        if prior is None or self.age <= c['startup_samples']:
            return dict(status='braking' if braking else 'startup', rule=None)
        heading = self.heading if self.drive_heading is None else self.drive_heading
        direction = np.array([math.cos(heading), math.sin(heading)])
        displacement = current[:2]-prior[:2]
        along = float(displacement@direction)
        lateral = float(displacement@np.array([-direction[1],direction[0]]))
        signed = (yaw_delta if axis else along)*np.sign(demand)
        measured_progress = (float(measured[2]) if axis else float(np.asarray(measured[:2])@direction))*np.sign(demand)
        self.samples.append((signed, abs(demand)*c['dt'], abs(lateral), measured_progress*c['dt']))
        self.samples = self.samples[-c['window_samples']:]
        actual, expected, sideways, velocity_progress = np.sum(self.samples,axis=0)
        rule = None
        if len(self.samples)==c['window_samples'] and expected>c['numeric_epsilon']:
            if actual < -c['numeric_epsilon']:
                rule='progress_reverse'
            elif actual < c['minimum_progress_ratio']*expected:
                rule='progress_lateral' if not axis and sideways>=c['minimum_progress_ratio']*expected else 'progress_stuck'
        return dict(status='triggered' if rule else 'tracking', rule=rule,
                    signed_progress=float(actual), commanded_progress=float(expected),
                    lateral_progress_m=float(sideways), measured_velocity_progress=float(velocity_progress),
                    measured_command_error=float(velocity_progress-expected),
                    unwrapped_yaw_rad=float(self.yaw), window_samples=len(self.samples))


class BaseResponseMonitor:
    """Robot-only first-order innovation; limits fixed on negative data before contacts.

    The 005/008 base response fit uses each sample's own body-forward axis.
    This diagnoses command tracking discrepancy, never contact or force.
    """
    CONTRACT = dict(version='r32-followup-response-v1',dt=.05,
        linear_rho=.6618654969317679,linear_beta=.33611604511281234,
        angular_rho=.037328111306414766,angular_beta=.9618894594236916,
        linear_innovation_limit=.06,angular_innovation_limit=.07,consecutive=2)

    def __init__(self, *, corroboration=False):
        self.corroboration = bool(corroboration)
        self.previous_pose = None
        self.previous = None
        self.last_step = None
        self.key = None
        self.counts = [0,0]

    def effective_contract(self):
        contract=self.CONTRACT.copy()
        if self.corroboration:
            contract.update(corroboration_version='r33-corroborated-response-v1',
                corroboration_position_model='trapezoidal_prior_and_nominal_velocity')
        return contract

    def update(self, step, phase, command, measured, pose, *, goal_arrived=False):
        if not np.isfinite(np.r_[command,measured[:3],pose[:3]]).all():
            self.previous=None;self.previous_pose=None;self.counts=[0,0];self.key=None
            return dict(status='nonfinite',rule=None)
        if self.last_step is not None and step<=self.last_step:
            self.previous=None;self.previous_pose=None;self.counts=[0,0];self.key=None
            return dict(status='stale_or_repeated',rule=None)
        gap=self.last_step is not None and step!=self.last_step+1
        self.last_step=step
        body=np.array([measured[0]*math.cos(pose[2])+measured[1]*math.sin(pose[2]),measured[2]])
        prior=self.previous;self.previous=body
        prior_pose=self.previous_pose;self.previous_pose=np.asarray(pose[:3],dtype=float).copy()
        key=(phase, int(np.sign(command[0])),int(np.sign(command[1])))
        same_phase=key==self.key and not gap
        if key!=self.key or gap:self.counts=[0,0]
        self.key=key
        if goal_arrived or phase not in ('base_turn','base_drive') or not any(command):
            self.counts=[0,0]
            return dict(status='goal_arrived' if goal_arrived else 'zero_or_inactive',rule=None)
        if prior is None or gap:
            return dict(status='awaiting_previous_sample',rule=None)
        innovations=[];predictions=[]
        for axis,label in enumerate(('linear','angular')):
            c=self.CONTRACT
            predicted=c[label+'_rho']*prior[axis]+c[label+'_beta']*command[axis]
            innovation=float(body[axis]-predicted);innovations.append(innovation);predictions.append(predicted)
            anomalous=command[axis]!=0 and abs(innovation)>c[label+'_innovation_limit']
            self.counts[axis]=self.counts[axis]+1 if anomalous else 0
        rule=next(('response_'+label for axis,label in enumerate(('linear','angular'))
                   if self.counts[axis]>=self.CONTRACT['consecutive']),None)
        evidence={}
        if self.corroboration and prior_pose is not None and same_phase:
            yaw_delta=wrap(pose[2]-prior_pose[2])
            heading=prior_pose[2]+yaw_delta/2
            direction=np.array([math.cos(heading),math.sin(heading)])
            displacement=np.array([float((np.asarray(pose[:2])-prior_pose[:2])@direction),yaw_delta])
            displacement_innovation=displacement/self.CONTRACT['dt']-(prior+np.asarray(predictions))/2
            corroborated=next(('response_corroborated_'+label for axis,label in enumerate(('linear','angular'))
                if command[axis]!=0 and innovations[axis]*np.sign(command[axis]) < -self.CONTRACT[label+'_innovation_limit']
                and displacement_innovation[axis]*np.sign(command[axis]) < -self.CONTRACT[label+'_innovation_limit']),None)
            rule=rule or corroborated
            evidence=dict(corroboration_enabled=True,linear_displacement_innovation_m_s=float(displacement_innovation[0]),
                angular_displacement_innovation_rad_s=float(displacement_innovation[1]))
        return dict(status='triggered' if rule else 'tracking',rule=rule,
            linear_innovation=innovations[0],angular_innovation=innovations[1],
            linear_count=self.counts[0],angular_count=self.counts[1],**evidence)


class BaseResponseDebtMonitor:
    """Open-loop nominal velocity discrepancy, independently opt-in.

    Fixed negative-only calibration plus the existing tool tolerances gives
    .04 m / .04 rad. These sampled integrals are not geometric endpoint errors
    or hard physical safety bounds. Never reseed prediction from slow motion.
    """
    CONTRACT=dict(version='r32-followup-response-debt-v1',dt=.05,
        linear_rho=.6618654969317679,linear_beta=.33611604511281234,
        angular_rho=.037328111306414766,angular_beta=.9618894594236916,
        linear_integral_limit_m=.04,angular_integral_limit_rad=.04)

    def __init__(self):
        self.predicted=None
        self.last_step=None
        self.debt=np.zeros(2)

    def update(self,step,phase,command,measured,pose,*,goal_arrived=False):
        if not np.isfinite(np.r_[command,measured[:3],pose[:3]]).all():
            self.predicted=None;self.debt[:]=0
            return dict(status='nonfinite',rule=None)
        if self.last_step is not None and step<=self.last_step:
            self.predicted=None;self.debt[:]=0
            return dict(status='stale_or_repeated',rule=None)
        gap=self.last_step is not None and step!=self.last_step+1
        self.last_step=step
        body=np.array([measured[0]*math.cos(pose[2])+measured[1]*math.sin(pose[2]),measured[2]])
        if self.predicted is None or gap:
            self.predicted=body;self.debt[:]=0
            return dict(status='awaiting_previous_sample',rule=None)
        c=self.CONTRACT
        self.predicted=np.array([c['linear_rho'],c['angular_rho']])*self.predicted+np.array([c['linear_beta'],c['angular_beta']])*command
        self.debt+=c['dt']*abs(self.predicted-body)
        active=phase in ('base_turn','base_drive') and not goal_arrived
        rule=('response_debt_linear' if active and command[0]!=0 and self.debt[0]>c['linear_integral_limit_m'] else
              'response_debt_angular' if active and command[1]!=0 and self.debt[1]>c['angular_integral_limit_rad'] else None)
        return dict(status='triggered' if rule else 'goal_arrived' if goal_arrived else 'tracking' if active and any(command) else 'zero_or_inactive',
            rule=rule,linear_integral_m=float(self.debt[0]),angular_integral_rad=float(self.debt[1]),
            predicted_linear_velocity=float(self.predicted[0]),predicted_angular_velocity=float(self.predicted[1]))


class BaseMotionAborted(Exception):
    pass


class BaseAbortMixin:
    """Opt-in wrapper; the frozen controller and all its formulas stay intact."""
    def __init__(self, *args, zero_base_velocity, **kwargs):
        self.zero_base_velocity = zero_base_velocity
        self._base_abort_guard = None
        super().__init__(*args, **kwargs)

    def _step(self, action, kind, **info):
        before = self.steps
        result = super()._step(action, kind, **info)
        if self._base_abort_guard is not None and self.steps > before:
            rule = self._base_abort_guard.update(kind, [action[11], action[12]*3.14], self.read_qvel())
            if rule:
                self.zero_base_velocity()  # Set velocity target only; no physics step.
                raise BaseMotionAborted(rule)
        return result

    def move_base(self, turn_rad, forward_m, *, max_steps=80):
        import math
        import numpy as np
        start = self.odometry().copy()
        before = self.steps
        self._base_abort_guard = BaseAbortGuard()
        try:
            return super().move_base(turn_rad, forward_m, max_steps=max_steps)
        except BaseMotionAborted as exc:
            current = self.odometry()
            heading = wrap(start[2] + turn_rad)
            direction = np.array([math.cos(heading), math.sin(heading)])
            delta = odometry(start, current)
            return dict(accepted=True, arrived=False,
                        reason='official_terminated' if self.stopped else 'base_motion_aborted',
                        base_abort_rule=str(exc), steps_used=self.steps-before,
                        requested_turn_rad=float(turn_rad), requested_forward_m=float(forward_m),
                        actual_turn_rad=float(delta[2]), actual_forward_m=float((current[:2]-start[:2])@direction),
                        position_error_m=float(np.linalg.norm(start[:2]+forward_m*direction-current[:2])),
                        rotation_error_rad=abs(wrap(heading-current[2])), official_terminated=self.stopped)
        finally:
            self._base_abort_guard = None

# ========================================================================
# Frozen source: eef_tools_v5.py
"""v5 synchronizes Cartesian and IK8 torso references with one release cursor."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .eef_tools import transform, vector
from .arm import ARM_IDX, periodic_delta, pose_error
from .arm import FetchToolsV4

EEF_TOOLS_V5_VERSION='eef-tool-v5-coupled-reference-r1'


class FetchToolsV5(FetchToolsV4):
    def coupled_schedule(self,plan):
        refs=plan['refs'];poses=np.asarray([self.fk.tcp(refs[0]),*plan['points']])
        speed=np.asarray(self.model['reference_speed_limits'])[:7]
        cost=np.max(abs(np.diff(refs[:,ARM_IDX],axis=0))/speed,axis=1)
        times=np.r_[0,np.cumsum(np.maximum(cost,1e-8))]
        start=max(1,int(np.ceil(times[-1])))
        # Stretch one shared path clock until its unpaused torso reference
        # satisfies the frozen A1 action bound. No separate end-goal clock.
        for count in range(start,201):
            ticks=np.linspace(0,times[-1],count+1)
            heights=np.interp(ticks,times,refs[:,3])
            delta=np.diff(heights)
            previous=np.r_[self.previous_delta[3],delta[:-1]]
            action=(delta-self.rho[-1]*previous)/(.1*self.beta[-1])
            if np.max(abs(action))<=1+1e-9:break
        else:raise ValueError('coupled torso reference exceeds 200-step model horizon')
        rotations=Slerp(times,Rotation.from_matrix(poses[:,:3,:3]))(ticks).as_matrix()
        points=[]
        for i,t in enumerate(ticks):
            point=np.eye(4);point[:3,:3]=rotations[i]
            point[:3,3]=[np.interp(t,times,poses[:,j,3]) for j in range(3)]
            points.append(point)
        return points,heights

    def move_eef_chunk(self,targets,*,max_steps):
        before=self.steps;self.deadline=min(self.max_steps,before+max_steps)
        plan=self._plan(targets)
        if not plan['accepted']:
            return {'arrived':False,'reason':plan['reason'],'steps_used':0,
                    'plan':{k:v for k,v in plan.items() if k not in ('refs','points')}}
        try:points,heights=self.coupled_schedule(plan)
        except ValueError as exc:
            return {'arrived':False,'reason':'reference_timing_infeasible','detail':str(exc),'steps_used':0}
        cursor=0;waits=0;reason='tool_step_limit'
        previous=vector(self.read_qpos(),15).copy()
        while cursor<len(points)-1 and self._available():
            target,solved=self._compensate(points[cursor+1])
            if solved.reason=='jump_rejected':reason='jump_rejected';break
            if not solved.accepted:waits+=1
            target[3]=heights[cursor+1]
            delta=np.zeros(15)
            if solved.accepted:delta[ARM_IDX]=periodic_delta(target,previous)[ARM_IDX]
            delta[3]=target[3]-previous[3]
            self._step(self.inverse_action(delta,previous),'move',reference_index=cursor+1,
                torso_reference_m=float(target[3]),ik_reason=solved.reason,
                reference_clock='shared_release_cursor',
                **({'reference_qpos':target.tolist()} if getattr(self,'execution_lifecycle',False) else {}))
            error=periodic_delta(target,vector(self.read_qpos(),15))
            released=np.max(abs(error[ARM_IDX]))<=self.p.release_rotation and abs(error[3])<=self.p.release_torso
            if solved.accepted and released:cursor+=1
            previous=target.copy()
        goal=transform(targets[-1]);self.last_goal=goal
        pe,re=pose_error(self.fk,self.read_qpos(),goal)
        while cursor==len(points)-1 and (pe>self.p.endpoint_position or re>self.p.endpoint_rotation) and self._available():
            target,solved=self._compensate(goal)
            if solved.reason=='jump_rejected':reason='jump_rejected';break
            if not solved.accepted:waits+=1
            self._step(self.inverse_action(np.zeros(15),target),'endpoint',ik_reason=solved.reason,
                **({'reference_qpos':target.tolist()} if getattr(self,'execution_lifecycle',False) else {}))
            pe,re=pose_error(self.fk,self.read_qpos(),goal)
        arrived=cursor==len(points)-1 and pe<=self.p.endpoint_position and re<=self.p.endpoint_rotation and reason!='jump_rejected'
        if arrived:self.last_plan_endpoint=plan['refs'][-1].copy()
        return {'arrived':bool(arrived),'reason':'ok' if arrived else reason,'steps_used':self.steps-before,
            'position_error_m':pe,'rotation_error_rad':re,'arm_ik_wait_steps':waits,
            'torso_target_m':float(plan['refs'][-1,3]),'reference_count':len(points),
            'reference_completed':cursor==len(points)-1,'reference_clock':'shared_release_cursor'}


# ========================================================================
# Frozen source: eef_tools_v6.py
"""v6: keyframe-local torso trapezoids, shared with Cartesian release progress."""
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from .dynamics import torso_trapezoid


EEF_TOOLS_V6_VERSION='eef-tool-v6-segment-coupled-r1'


class FetchToolsV6(FetchToolsV5):
    def _plan(self,targets):
        targets=list(targets)
        plan=super()._plan(targets)
        if plan['accepted']:
            ends=[];start=0
            for goal in targets:
                end=next(i+1 for i in range(start,len(plan['points']))
                    if np.allclose(plan['points'][i],goal,atol=1e-10,rtol=0))
                ends.append(end);start=end
            plan['keyframe_ends']=ends
        return plan

    def coupled_schedule(self,plan):
        refs=plan['refs'];all_points=[self.fk.tcp(refs[0])];all_z=[float(refs[0,3])]
        start=0;initial=float(self.previous_delta[3])
        for end in plan.get('keyframe_ends',[len(refs)-1]):
            segment={'refs':refs[start:end+1],'points':plan['points'][start:end]}
            cart=super()._schedule(segment)
            profile=torso_trapezoid(float(refs[start,3]),float(refs[end,3]),
                self.rho[-1],self.beta[-1],initial_delta=initial)
            source_z=profile['q'];count=max(len(cart)-1,len(source_z)-1,1)
            # Only stretch this segment, not unrelated constant-height segments.
            while count<=200:
                z=np.interp(np.linspace(0,len(source_z)-1,count+1),np.arange(len(source_z)),source_z)
                dq=np.diff(z);u=(dq-self.rho[-1]*np.r_[initial,dq[:-1]])/(.1*self.beta[-1])
                if np.max(abs(u))<=1+1e-9:break
                count+=1
            if count>200:raise ValueError('segment reference exceeds 200-step horizon')
            ticks=np.linspace(0,len(cart)-1,count+1);poses=np.asarray(cart)
            rotations=Slerp(np.arange(len(cart)),Rotation.from_matrix(poses[:,:3,:3]))(ticks).as_matrix()
            for k,t in enumerate(ticks[1:],1):
                p=np.eye(4);p[:3,:3]=rotations[k]
                p[:3,3]=[np.interp(t,np.arange(len(cart)),poses[:,j,3]) for j in range(3)]
                all_points.append(p);all_z.append(float(z[k]))
            initial=float(z[-1]-z[-2]);start=end
        return all_points,np.asarray(all_z)


# ========================================================================
# Frozen source: eef_tools_v7.py
"""v7 skips actions for already-reached references, retaining every IK check."""
import numpy as np
from .eef_tools import transform, vector
from .arm import ARM_IDX, periodic_delta, pose_error

EEF_TOOLS_V7_VERSION='eef-tool-v7-segment-coupled-release-r1'

class ProgressGuard:
    """CPU reuse of bvi/eef_arm_repair.py ProgressGuard (2026-10-01).

    State belongs to one command and phase. Inputs are robot residuals only.
    """
    def __init__(self, initial, *, reference_progress=False):
        from collections import deque
        self.reference_progress = reference_progress
        self.reference_index = None
        self.last_error = float(initial)
        self.best = float(initial)
        self.window = deque([self.best], maxlen=21)
        self.failures = 0

    def update(self, error, ik_ok, reference_index=None):
        if self.reference_progress and reference_index != self.reference_index:
            self.reference_index = reference_index
            self.best = self.last_error
            self.window.clear()
            self.window.append(self.best)
        self.last_error = float(error)
        self.failures = 0 if ik_ok else self.failures + 1
        self.best = min(self.best, float(error))
        self.window.append(self.best)
        if not self.reference_progress and self.failures >= 5:
            return 'ik_stalled'
        if len(self.window) == 21 and self.window[0] - self.best < .1:
            return 'motion_stalled'
        return None


class FetchToolsV7(FetchToolsV6):
    def move_eef_chunk(self,targets,*,max_steps):
        before=self.steps;self.deadline=min(self.max_steps,before+max_steps)
        plan=self._plan(targets)
        if not plan['accepted']:
            return {'arrived':False,'reason':plan['reason'],'steps_used':0,
                    'plan':{k:v for k,v in plan.items() if k not in ('refs','points')}}
        try:points,heights=self.coupled_schedule(plan)
        except ValueError as exc:
            return {'arrived':False,'reason':'reference_timing_infeasible','detail':str(exc),'steps_used':0}
        cursor=0;waits=0;reason='tool_step_limit'
        previous=vector(self.read_qpos(),15).copy()
        # Default-off: keep every old action, tolerance and release decision.
        guarded = self.p.stall_feedback
        phase = 'tracking'
        def residual():
            pe, re = pose_error(self.fk, self.read_qpos(), transform(targets[-1]))
            return max(pe/self.p.endpoint_position, re/self.p.endpoint_rotation)
        guard = ProgressGuard(residual(), reference_progress=self.p.stall_guard_mode == "reference_progress") if guarded else None
        while cursor<len(points)-1 and self._available():
            target,solved=self._compensate(points[cursor+1])
            if solved.reason=='jump_rejected':reason='jump_rejected';break
            if not solved.accepted:waits+=1
            target[3]=heights[cursor+1]
            # A reference already inside the frozen release tolerances needs
            # no extra environment action. Still solve/check every reference.
            at_reference=periodic_delta(target,vector(self.read_qpos(),15))
            if solved.accepted and np.max(abs(at_reference[ARM_IDX]))<=self.p.release_rotation and abs(at_reference[3])<=self.p.release_torso:
                cursor+=1
                continue
            delta=np.zeros(15)
            if solved.accepted:delta[ARM_IDX]=periodic_delta(target,previous)[ARM_IDX]
            delta[3]=target[3]-previous[3]
            if guard is not None:
                guard_reference = cursor + 1
            self._step(self.inverse_action(delta,previous),'move',reference_index=cursor+1,
                torso_reference_m=float(target[3]),ik_reason=solved.reason,
                reference_clock='shared_release_cursor')
            error=periodic_delta(target,vector(self.read_qpos(),15))
            released=np.max(abs(error[ARM_IDX]))<=self.p.release_rotation and abs(error[3])<=self.p.release_torso
            if solved.accepted and released:cursor+=1
            previous=target.copy()
            if guard is not None:
                stopped = guard.update(residual(), solved.accepted, reference_index=guard_reference)
                if stopped:
                    reason = stopped
                    break
        goal=transform(targets[-1]);self.last_goal=goal
        pe,re=pose_error(self.fk,self.read_qpos(),goal)
        if guarded and cursor==len(points)-1 and reason=='tool_step_limit':
            phase = 'endpoint'
            guard = ProgressGuard(residual(), reference_progress=self.p.stall_guard_mode == "reference_progress")
        while cursor==len(points)-1 and (not guarded or reason=='tool_step_limit') and (pe>self.p.endpoint_position or re>self.p.endpoint_rotation) and self._available():
            target,solved=self._compensate(goal)
            if solved.reason=='jump_rejected':reason='jump_rejected';break
            if not solved.accepted:waits+=1
            self._step(self.inverse_action(np.zeros(15),target),'endpoint',ik_reason=solved.reason)
            pe,re=pose_error(self.fk,self.read_qpos(),goal)
            if guard is not None:
                stopped = guard.update(residual(), solved.accepted, reference_index=cursor)
                if stopped:
                    reason = stopped
                    break
        if guarded and self.stopped: reason='official_terminated'
        arrived=cursor==len(points)-1 and pe<=self.p.endpoint_position and re<=self.p.endpoint_rotation and (reason=='tool_step_limit' if guarded else reason!='jump_rejected')
        if arrived:self.last_plan_endpoint=plan['refs'][-1].copy()
        return {'arrived':bool(arrived),'reason':'ok' if arrived else reason,'steps_used':self.steps-before,
            'position_error_m':pe,'rotation_error_rad':re,'arm_ik_wait_steps':waits,
            'torso_target_m':float(plan['refs'][-1,3]),'reference_count':len(points),
            'reference_completed':cursor==len(points)-1,'reference_clock':'shared_release_cursor',
            **({'execution_phase':phase,'consecutive_ik_failures':guard.failures} if guarded else {})}


# ========================================================================
# Frozen source: eef_tools_v8.py
"""v8 uses robot-mass-derived torso damping during gripper settling."""
import xml.etree.ElementTree as ET
import numpy as np
from .eef_tools import vector


EEF_TOOLS_V8_VERSION='eef-tool-v8-critical-torso-settling-r1'
TORSO_MOVING_MASS=29.157648076433205
TORSO_STIFFNESS=1000.
TORSO_DAMPING=100.
TORSO_VELOCITY_SECONDS=max(0.,(2*np.sqrt(TORSO_MOVING_MASS*TORSO_STIFFNESS)-TORSO_DAMPING)/TORSO_STIFFNESS)


def moving_torso_mass(urdf):
    xml=ET.parse(urdf).getroot();children={}
    for joint in xml.findall('joint'):
        children.setdefault(joint.find('parent').get('link'),[]).append(joint.find('child').get('link'))
    links={node.get('name'):node for node in xml.findall('link')}
    pending=['torso_lift_link'];mass=0.
    while pending:
        name=pending.pop();node=links[name].find('inertial/mass')
        if node is not None:mass+=float(node.get('value'))
        pending.extend(children.get(name,[]))
    return mass


class FetchToolsV8(FetchToolsV7):
    def settling_action(self,reference,previous_action):
        action=super().settling_action(reference,previous_action)
        q=vector(self.read_qpos(),15);velocity=vector(self.read_qvel(),15)
        # Extra velocity feedback brings the nominal scalar robot PD system
        # to critical damping. It is derived from URDF mass, not A1 refitting.
        offset=self.settling.position_gain*(vector(reference,15)[3]-q[3])-TORSO_VELOCITY_SECONDS*velocity[3]
        desired=np.clip(offset/self.settling.official_delta_scale,-self.settling.action_cap,self.settling.action_cap)
        previous=vector(previous_action,13)[10]
        action[10]=np.clip(previous+np.clip(desired-previous,-self.settling.action_change_cap,self.settling.action_change_cap),-self.settling.action_cap,self.settling.action_cap)
        return action


# ========================================================================
# Frozen source: eef_tools_v9.py
"""v9 parks torso at its robot-model stopping position during close alignment."""
import numpy as np
from .eef_tools import vector
from .dynamics import grip_band
from .arm import pose_error

EEF_TOOLS_V9_VERSION='eef-tool-v9-torso-stopping-anchor-r1'
class FetchToolsV9(FetchToolsV8):
    def set_gripper(self,command):
        band=grip_band(command); before=self.steps
        self.deadline=min(self.max_steps,before+self.p.close_align_cap+self.p.close_wait_cap)
        entering=band==0 and grip_band(self.gripper)!=0
        anchor=vector(self.read_qpos(),15).copy()
        if entering:
            # Park at the nominal critically damped stopping position, avoiding
            # the reverse excursion caused by returning to the entry position.
            omega=np.sqrt(TORSO_STIFFNESS/TORSO_MOVING_MASS)
            lo,hi,_=self.fk.limits['torso_lift_joint']
            anchor[3]=np.clip(anchor[3]+vector(self.read_qvel(),15)[3]/omega,lo,hi)
        previous_action=np.zeros(13)
        alignment=None
        if entering:
            pe,re=pose_error(self.fk,self.read_qpos(),self.last_goal)
            while ((pe>(self.p.endpoint_position if self.p.close_uses_endpoint_tolerance else self.p.position_tolerance)
                    or re>(self.p.endpoint_rotation if self.p.close_uses_endpoint_tolerance else self.p.rotation_tolerance)
                    or abs(self.previous_delta[3])>=self.p.torso_stable)
                   and self.steps-before<self.p.close_align_cap and self._available()):
                target,solved=self._compensate(self.last_goal)
                if solved.reason=='jump_rejected':
                    return {'accepted':False,'reason':'jump_rejected','steps_used':self.steps-before}
                # Keep these references fixed throughout this command. Arm IK
                # still uses actual measured torso height, never an ideal height.
                target[[3,4,6]]=anchor[[3,4,6]]
                action=self.settling_action(target,previous_action)
                self._step(action,'close_align',ik_reason=solved.reason,
                           control='damped_native_position',reference_qpos=target.tolist(),
                           torso_anchor_m=float(anchor[3]))
                previous_action=action
                pe,re=pose_error(self.fk,self.read_qpos(),self.last_goal)
            alignment={'position_error_m':pe,'rotation_error_rad':re,
                       'torso_delta_m':abs(float(self.previous_delta[3])),
                       'steps_used':self.steps-before,'torso_anchor_m':float(anchor[3])}
            if (pe>(self.p.endpoint_position if self.p.close_uses_endpoint_tolerance else self.p.position_tolerance)
                    or re>(self.p.endpoint_rotation if self.p.close_uses_endpoint_tolerance else self.p.rotation_tolerance)
                    or abs(self.previous_delta[3])>=self.p.torso_stable or not self._available()):
                return {'accepted':False,'reason':'close_alignment_failed','alignment':alignment,
                        'steps_used':self.steps-before}
        self.gripper=float(command)
        # Hold a single achieved arm posture during finger motion. The torso
        # anchor persists from command entry; it is never reset at every step.
        hold=vector(self.read_qpos(),15).copy(); hold[[3,4,6]]=anchor[[3,4,6]]
        wait_start=self.steps; stable=0; previous=float(self.read_finger())
        cap=self.p.close_wait_cap if entering else 1
        while self.steps-wait_start<cap and self._available():
            action=self.settling_action(hold,previous_action)
            self._step(action,'gripper',control='damped_native_position',
                       reference_qpos=hold.tolist(),torso_anchor_m=float(anchor[3]))
            previous_action=action
            current=float(self.read_finger())
            stable=stable+1 if abs(current-previous)<self.p.finger_stable else 0
            previous=current
            if entering and stable>=2:
                break
        return {'accepted':self.steps>wait_start,
                'reason':'ok' if self.steps>wait_start else 'terminal_or_budget',
                'steps_used':self.steps-before,'alignment':alignment,'enter_closed':entering,
                'command':self.gripper,'finger_distance_m':previous,
                'stable':stable>=2 if entering else None,'control':'damped_native_position'}


# ========================================================================
# Frozen source: eef_tools_v10.py
"""v10 removes regularization bias only for precision close alignment."""
from .eef_tools import vector
from .dynamics import grip_band
from .arm import SolutionV3, nearest_continuous, pose_error

EEF_TOOLS_V10_VERSION='eef-tool-v10-unbiased-close-ik-r1'
class FetchToolsV10(FetchToolsV9):
    def set_gripper(self,command):
        self._precision_close=grip_band(command)==0 and grip_band(self.gripper)!=0
        try:return super().set_gripper(command)
        finally:self._precision_close=False

    def _compensate(self,point):
        if not getattr(self,'_precision_close',False):return super()._compensate(point)
        measured=vector(self.read_qpos(),15)
        seed=self.last_arm_target.copy();seed[[0,1,2,3,4,6]]=measured[[0,1,2,3,4,6]]
        raw=self.ik7.raw.solve(point,seed)
        q=nearest_continuous(raw.qpos,seed)
        pe,re=pose_error(self.fk,q,point)
        jumped,dr,dz=self.ik7.jump(q,seed)
        reason='jump_rejected' if jumped else 'ok' if raw.accepted else 'ik_not_found'
        solved=SolutionV3(reason=='ok',q,pe,re,reason,True,dr,dz,int(raw.accepted),int(jumped),('precision_alignment_unregularized',))
        self.ik_records.append(solved.summary())
        if solved.accepted:self.last_arm_target=q.copy()
        target=self.last_arm_target.copy();target[[0,1,2,3,4,6]]=measured[[0,1,2,3,4,6]]
        return target,solved


# ========================================================================
# Frozen source: eef_tools_v11.py
"""v11 uses measured torso velocity in exact nominal sampled PD inversion."""
import numpy as np
from scipy.linalg import expm
from .eef_tools import vector


EEF_TOOLS_V11_VERSION='eef-tool-v11-measured-torso-momentum-r1'

def torso_response(dt):
    # q relative to its measured value at the beginning of this control step.
    # qdd = K/m*(0.1*u-q) - D/m*qdot. Augment with constant normalized u.
    a=np.array([[0.,1.,0.],[-TORSO_STIFFNESS/TORSO_MOVING_MASS,-TORSO_DAMPING/TORSO_MOVING_MASS,.1*TORSO_STIFFNESS/TORSO_MOVING_MASS],[0.,0.,0.]])
    transition=expm(a*dt)
    return float(transition[0,1]),float(transition[0,2])

class FetchToolsV11(FetchToolsV10):
    def inverse_action(self,delta,feedback_reference):
        action=super().inverse_action(delta,feedback_reference)
        q=vector(self.read_qpos(),15);v=vector(self.read_qvel(),15)
        coast,gain=torso_response(self.control_dt)
        desired=delta[3]+self.p.feedback*(feedback_reference[3]-q[3])
        action[10]=np.clip((desired-coast*v[3])/gain,-1.,1.)
        return action


# ========================================================================
# Frozen source: eef_tools_v12.py
"""E1-B robot-only executor: measured boundaries and attempt-relative base control.

No scene, map, world-pose, simulator or provider dependencies.
"""
import math
import numpy as np
from .eef_tools import vector, validate_official_controller


EEF_TOOLS_V12_VERSION = 'eef-tool-v12-odometry-base-r1'
LINEAR_LIMIT = .5
ANGULAR_LIMIT = 1.
BASE_STEP_CAP = 80


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def odometry(initial, current):
    """URDF root x/y translation followed by root z rotation, in initial base."""
    start, q = vector(initial, 3), vector(current, 3)
    c, s = math.cos(start[2]), math.sin(start[2])
    dx, dy = q[:2] - start[:2]
    return np.array([c*dx+s*dy, -s*dx+c*dy, wrap(q[2]-start[2])])


def validate_base_runtime(controller, control_freq):
    validate_official_controller(controller)
    cfg = controller.controllers['base'].config
    if (tuple(cfg.joint_names) != ('root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint')
            or not cfg.normalize_action or not np.allclose(cfg.lower, [-1., -3.14])
            or not np.allclose(cfg.upper, [1., 3.14]) or float(control_freq) != 20.):
        raise ValueError('frozen base mapping or measured 20 Hz control frequency mismatch')


class FetchToolsV12(FetchToolsV11):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not math.isclose(self.control_dt, .05, abs_tol=1e-12):
            raise ValueError('v12 requires verified 20 Hz control frequency')
        self._initial_base = vector(self.read_qpos(), 15)[:3].copy()
        self._base_active = False

    def odometry(self):
        return odometry(self._initial_base, self.read_qpos()[:3])

    def _sync_boundary(self):
        measured = vector(self.read_qpos(), 15).copy()
        self.last_plan_endpoint = measured
        self.last_arm_target = measured.copy()

    def move_eef_chunk(self, targets, *, max_steps):
        try:
            return super().move_eef_chunk(targets, max_steps=max_steps)
        finally:
            self._sync_boundary()

    def set_gripper(self, command):
        try:
            return super().set_gripper(command)
        finally:
            self._sync_boundary()

    def return_to_rest(self):
        try:
            return super().return_to_rest()
        finally:
            self._sync_boundary()

    def _step(self, action, kind, **info):
        # Shared authoritative action counter, including base steps and stops.
        if not self._available():
            return False
        action = vector(action, 13)
        if np.max(abs(action)) > 1+1e-10 or (np.any(action[11:] != 0) and not self._base_active):
            raise ValueError('action bound or unauthorized base channel')
        before = vector(self.read_qpos(), 15).copy()
        status = self.step_callback(action)
        self.previous_delta = vector(self.read_qpos(), 15)-before
        self.steps += 1
        self.stopped = bool(status.get('terminated') or status.get('truncated'))
        self.trace.append({'step': self.steps, 'kind': kind, 'qpos': self.read_qpos().tolist(),
                           'action': action.tolist(), 'delta': self.previous_delta.tolist(), **info})
        return not self.stopped

    def move_base(self, turn_rad, forward_m, *, max_steps=80):
        if (type(turn_rad) not in (float, int) or type(forward_m) not in (float, int)
                or not math.isfinite(turn_rad) or not math.isfinite(forward_m)
                or abs(turn_rad)>math.pi or abs(forward_m)>1.5):
            raise ValueError('move_base requires finite |turn_rad|<=pi and |forward_m|<=1.5')
        if type(max_steps) is not int or not 0 <= max_steps <= BASE_STEP_CAP:
            raise ValueError('base step budget must be an integer from 0 to 80')
        before = self.steps
        self.deadline = min(self.max_steps, before+max_steps)
        start = self.odometry()
        heading = wrap(start[2]+turn_rad)
        direction = np.array([math.cos(heading), math.sin(heading)])
        target = start[:2]+forward_m*direction
        hold = vector(self.read_qpos(), 15).copy()
        previous_action = np.zeros(13)
        phase = 'turn'
        turn_progress = 0.
        last_yaw = start[2]
        arrived = False
        self._base_active = True
        try:
            while not self.stopped:
                current = self.odometry()
                turn_progress += wrap(current[2]-last_yaw)
                last_yaw = current[2]
                yaw_error = wrap(heading-current[2])
                turn_error = turn_rad-turn_progress
                if phase == 'turn' and abs(turn_error) <= .03:
                    phase = 'drive'
                xy_error = float(np.linalg.norm(target-current[:2]))
                if phase == 'drive' and xy_error <= .02 and abs(yaw_error) <= .03:
                    arrived = True
                    break
                if not self._available():
                    break
                linear = 0.
                angular = float(np.clip(2*(turn_error if phase=='turn' else yaw_error), -ANGULAR_LIMIT, ANGULAR_LIMIT))
                if phase == 'drive' and xy_error > .02:
                    # Signed along-track correction supports backward and overshoot.
                    error = target-current[:2]
                    along = float(error @ direction)
                    sign = 1. if along >= 0 else -1.
                    desired = math.atan2(error[1], error[0])+(math.pi if sign < 0 else 0.)
                    steering = wrap(desired-current[2])
                    angular = float(np.clip(2*steering, -ANGULAR_LIMIT, ANGULAR_LIMIT))
                    linear = sign*min(LINEAR_LIMIT, 2*xy_error)*max(0., math.cos(steering))
                    if abs(steering)>math.pi/4:
                        linear = 0.
                action = self.settling_action(hold, previous_action)
                action[11:] = [linear, angular/3.14]
                self._step(action, 'base_'+phase, reference_qpos=hold.tolist())
                previous_action = action
            current = self.odometry()
            delta = odometry(start, current)
            return {'accepted': True, 'arrived': arrived,
                    'reason': 'official_terminated' if self.stopped else 'ok' if arrived else 'tool_step_limit',
                    'steps_used': self.steps-before, 'requested_turn_rad': float(turn_rad),
                    'requested_forward_m': float(forward_m), 'actual_turn_rad': float(delta[2]),
                    'actual_forward_m': float((current[:2]-start[:2]) @ direction),
                    'position_error_m': float(np.linalg.norm(target-current[:2])),
                    'rotation_error_rad': abs(wrap(heading-current[2])), 'official_terminated': self.stopped}
        finally:
            self._base_active = False
            self._sync_boundary()
            # Last EEF goal belonged to the old base frame; close at measured TCP.
            self.last_goal = self.fk.tcp(self.read_qpos())


# ========================================================================
# Frozen source: eef_tools_v13.py
"""v13: proprioceptive settling and braking-curve base motion. No scene inputs."""
import math
import numpy as np
from .eef_tools import vector


EEF_TOOLS_V13_VERSION = 'eef-tool-v13-settle-braking-r1'
# Derived by scripts/prepare_eef_base_v13_cpu.py from contact-free cases 5/8.
LINEAR_DECEL = 0.84
ANGULAR_DECEL = 4.81
LINEAR_LIMIT = .5
ANGULAR_LIMIT = 1.
STILL_LINEAR = .02
STILL_ANGULAR = .03
STILL_READINGS = 2
TAIL_GAIN = 2.


def braking_speed(error, deceleration, limit, tolerance):
    magnitude = abs(float(error))
    if magnitude <= tolerance:
        return 0.
    tail = max(2*tolerance, (.05/(1-.6618654969317679)+.05)*limit)
    speed = TAIL_GAIN*magnitude if magnitude <= tail else math.sqrt(2*deceleration*magnitude)
    return math.copysign(min(limit, speed), error)


def continuous_braking_speed(error, deceleration, limit, tolerance):
    """Opt-in continuous envelope with the existing limits and deadband."""
    remaining = max(0., abs(float(error)) - tolerance)
    speed = min(limit, TAIL_GAIN*remaining, math.sqrt(2*deceleration*remaining))
    return math.copysign(speed, error)


class FetchToolsV13(FetchToolsV12):
    def __init__(self, *args, settle_enabled=True, braking_enabled=True,
                 base_heading_hold=False, continuous_base_speed=False,
                 base_progress_monitor=False, base_progress_abort=False, base_response_monitor=False,
                 base_response_debt_monitor=False, base_response_corroboration=False,
                 execution_lifecycle=False, zero_base_velocity=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_response_debt_monitor = base_response_debt_monitor
        self._response_debt_monitor = None
        self._response_debt_samples = []
        self.base_response_corroboration = base_response_corroboration
        if base_response_corroboration and not base_response_monitor:
            raise ValueError("response corroboration requires response monitor")
        self.base_response_monitor = base_response_monitor
        self._response_monitor = None
        self._response_samples = []
        self.base_progress_monitor = base_progress_monitor
        self.base_progress_abort = base_progress_abort
        self.execution_lifecycle = execution_lifecycle
        if zero_base_velocity is not None:
            self.zero_base_velocity = zero_base_velocity
        if base_progress_abort and (not (base_progress_monitor or base_response_monitor or base_response_debt_monitor) or not callable(getattr(self, 'zero_base_velocity', None))):
            raise ValueError('progress abort requires monitor and zero_base_velocity callback')
        self._progress_monitor = None
        self._progress_samples = []
        self._progress_zeroed = False
        self.settle_enabled = settle_enabled
        self.braking_enabled = braking_enabled
        self.base_heading_hold = base_heading_hold
        self.continuous_base_speed = continuous_base_speed
        self.call_records = []
        self._guard_deadline = self.max_steps

    def _available(self):
        return super()._available() and self.steps < self._guard_deadline

    def _step(self, action, kind, **info):
        before=self.steps
        result=super()._step(action,kind,**info)
        if self.steps>before:
            velocity=vector(self.read_qvel(),15)
            self.trace[-1]['qvel']=velocity.tolist()
            self.trace[-1]['command_base_velocity']=[float(action[11]),float(action[12]*3.14)]
            self.trace[-1]['measured_root_velocity']=velocity[:3].tolist()
        if self.steps>before and getattr(self, '_response_monitor', None) is not None:
            velocity=vector(self.read_qvel(),15).copy();yaw=self._initial_base[2]
            velocity[:2]=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])@velocity[:2]
            pose=self.odometry()
            sample=self._response_monitor.update(self.steps,kind,[action[11],action[12]*3.14],velocity,pose,
                goal_arrived=bool((kind=='base_turn' and abs(wrap(self._progress_heading-pose[2]))<=.03) or
                    (kind=='base_drive' and np.linalg.norm(self._progress_target-pose[:2])<=.02 and
                     abs(wrap(self._progress_heading-pose[2]))<=.03)))
            self._response_samples.append(dict(step=self.steps,phase=kind,**sample))
        if self.steps>before and getattr(self, '_response_debt_monitor', None) is not None:
            velocity=vector(self.read_qvel(),15).copy();yaw=self._initial_base[2]
            velocity[:2]=np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])@velocity[:2]
            pose=self.odometry()
            sample=self._response_debt_monitor.update(self.steps,kind,[action[11],action[12]*3.14],velocity,pose,
                goal_arrived=bool((kind=='base_turn' and abs(wrap(self._progress_heading-pose[2]))<=.03) or
                    (kind=='base_drive' and np.linalg.norm(self._progress_target-pose[:2])<=.02 and
                     abs(wrap(self._progress_heading-pose[2]))<=.03)))
            self._response_debt_samples.append(dict(step=self.steps,phase=kind,**sample))
        if self.steps>before and getattr(self, '_progress_monitor', None) is not None:
            velocity = vector(self.read_qvel(),15).copy()
            yaw = self._initial_base[2]
            velocity[:2] = np.array([[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]])@velocity[:2]
            sample = self._progress_monitor.update(self.steps,kind,
                [action[11],action[12]*3.14],velocity,self.odometry(),
                goal_arrived=bool((kind=='base_turn' and abs(wrap(self._progress_heading-self.odometry()[2]))<=.03) or
                    (kind=='base_drive' and np.linalg.norm(self._progress_target-self.odometry()[:2])<=.02 and
                     abs(wrap(self._progress_heading-self.odometry()[2]))<=.03)))
            self._progress_samples.append(dict(step=self.steps,phase=kind,**sample))
            if sample['rule'] and getattr(self, 'base_progress_abort', False):
                self.zero_base_velocity()
                self._progress_zeroed = True
                raise BaseMotionAborted(sample['rule'])
        if (self.steps>before and getattr(self, '_response_monitor', None) is not None
                and self._response_samples[-1]['rule'] and getattr(self, 'base_progress_abort', False)):
            self.zero_base_velocity();self._progress_zeroed=True
            raise BaseMotionAborted(self._response_samples[-1]['rule'])
        if (self.steps>before and getattr(self, '_response_debt_monitor', None) is not None
                and self._response_debt_samples[-1]['rule'] and getattr(self, 'base_progress_abort', False)):
            self.zero_base_velocity();self._progress_zeroed=True
            raise BaseMotionAborted(self._response_debt_samples[-1]['rule'])
        return result

    def _execution_feedback(self, result, name, before, entry):
        if not self.execution_lifecycle:
            return result
        samples = [t for t in self.trace if t['step']>before]
        tracking = []
        for t in samples:
            if 'reference_qpos' in t:
                error = periodic_delta(vector(t['reference_qpos'],15),vector(t['qpos'],15))
                tracking.append(dict(step=t['step'],phase=t['kind'],
                    reference_index=t.get('reference_index'),
                    joint_error=error[3:].tolist(), max_joint_error=float(np.max(abs(error[3:]))),
                    remaining_steps=max(0,self.deadline-t['step'])))
        reason = result.get('reason')
        state = ('official_terminated' if self.stopped else
                 'aborted' if reason=='base_motion_aborted' else
                 'step_budget_exhausted' if self.steps>=self.max_steps else
                 'deadline_exhausted' if reason=='tool_step_limit' else
                 'completed' if reason=='ok' else 'rejected' if result.get('accepted') is False else 'ended')
        events = [dict(state='started',step=before)]
        if samples: events.append(dict(state='executing',step=samples[0]['step']))
        events.append(dict(state=state,step=self.steps))
        velocity = vector(self.read_qvel(),15)
        result['execution_lifecycle'] = dict(command=name,start_step=before,end_step=self.steps,
            state=state,events=events,entry_robot_joints=entry['qpos'][3:],exit_robot_joints=vector(self.read_qpos(),15)[3:].tolist(),
            tracking=tracking,tracking_basis='executed_reference_and_measured_joints',
            physical_stopped=bool(np.linalg.norm(velocity[:2])<STILL_LINEAR and abs(velocity[2])<STILL_ANGULAR),
            physical_stopped_scope='base_instantaneous_velocity_only',measured_at_step=self.steps,
            velocity_target_zeroed=bool(self._progress_zeroed))
        return result

    def _snapshot(self):
        return {'qpos':vector(self.read_qpos(),15).tolist(),
                'qvel':vector(self.read_qvel(),15).tolist(), 'step':self.steps}

    def settle_base(self, hold, phase):
        start = self.steps
        consecutive = 0
        if not self.settle_enabled:
            return True
        previous = np.zeros(13)
        while self._available():
            action = self.settling_action(hold, previous)
            action[11:] = 0.
            self._step(action, 'base_settle', phase=phase, reference_qpos=hold.tolist())
            previous = action
            velocity = vector(self.read_qvel(),15)
            below = np.linalg.norm(velocity[:2]) < STILL_LINEAR and abs(velocity[2]) < STILL_ANGULAR
            consecutive = consecutive+1 if below else 0
            if self.stopped:
                return False
            if consecutive >= STILL_READINGS:
                self.trace[-1]['settle_passed'] = True
                self.trace[-1]['settle_steps'] = self.steps-start
                return True
        return False

    def prepare_tool(self, max_steps):
        """Common call prelude, including first call of an attempt. Not check_path."""
        self.deadline = self._guard_deadline = min(self.max_steps,self.steps+max_steps)
        return self.settle_base(vector(self.read_qpos(),15).copy(), 'tool_entry')

    def _guarded(self, name, budget, callback):
        self.execution_lifecycle = getattr(self, 'execution_lifecycle', False)
        before = self.steps;entry=self._snapshot()
        self._progress_zeroed = False
        try:
            if not self.prepare_tool(budget):
                result={'accepted':False,'arrived':False,'reason':'official_terminated' if self.stopped else 'base_not_settled'}
            else:
                result=callback(max(0,budget-(self.steps-before)))
            result['steps_used']=self.steps-before
            return self._execution_feedback(result,name,before,entry) if self.execution_lifecycle else result
        finally:
            self.call_records.append({'tool':name,'entry':entry,'exit':self._snapshot(),
                'settle_steps':sum(t['kind']=='base_settle' for t in self.trace if t['step']>before)})
            self._guard_deadline=self.max_steps
            self._sync_boundary()

    def move_eef_chunk(self, targets, *, max_steps):
        return self._guarded('move_eef_chunk',max_steps,
            lambda left:super(FetchToolsV13,self).move_eef_chunk(targets,max_steps=left))

    def set_gripper(self, command):
        if type(command) not in (int,float) or not math.isfinite(command) or not 0<=command<=1:
            raise ValueError('gripper must be finite in [0,1]')
        return self._guarded('set_gripper',15,lambda left:super(FetchToolsV13,self).set_gripper(command))

    def return_to_rest(self):
        return self._guarded('return_to_rest',self.p.rest_cap,lambda left:super(FetchToolsV13,self).return_to_rest())

    def _speed(self,error,linear):
        limit=LINEAR_LIMIT if linear else ANGULAR_LIMIT
        if self.continuous_base_speed:
            return continuous_braking_speed(error,LINEAR_DECEL if linear else ANGULAR_DECEL,limit,.02 if linear else .03)
        if not self.braking_enabled:
            return float(np.clip(2*error,-limit,limit))
        return braking_speed(error,LINEAR_DECEL if linear else ANGULAR_DECEL,limit,.02 if linear else .03)

    def move_base(self, turn_rad, forward_m, *, max_steps=80):
        self.base_response_debt_monitor = getattr(self, 'base_response_debt_monitor', False)
        self.base_response_monitor = getattr(self, 'base_response_monitor', False)
        self.base_progress_monitor = getattr(self, 'base_progress_monitor', False)
        self.base_progress_abort = getattr(self, 'base_progress_abort', False)
        self.execution_lifecycle = getattr(self, 'execution_lifecycle', False)
        if (type(turn_rad) not in (int,float) or type(forward_m) not in (int,float)
                or not math.isfinite(turn_rad) or not math.isfinite(forward_m)
                or abs(turn_rad)>math.pi or abs(forward_m)>1.5):
            raise ValueError('move_base requires finite |turn_rad|<=pi and |forward_m|<=1.5')
        if type(max_steps) is not int or not 0<=max_steps<=80:
            raise ValueError('max_steps must be integer in [0,80]')
        before=self.steps;entry=self._snapshot();start=self.odometry()
        self.deadline=self._guard_deadline=min(self.max_steps,before+max_steps)
        hold=vector(self.read_qpos(),15).copy();previous=np.zeros(13)
        heading=wrap(start[2]+turn_rad);direction=np.array([math.cos(heading),math.sin(heading)])
        target=start[:2]+forward_m*direction
        self._base_active=True;arrived=False;reason='tool_step_limit'
        self._response_debt_monitor = BaseResponseDebtMonitor() if self.base_response_debt_monitor else None
        self._response_debt_samples = []
        self._response_monitor = BaseResponseMonitor(corroboration=getattr(self,"base_response_corroboration",False)) if self.base_response_monitor else None
        self._response_samples = []
        self._progress_monitor = BaseProgressMonitor(drive_heading=heading) if self.base_progress_monitor else None
        self._progress_heading=heading;self._progress_target=target.copy()
        self._progress_samples = []; self._progress_zeroed = False
        try:
            if not self.settle_base(hold,'tool_entry'):
                reason='base_not_settled'
            else:
                # Keep the original entry-relative metric goal through settling.
                progress=wrap(self.odometry()[2]-start[2]);last=self.odometry()[2]
                while self._available():
                    current=self.odometry();progress+=wrap(current[2]-last);last=current[2]
                    error=turn_rad-progress
                    if abs(error)<=.03:break
                    action=self.settling_action(hold,previous);action[11:]=[0.,self._speed(error,False)/3.14]
                    self._step(action,'base_turn',reference_qpos=hold.tolist());previous=action
                turn_ok=abs(wrap(heading-self.odometry()[2]))<=.03
                if turn_ok and not self.stopped:
                    settled=self.settle_base(hold,'before_drive' if forward_m else 'turn_exit')
                    if settled and forward_m:
                        previous=np.zeros(13)
                        while self._available():
                            current=self.odometry();error=target-current[:2];distance=float(np.linalg.norm(error))
                            yaw=wrap(heading-current[2])
                            if self.base_heading_hold:
                                remaining=float(error@direction)
                                # Preserve the requested travel direction. Crossing its
                                # endpoint never starts a reverse correction or chases
                                # lateral displacement with a new target bearing.
                                progress_complete=math.copysign(1.,forward_m)*remaining<=.02
                                if progress_complete and abs(yaw)<=.03:break
                            if distance<=.02 and abs(yaw)<=.03:break
                            linear=0.;angular=self._speed(yaw,False)
                            if self.base_heading_hold:
                                if not progress_complete and abs(yaw)<=math.pi/4:
                                    linear=self._speed(remaining,True)*max(0.,math.cos(yaw))
                            elif distance>.02:
                                sign=1. if error@direction>=0 else -1.
                                steer=wrap(math.atan2(error[1],error[0])+(math.pi if sign<0 else 0.)-current[2])
                                angular=self._speed(steer,False)
                                if abs(steer)<=math.pi/4:
                                    linear=self._speed(sign*distance,True)*max(0.,math.cos(steer))
                            action=self.settling_action(hold,previous);action[11:]=[linear,angular/3.14]
                            self._step(action,'base_drive',reference_qpos=hold.tolist());previous=action
                        if not self.stopped:
                            settled=self.settle_base(hold,'drive_exit')
                    current=self.odometry()
                    arrived=bool(settled and np.linalg.norm(target-current[:2])<=.02 and abs(wrap(heading-current[2]))<=.03)
            current=self.odometry();delta=odometry(start,current)
            if self.stopped:reason='official_terminated';arrived=False
            elif arrived:reason='ok'
            elif self.steps<self.deadline:reason='residual_after_settle'
            result={'accepted':True,'arrived':arrived,'reason':reason,'steps_used':self.steps-before,
                'requested_turn_rad':float(turn_rad),'requested_forward_m':float(forward_m),
                'actual_turn_rad':float(delta[2]),'actual_forward_m':float((current[:2]-start[:2])@direction),
                'position_error_m':float(np.linalg.norm(target-current[:2])),
                'rotation_error_rad':abs(wrap(heading-current[2])),'official_terminated':self.stopped}
            if self.base_progress_monitor:
                result['base_progress'] = dict(contract=BaseProgressMonitor.CONTRACT.copy(),
                    samples=self._progress_samples,trigger_rule=next((x['rule'] for x in self._progress_samples if x['rule']),None),
                    velocity_target_zeroed=self._progress_zeroed)
            if self.base_response_monitor:
                result['base_response']=dict(contract=self._response_monitor.effective_contract(),samples=self._response_samples,
                    trigger_rule=next((x['rule'] for x in self._response_samples if x['rule']),None),velocity_target_zeroed=self._progress_zeroed)
            if self.base_response_debt_monitor:
                result['base_response_debt']=dict(contract=BaseResponseDebtMonitor.CONTRACT.copy(),samples=self._response_debt_samples,
                    trigger_rule=next((x['rule'] for x in self._response_debt_samples if x['rule']),None),velocity_target_zeroed=self._progress_zeroed)
            return self._execution_feedback(result,'move_base',before,entry) if self.execution_lifecycle else result
        except BaseMotionAborted as exc:
            if not self.base_progress_monitor and not self.execution_lifecycle and not self.base_response_monitor and not self.base_response_debt_monitor:
                raise
            current=self.odometry();delta=odometry(start,current)
            result=dict(accepted=True,arrived=False,reason='official_terminated' if self.stopped else 'base_motion_aborted',
                base_abort_rule=str(exc),steps_used=self.steps-before,
                requested_turn_rad=float(turn_rad),requested_forward_m=float(forward_m),
                actual_turn_rad=float(delta[2]),actual_forward_m=float((current[:2]-start[:2])@direction),
                position_error_m=float(np.linalg.norm(target-current[:2])),
                rotation_error_rad=abs(wrap(heading-current[2])),official_terminated=self.stopped)
            if self.base_progress_monitor:
                result['base_progress']=dict(contract=BaseProgressMonitor.CONTRACT.copy(),samples=self._progress_samples,
                    trigger_rule=next((x['rule'] for x in self._progress_samples if x['rule']),None),velocity_target_zeroed=True)
            self._progress_zeroed=True
            if self.base_response_monitor:
                result['base_response']=dict(contract=self._response_monitor.effective_contract(),samples=self._response_samples,
                    trigger_rule=next((x['rule'] for x in self._response_samples if x['rule']),None),velocity_target_zeroed=self._progress_zeroed)
            if self.base_response_debt_monitor:
                result['base_response_debt']=dict(contract=BaseResponseDebtMonitor.CONTRACT.copy(),samples=self._response_debt_samples,
                    trigger_rule=next((x['rule'] for x in self._response_debt_samples if x['rule']),None),velocity_target_zeroed=self._progress_zeroed)
            return self._execution_feedback(result,'move_base',before,entry) if self.execution_lifecycle else result
        finally:
            self._response_debt_monitor = None
            self._response_monitor = None
            self._progress_monitor = None
            self._base_active=False;self._guard_deadline=self.max_steps;self._sync_boundary()
            self.last_goal=self.fk.tcp(self.read_qpos())
            self.call_records.append({'tool':'move_base','entry':entry,'exit':self._snapshot(),
                'settle_steps':sum(t['kind']=='base_settle' for t in self.trace if t['step']>before),
                'phase_steps':{k:sum(t['kind']==k for t in self.trace if t['step']>before) for k in ['base_settle','base_turn','base_drive']}})

