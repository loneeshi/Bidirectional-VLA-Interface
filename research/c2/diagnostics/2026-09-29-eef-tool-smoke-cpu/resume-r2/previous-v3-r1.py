"""Robot-only end-effector tools, smoke/E1 revision 1. No scene or simulator imports.

Official rest/static flags are supplied by the runtime; CPU flags must be labelled
as surrogate estimates. v1/v2 modules are immutable dependencies, never patched.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp
from .eef_tools import ARM, REST, FETCH_ORDER, FetchIK, densify, transform, vector
from .eef_response_model import QIND
from .eef_streaming import grip_band
from .eef_streaming_v2 import torso_trapezoid

VERSION = 'eef-tool-v3-smoke-r1'
ARM_IDX = np.asarray([FETCH_ORDER.index(n) for n in ARM])
CONTINUOUS = np.array([8, 10, 12])
ROTARY = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12])


@dataclass(frozen=True)
class ParametersV3:
    previous_weight: float = .005
    rest_weight: float = .02
    position_tolerance: float = .002
    rotation_tolerance: float = .02
    rotation_jump: float = .5
    torso_jump: float = .10
    translation_sample: float = .02
    rotation_sample: float = .1
    max_nfev: int = 180
    feedback: float = 1.
    release_rotation: float = .05
    release_torso: float = .01
    endpoint_position: float = .01
    endpoint_rotation: float = .05
    close_align_cap: int = 10
    close_wait_cap: int = 5
    finger_stable: float = .001
    torso_stable: float = .001
    rest_cap: int = 60


def periodic_delta(q, reference, indices=ROTARY):
    out = np.asarray(q, dtype=float) - np.asarray(reference, dtype=float)
    out = out.copy()
    out[indices] = (out[indices] + np.pi) % (2*np.pi) - np.pi
    return out


def nearest_continuous(q, reference):
    out = np.asarray(q, dtype=float).copy()
    out[CONTINUOUS] = np.asarray(reference)[CONTINUOUS] + periodic_delta(q, reference)[CONTINUOUS]
    return out


def pose_error(fk, q, target):
    actual = fk.tcp(q)
    return (float(np.linalg.norm(actual[:3, 3]-target[:3, 3])),
            float(Rotation.from_matrix(target[:3, :3]@actual[:3, :3].T).magnitude()))


@dataclass
class SolutionV3:
    accepted: bool
    qpos: np.ndarray
    position_error_m: float
    rotation_error_rad: float
    reason: str
    fallback_unregularized: bool
    max_rotation_delta: float
    torso_delta: float
    candidate_count: int

    def summary(self):
        result = asdict(self)
        result['qpos'] = self.qpos.tolist()
        return result


class PreferredIK:
    """Hard pose feasibility, then robot-only regularized candidate selection.

Two deterministic seeds (previous and robot rest) are frozen. A rest seed is a
robot constant, never a teacher joint posture. Jump rejection is applied after
ranking; no hidden alternative path or scene filtering is introduced.
    """
    def __init__(self, fk, *, torso=True, parameters=ParametersV3()):
        self.fk, self.p = fk, parameters
        self.raw = FetchIK(fk, torso=torso, method='dogbox',
            position_tol=self.p.position_tolerance, rotation_tol=self.p.rotation_tolerance,
            max_nfev=self.p.max_nfev)
        self.indices = self.raw.indices

    def jump(self, q, previous):
        delta = periodic_delta(q, previous)
        rotation = float(np.max(abs(delta[ARM_IDX])))
        torso = float(abs(delta[3]))
        return rotation > self.p.rotation_jump or torso > self.p.torso_jump, rotation, torso

    def _regularized(self, target, seed, previous):
        bounds = []
        for name, i in zip(self.raw.names, self.indices):
            lo, hi, kind = self.fk.limits[name]
            bounds.append((previous[i]-np.pi, previous[i]+np.pi) if kind == 'continuous' else (lo, hi))
        lo, hi = np.asarray(bounds).T
        def residual(x):
            q = previous.copy(); q[self.indices] = x
            tcp = self.fk.tcp(q)
            return np.r_[tcp[:3, 3]-target[:3, 3],
                Rotation.from_matrix(target[:3, :3]@tcp[:3, :3].T).as_rotvec(),
                self.p.previous_weight*periodic_delta(q, previous)[self.indices],
                self.p.rest_weight*periodic_delta(q, REST)[self.indices]]
        fit = least_squares(residual, np.clip(seed[self.indices], lo, hi), bounds=(lo, hi),
            method='dogbox', max_nfev=self.p.max_nfev, ftol=1e-10, xtol=1e-10, gtol=1e-10)
        q = previous.copy(); q[self.indices] = fit.x
        return nearest_continuous(q, previous)

    def solve(self, target, previous):
        target = transform(target); previous = vector(previous, 15).copy()
        rest_seed = previous.copy(); rest_seed[self.indices] = REST[self.indices]
        candidates = []; attempts = []
        for seed in (previous, nearest_continuous(rest_seed, previous)):
            q = self._regularized(target, seed, previous)
            pe, re = pose_error(self.fk, q, target)
            fallback = pe > self.p.position_tolerance or re > self.p.rotation_tolerance
            if fallback:
                raw = self.raw.solve(target, seed)
                q = nearest_continuous(raw.qpos, previous)
                pe, re = pose_error(self.fk, q, target)
            attempts.append((pe+re, q, pe, re, fallback))
            if pe <= self.p.position_tolerance and re <= self.p.rotation_tolerance:
                cost = (self.p.previous_weight**2*np.sum(periodic_delta(q, previous)[self.indices]**2)
                       + self.p.rest_weight**2*np.sum(periodic_delta(q, REST)[self.indices]**2))
                candidates.append((cost, q, pe, re, fallback))
        selected = min(candidates if candidates else attempts, key=lambda x:x[0])
        _, q, pe, re, fallback = selected
        jumped, rotation, torso = self.jump(q, previous)
        reason = 'ik_not_found' if not candidates else 'jump_rejected' if jumped else 'ok'
        return SolutionV3(reason == 'ok', q, pe, re, reason, fallback, rotation, torso, len(candidates))


class FetchToolsV3:
    def __init__(self, fk, read_qpos, step, read_finger, read_rest_status, model,
                 *, parameters=ParametersV3(), max_steps=200, gripper=1., initial_delta=None):
        if fk.order != FETCH_ORDER or not isinstance(max_steps, int) or max_steps < 0:
            raise ValueError('robot order or shared budget')
        self.fk, self.p, self.model = fk, parameters, model
        self.read_qpos, self.step_callback = read_qpos, step
        self.read_finger, self.read_rest_status = read_finger, read_rest_status
        self.ik8 = PreferredIK(fk, parameters=parameters)
        self.ik7 = PreferredIK(fk, torso=False, parameters=parameters)
        self.max_steps, self.steps, self.stopped = max_steps, 0, False
        self.deadline = max_steps
        self.previous_delta = np.zeros(15) if initial_delta is None else vector(initial_delta, 15).copy()
        self.last_arm_target = vector(read_qpos(), 15).copy()
        self.last_plan_endpoint = self.last_arm_target.copy()
        self.last_goal = fk.tcp(self.last_arm_target)
        grip_band(gripper); self.gripper = float(gripper)
        self.rho = np.array([x['rho'] for x in model['joint_parameters']])
        self.beta = np.array([x['beta'] for x in model['joint_parameters']])
        self.trace, self.ik_records = [], []

    def _available(self):
        return not self.stopped and self.steps < min(self.deadline, self.max_steps)

    def _step(self, action, kind, **info):
        if not self._available():return False
        action = vector(action, 13)
        if np.max(abs(action)) > 1+1e-10 or np.any(action[11:] != 0):
            raise ValueError('action bound or fixed-base violation')
        before = vector(self.read_qpos(), 15).copy()
        status = self.step_callback(action)
        self.previous_delta = vector(self.read_qpos(), 15)-before
        self.steps += 1
        self.stopped = bool(status.get('terminated') or status.get('truncated'))
        self.trace.append({'step':self.steps, 'kind':kind, 'qpos':self.read_qpos().tolist(),
            'action':action.tolist(), 'delta':self.previous_delta.tolist(), **info})
        return not self.stopped

    def inverse_action(self, delta, feedback_reference):
        measured = vector(self.read_qpos(), 15)
        error = periodic_delta(feedback_reference, measured, CONTINUOUS)
        u = (delta[QIND]+self.p.feedback*error[QIND]-self.rho*self.previous_delta[QIND])/(.1*self.beta)
        action = np.zeros(13); action[:7] = np.clip(u[:7], -1, 1); action[10] = np.clip(u[7], -1, 1)
        for qi, ai in ((4,8),(6,9)):
            action[ai] = np.clip((delta[qi]+self.p.feedback*error[qi])/(.1*np.median(self.beta[:7])), -1, 1)
        action[7] = 2*self.gripper-1
        return action

    def _plan(self, targets):
        measured = vector(self.read_qpos(), 15).copy()
        # The boundary jump guard sees the previous command endpoint; IK starts
        # at measured state and every intermediate point uses its previous solve.
        q = measured.copy(); points = densify(self.fk.tcp(q), targets,
            self.p.translation_sample, self.p.rotation_sample)
        if not points:raise ValueError('empty path')
        refs = [q.copy()]; records = []; lower_bound = 0
        for i, point in enumerate(points):
            solved = self.ik8.solve(point, q)
            records.append(solved.summary())
            if solved.accepted and i == 0:
                jumped, dr, dz = self.ik8.jump(solved.qpos, self.last_plan_endpoint)
                if jumped:
                    return {'accepted':False, 'reason':'boundary_jump_rejected', 'records':records}
            if not solved.accepted:return {'accepted':False, 'reason':solved.reason, 'records':records}
            delta = periodic_delta(solved.qpos, q)
            lower_bound += float(np.max(abs(delta[QIND]))/.1)
            q = solved.qpos.copy(); refs.append(q)
        accepted = int(np.ceil(lower_bound)) <= self.max_steps-self.steps and not self.stopped
        return {'accepted':accepted, 'reason':'ok' if accepted else 'step_budget_or_terminal',
                'records':records, 'points':points, 'refs':np.asarray(refs),
                'minimum_steps':int(np.ceil(lower_bound))}

    def check_path(self, targets):
        result = self._plan(targets)
        return {k:v for k,v in result.items() if k not in ('points','refs')}

    def _compensate(self, point):
        measured = vector(self.read_qpos(), 15)
        seed = self.last_arm_target.copy(); seed[[0,1,2,3,4,6]] = measured[[0,1,2,3,4,6]]
        solution = self.ik7.solve(point, seed)
        self.ik_records.append(solution.summary())
        if solution.accepted:self.last_arm_target = solution.qpos.copy()
        target = self.last_arm_target.copy(); target[[0,1,2,3,4,6]] = measured[[0,1,2,3,4,6]]
        return target, solution

    def _schedule(self, plan):
        refs = plan['refs']; poses = np.asarray([self.fk.tcp(refs[0]), *plan['points']])
        speed = np.asarray(self.model['reference_speed_limits'])[:7]
        cost = np.max(abs(np.diff(refs[:,ARM_IDX], axis=0))/speed, axis=1)
        times = np.r_[0,np.cumsum(np.maximum(cost,1e-8))]
        ticks = np.linspace(0,times[-1],max(1,int(np.ceil(times[-1])))+1)
        rotation = Slerp(times,Rotation.from_matrix(poses[:,:3,:3]))(ticks).as_matrix()
        output = []
        for i,t in enumerate(ticks):
            pose = np.eye(4); pose[:3,:3] = rotation[i]
            pose[:3,3] = [np.interp(t,times,poses[:,j,3]) for j in range(3)]
            output.append(pose)
        return output

    def move_eef_chunk(self, targets, *, max_steps):
        before = self.steps; self.deadline = min(self.max_steps, before+max_steps)
        plan = self._plan(targets)
        if not plan['accepted']:
            return {'arrived':False, 'reason':plan['reason'], 'steps_used':0, 'plan':self.check_path(targets)}
        points = self._schedule(plan); cursor = 0; clock = 0; waits = 0; reason = 'tool_step_limit'
        profile = torso_trapezoid(float(self.read_qpos()[3]),float(plan['refs'][-1,3]),
            self.rho[-1],self.beta[-1],float(self.previous_delta[3]),cap=200)
        previous = self.read_qpos().copy()
        while cursor < len(points)-1 and self._available():
            target, solved = self._compensate(points[cursor+1])
            if solved.reason == 'jump_rejected':reason='jump_rejected';break
            if not solved.accepted:waits += 1
            index = min(clock+1,len(profile['q'])-1)
            target[3] = profile['q'][index]
            delta = np.zeros(15); delta[ARM_IDX] = periodic_delta(target,previous)[ARM_IDX] if solved.accepted else 0
            delta[3] = profile['delta'][index]
            feedback = previous.copy(); feedback[3] = profile['q'][min(clock,len(profile['q'])-1)]
            self._step(self.inverse_action(delta,feedback),'move',reference_index=cursor+1,ik_reason=solved.reason)
            measured = vector(self.read_qpos(),15)
            error = periodic_delta(target,measured)
            released = (np.max(abs(error[ARM_IDX])) <= self.p.release_rotation
                        and abs(error[3]) <= self.p.release_torso)
            if solved.accepted and released:cursor += 1
            previous = target.copy(); clock += 1
        goal = transform(targets[-1]); self.last_goal = goal
        pe,re = pose_error(self.fk,self.read_qpos(),goal)
        while cursor == len(points)-1 and (pe>self.p.endpoint_position or re>self.p.endpoint_rotation) and self._available():
            target, solved = self._compensate(goal)
            if solved.reason == 'jump_rejected':reason='jump_rejected';break
            if not solved.accepted:waits += 1
            self._step(self.inverse_action(np.zeros(15),target),'endpoint',ik_reason=solved.reason)
            pe,re = pose_error(self.fk,self.read_qpos(),goal)
        arrived = cursor == len(points)-1 and pe<=self.p.endpoint_position and re<=self.p.endpoint_rotation and reason!='jump_rejected'
        if arrived:self.last_plan_endpoint = plan['refs'][-1].copy()
        return {'arrived':bool(arrived), 'reason':'ok' if arrived else reason,
            'steps_used':self.steps-before, 'position_error_m':pe,'rotation_error_rad':re,
            'arm_ik_wait_steps':waits,'torso_target_m':float(plan['refs'][-1,3]),
            'reference_count':len(points),'reference_completed':cursor==len(points)-1}

    def move_to(self, target, *, max_steps=20):
        return self.move_eef_chunk([target],max_steps=max_steps)

    def set_gripper(self, command):
        band = grip_band(command); before = self.steps
        self.deadline = min(self.max_steps,before+15)
        entering = band==0 and grip_band(self.gripper)!=0
        alignment = None
        if entering:
            pe,re = pose_error(self.fk,self.read_qpos(),self.last_goal)
            while (pe>self.p.position_tolerance or re>self.p.rotation_tolerance or abs(self.previous_delta[3])>=self.p.torso_stable) and self.steps-before<self.p.close_align_cap and self._available():
                target, solved = self._compensate(self.last_goal)
                if solved.reason=='jump_rejected':
                    return {'accepted':False,'reason':'jump_rejected','steps_used':self.steps-before}
                self._step(self.inverse_action(np.zeros(15),target),'close_align',ik_reason=solved.reason)
                pe,re = pose_error(self.fk,self.read_qpos(),self.last_goal)
            alignment = {'position_error_m':pe,'rotation_error_rad':re,'torso_delta_m':abs(float(self.previous_delta[3])),
                         'steps_used':self.steps-before}
            if pe>self.p.position_tolerance or re>self.p.rotation_tolerance or abs(self.previous_delta[3])>=self.p.torso_stable or not self._available():
                return {'accepted':False,'reason':'close_alignment_failed','alignment':alignment,'steps_used':self.steps-before}
        self.gripper=float(command)
        wait_start=self.steps;stable=0;previous=float(self.read_finger())
        cap=self.p.close_wait_cap if entering else 1
        while self.steps-wait_start<cap and self._available():
            held=vector(self.read_qpos(),15).copy()
            self._step(self.inverse_action(np.zeros(15),held),'gripper')
            current=float(self.read_finger());stable=stable+1 if abs(current-previous)<self.p.finger_stable else 0;previous=current
            if entering and stable>=2:break
        return {'accepted':self.steps>wait_start,'reason':'ok' if self.steps>wait_start else 'terminal_or_budget',
                'steps_used':self.steps-before,'alignment':alignment,'enter_closed':entering,
                'command':self.gripper,'finger_distance_m':previous,'stable':stable>=2 if entering else None}

    def return_to_rest(self):
        before=self.steps;self.deadline=min(self.max_steps,before+self.p.rest_cap)
        flags=self.read_rest_status()
        def passed():return all(bool(flags.get(k,False)) for k in ('ee_rest','robot_rest','is_static'))
        while not passed() and self._available():
            measured=vector(self.read_qpos(),15)
            target=REST.copy();target[:3]=measured[:3];target[-2:]=measured[-2:]
            # Official robot_rest compares raw qpos with REST: use its exact
            # joint representation here, not a periodic equivalent of REST.
            action=self.inverse_action(np.zeros(15),target)
            delta=target[ARM_IDX]-measured[ARM_IDX]
            action[:7]=np.clip((delta-self.rho[:7]*self.previous_delta[ARM_IDX])/(.1*self.beta[:7]),-1,1)
            self._step(action,'return_to_rest_joint')
            flags=self.read_rest_status()
        self.last_arm_target=vector(self.read_qpos(),15).copy()
        self.last_plan_endpoint=self.last_arm_target.copy();self.last_goal=self.fk.tcp(self.last_arm_target)
        return {'arrived':passed(),'steps_used':self.steps-before,'criteria':flags,
                'max_rest_joint_error':float(np.max(abs(self.read_qpos()[3:-2]-REST[3:-2]))),
                'method':'joint_space_no_ik'}
