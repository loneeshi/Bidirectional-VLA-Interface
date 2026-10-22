"""CPU-prepared FlowSDE IK/preview candidate; no environment or API entry point.

Original pure modules are hash-pinned by upstream.json. Existing Fetch action,
gripper, settling, budget and terminal handling remain inherited. This is NOT
the FlowSDE predictive servo or whole-body controller.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import numpy as np
from scipy.spatial.transform import Rotation
from astra_pick.robot.eef_tools import ARM, FETCH_ORDER, transform, vector
from astra_pick.robot.executor import ArmCoordinatedV2

HERE = Path(__file__).resolve().parent
def validate_runtime_inputs():
    manifest=json.loads((HERE/'runtime-inputs.json').read_text(encoding='utf-8'))
    root=HERE.parents[3]
    for name,digest in manifest['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('paired runtime source changed: '+name)

def load_upstream():
    manifest = json.loads((HERE/'upstream.json').read_text())
    root = Path(manifest['root'])
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError('upstream source changed: '+name)
    # No package __init__, simulator import, or synthetic simulator module.
    if 'flowsde_ctrl' in sys.modules:
        raise RuntimeError('load in a fresh process; flowsde_ctrl already imported')
    package = types.ModuleType('flowsde_ctrl'); package.__path__ = [str(root)]
    sys.modules['flowsde_ctrl'] = package
    from flowsde_ctrl.ik import DLSIK, IKConfig
    from flowsde_ctrl.reference import ReferencePreview
    return DLSIK, IKConfig, ReferencePreview
DLSIK, IKConfig, ReferencePreview = load_upstream()

class BoundedPreview(ReferencePreview):
    """Retain upstream spline inside horizon; hold endpoint outside it."""
    def build(self, inp, p, R):
        t=np.asarray(inp.ref_t, float); P=np.asarray(inp.ref_rel_p,float)
        Rs=np.asarray(inp.ref_rel_R,float)
        if (t.ndim!=1 or len(t)<2 or not np.isfinite(t).all()
                or not np.all(np.diff(t)>0) or t[0]!=0
                or P.shape!=(len(t),3) or Rs.shape!=(len(t),3,3)
                or not np.isfinite(P).all() or not np.isfinite(Rs).all()
                or not np.isfinite(inp.ref_dt) or inp.ref_dt<=0):
            raise ValueError('invalid preview')
        for r in [np.asarray(R),*Rs]:
            if not np.allclose(r.T@r,np.eye(3),atol=1e-7) or not np.isclose(np.linalg.det(r),1,atol=1e-7):
                raise ValueError('invalid rotation')
        vector(p,3)
        return super().build(inp,p,R)
    def _time(self,s):
        if not np.isfinite(s): raise ValueError('invalid preview time')
        return float(np.clip(s,self.t[0],self.t[-1]))
    def pos(self,s): return super().pos(self._time(s))
    def vel(self,s):
        self._time(s)
        return np.zeros(3) if s<self.t[0] or s>=self.t[-1] else super().vel(self._time(s))
    def ang_vel(self,s):
        self._time(s)
        return np.zeros(3) if s<self.t[0] or s>=self.t[-1] else super().ang_vel(self._time(s))
    def target_at(self,s,vlin,wz,anchor):
        # This candidate is fixed-base. Reject unsupported compensation explicitly.
        if np.any(vector(vlin,3)!=0) or not np.isfinite(wz) or wz!=0:
            raise ValueError('fixed-base preview requires zero base velocity')
        return super().target_at(self._time(s),vlin,wz,vector(anchor,3))

class FetchKinematics:
    def __init__(self,fk,full):
        self.robot=fk; self.full=vector(full,15).copy()
        self.indices=np.array([FETCH_ORDER.index(n) for n in ARM]+[3])
        limits=[fk.limits[FETCH_ORDER[i]][:2] for i in self.indices]
        self.q_lo,self.q_hi=np.array(limits).T
    def pose(self,q):
        full=self.full.copy(); full[self.indices]=q
        return self.robot.tcp(full)
    def fk(self,q,h):
        T=self.pose(q); J=np.zeros((6,8)); eps=1e-6
        for i in range(8):
            delta=np.zeros(8);delta[i]=eps
            a,b=self.pose(q+delta),self.pose(q-delta)
            J[:3,i]=(a[:3,3]-b[:3,3])/(2*eps)
            J[3:,i]=Rotation.from_matrix(a[:3,:3]@b[:3,:3].T).as_rotvec()/(2*eps)
        return T[:3,3],T[:3,:3],J

def bounded_ik_tick(model,q,target,dt,posture=None):
    """Cap TOTAL displacement, with separate angular/torso speed units."""
    q=vector(q,8); target=transform(target)
    if not np.isfinite(dt) or dt<=0: raise ValueError('invalid dt')
    if np.any(q<model.q_lo) or np.any(q>model.q_hi): raise ValueError('joint outside URDF limits')
    # Upstream assumes angular coordinates. Normalize the prismatic torso by
    # angular_speed / linear_speed, including its Jacobian and bounds.
    units=np.r_[np.ones(7),25.]
    class ScaledModel:
        q_lo=model.q_lo*units; q_hi=model.q_hi*units
        def fk(self,x,h):
            p,R,J=model.fk(x/units,h)
            return p,R,J/units
    cfg=IKConfig()
    solver=DLSIK(ScaledModel(),cfg,dt)
    solver.q_post=(q.copy() if posture is None else vector(posture,8))*units
    proposed=solver.solve(q*units,0,target[:3,3],target[:3,:3])/units
    delta=vector(proposed,8)-q
    # Continuous joints stay unwrapped near the measured reference; no +/-2pi box.
    continuous=~np.isfinite(model.q_lo)
    delta[continuous]=(delta[continuous]+np.pi)%(2*np.pi)-np.pi
    budget=np.r_[np.full(7,2.5),.1]*dt
    scale=max(1.,float(np.max(np.abs(delta)/budget)))
    out=np.clip(q+delta/scale,model.q_lo,model.q_hi)
    return out,{'total_step_scaled':scale>1,'max_budget_ratio':float(np.max(abs(out-q)/budget))}

class FlowFetchExecutor(ArmCoordinatedV2):
    VERSION='flowsde-ik-preview-fetch-cpu-v1'
    REFERENCE_LINEAR_SPEED=.2  # m/s, preview timing only; not measured TCP speed
    REFERENCE_ANGULAR_SPEED=.5 # rad/s, actual joint increments separately capped
    def move_base(self,*args,**kwargs):
        raise ValueError('navigation disabled for paired arm experiment')
    def move_eef_chunk(self,targets,*,max_steps):
        if type(max_steps) is not int or max_steps<=0: raise ValueError('invalid step cap')
        targets=[transform(t) for t in targets]
        if not targets: raise ValueError('empty path')
        def motion(left):
            before=self.steps
            entry=vector(self.read_qpos(),15); kin=FetchKinematics(self.fk,entry)
            start=self.fk.tcp(entry); poses=[start,*targets]
            durations=[max(self.control_dt,np.linalg.norm(b[:3,3]-a[:3,3])/self.REFERENCE_LINEAR_SPEED,
                Rotation.from_matrix(a[:3,:3].T@b[:3,:3]).magnitude()/self.REFERENCE_ANGULAR_SPEED)
                for a,b in zip(poses,poses[1:])]
            times=np.r_[0,np.cumsum(durations)]
            preview=BoundedPreview().build(types.SimpleNamespace(ref_t=times,
                ref_rel_p=np.array([t[:3,3]-start[:3,3] for t in poses]),
                ref_rel_R=np.array([start[:3,:3].T@t[:3,:3] for t in poses]),
                ref_dt=self.control_dt),start[:3,3],start[:3,:3])
            if times[-1]>left*self.control_dt:
                return dict(arrived=False,reason='reference_timing_infeasible',steps_used=0)
            reason='tool_step_limit'; pe=re=float('inf')
            while self._available():
                measured=vector(self.read_qpos(),15); kin.full=measured.copy()
                t=(self.steps-before+1)*self.control_dt
                p,R=preview.target_at(t,np.zeros(3),0,np.zeros(3))
                goal=np.eye(4); goal[:3,3]=p; goal[:3,:3]=R
                try:
                    solved,info=bounded_ik_tick(kin,measured[kin.indices],goal,self.control_dt,entry[kin.indices])
                except (ValueError,np.linalg.LinAlgError):
                    reason='ik_invalid';break
                desired=measured.copy();desired[kin.indices]=solved
                # Existing response-model action conversion, not FlowSDE predictive servo.
                action=self.inverse_action(np.zeros(15),desired);action[11:]=0
                self._step(action,'flowsde_move',reference_time_s=min(t,times[-1]),**info)
                actual=self.fk.tcp(self.read_qpos())
                pe=float(np.linalg.norm(actual[:3,3]-targets[-1][:3,3]))
                re=float(Rotation.from_matrix(actual[:3,:3].T@targets[-1][:3,:3]).magnitude())
                if self.stopped: reason='official_terminated';break
                if t>=times[-1] and pe<=self.p.endpoint_position and re<=self.p.endpoint_rotation:
                    reason='ok';break
            self.last_goal=targets[-1].copy()
            self.last_arm_target=vector(self.read_qpos(),15).copy()
            return dict(arrived=reason=='ok',reason=reason,steps_used=self.steps-before,
                position_error_m=pe if np.isfinite(pe) else None,
                rotation_error_rad=re if np.isfinite(re) else None)
        return self._guarded('move_eef_chunk',max_steps,motion)

def execute_script(executor,commands):
    """Same frozen command script for A/B, fail-stop on incomplete arm motion.

    Caller owns environment creation and physical authorization. This function
    itself only talks to the supplied executor; no API or reset is performed.
    """
    from astra_pick.robot.eef_tools import pose
    # Validate the complete script before the first callback.
    prepared=[]
    for command in commands:
        tool=command['tool']
        if tool not in ('move_to','move_eef_chunk','set_gripper','return_to_rest'):
            raise ValueError('script contains unsupported or navigation command')
        targets=[pose(x['position'],x['quaternion_xyzw']) for x in command.get('poses',[])]
        if tool in ('move_to','move_eef_chunk') and not targets:raise ValueError('missing target')
        if tool=='set_gripper':
            grip=command['gripper']
            if type(grip) not in (float,int) or not np.isfinite(grip) or not 0<=grip<=1:
                raise ValueError('invalid gripper')
        prepared.append((command,targets))
    rows=[]
    for command,targets in prepared:
        if executor.stopped or executor.steps>=executor.max_steps:break
        tool=command['tool']
        if tool in ('move_to','move_eef_chunk'):
            result=executor.move_eef_chunk(targets,max_steps=min(150,executor.max_steps-executor.steps))
        elif tool=='set_gripper':result=executor.set_gripper(command['gripper'])
        else:result=executor.return_to_rest()
        rows.append(dict(source_turn=command['source_turn'],tool=tool,result=result))
        if result.get('arrived') is False or result.get('accepted') is False:break
    return rows

def make_executor(condition,*args,**kwargs):
    """Explicit factory for the later reviewed harness; validates frozen sources."""
    validate_runtime_inputs()
    if condition not in ('A','B'):raise ValueError('condition must be A or B')
    return (ArmCoordinatedV2 if condition=='A' else FlowFetchExecutor)(*args,**kwargs)
