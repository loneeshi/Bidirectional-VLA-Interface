"""Bounded, observation-driven task executor; never calls an environment or API.
Every returned command requires one fresh acknowledged observation before another.
Task poses use the caller's current base frame; persistence uses local odometry.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from flowsde_ctrl.inputs import TrackInput
from fetch_driver import encode_flow_robot,flatten_groups
from task_rules import REST_CAP, rest_budget, close_command_result
ARM=np.array([5,7,8,9,10,11,12]);HEAD=np.array([4,6])
REST=np.array([0,0,0,.386,0,-.370,.562,-1.032,.695,.955,-.1,2.077,0,.015,.015])

@dataclass(frozen=True)
class Stage:
    phase: str
    cap: int
    target: object=None  # 4x4 robot command pose; no scene target input
    reference_frame: str='tool_base'  # or episode_base for frozen command replay

def rigid(T):
    T=np.asarray(T,float)
    if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(T[:3,:3]),1,atol=1e-6):
        raise ValueError('Invalid rigid pose')
    return T.copy()

def pose(p,R):
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=p;return T

class FetchEpisode:
    """Complete stage dispatch with persistent controller; no success surrogate.

    Caller supplies measured qpos/qvel, base-local odometry transform/velocity,
    official end flags and official rest flags. No object pose/grasp/contact IDs.
    max_steps counts ACKNOWLEDGED actions; produced pending actions reserve a slot.
    """
    def __init__(self,session,stages,action_mapping,*,max_steps):
        if type(max_steps) is not int or max_steps<1:raise ValueError('Invalid episode cap')
        if not stages:raise ValueError('Empty stage plan')
        self.stages=[]
        for stage in stages:
            if stage.phase not in ('open','pregrasp','descend','close','settling','lift','rest','hold') or type(stage.cap) is not int or stage.cap<1:
                raise ValueError('Invalid stage')
            if stage.reference_frame not in ('tool_base','episode_base'):raise ValueError('Invalid reference frame')
            target=rigid(stage.target) if stage.target is not None else None
            if stage.phase in ('pregrasp','descend','lift') and target is None:raise ValueError('Move needs target pose')
            self.stages.append(Stage(stage.phase,stage.cap,target,stage.reference_frame))
        self.session=session;self.mapping=action_mapping;self.max_steps=max_steps
        self.index=0;self.steps=0;self.stage_steps=0;self.pending=False;self.seq=None
        self.status='running';self.events=[];self.entered=False;self.closed=False
        self.goal_local=None;self.last_q=None;self.finger_stable=0;self.close_steps=0
        self.dt=session.ctrl.dtp;self.initial_base=None

    def _finish(self,reason):
        self.status=reason;return None

    def advance(self,*,sequence,qpos,qvel,local_from_base,base_velocity,terminated=False,truncated=False,rest_flags=None):
        if self.status!='running':return None
        q=np.asarray(qpos,float);v=np.asarray(qvel,float);B=rigid(local_from_base);bv=np.asarray(base_velocity,float)
        if q.shape!=(15,) or v.shape!=(15,) or bv.shape!=(3,) or not np.isfinite(np.r_[q,v,bv]).all():raise ValueError('Invalid measured state')
        if type(sequence) is not int or (self.seq is not None and sequence!=self.seq+1):raise ValueError('Missing/duplicate acknowledged observation')
        if type(terminated) is not bool or type(truncated) is not bool:raise ValueError('Invalid official flags')
        if self.initial_base is None:self.initial_base=B.copy()
        previous=self.last_q
        if self.pending:
            self.steps+=1;self.stage_steps+=1
            if self.stages[self.index].phase=='close' and self.closed:
                self.close_steps+=1
                gap=float(q[13]+q[14]);old=float(previous[13]+previous[14])
                self.finger_stable=self.finger_stable+1 if abs(gap-old)<.001 else 0
                self.events[-1].update(close_command_result(self.close_steps>=5 or
                    (self.close_steps>=2 and self.finger_stable>=2)))
        self.pending=False;self.seq=sequence
        if self.session.latch_official_end(terminated=terminated,truncated=truncated):
            return self._finish('official_terminated' if terminated else 'official_truncated')
        p,R=self.session.model.fk(q[ARM],q[3],jac=False);current=pose(p,R)
        for _ in range(len(self.stages)+1):
            stage=self.stages[self.index]
            if not self.entered:
                self.stage_steps=0;self.close_steps=0;self.finger_stable=0
                self.start_local=B@current
                if stage.target is not None:self.goal_local=(self.initial_base if stage.reference_frame=='episode_base' else B)@stage.target
                elif self.goal_local is None or stage.phase=='open':self.goal_local=B@current
                self.head_target=q[HEAD].copy()
                if stage.phase=='rest':
                    if self.session.first:raise ValueError('Rest needs an initialized control session')
                    rp,rR=self.session.begin_rest(REST[ARM],REST[3]);self.goal_local=B@pose(rp,rR);self.head_target=REST[HEAD].copy()
                target=np.linalg.inv(B)@self.goal_local
                angle=Rotation.from_matrix(current[:3,:3].T@target[:3,:3]).magnitude()
                self.duration=max(float(np.linalg.norm(current[:3,3]-target[:3,3]))/.2,float(angle)/.5,self.dt)
                if stage.phase not in ('pregrasp','descend','lift','rest'):self.duration=0.
                self.events.append(dict(stage=self.index,phase=stage.phase,start_step=self.steps))
                if stage.phase=='close':self.events[-1].update(close_command_result(False))
                self.entered=True
                if stage.phase=='rest':
                    budget=rest_budget(self.duration,self.dt,stage.cap,self.max_steps-self.steps)
                    self.events[-1]['rest_budget']=budget
                    if not budget['feasible']:
                        self.events[-1].update(end_step=self.steps,status='rest_budget_infeasible')
                        return self._finish('rest_budget_infeasible')
                if stage.phase=='open':self.closed=False
            target=np.linalg.inv(B)@self.goal_local
            pe=float(np.linalg.norm(current[:3,3]-target[:3,3]));re=float(Rotation.from_matrix(current[:3,:3].T@target[:3,:3]).magnitude())
            elapsed=self.stage_steps*self.dt;done=False
            if stage.phase in ('pregrasp','descend','lift'):done=elapsed>=self.duration and pe<=.01 and re<=.05
            elif stage.phase in ('open','settling','hold'):done=self.stage_steps>=stage.cap
            elif stage.phase=='close':
                torso_delta=abs(q[3]-previous[3]) if previous is not None else np.inf
                if not self.closed:
                    if pe<=.002 and re<=.02 and torso_delta<.001:self.closed=True
                    elif self.stage_steps>=min(10,stage.cap):return self._finish('close_alignment_failed')
                if self.closed:done=self.close_steps>=5 or (self.close_steps>=2 and self.finger_stable>=2)
            elif stage.phase=='rest':
                flags=rest_flags or {}
                done=all(flags.get(k) is True for k in ('ee_rest','robot_rest','is_static'))
            if done:
                self.events[-1].update(end_step=self.steps,status='completed')
                if stage.phase=='close':self.events[-1].update(close_command_result(True))
                self.index+=1;self.entered=False
                if self.index==len(self.stages):return self._finish('plan_complete')
                continue
            if self.steps>=self.max_steps:return self._finish('episode_step_limit')
            if self.stage_steps>=stage.cap:return self._finish('stage_step_limit')
            break
        else:raise AssertionError('Stage transition cycle')
        # Clamped continuous preview in local odometry; re-express in current base.
        times=np.arange(13)*.05
        alpha=np.ones(13) if self.duration==0 else np.clip((elapsed+times)/self.duration,0,1)
        P=(1-alpha[:,None])*self.start_local[:3,3]+alpha[:,None]*self.goal_local[:3,3]
        rots=Slerp([0.,1.],Rotation.from_matrix(np.stack([self.start_local[:3,:3],self.goal_local[:3,:3]])))(alpha).as_matrix()
        inv=np.linalg.inv(B);P=P@inv[:3,:3].T+inv[:3,3];rots=np.einsum('ij,njk->nik',inv[:3,:3],rots)
        # Match upstream reference anchor's pose-delay convention.
        pp,RR=self.session.model.fk(q[ARM]-self.session.ctrl.dl*v[ARM],q[3],jac=False)
        inp=TrackInput(False,q[ARM],v[ARM],times,P-pp,np.einsum('ij,njk->nik',RR.T,rots),h=q[3],base_vel=bv,grip_cmd=np.array([1.,1.]) if self.closed else np.array([-1.,-1.]))
        cmd,arm=self.session.tick(inp,sequence=sequence,phase=stage.phase)
        torso=cmd.vz
        if stage.phase=='rest':
            # Task-layer homing target for torso; arm still uses full IK/contact/servo chain.
            torso=float(np.clip((REST[3]-q[3])/.3,-.1,.1))
        robot=encode_flow_robot(arm,head_q=q[HEAD],head_target=self.head_target,torso_velocity=torso,base_velocity=[0.,0.] if stage.phase=='rest' else [cmd.base_cmd[0],cmd.base_cmd[2]],grip_closed=self.closed)
        action=flatten_groups(robot,self.mapping)
        self.pending=True;self.last_q=q.copy()
        return dict(action=action,phase=stage.phase,stage=self.index,sequence=sequence,controller_info=cmd.info,clipped_groups=robot.clipped_groups)


def stages_from_frozen_commands(commands,*,move_cap=150,rest_cap=REST_CAP):
    """Translate the three frozen near-pick scripts; no object state is consumed.

    Pose commands are bound to the episode's initial base frame for A/B target
    identity despite whole-body motion. This is an explicit replay-frame policy.
    Caps describe CPU preparation, not authorization to consume physical steps.
    """
    expected=['set_gripper','move_to','move_to','set_gripper','move_to','return_to_rest']
    if [c['tool'] for c in commands]!=expected or commands[0].get('gripper')!=1 or commands[3].get('gripper')!=0:
        raise ValueError('Frozen near-pick script shape changed')
    if move_cap<=2 or rest_cap<1:raise ValueError('Invalid stage cap')
    out=[];moves=iter(['pregrasp','descend','lift'])
    for command in commands:
        out.append(Stage('settling',2))
        if command['tool']=='move_to':
            if len(command['poses'])!=1:raise ValueError('Expected one frozen pose')
            item=command['poses'][0];T=pose(np.asarray(item['position'],float),Rotation.from_quat(item['quaternion_xyzw']).as_matrix())
            out.append(Stage(next(moves),move_cap-2,T,'episode_base'))
        elif command['tool']=='set_gripper':out.append(Stage('open',1) if command['gripper']==1 else Stage('close',15))
        else:out.append(Stage('rest',rest_cap))
    return out
