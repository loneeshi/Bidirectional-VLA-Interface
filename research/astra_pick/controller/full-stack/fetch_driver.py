"""CPU codec for the source-verified Fetch pd_joint_delta_pos_vel arm.
No environment dependency, no legacy response-model inverse. Body position, base velocity and gripper groups use the archived Fetch configuration.
No torso-velocity-to-position equivalence is assumed. Source: pd_joint_pos_vel.py.source + archived fetch.py.
"""
from dataclasses import dataclass
import numpy as np
@dataclass(frozen=True)
class ArmCommand:
    action: np.ndarray
    applied_position: np.ndarray
    applied_velocity: np.ndarray
    clipped: np.ndarray

def encode_arm(q, q_sp, qd_ff):
    q, p, v=(np.asarray(x,dtype=float) for x in (q,q_sp,qd_ff))
    if any(x.shape!=(7,) or not np.isfinite(x).all() for x in (q,p,v)):
        raise ValueError('Expected seven finite arm joint values')
    raw=np.r_[(p-q)/0.1,v]  # velocity range [-1,1] rad/s, not Panda [-2,2]
    action=np.clip(raw,-1,1).astype(np.float32)
    applied_p, applied_v=decode_arm(q,action)
    return ArmCommand(action,applied_p,applied_v,np.abs(raw)>1)

def decode_arm(q, action):
    q=np.asarray(q,dtype=float); a=np.asarray(action,dtype=float)
    if q.shape!=(7,) or a.shape!=(14,) or not np.isfinite(q).all() or not np.isfinite(a).all():
        raise ValueError('Invalid Fetch arm observation/action')
    a=np.clip(a,-1,1)
    return q+0.1*a[:7],a[7:].copy()


@dataclass(frozen=True)
class RobotCommand:
    groups: dict
    applied_body_position: np.ndarray
    applied_base_velocity: np.ndarray
    applied_gripper_position: float
    clipped_groups: tuple


def encode_robot(arm, *, body_q, body_target, base_velocity, grip_closed):
    """Official grouped Fetch action for pd_joint_delta_pos_vel.

    body order is head_pan, head_tilt, torso. body_target is an explicit position
    reference: this routine deliberately does not pretend position PD accepts
    torso velocity feed-forward. Base velocity is ego-frame [forward, yaw].
    grip_closed is intent, not simulator grasp state. Official gripper range
    [-.01,.05] means negative normalized action closes (upstream grip_cmd positive).
    """
    q=np.asarray(body_q,dtype=float);p=np.asarray(body_target,dtype=float);b=np.asarray(base_velocity,dtype=float)
    if q.shape!=(3,) or p.shape!=(3,) or b.shape!=(2,) or not np.isfinite(np.r_[q,p,b]).all():
        raise ValueError('Invalid body/base command')
    if not isinstance(grip_closed,(bool,np.bool_)):
        raise ValueError('grip_closed must be explicit boolean intent')
    if not isinstance(arm,ArmCommand):raise ValueError('Expected validated arm codec result')
    body_raw=(p-q)/.1; base_raw=b/np.array([1.,3.14])
    body=np.clip(body_raw,-1,1).astype(np.float32);base=np.clip(base_raw,-1,1).astype(np.float32)
    grip=np.array([-1. if grip_closed else 1.],dtype=np.float32)
    clipped=tuple(n for n,v in [('arm',arm.clipped),('body',np.abs(body_raw)>1),('base',np.abs(base_raw)>1)] if np.any(v))
    return RobotCommand(dict(arm=arm.action.copy(),gripper=grip,body=body,base=base),q+.1*body.astype(float),base.astype(float)*[1.,3.14],-.01 if grip_closed else .05,clipped)


def flatten_groups(command,action_mapping):
    """Use runtime controller.action_mapping; never assume group concatenation order."""
    if set(action_mapping)!=set(command.groups):raise ValueError('Unexpected controller action groups')
    spans={name:tuple(bounds) for name,bounds in action_mapping.items()}
    for name,(a,b) in spans.items():
        if type(a) is not int or type(b) is not int or a<0 or b-a!=len(command.groups[name]):
            raise ValueError('Invalid runtime action mapping')
    indices=[i for a,b in spans.values() for i in range(a,b)]
    if sorted(indices)!=list(range(20)):raise ValueError('Action mapping overlaps or has gaps')
    action=np.empty(20,dtype=np.float32)
    for name,(a,b) in spans.items():action[a:b]=command.groups[name]
    return action


@dataclass(frozen=True)
class FlowRobotCommand:
    groups: dict
    applied_head_position: np.ndarray
    applied_torso_velocity: float
    applied_base_velocity: np.ndarray
    applied_gripper_position: float
    clipped_groups: tuple

def encode_flow_robot(arm,*,head_q,head_target,torso_velocity,base_velocity,grip_closed):
    """flowsde_full mode: arm14 + grip1 + head2 + torso velocity1 + base2."""
    q=np.asarray(head_q,float);p=np.asarray(head_target,float);b=np.asarray(base_velocity,float)
    if q.shape!=(2,) or p.shape!=(2,) or b.shape!=(2,) or not np.isfinite(np.r_[q,p,b,torso_velocity]).all():
        raise ValueError('Invalid head/base/torso command')
    if not isinstance(grip_closed,(bool,np.bool_)) or not isinstance(arm,ArmCommand):
        raise ValueError('Invalid arm/gripper intent')
    raw_head=(p-q)/.1;raw_base=b/[1.,3.14];raw_torso=float(torso_velocity)/.1
    head=np.clip(raw_head,-1,1).astype(np.float32);base=np.clip(raw_base,-1,1).astype(np.float32)
    torso=np.array([np.clip(raw_torso,-1,1)],dtype=np.float32);grip=np.array([-1. if grip_closed else 1.],dtype=np.float32)
    clipped=tuple(n for n,v in [('arm',arm.clipped),('head',abs(raw_head)>1),('torso',abs(raw_torso)>1),('base',abs(raw_base)>1)] if np.any(v))
    return FlowRobotCommand(dict(arm=arm.action.copy(),gripper=grip,head=head,torso=torso,base=base),q+.1*head.astype(float),float(torso[0])*.1,base.astype(float)*[1.,3.14],-.01 if grip_closed else .05,clipped)
