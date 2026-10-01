"""Arm-only 600-step budget and one candidate change: joint-space approach.

No scene geometry. Existing coordinated-v2 control gains and execution remain
inherited. The candidate changes only reference path generation.
"""
from dataclasses import replace
import numpy as np
from .eef_coordinated_tools_v2 import CoordinatedVisualToolsV2
from .eef_tools import REST, densify, transform, vector
from .eef_tools_v3 import ARM_IDX, CONTINUOUS, nearest_continuous, periodic_delta, pose_error
from .eef_response_model import QIND
from .eef_tools_v3 import ParametersV3

PLANNING_CAP = 600
MOVE_CAP = 150
STRAIGHT_TAIL_M = .12


class ArmCoordinatedV2(CoordinatedVisualToolsV2):
    """Fallback: v2, with only the user-authorized reference horizon extended."""
    VERSION = 'arm-coordinated-v2-600'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('max_steps', 600)
        if kwargs['max_steps'] > 600:
            raise ValueError('arm attempt budget exceeds 600')
        kwargs['parameters'] = replace(kwargs.get('parameters', ParametersV3()), rest_cap=150)
        super().__init__(*args, **kwargs)

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
