"""Streaming v2: last mobile-torso candidate, no simulator or scene imports."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .eef_tools import ARM, BODY, REST, FETCH_ORDER, FetchIK, densify, vector
from .eef_response_model import QIND
from .eef_streaming import grip_band, teacher_segments

VERSION='eef-streaming-v2-last-mobile-torso'


def torso_trapezoid(start, target, rho, beta, initial_delta=0., cap=200):
    """Fastest discrete symmetric trapezoid within the scalar A1 action bound.

    Speed limit is beta*0.1/(1-rho). Ramp acceleration and peak speed are
    selected jointly: each inverse-model input must remain in [-1,1].
    Existing momentum is braked first, not silently set to zero.
    """
    if not (0<=rho<1 and beta>0 and cap>=1):raise ValueError('invalid scalar dynamics')
    scale=.1*beta;q=float(start);v=float(initial_delta);positions=[q];velocities=[v];inputs=[]
    while abs(v)>1e-10:
        if len(inputs)>=cap:raise ValueError('braking exceeds reference cap')
        u=float(np.clip(-rho*v/scale,-1,1));v=rho*v+scale*u;q+=v
        positions.append(q);velocities.append(v);inputs.append(u)
    distance=target-q
    if abs(distance)<1e-10:
        return {'q':np.asarray(positions),'delta':np.asarray(velocities),'action':np.asarray(inputs),
                'v_limit':scale/(1-rho),'a_peak':0.,'peak_velocity':0.,'ramp_steps':0}
    remaining=cap-len(inputs);best=None
    for n in range(1,remaining//2+1):
        feasible_peak=scale/(1-rho+rho/n)
        m=max(0,int(np.ceil(abs(distance)/feasible_peak-1e-10))-n)
        length=2*n+m
        if length<=remaining and (best is None or (length,n)<(best[0],best[1])):best=(length,n,m)
    if best is None:raise ValueError('torso trapezoid exceeds reference cap')
    _,n,m=best;peak=distance/(n+m)
    deltas=np.r_[peak*np.arange(1,n+1)/n,np.full(m,peak),peak*np.arange(n-1,-1,-1)/n]
    for desired in deltas:
        u=(desired-rho*v)/scale
        if abs(u)>1+1e-9:raise ValueError('dynamically infeasible reference')
        q+=desired;v=float(desired);positions.append(q);velocities.append(v);inputs.append(float(u))
    return {'q':np.asarray(positions),'delta':np.asarray(velocities),'action':np.asarray(inputs),
            'v_limit':scale/(1-rho),'a_peak':abs(peak/n),'peak_velocity':abs(peak),'ramp_steps':n}


@dataclass(frozen=True)
class V2Parameters:
    feedback:float=1.
    close_align_cap:int=10
    close_wait_cap:int=5
    close_stable_count:int=2
    position_tolerance:float=.002
    rotation_tolerance:float=.02
    torso_stable_m:float=.001
    finger_stable_m:float=.001
    joint_rest_tolerance:float=.6
    ee_rest_threshold:float=.05
    rest_steps_cap:int=30
    arm_wait_cap:int=200


class FetchStreamingExecutorV2:
    def __init__(self,ik8,read_qpos,step,read_finger,model,parameters=V2Parameters(),*,max_steps=200,gripper=1.):
        if ik8.fk.order!=FETCH_ORDER or not isinstance(max_steps,int) or max_steps<0:raise ValueError('robot/budget')
        self.ik8=ik8;self.fk=ik8.fk
        self.ik7=FetchIK(self.fk,torso=False,method='dogbox',position_tol=parameters.position_tolerance,
                         rotation_tol=parameters.rotation_tolerance,max_nfev=180)
        self.read_qpos,self.step,self.read_finger=read_qpos,step,read_finger
        self.model,self.p=model,parameters;self.max_steps=max_steps;self.steps=0;self.stopped=False
        grip_band(gripper);self.gripper=float(gripper)
        self.previous_delta=np.zeros(15)
        self.last_arm_target=vector(read_qpos(),15).copy()
        self.rho=np.array([p['rho'] for p in model['joint_parameters']])
        self.beta=np.array([p['beta'] for p in model['joint_parameters']])
        self.wait_steps=0;self.trace=[];self.alignments=[];self.grip_events=[]

    def _step(self,action,kind,**info):
        if self.stopped or self.steps>=self.max_steps:return False
        before=vector(self.read_qpos(),15).copy();status=self.step(np.asarray(action,dtype=float))
        self.previous_delta=vector(self.read_qpos(),15)-before
        self.steps+=1;self.stopped=bool(status.get('terminated') or status.get('truncated'))
        self.trace.append({'step':self.steps,'kind':kind,'torso_m':float(self.read_qpos()[3]),
                          'torso_delta_m':float(self.previous_delta[3]),**info})
        return not self.stopped

    def inverse_action(self,desired_delta,target_q):
        q=vector(self.read_qpos(),15);desired=vector(desired_delta,15)
        command=(desired[QIND]+self.p.feedback*(target_q[QIND]-q[QIND])-self.rho*self.previous_delta[QIND])/(.1*self.beta)
        a=np.zeros(13);a[:7]=np.clip(command[:7],-1,1);a[10]=np.clip(command[7],-1,1)
        # Frozen v1 auxiliary head model rho=0, beta=median arm beta.
        head_beta=float(np.median(self.beta[:7]))
        for qi,ai in ((4,8),(6,9)):
            a[ai]=np.clip((desired[qi]+self.p.feedback*(target_q[qi]-q[qi]))/(.1*head_beta),-1,1)
        # Gripper is absolute position, not a velocity-channel rho/beta model.
        # Preserve the segment semantic command; it has its separate A1 model.
        a[7]=2*self.gripper-1
        return a

    def compensated_target(self,pose):
        measured=vector(self.read_qpos(),15);seed=self.last_arm_target.copy()
        seed[3]=measured[3];seed[[0,1,2,4,6]]=measured[[0,1,2,4,6]]
        result=self.ik7.solve(pose,seed)
        if result.accepted:self.last_arm_target=result.qpos.copy()
        target=self.last_arm_target.copy();target[3]=measured[3]
        target[[0,1,2,4,6]]=measured[[0,1,2,4,6]]
        return target,result.accepted,result

    def _torso_profile(self,target):
        return torso_trapezoid(float(self.read_qpos()[3]),float(target),self.rho[-1],self.beta[-1],
                               initial_delta=float(self.previous_delta[3]),cap=200)

    def _pose_schedule(self,targets):
        q=vector(self.read_qpos(),15).copy();poses=[self.fk.tcp(q)];joint_refs=[q.copy()]
        for point in densify(poses[0],targets):
            solution=self.ik8.solve(point,q)
            if not solution.accepted:return None,'ik8_not_found'
            q=solution.qpos.copy();poses.append(point);joint_refs.append(q.copy())
        # Torso target is exactly the IK8 solution at the segment endpoint.
        refs=np.asarray(joint_refs);speeds=np.asarray(self.model['reference_speed_limits'])[:7]
        cost=np.max(abs(np.diff(refs[:,QIND[:7]],axis=0))/speeds,axis=1)
        times=np.r_[0,np.cumsum(cost)];valid=np.r_[True,np.diff(times)>1e-10]
        times=times[valid];pose_array=np.asarray(poses)[valid]
        if len(times)==1:return {'poses':[pose_array[0]],'torso_target':float(q[3])},'ok'
        n=max(1,int(np.ceil(times[-1])));ticks=np.linspace(0,times[-1],n+1)
        rotations=Slerp(times,Rotation.from_matrix(pose_array[:,:3,:3]))(ticks).as_matrix()
        output=[]
        for i,t in enumerate(ticks):
            point=np.eye(4);point[:3,:3]=rotations[i]
            point[:3,3]=[np.interp(t,times,pose_array[:,j,3]) for j in range(3)]
            output.append(point)
        return {'poses':output,'torso_target':float(q[3])},'ok'

    def _track(self,schedule,kind='track'):
        before=self.steps;wait_before=self.wait_steps;cursor=0;poses=schedule['poses'];profile=self._torso_profile(schedule['torso_target'])
        k=0;previous_target=vector(self.read_qpos(),15).copy()
        while cursor<len(poses)-1 and not self.stopped and self.steps<self.max_steps:
            point=poses[min(cursor+1,len(poses)-1)]
            target,feasible,result=self.compensated_target(point)
            desired=np.zeros(15)
            if feasible:
                desired[QIND[:7]]=target[QIND[:7]]-previous_target[QIND[:7]]
            # Infeasible IK keeps last feasible arm target; torso still progresses.
            else:self.wait_steps+=1
            index=min(k+1,len(profile['q'])-1)
            target[3]=profile['q'][index];desired[3]=profile['delta'][index]
            feedback_target=previous_target.copy()
            feedback_target[3]=profile['q'][min(k,len(profile['q'])-1)]
            self._step(self.inverse_action(desired,feedback_target),kind,arm_ik_feasible=feasible,
                       reference_index=min(cursor+1,len(poses)-1),arm_position_residual_m=float(result.position_error_m))
            previous_target=target.copy();k+=1
            if feasible:cursor+=1
            if self.wait_steps-wait_before>=self.p.arm_wait_cap:break
        return {'completed_reference':cursor==len(poses)-1,'steps_used':self.steps-before,
                'arm_ik_wait_steps':self.wait_steps-wait_before,'reference_points':len(poses),
                'torso_target_m':schedule['torso_target'],'torso_profile_steps':len(profile['action']),
                'torso_profile_v_limit':profile['v_limit'],'torso_profile_peak_velocity':profile['peak_velocity'],
                'torso_profile_peak_acceleration':profile['a_peak']}

    def move_eef_chunk(self,targets):
        schedule,reason=self._pose_schedule(targets)
        if schedule is None:return {'completed_reference':False,'reason':reason,'steps_used':0,'arm_ik_wait_steps':0}
        return self._track(schedule)

    def close_alignment(self,pose):
        before=self.steps
        def residuals():
            tcp=self.fk.tcp(self.read_qpos())
            return float(np.linalg.norm(tcp[:3,3]-pose[:3,3])),float(Rotation.from_matrix(pose[:3,:3]@tcp[:3,:3].T).magnitude()),abs(float(self.previous_delta[3]))
        pe,re,td=residuals()
        while (pe>self.p.position_tolerance or re>self.p.rotation_tolerance or td>=self.p.torso_stable_m) and self.steps-before<self.p.close_align_cap and not self.stopped and self.steps<self.max_steps:
            target,feasible,_=self.compensated_target(pose)
            if not feasible:self.wait_steps+=1
            target[3]=self.read_qpos()[3]
            self._step(self.inverse_action(np.zeros(15),target),'close_align',arm_ik_feasible=feasible)
            pe,re,td=residuals()
        result={'passed':pe<=self.p.position_tolerance and re<=self.p.rotation_tolerance and td<self.p.torso_stable_m,
                'position_error_m':pe,'rotation_error_rad':re,'torso_delta_m':td,'steps_used':self.steps-before}
        self.alignments.append(result);return result

    def set_gripper(self,command,hold_pose=None):
        before=self.steps;enter_closed=grip_band(command)==0 and grip_band(self.gripper)!=0
        self.gripper=float(command)
        if not enter_closed:
            result={'enter_closed':False,'steps_used':0,'stable':None};self.grip_events.append(result);return result
        stable=0;previous=float(self.read_finger())
        pose=self.fk.tcp(self.read_qpos()) if hold_pose is None else hold_pose
        while stable<self.p.close_stable_count and self.steps-before<self.p.close_wait_cap and not self.stopped and self.steps<self.max_steps:
            target,feasible,_=self.compensated_target(pose);target[3]=self.read_qpos()[3]
            if not feasible:self.wait_steps+=1
            self._step(self.inverse_action(np.zeros(15),target),'closed_gripper_wait',arm_ik_feasible=feasible)
            current=float(self.read_finger());stable=stable+1 if abs(current-previous)<self.p.finger_stable_m else 0;previous=current
        result={'enter_closed':True,'steps_used':self.steps-before,'stable':stable>=self.p.close_stable_count,
                'finger_distance_m':previous};self.grip_events.append(result);return result

    def rest_criteria(self):
        q=vector(self.read_qpos(),15);tcp=self.fk.tcp(q)
        ee=float(np.linalg.norm(tcp[:3,3]-np.array([.5,0,1.25])))
        errors=abs(q[3:-2]-REST[3:-2])
        return {'ee_rest':ee<=self.p.ee_rest_threshold,'robot_rest':bool(np.all(errors<self.p.joint_rest_tolerance)),
                'ee_rest_distance_m':ee,'max_rest_joint_error':float(max(errors)),
                'torso_height_m':float(q[3]),'joint_tolerance':self.p.joint_rest_tolerance,
                'evaluation':'official geometry thresholds on CPU surrogate state; no grasp/static/force physics'}

    def return_to_rest(self):
        before=self.steps;criteria=self.rest_criteria()
        seed=REST.copy();seed[3]=self.read_qpos()[3];seed[:3]=self.read_qpos()[:3]
        pose=self.fk.tcp(seed);pose[:3,3]=[.5,0,1.25]
        initial=self.ik7.solve(pose,seed)
        if initial.accepted:self.last_arm_target=initial.qpos.copy()
        while not (criteria['ee_rest'] and criteria['robot_rest']) and self.steps-before<self.p.rest_steps_cap and not self.stopped and self.steps<self.max_steps:
            target,feasible,_=self.compensated_target(pose);target[3]=self.read_qpos()[3]
            target[4]=REST[4];target[6]=REST[6]
            if not feasible:self.wait_steps+=1
            self._step(self.inverse_action(np.zeros(15),target),'return_to_rest',arm_ik_feasible=feasible)
            criteria=self.rest_criteria()
        return {**criteria,'passed':bool(criteria['ee_rest'] and criteria['robot_rest']),'steps_used':self.steps-before}

    def replay_teacher(self,frames):
        result=teacher_segments(frames);segments=[];all_done=True
        for s in result['segments']:
            before=self.steps;closed=grip_band(s['gripper'])==0 and grip_band(self.gripper)!=0
            if not closed:self.set_gripper(s['gripper'])
            targets=[np.asarray(frames[i]['tcp_base']) for i in s['indices'][1:]]
            motion=self.move_eef_chunk(targets)
            alignment=None;grip=None
            if closed and motion['completed_reference']:
                alignment=self.close_alignment(targets[-1])
                if alignment['passed']:grip=self.set_gripper(s['gripper'],targets[-1])
            ok=motion['completed_reference'] and (not closed or bool(alignment and alignment['passed']))
            segments.append({'indices':s['indices'],'command':s['gripper'],'enter_closed':closed,
                'motion':motion,'alignment':alignment,'gripper_wait':grip,'steps_used':self.steps-before})
            if not ok or self.stopped:all_done=False;break
        rest=self.return_to_rest() if all_done else None
        complete=all_done and rest is not None and rest['passed']
        return {'version':VERSION,'complete':bool(complete),'predicted_steps':self.steps if complete else None,
                'steps_until_end_or_abort':self.steps,'segments':segments,'arm_ik_wait_steps':self.wait_steps,
                'close_alignments':self.alignments,'rest':rest,'parameters':asdict(self.p)}
