"""Static grouping of frozen robot code; algorithms and inheritance preserved."""

# ========================================================================
# Frozen source: eef_response_model.py
"""CPU-only empirical joint/gripper dynamics; excludes contact and grasp physics."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import lsq_linear
from .eef_tools import ARM, FETCH_ORDER

NAMES = (*ARM, 'torso_lift_joint')
QIND = np.array([FETCH_ORDER.index(n) for n in NAMES])
AIND = np.array([0, 1, 2, 3, 4, 5, 6, 10])


def samples(frames):
    q = np.asarray([f['qpos'] for f in frames], dtype=float)
    actions = np.asarray([f['action'] for f in frames[1:]], dtype=float)
    # sample(action) occurs AFTER adapter.step(action); frame zero has no action.
    delta = np.diff(q[:, QIND], axis=0)
    lag = np.vstack([np.zeros(8), delta[:-1]])
    finger = np.asarray([f['finger_distance_m'] for f in frames])
    target = np.clip(.06 * (actions[:, 7]+1)-.02, 0, .1)
    return q, actions, delta, lag, finger, target


def fit_response(trajectories):
    rows = [samples(f) for f in trajectories]
    params = []
    for j, name in enumerate(NAMES):
        x = np.concatenate([np.column_stack([s[3][:, j], .1*s[1][:, AIND[j]]]) for s in rows])
        y = np.concatenate([s[2][:, j] for s in rows])
        fitted = lsq_linear(x, y, bounds=([0., .001], [.99, 2.]))
        params.append({'name': name, 'rho': float(fitted.x[0]), 'beta': float(fitted.x[1])})
    x = np.concatenate([np.column_stack([s[5]-s[4][:-1], np.ones(len(s[5]))]) for s in rows])
    y = np.concatenate([np.diff(s[4]) for s in rows])
    grip = lsq_linear(x, y, bounds=([.001, -.02], [1., .02])).x
    speeds = np.max(np.concatenate([np.abs(s[2]) for s in rows]), axis=0)
    return {'equation': 'dq[t] = rho*dq[t-1] + beta*0.1*action[t]',
            'joint_parameters': params, 'gripper': {'alpha': float(grip[0]), 'bias_m': float(grip[1]),
            'equation': 'd[t] = clip(d[t-1]+alpha*(clip(0.06*(a_grip+1)-0.02,0,0.1)-d[t-1])+bias,0,0.1)'},
            'reference_speed_limits': speeds.tolist(),
            'initial_delta': 'zero for first transition (unknown pre-snapshot velocity); no differences cross trajectories'}


def errors(model, frames):
    q, a, delta, lag, finger, target = samples(frames)
    rho = np.array([p['rho'] for p in model['joint_parameters']])
    beta = np.array([p['beta'] for p in model['joint_parameters']])
    predicted = rho*lag + beta*.1*a[:, AIND]
    residual = predicted-delta
    grip = model['gripper']
    finger_pred = np.clip(finger[:-1]+grip['alpha']*(target-finger[:-1])+grip['bias_m'],0,.1)
    # Also report free rollout, since a one-step fit need not remain accurate.
    plant = ResponsePlant(model,q[0],finger[0])
    free=[]; free_finger=[]
    for action in a:
        plant.step(action); free.append(plant.q[QIND].copy()); free_finger.append(plant.finger)
    ratios = [float(np.median(delta[:,j][np.abs(a[:,AIND[j]])>.1]/(.1*a[:,AIND[j]][np.abs(a[:,AIND[j]])>.1]))) for j in range(8)]
    return {'transitions':len(a),'joint_errors':[{'name':n,'one_step_rmse':float(np.sqrt(np.mean(residual[:,j]**2))),
        'one_step_mae':float(np.mean(abs(residual[:,j]))),'rollout_q_rmse':float(np.sqrt(np.mean((np.asarray(free)[:,j]-q[1:,QIND[j]])**2))),
        'observed_response_median':ratios[j]} for j,n in enumerate(NAMES)],
        'finger_one_step_rmse_m':float(np.sqrt(np.mean((finger_pred-finger[1:])**2))),
        'finger_rollout_rmse_m':float(np.sqrt(np.mean((np.asarray(free_finger)-finger[1:])**2)))}


@dataclass
class ResponsePlant:
    model: dict
    q: np.ndarray
    finger: float

    def __post_init__(self):
        self.q=np.asarray(self.q,dtype=float).copy()
        self.delta=np.zeros(8)
        self.steps=0

    def step(self, action):
        action=np.asarray(action,dtype=float)
        rho=np.array([p['rho'] for p in self.model['joint_parameters']])
        beta=np.array([p['beta'] for p in self.model['joint_parameters']])
        self.delta=rho*self.delta+beta*.1*action[AIND]
        self.q[QIND]+=self.delta
        self.q[3]=np.clip(self.q[3],0,.386)
        # Auxiliary head response is unidentifiable in stationary-head teachers.
        # Use median arm response, label it as extrapolated (not used to fit arm).
        for qi,ai in ((4,8),(6,9)):
            self.q[qi]+=float(np.median(beta[:7]))*.1*action[ai]
        g=self.model['gripper']; target=np.clip(.06*(action[7]+1)-.02,0,.1)
        self.finger=float(np.clip(self.finger+g['alpha']*(target-self.finger)+g['bias_m'],0,.1))
        self.q[-2:]=self.finger/2
        self.steps+=1
        return {'terminated':False,'truncated':False}


# ========================================================================
# Frozen source: eef_streaming.py
"""Independent streaming executor; robot-only planning, no scene or simulator API."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .eef_tools import ARM, BODY, REST, FETCH_ORDER, densify, vector



def grip_band(command):
    if not np.isfinite(command) or not 0<=command<=1:raise ValueError('invalid gripper command')
    return 0 if command<=.1 else 2 if command>=.9 else 1


def compress_pose_indices(frames, start, end, position_tol=.01, rotation_tol=.05):
    keep={start,end}
    def visit(a,b):
        if b-a<2:return
        poses=[np.asarray(frames[i]['tcp_base']) for i in (a,b)]
        slerp=Slerp([a,b],Rotation.from_matrix(np.stack([p[:3,:3] for p in poses])))
        errors=[]
        for i in range(a+1,b):
            t=(i-a)/(b-a); actual=np.asarray(frames[i]['tcp_base'])
            pe=np.linalg.norm(actual[:3,3]-((1-t)*poses[0][:3,3]+t*poses[1][:3,3]))/position_tol
            re=(Rotation.from_matrix(actual[:3,:3])*slerp(i).inv()).magnitude()/rotation_tol
            errors.append(max(pe,re))
        if max(errors)>1:
            i=a+1+int(np.argmax(errors));keep.add(i);visit(a,i);visit(i,b)
    visit(start,end)
    return sorted(keep)


def teacher_segments(frames):
    if len(frames)<2:raise ValueError('teacher needs a transition')
    events=[i for i in range(1,len(frames)) if grip_band(frames[i]['gripper_command'])!=grip_band(frames[i-1]['gripper_command'])]
    # Commands belong to the transition ending at frame i; segments partition
    # actions 1..N, not the pre-action initial sensor frame.
    starts=sorted({1,*events}); out=[]
    for index,start in enumerate(starts):
        end=starts[index+1]-1 if index+1<len(starts) else len(frames)-1
        out.append({'start_action':start,'end_action':end,
                    'gripper':float(np.median([f['gripper_command'] for f in frames[start:end+1]])),
                    'indices':compress_pose_indices(frames,start-1,end)})
    return {'event_indices':events,'segments':out,
            'keyframe_indices':sorted({i for s in out for i in s['indices']})}


@dataclass(frozen=True)
class StreamingParameters:
    gain: float
    lookahead: int=1
    release_tolerance: float=.05
    joint_tolerance: float=.002
    endpoint_steps_cap: int=10
    gripper_steps_cap: int=5
    gripper_stable_steps: int=2
    gripper_change_m: float=.001

    def __post_init__(self):
        if not np.isfinite(self.gain) or self.gain<=0 or self.lookahead<0:raise ValueError('invalid gain/lookahead')
        if self.endpoint_steps_cap<0 or self.gripper_steps_cap<1 or self.gripper_stable_steps<1:raise ValueError('invalid settling caps')


class FetchStreamingExecutor:
    def __init__(self,ik,read_qpos,step,read_finger,model,parameters,*,max_steps=200,gripper=1.):
        if ik.fk.order!=FETCH_ORDER or not isinstance(max_steps,int) or max_steps<0:raise ValueError('unsupported robot/budget')
        grip_band(gripper)
        self.ik,self.read_qpos,self.step,self.read_finger=ik,read_qpos,step,read_finger
        self.model,self.parameters=model,parameters
        self.max_steps,self.steps,self.stopped=max_steps,0,False
        self.gripper=gripper; self.events=[]
        self.controlled=np.array([ik.fk.index[n] for n in (*ARM,*BODY)])
        self.action_q=np.array([ik.fk.index[n] for n in (*ARM,*BODY)])

    def _step(self,action):
        if self.stopped or self.steps>=self.max_steps:return False
        status=self.step(action); self.steps+=1
        self.stopped=bool(status.get('terminated') or status.get('truncated'))
        return not self.stopped

    def action(self,target):
        q=vector(self.read_qpos(),15)
        a=np.zeros(13); delta=target-q
        a[:7]=np.clip(self.parameters.gain*delta[self.action_q[:7]]/.1,-1,1)
        a[8:11]=np.clip(self.parameters.gain*delta[self.action_q[7:]]/.1,-1,1)
        a[7]=2*self.gripper-1
        return a

    def _parameterize(self,refs):
        refs=np.asarray(refs,dtype=float)
        # Coordinate units: arm/head radians, torso metres. Limits are measured
        # independently for each controlled arm/torso channel, not scene truth.
        speed=np.maximum(np.asarray(self.model['reference_speed_limits']),1e-5)
        if len(refs)==1:return refs
        lengths=np.max(np.abs(np.diff(refs[:,QIND],axis=0))/speed,axis=1)
        head=np.max(abs(np.diff(refs[:,[4,6]],axis=0)),axis=1)/float(np.median(speed[:7]))
        lengths=np.maximum(lengths,head)
        times=np.r_[0,np.cumsum(lengths)]
        valid=np.r_[True,np.diff(times)>1e-10]; refs=refs[valid]; times=times[valid]
        if len(refs)==1:return refs
        ticks=np.linspace(0,times[-1],max(1,int(np.ceil(times[-1])))+1)
        return np.column_stack([np.interp(ticks,times,refs[:,j]) for j in range(15)])

    def _plan(self,targets,seed=None):
        q=vector(self.read_qpos() if seed is None else seed,15).copy(); refs=[q.copy()]
        points=densify(self.ik.fk.tcp(q),targets)
        if not points:raise ValueError('empty path')
        for point in points:
            result=self.ik.solve(point,q)
            if not result.accepted:return {'accepted':False,'reason':'ik_not_found','refs':None}
            q=result.qpos;refs.append(q.copy())
        return {'accepted':True,'reason':'ok','refs':self._parameterize(refs),'goal':points[-1]}

    def _arrived(self,target,goal=None):
        q=vector(self.read_qpos(),15)
        if np.max(abs(q[self.controlled]-target[self.controlled]))<=self.parameters.joint_tolerance:return True
        if goal is None:return False
        measured=self.ik.fk.tcp(q)
        pe=np.linalg.norm(measured[:3,3]-goal[:3,3])
        re=Rotation.from_matrix(goal[:3,:3]@measured[:3,:3].T).magnitude()
        return bool(pe<=self.ik.position_tol and re<=self.ik.rotation_tol)

    def _track(self,refs,goal=None):
        before=self.steps;cursor=0;last=len(refs)-1;p=self.parameters
        # Intermediate references never use the endpoint convergence loop.
        while cursor<last and not self.stopped and self.steps<self.max_steps:
            error=float(np.max(abs(vector(self.read_qpos(),15)[self.controlled]-refs[cursor,self.controlled])))
            if error<=p.release_tolerance:cursor+=1
            target=refs[min(cursor+p.lookahead,last)]
            self._step(self.action(target))
        converged=self._arrived(refs[-1],goal) if cursor==last else False
        settling=0
        while not converged and cursor==last and settling<p.endpoint_steps_cap and not self.stopped and self.steps<self.max_steps:
            self._step(self.action(refs[-1]));settling+=1
            converged=self._arrived(refs[-1],goal)
        record={'arrived':converged,'steps_used':self.steps-before,'endpoint_settle_steps':settling,
                'reference_points':len(refs),'stopped':self.stopped,'cursor':cursor,
                'max_controlled_joint_error':float(np.max(abs(vector(self.read_qpos(),15)[self.controlled]-refs[-1,self.controlled])))}
        if goal is not None:
            measured=self.ik.fk.tcp(self.read_qpos())
            record['endpoint_position_error_m']=float(np.linalg.norm(measured[:3,3]-goal[:3,3]))
            record['endpoint_rotation_error_rad']=float(Rotation.from_matrix(goal[:3,:3]@measured[:3,:3].T).magnitude())
        self.events.append(record)
        return record

    def move_eef_chunk(self,targets):
        plan=self._plan(targets)
        if not plan['accepted']:return {'arrived':False,'reason':plan['reason'],'steps_used':0}
        return self._track(plan['refs'],plan['goal'])

    def move_to(self,target):return self.move_eef_chunk([target])

    def set_gripper(self,openness):
        grip_band(openness);self.gripper=float(openness);before=self.steps
        stable=0;prior=float(self.read_finger());hold=vector(self.read_qpos(),15).copy()
        while stable<self.parameters.gripper_stable_steps and self.steps-before<self.parameters.gripper_steps_cap and not self.stopped and self.steps<self.max_steps:
            self._step(self.action(hold));current=float(self.read_finger())
            stable=stable+1 if abs(current-prior)<self.parameters.gripper_change_m else 0
            prior=current
        return {'steps_used':self.steps-before,'stable':stable>=self.parameters.gripper_stable_steps,'finger_distance_m':prior}

    def return_to_rest(self):
        seed=vector(self.read_qpos(),15).copy();target=seed.copy();target[3:-2]=REST[3:-2]
        return self._track(self._parameterize([seed,target]))

    def check_path(self,targets,*,gripper_commands=(),include_rest=False):
        # A separate response plant predicts without touching live state/callbacks.
        plan=self._plan(targets)
        if not plan['accepted']:return {**plan,'estimated_steps':None,'is_estimate':True}
        plant=ResponsePlant(self.model,self.read_qpos(),self.read_finger())
        copy=FetchStreamingExecutor(self.ik,lambda:plant.q,plant.step,lambda:plant.finger,
             self.model,self.parameters,max_steps=2000,gripper=self.gripper)
        for command in gripper_commands:copy.set_gripper(command)
        outcome=copy._track(plan['refs'],plan['goal'])
        rest=copy.return_to_rest() if include_rest else None
        accepted=outcome['arrived'] and (rest is None or rest['arrived']) and copy.steps<=self.max_steps-self.steps
        return {'accepted':accepted,'reason':'ok' if accepted else 'predicted_tracking_or_budget',
                'estimated_steps':copy.steps,'is_estimate':True,'contact_physics':False,
                'parameters':asdict(self.parameters)}


# ========================================================================
# Frozen source: eef_streaming_v2.py
"""Streaming v2: last mobile-torso candidate, no simulator or scene imports."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from .eef_tools import ARM, BODY, REST, FETCH_ORDER, FetchIK, densify, vector



EEF_STREAMING_V2_VERSION='eef-streaming-v2-last-mobile-torso'


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
            targets=[np.asarray(frames[i]['tcp_base']) for i in s['indices'][1:]]
            alignment=None;grip=None
            if closed:
                # Switch at the pre-action event frame, before the closed segment
                # (including the lift). Never postpone closure until its endpoint.
                event_pose=np.asarray(frames[s['indices'][0]]['tcp_base'])
                alignment=self.close_alignment(event_pose)
                if alignment['passed']:grip=self.set_gripper(s['gripper'],event_pose)
            else:grip=self.set_gripper(s['gripper'])
            motion=self.move_eef_chunk(targets) if not closed or alignment['passed'] else {'completed_reference':False,'steps_used':0,'reason':'close_alignment_failed'}
            ok=motion['completed_reference'] and (not closed or bool(alignment and alignment['passed']))
            segments.append({'indices':s['indices'],'command':s['gripper'],'enter_closed':closed,
                'motion':motion,'alignment':alignment,'gripper_wait':grip,'steps_used':self.steps-before})
            if not ok or self.stopped:all_done=False;break
        rest=self.return_to_rest() if all_done else None
        complete=all_done and rest is not None and rest['passed']
        return {'version':EEF_STREAMING_V2_VERSION,'complete':bool(complete),'predicted_steps':self.steps if complete else None,
                'steps_until_end_or_abort':self.steps,'segments':segments,'arm_ik_wait_steps':self.wait_steps,
                'close_alignments':self.alignments,'rest':rest,'parameters':asdict(self.p)}

