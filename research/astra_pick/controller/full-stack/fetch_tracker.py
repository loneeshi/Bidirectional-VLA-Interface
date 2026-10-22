"""Fetch full-core CPU integration. No environment runner or permission to execute physics.
Upstream modules remain byte-identical. Runtime must pass independently validated
RobotSpec/model; the provided kinematics-only model is for wiring diagnostics only.
"""
import copy
import numpy as np
import mujoco
import mink
from flowsde_ctrl.robot import RobotModel
from flowsde_ctrl.tracker import WholeBodyTracker
from flowsde_ctrl import variants
from fetch_driver import encode_arm

class CurrentMujocoModel(RobotModel):
    """Use current mj_fullM signature without changing upstream FK/servo math."""
    def fk(self,q,h=None,jac=True,inertia=False):
        out=super().fk(q,h,jac=jac,inertia=False)
        if inertia:
            mujoco.mj_crb(self.m,self.d)
            mujoco.mj_fullM(self.m,self.d,self.M)
            out.append(self.M[np.ix_(self.av,self.av)].copy())
        return out


def restore_loader_inertias(model,readback):
    """Restore original positive principal tensors after kinematics compilation.

    MuJoCo's XML compiler rejects triangle-invalid finger tensors accepted by
    PhysX. Its model arrays can represent them; restore every robot body's
    readback tensor/COM before computing mass matrices. No balancing is retained.
    This is a CPU nominal model, not evidence of integrator/drive equivalence.
    """
    facts={r['name']:r for r in readback['links']}
    restored=[]
    for bid in range(1,model.nbody):
        name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,bid)
        if name not in facts:raise ValueError('Missing loader inertia: '+name)
        row=facts[name];I=np.asarray(row['inertia'],float);mass=float(row['mass'])
        if I.shape!=(3,) or np.any(I<=0) or mass<=0 or not np.isfinite(np.r_[I,mass]).all():
            raise ValueError('Invalid loader inertia readback')
        model.body_mass[bid]=mass;model.body_inertia[bid]=I
        model.body_ipos[bid]=row['cmass_position'];model.body_iquat[bid]=row['cmass_quaternion_wxyz']
        restored.append(name)
    # Agent PDJointPosController sets drive damping=100 and friction=0. Those
    # replace URDF drive properties. The servo already includes KD=100.
    # Therefore do not add a second URDF damping/friction channel to the model.
    model.dof_damping[:]=0.;model.dof_frictionloss[:]=0.
    model.opt.gravity[:]=0.  # source-verified Fetch balance_passive_force behavior
    scratch=mujoco.MjData(model);mujoco.mj_setConst(model,scratch)
    for name in restored:
        bid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,name)
        if not np.array_equal(model.body_inertia[bid],facts[name]['inertia']):
            raise AssertionError('Readback inertia changed during constant rebuild')
    return restored


from flowsde_ctrl.servo import PredictiveServo

class FetchImplicitServo(PredictiveServo):
    """Original optimizer/DOB with Fetch's implicit-PD nominal transition.

    Frozen M, no torque saturation/contact constraints within this linear model.
    q+ = q + h v+; (M+h Kd+h^2 Kp)v+ = M v + h(u-Kp q).
    This is an isolated-drive nominal model, not an exact PhysX solver replica.
    DOB implementation and command optimization below are unchanged upstream.
    """
    def transition(self,M,kp_s=1.,D=None):
        n=len(self.KP);I=np.eye(n);h=self.h
        M=np.asarray(M,float);D=self.D_nom if D is None else np.asarray(D,float)
        K=kp_s*self.KP;Kd=self.KD+D
        if M.shape!=(n,n) or not np.isfinite(M).all() or not np.allclose(M,M.T,atol=1e-9):
            raise ValueError('Invalid nominal inertia')
        np.linalg.cholesky(M)
        A=M+h*np.diag(Kd)+h*h*np.diag(K)
        inv=np.linalg.solve(A,I);P=-h*inv*K;V=np.linalg.solve(A,M)
        F=np.block([[I+h*P,h*V],[P,V]])
        G=np.vstack([h*h*inv,h*inv])
        return F,G

    def model_FG(self,M):
        return self.transition(M)

    def drive_torque_preview(self,q,qd,q_sp,qd_ff):
        """Nominal implicit end-substep drive torque, not a measured contact force.

        Uses the same frozen M and applied command as the predictor. tau_d is
        an external residual, so it influences predicted motion but is excluded
        from the actuator torque itself. A threshold crossing makes the linear
        DOB inverse invalid; this preview does not simulate clipped dynamics.
        """
        if self.last_M is None or self.last_u is None:
            raise ValueError('No command prediction available')
        F,G=self.model_FG(self.last_M);state=np.r_[q,qd];trace=[]
        for _ in range(self.n_sub):
            state=F@state+G@self.last_u
            n=len(self.KP)
            trace.append(self.KP*(q_sp-state[:n])+self.KD*(qd_ff-state[n:]))
        return np.asarray(trace)

    def command(self,q,qd,M,q_t,qd_t,kp_s=1.,D=None):
        c=self.c;h=self.h;n=len(q);In=np.eye(n)
        K=kp_s*self.KP
        F,Gc=self.transition(M,kp_s,D)
        Fn = np.eye(2 * n); S = np.zeros((2 * n, 2 * n))
        for _ in range(self.n_sub):
            S += Fn; Fn = F @ Fn
        x0 = np.concatenate([q, qd])
        SG = S @ Gc
        W = np.concatenate([np.ones(n), c.mpc_wv * np.ones(n)])
        A = W[:, None] * SG; AA = A.T @ A
        cvec = np.linalg.solve(AA + c.mpc_reg * np.trace(AA) / n * In, A.T @ (W * (np.concatenate([q_t, qd_t]) - Fn @ x0)))
        fric = self.F_nom * np.tanh(qd_t / c.fric_v)
        win, vff = self.inner.sp_window, self.inner.vff_scale
        q_sp = np.clip(0.5 * (q + q_t), q - win, q + win)
        qd_ff = (cvec - K * q_sp - self.tau_d + fric) / self.KD
        qd_c = np.clip(qd_ff, -vff, vff)                                   # overflow -> position channel
        q_sp = np.clip(q_sp + self.KD * (qd_ff - qd_c) / K, q - win, q + win)
        self.pred = Fn @ x0 + SG @ (K * q_sp + self.KD * qd_c + self.tau_d - fric)
        self.SG = SG
        self.last_u = kp_s * self.KP * q_sp + self.KD * qd_c + self.tau_d - fric
        self.last_M = M
        return q_sp, qd_c


from flowsde_ctrl.ik import DLSIK

class FetchDLSIK(DLSIK):
    """Keep upstream DLS objective; bound the total reference increment per tick.

    Upstream caps each Gauss-Newton iteration separately, so two iterations can
    produce twice its configured speed. RobotSpec limits otherwise apply only
    to mink QP. Uniform line scaling preserves the solved joint-space direction.
    This is a reference bound, not a guarantee on measured actuator velocity.
    """
    def __init__(self,model,cfg,dt,velocity_limits):
        super().__init__(model,cfg,dt)
        limits=np.asarray(velocity_limits,float)
        if limits.shape!=(model.n,) or not np.isfinite(limits).all() or np.any(limits<=0):
            raise ValueError('Missing/invalid Fetch joint speed limits')
        self.velocity_limits=np.minimum(limits,float(cfg.qd_max))
        self.last_reference_scale=1.

    def solve(self,q0,h,pt,Rt):
        q0=np.asarray(q0,float)
        if np.any(q0<self.q_lo) or np.any(q0>self.q_hi):
            raise ValueError('IK reference starts outside robot joint bounds')
        proposed=super().solve(q0,h,pt,Rt)
        delta=proposed-q0
        ratio=float(np.max(abs(delta)/(self.velocity_limits*self.dt)))
        self.last_reference_scale=1./max(1.,ratio)
        return q0+self.last_reference_scale*delta


class FetchTrackerSession:
    """One episode owns one actual WholeBodyTracker. Phases never reset it.

    Input measurements must be fresh at spec.inner's real control period. The
    caller constructs previews in the measured base frame using the same pose
    delay convention. No target-object state or collision prefilter is accepted.
    """
    def __init__(self,spec,model,variant='cc_obs_v5'):
        if variant not in ('fair','combo_dob_bp','cc_obs_v5'):
            raise ValueError('This bridge supports fair/combo_dob_bp/cc_obs_v5; latency and F/T bridges remain unverified')
        if model.n != 7 or spec.base.type != 'differential':
            raise ValueError('Expected Fetch seven-joint arm and differential base')
        if spec.inner.sp_scale != .1 or spec.inner.vff_scale != 1.:
            raise ValueError('Fetch dual-channel bounds do not match RobotSpec')
        self.spec=spec; self.model=model
        cfg=variants.make(variant)
        self.ctrl=WholeBodyTracker(spec,model,cfg)
        self.ctrl.servo=FetchImplicitServo(model,spec.inner,cfg.servo)
        self.ctrl.ik=FetchDLSIK(model,cfg.ik,self.ctrl.dtp,spec.qd_max)
        if hasattr(self.ctrl.base,'qp'):
            # Model base axes are fixed to the current base frame. Enforce lateral
            # zero in the QP, before planning/servo, instead of discarding its plan.
            self.ctrl.base.qp.constraints.append(mink.DofFreezingTask(model.m,dof_indices=[int(model.bv[1])]))
        self.first=True; self.last_sequence=None; self.phase=None; self.phase_log=[]
        self.official_end=None
        self.drive_interval=None;self.dob_skipped_saturation=0

    def latch_official_end(self,*,terminated,truncated):
        """Consume only official stop flags. Never reset the episode after termination."""
        if type(terminated) is not bool or type(truncated) is not bool:
            raise ValueError('Official flags must be booleans')
        if terminated or truncated:
            self.official_end=dict(terminated=terminated,truncated=truncated)
        return self.official_end is not None

    def begin_rest(self,arm_target,torso_target):
        """Supply a robot-defined rest posture without resetting any observer.

        Original tracker has no Fetch task/rest primitive. Set its existing
        posture objective and return the corresponding robot-only EE target;
        caller must stream a bounded preview and command torso/head separately.
        This does not declare official robot_rest: redundant joints may lag.
        """
        if self.official_end is not None:raise StopIteration('Official episode ended')
        if self.first:raise ValueError('Initialize the episode before a rest transition')
        target=np.asarray(arm_target,float)
        if target.shape!=(7,) or not np.isfinite(target).all() or not np.isfinite(torso_target):
            raise ValueError('Invalid robot rest posture')
        if np.any(target<self.model.q_lo) or np.any(target>self.model.q_hi):
            raise ValueError('Rest posture outside robot joint bounds')
        # Use the exact raw representation; do not wrap continuous joints.
        self.ctrl.ik.q_post=target.copy()
        return self.model.fk(target,float(torso_target),jac=False)

    def tick(self,inp,*,sequence,phase):
        if self.official_end is not None:raise StopIteration('Official episode ended')
        if phase not in ('open','pregrasp','descend','close','settling','lift','rest','hold'):
            raise ValueError('Unknown task phase')
        if self.last_sequence is not None and sequence != self.last_sequence+1:
            raise ValueError('Missing/duplicate feedback tick: do not invent interpolated observations')
        if not self.first and inp.episode_start:
            raise ValueError('Unexpected episode reset; create a new session after restore')
        x=copy.deepcopy(inp); x.episode_start=self.first
        q=np.asarray(x.q);qd=np.asarray(x.qd); times=np.asarray(x.ref_t)
        if q.shape!=(7,) or qd.shape!=(7,) or not np.isfinite(np.r_[q,qd,x.h]).all():
            raise ValueError('Invalid measured joint state')
        horizon=2*self.ctrl.dtp+self.ctrl.dl
        if len(times)<4 or not np.isfinite(times).all() or np.any(np.diff(times)<=0) or times[0]>0 or times[-1]<max(.5,horizon):
            raise ValueError('Need monotone preview covering now, >=0.5s and both servo targets')
        if np.shape(x.ref_rel_p)!=(len(times),3) or np.shape(x.ref_rel_R)!=(len(times),3,3):
            raise ValueError('Preview shape mismatch')
        if not np.isfinite(x.ref_rel_p).all() or not np.isfinite(x.ref_rel_R).all():
            raise ValueError('Nonfinite reference preview')
        if self.phase!=phase:self.phase_log.append(dict(sequence=sequence,phase=phase))
        # The linear SG inverse cannot distinguish a clipped actuator from an
        # external disturbance. Skip this interval as a whole: dense inertia
        # couples even apparently unsaturated joints. Preserve learned tau_d.
        if self.drive_interval is not None and self.drive_interval['linear_dob_invalid']:
            self.ctrl.servo.pred=None
            self.dob_skipped_saturation+=1
        cmd=self.ctrl.step(x)
        arm=encode_arm(x.q,cmd.q_sp,cmd.qd_ff)
        # Reconcile float32 quantization/clipping with the actual driver command:
        # DOB must compare observations to what was sent, not the unclipped intent.
        delta_u=self.ctrl.servo.KP*(arm.applied_position-cmd.q_sp)+self.ctrl.servo.KD*(arm.applied_velocity-cmd.qd_ff)
        self.ctrl.servo.pred+=self.ctrl.servo.SG@delta_u
        self.ctrl.servo.last_u+=delta_u
        torque=self.ctrl.servo.drive_torque_preview(x.q,x.qd,arm.applied_position,arm.applied_velocity)
        limit=np.asarray(self.spec.inner.tau_max,float)
        fraction=self.ctrl.obs.c.sat_frac if self.ctrl.obs is not None else None
        fraction=.9 if fraction is None else fraction
        near=np.any(abs(torque)>fraction*limit,axis=0)
        self.drive_interval=dict(nominal_drive_torque=torque.tolist(),near_limit=near.tolist(),
            linear_dob_invalid=bool(np.any(near)),threshold_fraction=float(fraction),
            measured=False)
        if self.ctrl.obs is not None:
            # Replace the upstream instantaneous explicit-PD estimate with the
            # matching implicit substep estimate after float32 action encoding.
            self.ctrl.obs.sat_next=near.copy()
        self.first=False;self.last_sequence=sequence;self.phase=phase
        # Evaluation-side snapshots only; never feed these back into control.
        cmd.info['fetch_diagnostics']=dict(sequence=sequence,phase=phase,tick=self.ctrl.k,
            control_dt=self.ctrl.dtp,measured_q=q.copy(),measured_qd=qd.copy(),
            q_ref=self.ctrl.q_ref.copy(),q_sp=np.asarray(cmd.q_sp).copy(),
            qd_ff=np.asarray(cmd.qd_ff).copy(),tau_d=self.ctrl.servo.tau_d.copy(),
            predicted_state=self.ctrl.servo.pred.copy(),last_u=self.ctrl.servo.last_u.copy(),
            dob_skipped_saturation=self.dob_skipped_saturation,
            drive_interval=copy.deepcopy(self.drive_interval))
        return cmd,arm


from flowsde_ctrl.nav import NavTracker
from flowsde_ctrl.perception import Perception,HeightMap

class MeasuredDepthMap(Perception):
    """Upstream height-map fusion/queries fed by measured OpenCV depth, not rays.

    Inherits the original optimistic unknown=free semantics; no claim of lidar
    equivalence, complete coverage or collision avoidance. Extrinsics must come
    from robot calibration/FK plus local odometry, never a simulator camera pose.
    """
    def __init__(self,grid,threshold=.03):
        self.grid=grid;self.thr=float(threshold);self.map=HeightMap(grid,self.thr)
        self.sensor=None;self.last_frame_time=None
    def update(self,t):
        if self.last_frame_time is None:raise ValueError('No measured depth map available')
        if t+1e-9<self.last_frame_time:raise ValueError('Depth timestamp is in the future')
    def push_depth(self,depth_m,K,local_from_camera,*,timestamp,robot_mask=None):
        return self.push_batch([(depth_m,K,local_from_camera,robot_mask)],timestamp=timestamp)
    def push_batch(self,frames,*,timestamp):
        # Synchronous cameras share one timestamp. Validate every frame before
        # changing the persistent map; do not invent per-camera clock offsets.
        if not np.isfinite(timestamp) or (self.last_frame_time is not None and timestamp<=self.last_frame_time):
            raise ValueError('Depth frames must be strictly newer')
        if not frames:raise ValueError('No depth frames')
        points=np.concatenate([self.project_depth(*frame) for frame in frames],axis=0)
        if not len(points):raise ValueError('No valid depth measurements')
        self.map.fuse(('pts',points));self.last_frame_time=float(timestamp)
        return points
    @staticmethod
    def project_depth(depth_m,K,local_from_camera,robot_mask=None):
        z=np.asarray(depth_m,float);K=np.asarray(K,float);T=np.asarray(local_from_camera,float)
        if z.ndim!=2 or K.shape!=(3,3) or T.shape!=(4,4) or not np.isfinite(K).all() or not np.isfinite(T).all():
            raise ValueError('Invalid depth/calibration shapes')
        if K[0,0]<=0 or K[1,1]<=0 or not np.allclose(K[2],[0,0,1]) or abs(K[0,1])+abs(K[1,0])>1e-12:
            raise ValueError('Expected calibrated pinhole intrinsics without skew')
        R=T[:3,:3]
        if not np.allclose(T[3],[0,0,0,1]) or not np.allclose(R.T@R,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(R),1,atol=1e-6):
            raise ValueError('Invalid camera rigid transform')
        v,u=np.indices(z.shape);valid=np.isfinite(z)&(z>0)
        if robot_mask is not None:
            mask=np.asarray(robot_mask)
            if mask.dtype!=bool or mask.shape!=z.shape:raise ValueError('Robot self mask must match depth')
            valid &= ~mask
        # robot_mask must be constructed from calibrated robot geometry, not scene IDs.
        xyz=np.stack([(u[valid]-K[0,2])*z[valid]/K[0,0],(v[valid]-K[1,2])*z[valid]/K[1,1],z[valid]],axis=1)
        points=xyz@R.T+T[:3,3]
        return points

from flowsde_ctrl.perception import BandedMap

class _MeasuredBands(BandedMap):
    """Original band fusion/query methods, with measured depth and no fake lidar."""
    def __init__(self,grid,threshold):
        self.grid=grid;self.thr=threshold;self.dz=.05;self.nb=40
        self.occ_d=np.zeros((grid.nx,grid.ny,self.nb),bool)
        self.occ_l=np.zeros_like(self.occ_d)  # unavailable lidar stays empty
        self.version=0;self._clr=None;self._clr_ver=-1
    def fuse(self,frame):
        if frame[0]!='pts':raise ValueError('Measured depth points required')
        self._fuse_depth(frame[1]);self.version+=1

class MeasuredBandedDepthMap(MeasuredDepthMap):
    """Depth-only adaptation of upstream BandedMap; unknown remains free.

    It is not the upstream depth+360-degree-lidar sensing condition.
    """
    def __init__(self,grid,threshold=.03):
        super().__init__(grid,threshold);self.map=_MeasuredBands(grid,self.thr)
    def sphere_query(self,P,rinf=.35):return self.map.sphere_query(P,rinf)


class MeasuredNavTracker(NavTracker):
    """Native DP/QP path with explicit measured-map injection.

    world supplies local odometry and planner reference only. Original reset's
    PC.make(world) is replaced to avoid exact-map/raycast sensor construction.
    Robot spheres are body-local calibration data, not scene collision meshes.
    """
    def __init__(self,spec,model,cfg,world,measured_map,robot_spheres):
        if not isinstance(measured_map,MeasuredDepthMap):raise ValueError('Expected measured depth map')
        if cfg.level!='measured_depth':raise ValueError('Explicit measured_depth level required')
        if not robot_spheres:raise ValueError('Calibrated robot spheres required')
        self.measured_map=measured_map;self.body_spheres=robot_spheres
        super().__init__(spec,model,cfg,world)
        self.qp.constraints.append(mink.DofFreezingTask(model.m,dof_indices=[int(model.bv[1])]))
    def _spheres_def(self):
        self.spheres=[]
        for name,offset,radius,kind in self.body_spheres:
            b=mujoco.mj_name2id(self.mod.m,mujoco.mjtObj.mjOBJ_BODY,name)
            off=np.asarray(offset,float)
            if b<0 or off.shape!=(3,) or not np.isfinite(off).all() or not np.isfinite(radius) or radius<=0 or kind not in (0,1):
                raise ValueError('Invalid robot sphere definition')
            if kind==0 or self.c.arm_spheres:self.spheres.append((b,off,float(radius),kind))
    def reset(self,inp):
        self.base_map.reset();self.qp.reset(inp.q,inp.h)
        self.perc=self.measured_map
        self.plan=None;self.t_plan=-1e9;self.qp.n_fail=0;self.min_clear=np.inf


class FetchNavSession:
    """Persistent native DP/QP -> Fetch driver bridge, no predictive servo.

    Caller updates measured-map/world odometry before each tick. Execution and
    task termination remain the outer runner's responsibility.
    """
    def __init__(self,tracker,action_mapping):
        if not isinstance(tracker,MeasuredNavTracker):raise ValueError('Native measured nav tracker required')
        self.tracker=tracker;self.mapping=action_mapping;self.last_sequence=None;self.ended=False
    def official_stop(self):self.ended=True
    def tick(self,inp,*,sequence,head_q,head_target,grip_closed):
        from fetch_driver import encode_flow_robot,flatten_groups
        if self.ended:raise StopIteration('Official episode ended')
        if type(sequence) is not int or sequence<0 or (self.last_sequence is not None and sequence!=self.last_sequence+1):
            raise ValueError('Missing/duplicate navigation feedback')
        if self.last_sequence is not None and inp.episode_start:raise ValueError('Unexpected nav reset')
        x=copy.deepcopy(inp);x.episode_start=self.last_sequence is None
        if np.shape(x.q)!=(7,) or np.shape(x.qd)!=(7,) or not np.isfinite(np.r_[x.q,x.qd,x.h]).all():
            raise ValueError('Invalid nav measured state')
        command=self.tracker.step(x)
        arm=encode_arm(x.q,command.q_sp,command.qd_ff)
        if np.shape(command.base_cmd)!=(3,) or abs(command.base_cmd[1])>1e-6:
            raise ValueError('Differential Fetch received lateral nav command')
        robot=encode_flow_robot(arm,head_q=head_q,head_target=head_target,torso_velocity=command.vz,
            base_velocity=[command.base_cmd[0],command.base_cmd[2]],grip_closed=grip_closed)
        action=flatten_groups(robot,self.mapping)
        # QP integrates an arm reference. Retain the setpoint actually encoded
        # after float32 conversion/clipping instead of an unreachable command.
        self.tracker.qp.q_ik=arm.applied_position.copy()
        self.last_sequence=sequence
        return dict(action=action,controller_info=command.info,clipped_groups=robot.clipped_groups,
            sequence=sequence,native_nav=True)
