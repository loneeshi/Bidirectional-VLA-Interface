"""Robot description layer: everything EMBODIMENT-SPECIFIC lives here (and in the adapters / embodiment presets).

A `RobotSpec` names the joints / frames of an MJCF model and states the hardware facts the controllers rely on:
joint groups, end-effector site, inner-loop PD gains and rates, actuator limits, the identified dynamics, the base
type and actuator model, sensor availability and the nominal sensing latency. `RobotModel` is the controller's private
copy of the (identified) model, used for forward kinematics, Jacobians and the joint-space inertia only. It never reads
the simulator's state; the controller writes the measured arm q and torso height into it.

Clean generalisation of flowsde/classical.py `Mirror` (same arithmetic for the reference embodiment, see
flowsde_ctrl/adapters/unified.py). Task-agnostic components (servo, observer, contact laws, planners) only see
`RobotSpec` / `RobotModel` and the per-tick `TrackInput`.
"""
import copy
from dataclasses import dataclass, field
from typing import Optional, Sequence

import mujoco
import numpy as np


# ---------------------------------------------------------------------------------------------------- spec
@dataclass
class InnerLoopSpec:
    """The robot's low-level joint loop that the 100 Hz controller commands (servo model; see TUNING_GUIDE §2).

    tau = kp_scale * kp (q_sp - q) + kd (qd_ff - qd) + bias(q, qd),  |tau| <= tau_max, run every `physics_dt`
    for `n_sub` steps per controller tick. The setpoint is sent as an offset a in [-1, 1] (q_sp = q + sp_scale a)
    and the velocity feed-forward as qd_ff = vff_scale a. `sp_window` is the largest |q_sp - q| the servo uses."""
    kp: Sequence[float]                      # N m / rad, per arm joint
    kd: Sequence[float]                      # N m s / rad, per arm joint
    tau_max: Sequence[float]                 # actuator torque limits (N m), used by the observer's saturation mask
    physics_dt: float = 0.002                # inner-loop period (s)
    n_sub: int = 5                           # inner steps per controller tick -> controller rate 1 / (n_sub physics_dt)
    sp_scale: float = 0.3                    # setpoint-offset action scale (rad per unit action)
    vff_scale: float = 2.0                   # velocity feed-forward action scale (rad/s per unit action)
    sp_window: float = 0.3                   # |q_sp - q| bound used by the servo (rad)
    pose_delay_steps: int = 1                # staleness of the plant's reported EE pose in physics steps (MuJoCo
                                             # reports site poses from before the last integration step: 1); the
                                             # tracker evaluates the EE at q - pose_delay_steps * physics_dt * qd


@dataclass
class BaseSpec:
    """Mobile base / torso. type: "holonomic" (x / y slides + yaw), "differential" (no lateral motion) or "fixed".
    The base actuator is a velocity source with Coulomb friction; the controller feeds forward `fric_ff` (in command
    units per joint) with an on/off hysteresis (v_on / v_off) to avoid stick-slip chatter."""
    type: str = "fixed"
    joints: Sequence[str] = ()               # (forward slide, side slide, yaw) joint names in the MJCF (holonomic order)
    torso_joint: Optional[str] = None        # prismatic torso lift (or None)
    body: Optional[str] = None               # base frame body (fixed arms: the arm's mounting body / root link)
    vmax: float = 0.5                        # planar speed limit (m/s)
    wmax: float = 0.6                        # yaw-rate limit (rad/s)
    torso_vmax: float = 0.1                  # torso speed limit (m/s)
    fric_ff: Sequence[float] = (0.25, 0.25, 250.0 / 1500.0)   # Coulomb feed-forward per base joint (command units)
    v_on: float = 0.03
    v_off: float = 0.012
    w_on: float = 0.04
    w_off: float = 0.015
    yaw_action_scale: float = 1.5            # yaw command = yaw_action_scale * action
    acc_max: Optional[float] = None          # documented hardware limits (used by the tuning guide checklist)
    yaw_acc_max: Optional[float] = None


@dataclass
class SensorSpec:
    """Which sensors exist. The fair tier needs none beyond joint encoders (q, qd) and base odometry velocities."""
    wrist_ft: bool = False                   # 6-axis F/T between the last link and the hand
    ft_body: Optional[str] = None            # body whose frame is the F/T sensor frame (hand)
    depth: Optional[dict] = None             # depth camera mount, see perception.DepthCameraSim (pos, pitch, fov, res, ...)
    lidar: Optional[dict] = None             # planar lidar mount, see perception.Lidar2DSim (z, n, rmax, ...)


@dataclass
class LatencySpec:
    """Nominal proprioceptive latency (ms, < one controller tick) and how the setpoint is anchored by the plant
    interface ("obs": q_sp = q_observed + offset; "true": q_sp = q_true + offset)."""
    ms: float = 0.0
    anchor: str = "obs"


@dataclass
class RobotSpec:
    name: str
    arm_joints: Sequence[str]
    ee_site: str
    inner: InnerLoopSpec
    base: BaseSpec = field(default_factory=BaseSpec)
    sensors: SensorSpec = field(default_factory=SensorSpec)
    latency: LatencySpec = field(default_factory=LatencySpec)
    qd_max: Optional[Sequence[float]] = None # joint speed limits (rad/s); QP velocity limits
    shoulder_body: Optional[str] = None      # reach-band planner centre (first arm link)
    wrist_body: Optional[str] = None         # wrist centre (last arm link origin)
    hand_body: Optional[str] = None          # hand / tool subtree root (payload, F/T)
    link_bodies: Sequence[str] = ()          # arm link bodies in order (observer gravity columns)
    identified: dict = field(default_factory=dict)   # {"armature": [...], "damping": [...], "friction": [...]} absolute,
                                                     # optional "payload": dict(body, mass, pos)
    q_limits: Optional[Sequence[Sequence[float]]] = None   # (n, 2) joint ranges overriding the MJCF; unlimited
                                                     # (continuous) MJCF joints default to +-2 pi
    gripper: Optional[dict] = None           # informational (the controllers never actuate the gripper)
    notes: str = ""


# ---------------------------------------------------------------------------------------------------- model
class RobotModel:
    """Controller-side nominal / identified model (private MjData). Only robot joints are ever written: arm q and torso
    height from the measurements, base joints at zero, so every quantity is expressed in the CURRENT base frame
    (odometry-free). Used for kinematics and inertia (mass, armature) only.

    model: an MjModel (it is copied unless share=True), the embodiment's MJCF possibly including a scene.
    init_qpos: configuration that defines the base frame and the start posture (default: model.qpos0)."""

    def __init__(self, spec: RobotSpec, model, init_qpos=None, share=False):
        self.spec = spec
        m = model if share else copy.copy(model)
        jid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.arm_j = [jid(n) for n in spec.arm_joints]
        assert min(self.arm_j) >= 0, f"unknown arm joint in {spec.arm_joints}"
        self.aq = np.array([m.jnt_qposadr[j] for j in self.arm_j]); self.av = np.array([m.jnt_dofadr[j] for j in self.arm_j])
        self.n = len(self.arm_j)
        idf = spec.identified or {}
        for key, arr in (("armature", m.dof_armature), ("damping", m.dof_damping), ("friction", m.dof_frictionloss)):
            if key in idf and idf[key] is not None:
                arr[self.av] = np.asarray(idf[key], float)
        if idf.get("payload"):                       # identified payload: point mass on a body (mass, body, pos)
            pl = idf["payload"]; b = bid(pl["body"]); mp = float(pl["mass"]); pos = np.asarray(pl.get("pos", (0, 0, 0)), float)
            mb = m.body_mass[b]
            m.body_ipos[b] = (mb * m.body_ipos[b] + mp * pos) / (mb + mp); m.body_mass[b] = mb + mp
        self.m = m
        b = spec.base
        names = list(b.joints) + ([b.torso_joint] if b.torso_joint else [])
        self.base_j = [jid(n) for n in names]
        self.bq = np.array([m.jnt_qposadr[j] for j in self.base_j], int)
        self.bv = np.array([m.jnt_dofadr[j] for j in self.base_j], int)
        self.n_base = len(b.joints)                       # planar base dofs (0 or 3)
        self.has_torso = b.torso_joint is not None
        self.tq = int(m.jnt_qposadr[jid(b.torso_joint)]) if self.has_torso else None
        self.site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, spec.ee_site)
        assert self.site >= 0, spec.ee_site
        self.bb = bid(b.body) if b.body else 0
        self.q_lo = m.jnt_range[self.arm_j, 0].copy(); self.q_hi = m.jnt_range[self.arm_j, 1].copy()
        unl = m.jnt_limited[self.arm_j] == 0
        self.q_lo[unl], self.q_hi[unl] = -2 * np.pi, 2 * np.pi
        if spec.q_limits is not None:
            ql = np.asarray(spec.q_limits, float); self.q_lo, self.q_hi = ql[:, 0].copy(), ql[:, 1].copy()
        self.d = mujoco.MjData(m)
        q0 = m.qpos0 if init_qpos is None else init_qpos
        self.init_qpos = np.array(q0, float).copy()
        self.d.qpos[:] = self.init_qpos
        if self.n_base:
            self.d.qpos[self.bq[:3]] = 0.0
        mujoco.mj_kinematics(m, self.d)
        self.R0 = self.d.xmat[self.bb].reshape(3, 3).copy(); self.p0 = self.d.xpos[self.bb].copy()
        if self.n_base:
            self.anchor = self.R0.T @ (self.d.xanchor[self.base_j[2]] - self.p0)          # yaw axis point, base frame
            self.ax_w = np.stack([self.d.xaxis[self.base_j[0]], self.d.xaxis[self.base_j[1]]])   # slide axes (world)
        else:
            self.anchor = np.zeros(3); self.ax_w = np.zeros((2, 3))
        sh = bid(spec.shoulder_body) if spec.shoulder_body else int(m.jnt_bodyid[self.arm_j[0]])
        wr = bid(spec.wrist_body) if spec.wrist_body else int(m.jnt_bodyid[self.arm_j[-1]])
        self.shoulder = self.R0.T @ (self.d.xpos[sh] - self.p0)
        self.h0 = float(self.d.qpos[self.tq]) if self.has_torso else 0.0
        Rs = self.d.site_xmat[self.site].reshape(3, 3)
        self.wrist_off = Rs.T @ (self.d.xpos[wr] - self.d.site_xpos[self.site])      # wrist centre in the EE frame
        self.jp = np.zeros((3, m.nv)); self.jr = np.zeros((3, m.nv)); self.M = np.zeros((m.nv, m.nv))
        # identified joint-space parameters seen by the servo / observer
        self.D_nom = m.dof_damping[self.av].copy()
        self.F_nom = m.dof_frictionloss[self.av].copy()
        self.links = [bid(n) for n in spec.link_bodies] if spec.link_bodies else \
            [int(m.jnt_bodyid[j]) for j in self.arm_j]
        hb = bid(spec.hand_body) if spec.hand_body else int(m.site_bodyid[self.site])
        self.hand = hb
        sub, todo = [], [hb]
        while todo:
            x = todo.pop(); sub.append(x); todo += [k for k in range(m.nbody) if m.body_parentid[k] == x and k != x]
        self.hand_sub = [x for x in sub if m.body_mass[x] > 0]

    def fk(self, q, h=None, jac=True, inertia=False):
        """EE position / rotation in the base frame (+ 6 x n arm Jacobian, + n x n arm inertia incl. armature)."""
        d = self.d
        d.qpos[self.aq] = q
        if self.has_torso:
            d.qpos[self.tq] = h
        mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
        out = [self.R0.T @ (d.site_xpos[self.site] - self.p0), self.R0.T @ d.site_xmat[self.site].reshape(3, 3)]
        if jac:
            mujoco.mj_jacSite(self.m, d, self.jp, self.jr, self.site)
            out.append(np.vstack([self.R0.T @ self.jp[:, self.av], self.R0.T @ self.jr[:, self.av]]))
        if inertia:
            mujoco.mj_crb(self.m, d); mujoco.mj_fullM(self.m, self.M, d.qM)
            out.append(self.M[np.ix_(self.av, self.av)].copy())
        return out

    def jac_wb(self, q, h=None):
        """6 x (n + n_base) base-frame EE Jacobian incl. the planar base dofs (generic embodiments; the reference
        adapter takes it from the observation instead)."""
        self.fk(q, h, jac=False)
        mujoco.mj_jacSite(self.m, self.d, self.jp, self.jr, self.site)
        cols = np.r_[self.av, self.bv[:self.n_base]].astype(int)
        return np.vstack([self.R0.T @ self.jp[:, cols], self.R0.T @ self.jr[:, cols]])

    def gravity(self, q, h=None):
        """Nominal gravity torque on the arm joints (links + hand subtree)."""
        m, d = self.m, self.d
        self.fk(q, h, jac=False)
        g = np.zeros(self.n); gv = m.opt.gravity
        for b in self.links + self.hand_sub:
            mujoco.mj_jacBodyCom(m, d, self.jp, None, b)
            g -= self.jp[:, self.av].T @ (m.body_mass[b] * gv)
        return g
