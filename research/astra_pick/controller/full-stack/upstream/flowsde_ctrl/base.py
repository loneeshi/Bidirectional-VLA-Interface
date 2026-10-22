"""Base / torso planning and the base actuator map.

  ReachBandPlanner  classical reach-band planner (flowsde/classical.py ClassicalTracker._base_plan): the base moves only
                    when the previewed WRIST centre leaves a band around the shoulder, plus the joint-limit-induced share
                    of a whole-body DLS. Used by the precise servo tier (cls fair, fair_dc). Embodiment-specific: the band
                    radii / bearing (arm reach), torso band.
  WholeBodyQP       mink (kevinzakka/mink, Apache-2.0) whole-body differential-IK QP on the identified model
                    (flowsde/zoo/mink_wbik.py MinkWBIK): EE FrameTask + PostureTask + base DampingTask, joint-range /
                    joint-speed limits, scene dofs frozen. The mink LIBRARY is used unchanged (pip dependency).
  MinkBasePlanner   the zoo BASE_PLAN as used by flowsde/combo.py (combo_ft_bp / combo_dob_bp / cc_obs_v5): open-loop
                    QP, posture cost 1e-3, base damping 0.3; only its base / yaw / torso velocities are used, with
                    optional command conditioning (low-pass, acceleration limits).
  BaseActuatorMap   desired base-frame velocity -> per-joint command with Coulomb feed-forward and on/off hysteresis
                    (flowsde/classical.py _base_action). Differential bases drop the lateral component.

THE trade-off (TUNING_GUIDE §6): a low posture cost (1e-3) lets the QP move the base early and keeps it off tables
(Square base contact 2 % vs 92 % for the reach band) but costs free-space precision (DROID >= 0.37 mm vs 0.04 mm);
a stiffer posture (3e-3) recovers precision and gives the clearance back (base contact 69 % on a counter-reach proxy).
"""
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from flowsde_ctrl.math3d import rotvec, slerp_R


# ---------------------------------------------------------------------------------------------------- reach band
@dataclass
class ReachBandConfig:
    wrist: bool = True        # use the wrist centre (EE + wrist offset) instead of the EE
    r_lo: float = 0.5         # inner reach radius around the shoulder, horizontal (m)
    r_hi: float = 0.65        # outer reach radius (m)
    phi_max: float = 0.6      # max bearing of the wrist w.r.t. the base heading (rad)
    z_lo: float = -0.3        # torso band: reference height relative to the shoulder (m)
    z_hi: float = 0.0
    T_base: float = 0.3       # time constant of the band correction (s)
    ff_gain: float = 1.0      # gated feed-forward of the previewed reference velocity near the band edges
    ff_band: float = 0.08     # width of the feed-forward gate (m)
    ff_phi: float = 0.15      # bearing gate width (rad)
    wb_gain: float = 1.0      # whole-body term: share of a limit-weighted whole-body DLS shifted to the base
    wb_excess: bool = True    # only the share induced by the limit weights
    wb_s: float = 0.15        # preview time of the reference twist for the whole-body term (s)
    wb_k: float = 1.0
    wb_margin: float = 1.0    # rad from a joint limit where the arm weight starts to grow
    wb_base: float = 20.0     # weight of the base translation dofs in the whole-body DLS
    wb_yaw: float = 20.0      # weight of the yaw dof
    vmax: float = 0.5
    wmax: float = 0.6
    torso_vmax: float = 0.1


class ReachBandPlanner:
    def __init__(self, model, cfg: ReachBandConfig):
        self.mod, self.c = model, cfg

    def reset(self, inp):
        pass

    def plan(self, inp, ref, q):
        c = self.c; mod = self.mod; n = mod.n
        sh = mod.shoulder + np.array([0, 0, inp.h - mod.h0])
        P, Rs, t = ref.P, ref.Rs, ref.t
        m = (t >= 0) & (t <= np.inf)
        fut, tf = P[m], t[m]
        if c.wrist:
            fut = fut + np.einsum("nij,j->ni", Rs[m], mod.wrist_off)
        rel = fut[:, :2] - sh[:2]
        r = np.linalg.norm(rel, axis=1); phi = np.arctan2(rel[:, 1], rel[:, 0])
        dx = np.zeros(2); dyaw = 0.0
        k = int(np.argmax(r))
        if r[k] > c.r_hi:
            dx += (r[k] - c.r_hi) * rel[k] / r[k]
        k2 = int(np.argmin(r))
        if r[k2] < c.r_lo:
            dx += (r[k2] - c.r_lo) * rel[k2] / max(r[k2], 1e-3)
        kk = int(np.argmax(np.abs(phi)))
        if abs(phi[kk]) > c.phi_max:
            dyaw = phi[kk] - np.sign(phi[kk]) * c.phi_max
        v = dx / c.T_base; w = dyaw / c.T_base
        if c.ff_gain > 0 and len(tf) >= 3:
            A = np.stack([tf - tf.mean(), np.ones_like(tf)], 1)
            vel = np.linalg.lstsq(A, fut[:, :2], rcond=None)[0][0]
            nn = rel[0] / max(r[0], 1e-6); vr = vel @ nn; vt = vel @ np.array([-nn[1], nn[0]])
            g_out = np.clip((r.max() - (c.r_hi - c.ff_band)) / c.ff_band, 0, 1)
            g_in = np.clip(((c.r_lo + c.ff_band) - r.min()) / c.ff_band, 0, 1)
            v += c.ff_gain * (g_out * max(vr, 0) + g_in * min(vr, 0)) * nn
            pm = phi[kk]; g_phi = np.clip((abs(pm) - (c.phi_max - c.ff_phi)) / c.ff_phi, 0, 1)
            wt = vt / max(r[0], 0.2)
            if np.sign(wt) == np.sign(pm):
                w += c.ff_gain * g_phi * wt
        if c.wb_gain > 0 and inp.J_wb is not None:
            J = inp.J_wb[:, :n + 3]
            ts = float(np.clip(c.wb_s, ref.t.min(), ref.t.max()))
            xd = np.concatenate([ref.sp.derivative()(ts), ref.Rc @ ref.sw.derivative()(ts)])
            dist = np.minimum(q - mod.q_lo, mod.q_hi - q)
            wa = 1.0 + c.wb_k * np.maximum(c.wb_margin / np.maximum(dist, 1e-3) - 1.0, 0.0) ** 2
            sol = lambda wv: (J / wv).T @ np.linalg.solve((J / wv) @ J.T + 1e-4 * np.eye(6), xd)
            wbase = [c.wb_base] * 2 + [c.wb_yaw]
            qdot = sol(np.concatenate([wa, wbase]))
            if c.wb_excess:
                qdot = qdot - sol(np.concatenate([np.ones(n), wbase]))
            ax = inp.base_axes
            v += c.wb_gain * (qdot[n] * ax[0][:2] + qdot[n + 1] * ax[1][:2]); w += c.wb_gain * qdot[n + 2]
        vn = np.linalg.norm(v)
        if vn > c.vmax:
            v *= c.vmax / vn
        w = float(np.clip(w, -c.wmax, c.wmax))
        zr = fut[:, 2] - sh[2]; vz = 0.0
        if zr.min() < c.z_lo:
            vz = (zr.min() - c.z_lo) / c.T_base
        elif zr.max() > c.z_hi:
            vz = (zr.max() - c.z_hi) / c.T_base
        return v, w, float(np.clip(vz, -c.torso_vmax, c.torso_vmax))


# ---------------------------------------------------------------------------------------------------- actuator map
class BaseActuatorMap:
    """Base-frame planar velocity v (2) and yaw rate w -> per-joint commands [ax0, ax1, yaw] incl. Coulomb
    feed-forward with on/off hysteresis (the real base is a velocity actuator with Coulomb friction)."""

    def __init__(self, base_spec):
        self.b = base_spec
        self.on = np.zeros(3, bool)

    def reset(self):
        self.on = np.zeros(3, bool)

    def __call__(self, inp, v, w):
        b = self.b; ax = inp.base_axes
        if b.type == "fixed":
            return np.zeros(3)
        if b.type == "differential":                     # non-holonomic: no lateral motion (heading via yaw)
            v = np.array([v[0], 0.0])
        vj = np.array([ax[0][:2] @ v, ax[1][:2] @ v, w])
        on = np.array([b.v_on, b.v_on, b.w_on]); off = np.array([b.v_off, b.v_off, b.w_off])
        fr = np.array(b.fric_ff, float)
        self.on = np.where(np.abs(vj) > on, True, np.where(np.abs(vj) < off, False, self.on))
        return np.where(self.on, vj + fr * np.sign(vj), 0.0)


# ---------------------------------------------------------------------------------------------------- mink whole-body QP
@dataclass
class WholeBodyQPConfig:
    pos_cost: float = 1.0
    ori_cost: float = 0.5
    posture_cost: float = 1e-2
    base_cost: float = 1.0     # damping cost of the base dofs (arm-vs-base split)
    torso_cost: float = 1.0
    lm_damping: float = 1e-6
    damping: float = 1e-4
    gain: float = 1.0
    iters: int = 1
    lead: float = 1.0
    resync: float = 0.3        # re-sync the open-loop configuration to the measurement beyond this (rad)
    closed_loop: bool = False  # True: configuration re-synced to the measured q every tick
    vmax: float = 0.5
    wmax: float = 0.6
    torso_vmax: float = 0.1
    solver: str = "daqp"


class WholeBodyQP:
    """mink QP on the identified model; the QP's world is the CURRENT base frame (base joints held at zero)."""

    def __init__(self, model, spec, cfg: WholeBodyQPConfig):
        import mink
        self.mink = mink
        self.mod, self.spec, self.c = model, spec, cfg
        m = model.m; c = cfg
        self.m = m
        self.cfg = mink.Configuration(m)
        self.av, self.bv, self.aq, self.bq = model.av, model.bv, model.aq, model.bq
        site = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SITE, model.site)
        self.frame = mink.FrameTask(site, "site", position_cost=c.pos_cost, orientation_cost=c.ori_cost,
                                    gain=c.gain, lm_damping=c.lm_damping)
        self.posture = mink.PostureTask(m, cost=c.posture_cost)
        dcost = np.zeros(m.nv)
        if model.n_base:
            dcost[self.bv[:3]] = c.base_cost
        if model.has_torso:
            dcost[self.bv[3]] = c.torso_cost
        self.damp = mink.DampingTask(m, dcost)
        robot_dofs = set(int(x) for x in np.r_[self.av, self.bv])
        frozen = [i for i in range(m.nv) if i not in robot_dofs]
        self.constraints = [mink.DofFreezingTask(m, dof_indices=frozen)] if frozen else []
        jn = lambda j: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        qd_max = np.asarray(spec.qd_max, float)
        vel = {jn(model.arm_j[i]): qd_max[i] for i in range(model.n)}
        if model.n_base:
            vel.update({jn(model.base_j[0]): c.vmax, jn(model.base_j[1]): c.vmax, jn(model.base_j[2]): c.wmax})
        if model.has_torso:
            vel.update({jn(model.base_j[3]): c.torso_vmax})
        lim = mink.ConfigurationLimit(m)
        keep = [int(i) for i in lim.indices if int(i) in robot_dofs]
        lim.indices = np.array(keep, int); lim.projection_matrix = np.eye(m.nv)[keep] if keep else None
        lim._ball_qposadr = lim._ball_qposadr[:0]; lim._ball_dofadr = lim._ball_dofadr[:0]
        lim._ball_max_angle = lim._ball_max_angle[:0]; lim._ball_G = np.zeros((0, m.nv)); lim._ball_h = np.empty(0)
        self.limits = [lim, mink.VelocityLimit(m, vel)]
        self.q_full = model.init_qpos.copy()
        if model.n_base:
            self.q_full[self.bq[:3]] = 0.0
        self.R0, self.p0 = model.R0, model.p0
        self.n_fail = 0

    def reset(self, q, h):
        self.q_ik = q.copy()
        self.q_full[self.aq] = q
        if self.mod.has_torso:
            self.q_full[self.bq[3]] = h
        self.posture.set_target(self.q_full.copy())

    @staticmethod
    def ref_at(t, P, Rs, s):
        """piecewise-linear position / geodesic rotation interpolation of the preview at time s (clamped)."""
        s = float(np.clip(s, t[0], t[-1]))
        i = int(np.clip(np.searchsorted(t, s) - 1, 0, len(t) - 2))
        w = (s - t[i]) / max(t[i + 1] - t[i], 1e-9)
        return (1 - w) * P[i] + w * P[i + 1], slerp_R(Rs[i], Rs[i + 1], w)

    def set_config(self, q, h):
        c = self.c
        if c.closed_loop or np.abs(self.q_ik - q).max() > c.resync:
            self.q_ik = q.copy()
        qf = self.q_full; qf[self.aq] = self.q_ik
        if self.mod.n_base:
            qf[self.bq[:3]] = 0.0
        if self.mod.has_torso:
            qf[self.bq[3]] = h
        self.cfg.update(qf)

    def target(self, inp, s):
        q, h = inp.q, inp.h
        p, R = self.mod.fk(q, h, jac=False)
        P = p + inp.ref_rel_p
        Rs = np.einsum("ij,njk->nik", R, inp.ref_rel_R)
        return self.ref_at(inp.ref_t, P, Rs, self.c.lead * s)

    def solve(self, inp, s, extra_tasks=(), extra_limits=None):
        """one tick: (arm qd, base-frame v (2), yaw rate, torso rate, arm configuration after s)."""
        mink = self.mink; c = self.c
        self.set_config(inp.q, inp.h)
        pt, Rt = self.target(inp, s)
        T = mink.SE3.from_rotation_and_translation(mink.SO3.from_matrix(self.R0 @ Rt), self.p0 + self.R0 @ pt)
        self.frame.set_target(T)
        tasks = [self.frame, self.posture, self.damp] + list(extra_tasks)
        lim_sets = extra_limits if extra_limits is not None else [[]]
        dv = None
        for extra in lim_sets:
            try:
                with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                    dv = mink.solve_ik(self.cfg, tasks, s, c.solver, damping=c.damping, limits=self.limits + list(extra),
                                       constraints=self.constraints)
                break
            except mink.exceptions.NoSolutionFound:        # infeasible QP: next fallback / hold
                self.n_fail += 1
        if dv is None:
            dv = np.zeros(self.m.nv)
        self.cfg.integrate_inplace(dv, s)
        qd = dv[self.av]
        q_next = self.q_ik + qd * s
        if self.mod.n_base:
            ax = self.mod.ax_w
            vb = self.R0.T @ (dv[self.bv[0]] * ax[0] + dv[self.bv[1]] * ax[1])
            w = float(dv[self.bv[2]])
        else:
            vb, w = np.zeros(3), 0.0
        vz = float(dv[self.bv[3]]) if self.mod.has_torso else 0.0
        return qd, vb[:2], w, vz, q_next


# ---------------------------------------------------------------------------------------------------- mink base plan
@dataclass
class MinkBaseConfig:
    qp: WholeBodyQPConfig = field(default_factory=lambda: WholeBodyQPConfig(closed_loop=False, posture_cost=1e-3, base_cost=0.3))
    base_vmax: float = 0.5
    base_wmax: float = 0.6
    base_tau: float = 0.0          # first-order low-pass of the base command (s); 0 = off
    base_acc: Optional[float] = None    # planar acceleration limit of the command (m/s^2)
    yaw_acc: Optional[float] = None     # yaw acceleration limit (rad/s^2)
    torso_vmax: float = 0.1


class MinkBasePlanner:
    def __init__(self, model, spec, cfg: MinkBaseConfig, dt):
        self.mod, self.c, self.dt = model, cfg, dt
        self.qp = WholeBodyQP(model, spec, cfg.qp)

    def reset(self, inp):
        self.qp.reset(inp.q, inp.h)
        self.q_mink = inp.q.copy()
        self.vb_f = np.zeros(2); self.w_f = 0.0

    def plan(self, inp, ref=None, q=None):
        c = self.c; dt = self.dt
        _, vb, w, vz, q_next = self.qp.solve(inp, dt)
        self.qp.q_ik = np.clip(q_next, self.mod.q_lo, self.mod.q_hi)
        self.q_mink = self.qp.q_ik.copy()
        vn = np.linalg.norm(vb)
        if vn > c.base_vmax:
            vb = vb * c.base_vmax / vn
        w = float(np.clip(w, -c.base_wmax, c.base_wmax))
        if c.base_tau > 0:
            a = min(1.0, dt / c.base_tau)
            vb = self.vb_f + a * (vb - self.vb_f); w = self.w_f + a * (w - self.w_f)
        if c.base_acc:
            dv = vb - self.vb_f; n = np.linalg.norm(dv)
            if n > c.base_acc * dt:
                vb = self.vb_f + dv * c.base_acc * dt / n
        if c.yaw_acc:
            w = self.w_f + float(np.clip(w - self.w_f, -c.yaw_acc * dt, c.yaw_acc * dt))
        self.vb_f, self.w_f = np.array(vb, float), float(w)
        return vb, w, float(np.clip(vz, -c.torso_vmax, c.torso_vmax))


class NoBase:
    """fixed-base embodiments: no base / torso motion."""
    vb_f = np.zeros(2); w_f = 0.0

    def reset(self, inp):
        pass

    def plan(self, inp, ref=None, q=None):
        return np.zeros(2), 0.0, 0.0
