"""NavTracker: perception-driven whole-body base layer (clean copy of flowsde/zoo/mink_nav.py MinkNav).

  perception (perception.py; a level "L0".."L5", a fusion "La+Lb", or "banded")  -> world-frame map
  base-path layer (every `replan` s): dynamic programming over the EE reference horizon (default 3 s in 0.25 s
      stages); per stage, candidate base poses on a polar lattice around the reference EE (distance `radii`, heading
      within +-bear_max of the reference yaw); node cost = reach preference + heading preference + base clearance
      (perceived map vs the base disc radius + plan_margin); edge cost = motion + speed limit + turning.
  whole-body QP (every tick): mink FrameTask (EE) + PostureTask + DampingTask + a base-target PostureTask (plan at
      now + tau) + SPHERE CLEARANCE velocity dampers for the robot's collision spheres within d_inf of perceived
      obstacles: n^T J v >= -xi (d - d_s) / (d_inf - d_s). Infeasible -> base spheres only -> no clearance.
Information (labelled tier "nav"): the controller observation + (i) the EE reference over the planner horizon,
(ii) the base pose in the world (localisation), (iii) the perception map. The arm is driven by the QP's integrated
configuration (position setpoint + velocity feed-forward), not by the predictive servo.

Embodiment-specific: the collision spheres (`spheres`), the base disc radius (`base_r`), the lattice radii (arm
reach), the QP joint-speed limits (RobotSpec.qd_max) and sensor mounts. Task-agnostic: everything else.
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from flowsde_ctrl.base import BaseActuatorMap, WholeBodyQP, WholeBodyQPConfig
from flowsde_ctrl.inputs import Command
from flowsde_ctrl import perception as PC
from flowsde_ctrl.math3d import Rz, wrap

DT_REF = 0.05


@dataclass
class NavConfig:
    qp: WholeBodyQPConfig = field(default_factory=lambda: WholeBodyQPConfig(closed_loop=True, posture_cost=1e-3,
                                                                            base_cost=0.3, ori_cost=1.0))
    level: str = "L0"           # perception level / fusion ("L2+L4" = depth + 2D lidar, the deployable pick)
    planner: bool = True
    clearance: bool = True
    horizon: float = 3.0        # planner horizon over the reference (s)
    stage: float = 0.25         # DP stage (s)
    replan: float = 0.2         # replanning period (s)
    tau: float = 0.3            # base-target lead (s)
    base_task_cost: float = 0.5
    plan_margin: float = 0.06   # extra clearance of the base disc in the planner (m); 0.15 for noisy / coarse maps
    base_r: float = 0.43        # base disc radius around the yaw axis (m)
    r_nom: float = 0.46         # preferred base(yaw axis)-to-EE distance (m)
    w_r: float = 20.0
    w_y: float = 1.0
    w_clr: float = 400.0
    w_move: float = 2.0
    w_vmax: float = 200.0
    w_turn: float = 0.5
    v_plan: float = 0.45        # planner speed (m/s)
    n_bear: int = 11
    bear_max: float = float(np.radians(75.0))
    radii: tuple = (0.38, 0.46, 0.54, 0.62, 0.70)
    d_inf: float = 0.25         # clearance damper influence distance (m)
    d_s_base: float = 0.04      # security distance, base spheres (m)
    d_s_arm: float = 0.03       # security distance, arm spheres (m)
    xi: float = 0.4             # damper gain (1/s)
    arm_spheres: bool = True
    spheres: tuple = ()         # ((geom name, [(dx, dy, dz) offsets], radius, kind 0 base / 1 arm), ...)
    perc: dict = field(default_factory=dict)


class _Clearance:
    def __init__(self, ctrl, which="all"):
        import mink
        self.ctrl, self.which, self.mink = ctrl, which, mink

    def compute_qp_inequalities(self, configuration, dt):
        G, h = self.ctrl._clearance_rows(configuration, dt, self.which)
        if G is None:
            return self.mink.Constraint()
        return self.mink.Constraint(G=G, h=h)


def _limit_cls():
    import mink

    class SphereClearance(_Clearance, mink.Limit):
        pass
    return SphereClearance


class NavTracker:
    """world: adapter object with base_state() -> (cx, cy, th) (yaw-axis point, world yaw), base_origin(cx, cy, th),
    base_z, yaw_anchor_b, carry_R (EE rotation in the base frame for the reference yaw), ref_pos / ref_rot (full
    reference, world, at DT_REF), time(), m / d (sim, for simulated sensors), grid0, exact_H(grid), seed."""

    def __init__(self, spec, model, cfg: NavConfig, world):
        self.spec, self.mod, self.c, self.w = spec, model, cfg, world
        self.dtp = spec.inner.physics_dt * spec.inner.n_sub
        self.qp = WholeBodyQP(model, spec, cfg.qp)
        import mink
        m = model.m
        v = np.zeros(m.nv); v[model.bv[:3]] = cfg.base_task_cost
        self.base_task = mink.PostureTask(m, cost=v, gain=1.0)
        self.base_map = BaseActuatorMap(spec.base)
        self._SC = _limit_cls()
        self._spheres_def()
        self.perc = None

    def _spheres_def(self):
        m = self.mod.m; S = []
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
        for name, offs, r, kind in self.c.spheres:
            if kind == 1 and not self.c.arm_spheres:
                continue
            g = gid(name)
            for off in offs:
                S.append((int(m.geom_bodyid[g]), m.geom_pos[g] + np.array(off, float), r, kind))
        self.spheres = S

    # ------------------------------------------------------------------ frames
    def _world_tf(self):
        w = self.w
        cx, cy, th = w.base_state()
        o = np.r_[w.base_origin(cx, cy, th), w.base_z]
        Rb = Rz(th)
        return o, Rb @ self.mod.R0.T, (cx, cy, th)

    def _clearance_rows(self, cfg, dt, which):
        if self.perc is None or not self.c.clearance or self.c.level == "L5":
            return None, None
        c = self.c; d = cfg.data; m = self.mod.m
        o, A, _ = self._world_tf()
        pts_m, idx = [], []
        for k, (b, off, r, kind) in enumerate(self.spheres):
            if which == "base" and kind != 0:
                continue
            pts_m.append(d.xpos[b] + d.xmat[b].reshape(3, 3) @ off); idx.append(k)
        if not pts_m:
            return None, None
        pts_m = np.array(pts_m)
        pts_w = o[None] + (pts_m - self.mod.p0[None]) @ A.T
        dist, nrm = self.perc.sphere_query(pts_w, rinf=c.d_inf + 0.25)
        G, h = [], []
        jp = np.zeros((3, m.nv))
        self.min_clear = min(self.min_clear, float(np.min([dist[i] - self.spheres[k][2] for i, k in enumerate(idx)])))
        for i, k in enumerate(idx):
            b, off, r, kind = self.spheres[k]
            dd = dist[i] - r
            if not np.isfinite(dd) or dd > c.d_inf:
                continue
            ds = c.d_s_base if kind == 0 else c.d_s_arm
            n_m = A.T @ nrm[i]
            mujoco.mj_jac(m, d, jp, None, pts_m[i], b)
            Jn = n_m @ jp
            G.append(-Jn); h.append(dt * c.xi * (dd - ds) / (c.d_inf - ds))
        if not G:
            return None, None
        return np.array(G), np.array(h)

    # ------------------------------------------------------------------ base-path layer
    def _ref_yaw(self, Rr):
        Rb = Rr @ self.w.carry_R.T
        return float(np.arctan2(Rb[1, 0], Rb[0, 0]))

    def _plan(self, t_now):
        w, c = self.w, self.c
        cx, cy, th = w.base_state()
        T = len(w.ref_pos)
        N = int(round(c.horizon / c.stage))
        ks = [min(int(round(t_now / DT_REF + (n + 1) * c.stage / DT_REF)), T - 1) for n in range(N)]
        anchor = -w.yaw_anchor_b[0]
        radii = np.array(c.radii); nb = c.n_bear
        need = c.base_r + c.plan_margin
        prev_c = np.array([[cx, cy]]); prev_th = np.array([th]); acc = np.zeros(1); back = []
        cands = []
        for n, k in enumerate(ks):
            E = w.ref_pos[k][:2]; psi = self._ref_yaw(w.ref_rot[k])
            th_c = psi + np.linspace(-c.bear_max, c.bear_max, nb)
            TH, RR = np.meshgrid(th_c, radii, indexing="ij"); TH, RR = TH.ravel(), RR.ravel()
            C = E[None] - (RR + anchor)[:, None] * np.stack([np.cos(TH), np.sin(TH)], 1)
            clr = self.perc.base_clearance(C) if self.perc is not None else np.full(len(C), 99.0)
            node = c.w_r * (RR - c.r_nom) ** 2 + c.w_y * wrap(TH - psi) ** 2 \
                + c.w_clr * np.maximum(need - clr, 0.0) ** 2 + 1e3 * (clr < c.base_r + 0.01)
            D = np.linalg.norm(C[None, :, :] - prev_c[:, None, :], axis=2)
            dth = np.abs(wrap(TH[None, :] - prev_th[:, None]))
            vm = c.v_plan * c.stage
            edge = c.w_move * D ** 2 + c.w_vmax * np.maximum(D - vm, 0.0) ** 2 + c.w_turn * np.maximum(dth - 0.6 * c.stage, 0) ** 2
            tot = acc[:, None] + edge
            arg = np.argmin(tot, 0)
            acc = tot[arg, np.arange(len(C))] + node
            back.append(arg); cands.append((C, TH)); prev_c, prev_th = C, TH
        j = int(np.argmin(acc)); path = []
        for n in range(N - 1, -1, -1):
            C, TH = cands[n]; path.append((C[j], TH[j])); j = back[n][j]
        path = path[::-1]
        ts = t_now + np.arange(N + 1) * c.stage
        P = np.array([[cx, cy]] + [p[0] for p in path])
        Th = np.unwrap(np.array([th] + [th + wrap(p[1] - th) for p in path]))
        self.plan = (ts, P, Th)
        self.plan_cost = float(acc.min())

    def _base_target(self, t):
        ts, P, Th = self.plan
        s = t + self.c.tau
        return np.array([np.interp(s, ts, P[:, 0]), np.interp(s, ts, P[:, 1])]), float(np.interp(s, ts, Th))

    # ------------------------------------------------------------------ main
    def reset(self, inp):
        self.base_map.reset()
        self.qp.reset(inp.q, inp.h)
        self.perc = PC.make(self.w, self.c.level, seed=self.w.seed, **(self.c.perc or {}))
        self.plan = None; self.t_plan = -1e9; self.qp.n_fail = 0; self.min_clear = np.inf

    @property
    def n_fallback(self):
        return self.qp.n_fail

    def step(self, inp):
        c, w, mod = self.c, self.w, self.mod
        if inp.episode_start:
            self.reset(inp)
        t = w.time()
        self.perc.update(t)
        if c.planner and (t - self.t_plan >= c.replan - 1e-9 or self.plan is None):
            self._plan(t); self.t_plan = t
        extra = []
        if c.planner and self.plan is not None:
            cx, cy, th = w.base_state()
            cd, thd = self._base_target(t)
            Rb = Rz(th)
            dcb = Rb.T @ np.r_[cd - np.array([cx, cy]), 0.0]
            dm = mod.R0 @ dcb
            ax = mod.ax_w[:, :2]
            qs = np.linalg.solve(ax.T, dm[:2])
            tgt = self.qp.q_full.copy(); tgt[mod.aq] = inp.q; tgt[mod.bq[3]] = inp.h
            tgt[mod.bq[0]] = qs[0]; tgt[mod.bq[1]] = qs[1]; tgt[mod.bq[2]] = wrap(thd - th)
            self.base_task.set_target(tgt)
            self.base_task.gain = min(1.0, self.dtp / c.tau)
            extra = [self.base_task]
        SC = self._SC
        qd, vb, wz, vz, q_next = self.qp.solve(inp, self.dtp, extra,
                                               extra_limits=[[SC(self, "all")], [SC(self, "base")], []])
        self.qp.q_ik = np.clip(q_next, mod.q_lo, mod.q_hi)
        b = self.spec.base
        vn = np.linalg.norm(vb)
        if vn > b.vmax:
            vb = vb * b.vmax / vn
        wz = float(np.clip(wz, -b.wmax, b.wmax))
        base_cmd = self.base_map(inp, vb, wz)
        return Command(q_sp=self.qp.q_ik, qd_ff=qd, base_cmd=base_cmd, vz=vz,
                       info=dict(min_clear=self.min_clear, n_fallback=self.qp.n_fail))
