"""Proprioceptive latency compensation (task-agnostic): Smith-predictor-style forward prediction of the delayed
measurement with the identified model of the inner PD loop. Clean copy of flowsde/classical_delay.py DelayComp.

The measurement (q, qd, torso, base velocity, and the EE / base pose the reference preview is expressed in) is
`ms` old (nominal; jitter unknown). Per tick:
  1. x_m = (q, qd) at t - tau;  2. roll the nominal inner-loop model forward over k = tau / physics_dt steps with the
  LAST command (absolute setpoint, feed-forward, DOB torque, friction feed-forward);  3. base / torso: constant
  velocity over tau; the reference preview is re-expressed in the predicted base frame relative to the predicted EE;
  4. the wrapped controller runs on the predicted input;  5. DOB bookkeeping at the MEASUREMENT instant: the measured
  state is compared with the one-interval-ahead prediction made a tick earlier for that instant, so the observer sees
  model error, not the delay;  6. the setpoint is re-anchored to the q the plant adds the offset to.
Requires tau < one controller tick. Nothing is tuned: the only parameter is the nominal latency (measure it!).
"""
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from flowsde_ctrl.math3d import Rz, rot6d, from_rot6d


@dataclass
class LatencyConfig:
    ms: Optional[float] = None     # nominal latency (ms); None -> RobotSpec.latency.ms
    anchor: Optional[str] = None   # "obs" | "true"; None -> RobotSpec.latency.anchor
    canonical: bool = True         # pass predicted q / relative rotations through the observation encoding (sin/cos
                                   # angles, rot6d) as the reference implementation does (bit-exact regression)


class LatencyCompensator:
    def __init__(self, tracker, cfg: LatencyConfig):
        self.t, self.c = tracker, cfg

    def init(self):
        tr = self.t; spec = tr.spec
        ms = spec.latency.ms if self.c.ms is None else self.c.ms
        self.k = int(round(float(ms) * 1e-3 / tr.h_phys))
        assert self.k < tr.n_sub, "latency compensation assumes nominal latency < one controller tick"
        self.anchor = self.c.anchor or (spec.latency.anchor if ms else "true")
        self.last = None; self.xhat = None; self.prev_h = None

    def pre(self, inp):
        tr = self.t; mod = tr.mod
        if inp.episode_start:
            self.init()
        k = self.k
        if k == 0:
            return inp, None
        q_m, qd_m, h_m = inp.q, inp.qd, inp.h
        x_m = np.concatenate([q_m, qd_m])
        tau = k * tr.h_phys
        hd = 0.0 if self.prev_h is None else (h_m - self.prev_h) / tr.dtp
        self.prev_h = h_m
        x_p = x_m.copy() if self.last is None else tr.servo.rollout(x_m, self.last[1], self.last[0], k)
        n = len(q_m)
        q_p, qd_p = x_p[:n], x_p[n:]
        h_p = h_m + hd * tau
        dl = tr.dl
        p_m, R_m = mod.fk(q_m - dl * qd_m, h_m, jac=False)
        P = p_m + inp.ref_rel_p; Rs = np.einsum("ij,njk->nik", R_m, inp.ref_rel_R)
        bvj = inp.base_vel; ax = inp.base_axes
        vlin, wz = bvj[0] * ax[0] + bvj[1] * ax[1], bvj[2]
        a_ = mod.anchor; Rinv = Rz(-wz * tau)
        P = a_ + (P - a_ - vlin * tau) @ Rinv.T
        Rs = np.einsum("ij,njk->nik", Rinv, Rs)
        p_p, R_p = mod.fk(q_p - dl * qd_p, h_p, jac=False)
        rel_p = P - p_p
        rel_R = np.einsum("ji,njk->nik", R_p, Rs)
        q_in, qd_in = q_p, qd_p
        if self.c.canonical:
            six = np.stack([rot6d(Ri) for Ri in rel_R])
            rel_R = from_rot6d(six[:, 0:3], six[:, 3:6])
            lo, hi = mod.q_lo, mod.q_hi
            qn = 2 * (q_p - lo) / (hi - lo) - 1
            q_lin = lo + 0.5 * (qn + 1) * (hi - lo)
            q_in = np.arctan2(np.sin(q_p), np.cos(q_p)); q_in = q_in + 2 * np.pi * np.round((q_lin - q_in) / (2 * np.pi))
            qd_in = 2.0 * (qd_p / 2.0)
        if self.xhat is not None and tr.servo.pred is not None:
            tr.servo.pred = x_p - (x_m - self.xhat)
        ctx = dict(x_p=x_p, q_p=q_p, q_m=q_m, q_in=q_in)
        return replace(inp, q=q_in, qd=qd_in, h=h_p, ref_rel_p=rel_p, ref_rel_R=rel_R), ctx

    def post(self, cmd, ctx):
        if ctx is None:
            return cmd
        tr = self.t; sv = tr.servo; k = self.k
        self.xhat = sv.rollout(ctx["x_p"], sv.last_M, sv.last_u, tr.n_sub - k)
        self.last = (sv.last_u, sv.last_M)
        s = tr.spec.inner.sp_scale
        a = np.clip((cmd.q_sp - ctx["q_in"]) / s, -1, 1)               # the offset the controller asks for
        if self.anchor == "obs":                                       # plant adds it to the OBSERVED q
            cmd.q_sp = ctx["q_p"] + s * a
            cmd.info["q_anchor"] = ctx["q_m"]
        else:                                                          # plant adds it to the TRUE q (~ predicted)
            cmd.q_sp = ctx["q_in"] + s * a
            cmd.info["q_anchor"] = ctx["q_in"]
        return cmd
