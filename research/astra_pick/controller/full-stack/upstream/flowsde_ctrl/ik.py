"""Closed-loop differential IK on the identified kinematics (task-agnostic).

Clean copy of flowsde/classical.py ClassicalTracker._ik. The IK state is INTEGRATED (q_ref is carried from tick to tick,
not reset to the measured q; reset only when it drifts > `resync` from the measurement, e.g. in a hard contact).
"""
from dataclasses import dataclass

import numpy as np

from flowsde_ctrl.math3d import rotvec


@dataclass
class IKConfig:
    w_rot: float = 0.3        # task weight of orientation vs position (m per rad)
    lam: float = 0.005        # DLS damping (constant part)
    lam_s: float = 0.05       # extra damping near singularities (smallest singular value < sv0)
    sv0: float = 0.05         # singular-value threshold for the extra damping
    ik_iters: int = 2         # Gauss-Newton iterations per target
    k_post: float = 1.0       # null-space pull to the posture (episode's initial q), 1/s
    k_lim: float = 10.0       # null-space joint-limit repulsion gain, 1/s
    k_wlim: float = 1.0       # joint-limit weighting strength (weights grow inside lim_margin)
    lim_margin: float = 0.3   # rad from a limit where weighting / repulsion start
    qd_max: float = 2.5       # joint speed cap of the IK step (rad/s, uniform)
    resync: float = 0.3       # re-sync the integrated reference to the measured q beyond this (rad)


class DLSIK:
    def __init__(self, model, cfg: IKConfig, dt):
        self.mod, self.c, self.dt = model, cfg, dt
        self.q_lo, self.q_hi = model.q_lo, model.q_hi
        self.q_post = None

    def solve(self, q0, h, pt, Rt):
        """Joint-limit-weighted, singularity-robust DLS with null-space posture + limit repulsion and speed cap."""
        c = self.c; q = q0.copy(); n = len(q)
        for _ in range(c.ik_iters):
            p, R, J = self.mod.fk(q, h)
            e = np.concatenate([pt - p, c.w_rot * (R @ rotvec(R.T @ Rt))])
            J = J * np.r_[1, 1, 1, [c.w_rot] * 3][:, None]
            lo_d, hi_d = q - self.q_lo, self.q_hi - q
            dist = np.minimum(lo_d, hi_d)
            w = 1.0 + c.k_wlim * np.maximum(c.lim_margin / np.maximum(dist, 1e-3) - 1.0, 0.0) ** 2
            JW = J / w
            sv = np.linalg.svd(J, compute_uv=False)[-1]
            lam2 = c.lam ** 2 + c.lam_s ** 2 * max(0.0, 1 - (sv / c.sv0) ** 2)
            Jp = JW.T @ np.linalg.inv(JW @ J.T + lam2 * np.eye(6))
            dq = Jp @ e
            lim = c.k_lim * (np.maximum(c.lim_margin - lo_d, 0) - np.maximum(c.lim_margin - hi_d, 0))
            dq += (np.eye(n) - Jp @ J) @ ((c.k_post * (self.q_post - q) + lim) * self.dt)
            mx = np.abs(dq).max() / (c.qd_max * self.dt)
            if mx > 1:
                dq /= mx
            q = np.clip(q + dq, self.q_lo + 1e-3, self.q_hi - 1e-3)
        return q
