"""Predictive joint servo of the inner PD loop + disturbance observer (DOB). Task-agnostic; embodiment enters only
through the identified model (inertia incl. armature, damping, friction) and the inner-loop spec (Kp, Kd, rates).

Clean copy of flowsde/classical.py ClassicalTracker._mpc (servo="mpc") and of the DOB update variants in
flowsde/classical_ft.py FTTracker._mpc (unclamped estimate + contact-normal clamp) and flowsde/constraint_comp.py
CCTracker._pre_observe (early update).

Model. Within one controller tick the inner loop  tau = Kp (q_sp - q) + Kd (qd_ff - qd) + bias  runs n_sub times with
constant (q_sp, qd_ff). With the bias cancelling gravity / Coriolis and the identified joint-space inertia M(q), viscous
damping D and Coulomb friction F tanh(qd / 0.02), the semi-implicit Euler update is LINEAR in the commands, so the
joint state after the tick is predicted exactly:  x+ = F^n x0 + S G (Kp q_sp + Kd qd_ff + tau_d - fric).
The setpoint is placed mid-interval and the velocity feed-forward (whose gain Kd is not affected by a stiffness error)
is solved in weighted least squares so the predicted state lands on (q_ref(t + dt), qd_ref). Overflow of the
feed-forward range goes into the position channel.

DOB. tau_d is the lumped residual torque (model error, held loads, contact, base coupling) learned from the previous
tick's prediction error ex = x_meas - x_pred:  tau_d += gain * lstsq(S G, ex), |tau_d| <= dob_max; skipped when the
position error exceeds `dob_gate` (resync / impact). In contact, the update component along the contact normal can be
removed ("clamp") so the integrator does not wind up against a constraint.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ServoConfig:
    mpc_wv: float = 0.01      # weight of the velocity target vs position target in the feed-forward LS
    mpc_reg: float = 1e-6     # relative Tikhonov regulariser of the LS
    dob_gain: float = 0.5     # DOB update gain per tick (0..1); 0 disables the observer
    dob_max: float = 40.0     # |tau_d| bound per joint (N m)
    dob_gate: float = 0.05    # skip the update if max |q error| of the prediction exceeds this (rad)
    fric_v: float = 0.02      # velocity scale of the tanh Coulomb friction model (rad/s)


class PredictiveServo:
    def __init__(self, model, inner, cfg: ServoConfig, kp=None, kd=None):
        self.mod, self.inner, self.c = model, inner, cfg
        self.KP = np.array(inner.kp if kp is None else kp, float)
        self.KD = np.array(inner.kd if kd is None else kd, float)
        self.h = inner.physics_dt; self.n_sub = inner.n_sub
        self.D_nom = model.D_nom.copy(); self.F_nom = model.F_nom.copy()
        self.reset()

    def reset(self):
        n = len(self.KP)
        self.tau_d = np.zeros(n)
        self.pred = None          # predicted (q, qd) at the next tick (None until the first command)
        self.SG = None
        self.last_u = None        # input of the nominal inner-loop model for the last command (latency predictor)
        self.last_M = None

    # ------------------------------------------------------------------ disturbance observer
    def dob_update(self, q, qd, Jn=None):
        """Consume the last prediction. Returns (tau_d before, unclamped update) or None if no update happened.
        Jn: joint-space direction of a force along the contact normal; its component is removed from the APPLIED
        update (clamp)."""
        if self.pred is None:
            return None
        c = self.c
        ex = np.concatenate([q, qd]) - self.pred
        self.pred = None
        if np.abs(ex[:len(q)]).max() >= c.dob_gate:
            return None
        td0 = self.tau_d.copy()
        upd = c.dob_gain * np.linalg.lstsq(self.SG, ex, rcond=None)[0]
        upd_raw = upd
        if Jn is not None:
            nn = Jn @ Jn
            if nn > 1e-9:
                upd = upd - (upd @ Jn) / nn * Jn
        self.tau_d = np.clip(self.tau_d + upd, -c.dob_max, c.dob_max)
        return td0, upd_raw

    # ------------------------------------------------------------------ servo
    def command(self, q, qd, M, q_t, qd_t, kp_s=1.0, D=None):
        """(q_sp, qd_ff) for the next tick so that the predicted state lands on (q_t, qd_t)."""
        c = self.c; h = self.h; n = len(q); In = np.eye(n)
        D = self.D_nom if D is None else D
        K = kp_s * self.KP; Kd = self.KD + D
        Minv = np.linalg.inv(M + h * np.diag(D))
        F = np.block([[In - h * h * Minv * K, h * In - h * h * Minv * Kd], [-h * Minv * K, In - h * Minv * Kd]])
        Gc = np.vstack([h * h * Minv, h * Minv])
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

    # ------------------------------------------------------------------ nominal inner-loop rollout (latency)
    def model_FG(self, M):
        h = self.h; n = len(self.KP); In = np.eye(n)
        K = self.KP * 1.0; Kd = self.KD + self.D_nom
        Minv = np.linalg.inv(M + h * np.diag(self.D_nom))
        F = np.block([[In - h * h * Minv * K, h * In - h * h * Minv * Kd], [-h * Minv * K, In - h * Minv * Kd]])
        G = np.vstack([h * h * Minv, h * Minv])
        return F, G

    def rollout(self, x, M, u, k):
        if k <= 0:
            return x.copy()
        F, G = self.model_FG(M)
        for _ in range(k):
            x = F @ x + G @ u
        return x
