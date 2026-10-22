"""Contact-force sources (task-agnostic): proprioceptive disturbance observers and the wrist F/T reading.

  WristFT         HW upgrade: external force on the tool from a wrist F/T stream (flowsde/classical_ft.py _force_ft).
  SimpleDobForce  fair tier, first-generation observer: unclamped DOB estimate minus a slow free-space EMA baseline,
                  mapped to an EE force (flowsde/classical_ft.py ft_src="dob"); used by combo_dob_bp.
  ForceObserver   fair tier, the frozen cc_obs_v5 observer (flowsde/constraint_comp.py, obs="ema" path): raw
                  per-interval disturbance (not the filtered DOB state), base-coupling compensated, early (start-of-tick)
                  update, free-space EMA baseline frozen in contact, joint-limit and actuator-saturation masks,
                  wrist-down-weighted force-only map.
  BaseCoupling    inertial torque on the arm from the measured base acceleration, M_arm,base(q) a_base
                  (flowsde/combo.py ComboTracker._coupling). The arm model is fixed-base: without this, every base
                  acceleration reads as an EE contact (50-120 N false contacts on DROID; TUNING_GUIDE §4).

All forces are "force ON the robot BY the environment", base frame (they point out of a surface).
Pitfalls these components encode (TUNING_GUIDE §4): the residual of a saturated actuator is the clipped deficit, not a
contact; a joint at a mechanical limit carries an unmodelled limit torque; the commanded (not measured) torque is what
the observer sees; a load present from initialisation is absorbed into the baseline during warm-up.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------------------------------- base coupling
class BaseCoupling:
    def __init__(self, model):
        self.mod = model
        self.reset()

    def reset(self):
        n = self.mod.n
        self.bvj_prev = None; self.cp = np.zeros(n); self.cp_f = np.zeros(n)

    def update(self, inp, dt):
        """cp = M_arm,base(q) a_base, a_base differenced from the measured base joint velocities (nominal model)."""
        mod = self.mod
        bvj = inp.base_vel
        if self.bvj_prev is None:
            self.bvj_prev = bvj
        acc = (bvj - self.bvj_prev) / dt; self.bvj_prev = bvj
        ax = inp.base_axes
        a_b = ax[0] * acc[0] + ax[1] * acc[1]                       # base-frame linear acceleration
        A = (mod.R0.T @ mod.ax_w.T)[:2]                               # model slide axes in the model base frame
        s = np.linalg.lstsq(A, a_b[:2], rcond=None)[0]
        mod.fk(inp.q, inp.h, jac=False, inertia=True)                # full nominal M at the current posture
        Mab = mod.M[np.ix_(mod.av, mod.bv[:3])]
        self.cp = Mab @ np.array([s[0], s[1], acc[2]])
        return self.cp

    def filt(self, gain):
        """pass the coupling through the observer's own first-order update (gain = DOB gain)."""
        self.cp_f = self.cp_f + gain * (self.cp - self.cp_f)
        return self.cp_f


# ---------------------------------------------------------------------------------------------------- wrist F/T
@dataclass
class WristFTConfig:
    oracle: bool = False      # labelled upper bound: the adapter passes the true contact wrench instead


class WristFT:
    """External force on the tool from the F/T samples since the last tick, rotated to the base frame with the
    nominal kinematics. Returns (latest force, peak |force| over the samples) -- the reaction law triggers on the peak
    of the 500 Hz stream."""

    def __init__(self, model, R_site_sensor):
        self.mod = model; self.R_site_sensor = R_site_sensor

    def read(self, samples, R_ee):
        if samples is None:
            return np.zeros(3), 0.0
        Rb = R_ee @ self.R_site_sensor
        Fs = np.array([Rb @ b[:3] for b in samples])
        return Fs[-1], float(np.linalg.norm(Fs, axis=1).max())


# ---------------------------------------------------------------------------------------------------- simple DOB force
@dataclass
class SimpleDobConfig:
    me_tau: float = 1.0       # free-space baseline time constant (s); frozen while in contact
    F_lp: float = 0.02        # force low-pass time constant (s)
    dob_eps: float = 1e-3     # damping of the force map (J J^T + eps)^-1
    warm: float = 0.5         # warm-up (s): the baseline converges onto model error, no contact action
    base_comp: bool = False   # add the base-coupling torque (mobile bases with a whole-body planner)


class SimpleDobForce:
    def __init__(self, cfg: SimpleDobConfig, dt):
        self.c, self.dt = cfg, dt

    def reset(self, n):
        self.tau_hat = np.zeros(n); self.tau_me = np.zeros(n); self.F = np.zeros(3)

    def after_dob(self, res, dob_max, gain, coupling=None):
        """servo DOB update result (td0, upd_raw) -> unclamped estimate (+ filtered base coupling)."""
        if res is None:
            return
        th0 = self.tau_hat.copy()
        td0, upd = res
        self.tau_hat = np.clip(td0 + upd, -dob_max, dob_max)
        if self.c.base_comp and coupling is not None and not np.array_equal(th0, self.tau_hat):
            self.tau_hat = self.tau_hat + coupling.filt(gain)

    def rebaseline(self):
        """watchdog: a 'contact' that persists is re-classified as model error."""
        self.tau_me = self.tau_hat.copy(); self.F[:] = 0.0

    def read(self, J, t_elapsed, in_contact):
        c = self.c
        if t_elapsed < c.warm:
            self.tau_me = self.tau_hat.copy()
        elif not in_contact:
            self.tau_me += (self.tau_hat - self.tau_me) * min(1.0, self.dt / c.me_tau)
        tau = self.tau_hat - self.tau_me + 0.0
        F = np.linalg.solve(J @ J.T + c.dob_eps * np.eye(6), J @ tau)[:3]
        self.F += (F - self.F) * min(1.0, self.dt / c.F_lp)
        return self.F.copy(), float(np.linalg.norm(self.F))


# ---------------------------------------------------------------------------------------------------- cc_obs_v5 observer
@dataclass
class ObserverConfig:
    me_tau: float = 1.0       # free-space baseline (EMA) time constant (s)
    F_lp: float = 0.02        # force estimate low-pass (s)
    dob_eps: float = 1e-3     # force-map damping
    warm: float = 0.5         # warm-up (s): baseline := raw, force := 0
    gate_F: float = 12.0      # baseline adapts only while |F_hat| < gate_F and no contact is latched (gate latch+thr)
    w_wrist: float = 0.2      # weight of the wrist joints in the force map (short levers, most noise)
    lim_eps: float = 0.03     # joint-limit mask: weight 0 within lim_eps of a limit, ramps to 1 at 2 lim_eps (rad)
    sat_frac: Optional[float] = 0.9   # saturation mask: joints whose predicted torque > sat_frac * tau_max are excluded
    base_comp: bool = True    # base-coupling compensation (mobile bases)
    early: bool = True        # update at the start of the tick (one tick fresher force estimate)


class ForceObserver:
    def __init__(self, model, cfg: ObserverConfig, gain, dob_max, dt, tau_max, kp, kd):
        self.mod, self.c, self.g, self.dob_max, self.dt = model, cfg, gain, dob_max, dt
        self.TAU = np.asarray(tau_max, float); self.KP = np.asarray(kp, float); self.KD = np.asarray(kd, float)

    def reset(self, n):
        self.tau_hat = np.zeros(n); self.tau_me = np.zeros(n); self.bias = np.zeros(n)
        self.F_hat = np.zeros(3); self.F_raw = np.zeros(3); self._res_f = np.zeros(n); self._res = np.zeros(n)
        self._qd_prev = None; self.a_ee = np.zeros(3)
        self.sat_next = np.zeros(n, bool); self.sat_cur = np.zeros(n, bool)

    def pre_tick(self):
        self.sat_cur = self.sat_next.copy()

    def update(self, res, coupling, q, qd, J7, k, in_contact):
        """res = servo.dob_update(...) output (td0, upd_raw). Builds the raw per-interval disturbance and observes."""
        if res is None:
            return
        td0, upd = res
        self.tau_hat = np.clip(td0 + upd, -self.dob_max, self.dob_max)
        tau_ft = self.tau_hat.copy()
        if self.c.base_comp:
            self.tau_hat = self.tau_hat + coupling.filt(self.g)
        raw = td0 + (tau_ft - td0) / self.g + (coupling.cp if self.c.base_comp else 0.0)
        self._observe(raw, q, qd, J7, k, in_contact)

    def _observe(self, raw, q, qd, J, k, inc):
        c = self.c; dt = self.dt
        qdd = (qd - self._qd_prev) / dt if self._qd_prev is not None else np.zeros(len(q))
        self.a_ee = J[:3] @ qdd
        self._qd_prev = qd.copy()
        warm = k * dt < c.warm
        upd = (float(np.linalg.norm(self.F_hat)) < c.gate_F and not inc) or warm
        if warm:
            self.tau_me = raw.copy()
        e = self.tau_me
        res = raw - e - self.bias
        if not warm and upd:
            self.tau_me += (raw - self.tau_me) * min(1.0, dt / c.me_tau)
            res = raw - self.tau_me
        self._res = res
        dlim = np.minimum(q - self.mod.q_lo, self.mod.q_hi - q)
        wl = np.clip((dlim - c.lim_eps) / c.lim_eps, 0.0, 1.0)
        if c.sat_frac is not None:
            wl = wl * (~self.sat_cur)
        n = len(q)
        Jp = J[:3]; w = np.array([1.0] * 4 + [c.w_wrist] * (n - 4)) * wl
        F = np.linalg.solve((Jp * w) @ Jp.T + c.dob_eps * np.eye(3), (Jp * w) @ res)
        a = min(1.0, dt / c.F_lp)
        self.F_hat += (F - self.F_hat) * a
        self.F_raw = F.copy()
        self._res_f = self._res_f + (res - self._res_f) * a
        if warm:
            self.F_hat[:] = 0.0; self.F_raw[:] = 0.0

    def absorb(self):
        """not-a-constraint veto: the persisting residual is model error / a constant load -> into the baseline."""
        self.tau_me = self.tau_me + self._res_f
        self.F_hat[:] = 0.0

    def predict_saturation(self, q, qd, h, q_sp, qd_c):
        """joints whose commanded torque in the coming interval will hit the actuator limit: their DOB residual is
        the clipped deficit, not an external torque."""
        if self.c.sat_frac is None:
            return
        grav = self.mod.gravity(q, h)
        tau0 = self.KP * (q_sp - q) + self.KD * (qd_c - qd) + grav
        self.sat_next = np.abs(tau0) > self.c.sat_frac * self.TAU
