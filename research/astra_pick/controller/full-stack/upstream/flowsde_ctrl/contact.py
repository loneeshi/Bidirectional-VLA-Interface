"""Contact laws (task-agnostic): turn a force estimate into an EE reference OFFSET o (base frame). The stiff servo then
tracks reference + o; in free space o returns to zero (precision unchanged). No mechanism model, no task identity, no
approach-speed limit. Force convention: F is the force ON the robot BY the environment.

  ReactLaw  flowsde/classical_ft.py FTTracker._contact (modes sat / react / dir). Used with the wrist F/T (ft_react,
            combo_ft_bp) and with the simple DOB (combo_dob_bp).
  GCCLaw    flowsde/constraint_comp.py CCTracker._gcc: generic constraint-direction compliance (cc_obs_v5).

Why an offset admittance on a stiff servo (impedance with explicit force saturation) and not a soft impedance or a
pure admittance: soft impedance reads model error as contact and yields (QDD mismatch: 75 % success, 12 mm error);
admittance against stiff contacts chatters when its loop gain k_eff dt / D approaches 1 (D 300 -> 20-240 N
oscillation; D 500-1000 stable); energy tanks do not cap a STATIC push (no motion, the tank never drains), so an
explicit force cap is needed. See TUNING_GUIDE §5.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from flowsde_ctrl.math3d import unit


# ---------------------------------------------------------------------------------------------------- react / sat / dir
@dataclass
class ReactConfig:
    mode: str = "react"       # "sat": force saturation only; "react": + stop pushing on contact; "dir": + give/resist
    f_ref: float = 15.0       # force cap along the measured force direction (N)
    D_adm: float = 400.0      # offset admittance damping (N s/m): o' = (|F| - f_ref) / D_adm
    v_adm: float = 0.25       # offset rate bound (m/s)
    o_max: float = 0.04       # |o| bound (m)
    D_ret: Optional[float] = None   # optional separate return damping out of contact
    f_on: float = 10.0        # contact trigger on the PEAK of the sensor stream (N)
    f_off: float = 4.0        # release threshold (N)
    react_ff: bool = True     # cancel the reference motion INTO the surface while in contact
    dob_clamp: bool = True    # project the DOB update off the contact normal while in contact
    block_gate: bool = True   # cancel only the part of the reference motion the hand does NOT achieve
    f_move: float = 25.0      # dir: push budget while the contact gives
    f_hold: float = 6.0       # dir: light-contact force after a stall
    v_move: float = 0.01      # dir: "gives" speed threshold (m/s)
    t_hold: float = 0.25      # dir: stall time before yielding (s)
    t_reprobe: float = 1.0    # dir: re-probe after (s)
    f_burst: float = 45.0     # dir: transient allowance (N) ...
    t_burst: float = 0.04     # ... for at most (s)
    inc_tmax: Optional[float] = None   # observer watchdog (dob source only): persistent 'contact' -> model error
    warm: float = 0.0         # no contact action before this time (s)


class ReactLaw:
    def __init__(self, cfg: ReactConfig, dt, rebaseline=None):
        self.c, self.dt, self.rebaseline = cfg, dt, rebaseline

    def reset(self):
        self.o = np.zeros(3); self.n = np.zeros(3); self.inc = False; self.t_c = 0.0; self.t_stall = 0.0
        self.t_burst_used = 0.0; self.yield_ = False; self.t_yield = 0.0; self.t_inc = 0.0

    def step(self, F, Fpk, v_ref_base, v_ee):
        c = self.c; dt = self.dt
        Fn_ = np.linalg.norm(F)
        if not self.inc and Fpk > c.f_on and c.mode in ("react", "dir"):
            self.inc = True; self.t_c = 0.0; self.t_stall = 0.0; self.t_burst_used = 0.0; self.yield_ = False
        elif self.inc and Fn_ < c.f_off and Fpk < c.f_on:
            self.inc = False
        if self.inc:
            self.t_inc = self.t_inc + dt
            if c.inc_tmax and self.t_inc > c.inc_tmax and self.rebaseline is not None:
                self.rebaseline(); self.inc = False; Fn_ = 0.0; F = np.zeros(3)
        else:
            self.t_inc = 0.0
        if Fn_ > 1.0:
            self.n = F / Fn_
        n = self.n
        f_ref = c.f_ref
        if c.mode == "dir" and self.inc:
            self.t_c += dt
            give = (-v_ee @ n) > c.v_move
            self.t_stall = 0.0 if give else self.t_stall + dt
            if self.yield_:
                self.t_yield += dt
                if give or self.t_yield > c.t_reprobe:
                    self.yield_ = False; self.t_stall = 0.0
            elif self.t_stall > c.t_hold:
                self.yield_ = True; self.t_yield = 0.0
            if self.yield_:
                f_ref = c.f_hold
            elif give:
                f_ref = c.f_move
            if self.t_burst_used < c.t_burst and not self.yield_:
                f_ref = max(f_ref, c.f_burst); self.t_burst_used += dt
        # (a) explicit force saturation along n: offset admittance
        rate = (Fn_ - f_ref) / c.D_adm
        rate = float(np.clip(rate, -c.v_adm, c.v_adm))
        if Fn_ > f_ref:
            self.o = self.o + rate * dt * n
        else:
            on_ = np.linalg.norm(self.o)
            if on_ > 0:
                if not self.inc and c.D_ret:
                    rate = -min(c.v_adm, (c.f_ref - Fn_) / c.D_ret)
                step = min(on_, -rate * dt)
                self.o = self.o - step * self.o / on_
        # (b) stop actively pushing: cancel the reference's motion into the surface while in contact
        if self.inc and c.react_ff and c.mode in ("react", "dir"):
            vin = -(v_ref_base @ n)
            if c.block_gate:
                vin -= max(0.0, -(v_ee @ n))
            if vin > 0:
                self.o = self.o + vin * dt * n
        on_ = np.linalg.norm(self.o)
        if on_ > c.o_max:
            self.o *= c.o_max / on_
        return self.o

    def clamp_active(self):
        return self.inc and self.c.mode in ("react", "dir") and self.c.dob_clamp


# ---------------------------------------------------------------------------------------------------- generic constraint compliance
@dataclass
class GCCConfig:
    # contact latch (on the observer's force estimate)
    f_on: float = 15.0        # latch threshold (N), raised by the motion-scaled bound below
    t_on: float = 0.03        # persistence above the threshold before latching (s)
    f_off: float = 6.0        # release threshold (N) ...
    t_off: float = 0.3        # ... held for (s)
    fm_vb: float = 8.0        # motion-scaled detection bound: + fm_vb * min(1, base speed / 0.05) (N)
    fm_ab: float = 8.0        #   + fm_ab * base acceleration (N per m/s^2)
    fm_ae: float = 4.0        #   + fm_ae * EE acceleration (N per m/s^2), all capped at fm_max
    fm_max: float = 25.0
    f_fast: Optional[float] = 40.0   # immediate latch on the UNFILTERED force above this (+ the motion bound)
    f_snap: Optional[float] = 35.0   # overload reflex: above this, move the target to the measured EE along F
    snap_raw: bool = True
    snap_max: float = 0.01    # reflex step bound per tick (m)
    # constraint law
    f_perp: float = 5.0       # force target orthogonal to the progress direction (constraint force) (N)
    D_perp: float = 500.0     # admittance damping orthogonal (N s/m)
    f_push: float = 18.0      # push budget along the progress direction (N)
    D_par: float = 500.0      # admittance damping along progress (N s/m)
    f_hold: float = 8.0       # light contact after giving up (N)
    v_adm: float = 0.1        # offset rate bound (m/s)
    t_win: float = 0.15       # window of the realized EE displacement defining the progress direction (s)
    d_win: float = 0.004      # minimum realized displacement (m)
    v_refmin: float = 0.01    # minimum reference speed to use the reference direction (m/s)
    v_move: float = 0.01      # progress speed below which the push counts as stalled (m/s)
    t_probe: float = 1.5      # stall time before giving up the push (s)
    t_rest: float = 1.0       # rest at f_hold before re-probing (s)
    cancel: bool = True       # blocked-reference cancel (no wind-up of the stiff servo)
    o_max: float = 0.08       # |o| bound with the gripper closed (m)
    o_max_open: Optional[float] = 0.02   # |o| bound with the gripper open (m)
    v_bound: float = 0.05     # rate of shrinking the bound (m/s)
    v_ret: float = 0.03       # return speed of the offset out of contact (m/s)
    d_veto: float = 0.015     # not-a-constraint veto: force persists while the EE moves this far along it (m)
    warm: float = 0.5         # no contact action before this time (s)


class GCCLaw:
    def __init__(self, cfg: GCCConfig, dt, observer=None):
        self.c, self.dt, self.obs = cfg, dt, observer

    def reset(self):
        self.o = np.zeros(3); self.inc = False; self.t_low = 0.0; self.d_away = 0.0; self.n_veto = 0; self.t_on = 0.0
        self._vb_prev = None; self._aee = 0.0; self._fon = 0.0
        self.vbuf = []; self.t_stall = 0.0; self.giveup = False; self.t_give = 0.0; self.n = np.zeros(3)

    def step(self, F, v_ref, v_ee, vb, wb, p_now, ref_now, closed, observer_force=True):
        """F: filtered force estimate; v_ref / v_ee: reference / realized EE velocity (base axes, world-fixed points);
        vb, wb: the base planner's (filtered) commanded base velocity / yaw rate; p_now / ref_now: measured EE and the
        reference now (overload reflex); closed: gripper command closed."""
        c = self.c; dt = self.dt; ob = self.obs
        Fn = float(np.linalg.norm(F))
        vb, wb = np.asarray(vb, float), float(wb)
        if self._vb_prev is None:
            self._vb_prev = (vb.copy(), wb)
        ab = np.linalg.norm(vb - self._vb_prev[0]) / dt + 0.5 * abs(wb - self._vb_prev[1]) / dt
        self._vb_prev = (vb.copy(), wb)
        moving = min(1.0, (np.linalg.norm(vb) + 0.5 * abs(wb)) / 0.05)
        a_ee = ob.a_ee if ob is not None else np.zeros(3)
        self._aee += (float(np.linalg.norm(a_ee)) - self._aee) * min(1.0, dt / 0.05)
        f_mot = min(c.fm_max, c.fm_vb * moving + c.fm_ab * ab + c.fm_ae * self._aee)
        self._fon = c.f_on + f_mot
        self.t_on = self.t_on + dt if Fn > self._fon else 0.0
        F_raw = ob.F_raw if ob is not None else F
        fast = c.f_fast is not None and float(np.linalg.norm(F_raw)) > c.f_fast + (self._fon - c.f_on)
        if not self.inc and (self.t_on >= c.t_on - 1e-9 or fast):
            self.inc = True; self.t_low = 0.0; self.t_stall = 0.0; self.giveup = False; self.d_away = 0.0
        elif self.inc:
            self.t_low = self.t_low + dt if Fn < c.f_off else 0.0
            if self.t_low > c.t_off:
                self.inc = False
            if Fn > c.f_off:
                self.d_away += max(0.0, float(v_ee @ F) / Fn) * dt
            if c.d_veto and self.d_away > c.d_veto and Fn > c.f_off and observer_force and ob is not None:
                ob.absorb(); F = np.zeros(3); Fn = 0.0
                self.inc = False; self.n_veto += 1
        self.vbuf.append(v_ee.copy())
        nw = max(1, int(round(c.t_win / dt)))
        if len(self.vbuf) > nw:
            self.vbuf = self.vbuf[-nw:]
        disp = np.sum(self.vbuf, 0) * dt
        if np.linalg.norm(disp) > c.d_win:
            s = unit(disp)
        elif np.linalg.norm(v_ref) > c.v_refmin:
            s = unit(v_ref)
        else:
            s = None
        if not self.inc:
            on = np.linalg.norm(self.o)
            if on > 0:
                self.o -= min(on, c.v_ret * dt) * self.o / on
            self.t_stall = 0.0; self.giveup = False
            return self.o, F
        do = np.zeros(3)
        Fr = F_raw if c.snap_raw else F
        Frn = float(np.linalg.norm(Fr))
        if c.f_snap is not None and max(Fn, Frn) > c.f_snap:
            nn_ = Fr / Frn if Frn >= Fn else F / max(Fn, 1e-9)
            pen = float((p_now - (ref_now + self.o)) @ nn_)
            if pen > 0:
                do += min(pen, c.snap_max) * nn_
        if s is None and np.linalg.norm(self.o) > 1e-3:
            s_eff = -unit(self.o); Fpar = float(F @ s_eff); Fperp = F - Fpar * s_eff
        elif s is None:
            Fpar, Fperp, s_eff = 0.0, F.copy(), None
        else:
            Fpar = float(F @ s); Fperp = F - Fpar * s; s_eff = s
        R = max(0.0, -Fpar) if s_eff is not None else 0.0
        prog = float(v_ee @ s_eff) if s_eff is not None else 0.0
        if prog > c.v_move or R < c.f_hold:
            self.t_stall = 0.0; self.giveup = False; self.t_give = 0.0
        else:
            self.t_stall += dt
            if self.giveup:
                self.t_give += dt
                if self.t_give > c.t_rest:
                    self.giveup = False; self.t_stall = 0.0
            elif self.t_stall > c.t_probe:
                self.giveup = True; self.t_give = 0.0
        f_cap = c.f_hold if self.giveup else c.f_push
        fp = float(np.linalg.norm(Fperp))
        if fp > 1e-6:
            npp = Fperp / fp
            rate = float(np.clip((fp - c.f_perp) / c.D_perp, -c.v_adm, c.v_adm))
            if rate > 0:
                do += rate * dt * npp
                if c.cancel:
                    vin = -(v_ref @ npp) - max(0.0, -(v_ee @ npp))
                    if vin > 0:
                        do += vin * dt * npp
            else:
                on = self.o @ npp
                if on > 0:
                    do -= min(on, -rate * dt) * npp
        if R > 0:
            rate = float(np.clip((R - f_cap) / c.D_par, -c.v_adm, c.v_adm))
            if rate > 0:
                do += rate * dt * (-s_eff)
                if c.cancel:
                    vin = (v_ref @ s_eff) - max(0.0, v_ee @ s_eff)
                    if vin > 0:
                        do += vin * dt * (-s_eff)
            else:
                on = self.o @ (-s_eff)
                if on > 0:
                    do -= min(on, -rate * dt) * (-s_eff)
        elif s_eff is not None and Fpar > c.f_push:
            do += min(c.v_adm, (Fpar - c.f_push) / c.D_par) * dt * s_eff
        self.o = self.o + do
        om = c.o_max if (closed or c.o_max_open is None) else c.o_max_open
        on = np.linalg.norm(self.o)
        if on > om:
            on0 = float(np.linalg.norm(self.o - do))
            lim = max(om, on0 - c.v_bound * dt)
            if on > lim:
                self.o *= lim / on
        return self.o, F
