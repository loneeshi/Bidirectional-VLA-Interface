"""WholeBodyTracker: the classical whole-body EE tracking controller, composed from components.

  reference preview  -> ReferencePreview (C2 spline, predicted base frame)
  base / torso       -> ReachBandPlanner | MinkBasePlanner | NoBase,  then BaseActuatorMap
  force source       -> None | WristFT | SimpleDobForce | ForceObserver   (+ BaseCoupling for mobile bases)
  contact law        -> None | ReactLaw | GCCLaw   (EE reference offset; zero in free space)
  arm reference      -> DLSIK (integrated closed-loop IK, null-space posture)
  joint servo        -> PredictiveServo (+ DOB)
  latency            -> LatencyCompensator (optional forward prediction)

Per tick (controller period dt = n_sub * physics_dt; the order below reproduces the frozen research controllers
flowsde/classical.py, classical_ft.py, combo.py, constraint_comp.py, classical_delay.py exactly):
   1. kinematics on the identified model; the EE pose is evaluated one physics step stale (q - h qd), as the
      measurement pipeline of the reference plant reports it;
   2. reference spline; 3. base plan + actuator map (+ base-coupling torque);
   4. early observer update (ForceObserver); 5. force source; 6. contact law -> offset o;
   7. IK to the reference at t + dt and t + 2 dt (+ o); 8. DOB update (late variants) + predictive servo.
Every component has a config dataclass with documented defaults; `variants.py` holds the frozen presets.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from flowsde_ctrl.base import (BaseActuatorMap, MinkBaseConfig, MinkBasePlanner, NoBase, ReachBandConfig,
                               ReachBandPlanner)
from flowsde_ctrl.contact import GCCConfig, GCCLaw, ReactConfig, ReactLaw
from flowsde_ctrl.ik import DLSIK, IKConfig
from flowsde_ctrl.inputs import Command, TrackInput
from flowsde_ctrl.latency import LatencyCompensator, LatencyConfig
from flowsde_ctrl.math3d import unit
from flowsde_ctrl.observer import (BaseCoupling, ForceObserver, ObserverConfig, SimpleDobConfig, SimpleDobForce,
                                   WristFT)
from flowsde_ctrl.reference import ReferencePreview
from flowsde_ctrl.servo import PredictiveServo, ServoConfig


@dataclass
class TrackerConfig:
    """Composition of a tracker. Component configs not selected are ignored.
    tier: "fair" (proprioception + identified model), "hw:wrist_ft" (+ wrist F/T sensor)."""
    ik: IKConfig = field(default_factory=IKConfig)
    servo: ServoConfig = field(default_factory=ServoConfig)
    base: str = "reach"                       # "reach" | "mink" | "none"
    reach: ReachBandConfig = field(default_factory=ReachBandConfig)
    mink: MinkBaseConfig = field(default_factory=MinkBaseConfig)
    force: Optional[str] = None               # None | "ft" | "dob_simple" | "observer"
    simple_dob: SimpleDobConfig = field(default_factory=SimpleDobConfig)
    observer: ObserverConfig = field(default_factory=ObserverConfig)
    contact: Optional[str] = None             # None | "react" | "gcc"
    react: ReactConfig = field(default_factory=ReactConfig)
    gcc: GCCConfig = field(default_factory=GCCConfig)
    dob_clamp: bool = True                    # gcc: project the DOB update off the contact normal while latched
    latency: Optional[LatencyConfig] = None   # forward prediction (None = off)
    tier: str = "fair"


class WholeBodyTracker:
    def __init__(self, spec, model, cfg: TrackerConfig, R_site_sensor=None):
        self.spec, self.mod, self.cfg = spec, model, cfg
        inner = spec.inner
        self.h_phys, self.n_sub = inner.physics_dt, inner.n_sub
        self.dtp = inner.physics_dt * inner.n_sub
        self.dl = inner.pose_delay_steps * inner.physics_dt      # EE pose staleness of the plant (s)
        self.ref = ReferencePreview()
        self.ik = DLSIK(model, cfg.ik, self.dtp)
        self.servo = PredictiveServo(model, inner, cfg.servo)
        if spec.base.type == "fixed" or cfg.base == "none":
            self.base = NoBase()
        elif cfg.base == "mink":
            self.base = MinkBasePlanner(model, spec, cfg.mink, self.dtp)
        else:
            self.base = ReachBandPlanner(model, cfg.reach)
        self.base_map = BaseActuatorMap(spec.base)
        self.coupling = None
        self.ft = self.sdob = self.obs = None
        if cfg.force == "ft":
            if R_site_sensor is None:
                R_site_sensor = np.eye(3)
            self.ft = WristFT(model, R_site_sensor)
        elif cfg.force == "dob_simple":
            self.sdob = SimpleDobForce(cfg.simple_dob, self.dtp)
            if cfg.simple_dob.base_comp and model.n_base:
                self.coupling = BaseCoupling(model)
        elif cfg.force == "observer":
            self.obs = ForceObserver(model, cfg.observer, cfg.servo.dob_gain, cfg.servo.dob_max, self.dtp,
                                     inner.tau_max, inner.kp, inner.kd)
            if cfg.observer.base_comp and model.n_base:
                self.coupling = BaseCoupling(model)
            else:
                self.cfg.observer.base_comp = False
        self.react = self.gcc = None
        if cfg.contact == "react":
            self.react = ReactLaw(cfg.react, self.dtp, rebaseline=self.sdob.rebaseline if self.sdob else None)
        elif cfg.contact == "gcc":
            self.gcc = GCCLaw(cfg.gcc, self.dtp, observer=self.obs)
        self.lat = LatencyCompensator(self, cfg.latency) if cfg.latency is not None else None
        self.k = 0

    # ------------------------------------------------------------------ episode state
    def reset(self, inp):
        q = inp.q; n = len(q)
        self.q_ref = q.copy(); self.ik.q_post = q.copy()
        self.prev_h = inp.h; self.k = 0
        self.servo.reset(); self.base_map.reset(); self.base.reset(inp)
        self.n = np.zeros(3); self.o = np.zeros(3)
        if self.coupling is not None:
            self.coupling.reset()
        if self.sdob is not None:
            self.sdob.reset(n)
        if self.obs is not None:
            self.obs.reset(n)
        if self.react is not None:
            self.react.reset()
        if self.gcc is not None:
            self.gcc.reset()

    def in_contact(self):
        law = self.react or self.gcc
        return bool(law.inc) if law is not None else False

    # ------------------------------------------------------------------ main
    def step(self, inp: TrackInput) -> Command:
        ctx = None
        if self.lat is not None:
            inp, ctx = self.lat.pre(inp)
        cmd = self._step(inp)
        if self.lat is not None:
            cmd = self.lat.post(cmd, ctx)
        return cmd

    def _step(self, inp):
        cfg = self.cfg; mod = self.mod
        if inp.episode_start:
            self.reset(inp)
        q, qd, h = inp.q, inp.qd, inp.h
        closed = bool(inp.grip_cmd[0] > 0)
        dl = self.dl; D = self.dtp
        p, R = mod.fk(q - dl * qd, h, jac=False)
        _, R_now, J7, M = mod.fk(q, h, inertia=True)
        bvj = inp.base_vel; ax = inp.base_axes
        vlin, wz = bvj[0] * ax[0] + bvj[1] * ax[1], bvj[2]
        self.ref.build(inp, p, R)
        if self.coupling is not None:
            self.coupling.update(inp, D)
        v_b, w_b, vz = self.base.plan(inp, self.ref, q)
        base_cmd = self.base_map(inp, v_b, w_b)
        hd = (h - self.prev_h) / D if self.k > 0 else 0.0; self.prev_h = h
        if np.abs(self.q_ref - q).max() > cfg.ik.resync:
            self.q_ref = q.copy()
        # early (start-of-tick) observer update: one tick fresher force estimate
        if self.obs is not None and cfg.observer.early:
            if self.servo.pred is not None:
                self.obs.pre_tick()
            Jn = J7[:3].T @ self.n if (self.gcc is not None and self.gcc.inc and cfg.dob_clamp) else None
            res = self.servo.dob_update(q, qd, Jn)
            self.obs.update(res, self.coupling, q, qd, J7, self.k, self.in_contact())
        # force source
        t_el = self.k * D
        F, Fpk = np.zeros(3), 0.0
        if self.ft is not None:
            F, Fpk = self.ft.read(inp.ft_samples, R_now)
        elif self.sdob is not None:
            F, Fpk = self.sdob.read(J7, t_el, self.in_contact())
        elif self.obs is not None:
            F = self.obs.F_hat.copy()
        # contact law -> EE reference offset
        off = np.zeros(3)
        if self.react is not None and t_el >= cfg.react.warm:
            v_ref = self.ref.vel(D + dl)
            v_ee = J7[:3] @ qd + vlin
            off = self.react.step(F, Fpk, v_ref - vlin, v_ee)
        elif self.gcc is not None and t_el >= cfg.gcc.warm:
            p_now, ref_now = p, self.ref.pos(dl)
            v_ref = self.ref.vel(D + dl)
            v_ee = J7[:3] @ qd + vlin
            off, _ = self.gcc.step(F, v_ref, v_ee, getattr(self.base, "vb_f", np.zeros(2)), getattr(self.base, "w_f", 0.0),
                                   p_now, ref_now, closed,
                                   observer_force=self.obs is not None)
            self.n = unit(F) if self.gcc.inc else self.n
        self.o = off
        # arm reference: integrated IK at t + dt, t + 2 dt in the predicted base frame
        (p1, R1), (p2, R2) = self.ref.target_at(D + dl, vlin, wz, mod.anchor), self.ref.target_at(2 * D + dl, vlin, wz, mod.anchor)
        q1 = self.ik.solve(self.q_ref, h + hd * (D + dl), p1 + off, R1)
        q2 = self.ik.solve(q1, h + hd * (2 * D + dl), p2 + off, R2)
        q0 = self.q_ref
        # late DOB update (fair / F/T / simple-DOB variants), then the predictive servo
        if not (self.obs is not None and cfg.observer.early):
            Jn = J7[:3].T @ self.react.n if (self.react is not None and self.react.clamp_active()) else None
            res = self.servo.dob_update(q, qd, Jn)
            if self.sdob is not None:
                self.sdob.after_dob(res, cfg.servo.dob_max, cfg.servo.dob_gain, self.coupling)
        q_sp, qd_ff = self.servo.command(q, qd, M, q1, 0.5 * (q2 - q0) / D, 1.0, self.servo.D_nom)
        if self.obs is not None:
            self.obs.predict_saturation(q, qd, h, q_sp, qd_ff)
        self.q_ref = q1; self.k += 1
        return Command(q_sp=q_sp, qd_ff=qd_ff, base_cmd=base_cmd, vz=vz,
                       info=dict(F=np.asarray(F, float), contact=self.in_contact(), offset=np.array(off, float)))
