"""Generic adapter for a fixed-base arm (no base, no torso): measurements -> TrackInput, Command -> joint commands.

On hardware, replace `FixedArmPlant` by the robot driver: it must provide the measured (q, qd), the reference preview
(times relative to now, EE-relative positions / rotations in the base frame), the gripper command, and accept an
absolute joint setpoint + velocity feed-forward for the low-level joint loop.
"""
import numpy as np

from flowsde_ctrl.inputs import TrackInput
from flowsde_ctrl.tracker import WholeBodyTracker


class FixedArmController:
    def __init__(self, spec, model, cfg):
        self.spec, self.mod = spec, model
        self.ctrl = WholeBodyTracker(spec, model, cfg)
        self.first = True

    def reset(self):
        self.first = True

    def step(self, q, qd, ref_t, rel_p, rel_R, ref_dt=0.05, grip=(-1.0, -1.0)):
        inp = TrackInput(episode_start=self.first, q=np.asarray(q, float), qd=np.asarray(qd, float), ref_t=ref_t,
                         ref_rel_p=rel_p, ref_rel_R=rel_R, ref_dt=ref_dt, grip_cmd=np.asarray(grip, float))
        self.first = False
        cmd = self.ctrl.step(inp)
        s = self.spec.inner
        anchor = cmd.info.get("q_anchor", inp.q)
        # the plant interface takes an offset in [-1, 1] x sp_scale around the measured q and a bounded feed-forward
        a = np.clip((cmd.q_sp - anchor) / s.sp_scale, -1, 1)
        return anchor + s.sp_scale * a, np.clip(cmd.qd_ff, -s.vff_scale, s.vff_scale), cmd


def run_episode(plant, ctrl, ref_pos, ref_rot, q0=None, dt_pol=0.01, grip=(-1.0, -1.0)):
    """closed loop on a FixedArmPlant; returns the plant metrics (+ contact / force info)."""
    plant.reset(ref_pos, ref_rot, q0)
    ctrl.reset()
    done = False; n_in = 0; Fh = []
    while not done:
        q, qd = plant.measure()
        t, rp, rR = plant.preview(q)
        q_sp, qd_ff, cmd = ctrl.step(q, qd, t, rp, rR, grip=grip)
        n_in += int(bool(cmd.info.get("contact", False)))
        Fh.append(float(np.linalg.norm(cmd.info.get("F", np.zeros(3)))))
        done = plant.step(q_sp, qd_ff)
    out = plant.metrics(dt_pol)
    Fh = np.asarray(Fh)
    out.update(contact_ticks=n_in, F_hat_p50=float(np.median(Fh)), F_hat_p99=float(np.percentile(Fh, 99)),
               F_hat_max=float(Fh.max()), ticks=len(Fh))
    return out
