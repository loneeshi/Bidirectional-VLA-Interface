"""Reference preview handling (task-agnostic): C2 cubic spline through the previewed EE poses.

Clean copy of flowsde/classical.py ClassicalTracker._ref / _target_at. Positions are splined directly; rotations as
rotation vectors relative to the sample nearest to one reference period ahead. The reference is expressed in the
CURRENT base frame; `target_at` re-expresses it in the base frame PREDICTED at time s from the measured base velocity
(constant-velocity assumption over the short servo horizon).
"""
import numpy as np
from scipy.interpolate import CubicSpline

from flowsde_ctrl.math3d import rotvec, expm, Rz


class ReferencePreview:
    def build(self, inp, p, R):
        """p, R: current EE pose (base frame, nominal kinematics; one physics step stale, see tracker)."""
        t = inp.ref_t
        P = p + inp.ref_rel_p
        Rs = np.einsum("ij,njk->nik", R, inp.ref_rel_R)
        j0 = int(np.argmin(np.abs(t - inp.ref_dt)))
        Rc = Rs[j0]
        W = np.array([rotvec(Rc.T @ Ri) for Ri in Rs])
        self.sp, self.sw, self.Rc, self.P, self.Rs, self.t = CubicSpline(t, P), CubicSpline(t, W), Rc, P, Rs, t
        return self

    def pos(self, s):
        return self.sp(s)

    def vel(self, s):
        return self.sp.derivative()(s)

    def ang_vel(self, s):
        return self.Rc @ self.sw.derivative()(s)

    def target_at(self, s, vlin, wz, anchor):
        """Reference pose at time s expressed in the PREDICTED base frame at time s (constant base velocity
        vlin (base frame) and yaw rate wz about the yaw axis point `anchor`)."""
        pt = self.sp(s); Rt = self.Rc @ expm(self.sw(s))
        a = anchor; Rinv = Rz(-wz * s)
        return a + Rinv @ (pt - a - vlin * s), Rinv @ Rt
