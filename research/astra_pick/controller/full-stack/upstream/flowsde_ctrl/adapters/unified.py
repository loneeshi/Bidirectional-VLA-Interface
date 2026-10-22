"""Reference adapter: UnifiedTrack-v0 / TaskTrackV1 / NavTrack-v0 (flowsde/task_track_v1.py, PandaOmron).

Everything specific to that plant interface lives here:
  * the identified model: the env's nominal model (env.m_nom under hw_rand, never the per-episode draw; know_plant=True
    = the true nominal plant incl. arm_dyn armature / friction / damping; know_plant=False = robosuite-Panda arm values,
    the model-mismatch test), the start configuration and the nominal joint damping env.damp0;
  * the observation layout (flowsde/classical.py parse_obs, copied) -> TrackInput;
  * Command -> the 18-dim v1 action (setpoint offsets x0.3 rad, base body-frame velocity + yaw / 1.5, torso / 0.1,
    velocity feed-forward / 2 rad/s);
  * the wrist F/T driver buffer (env._ft) and the nominal latency (env.obs_latency_ms).
make_policy(variant) returns act(o, env) with one controller per env instance, a drop-in replacement for the research
controllers in scripts/eval_unified_ext.py / scripts/eval_nav.py (bit-identical actions, see flowsde_ctrl/tests).
"""
import copy

import mujoco
import numpy as np

from flowsde_ctrl import variants as V
from flowsde_ctrl.embodiments import pandaomron as PO
from flowsde_ctrl.inputs import TrackInput
from flowsde_ctrl.robot import RobotModel
from flowsde_ctrl.tracker import WholeBodyTracker

SEG = np.arange(-2, 11).astype(float)        # reference preview sample offsets (flowsde/task_track.py SEG)
REF_DT = 0.05                                # 20 Hz reference
N_BASE_OBS = len(SEG) * 9 + 14 + 7 + 7 + 4 + 1 + 2 + 2
EZ = np.array([0.0, 0.0, 1.0])


def robot_spec(u, ft=None, latency_ms=None):
    L = getattr(u, "obs_latency_ms", None) or {}
    ms = L.get("ms", 0.0) if latency_ms is None else latency_ms
    s = PO.spec(ft=bool(getattr(u, "_ft", None) is not None) if ft is None else ft, latency_ms=ms, n_sub=u.n_sub)
    s.latency.anchor = L.get("anchor", "obs") if L else "true"
    return s


def robot_model(u, spec, know_plant=True):
    """The identified model the controllers use (see module docstring)."""
    m = u.m_nom if getattr(u, "hw", None) else u.m
    m = copy.copy(m)
    dyn = getattr(u, "arm_dyn", {}) or {}
    if dyn and not know_plant:
        m.dof_armature[u.arm_v] /= np.asarray(dyn.get("armature", 1.0), float)
    mod = RobotModel(spec, m, init_qpos=u.init_qpos, share=True)
    mod.D_nom = u.damp0[u.arm_v].copy()
    mod.F_nom = (u.m_nom if getattr(u, "hw", None) else u.m).dof_frictionloss[u.arm_v].copy()
    if dyn and not know_plant:
        mod.D_nom /= np.asarray(dyn.get("damping", 1.0), float); mod.F_nom /= np.asarray(dyn.get("friction", 1.0), float)
    return mod


def parse_obs(o, mod):
    """UnifiedTrack-v0 observation -> dict (flowsde/classical.py parse_obs, copied)."""
    o = np.asarray(o, np.float64)
    q_lo, q_hi = mod.q_lo, mod.q_hi
    n = len(SEG)
    seg = o[:9 * n].reshape(n, 9)
    rel_p = seg[:, :3]
    c0, c1 = seg[:, 3:6], seg[:, 6:9]
    rel_R = np.stack([c0, c1, np.cross(c0, c1)], -1)
    i = 9 * n
    sq, cq, qn = o[i:i + 7], o[i + 7:i + 14], o[i + 14:i + 21]; i += 21
    q_lin = q_lo + 0.5 * (qn + 1) * (q_hi - q_lo)
    q = np.arctan2(sq, cq); q = q + 2 * np.pi * np.round((q_lin - q) / (2 * np.pi))
    qd = 2.0 * o[i:i + 7]; i += 7
    bv = o[i:i + 3]; i += 3
    wz = o[i]; h = o[i + 1]; i += 2
    i += 2
    g = o[i:i + 2]; i += 2
    assert i == N_BASE_OBS
    J = o[i:i + 66].reshape(6, 11); i += 66
    i += 12
    return dict(rel_p=rel_p, rel_R=rel_R, q=q, qd=qd, bv=bv, wz=wz, h=h, g=g, J=J, i_phase=i)


def to_input(o, u, mod):
    ob = parse_obs(o, mod)
    oa = np.asarray(o, np.float64)
    ph = float(oa[ob["i_phase"]:ob["i_phase"] + u.R + 1][-1])
    J = ob["J"]
    Mb = np.stack([J[:3, 7], J[:3, 8], EZ], 1)
    Mw = np.stack([mod.ax_w[0], mod.ax_w[1], EZ], 1)
    RbT = Mb @ np.linalg.inv(Mw)
    v = RbT.T @ ob["bv"]
    return TrackInput(episode_start=bool(u.t == 0 and u.sub == 0), q=ob["q"], qd=ob["qd"],
                      ref_t=(SEG + ph) * REF_DT, ref_rel_p=ob["rel_p"], ref_rel_R=ob["rel_R"], ref_dt=REF_DT,
                      h=ob["h"], base_vel=np.array([v[0], v[1], ob["wz"]]),
                      base_axes=np.stack([J[:3, 7], J[:3, 8]]), J_wb=J[:, :10], grip_cmd=ob["g"])


def ft_samples(u, oracle=False):
    """the F/T driver's 500 Hz samples since the last tick (consumed), as flowsde/classical_ft.py reads them."""
    if getattr(u, "_ft", None) is None:
        return None
    buf = list(getattr(u._ft, "buffer", []))
    u._ft.buffer = []
    if oracle:
        buf = [u._ft.true_last.copy()]
    if not buf:
        w = u.ft_wrench(); buf = [w if w is not None else np.zeros(6)]
    return buf


def to_action(cmd, inp, spec):
    s = spec.inner
    q_anchor = cmd.info.get("q_anchor", inp.q)
    ax = inp.base_axes; ctrl = cmd.base_cmd
    a = np.zeros(18)
    a[:7] = np.clip((cmd.q_sp - q_anchor) / s.sp_scale, -1, 1)
    ab = np.zeros(3)
    ab[:2] = ctrl[0] * ax[0][:2] + ctrl[1] * ax[1][:2]; ab[2] = ctrl[2] / spec.base.yaw_action_scale
    a[7:10] = np.clip(ab, -1, 1)
    a[10] = np.clip(cmd.vz / spec.base.torso_vmax, -1, 1)
    a[11:18] = np.clip(cmd.qd_ff / s.vff_scale, -1, 1)
    return a


def R_site_hand(mod, hand="robot0_right_hand"):
    m, d = mod.m, mod.d
    hb = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, hand)
    mujoco.mj_kinematics(m, d)
    Rs = d.site_xmat[mod.site].reshape(3, 3); Rh = d.xmat[hb].reshape(3, 3)
    return Rs.T @ Rh


class UnifiedController:
    """One controller bound to one env instance."""

    def __init__(self, env, cfg, know_plant=True, oracle_ft=False):
        u = env.unwrapped
        self.u = u
        self.spec = robot_spec(u)
        self.mod = robot_model(u, self.spec, know_plant)
        self.oracle_ft = oracle_ft
        Rsh = R_site_hand(self.mod) if cfg.force == "ft" else None
        self.ctrl = WholeBodyTracker(self.spec, self.mod, cfg, R_site_sensor=Rsh)
        self.last_inp = None

    def act(self, o, env=None):
        inp = to_input(o, self.u, self.mod)
        if self.ctrl.ft is not None:
            inp.ft_samples = ft_samples(self.u, self.oracle_ft)
        cmd = self.ctrl.step(inp)
        self.last_inp, self.last_cmd = inp, cmd
        return to_action(cmd, inp, self.spec)


def make_policy(variant="fair", cfg=None, know_plant=True, **kw):
    """act(o, env) for a frozen preset (variants.VARIANTS) or an explicit TrackerConfig `cfg`."""
    cache = {}

    def act(o, env):
        u = env.unwrapped
        if id(u) not in cache:
            c = cfg if cfg is not None else V.make(variant, **kw)
            cache[id(u)] = UnifiedController(env, c, know_plant=know_plant)
        act.ctrl = cache[id(u)]
        return cache[id(u)].act(o, env)
    act.variant = variant
    act.trackers = cache
    return act


# ---------------------------------------------------------------------------------------------------- NavTrack
class NavWorld:
    """NavTrack-v0 (flowsde/nav_scenes.py) world interface for flowsde_ctrl.nav.NavTracker: localisation (perfect,
    labelled), the reference over the planner horizon, the sim handles for the ray-cast sensors and the task region."""

    def __init__(self, u):
        self.u = u

    def base_state(self):
        return self.u.base_state()

    def base_origin(self, cx, cy, th):
        return self.u.base_origin(cx, cy, th)

    @property
    def base_z(self):
        return self.u.base_z

    @property
    def yaw_anchor_b(self):
        return self.u.yaw_anchor_b

    @property
    def carry_R(self):
        return self.u.carry_R

    @property
    def ref_pos(self):
        return self.u.ref["pos"]

    @property
    def ref_rot(self):
        return self.u.ref["rot"]

    def time(self):
        return self.u.t * REF_DT + self.u.sub * self.u.dt_pol

    @property
    def m(self):
        return self.u.m

    @property
    def d(self):
        return self.u.d

    @property
    def grid0(self):
        return self.u.nav_task["grid"]

    @property
    def seed(self):
        return getattr(self.u, "nav_idx", 0)

    def exact_H(self, grid):
        """PRIVILEGED (L0 only): exact height map of the scene."""
        from flowsde import nav_scenes as NS
        task = self.u.nav_task
        if task.get("prims") is not None:
            return NS.rasterize(task["prims"], NS.Grid(grid.x0, grid.y0, grid.nx, grid.ny, grid.cell))
        H0 = task["H0"]; x0, y0, nx, ny, c0 = task["grid"]
        f = int(round(grid.cell / c0))
        if f <= 1:
            return H0.copy()
        nx2, ny2 = grid.nx, grid.ny
        Hp = np.zeros((nx2 * f, ny2 * f), np.float32); Hp[:nx, :ny] = H0[:nx2 * f, :ny2 * f]
        return Hp.reshape(nx2, f, ny2, f).max(axis=(1, 3))


class UnifiedNavController:
    def __init__(self, env, cfg):
        from flowsde_ctrl.nav import NavTracker
        u = env.unwrapped
        self.u = u
        self.spec = robot_spec(u)
        self.mod = robot_model(u, self.spec, True)
        if not cfg.spheres:
            cfg.spheres = PO.NAV_SPHERES
        self.ctrl = NavTracker(self.spec, self.mod, cfg, NavWorld(u))

    def act(self, o, env=None):
        inp = to_input(o, self.u, self.mod)
        cmd = self.ctrl.step(inp)
        return to_action(cmd, inp, self.spec)


def make_nav_policy(level="L2+L4", **over):
    """act(o, env) of the perception-driven base layer (results/nav/FROZEN_mink_nav.json: plan_margin 0.15 for depth-only
    L2 / L3 and the L2+L4 fusion as run in the study, 0.06 otherwise; d_s_base 0.04)."""
    from flowsde_ctrl.nav import NavConfig
    cache = {}

    def act(o, env):
        u = env.unwrapped
        if id(u) not in cache:
            c = NavConfig(level=level)
            for k, v in over.items():
                setattr(c, k, v)
            cache[id(u)] = UnifiedNavController(env, c)
        return cache[id(u)].act(o, env)
    act.trackers = cache
    return act
