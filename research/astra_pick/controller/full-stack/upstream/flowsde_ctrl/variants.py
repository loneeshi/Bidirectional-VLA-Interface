"""Frozen controller presets (TrackerConfig factories). Values are the frozen research configs:

  fair          flowsde/classical.py "fair" (results/unified/classical_fair/FROZEN.json == in-code defaults), run with
                the identified plant model (know_plant). Tier: fair.  Precise servo, reach-band base.
  fair_dc       + forward-prediction latency compensation (flowsde/classical_delay.py "fair_dc"; nominal latency from
                RobotSpec.latency.ms). Tier: fair + known nominal latency.
  ft_react      wrist-F/T reaction arm (flowsde/classical_delay.py "ft_react": classical_ft ft_react, f_ref 15).
                Tier: HW upgrade (wrist F/T).
  combo_ft_bp   mink BASE_PLAN + ft_react arm (results/combo_2026-10-06/combo_ft_bp/FROZEN.json). Tier: HW (wrist F/T).
  combo_dob_bp  mink BASE_PLAN + simple observer reaction arm, base-coupling compensated, base command conditioning
                (results/combo_2026-10-06/combo_dob_bp/FROZEN.json). Tier: fair.
  cc_obs_v5     observer-based generic constraint compliance on the combo_dob_bp base
                (results/constraint_2026-10-06/cc_obs_v5/FROZEN.json). Tier: fair. Best deployable without F/T.

Each factory returns a fresh TrackerConfig; override fields freely (dataclasses.replace or attribute assignment).
"""
from flowsde_ctrl.base import MinkBaseConfig, WholeBodyQPConfig
from flowsde_ctrl.contact import GCCConfig, ReactConfig
from flowsde_ctrl.ik import IKConfig
from flowsde_ctrl.latency import LatencyConfig
from flowsde_ctrl.observer import ObserverConfig, SimpleDobConfig
from flowsde_ctrl.tracker import TrackerConfig

BASE_PLAN = dict(closed_loop=False, posture_cost=1e-3, base_cost=0.3)      # flowsde/zoo/__init__.py BASE_PLAN
# NOTE (reproduced quirk): in the research code the combo / constraint controllers merge DEFAULT_COMBO["lim_margin"] = 0.0
# (mink arm-range shrink, unused there) over the classical IK's lim_margin = 0.3, so their arm IK runs WITHOUT
# joint-limit weighting / repulsion. The frozen presets keep that (IK_MINK); see TUNING_GUIDE §8.
IK_MINK = dict(lim_margin=0.0)


def fair():
    return TrackerConfig(base="reach", tier="fair")


def fair_dc(ms=None):
    return TrackerConfig(base="reach", latency=LatencyConfig(ms=ms), tier="fair + nominal latency")


def ft_react():
    return TrackerConfig(base="reach", force="ft", contact="react",
                         react=ReactConfig(mode="react", f_ref=15.0), tier="hw:wrist_ft")


def combo_ft_bp():
    return TrackerConfig(base="mink", mink=MinkBaseConfig(qp=WholeBodyQPConfig(**BASE_PLAN)), ik=IKConfig(**IK_MINK),
                         force="ft", contact="react", react=ReactConfig(mode="react", f_ref=15.0, o_max=0.1),
                         tier="hw:wrist_ft")


def combo_dob_bp():
    return TrackerConfig(base="mink", ik=IKConfig(**IK_MINK),
                         mink=MinkBaseConfig(qp=WholeBodyQPConfig(**BASE_PLAN), base_tau=0.1, base_acc=1.0, yaw_acc=1.5),
                         force="dob_simple", simple_dob=SimpleDobConfig(warm=0.5, base_comp=True),
                         contact="react", react=ReactConfig(mode="react", f_ref=15.0, f_on=20.0, inc_tmax=0.5,
                                                            o_max=0.04, warm=0.5),
                         tier="fair")


def cc_obs_v5():
    return TrackerConfig(base="mink", ik=IKConfig(**IK_MINK),
                         mink=MinkBaseConfig(qp=WholeBodyQPConfig(**BASE_PLAN), base_tau=0.1, base_acc=1.0, yaw_acc=1.5),
                         force="observer", observer=ObserverConfig(me_tau=1.0, F_lp=0.02, dob_eps=1e-3, warm=0.5,
                                                                   gate_F=12.0, w_wrist=0.2, lim_eps=0.03, sat_frac=0.9,
                                                                   base_comp=True, early=True),
                         contact="gcc", gcc=GCCConfig(f_on=20.0, t_on=0.03, f_off=6.0, t_off=0.3, fm_vb=8, fm_ab=8, fm_ae=4,
                                                      fm_max=25.0, f_fast=40, f_snap=35, snap_raw=True, snap_max=0.01,
                                                      f_perp=5.0, D_perp=500, f_push=18, D_par=500, f_hold=8.0,
                                                      v_adm=0.1, t_win=0.15, d_win=0.004, v_refmin=0.01, v_move=0.01,
                                                      t_probe=1.5, t_rest=1.0, cancel=True, o_max=0.08, o_max_open=0.02,
                                                      v_bound=0.05, v_ret=0.03, d_veto=0.015, warm=0.5),
                         dob_clamp=True, tier="fair")


VARIANTS = dict(fair=fair, fair_dc=fair_dc, ft_react=ft_react, combo_ft_bp=combo_ft_bp, combo_dob_bp=combo_dob_bp,
                cc_obs_v5=cc_obs_v5)
NEEDS_FT = {"ft_react", "combo_ft_bp"}


def make(name, **kw):
    return VARIANTS[name](**kw)
