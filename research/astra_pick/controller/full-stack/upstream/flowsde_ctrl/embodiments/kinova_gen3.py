"""Second embodiment: Kinova Gen3 7-DoF (MuJoCo Menagerie `kinova_gen3/gen3.xml`, BSD-3-Clause), fixed base.

Brought up with the TUNING_GUIDE procedure by flowsde_ctrl/sim/bringup_gen3.py (results in results/flowsde_ctrl_gen3/):
identification of armature / viscous damping / Coulomb friction / payload from an excitation run, inner-loop gains from
the identified inertia (target bandwidth, critical damping), observer calibration on free-space DROID references,
no base planner (fixed base), forward-prediction latency compensation at the measured latency.

TRUE_HW is the hidden "hardware" of the simulated plant (sim/fixed_arm.py) -- the controller never reads it; the
bring-up identifies it. Values are plausible for harmonic-drive actuators (~100:1): reflected rotor inertia
0.1-0.35 kg m^2, 1-2 N m s/rad damping, 0.8-1.5 N m Coulomb friction, plus a 0.4 kg tool payload.
"""
import os

import numpy as np

from flowsde_ctrl.robot import BaseSpec, InnerLoopSpec, LatencySpec, RobotSpec, SensorSpec

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
XML = os.path.join(ROOT, "assets", "mujoco_menagerie", "kinova_gen3", "gen3.xml")
ARM_JOINTS = [f"joint_{i}" for i in range(1, 8)]
EE_SITE = "pinch_site"
TAU_MAX = [105.0] * 4 + [52.0] * 3                     # MJCF actuator force ranges (large / small actuators)
QD_MAX = [1.39] * 4 + [1.22] * 3                       # Kinova Gen3 joint speed limits (rad/s, ~80 / 70 deg/s)
VENDOR_KP = [2000.0] * 4 + [500.0] * 3                 # MJCF position-actuator gains (used only for the excitation run)
VENDOR_KD = [100.0] * 4 + [50.0] * 3
Q_START = [0.0, 0.35, np.pi, -1.95, 0.0, 1.05, np.pi / 2]   # start posture (>= 0.6 rad from the joint-4 / joint-6 limits)
LINKS = ["shoulder_link", "half_arm_1_link", "half_arm_2_link", "forearm_link", "spherical_wrist_1_link",
         "spherical_wrist_2_link", "bracelet_link"]

TRUE_HW = dict(armature=[0.35, 0.35, 0.30, 0.30, 0.12, 0.12, 0.12],
               damping=[2.0, 2.0, 1.5, 1.5, 0.8, 0.8, 0.8],
               friction=[1.5, 1.5, 1.2, 1.2, 0.8, 0.8, 0.8])
TRUE_PAYLOAD = dict(body="bracelet_link", mass=0.4, pos=[0.0, 0.0, -0.10], size=0.03)


def spec(kp, kd, identified=None, latency_ms=0.0, n_sub=5):
    return RobotSpec(
        name="KinovaGen3",
        arm_joints=ARM_JOINTS, ee_site=EE_SITE,
        inner=InnerLoopSpec(kp=list(kp), kd=list(kd), tau_max=TAU_MAX, physics_dt=0.002, n_sub=n_sub, sp_scale=0.3,
                            vff_scale=2.0, sp_window=0.3, pose_delay_steps=0),
        base=BaseSpec(type="fixed", body="base_link"),
        sensors=SensorSpec(),                          # proprioception only (fair tier)
        latency=LatencySpec(ms=latency_ms, anchor="obs"),
        qd_max=QD_MAX, shoulder_body="shoulder_link", wrist_body="bracelet_link", hand_body="bracelet_link",
        link_bodies=LINKS, identified=dict(identified or {}),
        notes="Menagerie gen3.xml; continuous joints 1/3/5/7 limited to +-2 pi by the controller",
    )
