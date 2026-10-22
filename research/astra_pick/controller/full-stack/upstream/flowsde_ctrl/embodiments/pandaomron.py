"""Reference embodiment: robosuite PandaOmron (Franka Panda arm on the Omron holonomic base with a torso lift), as
simulated by flowsde/task_track*.py (UnifiedTrack-v0, NavTrack-v0). All values below are copied from the env
(flowsde/task_track.py KP / KD / TAU / ARM_DYN, flowsde/zoo/mink_wbik.py PANDA_QD_MAX, flowsde/classical.py base
actuator constants) -- the spec is only a description; the adapter takes the model itself from the env.

Plants: arm_dyn=None (robosuite Panda), "qdd" (uniform 1/20 armature), "openarm" (OpenArm-like QDD, the main plant of
the later studies). The identified model ("know_plant") is the env's nominal model for that plant.
"""
import numpy as np

from flowsde_ctrl.robot import BaseSpec, InnerLoopSpec, LatencySpec, RobotSpec, SensorSpec

KP = [600, 600, 600, 600, 250, 150, 50.0]
KD = [50, 50, 50, 50, 30, 25, 15.0]
TAU = [80, 80, 80, 80, 80, 12, 12.0]
PANDA_QD_MAX = [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]
# robosuite Panda joint parameters (before arm_dyn multipliers) and the OpenArm-like multipliers (task_track.ARM_DYN)
PANDA_ARMATURE = [5.0, 2.5, 1.667, 1.25, 1.0, 0.833, 0.714]
OPENARM_MULT = dict(armature=[0.0032, 0.0064, 0.096, 0.128, 0.010, 0.012, 0.014],
                    friction=[1.0, 1.0, 5.0, 5.0, 0.5, 0.5, 0.5], damping=[1.0, 1.0, 2.0, 2.0, 0.5, 0.5, 0.5])

DEPTH_CAM = dict(pos=(0.05, 0.0, 1.45), pitch=np.radians(40.0), hfov=np.radians(87.0), vfov=np.radians(58.0),
                 rmin=0.2, rmax=4.0, rate=0.1)
LIDAR = dict(z=0.25, n=720, rmax=6.0, rate=0.1)


def spec(ft=False, latency_ms=0.0, depth=False, lidar=False, n_sub=5):
    return RobotSpec(
        name="PandaOmron",
        arm_joints=[f"robot0_joint{i}" for i in range(1, 8)],
        ee_site="gripper0_right_grip_site",
        inner=InnerLoopSpec(kp=KP, kd=KD, tau_max=TAU, physics_dt=0.002, n_sub=n_sub, sp_scale=0.3, vff_scale=2.0,
                            sp_window=0.3),
        base=BaseSpec(type="holonomic",
                      joints=("mobilebase0_joint_mobile_forward", "mobilebase0_joint_mobile_side",
                              "mobilebase0_joint_mobile_yaw"),
                      torso_joint="mobilebase0_joint_torso_height", body="mobilebase0_base",
                      vmax=0.5, wmax=0.6, torso_vmax=0.1, fric_ff=(0.25, 0.25, 250.0 / 1500.0),
                      yaw_action_scale=1.5, acc_max=1.0, yaw_acc_max=1.5),
        sensors=SensorSpec(wrist_ft=ft, ft_body="robot0_right_hand", depth=DEPTH_CAM if depth else None,
                           lidar=LIDAR if lidar else None),
        latency=LatencySpec(ms=latency_ms, anchor="obs"),
        qd_max=PANDA_QD_MAX,
        shoulder_body="robot0_link1", wrist_body="robot0_link7", hand_body="robot0_right_hand",
        link_bodies=[f"robot0_link{i}" for i in range(1, 8)],
        gripper=dict(joints=["gripper0_right_finger_joint1", "gripper0_right_finger_joint2"]),
        notes="identified model = the env's nominal model of the selected plant (arm_dyn); see adapters/unified.py",
    )

# collision spheres for the perception-driven base layer (flowsde/zoo/mink_nav.py): (geom, offsets in the body frame
# relative to the geom position, radius, kind 0 = base / 1 = arm)
NAV_SPHERES = (
    ("mobilebase0_pedestal_feet_col", [(x, y, 0.0) for x in (-0.2, 0.0, 0.2) for y in (-0.1, 0.1)], 0.22, 0),
    ("mobilebase0_g0_col", [(0, 0, -0.12), (0, 0, 0.12)], 0.13, 0),
    ("mobilebase0_g1_col", [(0, 0, -0.12), (0, 0, 0.12)], 0.13, 0),
) + tuple((f"robot0_link{k}_collision", [(0.0, 0.0, 0.0)], r, 1)
          for k, r in ((0, 0.11), (1, 0.09), (2, 0.09), (3, 0.09), (4, 0.09), (5, 0.09), (6, 0.08)))
BASE_R = 0.43            # disc around the yaw axis covering the 0.70 x 0.50 x 0.38 m base box
