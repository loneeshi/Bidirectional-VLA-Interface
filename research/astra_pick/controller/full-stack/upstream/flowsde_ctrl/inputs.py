"""Per-tick controller input / output (embodiment-agnostic interface between an adapter and the controllers).

The controllers consume ONLY what a real robot's control computer has: joint encoders, base odometry velocities, the
reference preview from the planner, the gripper command, and (if fitted) a wrist F/T stream. Nothing about the scene or
the task. Adapters (flowsde_ctrl/adapters/) build a TrackInput from their plant's observation and turn the Command into
their plant's action.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class TrackInput:
    episode_start: bool                  # first tick of an episode (controller state is reset)
    q: np.ndarray                        # arm joint positions (rad)
    qd: np.ndarray                       # arm joint velocities (rad/s)
    ref_t: np.ndarray                    # (N,) times of the reference preview samples relative to now (s), increasing
    ref_rel_p: np.ndarray                # (N, 3) reference position minus current EE position, base frame (m)
    ref_rel_R: np.ndarray                # (N, 3, 3) R_ee^T R_ref (relative rotation of each sample)
    ref_dt: float = 0.05                 # nominal sample period of the reference (s)
    h: float = 0.0                       # torso height (m), 0 without a torso
    base_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))   # base joint velocities [v_ax0, v_ax1, wz]
    base_axes: np.ndarray = field(default_factory=lambda: np.eye(3)[:2])   # (2, 3) base translation dof axes, base frame
    J_wb: Optional[np.ndarray] = None    # (6, n + 3) base-frame EE Jacobian incl. base dofs (whole-body planner term)
    grip_cmd: np.ndarray = field(default_factory=lambda: np.zeros(2))   # gripper command now / in 3 reference steps (>0 closed)
    ft_samples: Optional[list] = None    # wrist F/T samples since the last tick: [F(3), T(3)] in the sensor frame


@dataclass
class Command:
    q_sp: np.ndarray                     # absolute arm joint setpoint for the inner PD loop
    qd_ff: np.ndarray                    # arm velocity feed-forward (rad/s), already clipped to the interface range
    base_cmd: np.ndarray                 # (3,) base joint commands incl. friction feed-forward [ax0, ax1, yaw]
    vz: float = 0.0                      # torso velocity (m/s)
    info: dict = field(default_factory=dict)
