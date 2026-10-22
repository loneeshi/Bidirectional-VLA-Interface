"""flowsde_ctrl: classical whole-body EE tracking controllers, packaged for reuse on other embodiments.

Layers
  robot.py        RobotSpec (embodiment description) + RobotModel (identified model, kinematics / inertia)
  inputs.py       TrackInput / Command (adapter <-> controller interface)
  components      reference.py, ik.py, servo.py, latency.py, observer.py, contact.py, base.py, perception.py, nav.py
  tracker.py      WholeBodyTracker (composition)  |  nav.py NavTracker (perception-driven base layer)
  variants.py     frozen presets: fair, fair_dc, ft_react, combo_ft_bp, combo_dob_bp, cc_obs_v5
  adapters/       unified.py (reference: UnifiedTrack / PandaOmron), fixed_arm.py (generic fixed-base MuJoCo arm)
  embodiments/    pandaomron.py (reference), kinova_gen3.py (second embodiment)
See README.md and docs/TUNING_GUIDE.md.
"""
from flowsde_ctrl.robot import RobotSpec, RobotModel, InnerLoopSpec, BaseSpec, SensorSpec, LatencySpec  # noqa: F401
from flowsde_ctrl.inputs import TrackInput, Command  # noqa: F401
from flowsde_ctrl.tracker import WholeBodyTracker, TrackerConfig  # noqa: F401
from flowsde_ctrl import variants  # noqa: F401

__version__ = "0.1.0"
