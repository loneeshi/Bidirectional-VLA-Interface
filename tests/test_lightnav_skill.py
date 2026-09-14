import math
import unittest
from types import SimpleNamespace as NS

from bvi.lightnav_skill import (FetchNavigationControl, normalize, track_waypoint,
                               world_waypoint)
from bvi.protocol import ActionBounds, ProtocolError


class TrackerTests(unittest.TestCase):
    def test_forward_left_and_in_place_turn(self):
        self.assertEqual(track_waypoint((0, 0, 0), (1, 0, 0)), (.25, 0., False))
        v, w, _ = track_waypoint((0, 0, 0), (0, 1, 0))
        self.assertEqual(v, 0)
        self.assertGreater(w, 0)
        v, w, done = track_waypoint((0, 0, 0), (0, 0, -.3))
        self.assertEqual(v, 0)
        self.assertLess(w, 0)
        self.assertFalse(done)

    def test_world_rotation_and_yaw_wrap(self):
        x, y, yaw = world_waypoint((4, 5, math.pi/2), (1, 0, math.pi))
        self.assertAlmostEqual(x, 4)
        self.assertAlmostEqual(y, 6)
        self.assertAlmostEqual(yaw, -math.pi/2)
        # Row2 is cumulative relative to capture, not relative to row1.
        self.assertEqual(world_waypoint((1, 1, 0), (2, 0, 0))[:2], (3, 1))

    def test_nonfinite_and_normalization(self):
        with self.assertRaises(ProtocolError):
            track_waypoint((0, 0, float('nan')), (1, 0, 0))
        self.assertEqual(normalize((-.01,), (-.01,), (.05,)), (-1.,))
        self.assertAlmostEqual(normalize((0.,), (-.01,), (.05,))[0], -2/3)
        self.assertEqual(normalize((.25,), (-1.,), (1.,)), (.25,))

    def adapter(self):
        def ctrl(cls, qpos, low, high, names=(), delta=True):
            c = type(cls, (), {})()
            c._normalize_action = True
            c._original_single_action_space = NS(shape=(len(low),), low=low, high=high)
            c.config = NS(use_delta=delta, use_target=False, joint_names=list(names))
            c.control_freq = 20
            c.qpos = [qpos]
            return c
        controllers = dict(
            arm=ctrl('PDJointPosController', [0]*7, [-.1]*7, [.1]*7),
            gripper=ctrl('PDJointPosMimicController', [.005]*2, [-.01], [.05], delta=False),
            body=ctrl('PDJointPosController', [0,0,.3], [-.1]*3, [.1]*3,
                      ['head_pan_joint','head_tilt_joint','torso_lift_joint']),
            base=ctrl('PDBaseForwardVelController', [0,0,0], [-1,-3.14], [1,3.14],
                      ['root_x_axis_joint','root_y_axis_joint','root_z_rotation_joint']))
        controllers['gripper']._target_qpos = [[-.01, -.01]]
        return NS(uenv=NS(agent=NS(controller=NS(controllers=controllers))),
                  logger=NS(emit=lambda *a, **k: None),
                  action_bounds=ActionBounds((-1.,)*13, (1.,)*13))

    def test_hold_preserves_closed_target_and_compensates_torso_drift(self):
        adapter = self.adapter()
        control = FetchNavigationControl(adapter)
        control.capture_hold()
        control.controllers['body'].qpos = [[0,0,.28]]
        action = control.action(.25, .314)
        self.assertEqual(len(action), 13)
        self.assertEqual(action[7], -1)  # Not zero, which would command opening.
        self.assertAlmostEqual(action[10], .2)
        self.assertAlmostEqual(action[11], .25)
        self.assertAlmostEqual(action[12], .1)

    def test_changed_controller_contract_is_rejected(self):
        adapter = self.adapter()
        adapter.uenv.agent.controller.controllers['base'].config.joint_names.reverse()
        with self.assertRaises(ProtocolError):
            FetchNavigationControl(adapter)
