import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_pick_matrix import check_start, component_action


class StartGateTests(unittest.TestCase):
    def test_teacher_channels_and_window_are_bounded(self):
        for component, channels, end in [('base', [11, 12], 30), ('arm-torso', list(range(7))+[10], 30), ('gripper', [7], 40)]:
            action, selected = component_action([0.]*13, [.5]*13, component, end)
            self.assertEqual(selected, channels)
            self.assertEqual([i for i,v in enumerate(action) if v], channels)
            self.assertEqual(component_action([0.]*13, [.5]*13, component, end+1), ((0.,)*13, []))
    def test_rejects_velocity_drift_even_when_positions_match(self):
        reference = {k: [[0., 1.]] for k in ('qpos', 'qvel', 'tcp_pose', 'object_pose')}
        current = dict(reference, qvel=[[0., 1.01]])
        errors, passed = check_start(reference, current)
        self.assertFalse(passed)
        self.assertGreater(errors['qvel'], .009)

    def test_accepts_numerical_tolerance(self):
        reference = {k: [[0., 1.]] for k in ('qpos', 'qvel', 'tcp_pose', 'object_pose')}
        current = dict(reference, object_pose=[[0., 1.000001]])
        self.assertTrue(check_start(reference, current)[1])

    def test_nonfinite_state_is_not_comparable(self):
        reference = {k: [[0., 1.]] for k in ('qpos', 'qvel', 'tcp_pose', 'object_pose')}
        with self.assertRaises(ValueError):
            check_start(reference, dict(reference, qpos=[[float('nan'), 1.]]))
