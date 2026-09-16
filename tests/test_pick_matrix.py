import importlib.util
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_pick_matrix import check_start, component_action, compare_state_tree, controller_snapshot


class StartGateTests(unittest.TestCase):
    def test_float_jitter_is_logged_but_discrete_state_drift_fails(self):
        diffs, ok = compare_state_tree({'pose':[1.0], 'pointer':1}, {'pose':[1.00000001], 'pointer':1})
        self.assertTrue(ok)
        self.assertEqual(diffs[0]['path'], '/pose/0')
        for current in ({'pose':[1.01],'pointer':1}, {'pose':[1.0],'pointer':2},
                        {'pose':[float('nan')],'pointer':1}, {'pose':[1.0],'pointer':True}, {'pose':[1.0]}):
            self.assertFalse(compare_state_tree({'pose':[1.0],'pointer':1}, current)[1])

    def test_empty_public_controller_state_does_not_hide_targets(self):
        class C:
            _target_qpos=[.4]; _start_qpos=[.3]; _step=5
            def get_state(self): return {}
        snap=controller_snapshot(C(), lambda x:x)
        self.assertEqual(snap['_target_qpos'], [.4])
        other=dict(snap, _target_qpos=[.5])
        self.assertFalse(compare_state_tree(snap, other)[1])

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
