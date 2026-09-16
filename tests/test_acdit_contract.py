import unittest

import numpy as np

from bvi.acdit_contract import (
    STATE_JOINTS, UNIFIED_INDICES, NativeCallBoundary, controller_chunk,
    fetch_proprio, from_unified, privileged_context, to_unified,
    validate_pointcloud,
)
from bvi.protocol import ProtocolError


class ACDiTContractTests(unittest.TestCase):
    def test_root_joint_insertion_does_not_change_proprio(self):
        names = ('torso_lift_joint', 'head_pan_joint', 'shoulder_pan_joint',
                 'head_tilt_joint', 'shoulder_lift_joint', 'upperarm_roll_joint',
                 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint',
                 'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint')
        qpos = np.arange(12, dtype=np.float32) / 20
        expected = np.r_[qpos[[2, 4, 5, 6, 7, 8, 9, 10, 1, 3, 0]], .123, -.456]
        roots = ('root_x_axis_joint', 'root_y_axis_joint', 'root_z_rotation_joint')
        for qs, ns in [(qpos, names), (np.r_[99, 88, 77, qpos], roots + names)]:
            result = fetch_proprio(qs, ns, [.123, 2, 3], [4, 5, -.456])
            np.testing.assert_allclose(result, expected)
        # qpos input order may vary; name-based mapping must remain invariant.
        np.testing.assert_allclose(fetch_proprio(qpos[::-1], names[::-1],
                                                [.123, 2, 3], [4, 5, -.456]), expected)

    def test_missing_duplicate_nonfinite_joints_fail(self):
        for names, qpos in [(STATE_JOINTS[:-1], [0]*10),
                            (STATE_JOINTS + STATE_JOINTS[:1], [0]*12),
                            (STATE_JOINTS, [float('nan')]*11)]:
            with self.assertRaises(ProtocolError):
                fetch_proprio(qpos, names, [0]*3, [0]*3)

    def test_scatter_and_gather_have_no_normalization(self):
        values = np.arange(13, dtype=np.float32) * 2 - 5
        unified = to_unified(values)
        np.testing.assert_array_equal(from_unified(unified), values)
        self.assertEqual(unified[10], values[7])
        self.assertEqual(unified[125], values[8])
        self.assertEqual(unified[100], values[11])
        self.assertEqual(unified[102], values[12])
        self.assertTrue(np.all(unified[[i for i in range(128) if i not in UNIFIED_INDICES]] == 0))

    def context(self):
        return dict(goal_pos_wrt_base=[[1, 2, 3]], is_grasped=[True],
                    obj_pose_wrt_base=[[4, 5, 6, 1, 0, 0, 0]],
                    tcp_pose_wrt_base=[[7, 8, 9, 1, 0, 0, 0]])

    def test_privilege_consent_missing_fields_and_quaternion_order(self):
        with self.assertRaises(ProtocolError):
            privileged_context(self.context(), allow_privileged=False)
        missing = self.context()
        del missing['obj_pose_wrt_base']
        with self.assertRaises(ProtocolError):
            privileged_context(missing, allow_privileged=True)
        result = privileged_context(self.context(), allow_privileged=True)
        np.testing.assert_array_equal(result, [1, 2, 3, 1, 4, 5, 6, 1, 0, 0, 0, 7, 8, 9, 1, 0, 0, 0])

    def test_pointcloud_rejects_wrong_shape_rgb_and_nan(self):
        self.assertEqual(validate_pointcloud(np.zeros((1024, 6))).shape, (1024, 6))
        for value in [np.zeros((1023, 6)), np.full((1024, 6), np.nan),
                      np.full((1024, 6), 255)]:
            with self.assertRaises(ProtocolError):
                validate_pointcloud(value)

    def test_clipping_is_explicit_whole_body_and_does_not_mutate_labels(self):
        raw = np.zeros((2, 13), dtype=np.float32)
        raw[0, 7], raw[1, 11], raw[1, 12] = -20, .37, 2
        chunk = controller_chunk(raw, [-1]*13, [1]*13)
        self.assertEqual(chunk.clipped_indices, ((0, 7), (1, 12)))
        self.assertEqual(raw[0, 7], -20)
        self.assertEqual(chunk.executed[0, 7], -1)
        self.assertAlmostEqual(chunk.executed[1, 11], .37)
        for bad in [np.zeros((1, 13)), np.zeros((2, 7)), np.full((2, 13), np.inf)]:
            with self.assertRaises(ProtocolError):
                controller_chunk(bad, [-1]*13, [1]*13)

    def test_original_unicode_instruction_reaches_encoder_and_hashes_bytes(self):
        seen = []
        boundary = NativeCallBoundary(lambda text: seen.append(text) or 'embedding')
        text = '  抓住苹果，lift it gently。'
        boundary.begin('a', text, control_owner='acdit_whole_body')
        self.assertEqual(seen, [text])
        self.assertEqual(boundary.instruction, text)
        self.assertEqual(boundary.embedding, 'embedding')
        self.assertEqual(len(boundary.instruction_sha256), 64)

    def test_switch_and_interrupt_drop_queued_and_stale_commands(self):
        boundary = NativeCallBoundary(lambda text: text)
        boundary.begin('a', 'pick apple', control_owner='acdit_whole_body')
        boundary.enqueue('a', np.ones((2, 13)), [-1]*13, [1]*13)
        boundary.pop()
        boundary.begin('b', 'place apple', control_owner='acdit_whole_body')
        with self.assertRaises(ProtocolError):
            boundary.pop()
        with self.assertRaises(ProtocolError):
            boundary.enqueue('a', np.ones((2, 13)), [-1]*13, [1]*13)
        boundary.interrupt()
        self.assertIsNone(boundary.embedding)

    def test_native_model_cannot_impersonate_tapt_or_share_control(self):
        boundary = NativeCallBoundary(lambda text: text)
        for kwargs in [dict(control_owner='lightnav'),
                       dict(control_owner='acdit_whole_body', tool_family='grasp')]:
            with self.assertRaises(ProtocolError):
                boundary.begin('a', 'pick apple', **kwargs)
        with self.assertRaises(ProtocolError):
            boundary.begin('a', '\ud800', control_owner='acdit_whole_body')


if __name__ == '__main__':
    unittest.main()
