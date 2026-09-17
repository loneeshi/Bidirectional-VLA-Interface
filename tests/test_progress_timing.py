import unittest
from dataclasses import FrozenInstanceError

from bvi.progress_monitor import ProgressMonitor
from bvi.progress_timing import PostActionProgressGate, current_observation_progress
from bvi.progress_timing import require_post_action_contract, LEGACY_FETCH_CHECKPOINT_SHA256


class ProgressTimingTest(unittest.TestCase):
    def setUp(self):
        self.monitor = ProgressMonitor("reach", 1)
        self.gate = PostActionProgressGate(self.monitor)

    def test_author_observation_targets(self):
        self.assertEqual(current_observation_progress(0, 5), 0)
        self.assertEqual(current_observation_progress(2, 5), 0.5)
        self.assertEqual(current_observation_progress(2, 5, 2), 1)
        for args in [(0, 1), (-1, 5), (4, 5, 1), (0.5, 5), (0, 5, -1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                current_observation_progress(*args)

    def test_checkpoint_semantics_cannot_be_silently_reinterpreted(self):
        self.assertEqual(require_post_action_contract({}, LEGACY_FETCH_CHECKPOINT_SHA256), 'post_action_position_v1')
        self.assertEqual(require_post_action_contract({'progress_label_contract':'post_action_position_v1'}, 'new'), 'post_action_position_v1')
        for report, digest in [({}, 'unknown'), ({'progress_label_contract':'current_observation_v2'}, LEGACY_FETCH_CHECKPOINT_SHA256)]:
            with self.assertRaises(ValueError):
                require_post_action_contract(report, digest)

    def test_action_sequence_threshold_interrupts_remaining_actions(self):
        # Two predictions, each with a three-action chunk; stop only after the
        # action corresponding to the second prediction's first progress value.
        executed = []
        step = 0
        for call in ("first", "second"):
            self.gate.stage(call, step, [0.95, 0.98, 1.0])
            history = list(self.monitor.history)
            self.assertIsNone(self.gate.observe(call, step))
            self.assertEqual(self.monitor.history, history)
            for action in range(3):
                executed.append((call, action))
                step += 1
                reason = self.gate.observe(call, step) if self.gate.pending else None
                if reason:
                    break
        self.assertEqual(executed, [("first", 0), ("first", 1), ("first", 2), ("second", 0)])
        self.assertEqual(reason, "learned_threshold")
        self.assertEqual(self.monitor.history, [0.95, 0.95])
        self.assertTrue(self.gate.consumed)
        self.assertIsNone(self.gate.due_step)

    def test_drop_waits_until_action(self):
        for step, value in enumerate([0.2, 0.3, 0.4]):
            self.gate.stage("call", step, [value])
            self.gate.observe("call", step + 1)
        self.gate.stage("call", 3, [0.2])
        self.assertIsNone(self.gate.observe("call", 3))
        self.assertEqual(self.monitor.history, [0.2, 0.3, 0.4])
        self.assertEqual(self.gate.observe("call", 4), "learned_drop")

    def test_bad_observations_cannot_advance_monitor(self):
        self.gate.stage("call", 4, [0.95])
        for call, step in [("wrong", 5), ("call", 3), ("call", 6), ("call", 4.5)]:
            with self.subTest(call=call, step=step), self.assertRaises(ValueError):
                self.gate.observe(call, step)
            self.assertEqual(self.monitor.history, [])
            self.assertEqual(self.gate.due_step, 5)
        self.gate.observe("call", 5)
        with self.assertRaises(ValueError):
            self.gate.observe("call", 5)
        for step in (3, 4):
            with self.assertRaises(ValueError):
                self.gate.stage("call", step, [0.95])
        self.assertEqual(self.monitor.history, [0.95])

    def test_pending_is_immutable_and_cannot_be_overwritten(self):
        values = [0.2, 0.8]
        self.gate.stage("call", 0, values)
        values[0] = 1
        self.assertEqual(self.gate.pending.values, (0.2, 0.8))
        with self.assertRaises(FrozenInstanceError):
            self.gate.pending.prediction_step = 9
        with self.assertRaises(ValueError):
            self.gate.stage("other", 1, [0.99])
        self.gate.observe("call", 1)
        self.assertEqual(self.monitor.history, [0.2])

    def test_invalid_vectors_fail_closed(self):
        for values in ([], None, 0.5, "0.5", [[0.5]], [True], [complex(0.5)],
                       [float("nan")], [float("inf")], [-0.1], [1.1],
                       [0.5, float("nan")]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.gate.stage("call", 0, values)
            self.assertIsNone(self.gate.pending)
            self.assertEqual(self.monitor.history, [])
        self.gate.stage("valid", 0, [0.5])
        self.gate.observe("valid", 1)

    def test_native_termination_cancels_future_progress(self):
        self.gate.stage("call", 0, [0.99])
        self.gate.cancel()  # Native termination after action wins over progress.
        with self.assertRaises(ValueError):
            self.gate.observe("call", 1)
        self.assertEqual(self.monitor.history, [])
        self.assertFalse(self.gate.consumed)
        self.assertIsNone(self.gate.due_step)
        self.gate.reset()
        self.gate.stage("new", 0, [0.1])
        self.gate.observe("new", 1)
        self.assertEqual(self.monitor.history, [0.1])


if __name__ == "__main__":
    unittest.main()
def test_current_contract_cannot_reinterpret_legacy_weights():
    from bvi.progress_timing import require_progress_contract, LEGACY_FETCH_CHECKPOINT_SHA256
    import pytest
    assert require_progress_contract({'progress_label_contract':'current_observation_v2'}, 'new', 'current_observation_v2') == 'current_observation_v2'
    with pytest.raises(ValueError):
        require_progress_contract({}, LEGACY_FETCH_CHECKPOINT_SHA256, 'current_observation_v2')
    with pytest.raises(ValueError):
        require_progress_contract({'progress_label_contract':'current_observation_v2'}, 'new', 'post_action_v1')
