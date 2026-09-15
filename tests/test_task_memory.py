import unittest

from bvi.task_memory import object_memory, transition_rejection


class TaskMemoryTests(unittest.TestCase):
    def test_old_release_survives_other_object_calls_without_claiming_success(self):
        history = [
            {
                "target": "cheese",
                "family": "release",
                "reason": "learned_threshold",
                "progress": 0.96,
                "source": "learned",
                "end_step": 135,
            }
        ]
        history += [{"target": "butter", "family": "reach", "reason": "step_limit"}] * 8
        memory = object_memory(history)
        self.assertEqual(memory["cheese"]["last_by_family"]["release"]["end_step"], 135)
        self.assertEqual(memory["cheese"]["native_goal"], "unknown")

    def test_low_progress_timeout_cannot_advance_but_can_relocalize(self):
        history = [
            {
                "target": "cheese",
                "family": "reach",
                "reason": "step_limit",
                "progress": 0.159,
            }
        ]
        self.assertIsNotNone(transition_rejection(history, "grasp", "cheese"))
        self.assertIsNone(transition_rejection(history, "reach", "cheese"))
        self.assertIsNone(transition_rejection(history, "grasp", "butter"))
        history.append(
            {
                "target": "cheese",
                "family": "reach",
                "reason": "learned_threshold",
                "progress": 0.98,
            }
        )
        self.assertIsNone(transition_rejection(history, "grasp", "cheese"))

    def test_rule_feedback_without_progress_is_not_misread_as_zero(self):
        history = [
            {
                "target": "cheese",
                "family": "reach",
                "reason": "step_limit",
                "progress": None,
            }
        ]
        self.assertIsNone(transition_rejection(history, "grasp", "cheese"))
