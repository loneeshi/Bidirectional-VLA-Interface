import unittest
from bvi.tool_family import (
    FamilyInvocation,
    FamilySession,
    ProgressChunk,
    validate_instruction_length,
)


class Backend:
    def select_family(self, family):
        self.family = family
        return "a" * 64

    def infer(self, observation, **kwargs):
        self.received = kwargs
        return {
            "actions": [[0.0] * 7, [1.0] * 7],
            "progress": [0.2, 0.8],
            "progress_source": "learned",
        }


class FamilyTests(unittest.TestCase):
    def test_unicode_schema_limit_preserves_exact_instruction(self):
        text = "a" * 159 + "\uc5f4"
        self.assertEqual(len(text), 160)
        self.assertEqual(len(text.encode("utf-8")), 162)
        validate_instruction_length(text)
        self.session.start(FamilyInvocation("unicode", "reach", text, 10))
        self.session.act({})
        self.assertEqual(self.backend.received["instruction"], text)
        with self.assertRaisesRegex(ValueError, "Instruction too long"):
            validate_instruction_length(text, "legacy-bytes")

    def test_schema_limit_rejects_161_characters(self):
        with self.assertRaisesRegex(ValueError, "Instruction too long"):
            validate_instruction_length("a" * 161)

    def setUp(self):
        self.backend = Backend()
        self.events = []
        self.session = FamilySession(
            self.backend, lambda e, **kw: self.events.append((e, kw))
        )

    def test_instruction_and_family_reach_backend(self):
        text = "grasp the cream cheese box, leaving the butter alone"
        self.session.start(FamilyInvocation("one", "grasp", text, 10))
        self.session.act({})
        self.assertEqual(
            self.backend.received, {"instruction": text, "tool_family": "grasp"}
        )

    def test_interrupt_discards_old_actions(self):
        self.session.start(FamilyInvocation("one", "grasp", "grasp cheese", 10))
        self.session.act({})
        with self.assertRaises(RuntimeError):
            self.session.start(FamilyInvocation("two", "move", "move cheese", 10))
        self.session.finish("stagnation")
        self.session.start(FamilyInvocation("two", "move", "move cheese", 10))
        action, _ = self.session.act({})
        self.assertEqual(action, [0.0] * 7)

    def test_budget_is_enforced_inside_chunk(self):
        self.session.start(FamilyInvocation("one", "grasp", "grasp cheese", 1))
        self.session.act({})
        with self.assertRaises(RuntimeError):
            self.session.act({})
        self.assertFalse(self.session.queue)

    def test_invalid_family_and_progress(self):
        with self.assertRaises(ValueError):
            FamilyInvocation("one", "unknown", "grasp cheese", 10)
        for value in (float("nan"), -1.0, 2.0):
            with self.assertRaises(ValueError):
                ProgressChunk("one", (value,), "learned")

    def test_backend_failure_clears_control(self):
        self.backend.infer = lambda *a, **kw: {"actions": [[float("nan")] * 7]}
        self.session.start(FamilyInvocation("one", "grasp", "grasp cheese", 10))
        with self.assertRaises(ValueError):
            self.session.act({})
        self.assertIsNone(self.session.active)

    def test_call_ids_cannot_be_reused(self):
        call = FamilyInvocation("one", "grasp", "grasp cheese", 10)
        self.session.start(call)
        self.session.finish("completed")
        with self.assertRaises(ValueError):
            self.session.start(call)
