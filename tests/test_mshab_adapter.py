"""Offline checks of the adapter's semantic and oracle-feedback boundaries."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bvi import (Observation, Requirement, RequirementState, SkillRequest,
                 SkillStatus, Transition)
from bvi.mshab_adapter import (benchmark_feedback, describe_target,
                               load_checkpoint_config, scalar)


class MSHABBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.request = SkillRequest("test-call", "pick", "apple", "f0",
                                    (Requirement("done", "benchmark_success"),))

    def feedback(self, before=0, after=0, fail=False, truncated=False):
        transition = Transition(Observation("f1", 1), truncated=truncated,
            info={"adapter_subtask_before": before, "adapter_subtask_after": after,
                  "fail": [fail]})
        return benchmark_feedback(self.request, transition, 0)

    def test_only_one_environment_scalar_is_accepted(self):
        self.assertEqual(scalar([[3]]), 3)
        with self.assertRaises(ValueError):
            scalar([3, 4])

    def test_pointer_advance_is_oracle_success_with_evidence(self):
        feedback = self.feedback(after=1)
        self.assertEqual(feedback.status, SkillStatus.SUCCEEDED)
        self.assertEqual(feedback.source, "oracle_benchmark")
        self.assertEqual(feedback.requirements[0].state, RequirementState.SATISFIED)
        self.assertIn("f1", feedback.requirements[0].evidence[0])

    def test_benchmark_failure_wins_over_apparent_pointer_advance(self):
        feedback = self.feedback(after=1, fail=True)
        self.assertEqual(feedback.status, SkillStatus.FAILED)
        self.assertEqual(feedback.requirements[0].state, RequirementState.UNSATISFIED)

    def test_no_advance_is_not_success(self):
        self.assertEqual(self.feedback().status, SkillStatus.EXECUTING)
        self.assertEqual(self.feedback(truncated=True).status, SkillStatus.TIMED_OUT)

    def test_unexpected_pointer_jumps_are_not_counted_as_success(self):
        self.assertEqual(self.feedback(after=2).status, SkillStatus.FAILED)
        self.assertEqual(self.feedback(before=1, after=2).status, SkillStatus.FAILED)

    def test_navigation_target_describes_next_semantic_object_without_pose(self):
        plan = SimpleNamespace(subtasks=[
            SimpleNamespace(type="navigate", uid="nav-1"),
            SimpleNamespace(type="pick", uid="pick-1", obj_id="013_apple")])
        target = describe_target(plan, 0)
        self.assertEqual(target.source, "oracle_task_plan")
        self.assertIn("013_apple", target.description)
        self.assertEqual(target.id, "subtask-0-nav-1")

    def test_placement_target_preserves_identity_without_exposing_goal_coordinates(self):
        plan = SimpleNamespace(subtasks=[SimpleNamespace(type="place", uid="place-1",
            obj_id="024_bowl", goal_pos=[1.23, 4.56, 7.89])])
        target = describe_target(plan, 0)
        self.assertIn("024_bowl", target.description)
        self.assertNotIn("1.23", target.description)

    def test_checkpoint_inheritance_ignores_process_cli(self):
        class ConfigAPI:
            load = staticmethod(lambda path: json.loads(path.read_text()))
            create = staticmethod(dict)

            @staticmethod
            def merge(left, right):
                out = copy.deepcopy(left)
                for key, value in right.items():
                    if isinstance(value, dict) and isinstance(out.get(key), dict):
                        out[key] = ConfigAPI.merge(out[key], value)
                    else:
                        out[key] = copy.deepcopy(value)
                return out

        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory) / "base.json"
            base.write_text(json.dumps({"algo": {"name": "ppo", "hidden": [128]}}))
            child = Path(directory) / "config.json"
            child.write_text(json.dumps({"base_config": "base.json", "algo": {"hidden": [256]}}))
            with patch.object(sys, "argv", ["run_coordinator.py", "--dry-run", "--output", "/tmp/x",
                                           "algo.name=untrusted_cli_override"]):
                config = load_checkpoint_config(child, ConfigAPI)
            self.assertEqual(config["algo"], {"name": "ppo", "hidden": [256]})
            self.assertNotIn("--dry-run", config)
            base.write_text(json.dumps({"base_config": "config.json"}))
            with self.assertRaisesRegex(ValueError, "cycle"):
                load_checkpoint_config(child, ConfigAPI)


if __name__ == "__main__":
    unittest.main()
