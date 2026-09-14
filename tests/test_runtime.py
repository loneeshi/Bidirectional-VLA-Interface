"""CPU contract tests. These do not claim navigation or manipulation success."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from bvi import (ActionBounds, AllowedCall, JsonlLogger, Observation, ProtocolError,
                 Requirement, RequirementResult, RequirementState, SerialRuntime,
                 SkillFeedback, SkillRequest, SkillSpec, SkillStatus, Target, Transition)


class FakeEnv:
    def __init__(self):
        self.action_bounds = ActionBounds((-1.0,) * 13, (1.0,) * 13)
        self.steps = 0
        self.actions = []
        self.terminated = False
        self.truncated = False
        self.change_admissibility = False

    def observe(self):
        pairs = () if self.change_admissibility and self.steps else (AllowedCall("pick", "cup"),)
        return Observation(f"f{self.steps}", self.steps,
                           targets=(Target("cup", "cup", "fake_fixture"),), allowed_calls=pairs)

    def step(self, action):
        self.actions.append(action)
        self.steps += 1
        return Transition(self.observe(), terminated=self.terminated, truncated=self.truncated)


class FakeSkill:
    def __init__(self, successful_at=2, action=None):
        self.successful_at = successful_at
        self.action = action if action is not None else [0.0] * 11 + [0.3, -0.2]
        self.starts = 0
        self.steps = 0
        self.force_unknown = False

    def start(self, request, observation):
        self.starts += 1
        self.steps = 0

    def act(self, observation):
        return self.action

    def feedback(self, request, transition):
        self.steps += 1
        succeeded = self.steps >= self.successful_at
        state = RequirementState.SATISFIED if succeeded else RequirementState.UNSATISFIED
        if self.force_unknown:
            state = RequirementState.UNKNOWN
        return SkillFeedback(SkillStatus.SUCCEEDED if succeeded else SkillStatus.EXECUTING,
            tuple(RequirementResult(r.id, state, (transition.observation.frame_id,))
                  for r in request.requirements), source="fake_fixture")


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.logger = JsonlLogger(Path(self.temp.name) / "events.jsonl", "fake-unit-test")
        self.env = FakeEnv()
        self.skill = FakeSkill()
        self.specs = {"pick": SkillSpec("pick", max_steps=10)}
        self.runtime = SerialRuntime(self.env, {"pick": self.skill}, self.specs, self.logger)
        self.request = SkillRequest("c1", "pick", "cup", "f0",
                                    (Requirement("r1", "benchmark_success"),), max_steps=5)

    def test_success_retains_base_control_and_logs_actual_actions(self):
        result = self.runtime.execute(self.request)
        self.assertEqual(result.feedback.status, SkillStatus.SUCCEEDED)
        self.assertEqual(result.steps, 2)
        self.assertEqual(self.env.actions[0][-2:], (0.3, -0.2))
        records = [json.loads(line) for line in self.logger.path.read_text().splitlines()]
        self.assertEqual([row["event"] for row in records],
                         ["skill_started", "skill_step", "skill_step", "skill_finished"])
        self.assertEqual(records[-1]["feedback"]["source"], "fake_fixture")

    def test_invalid_actions_never_reach_environment(self):
        for action in ([0.0] * 12, [0.0] * 14, [float("nan")] * 13,
                       [float("inf")] * 13, [1.01] * 13, [[0.0]] * 13):
            with self.subTest(action=action):
                env = FakeEnv()
                runtime = SerialRuntime(env, {"pick": FakeSkill(action=action)}, self.specs, self.logger)
                result = runtime.execute(self.request)
                self.assertEqual(result.feedback.status, SkillStatus.FAILED)
                self.assertTrue(result.feedback.reason.startswith("invalid_action"))
                self.assertEqual(env.steps, 0)

    def test_step_timeout_has_unknown_completion(self):
        self.skill.successful_at = 99
        result = self.runtime.execute(replace(self.request, max_steps=3))
        self.assertEqual(result.feedback.status, SkillStatus.TIMED_OUT)
        self.assertEqual(result.feedback.reason, "step_limit")
        self.assertEqual(self.env.steps, 3)
        self.assertEqual(result.feedback.requirements[0].state, RequirementState.UNKNOWN)

    def test_slow_policy_does_not_execute_late_action(self):
        now = [0.0]
        self.runtime.clock = lambda: now[0]
        def slow_act(observation):
            now[0] = 5.0
            return [0.0] * 13
        self.skill.act = slow_act
        result = self.runtime.execute(replace(self.request, timeout_seconds=1.0))
        self.assertEqual(result.feedback.status, SkillStatus.TIMED_OUT)
        self.assertEqual(self.env.steps, 0)

    def test_slow_environment_does_not_report_late_success(self):
        now = [0.0]
        self.runtime.clock = lambda: now[0]
        original_step = self.env.step
        def slow_step(action):
            now[0] = 5.0
            return original_step(action)
        self.env.step = slow_step
        self.skill.successful_at = 1
        result = self.runtime.execute(replace(self.request, timeout_seconds=1.0))
        self.assertEqual(result.feedback.status, SkillStatus.TIMED_OUT)
        self.assertEqual(self.env.steps, 1)

    def test_unknown_feedback_cannot_end_as_success(self):
        self.skill.successful_at = 1
        self.skill.force_unknown = True
        result = self.runtime.execute(self.request)
        self.assertEqual(result.feedback.status, SkillStatus.FAILED)
        self.assertEqual(result.feedback.reason, "adapter_error:ProtocolError")

    def test_stale_call_is_rejected_without_skill_start(self):
        self.env.steps = 2
        result = self.runtime.execute(self.request)
        self.assertEqual(result.feedback.status, SkillStatus.REJECTED)
        self.assertEqual(self.skill.starts, 0)

    def test_request_cannot_weaken_required_success(self):
        result = self.runtime.execute(replace(self.request, requirements=()))
        self.assertEqual(result.feedback.status, SkillStatus.REJECTED)
        self.assertEqual(self.env.steps, 0)

    def test_repeated_call_id_requires_explicit_new_retry(self):
        self.runtime.execute(self.request)
        result = self.runtime.execute(replace(self.request, observation_id="f2"))
        self.assertEqual(result.feedback.status, SkillStatus.REJECTED)
        self.assertEqual(self.env.steps, 2)

    def test_nested_skill_cannot_take_control(self):
        self.runtime._active = True
        with self.assertRaises(ProtocolError):
            self.runtime.execute(self.request)
        self.assertEqual(self.env.steps, 0)

    def test_environment_termination_without_success_is_failure(self):
        self.env.terminated = True
        result = self.runtime.execute(self.request)
        self.assertEqual(result.feedback.status, SkillStatus.FAILED)
        self.assertEqual(self.env.steps, 1)

    def test_environment_truncation_is_timeout(self):
        self.env.truncated = True
        result = self.runtime.execute(self.request)
        self.assertEqual(result.feedback.status, SkillStatus.TIMED_OUT)
        self.assertEqual(result.feedback.reason, "environment_truncated")

    def test_task_change_without_completion_stops_before_second_action(self):
        self.env.change_admissibility = True
        result = self.runtime.execute(self.request)
        self.assertEqual(result.feedback.reason, "admissibility_changed_without_completion")
        self.assertEqual(self.env.steps, 1)

    def test_runtime_releases_control_after_adapter_exception(self):
        def broken(observation):
            raise RuntimeError("not a valid policy")
        self.skill.act = broken
        first = self.runtime.execute(self.request)
        second = self.runtime.execute(replace(self.request, call_id="c2"))
        self.assertEqual(first.feedback.status, SkillStatus.FAILED)
        self.assertEqual(second.feedback.status, SkillStatus.FAILED)
        self.assertEqual(self.skill.starts, 2)

    def test_bounds_are_external_box_and_not_assumed_normalized(self):
        bounds = ActionBounds((-2.0,) * 13, (3.0,) * 13)
        self.assertEqual(bounds.validate([2.0] * 13), (2.0,) * 13)
        for low, high in (((0.0,) * 12, (1.0,) * 13),
                          ((2.0,) * 13, (1.0,) * 13)):
            with self.assertRaises(ProtocolError):
                ActionBounds(low, high)


if __name__ == "__main__":
    unittest.main()

