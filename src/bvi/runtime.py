"""One skill owns the complete action vector until it ends.

This synchronous simulator runtime stops stepping on failure/timeout. It does
not claim to stop a remote provider bill or implement a real-robot safety hold.
Blocking policy/environment calls must enforce their own I/O deadlines; the
wall-clock gate runs before and after each action-producing call.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import replace

from .logging import JsonlLogger
from .protocol import (AllowedCall, Environment, Observation, ProtocolError, RequirementResult,
                       RequirementState, Skill, SkillFeedback, SkillRequest,
                       SkillResult, SkillSpec, SkillStatus, validate_feedback,
                       validate_request)


class SerialRuntime:
    def __init__(self, env: Environment, skills: Mapping[str, Skill],
                 specs: Mapping[str, SkillSpec], logger: JsonlLogger,
                 clock: Callable[[], float] = time.monotonic):
        if set(skills) != set(specs):
            raise ProtocolError("Every registered skill requires exactly one specification")
        self.env, self.skills, self.specs, self.logger = env, skills, specs, logger
        self.clock = clock
        self._active = False
        self._seen_calls: set[str] = set()
        self.last_observation: Observation | None = None

    def execute(self, request: SkillRequest,
                observation: Observation | None = None) -> SkillResult:
        if self._active:
            raise ProtocolError("Serial runtime already has an active skill")
        # Always read current state, so a caller cannot bypass the freshness gate
        # by providing an old observation. observe() must be side-effect free.
        current = self.env.observe()
        self.last_observation = current
        started = self.clock()
        steps = 0
        last_feedback: SkillFeedback | None = None

        def terminalize(feedback: SkillFeedback | None, status: SkillStatus,
                        reason: str) -> SkillFeedback | None:
            if feedback is None:
                return None
            return replace(feedback, status=status, reason=reason,
                           requirements=tuple(RequirementResult(
                               item.requirement_id, RequirementState.UNKNOWN,
                               item.evidence) for item in feedback.requirements))

        def finish(status: SkillStatus, reason: str,
                   feedback: SkillFeedback | None = None) -> SkillResult:
            if feedback is None:
                feedback = SkillFeedback(status, tuple(
                    RequirementResult(r.id, RequirementState.UNKNOWN)
                    for r in request.requirements), reason, "runtime")
            result = SkillResult(request, feedback, steps, self.clock() - started, current)
            self.last_observation = current
            self.logger.emit("skill_finished", call_id=request.call_id,
                             status=feedback.status, feedback=feedback, steps=steps,
                             elapsed_seconds=result.elapsed_seconds, frame_id=current.frame_id)
            return result

        try:
            if observation is not None and observation.frame_id != current.frame_id:
                raise ProtocolError("Caller supplied a stale observation")
            validate_request(request, current, self.specs)
            if request.call_id in self._seen_calls:
                raise ProtocolError("Duplicate call ID; issue a new ID for an explicit retry")
        except ProtocolError as exc:
            return finish(SkillStatus.REJECTED, str(exc))

        self._seen_calls.add(request.call_id)
        self.logger.emit("skill_started", request=request,
                         control_resources=self.specs[request.skill].control_resources)
        self._active = True
        try:
            skill = self.skills[request.skill]
            skill.start(request, current)
            while steps < request.max_steps:
                if self.clock() - started >= request.timeout_seconds:
                    return finish(SkillStatus.TIMED_OUT, "wall_clock_limit")
                raw_action = skill.act(current)
                if self.clock() - started >= request.timeout_seconds:
                    return finish(SkillStatus.TIMED_OUT, "policy_exceeded_wall_clock_limit")
                try:
                    action = self.env.action_bounds.validate(raw_action)
                except ProtocolError as exc:
                    return finish(SkillStatus.FAILED, f"invalid_action: {exc}")
                transition = self.env.step(action)
                steps += 1
                previous_step = current.sim_step
                current = transition.observation
                if current.sim_step <= previous_step:
                    return finish(SkillStatus.FAILED, "nonmonotonic_environment_step")
                feedback = skill.feedback(request, transition)
                last_feedback = feedback
                validate_feedback(request, feedback)
                self.logger.emit("skill_step", call_id=request.call_id, step=steps,
                                 frame_id=current.frame_id, action=action,
                                 feedback=feedback, terminated=transition.terminated,
                                 truncated=transition.truncated)
                if self.clock() - started >= request.timeout_seconds:
                    return finish(SkillStatus.TIMED_OUT, "environment_exceeded_wall_clock_limit",
                                  terminalize(last_feedback, SkillStatus.TIMED_OUT,
                                              "environment_exceeded_wall_clock_limit"))
                if feedback.status is not SkillStatus.EXECUTING:
                    return finish(feedback.status, feedback.reason or "skill_ended", feedback)
                if transition.truncated:
                    return finish(SkillStatus.TIMED_OUT, "environment_truncated",
                                  terminalize(last_feedback, SkillStatus.TIMED_OUT,
                                              "environment_truncated"))
                if transition.terminated:
                    return finish(SkillStatus.FAILED, "environment_terminated_without_skill_success",
                                  terminalize(last_feedback, SkillStatus.FAILED,
                                              "environment_terminated_without_skill_success"))
                if AllowedCall(request.skill, request.target_id) not in current.allowed_calls:
                    return finish(SkillStatus.FAILED, "admissibility_changed_without_completion")
            return finish(SkillStatus.TIMED_OUT, "step_limit",
                          terminalize(last_feedback, SkillStatus.TIMED_OUT, "step_limit"))
        except Exception as exc:
            # Exception type is sufficient for public event logs; messages may
            # contain provider keys or private paths. Full traceback stays with
            # the application runner if it opts into debug logging.
            if os.getenv("BVI_DEBUG_ADAPTER_EXCEPTION") == "1":
                self.logger.emit("adapter_exception_debug", exception_type=type(exc).__name__,
                                 message=str(exc))
            return finish(SkillStatus.FAILED, f"adapter_error:{type(exc).__name__}")
        finally:
            self._active = False
