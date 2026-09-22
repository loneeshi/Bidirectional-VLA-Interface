"""Continuous goal-tool execution for the C0/C2 development experiment.

Native evaluator signals remain feedback/scoring evidence.  A native fail or
force violation is latched but does not by itself end simulator execution.

STATUS: active — evaluation
"""
from __future__ import annotations

from dataclasses import replace

from .goal_tools import GoalToolAdapter, GoalRLSkill, scalar
from .mshab_adapter import jsonable
from .protocol import (AllowedCall, ProtocolError, RequirementResult,
                       RequirementState, SkillFeedback, SkillStatus)


class RetryLedger:
    """Count full-horizon retries after an unsuccessful tool invocation."""

    def __init__(self, max_retries: int):
        if max_retries not in (0, 3):
            raise ValueError('C0 uses 0 retries; C2 uses 3')
        self.max_retries = max_retries
        self.navigation_retries = max_retries
        self.active = None
        self.attempted = set()
        self.completed = set()
        self.retries = {}

    def can_call(self, pair):
        if pair in self.completed:
            return False
        limit = self.navigation_retries if pair[0] == 'navigate' else self.max_retries
        return pair not in self.attempted or self.retries.get(pair, 0) < limit

    def begin(self, pair):
        if not self.can_call(pair):
            raise ProtocolError('Skill/target attempt allowance exhausted')
        if pair not in self.attempted:
            event = 'attempt_started'
        else:
            count = self.retries.get(pair, 0) + 1
            if count > self.max_retries:
                raise ProtocolError('Recovery retry limit exhausted')
            self.retries[pair] = count
            event = 'retried'
        self.active = pair
        return event

    def finish(self, pair, succeeded):
        self.attempted.add(pair)
        self.active = None
        if succeeded:
            self.completed.add(pair)


class ContinuationGoalAdapter(GoalToolAdapter):
    CONDITIONS = {'C0': 0, 'C2': 3}

    def __init__(self, adapter, condition, capability_prior=None, recovery_backend=None,
                 structured_recovery=False):
        if condition not in self.CONDITIONS:
            raise ValueError('Unknown continuation condition')
        if structured_recovery and (condition != 'C2' or capability_prior is not None):
            raise ValueError('Structured recovery is C2-only and does not consume spawn priors')
        if not structured_recovery and (condition == 'C2') != (capability_prior is not None):
            raise ValueError('Only C2 receives the frozen capability prior')
        super().__init__(adapter)
        self.condition = condition
        self.capability_prior = capability_prior
        self.retry_ledger = RetryLedger(self.CONDITIONS[condition])
        if structured_recovery:
            self.retry_ledger.navigation_retries = 0
        self.official_episode_valid = True
        self.force_violation_steps = 0
        self.native_fail_steps = 0
        self.ever_satisfied = set()
        self.recovery = None
        if condition == 'C2':
            from .pose_recovery import PoseRecovery
            if structured_recovery:
                from .pose_goal import GoalPoseRecovery
                self.recovery = GoalPoseRecovery(self, recovery_backend)
            else:
                self.recovery = PoseRecovery(self, capability_prior, recovery_backend)

    def begin_request(self, request):
        pair = (request.skill, request.target_id)
        if self.recovery is not None:
            if (request.skill in ('pick', 'place') and pair in self.retry_ledger.attempted
                    and not self.recovery.can_retry(pair, self.steps)):
                raise ProtocolError('SAC retry requires a fresh verified different pose')
            self.recovery.invalidate()
            self.recovery.record_sac_start(request)
        event = self.retry_ledger.begin(pair)
        self.logger.emit(event, call_id=request.call_id, skill=request.skill,
                         target_id=request.target_id,
                         retry_number=self.retry_ledger.retries.get(pair, 0))

    def finish_request(self, request, succeeded):
        if request.skill == 'reposition' and self.recovery is not None:
            self.recovery.finish(request, succeeded)
            return
        pair = (request.skill, request.target_id)
        self.retry_ledger.finish(pair, succeeded)
        if succeeded:
            self.ever_satisfied.add(pair)

    def admit_request(self, request):
        if request.skill == 'reposition' and hasattr(self.recovery, 'prepare'):
            self.recovery.prepare(request)

    def view(self, observation):
        viewed = super().view(observation)
        calls = tuple(call for call in viewed.allowed_calls
                      if self.retry_ledger.can_call((call.skill, call.target_id)))
        recovery_state = None
        if self.recovery is not None:
            # Keep the active SAC invocation admissible while it takes actions;
            # consume its ticket only once in begin_request.
            calls = tuple(c for c in calls if c.skill not in ('pick', 'place') or
                (c.skill, c.target_id) not in self.retry_ledger.attempted or
                self.retry_ledger.active == (c.skill, c.target_id) or
                self.recovery.can_retry((c.skill, c.target_id), observation.sim_step))
            recovery_state = self.recovery.snapshot()
            targets = {c['target_id'] for c in recovery_state['candidates']}
            targets.update(c['target_id'] for c in recovery_state.get('eligible_targets', []))
            if self.recovery.active is not None:
                targets.add(self.recovery.active['pair'][1])
            calls += tuple(AllowedCall('reposition', t) for t in sorted(targets))
        text = (f' Continuation condition {self.condition}: native subtask failure and force '
                'violations are recorded but do not stop this simulated episode. Each invocation '
                'runs until native completion or the official tool horizon: navigate 500 actions, '
                'pick/place 200 actions. The executor owns this horizon. Cover unfinished object '
                f'goals before spending recovery retries; at most {self.retry_ledger.max_retries} '
                'retries after an unsuccessful full-horizon invocation.')
        if self.recovery is not None and hasattr(self.recovery, 'prepare'):
            text += (' Navigation has ONE invocation per target, with no retries. '
                     'Manipulation has no independent in-place retry pool. Each of at most three '
                     'recovery rounds consists of reposition, then a separate GPT decision to '
                     'invoke SAC only after measured arrival. Failed preparations consume a round.')
        return replace(viewed, task=viewed.task + text, allowed_calls=calls,
                       metadata={**viewed.metadata, 'continuation_condition': self.condition,
                                 'native_scoring':'independent_final_object_predicates',
                                  'official_episode_valid': self.official_episode_valid,
                                  **({'recovery': recovery_state} if self.recovery is not None else {})})

    def step(self, action):
        transition = self.base.step(action)
        info = transition.info
        force_ok = bool(scalar(info.get('cumulative_force_within_limit', True)))
        failed = bool(scalar(info.get('fail', False)))
        self.force_violation_steps += int(not force_ok)
        self.native_fail_steps += int(failed)
        self.official_episode_valid &= force_ok and not failed
        whole_success = bool(scalar(info.get('success', False)))
        if self.base.ended and failed and not whole_success:
            if transition.terminated or transition.truncated:
                raise ProtocolError('Native wrapper terminated despite continuous-task configuration')
            self.base.ended = False
            self.base._observation = self.base._snapshot(transition.observation.policy)
            transition = replace(transition, observation=self.view(self.base._observation))
            self.logger.emit('native_failure_continued', frame_id=transition.observation.frame_id,
                             force_ok=force_ok, official_episode_valid=False)
            return transition
        return replace(transition, observation=self.view(transition.observation))

    def final_object_score(self):
        rows = []
        for goal in self.catalog.goals:
            index = self.catalog.bindings[('place', goal['destination_id'])]
            satisfied, checkers = self.predicate(index)
            rows.append({'object_id': goal['object_id'],
                         'destination_id': goal['destination_id'],
                         'satisfied': bool(scalar(satisfied)),
                         'checkers': checkers})
        return rows


class ContinuationGoalRLSkill(GoalRLSkill):
    """Evaluator feedback without native-fail early termination."""

    def start(self, request, observation):
        self.adapter.begin_request(request)
        super().start(request, observation)

    def feedback(self, request, transition):
        satisfied, checkers = self.adapter.predicate(self.selected_index)
        done = bool(scalar(satisfied))
        evidence=(f'events.jsonl:continuation_predicate:{request.call_id}:'
                  f'{transition.observation.frame_id}',)
        self.adapter.logger.emit('continuation_predicate', call_id=request.call_id,
            frame_id=transition.observation.frame_id, target_id=request.target_id,
            selected_index=self.selected_index, checkers=jsonable(checkers),
            native_fail=bool(scalar(transition.info.get('fail', False))))
        status = SkillStatus.SUCCEEDED if done else SkillStatus.EXECUTING
        state = RequirementState.SATISFIED if done else RequirementState.UNSATISFIED
        reason = 'requested_native_predicate' if done else None
        return SkillFeedback(status, tuple(RequirementResult(r.id,state,evidence)
                                            for r in request.requirements),
                             reason, 'native_requested_target_predicate')
