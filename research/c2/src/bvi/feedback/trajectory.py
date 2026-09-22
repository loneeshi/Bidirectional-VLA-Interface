"""Invocation-local summaries from caller-supplied observations.

STATUS: active — feedback
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class InvocationDigest:
    skill: str
    target_id: str
    steps: int
    distance_start: float
    distance_end: float
    distance_min: float
    distance_at_min_step: int
    grasped_ever: bool
    stalled_steps: int
    end_reason: str
    source: str = 'simulator_policy_observation_geometry'
    distance_unit: str = 'm'


class TrajectoryAccumulator:
    def __init__(self, skill, target_id, snapshot, epsilon=1e-4):
        if not math.isfinite(epsilon) or epsilon <= 0:
            raise ValueError('Positive finite epsilon required')
        self.skill, self.target_id, self.epsilon = skill, target_id, epsilon
        self.start = self.end = self.minimum = self.distance(snapshot)
        self.steps = self.minimum_step = self.stalled = 0
        self.grasped = bool(snapshot['grasped'])

    @staticmethod
    def distance(snapshot):
        value = float(snapshot['distance'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('Distance must be finite and nonnegative')
        return value

    def update(self, snapshot):
        value = self.distance(snapshot)
        self.steps += 1
        self.stalled = self.stalled + 1 if abs(value-self.end) < self.epsilon else 0
        self.end = value
        if value < self.minimum:
            self.minimum, self.minimum_step = value, self.steps
        self.grasped |= bool(snapshot['grasped'])

    def finish(self, reason):
        return asdict(InvocationDigest(self.skill, self.target_id, self.steps,
            self.start, self.end, self.minimum, self.minimum_step, self.grasped,
            self.stalled, reason))


class TrajectorySkill:
    """Optional instrumentation around either feedback mode; no predicate reads."""
    def __init__(self, skill, adapter):
        self.skill, self.adapter = skill, adapter

    def start(self, request, observation):
        self.skill.start(request, observation)
        self.index = self.adapter.resolve_request_index(request)
        self.trace = TrajectoryAccumulator(request.skill, request.target_id,
                                          self.adapter.progress_snapshot(self.index))

    def act(self, observation):
        return self.skill.act(observation)

    def feedback(self, request, transition):
        feedback = self.skill.feedback(request, transition)
        self.trace.update(self.adapter.progress_snapshot(self.index))
        return feedback

    def invocation_digest(self, reason):
        return self.trace.finish(reason)
