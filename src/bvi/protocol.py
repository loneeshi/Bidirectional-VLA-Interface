"""Versioned boundary between the coordinator, skill implementations and simulator.

Policy observations may contain benchmark privileges. They are deliberately not
serialized into VLM prompts. Targets/admissible calls are adapter-provided and
must be declared as oracle metadata when derived from a benchmark task plan.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

PROTOCOL_VERSION = "0.1"
FETCH_ACTION_DIM = 13


class ProtocolError(ValueError):
    """An invocation or action cannot be executed under the current contract."""


class SkillStatus(str, Enum):
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    INTERRUPTED = "interrupted"
    REJECTED = "rejected"


class RequirementState(str, Enum):
    SATISFIED = "satisfied"
    UNSATISFIED = "unsatisfied"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Requirement:
    id: str
    predicate: str


@dataclass(frozen=True)
class RequirementResult:
    requirement_id: str
    state: RequirementState
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImageFrame:
    """Actual encoded camera image, not a filename masquerading as vision input."""
    camera: str
    data: bytes
    media_type: str = "image/png"

    def validate(self) -> None:
        signatures = {"image/png": b"\x89PNG\r\n\x1a\n", "image/jpeg": b"\xff\xd8\xff"}
        signature = signatures.get(self.media_type)
        if not self.camera or not signature or not self.data.startswith(signature):
            raise ProtocolError("An encoded PNG/JPEG camera image is required")


@dataclass(frozen=True)
class Target:
    id: str
    description: str
    source: str = "unspecified"


@dataclass(frozen=True)
class AllowedCall:
    skill: str
    target_id: str


@dataclass(frozen=True)
class Observation:
    frame_id: str
    sim_step: int
    policy: Any = field(default=None, repr=False, compare=False)
    images: tuple[ImageFrame, ...] = field(default=(), repr=False)
    targets: tuple[Target, ...] = ()
    allowed_calls: tuple[AllowedCall, ...] = ()
    task: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillRequest:
    call_id: str
    skill: str
    target_id: str
    observation_id: str
    requirements: tuple[Requirement, ...]
    max_steps: int = 500
    timeout_seconds: float = 120.0


@dataclass(frozen=True)
class SkillSpec:
    name: str
    supported_requirements: tuple[str, ...] = ("benchmark_success",)
    required_requirements: tuple[str, ...] = ("benchmark_success",)
    max_steps: int = 500
    timeout_seconds: float = 120.0
    # Reserving all joints permits whole-body low-level policies. Serial skills
    # do not imply that manipulation locks the base.
    control_resources: tuple[str, ...] = ("arm", "gripper", "head", "torso", "base")


@dataclass(frozen=True)
class SkillFeedback:
    status: SkillStatus
    requirements: tuple[RequirementResult, ...] = ()
    reason: str | None = None
    source: str = "unspecified"


@dataclass(frozen=True)
class Transition:
    observation: Observation
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillResult:
    request: SkillRequest
    feedback: SkillFeedback
    steps: int
    elapsed_seconds: float
    observation: Observation


@dataclass(frozen=True)
class ActionBounds:
    """Use the installed env's external Box; never substitute physical ranges."""
    low: tuple[float, ...]
    high: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.low) != FETCH_ACTION_DIM or len(self.high) != FETCH_ACTION_DIM:
            raise ProtocolError("MS-HAB Fetch requires exactly 13 action bounds")
        if any(not math.isfinite(v) for v in (*self.low, *self.high)):
            raise ProtocolError("Action bounds must be finite")
        if any(lo > hi for lo, hi in zip(self.low, self.high)):
            raise ProtocolError("Action low exceeds high")

    def validate(self, action: Sequence[float]) -> tuple[float, ...]:
        try:
            if len(action) != FETCH_ACTION_DIM:
                raise ProtocolError("Expected exactly 13 action values")
            values = tuple(float(v) for v in action)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ProtocolError("Action must be a flat sequence of 13 numbers") from exc
        if any(not math.isfinite(v) for v in values):
            raise ProtocolError("Action contains NaN or infinity")
        if any(v < lo or v > hi for v, lo, hi in zip(values, self.low, self.high)):
            raise ProtocolError("Action outside environment Box; values are not clipped")
        return values


class Environment(Protocol):
    action_bounds: ActionBounds

    def observe(self) -> Observation: ...

    def step(self, action: tuple[float, ...]) -> Transition: ...


class Skill(Protocol):
    def start(self, request: SkillRequest, observation: Observation) -> None: ...

    def act(self, observation: Observation) -> Sequence[float]: ...

    def feedback(self, request: SkillRequest, transition: Transition) -> SkillFeedback: ...


def validate_request(request: SkillRequest, observation: Observation,
                     specs: Mapping[str, SkillSpec]) -> None:
    if not request.call_id or not request.observation_id:
        raise ProtocolError("Call ID and observation ID are required")
    if request.observation_id != observation.frame_id:
        raise ProtocolError("Invocation refers to a stale observation")
    if request.skill not in specs:
        raise ProtocolError(f"Unknown skill: {request.skill}")
    target_ids = [target.id for target in observation.targets]
    if len(target_ids) != len(set(target_ids)):
        raise ProtocolError("Observation contains ambiguous duplicate target IDs")
    if request.target_id not in set(target_ids):
        raise ProtocolError(f"Unknown target: {request.target_id}")
    if AllowedCall(request.skill, request.target_id) not in observation.allowed_calls:
        raise ProtocolError("Skill/target pair is not currently admissible")
    spec = specs[request.skill]
    if type(request.max_steps) is not int or not 0 < request.max_steps <= spec.max_steps:
        raise ProtocolError("Step budget exceeds the skill contract")
    if (type(request.timeout_seconds) not in (float, int)
            or not math.isfinite(request.timeout_seconds)
            or not 0 < request.timeout_seconds <= spec.timeout_seconds):
        raise ProtocolError("Wall-clock budget exceeds the skill contract")
    ids = [item.id for item in request.requirements]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ProtocolError("Requirement IDs must be nonempty and unique")
    predicates = {item.predicate for item in request.requirements}
    if not predicates <= set(spec.supported_requirements):
        raise ProtocolError("Invocation contains unsupported requirements")
    if not set(spec.required_requirements) <= predicates:
        raise ProtocolError("Invocation cannot weaken mandatory completion requirements")


def validate_feedback(request: SkillRequest, feedback: SkillFeedback) -> None:
    if not isinstance(feedback.status, SkillStatus):
        raise ProtocolError("Feedback status must be SkillStatus")
    expected = {item.id for item in request.requirements}
    actual = [item.requirement_id for item in feedback.requirements]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ProtocolError("Feedback must address every requirement exactly once")
    for item in feedback.requirements:
        if not isinstance(item.state, RequirementState):
            raise ProtocolError("Requirement state must be RequirementState")
        if item.state is RequirementState.SATISFIED and not item.evidence:
            raise ProtocolError("Satisfied requirements require evidence references")
    if feedback.status is SkillStatus.SUCCEEDED and any(
        item.state is not RequirementState.SATISFIED for item in feedback.requirements
    ):
        raise ProtocolError("Success requires verified satisfaction of every requirement")
