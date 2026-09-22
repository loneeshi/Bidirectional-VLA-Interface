"""Paper-style invocation/progress contract, isolated from legacy MS-HAB v0.1.

STATUS: frozen — historical training and diagnostics (retained)

This contract does not itself implement a trained adapter or progress predictor.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import math

VERSION = "vla-tool-family/1"
FAMILIES = ("reach", "grasp", "move", "release")


def validate_instruction_length(text, mode="schema-characters"):
    """Match JSON Schema maxLength without rewriting the model's instruction.

    legacy-bytes is retained only to reproduce run01's stricter, flawed guard.
    The separate API request-envelope byte budget still applies in either mode.
    """
    if mode not in ("schema-characters", "legacy-bytes"):
        raise ValueError("Unknown instruction limit mode")
    size = len(text.encode("utf-8")) if mode == "legacy-bytes" else len(text)
    if size > 160:
        raise ValueError("Instruction too long")


@dataclass(frozen=True)
class FamilyInvocation:
    call_id: str
    tool_family: str
    instruction: str
    max_steps: int
    version: str = VERSION

    def __post_init__(self):
        if self.version != VERSION or self.tool_family not in FAMILIES:
            raise ValueError("Unsupported protocol version or tool family")
        if not isinstance(self.call_id, str) or not self.call_id.strip():
            raise ValueError("call_id is required")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("A grounded instruction is required")
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 520:
            raise ValueError("Invocation budget must be 1..520")


@dataclass(frozen=True)
class ProgressChunk:
    call_id: str
    values: tuple[float, ...]
    source: str

    def __post_init__(self):
        if self.source not in ("learned", "simulator", "heuristic"):
            raise ValueError("Explicit progress provenance is required")
        if not self.values or any(
            not math.isfinite(x) or not 0 <= x <= 1 for x in self.values
        ):
            raise ValueError("Progress must be finite and bounded")


class FamilySession:
    """One owner of control; invalid requests never activate a fallback policy."""

    def __init__(self, backend, emit):
        self.backend, self.emit = backend, emit
        self.active = None
        self.queue = deque()
        self.steps = 0
        self.seen_ids = set()

    def start(self, invocation: FamilyInvocation):
        if self.active is not None:
            raise RuntimeError(
                "Finish or interrupt the active invocation before switching"
            )
        if invocation.call_id in self.seen_ids:
            raise ValueError("Invocation IDs cannot be reused")
        self.queue.clear()
        digest = self.backend.select_family(invocation.tool_family)
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError("Backend must return selected adapter SHA256")
        self.seen_ids.add(invocation.call_id)
        self.active, self.steps = invocation, 0
        self.emit(
            "family_selected",
            call_id=invocation.call_id,
            family=invocation.tool_family,
            instruction=invocation.instruction,
            adapter_sha256=digest,
            protocol=VERSION,
        )

    def act(self, observation):
        if self.active is None:
            raise RuntimeError("No active invocation")
        if self.steps >= self.active.max_steps:
            self.finish("step_limit")
            raise RuntimeError("Invocation exhausted")
        if not self.queue:
            try:
                result = self.backend.infer(
                    observation,
                    instruction=self.active.instruction,
                    tool_family=self.active.tool_family,
                )
                actions = result["actions"]
                if not len(actions) or any(
                    len(a) != 7 or any(not math.isfinite(float(v)) for v in a)
                    for a in actions
                ):
                    raise ValueError("Expected finite LIBERO 7D actions")
                progress = ProgressChunk(
                    self.active.call_id,
                    tuple(result["progress"]),
                    result["progress_source"],
                )
                if len(progress.values) != len(actions):
                    raise ValueError("Action and progress chunks must align")
                self.queue.extend(zip(actions, progress.values))
                self.emit(
                    "progress_chunk",
                    call_id=progress.call_id,
                    values=progress.values,
                    source=progress.source,
                )
            except Exception:
                self.finish("backend_error")
                raise
        action, predicted_progress = self.queue.popleft()
        self.steps += 1
        return action, predicted_progress

    def finish(self, reason):
        if self.active is not None:
            self.emit(
                "invocation_finished",
                call_id=self.active.call_id,
                reason=reason,
                steps=self.steps,
            )
        self.queue.clear()
        self.active = None
