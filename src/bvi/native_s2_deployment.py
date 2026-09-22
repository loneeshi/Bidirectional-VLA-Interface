"""Reusable native24 S2 deployment adapter, independent of any transport.

STATUS: frozen — historical training and diagnostics (retained)

The adapter owns request validation, explicit family/bank selection, queue
invalidation, preprocessing, response postprocessing and response identities.
A future Unix-socket/WebSocket service should wrap this class rather than
reimplement those semantics.  Model construction and JIT compilation stay in
the executable and are injected as callables, which keeps this module CPU
testable and makes transport coverage explicit rather than implied.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Callable, Mapping

import numpy as np


FAMILIES = ("reach", "grasp", "move", "release")


class NativeS2DeploymentAdapter:
    """Transport-free production boundary for native24 family inference."""

    def __init__(
        self,
        *,
        transform: Callable[[Mapping], Mapping],
        observation_factory: Callable[[Mapping], object],
        predict: Callable[[str, object, object, object], tuple[object, object]],
        unnormalize: Callable[[Mapping], Mapping],
        observation_digest: Callable[[object], str],
        bank_sha256: Mapping[str, str],
        head_sha256: str,
        checkpoint_sha256: str,
        normalizer_sha256: str,
        state_contract_sha256: str,
    ):
        if set(bank_sha256) != set(FAMILIES):
            raise ValueError("Exactly four family bank identities required")
        identities = [*bank_sha256.values(), head_sha256, checkpoint_sha256,
                      normalizer_sha256, state_contract_sha256]
        if any(not isinstance(value, str) or len(value) != 64 for value in identities):
            raise ValueError("Deployment identities must be SHA256 strings")
        self.transform = transform
        self.observation_factory = observation_factory
        self.predict = predict
        self.unnormalize = unnormalize
        self.observation_digest = observation_digest
        self.bank_sha256 = dict(bank_sha256)
        self.head_sha256 = head_sha256
        self.checkpoint_sha256 = checkpoint_sha256
        self.normalizer_sha256 = normalizer_sha256
        self.state_contract_sha256 = state_contract_sha256
        self.family = None
        self.call_id = None
        self.instruction = None
        self.pending = deque()
        self.seen = set()

    def begin(self, *, call_id: str, family: str, instruction: str) -> dict:
        if not isinstance(call_id, str) or not call_id or call_id in self.seen:
            raise ValueError("Fresh nonempty call_id required")
        if family not in FAMILIES:
            raise ValueError("Unknown family")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Nonempty instruction required")
        discarded = len(self.pending)
        previous = self.family
        self.pending.clear()
        self.call_id, self.family, self.instruction = call_id, family, instruction
        self.seen.add(call_id)
        return {
            "previous_family": previous,
            "selected_family": family,
            "selected_bank_sha256": self.bank_sha256[family],
            "discarded_actions": discarded,
            "queue_empty_before_inference": not self.pending,
        }

    @staticmethod
    def _validate_request(request: Mapping) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
        if not isinstance(request, Mapping):
            raise ValueError("Deployment request must be a mapping")
        head = np.asarray(request.get("head_rgb"))
        wrist = np.asarray(request.get("wrist_rgb"))
        state = np.asarray(request.get("state"), dtype=np.float32)
        family, call_id, prompt = request.get("tool_family"), request.get("call_id"), request.get("prompt")
        if head.shape != (128, 128, 3) or wrist.shape != (128, 128, 3):
            raise ValueError("Deployment request requires two 128x128 RGB images")
        if head.dtype != np.uint8 or wrist.dtype != np.uint8:
            raise ValueError("Deployment RGB inputs must be uint8")
        if state.shape != (24,) or not np.isfinite(state).all():
            raise ValueError("Deployment request requires finite state24")
        if family not in FAMILIES or not isinstance(call_id, str) or not call_id:
            raise ValueError("Deployment request family/call identity invalid")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Deployment request prompt is required")
        return head, wrist, state, family, call_id, prompt

    def infer(self, request: Mapping, *, rng_key, noise) -> dict:
        head, wrist, state, family, call_id, prompt = self._validate_request(request)
        if (call_id, family, prompt) != (self.call_id, self.family, self.instruction):
            raise ValueError("Request differs from active call/family/instruction binding")
        if self.pending:
            raise ValueError("Replan requires the previous deployment queue to be cleared")
        transformed = self.transform({
            "observation/image": head,
            "observation/wrist_image": wrist,
            "observation/state": state,
            "prompt": prompt,
        })
        observation = self.observation_factory(transformed)
        normalized, progress = self.predict(family, rng_key, observation, noise)
        normalized = np.asarray(normalized)
        progress = np.asarray(progress)
        if normalized.shape != (1, 10, 32) or progress.shape != (1, 10):
            raise ValueError("Deployment model returned an invalid chunk shape")
        if not np.isfinite(normalized).all() or not np.isfinite(progress).all():
            raise ValueError("Deployment model returned nonfinite values")
        external = np.asarray(self.unnormalize({
            "actions": normalized[0], "state": np.asarray(transformed["state"]),
        })["actions"][:, :13])
        if external.shape != (10, 13) or not np.isfinite(external).all():
            raise ValueError("Deployment response requires finite Fetch13 chunks")
        self.pending.extend(external.tolist())
        return {
            "status": "ok",
            "call_id": call_id,
            "tool_family": family,
            "instruction": prompt,
            "normalized_actions": normalized,
            "actions": external,
            "progress": progress,
            "progress_source": "learned_chunk_prefix",
            "observation_sha256": self.observation_digest(observation),
            "adapter_sha256": self.bank_sha256[family],
            "progress_head_sha256": self.head_sha256,
            "checkpoint_sha256": self.checkpoint_sha256,
            "normalizer_sha256": self.normalizer_sha256,
            "state_contract_sha256": self.state_contract_sha256,
            "controller_clip_fraction": float(np.mean(np.abs(external) > 1.0)),
            "controller_clip_max_delta": float(np.max(np.abs(external - np.clip(external, -1.0, 1.0)))),
            "queued_actions": len(self.pending),
            "transport_evaluated": False,
        }

    def clear(self, reason: str) -> dict:
        if not isinstance(reason, str) or not reason:
            raise ValueError("Queue clear reason required")
        discarded = len(self.pending)
        self.pending.clear()
        return {"call_id": self.call_id, "family": self.family, "reason": reason,
                "discarded_actions": discarded, "queue_empty": not self.pending}
