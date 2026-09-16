"""Auditable AC-DiT native MS-HAB boundary; no trained TAPT adapters implied.

Matches PKU-HMI-Lab/AC-DiT@90ad00a. The upstream policy consumes physical
proprioception and predicts normalized Fetch controller commands. Neither uses
the pi05 quantile transform. This module never changes the legacy pi05 runner.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Callable

import numpy as np

from .protocol import ProtocolError

SOURCE_COMMIT = "90ad00a926f34da04816ed9c3312aaf3bc845b7f"
UNIFIED_INDICES = (0, 1, 2, 3, 4, 5, 6, 10, 125, 126, 127, 100, 102)
ARM_JOINTS = (
    "shoulder_pan_joint", "shoulder_lift_joint", "upperarm_roll_joint",
    "elbow_flex_joint", "forearm_roll_joint", "wrist_flex_joint", "wrist_roll_joint",
)
STATE_JOINTS = ARM_JOINTS + (
    "r_gripper_finger_joint", "head_pan_joint", "head_tilt_joint", "torso_lift_joint",
)
CONTEXT_FIELDS = (("goal_pos_wrt_base", 3), ("is_grasped", 1),
                  ("obj_pose_wrt_base", 7), ("tcp_pose_wrt_base", 7))


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ProtocolError(f"{name}: expected finite {shape}, got {array.shape}")
    return array.copy()


def fetch_proprio(qpos, joint_names, base_linear_velocity_world, base_angular_velocity_world):
    """Name-based mapping accepts both full and rootless qpos, never guesses.

    Upstream training stores WORLD x velocity and WORLD z angular velocity.
    Do not silently substitute body-forward velocity to 'fix' this convention.
    The finger position is one physical joint, not doubled aperture or [-1,1].
    """
    names = tuple(joint_names)
    if len(set(names)) != len(names) or not set(STATE_JOINTS).issubset(names):
        raise ProtocolError("Missing/duplicate named Fetch joints")
    qpos = finite_array(qpos, (len(names),), "qpos")
    linear = finite_array(base_linear_velocity_world, (3,), "world linear velocity")
    angular = finite_array(base_angular_velocity_world, (3,), "world angular velocity")
    return np.array([qpos[names.index(name)] for name in STATE_JOINTS]
                    + [linear[0], angular[2]], dtype=np.float32)


def to_unified(native):
    native = finite_array(native, (13,), "native state/action")
    result = np.zeros(128, dtype=np.float32)
    result[list(UNIFIED_INDICES)] = native
    return result


def from_unified(unified):
    unified = np.asarray(unified, dtype=np.float32)
    if unified.ndim not in (1, 2) or unified.shape[-1] != 128 or not np.isfinite(unified).all():
        raise ProtocolError("Expected finite 128D vector or action chunk")
    return unified[..., list(UNIFIED_INDICES)].copy()


def privileged_context(extra, *, allow_privileged):
    """Preserve upstream vectorize_pose order; do not reorder quaternion fields."""
    if not allow_privileged:
        raise ProtocolError("Native AC-DiT requires declared simulator privileges")
    fields = []
    for name, dim in CONTEXT_FIELDS:
        if name not in extra:
            raise ProtocolError(f"Missing required privileged field: {name}")
        value = np.asarray(extra[name])
        if value.shape == (1, dim):
            value = value[0]
        if name == "is_grasped" and value.shape == ():
            value = value.reshape(1)
        fields.append(finite_array(value, (dim,), name))
    if fields[1][0] not in (0, 1):
        raise ProtocolError("is_grasped must be boolean/0/1")
    return np.concatenate(fields)


def validate_pointcloud(value):
    """1024 XYZRGB points in world coordinates, metres and RGB in [0,1].

    Caller must use the pinned upstream crop/sampling, not a new FPS sampler.
    Numeric shape checks alone cannot verify the coordinate frame or sampling.
    """
    cloud = finite_array(value, (1024, 6), "world XYZRGB")
    if np.any(cloud[:, 3:] < 0) or np.any(cloud[:, 3:] > 1):
        raise ProtocolError("Pointcloud RGB must already be in [0,1]")
    return cloud


@dataclass(frozen=True)
class ClippedChunk:
    raw: np.ndarray
    executed: np.ndarray
    clipped_indices: tuple[tuple[int, int], ...]


def controller_chunk(actions, low, high):
    """Single explicit environment-bound clipping; retain raw outputs for audit.

    This is not a training-label transform, and does not add a gripper threshold
    or freeze the base. The runtime must separately assert controller semantics.
    """
    raw = finite_array(actions, (2, 13), "native AC-DiT two-step action chunk")
    low, high = finite_array(low, (13,), "action low"), finite_array(high, (13,), "action high")
    if not np.array_equal(low, -np.ones(13)) or not np.array_equal(high, np.ones(13)):
        raise ProtocolError("Expected normalized Fetch bounds [-1,1] on all 13 channels")
    executed = np.clip(raw, low, high)
    indices = tuple(tuple(int(x) for x in row) for row in np.argwhere(raw != executed))
    return ClippedChunk(raw, executed, indices)


class NativeCallBoundary:
    """Instruction binding and queued native commands, before TAPT training.

    `encoder` must encode the supplied string with the checkpoint's SigLIP.
    This object supplies no fake family adapter and no simulated progress.
    A model server must use `embedding` in its real MSHabModel.step invocation.
    """
    def __init__(self, encoder: Callable):
        self.encoder = encoder
        self.pending = deque()
        self.call_id = None
        self.instruction = None
        self.embedding = None
        self.instruction_sha256 = None

    def begin(self, call_id, instruction, *, control_owner, tool_family=None):
        # Fail closed: even a rejected replacement cannot leave stale actions.
        self.interrupt()
        if control_owner != "acdit_whole_body":
            raise ProtocolError("AC-DiT must own all Fetch channels during manipulation")
        if tool_family is not None:
            raise ProtocolError("Native checkpoint has no trained TAPT family adapters")
        if not isinstance(call_id, str) or not call_id:
            raise ProtocolError("Nonempty call_id required")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ProtocolError("An explicit instruction is required")
        try:
            encoded = instruction.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ProtocolError("Instruction is not valid UTF-8") from exc
        if len(encoded) > 8192:
            raise ProtocolError("Instruction exceeds 8192 UTF-8 bytes")
        # Do not substitute a random stored language embedding or strip wording.
        embedding = self.encoder(instruction)
        if embedding is None:
            raise ProtocolError("Encoder returned no embedding")
        self.embedding = embedding
        self.call_id, self.instruction = call_id, instruction
        self.instruction_sha256 = hashlib.sha256(encoded).hexdigest()

    def enqueue(self, call_id, actions, low, high):
        if self.call_id is None or call_id != self.call_id:
            raise ProtocolError("Stale prediction from another invocation")
        if self.pending:
            raise ProtocolError("Previous action chunk has not been consumed")
        chunk = controller_chunk(actions, low, high)
        self.pending.extend(row.copy() for row in chunk.executed)
        return chunk

    def pop(self):
        if self.call_id is None or not self.pending:
            raise ProtocolError("No command for active invocation")
        return self.pending.popleft()

    def interrupt(self):
        self.pending.clear()
        self.call_id = self.instruction = self.embedding = self.instruction_sha256 = None
