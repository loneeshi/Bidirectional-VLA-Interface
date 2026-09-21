"""Pure helpers for the frozen S1-IA D1/D2 diagnostic cards."""

from __future__ import annotations

import math
import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


GRIPPER_INDEX = 7
ACTION_DIM = 13
GRIPPER_LOWER_M = -0.01
GRIPPER_UPPER_M = 0.05
FORCE_LIMIT = 5000.0


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(value):
    """Convert nested numpy/torch evidence to JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalized_gripper_target(
    position_m: float,
    lower_m: float = GRIPPER_LOWER_M,
    upper_m: float = GRIPPER_UPPER_M,
) -> float:
    """Map an absolute mimic-controller target to its normalized action."""
    if not math.isfinite(position_m) or not lower_m < upper_m:
        raise ValueError("Invalid gripper position or controller bounds")
    return float(np.clip(2.0 * (position_m - lower_m) / (upper_m - lower_m) - 1.0, -1.0, 1.0))


def fixed_action(mode: str, initial_finger_qpos: Sequence[float]) -> np.ndarray:
    """Construct one frozen action for zero, pose-hold, or constant-close."""
    action = np.zeros(ACTION_DIM, dtype=np.float32)
    if mode == "zero":
        return action
    if mode == "close":
        action[GRIPPER_INDEX] = -1.0
        return action
    if mode != "hold":
        raise ValueError(f"Unknown fixed-action mode: {mode}")
    fingers = np.asarray(initial_finger_qpos, dtype=np.float64).reshape(-1)
    if fingers.size != 2 or not np.isfinite(fingers).all():
        raise ValueError("Expected two finite initial finger positions")
    action[GRIPPER_INDEX] = normalized_gripper_target(float(fingers.mean()))
    return action


def perturbation_targets(start_distance_m: float, offsets_m: Iterable[float] = (0.03, 0.05)) -> list[float]:
    """Return the preregistered +3/+5 cm targets, both inside the 5--8 cm band."""
    targets = [start_distance_m + float(offset) for offset in offsets_m]
    if not math.isfinite(start_distance_m) or any(not 0.05 <= target <= 0.08 for target in targets):
        return []
    return targets


def force_fraction(peak: float, limit: float = FORCE_LIMIT) -> float:
    if limit <= 0:
        raise ValueError("Force limit must be positive")
    return float(peak) / float(limit)


def d1_adjudication(a0_force_limit_steps: Sequence[int], sac_peak_forces: Sequence[float]) -> str:
    """Apply the D1 branches without inventing a result for mixed evidence."""
    if any(0 < int(step) <= 40 for step in a0_force_limit_steps):
        return "threshold_or_wrapper_layer"
    if not sac_peak_forces:
        return "uncertain"
    ratios = np.asarray(sac_peak_forces, dtype=float) / FORCE_LIMIT
    if np.max(ratios) >= 0.8:
        return "knife_edge_threshold_policy_precision_insufficient"
    if np.max(ratios) < 0.5:
        return "ia_collision_requires_geometry_attribution"
    return "uncertain"


def d2_adjudication(b1_successes: int, evaluable: int, b2_successes: int | None = None) -> str:
    if evaluable <= 0:
        return "uncertain"
    if b1_successes >= 3:
        return "constant_close_not_distinguishable"
    if b1_successes <= 1 and b2_successes is not None:
        if b2_successes >= 2:
            return "weak_near_start_grasp_evidence"
        if b2_successes == 0:
            return "narrow_start_coupled_capability"
    return "uncertain"
