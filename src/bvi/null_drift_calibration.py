"""Pure validation and threshold generation for Level-3 null-drift controls."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

from bvi.closed_loop_divergence import FETCH_QPOS_CHANNELS, FETCH_QPOS_UNITS


RULE_VERSION = "same-action-fresh-env-pairwise-max-v1"
MIN_REPEATS = 3
CALIBRATION_ACTIONS = 35
NULL_MULTIPLIER = 2.0
UNIT_FLOOR = 1e-4


def _flat(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flat(item))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected nested finite numeric values")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Null-drift evidence contains a nonfinite value")
    return [number]


def _scalar(value) -> float:
    values = _flat(value)
    if len(values) != 1:
        raise ValueError("Expected scalar null-drift evidence")
    return values[0]


def validate_recorded_actions(rows: Iterable[Mapping], required_actions: int = CALIBRATION_ACTIONS) -> list[list[float]]:
    """Validate the immutable SAC prefix before any simulator process starts."""
    if required_actions != CALIBRATION_ACTIONS:
        raise ValueError(f"Rule v1 is frozen to exactly {CALIBRATION_ACTIONS} actions")
    rows = list(rows)
    if len(rows) < required_actions:
        raise ValueError(f"Recorded SAC source has only {len(rows)} actions")
    all_actions = []
    for expected, row in enumerate(rows, 1):
        if row.get("step") != expected:
            raise ValueError("Recorded SAC steps must be contiguous from one")
        action = _flat(row.get("action"))
        if len(action) != 13:
            raise ValueError("Recorded SAC actions must have 13 channels")
        if any(abs(value) > 1 for value in action):
            raise ValueError("Recorded SAC action is outside the normalized controller box")
        if action[8] != 0 or action[9] != 0:
            raise ValueError("Recorded SAC source violates the stationary-head contract")
        all_actions.append(action)
    return all_actions[:required_actions]


def _validate_repeat(
    rows: Sequence[Mapping],
    source_actions: Sequence[Sequence[float]],
    channel_names: Sequence[str],
) -> dict:
    if len(rows) != CALIBRATION_ACTIONS:
        raise ValueError(f"Each null repeat must contain exactly {CALIBRATION_ACTIONS} rows")
    qpos = []
    distances = []
    forces = []
    for expected, (row, source_action) in enumerate(zip(rows, source_actions), 1):
        if row.get("step") != expected:
            raise ValueError("Null-repeat steps must be contiguous from one")
        action = _flat(row.get("action"))
        if action != list(source_action):
            raise ValueError(f"Null-repeat action differs from frozen SAC source at step {expected}")
        joints = _flat(row.get("qpos"))
        if len(joints) != len(channel_names):
            raise ValueError(f"Null-repeat qpos dimension changed at step {expected}")
        if "tcp_target_distance_m" not in row or "robot_cumulative_force" not in row:
            raise ValueError(f"Null-repeat scalar diagnostic missing at step {expected}")
        qpos.append(joints)
        distances.append(_scalar(row["tcp_target_distance_m"]))
        forces.append(_scalar(row["robot_cumulative_force"]))
    return {"qpos": qpos, "tcp_target_distance_m": distances, "robot_cumulative_force": forces}


def _pairwise_max(series: Sequence[Sequence[float]]) -> list[float]:
    width = len(series[0])
    maxima = [0.0] * width
    for left in range(len(series)):
        for right in range(left + 1, len(series)):
            for index in range(width):
                maxima[index] = max(maxima[index], abs(series[left][index] - series[right][index]))
    return maxima


def generate_null_drift_thresholds(
    repeats: Sequence[Sequence[Mapping]],
    source_actions: Sequence[Sequence[float]],
    *,
    channel_names: Sequence[str] = FETCH_QPOS_CHANNELS,
    units: Mapping[str, str] = FETCH_QPOS_UNITS,
    null_multiplier: float = NULL_MULTIPLIER,
    translation_floor_m: float = UNIT_FLOOR,
    rotation_floor_rad: float = UNIT_FLOOR,
) -> dict:
    """Generate thresholds only from same-action null repeats.

    For each qpos channel, the null envelope is the largest pairwise difference
    across all repeat pairs and all 35 post-action states.  The frozen threshold
    is max(unit floor, multiplier * envelope).  No policy-under-test trajectory
    is accepted by this function.
    """
    names = list(channel_names)
    if len(repeats) < MIN_REPEATS:
        raise ValueError(f"At least {MIN_REPEATS} fresh-env repeats are required")
    if len(source_actions) != CALIBRATION_ACTIONS:
        raise ValueError(f"Exactly {CALIBRATION_ACTIONS} frozen source actions are required")
    if not names or len(names) != len(set(names)) or set(names) != set(units):
        raise ValueError("Channel names and units must be complete and unique")
    for label, value in (
        ("null_multiplier", null_multiplier),
        ("translation_floor_m", translation_floor_m),
        ("rotation_floor_rad", rotation_floor_rad),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{label} must be numeric")
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{label} must be finite and positive")
    parsed = [_validate_repeat(list(rows), source_actions, names) for rows in repeats]
    qpos_envelope = [0.0] * len(names)
    distance_envelope = 0.0
    force_envelope = 0.0
    per_step = []
    for step in range(CALIBRATION_ACTIONS):
        qpos_at_step = [repeat["qpos"][step] for repeat in parsed]
        distance_at_step = [[repeat["tcp_target_distance_m"][step]] for repeat in parsed]
        force_at_step = [[repeat["robot_cumulative_force"][step]] for repeat in parsed]
        qpos_max = _pairwise_max(qpos_at_step)
        distance_max = _pairwise_max(distance_at_step)[0]
        force_max = _pairwise_max(force_at_step)[0]
        qpos_envelope = [max(old, new) for old, new in zip(qpos_envelope, qpos_max)]
        distance_envelope = max(distance_envelope, distance_max)
        force_envelope = max(force_envelope, force_max)
        per_step.append(
            {
                "step": step + 1,
                "qpos_max_pairwise_abs": dict(zip(names, qpos_max)),
                "tcp_target_distance_max_pairwise_abs_m": distance_max,
                "robot_cumulative_force_max_pairwise_abs": force_max,
            }
        )
    floors = {
        name: float(translation_floor_m if units[name] == "m" else rotation_floor_rad)
        for name in names
    }
    thresholds = {
        name: max(floors[name], float(null_multiplier) * envelope)
        for name, envelope in zip(names, qpos_envelope)
    }
    return {
        "status": "completed_null_only_calibration",
        "consumes_policy_under_test": False,
        "generation_rule": {
            "version": RULE_VERSION,
            "expression": "threshold[channel] = max(unit_floor, null_multiplier * max_pairwise_abs_over_repeat_pairs_and_steps)",
            "strict_comparison_operator": ">",
            "repeat_count": len(repeats),
            "repeat_pair_count": len(repeats) * (len(repeats) - 1) // 2,
            "calibration_actions": CALIBRATION_ACTIONS,
            "null_multiplier": float(null_multiplier),
            "translation_floor_m": float(translation_floor_m),
            "rotation_floor_rad": float(rotation_floor_rad),
            "scope": "same GPU/backend/env config/exact snapshot/frozen SAC action prefix only",
        },
        "qpos_abs_error": thresholds,
        "qpos_null_max_pairwise_abs": dict(zip(names, qpos_envelope)),
        "unit_floor": floors,
        "tcp_target_distance_null_max_pairwise_abs_m": distance_envelope,
        "robot_cumulative_force_null_max_pairwise_abs": force_envelope,
        "per_step_null_envelope": per_step,
    }
