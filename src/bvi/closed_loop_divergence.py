"""Strictly paired, observation-only closed-loop divergence diagnostics.

This module never changes an action, environment, or success predicate.  It can
be used after a rollout or instantiated by a runner and fed one policy event at
a time.  A state comparison is emitted only when both runs identify the exact
same saved initial-state artifact and document bounded restoration error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import re
from typing import Iterable, Mapping, Sequence


FETCH_QPOS_CHANNELS = (
    "root_x_axis_joint",
    "root_y_axis_joint",
    "root_z_rotation_joint",
    "torso_lift_joint",
    "head_pan_joint",
    "shoulder_pan_joint",
    "head_tilt_joint",
    "shoulder_lift_joint",
    "upperarm_roll_joint",
    "elbow_flex_joint",
    "forearm_roll_joint",
    "wrist_flex_joint",
    "wrist_roll_joint",
    "r_gripper_finger_joint",
    "l_gripper_finger_joint",
)

FETCH_QPOS_UNITS = {
    "root_x_axis_joint": "m",
    "root_y_axis_joint": "m",
    "root_z_rotation_joint": "rad",
    "torso_lift_joint": "m",
    "head_pan_joint": "rad",
    "shoulder_pan_joint": "rad",
    "head_tilt_joint": "rad",
    "shoulder_lift_joint": "rad",
    "upperarm_roll_joint": "rad",
    "elbow_flex_joint": "rad",
    "forearm_roll_joint": "rad",
    "wrist_flex_joint": "rad",
    "wrist_roll_joint": "rad",
    "r_gripper_finger_joint": "m",
    "l_gripper_finger_joint": "m",
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def run_identity(result: Mapping) -> dict:
    """Extract only evidence that is relevant to exact-start pairing."""
    errors = result.get("reset_state_max_errors")
    maximum = None
    if isinstance(errors, Mapping) and errors:
        try:
            values = [float(value) for value in errors.values()]
            if values and all(math.isfinite(value) and value >= 0 for value in values):
                maximum = max(values)
        except (TypeError, ValueError):
            maximum = None
    return {
        "task": result.get("task"),
        "seed": result.get("seed"),
        "reference_state_sha256": result.get("reference_state_sha256"),
        "restore_max_abs_error": maximum,
    }


def strict_pairing(policy: Mapping, expert: Mapping, restore_tolerance: float = 1e-5) -> dict:
    """Validate exact saved-start identity; absence is a negative result."""
    if not math.isfinite(restore_tolerance) or restore_tolerance < 0:
        raise ValueError("restore_tolerance must be finite and nonnegative")
    reasons = []
    task_ok = isinstance(policy.get("task"), str) and policy.get("task") == expert.get("task")
    if not task_ok:
        reasons.append("task_identity_missing_or_mismatch")
    seed_ok = isinstance(policy.get("seed"), int) and policy.get("seed") == expert.get("seed")
    if not seed_ok:
        reasons.append("seed_identity_missing_or_mismatch")
    policy_sha = policy.get("reference_state_sha256")
    expert_sha = expert.get("reference_state_sha256")
    policy_sha_ok = isinstance(policy_sha, str) and _SHA256.fullmatch(policy_sha) is not None
    expert_sha_ok = isinstance(expert_sha, str) and _SHA256.fullmatch(expert_sha) is not None
    if not policy_sha_ok:
        reasons.append("policy_reference_state_sha256_missing_or_invalid")
    if not expert_sha_ok:
        reasons.append("expert_reference_state_sha256_missing_or_invalid")
    sha_ok = policy_sha_ok and expert_sha_ok and policy_sha == expert_sha
    if policy_sha_ok and expert_sha_ok and not sha_ok:
        reasons.append("reference_state_sha256_mismatch")
    restore_checks = {}
    for label, identity in (("policy", policy), ("expert", expert)):
        value = identity.get("restore_max_abs_error")
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        valid = valid and math.isfinite(float(value)) and 0 <= float(value) <= restore_tolerance
        restore_checks[label] = bool(valid)
        if not valid:
            reasons.append(f"{label}_restore_error_missing_or_above_tolerance")
    return {
        "status": "paired" if not reasons else "not_paired",
        "comparable": not reasons,
        "reasons": reasons,
        "restore_tolerance": restore_tolerance,
        "criteria": {
            "task_equal": task_ok,
            "seed_equal": seed_ok,
            "exact_reference_state_sha256_equal": sha_ok,
            "restore_error_within_tolerance": restore_checks,
        },
        "policy": dict(policy),
        "expert": dict(expert),
    }


def _flatten_numbers(value) -> list[float]:
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_flatten_numbers(item))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Expected a finite numeric scalar or nested sequence")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite trajectory value")
    return [number]


def _scalar(value) -> float:
    values = _flatten_numbers(value)
    if len(values) != 1:
        raise ValueError("Expected a scalar trajectory value")
    return values[0]


def _distance(row: Mapping) -> float | None:
    if "tcp_target_distance_m" in row:
        try:
            return _scalar(row["tcp_target_distance_m"])
        except ValueError:
            return None
    extra = row.get("extra")
    if not isinstance(extra, Mapping):
        return None
    try:
        tcp = _flatten_numbers(extra["tcp_pose_wrt_base"])
        target = _flatten_numbers(extra["obj_pose_wrt_base"])
    except (KeyError, ValueError):
        return None
    if len(tcp) < 3 or len(target) < 3:
        return None
    return math.sqrt(sum((tcp[index] - target[index]) ** 2 for index in range(3)))


def _force(row: Mapping) -> float | None:
    value = row.get("robot_cumulative_force")
    if value is None and isinstance(row.get("info"), Mapping):
        value = row["info"].get("robot_cumulative_force")
    if value is None:
        return None
    try:
        return _scalar(value)
    except ValueError:
        return None


def _observed_curve(rows: Sequence[Mapping], extractor) -> list[dict]:
    curve = []
    for row in rows:
        value = extractor(row)
        if value is not None:
            curve.append({"step": int(row["step"]), "value": value})
    return curve


def _paired_curve(policy_rows: Sequence[Mapping], expert_rows: Sequence[Mapping], extractor) -> dict:
    policy_by_step = {int(row["step"]): extractor(row) for row in policy_rows}
    expert_by_step = {int(row["step"]): extractor(row) for row in expert_rows}
    common = sorted(set(policy_by_step) & set(expert_by_step))
    if not common or any(policy_by_step[step] is None or expert_by_step[step] is None for step in common):
        return {
            "available": False,
            "reason": "one_or_both_trajectories_lack_the_metric_at_a_paired_step",
            "values": [],
        }
    return {
        "available": True,
        "values": [
            {
                "step": step,
                "policy": policy_by_step[step],
                "expert": expert_by_step[step],
                "delta": policy_by_step[step] - expert_by_step[step],
            }
            for step in common
        ],
    }


def _row_contract(rows: Sequence[Mapping], label: str) -> list[str]:
    reasons = []
    if not rows:
        return [f"{label}_trajectory_empty"]
    expected = list(range(1, len(rows) + 1))
    steps = [row.get("step") for row in rows]
    if steps != expected:
        reasons.append(f"{label}_steps_not_contiguous_from_one")
    for index, row in enumerate(rows, 1):
        if "qpos" not in row:
            reasons.append(f"{label}_qpos_missing_at_step_{index}")
            break
        try:
            _flatten_numbers(row["qpos"])
        except ValueError:
            reasons.append(f"{label}_qpos_invalid_at_step_{index}")
            break
    return reasons


def _validated_thresholds(
    channel_names: Sequence[str], thresholds: Mapping[str, float], units: Mapping[str, str]
) -> tuple[list[float], list[str]]:
    if not channel_names or len(channel_names) != len(set(channel_names)):
        raise ValueError("channel_names must be nonempty and unique")
    if set(thresholds) != set(channel_names):
        raise ValueError("thresholds must name every and only configured channel")
    if set(units) != set(channel_names):
        raise ValueError("units must name every and only configured channel")
    values = []
    for name in channel_names:
        value = thresholds[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"Non-numeric threshold for {name}")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Threshold for {name} must be finite and positive")
        if not isinstance(units[name], str) or not units[name]:
            raise ValueError(f"Unit for {name} must be a nonempty string")
        values.append(value)
    return values, [units[name] for name in channel_names]


def analyze_closed_loop_divergence(
    policy_rows: Iterable[Mapping],
    expert_rows: Iterable[Mapping],
    policy_identity: Mapping,
    expert_identity: Mapping,
    thresholds: Mapping[str, float],
    *,
    channel_names: Sequence[str] = FETCH_QPOS_CHANNELS,
    units: Mapping[str, str] = FETCH_QPOS_UNITS,
    restore_tolerance: float = 1e-5,
) -> dict:
    """Compare qpos by step and preserve TCP-target/force curves.

    Thresholds are intentionally mandatory and explicit.  This function does
    not choose a scientific divergence threshold on the caller's behalf.
    """
    names = list(channel_names)
    threshold_values, unit_values = _validated_thresholds(names, thresholds, units)
    policy_rows = [dict(row) for row in policy_rows]
    expert_rows = [dict(row) for row in expert_rows]
    pairing = strict_pairing(policy_identity, expert_identity, restore_tolerance)
    contract_reasons = _row_contract(policy_rows, "policy") + _row_contract(expert_rows, "expert")
    common_steps = min(len(policy_rows), len(expert_rows)) if not contract_reasons else 0
    if common_steps:
        for index in range(common_steps):
            try:
                policy_qpos = _flatten_numbers(policy_rows[index]["qpos"])
                expert_qpos = _flatten_numbers(expert_rows[index]["qpos"])
            except ValueError:
                contract_reasons.append(f"qpos_invalid_at_step_{index + 1}")
                break
            if len(policy_qpos) != len(names) or len(expert_qpos) != len(names):
                contract_reasons.append(
                    f"qpos_channel_count_mismatch_at_step_{index + 1}:"
                    f"policy={len(policy_qpos)},expert={len(expert_qpos)},configured={len(names)}"
                )
                break
    comparable = pairing["comparable"] and not contract_reasons
    curves = {
        "policy": {
            "tcp_target_distance_m": _observed_curve(policy_rows, _distance),
            "robot_cumulative_force": _observed_curve(policy_rows, _force),
        },
        "expert": {
            "tcp_target_distance_m": _observed_curve(expert_rows, _distance),
            "robot_cumulative_force": _observed_curve(expert_rows, _force),
        },
        "paired": {},
    }
    report = {
        "status": "comparable" if comparable else "not_comparable",
        "comparison_performed": comparable,
        "pairing": pairing,
        "trajectory_contract_reasons": contract_reasons,
        "policy_steps": len(policy_rows),
        "expert_steps": len(expert_rows),
        "common_steps": common_steps if comparable else 0,
        "state_channel": "post_action_qpos",
        "thresholds": [
            {"channel": name, "unit": unit, "absolute_error_threshold": threshold}
            for name, unit, threshold in zip(names, unit_values, threshold_values)
        ],
        "first_divergence": None,
        "qpos_error_curve": [],
        "curves": curves,
    }
    if not comparable:
        curves["paired"] = {
            "status": "not_comparable",
            "reason": "strict_same_start_or_trajectory_contract_failed",
        }
        return report

    qpos_curve = []
    first = None
    for index in range(common_steps):
        step = index + 1
        policy_qpos = _flatten_numbers(policy_rows[index]["qpos"])
        expert_qpos = _flatten_numbers(expert_rows[index]["qpos"])
        channels = []
        crossed = []
        for channel, unit, threshold, policy_value, expert_value in zip(
            names, unit_values, threshold_values, policy_qpos, expert_qpos
        ):
            error = abs(policy_value - expert_value)
            row = {
                "channel": channel,
                "unit": unit,
                "policy": policy_value,
                "expert": expert_value,
                "absolute_error": error,
                "threshold": threshold,
                "normalized_error": error / threshold,
                "crossed": error > threshold,
            }
            channels.append(row)
            if row["crossed"]:
                crossed.append(row)
        qpos_curve.append({"step": step, "channels": channels})
        if first is None and crossed:
            primary = max(crossed, key=lambda item: item["normalized_error"])
            first = {
                "step": step,
                "channels": [item["channel"] for item in crossed],
                "primary_channel": primary["channel"],
                "primary": primary,
            }
    report["qpos_error_curve"] = qpos_curve
    report["first_divergence"] = first
    curves["paired"] = {
        "status": "comparable",
        "tcp_target_distance_m": _paired_curve(policy_rows, expert_rows, _distance),
        "robot_cumulative_force": _paired_curve(policy_rows, expert_rows, _force),
    }
    return report


@dataclass
class ClosedLoopDivergenceRecorder:
    """Small in-run adapter: call ``record`` after each environment step."""

    expert_rows: Sequence[Mapping]
    policy_identity: Mapping
    expert_identity: Mapping
    thresholds: Mapping[str, float]
    channel_names: Sequence[str] = FETCH_QPOS_CHANNELS
    units: Mapping[str, str] = field(default_factory=lambda: FETCH_QPOS_UNITS)
    restore_tolerance: float = 1e-5
    _policy_rows: list[dict] = field(default_factory=list, init=False)

    def record(self, row: Mapping) -> None:
        expected = len(self._policy_rows) + 1
        if row.get("step") != expected:
            raise ValueError(f"Expected policy step {expected}, got {row.get('step')}")
        self._policy_rows.append(dict(row))

    def summary(self) -> dict:
        return analyze_closed_loop_divergence(
            self._policy_rows,
            self.expert_rows,
            self.policy_identity,
            self.expert_identity,
            self.thresholds,
            channel_names=self.channel_names,
            units=self.units,
            restore_tolerance=self.restore_tolerance,
        )
