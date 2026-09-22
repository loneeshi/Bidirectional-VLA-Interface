"""Pure contracts for the native24 S2 step0/step20 offline decomposition.

STATUS: frozen — historical training and diagnostics (retained)

The GPU entry point lives in ``scripts/audit_native_s2_decomposition.py``.
This module deliberately contains no JAX, optimizer, simulator, or model code so
the roster, identity, parity, queue, and reporting gates are unit-testable on
CPU before any accelerator is reserved.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np


FAMILIES = ("reach", "grasp", "move", "release")
COMBINATIONS = (
    ("H0_B0", 0, 0),
    ("H20_B0", 20, 0),
    ("H0_B20", 0, 20),
    ("H20_B20", 20, 20),
)
HEAD_CHANNELS = (8, 9)
ACTION_CHANNELS = (
    ("shoulder_pan", "rad_delta", -0.1, 0.1),
    ("shoulder_lift", "rad_delta", -0.1, 0.1),
    ("upperarm_roll", "rad_delta", -0.1, 0.1),
    ("elbow_flex", "rad_delta", -0.1, 0.1),
    ("forearm_roll", "rad_delta", -0.1, 0.1),
    ("wrist_flex", "rad_delta", -0.1, 0.1),
    ("wrist_roll", "rad_delta", -0.1, 0.1),
    ("gripper", "m_joint_target", -0.01, 0.05),
    ("head_pan", "rad_delta", -0.1, 0.1),
    ("head_tilt", "rad_delta", -0.1, 0.1),
    ("torso_lift", "m_delta", -0.1, 0.1),
    ("base_forward", "m_per_s", -1.0, 1.0),
    ("base_yaw", "rad_per_s", -3.14, 3.14),
)
SIGN_CHANNELS = {"gripper": 7, "torso_lift": 10, "base_yaw": 12}
SCHEMA = "bvi.native24-s2-offline-decomposition/1"


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fixed_validation_roster(records: Sequence[Mapping], tables: Mapping[int, Mapping]) -> list[dict]:
    """Reproduce the immutable validation traversal in ``train_native_s2.py``."""
    rows = []
    for family in FAMILIES:
        candidates = [
            n for n, record in enumerate(records)
            if record.get("split") == "validation" and record.get("family") == family
        ][:10]
        if not candidates:
            raise ValueError(f"Missing validation family: {family}")
        for window in candidates:
            length = len(tables[window]["state"])
            if length < 2:
                raise ValueError("Validation invocation must contain at least two observations")
            for index in sorted({0, length // 2, length - 1}):
                rows.append({
                    "row_id": f"w{window:03d}-i{index:04d}-{family}",
                    "window": window,
                    "index": index,
                    "family": family,
                    "loss_rng_seed": 123 + window * 1000 + index,
                    "sample_rng_seed": 700_123 + window * 1000 + index,
                    "noise_rng_seed": 900_123 + window * 1000 + index,
                    "action_label_valid": bool(tables[window]["action_valid"][index]),
                })
    if len({row["row_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate held-out row identity")
    return rows


def combination_spec() -> list[dict]:
    return [
        {"name": name, "head_step": head_step, "bank_step": bank_step}
        for name, head_step, bank_step in COMBINATIONS
    ]


def controller_to_physical(actions) -> np.ndarray:
    values = np.asarray(actions, dtype=np.float64)
    if values.shape[-1] != 13 or not np.isfinite(values).all():
        raise ValueError("Expected finite Fetch13 controller commands")
    lower = np.asarray([row[2] for row in ACTION_CHANNELS])
    upper = np.asarray([row[3] for row in ACTION_CHANNELS])
    return lower + (values + 1.0) * (upper - lower) / 2.0


def sign_label(value: float, *, atol: float = 1e-8) -> int:
    if not math.isfinite(float(value)):
        raise ValueError("Sign input must be finite")
    return 0 if abs(float(value)) <= atol else (1 if value > 0 else -1)


def first_action_diagnostics(predicted_raw, expert_raw, *, eligible: bool) -> dict:
    predicted = np.asarray(predicted_raw, dtype=np.float64)
    expert = np.asarray(expert_raw, dtype=np.float64)
    if predicted.shape != (13,) or expert.shape != (13,) or not (
        np.isfinite(predicted).all() and np.isfinite(expert).all()
    ):
        raise ValueError("First actions must be finite Fetch13 vectors")
    oob = np.abs(predicted) > 1.0
    result = {
        "eligible": bool(eligible),
        "predicted_raw": predicted.tolist(),
        "raw_oob_count": int(oob.sum()),
        "raw_oob_fraction": float(oob.mean()),
        "raw_max_abs": float(np.max(np.abs(predicted))),
        "raw_oob_channels": [ACTION_CHANNELS[i][0] for i in np.flatnonzero(oob)],
    }
    if not eligible:
        result.update(expert_raw=None, physical_abs_error=None, physical_rmse=None,
                      physical_l1_mean=None, sign_diagnostics=None)
        return result
    predicted_physical = controller_to_physical(predicted)
    expert_physical = controller_to_physical(expert)
    error = np.abs(predicted_physical - expert_physical)
    result.update(
        expert_raw=expert.tolist(),
        predicted_physical=predicted_physical.tolist(),
        expert_physical=expert_physical.tolist(),
        physical_abs_error=error.tolist(),
        physical_rmse=float(np.sqrt(np.mean(np.square(predicted_physical - expert_physical)))),
        physical_l1_mean=float(np.mean(error)),
        sign_diagnostics={
            name: {
                "channel": channel,
                "predicted": sign_label(predicted[channel]),
                "expert": sign_label(expert[channel]),
                "match": sign_label(predicted[channel]) == sign_label(expert[channel]),
            }
            for name, channel in SIGN_CHANNELS.items()
        },
    )
    return result


def assert_path_parity(training: Mapping, deployment: Mapping, *, atol: float = 0.0) -> dict:
    """Fail closed on model input, normalized action, external action and progress."""
    if training.get("observation_sha256") != deployment.get("observation_sha256"):
        raise ValueError("Training/deployment transformed-observation hash mismatch")
    fields = ("normalized_actions", "external_actions", "progress")
    maximum = {}
    exact = {}
    for field in fields:
        left = np.asarray(training[field])
        right = np.asarray(deployment[field])
        if left.shape != right.shape or not (np.isfinite(left).all() and np.isfinite(right).all()):
            raise ValueError(f"Invalid parity field: {field}")
        delta = float(np.max(np.abs(left - right))) if left.size else 0.0
        maximum[field] = delta
        exact[field] = bool(np.array_equal(left, right))
        if delta > atol:
            raise ValueError(f"Training/deployment {field} mismatch: {delta}")
    return {
        "passed": True,
        "atol": float(atol),
        "observation_sha256": training["observation_sha256"],
        "max_abs_diff": maximum,
        "bit_exact": exact,
    }


def summarize_rows(rows: Sequence[Mapping]) -> dict:
    if not rows:
        raise ValueError("No decomposition rows")
    combinations = {row.get("combination") for row in rows}
    expected = {row[0] for row in COMBINATIONS}
    if combinations != expected:
        raise ValueError("Incomplete 2x2 combinations")
    result = {}
    for combination in sorted(combinations):
        combo_rows = [row for row in rows if row["combination"] == combination]
        per_family = {}
        for family in FAMILIES:
            family_rows = [row for row in combo_rows if row["family"] == family]
            if not family_rows:
                raise ValueError(f"Missing {combination}/{family}")
            metrics = {
                key: float(np.mean([row[key] for row in family_rows]))
                for key in ("action_loss", "progress_loss", "joint_loss")
            }
            eligible = [row["first_action"] for row in family_rows if row["first_action"]["eligible"]]
            metrics.update(
                rows=len(family_rows),
                first_action_rows=len(eligible),
                first_action_physical_rmse_mean=(
                    None if not eligible else float(np.mean([row["physical_rmse"] for row in eligible]))
                ),
                raw_oob_fraction_mean=float(np.mean([
                    row["first_action"]["raw_oob_fraction"] for row in family_rows
                ])),
                sign_match_fraction={
                    name: (None if not eligible else float(np.mean([
                        row["sign_diagnostics"][name]["match"] for row in eligible
                    ])))
                    for name in SIGN_CHANNELS
                },
            )
            per_family[family] = metrics
        result[combination] = {
            "per_family": per_family,
            "macro_action_loss": float(np.mean([per_family[f]["action_loss"] for f in FAMILIES])),
            "macro_progress_loss": float(np.mean([per_family[f]["progress_loss"] for f in FAMILIES])),
            "macro_joint_loss": float(np.mean([per_family[f]["joint_loss"] for f in FAMILIES])),
        }
    return result


def compare_historical_rows(rows: Sequence[Mapping], historical: Mapping, *, atol: float = 1e-6) -> dict:
    expected = {(int(row["window"]), int(row["index"]), row["family"]): row for row in historical["rows"]}
    actual = {(int(row["window"]), int(row["index"]), row["family"]): row for row in rows}
    if set(expected) != set(actual):
        raise ValueError("Historical validation roster mismatch")
    maximum = {key: 0.0 for key in ("action_loss", "progress_loss", "joint_loss")}
    for identity in expected:
        for key in maximum:
            maximum[key] = max(maximum[key], abs(float(expected[identity][key]) - float(actual[identity][key])))
    if any(value > atol for value in maximum.values()):
        raise ValueError(f"Historical validation reproduction failed: {maximum}")
    return {"passed": True, "atol": float(atol), "max_abs_diff": maximum, "rows": len(actual)}
