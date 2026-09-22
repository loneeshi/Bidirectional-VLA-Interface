"""Predeclared functional confirmation for same-key BF16 action clusters.

STATUS: frozen — historical training and diagnostics (retained)
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from bvi.first_action_probe import ACTIVE_CHANNELS, GATE_SPEC, applied_actions


CONFIRMATION_SPEC = {
    "expected_cases": 5,
    "fresh_inference_calls_per_case": 1,
    "distance": "RMSE over 11 applied non-head Fetch13 channels",
    "anchor_rule": (
        "historical deployed action and exploratory same-key sample0 must be mutual unique nearest "
        "neighbors versus exploratory different-key samples1..15"
    ),
    "fresh_rule": (
        "fresh same-key action must be closer to both historical and exploratory sample0 than to "
        "every exploratory different-key sample1..15"
    ),
    "strict_inequalities": True,
    "material_sign_floor": GATE_SPEC["z_scale_floor"],
    "require_all_cases": True,
}


def _action(value, label: str) -> np.ndarray:
    array = np.asarray(value, np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (13,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite Fetch13 action")
    return applied_actions(array)[list(ACTIVE_CHANNELS)]


def _sample_actions(value) -> np.ndarray:
    array = np.asarray(value, np.float64)
    if array.shape != (GATE_SPEC["samples_per_case"], 13) or not np.isfinite(array).all():
        raise ValueError(
            f"exploratory samples must be [{GATE_SPEC['samples_per_case']},13] finite actions"
        )
    return applied_actions(array)[:, list(ACTIVE_CHANNELS)]


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def material_sign_flips(left, right, floor: float = CONFIRMATION_SPEC["material_sign_floor"]):
    left, right = _action(left, "left action"), _action(right, "right action")
    return _material_sign_flips_active(left, right, floor)


def _material_sign_flips_active(left: np.ndarray, right: np.ndarray, floor: float):
    return [
        int(ACTIVE_CHANNELS[index])
        for index in range(len(ACTIVE_CHANNELS))
        if left[index] * right[index] < 0
        and min(abs(left[index]), abs(right[index])) >= floor
    ]


def anchor_cluster_preflight(historical, exploratory_samples) -> dict:
    """Check the already-observed mutual-nearest same-key anchor relation."""
    historical = _action(historical, "historical action")
    samples = _sample_actions(exploratory_samples)
    same_key = _rmse(historical, samples[0])
    historical_to_other = [_rmse(historical, row) for row in samples[1:]]
    sample0_to_other = [_rmse(samples[0], row) for row in samples[1:]]
    flips = _material_sign_flips_active(
        historical, samples[0], CONFIRMATION_SPEC["material_sign_floor"]
    )
    mutual_unique = (
        same_key < min(historical_to_other)
        and same_key < min(sample0_to_other)
    )
    return {
        "passed": bool(mutual_unique and not flips),
        "mutual_unique_nearest": bool(mutual_unique),
        "historical_to_exploratory_sample0_rmse": same_key,
        "historical_to_nearest_different_key_rmse": min(historical_to_other),
        "sample0_to_nearest_different_key_rmse": min(sample0_to_other),
        "historical_same_key_margin_ratio": same_key / min(historical_to_other),
        "sample0_same_key_margin_ratio": same_key / min(sample0_to_other),
        "material_sign_flip_channels": flips,
    }


def confirm_fresh_cluster(historical, exploratory_samples, fresh) -> dict:
    """Test one fresh same-key output against both anchors and all other keys."""
    historical_applied = _action(historical, "historical action")
    samples = _sample_actions(exploratory_samples)
    fresh_applied = _action(fresh, "fresh action")
    to_historical = _rmse(fresh_applied, historical_applied)
    to_sample0 = _rmse(fresh_applied, samples[0])
    to_other = [_rmse(fresh_applied, row) for row in samples[1:]]
    nearest_other = min(to_other)
    flips_historical = material_sign_flips(fresh, historical)
    flips_sample0 = material_sign_flips(fresh, exploratory_samples[0])
    closer_to_both = max(to_historical, to_sample0) < nearest_other
    return {
        "passed": bool(closer_to_both and not flips_historical and not flips_sample0),
        "closer_to_both_same_key_anchors_than_any_different_key": bool(closer_to_both),
        "fresh_to_historical_rmse": to_historical,
        "fresh_to_exploratory_sample0_rmse": to_sample0,
        "fresh_to_nearest_different_key_rmse": nearest_other,
        "worst_same_key_margin_ratio": (
            max(to_historical, to_sample0) / nearest_other if nearest_other > 0 else None
        ),
        "material_sign_flips_vs_historical": flips_historical,
        "material_sign_flips_vs_exploratory_sample0": flips_sample0,
    }


def adjudicate_confirmed_tail(seed_reports: Sequence[Mapping]) -> dict:
    """Apply the original frozen SAC-distance tail rule after confirmation."""
    reports = [dict(row) for row in seed_reports]
    reasons = []
    if len(reports) != CONFIRMATION_SPEC["expected_cases"]:
        reasons.append("case_count_mismatch")
    if any(not row.get("anchor_preflight", {}).get("passed", False) for row in reports):
        reasons.append("one_or_more_anchor_preflights_failed")
    if any(not row.get("fresh_confirmation", {}).get("passed", False) for row in reports):
        reasons.append("one_or_more_fresh_confirmations_failed")
    eligible = [row for row in reports if row.get("sac_reference_native_success") is True]
    if len(eligible) != GATE_SPEC["expected_success_reference_case_count"]:
        reasons.append("success_reference_case_count_mismatch")
    candidates = [
        row["seed"]
        for row in eligible
        if row["deployed_sac_distance_percentile"]
        >= GATE_SPEC["sac_distance_tail_percentile"]
        and row["sample_median_over_deployed"]
        <= GATE_SPEC["median_sac_rmse_ratio_for_tail_candidate"]
    ]
    if reasons:
        decision = "confirmation_invalid"
    elif len(candidates) >= GATE_SPEC["tail_cases_required"]:
        decision = "stochastic_tail_supported"
    else:
        decision = "stochastic_tail_not_supported"
    return {
        "decision": decision,
        "contract_reasons": reasons,
        "tail_candidate_seeds": candidates,
        "tail_candidate_count": len(candidates),
        "success_reference_seeds": [row["seed"] for row in eligible],
        "scope_warning": (
            "This confirms functional same-key clustering, not bitwise GPU determinism. The SAC-distance "
            "rule is unchanged and uses only native-success SAC references."
        ),
    }
