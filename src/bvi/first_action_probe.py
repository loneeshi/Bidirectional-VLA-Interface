"""Exact-start stochastic first-action attribution for Fetch13 policies.

STATUS: frozen — historical training and diagnostics (retained)

The successful SAC action is a *reference*, not a unique ground-truth label.
This module only asks whether the deployed S1 action looks like an exceptional
draw from its own frozen stochastic policy.  It must not be used to claim that
matching SAC is necessary for task success.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


ACTION_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "upperarm_roll",
    "elbow_flex",
    "forearm_roll",
    "wrist_flex",
    "wrist_roll",
    "gripper",
    "head_pan",
    "head_tilt",
    "torso_lift",
    "base_forward",
    "base_yaw",
)
HEAD_CHANNELS = (8, 9)
ACTIVE_CHANNELS = tuple(index for index in range(13) if index not in HEAD_CHANNELS)

# Fixed before the probe.  These are diagnostic thresholds in normalized
# controller coordinates, not success thresholds or benchmark claims.
GATE_SPEC = {
    "expected_identity_case_count": 5,
    "expected_success_reference_case_count": 4,
    "samples_per_case": 16,
    "exact_reproduction_atol": 1e-6,
    # With 16 draws, >=15/16 means the deployed draw is one of the two most
    # distant from the SAC reference (ties are inclusive). Cohort voting later
    # keeps only references whose SAC replay reached native success.
    "sac_distance_tail_percentile": 15 / 16,
    "tail_cases_required": 3,
    "median_sac_rmse_ratio_for_tail_candidate": 0.75,
    "sac_action_magnitude_for_sign_test": 0.25,
    "opposite_sign_same_sign_fraction_max": 0.10,
    "minimum_opposed_channels_for_stable_split": 3,
    "central_interval_quantiles": [0.05, 0.95],
    "central_support_fraction_max_for_stable_split": 0.50,
    "z_scale_floor": 0.05,
}


def _action(value, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    while array.ndim > 1 and array.shape[0] == 1:
        array = array[0]
    if array.shape != (13,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must be one finite Fetch13 action")
    return array


def _samples(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 13 or not np.isfinite(array).all():
        raise ValueError("samples must be finite [sample,13] Fetch13 actions")
    if not len(array):
        raise ValueError("samples must be nonempty")
    return array


def applied_actions(value) -> np.ndarray:
    """Apply the deployed clipping and stationary-head mask."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape[-1] != 13 or not np.isfinite(array).all():
        raise ValueError("Expected finite Fetch13 actions")
    result = np.clip(array, -1.0, 1.0)
    result[..., list(HEAD_CHANNELS)] = 0.0
    return result


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def model_identity_checks(current: Mapping, recorded: Mapping) -> dict:
    """Fail closed on the three immutable identities recorded by the server."""
    fields = (
        "pretrained_parameters_sha256",
        "normalizer_sha256",
        "state_contract_sha256",
    )
    checks = {}
    reasons = []
    for field in fields:
        left, right = current.get(field), recorded.get(field)
        valid = (
            isinstance(left, str)
            and isinstance(right, str)
            and len(left) == 64
            and len(right) == 64
            and left == right
        )
        checks[field] = {
            "equal": valid,
            "current": left,
            "recorded": right,
        }
        if not valid:
            reasons.append(f"{field}_missing_or_mismatch")
    return {
        "status": "matched" if not reasons else "mismatch",
        "matched": not reasons,
        "reasons": reasons,
        "checks": checks,
    }


def first_event_actions(policy_event: Mapping, sac_event: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """Extract the deployed raw first action and applied SAC first action."""
    raw_chunk = np.asarray(policy_event.get("raw_actions"), dtype=np.float64)
    if raw_chunk.ndim == 3 and raw_chunk.shape[0] == 1:
        raw_chunk = raw_chunk[0]
    if raw_chunk.ndim != 2 or raw_chunk.shape[1] != 13 or not len(raw_chunk):
        raise ValueError("Policy first event lacks a finite [horizon,13] raw_actions chunk")
    deployed = _action(raw_chunk[0], "deployed first action")
    sac = _action(sac_event.get("action"), "SAC first action")
    return deployed, applied_actions(sac)


def analyze_first_action_distribution(
    raw_samples: Sequence[Sequence[float]],
    deployed_raw: Sequence[float],
    sac_action: Sequence[float],
    *,
    first_sample_rng_matches: bool,
    model_identity_matched: bool,
    exact_reproduction_atol: float = GATE_SPEC["exact_reproduction_atol"],
) -> dict:
    """Classify RNG-tail vs stable policy split under a predeclared gate.

    ``raw_samples[0]`` must be generated after resetting the frozen server to
    the same episode seed as the deployed action.  A classification is emitted
    only when it reproduces that action and its recorded RNG identity.
    """
    samples_raw = _samples(raw_samples)
    deployed_raw = _action(deployed_raw, "deployed first action")
    sac = applied_actions(_action(sac_action, "SAC first action"))
    samples = applied_actions(samples_raw)
    deployed = applied_actions(deployed_raw)
    active = np.asarray(ACTIVE_CHANNELS, dtype=np.int64)

    reproduction_error = float(np.max(np.abs(samples_raw[0] - deployed_raw)))
    exact_reproduction = reproduction_error <= exact_reproduction_atol
    identity_ok = bool(exact_reproduction and first_sample_rng_matches and model_identity_matched)

    active_samples = samples[:, active]
    active_deployed = deployed[active]
    active_sac = sac[active]
    mean = np.mean(active_samples, axis=0)
    median = np.median(active_samples, axis=0)
    std = np.std(active_samples, axis=0, ddof=1) if len(samples) > 1 else np.zeros(len(active))
    scale = np.maximum(std, float(GATE_SPEC["z_scale_floor"]))
    radial = np.sqrt(np.mean(np.square((active_samples - mean) / scale), axis=1))
    deployed_radial = float(radial[0])
    # Inclusive empirical percentile: 1.0 means the deployed draw is tied for
    # or is the most radially extreme of the sampled actions.
    deployed_tail_percentile = float(np.mean(radial <= deployed_radial))

    q_low, q_high = GATE_SPEC["central_interval_quantiles"]
    low = np.quantile(active_samples, q_low, axis=0)
    high = np.quantile(active_samples, q_high, axis=0)
    central_support = (active_sac >= low) & (active_sac <= high)
    sign_eligible = np.abs(active_sac) >= GATE_SPEC["sac_action_magnitude_for_sign_test"]
    same_sign = np.mean(active_samples * active_sac[None, :] > 0.0, axis=0)
    opposed = sign_eligible & (
        same_sign <= GATE_SPEC["opposite_sign_same_sign_fraction_max"]
    )
    eligible_count = int(np.sum(sign_eligible))
    central_support_fraction = (
        float(np.mean(central_support[sign_eligible])) if eligible_count else None
    )

    deployed_sac_rmse = _rmse(active_deployed, active_sac)
    median_sac_rmse = _rmse(median, active_sac)
    mean_sac_rmse = _rmse(mean, active_sac)
    sample_sac_rmse = np.sqrt(np.mean(np.square(active_samples - active_sac), axis=1))
    median_ratio = (
        median_sac_rmse / deployed_sac_rmse if deployed_sac_rmse > 1e-12 else None
    )

    sac_distance_tail_percentile = float(np.mean(sample_sac_rmse <= deployed_sac_rmse))

    if not identity_ok:
        decision = "identity_failed"
        interpretation = (
            "The exact online first action was not reproduced under the recorded model/RNG identity; "
            "distribution attribution is invalid."
        )
    elif len(samples) != GATE_SPEC["samples_per_case"]:
        decision = "wrong_sample_count"
        interpretation = "Identity passed, but the predeclared per-start sample count was not met."
    elif (
        sac_distance_tail_percentile >= GATE_SPEC["sac_distance_tail_percentile"]
        and median_ratio is not None
        and median_ratio <= GATE_SPEC["median_sac_rmse_ratio_for_tail_candidate"]
    ):
        decision = "sac_distant_tail_candidate"
        interpretation = (
            "The deployed action is one of the two samples farthest from the official SAC reference, "
            "and the policy median is materially closer. This seed counts toward the cohort RNG-tail gate."
        )
    else:
        decision = "not_sac_distant_tail"
        interpretation = (
            "The deployed action does not meet both predeclared SAC-distance tail conditions for this seed."
        )

    channel_rows = []
    for local, channel in enumerate(active):
        channel_rows.append(
            {
                "channel": int(channel),
                "name": ACTION_NAMES[int(channel)],
                "deployed_applied": float(active_deployed[local]),
                "sac_applied": float(active_sac[local]),
                "sample_mean": float(mean[local]),
                "sample_median": float(median[local]),
                "sample_std": float(std[local]),
                "sample_q05": float(low[local]),
                "sample_q95": float(high[local]),
                "sac_in_central_90pct": bool(central_support[local]),
                "sac_sign_eligible": bool(sign_eligible[local]),
                "same_sign_as_sac_fraction": float(same_sign[local]),
                "confidently_opposed_to_sac": bool(opposed[local]),
                "raw_out_of_bounds_fraction": float(
                    np.mean(np.abs(samples_raw[:, int(channel)]) > 1.0)
                ),
            }
        )

    return {
        "decision": decision,
        "interpretation": interpretation,
        "scope_warning": (
            "One exact start and one official SAC reference action; SAC is not a unique target, and "
            "this is not a success-rate estimate."
        ),
        "gate_spec": dict(GATE_SPEC),
        "sample_count": int(len(samples)),
        "identity": {
            "valid": identity_ok,
            "model_identity_matched": bool(model_identity_matched),
            "first_sample_rng_matches": bool(first_sample_rng_matches),
            "first_sample_reproduction_max_abs_raw": reproduction_error,
            "exact_reproduction_atol": exact_reproduction_atol,
            "first_sample_reproduced": exact_reproduction,
        },
        "active_channels": list(ACTIVE_CHANNELS),
        "deployed_radial_tail_percentile": deployed_tail_percentile,
        "deployed_sac_distance_percentile": sac_distance_tail_percentile,
        "opposed_channel_count": int(np.sum(opposed)),
        "opposed_channels": [ACTION_NAMES[int(active[i])] for i in np.flatnonzero(opposed)],
        "sign_eligible_channel_count": eligible_count,
        "sac_central_90pct_support_fraction_on_sign_eligible_channels": central_support_fraction,
        "active_rmse_to_sac": {
            "deployed": deployed_sac_rmse,
            "sample_mean": mean_sac_rmse,
            "sample_median": median_sac_rmse,
            "sample_median_over_deployed": median_ratio,
            "nearest_sample": float(np.min(sample_sac_rmse)),
            "sample_q05": float(np.quantile(sample_sac_rmse, 0.05)),
            "sample_q50": float(np.quantile(sample_sac_rmse, 0.50)),
            "sample_q95": float(np.quantile(sample_sac_rmse, 0.95)),
        },
        "channel_metrics": channel_rows,
    }


def analyze_probe_cohort(seed_reports: Sequence[Mapping]) -> dict:
    """Apply 5-case identity and 4-success-reference tail gates."""
    reports = [dict(report) for report in seed_reports]
    seeds = [report.get("seed") for report in reports]
    contract_reasons = []
    if len(reports) != GATE_SPEC["expected_identity_case_count"]:
        contract_reasons.append("case_count_mismatch")
    if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
        contract_reasons.append("seed_missing_or_invalid")
    elif len(set(seeds)) != len(seeds):
        contract_reasons.append("duplicate_seed")
    if any(report.get("sample_count") != GATE_SPEC["samples_per_case"] for report in reports):
        contract_reasons.append("per_case_sample_count_mismatch")
    if any(not report.get("identity", {}).get("valid", False) for report in reports):
        contract_reasons.append("one_or_more_exact_reproduction_identity_failures")

    success_flags = [report.get("sac_reference_native_success") for report in reports]
    if any(not isinstance(flag, bool) for flag in success_flags):
        contract_reasons.append("sac_reference_success_flag_missing_or_invalid")
    eligible = [report for report in reports if report.get("sac_reference_native_success") is True]
    if len(eligible) != GATE_SPEC["expected_success_reference_case_count"]:
        contract_reasons.append("success_reference_case_count_mismatch")
    tail_seeds = [
        report["seed"]
        for report in eligible
        if report.get("decision") == "sac_distant_tail_candidate"
    ]
    if contract_reasons:
        decision = "invalid_cohort"
        interpretation = "The predeclared five-start exact-reproduction cohort contract was not met."
    elif len(tail_seeds) >= GATE_SPEC["tail_cases_required"]:
        decision = "stochastic_tail_supported"
        interpretation = (
            "At least three of four deployed actions with native-success SAC references are among the two most "
            "SAC-reference-distant of sixteen draws, with the per-start median at least 25% closer. "
            "Test aggregation/RNG policy before adaptation."
        )
    else:
        decision = "stochastic_tail_not_supported"
        interpretation = (
            "Fewer than three of the four native-success SAC-reference starts meet the predeclared tail rule. "
            "The observed first-action split is not explained by the deployed draw repeatedly landing in the "
            "frozen policy's SAC-distant tail; continue input/conditioning or adaptation attribution."
        )

    # Reference only: ranks need not be independent across scenarios.  If they
    # were independent and exchangeable, top-two probability is 2/16.
    p = 2 / GATE_SPEC["samples_per_case"]
    n = GATE_SPEC["expected_success_reference_case_count"]
    k = GATE_SPEC["tail_cases_required"]
    reference_probability = sum(
        math.comb(n, count) * p**count * (1 - p) ** (n - count)
        for count in range(k, n + 1)
    )
    return {
        "decision": decision,
        "interpretation": interpretation,
        "scope_warning": (
            "All five starts are retained for exact-reproduction identity, but SAC-distance voting uses only "
            "the four native-success SAC references. SAC is not a supervised label or a necessary action."
        ),
        "gate_spec": dict(GATE_SPEC),
        "contract_reasons": contract_reasons,
        "seeds": seeds,
        "success_reference_seeds": [report["seed"] for report in eligible],
        "diagnostic_failed_reference_seeds": [
            report["seed"]
            for report in reports
            if report.get("sac_reference_native_success") is False
        ],
        "tail_candidate_seeds": tail_seeds,
        "tail_candidate_count": len(tail_seeds),
        "independent_exchangeable_rank_reference_probability": reference_probability,
        "independence_claimed": False,
    }


def validate_gate_spec() -> None:
    """Catch accidental nonsensical edits to the predeclared gate."""
    probabilities = (
        "sac_distance_tail_percentile",
        "median_sac_rmse_ratio_for_tail_candidate",
        "opposite_sign_same_sign_fraction_max",
        "central_support_fraction_max_for_stable_split",
    )
    for key in probabilities:
        value = GATE_SPEC[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid probability/ratio gate: {key}")
    low, high = GATE_SPEC["central_interval_quantiles"]
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid central interval quantiles")
    if (
        GATE_SPEC["samples_per_case"] < 2
        or GATE_SPEC["expected_identity_case_count"] < 1
        or GATE_SPEC["expected_success_reference_case_count"] < 1
        or GATE_SPEC["expected_success_reference_case_count"]
        > GATE_SPEC["expected_identity_case_count"]
        or not 1
        <= GATE_SPEC["tail_cases_required"]
        <= GATE_SPEC["expected_success_reference_case_count"]
        or GATE_SPEC["z_scale_floor"] <= 0
    ):
        raise ValueError("Invalid sample floor or z scale")


validate_gate_spec()
