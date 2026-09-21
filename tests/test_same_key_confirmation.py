import numpy as np

from bvi.first_action_probe import GATE_SPEC
from bvi.same_key_confirmation import (
    adjudicate_confirmed_tail,
    anchor_cluster_preflight,
    confirm_fresh_cluster,
    material_sign_flips,
)


def clustered_actions():
    samples = np.zeros((16, 13), np.float64)
    samples[0, :8] = 0.501
    samples[0, 10:] = 0.501
    for index in range(1, 16):
        samples[index, :8] = -0.5 + index * 0.02
        samples[index, 10:] = -0.5 + index * 0.02
    historical = np.zeros(13)
    historical[:8] = 0.5
    historical[10:] = 0.5
    fresh = historical.copy()
    fresh[:8] += 0.002
    fresh[10:] += 0.002
    return historical, samples, fresh


def test_same_key_anchor_and_fresh_are_strictly_clustered():
    historical, samples, fresh = clustered_actions()
    anchor = anchor_cluster_preflight(historical, samples)
    confirmation = confirm_fresh_cluster(historical, samples, fresh)
    assert anchor["passed"] is True
    assert confirmation["passed"] is True
    assert anchor["historical_same_key_margin_ratio"] < 0.01
    assert confirmation["worst_same_key_margin_ratio"] < 0.02


def test_fresh_different_key_like_action_fails_cluster():
    historical, samples, _ = clustered_actions()
    fresh = samples[4].copy()
    result = confirm_fresh_cluster(historical, samples, fresh)
    assert result["passed"] is False
    assert result["closer_to_both_same_key_anchors_than_any_different_key"] is False


def test_material_sign_flips_ignore_only_near_zero_changes():
    left = np.zeros(13)
    right = np.zeros(13)
    left[0], right[0] = 0.5, -0.5
    left[1], right[1] = 0.01, -0.01
    assert material_sign_flips(left, right) == [0]


def report(seed, tail, success=True, passed=True):
    return {
        "seed": seed,
        "anchor_preflight": {"passed": passed},
        "fresh_confirmation": {"passed": passed},
        "sac_reference_native_success": success,
        "deployed_sac_distance_percentile": (
            GATE_SPEC["sac_distance_tail_percentile"] if tail else 0.5
        ),
        "sample_median_over_deployed": (
            GATE_SPEC["median_sac_rmse_ratio_for_tail_candidate"] if tail else 1.0
        ),
    }


def test_tail_adjudication_needs_three_of_four_success_references():
    rows = [report(2024, True, success=False)] + [
        report(seed, seed <= 2027) for seed in range(2025, 2029)
    ]
    result = adjudicate_confirmed_tail(rows)
    assert result["decision"] == "stochastic_tail_supported"
    assert result["tail_candidate_seeds"] == [2025, 2026, 2027]
    rows[3]["fresh_confirmation"]["passed"] = False
    assert adjudicate_confirmed_tail(rows)["decision"] == "confirmation_invalid"
