import numpy as np

from bvi.first_action_probe import (
    ACTION_NAMES,
    GATE_SPEC,
    analyze_first_action_distribution,
    analyze_probe_cohort,
    applied_actions,
    first_event_actions,
    model_identity_checks,
)


def analyze(samples, deployed, sac, **kwargs):
    return analyze_first_action_distribution(
        samples,
        deployed,
        sac,
        first_sample_rng_matches=kwargs.get("rng", True),
        model_identity_matched=kwargs.get("model", True),
    )


def test_applied_actions_clips_and_masks_head():
    action = np.linspace(-2, 2, 13)
    result = applied_actions(action)
    assert np.max(np.abs(result)) <= 1
    assert result[8] == 0
    assert result[9] == 0


def test_exact_seed_tail_candidate_uses_sac_distance_rank_and_median_improvement():
    samples = np.zeros((16, 13), np.float64)
    samples[0, :] = 1.0
    deployed = samples[0].copy()
    report = analyze(samples, deployed, np.zeros(13))
    assert report["identity"]["valid"] is True
    assert report["decision"] == "sac_distant_tail_candidate"
    assert report["deployed_sac_distance_percentile"] == 1
    assert report["active_rmse_to_sac"]["sample_median_over_deployed"] == 0


def test_central_draw_opposing_sac_is_not_mislabeled_as_rng_tail():
    samples = np.zeros((16, 13), np.float64)
    samples[:, :3] = 0.8
    deployed = samples[0].copy()
    sac = np.zeros(13)
    sac[:3] = -0.8
    report = analyze(samples, deployed, sac)
    assert report["decision"] == "not_sac_distant_tail"
    assert report["opposed_channel_count"] == 3
    assert report["opposed_channels"] == list(ACTION_NAMES[:3])


def test_identity_failure_blocks_tail_classification():
    samples = np.zeros((16, 13), np.float64)
    deployed = np.ones(13)
    report = analyze(samples, deployed, np.zeros(13), rng=False)
    assert report["decision"] == "identity_failed"
    assert report["identity"]["valid"] is False
    assert report["identity"]["first_sample_rng_matches"] is False


def _cohort_report(seed, tail=True, valid=True):
    return {
        "seed": seed,
        "sample_count": GATE_SPEC["samples_per_case"],
        "identity": {"valid": valid},
        "decision": "sac_distant_tail_candidate" if tail else "not_sac_distant_tail",
        "sac_reference_native_success": seed != 2024,
    }


def test_cohort_requires_three_of_four_success_reference_tail_candidates():
    reports = [_cohort_report(seed, seed < 2028) for seed in range(2024, 2029)]
    result = analyze_probe_cohort(reports)
    assert result["decision"] == "stochastic_tail_supported"
    assert result["tail_candidate_count"] == 3
    assert result["diagnostic_failed_reference_seeds"] == [2024]
    reports[3]["decision"] = "not_sac_distant_tail"
    result = analyze_probe_cohort(reports)
    assert result["decision"] == "stochastic_tail_not_supported"
    assert result["independence_claimed"] is False


def test_invalid_exact_reproduction_invalidates_cohort():
    reports = [_cohort_report(seed) for seed in range(2024, 2029)]
    reports[0]["identity"]["valid"] = False
    result = analyze_probe_cohort(reports)
    assert result["decision"] == "invalid_cohort"
    assert "one_or_more_exact_reproduction_identity_failures" in result["contract_reasons"]


def test_model_identity_requires_all_three_hashes():
    current = {
        "pretrained_parameters_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "state_contract_sha256": "c" * 64,
    }
    assert model_identity_checks(current, dict(current))["matched"] is True
    recorded = dict(current, normalizer_sha256="d" * 64)
    result = model_identity_checks(current, recorded)
    assert result["matched"] is False
    assert result["reasons"] == ["normalizer_sha256_missing_or_mismatch"]


def test_first_event_action_extraction_accepts_nested_sac_action():
    chunk = np.zeros((10, 13), np.float64)
    chunk[0, 0] = 0.25
    deployed, sac = first_event_actions(
        {"raw_actions": chunk.tolist()}, {"action": [[2.0] + [0.0] * 12]}
    )
    assert deployed[0] == 0.25
    assert sac[0] == 1.0
    assert sac[8] == 0.0
