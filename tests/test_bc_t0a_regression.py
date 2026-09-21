"""The BC T0a CI gate is exact where it claims exactness and stays diagnostic."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_bc_t0a_regression", REPOSITORY / "scripts/check_bc_t0a_regression.py"
)
gate = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gate)


def artifacts():
    official = REPOSITORY / "docs/results/bc-t0a-2026-09-18-run01"
    acdit = REPOSITORY / "docs/results/bc-t0a-2026-09-18-acdit-fork-run01"
    return (*gate.load_artifacts(official), *gate.load_artifacts(acdit))


def test_checked_in_three_path_artifacts_pass_as_diagnostic_not_benchmark():
    result = gate.validate_artifacts(*artifacts())
    assert result["status"] == "passed"
    assert result["diagnostic_only"] is True
    assert result["benchmark_success_rate_estimate"] is False
    assert result["distinct_fixed_initial_conditions"] == 5
    assert result["invariants"]["raw_action_max_abs_difference"] == 0
    assert result["known_good_nonzero_diagnostic"]["successful_seed_count"] == 4
    assert "statistical" in result["known_good_nonzero_diagnostic"]["criterion"]


def test_nonzero_action_delta_fails_even_when_outcomes_match():
    official_panel, official_comparison, acdit_panel, acdit_comparison = artifacts()
    official_comparison = deepcopy(official_comparison)
    official_comparison["pairs"][0]["raw_action_max_abs_difference"] = 1e-12
    with pytest.raises(gate.GateFailure, match="raw action delta is not zero"):
        gate.validate_artifacts(
            official_panel, official_comparison, acdit_panel, acdit_comparison
        )


def test_three_path_hash_drift_fails():
    official_panel, official_comparison, acdit_panel, acdit_comparison = artifacts()
    acdit_panel = deepcopy(acdit_panel)
    acdit_panel["episodes"][0]["initial_policy_obs_sha256"] = "0" * 64
    with pytest.raises(gate.GateFailure, match="initial_policy_obs_sha256 differs"):
        gate.validate_artifacts(
            official_panel, official_comparison, acdit_panel, acdit_comparison
        )


def test_all_zero_outcomes_fail_nonzero_diagnostic_without_rate_claim():
    official_panel, official_comparison, acdit_panel, acdit_comparison = map(
        deepcopy, artifacts()
    )
    for panel in (official_panel, acdit_panel):
        for episode in panel["episodes"]:
            episode["native_success"] = False
    for comparison in (official_comparison, acdit_comparison):
        for row in comparison["pairs"]:
            row["official_success"] = False
            row["project_success"] = False
    with pytest.raises(gate.GateFailure, match="produced no native success"):
        gate.validate_artifacts(
            official_panel, official_comparison, acdit_panel, acdit_comparison
        )


def test_missing_seed_and_provenance_drift_fail_closed():
    official_panel, official_comparison, acdit_panel, acdit_comparison = artifacts()
    missing = deepcopy(acdit_panel)
    missing["episodes"].pop()
    with pytest.raises(gate.GateFailure, match="seed/path roster differs"):
        gate.validate_artifacts(
            official_panel, official_comparison, missing, acdit_comparison
        )
    drifted = deepcopy(acdit_panel)
    drifted["episodes"][0]["checkpoint_sha256"] = "f" * 64
    with pytest.raises(gate.GateFailure, match="checkpoint bytes drift"):
        gate.validate_artifacts(
            official_panel, official_comparison, drifted, acdit_comparison
        )


def test_gate_output_is_strict_json():
    result = gate.validate_artifacts(*artifacts())
    json.dumps(result, allow_nan=False)
