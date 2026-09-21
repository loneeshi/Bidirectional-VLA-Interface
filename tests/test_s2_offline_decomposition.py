import ast
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bvi.native_s2_deployment import NativeS2DeploymentAdapter
from bvi.s2_offline_decomposition import (
    assert_path_parity,
    combination_spec,
    compare_historical_rows,
    first_action_diagnostics,
    fixed_validation_roster,
    summarize_rows,
)


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_native_s2_decomposition.py"


def _dataset():
    records, tables = [], {}
    for family in ("reach", "grasp", "move", "release"):
        records.append({"split": "train", "family": family})
        tables[len(records) - 1] = {
            "state": np.zeros((2, 24)), "action_valid": np.array([1, 0]),
        }
        records.append({"split": "validation", "family": family})
        tables[len(records) - 1] = {
            "state": np.zeros((4, 24)), "action_valid": np.array([1, 1, 1, 0]),
        }
    return records, tables


def test_fixed_roster_matches_training_traversal_and_rng():
    records, tables = _dataset()
    rows = fixed_validation_roster(records, tables)
    assert len(rows) == 12
    assert [(row["window"], row["index"]) for row in rows[:3]] == [(1, 0), (1, 2), (1, 3)]
    assert rows[0]["loss_rng_seed"] == 1123
    assert rows[0]["sample_rng_seed"] == 701123
    assert rows[0]["noise_rng_seed"] == 901123
    assert rows[2]["action_label_valid"] is False


def test_2x2_spec_is_complete_and_unambiguous():
    spec = combination_spec()
    assert {(row["head_step"], row["bank_step"]) for row in spec} == {
        (0, 0), (20, 0), (0, 20), (20, 20)
    }
    assert len({row["name"] for row in spec}) == 4


def test_first_action_reports_physical_error_sign_and_oob():
    predicted = np.zeros(13)
    predicted[7], predicted[10], predicted[12] = -1.2, 0.5, -0.25
    expert = np.zeros(13)
    expert[7], expert[10], expert[12] = 1.0, -0.5, -0.25
    result = first_action_diagnostics(predicted, expert, eligible=True)
    assert result["raw_oob_channels"] == ["gripper"]
    assert result["physical_abs_error"][7] == pytest.approx(0.066)
    assert result["sign_diagnostics"]["gripper"]["match"] is False
    assert result["sign_diagnostics"]["torso_lift"]["match"] is False
    assert result["sign_diagnostics"]["base_yaw"]["match"] is True


def test_endpoint_keeps_oob_but_has_no_fake_expert_error():
    result = first_action_diagnostics(np.full(13, 1.1), np.zeros(13), eligible=False)
    assert result["raw_oob_count"] == 13
    assert result["expert_raw"] is None
    assert result["physical_rmse"] is None
    assert result["sign_diagnostics"] is None


def test_deployment_parity_is_exact_and_fails_closed():
    values = {
        "observation_sha256": "a" * 64,
        "normalized_actions": np.zeros((1, 10, 32)),
        "external_actions": np.zeros((10, 13)),
        "progress": np.zeros((1, 10)),
    }
    assert assert_path_parity(values, values)["passed"] is True
    changed = dict(values, external_actions=np.ones((10, 13)))
    with pytest.raises(ValueError, match="external_actions"):
        assert_path_parity(values, changed)
    with pytest.raises(ValueError, match="observation"):
        assert_path_parity(values, dict(values, observation_sha256="b" * 64))


def test_family_switch_discards_prior_chunk_and_binds_bank():
    adapter = NativeS2DeploymentAdapter(
        transform=lambda value: {**value, "state": value["observation/state"]},
        observation_factory=lambda value: value,
        predict=lambda family, key, observation, noise: (
            np.zeros((1, 10, 32)), np.zeros((1, 10))
        ),
        unnormalize=lambda value: value,
        observation_digest=lambda value: "d" * 64,
        bank_sha256={family: chr(97 + index) * 64 for index, family in enumerate(
            ("reach", "grasp", "move", "release")
        )},
        head_sha256="e" * 64,
        checkpoint_sha256="f" * 64,
        normalizer_sha256="1" * 64,
        state_contract_sha256="2" * 64,
    )
    first = adapter.begin(call_id="one", family="reach", instruction="Reach apple")
    assert first["discarded_actions"] == 0
    response = adapter.infer({
        "head_rgb": np.zeros((128, 128, 3), np.uint8),
        "wrist_rgb": np.zeros((128, 128, 3), np.uint8),
        "state": np.zeros(24, np.float32), "prompt": "Reach apple",
        "tool_family": "reach", "call_id": "one",
    }, rng_key=object(), noise=object())
    assert response["queued_actions"] == 10
    assert response["transport_evaluated"] is False
    second = adapter.begin(call_id="two", family="grasp", instruction="Grasp apple")
    assert second["discarded_actions"] == 10
    assert second["queue_empty_before_inference"] is True
    assert second["selected_bank_sha256"] == "b" * 64


def _metric_rows():
    rows = []
    for combination in ("H0_B0", "H20_B0", "H0_B20", "H20_B20"):
        for family in ("reach", "grasp", "move", "release"):
            rows.append({
                "combination": combination, "family": family,
                "action_loss": 1.0, "progress_loss": 2.0, "joint_loss": 1.2,
                "first_action": {
                    "eligible": True, "physical_rmse": 0.25, "raw_oob_fraction": 0.1,
                    "sign_diagnostics": {
                        name: {"match": True} for name in ("gripper", "torso_lift", "base_yaw")
                    },
                },
            })
    return rows


def test_summary_requires_every_combination_and_family():
    summary = summarize_rows(_metric_rows())
    assert summary["H20_B0"]["per_family"]["grasp"]["action_loss"] == 1.0
    assert summary["H0_B20"]["macro_joint_loss"] == pytest.approx(1.2)
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_rows(_metric_rows()[:-4])


def test_historical_comparison_is_roster_bound():
    actual = [{"window": 1, "index": 0, "family": "reach",
               "action_loss": 1.0, "progress_loss": 2.0, "joint_loss": 1.2}]
    historical = {"rows": [dict(actual[0])]}
    assert compare_historical_rows(actual, historical)["passed"] is True
    historical["rows"][0]["joint_loss"] = 1.3
    with pytest.raises(ValueError, match="reproduction"):
        compare_historical_rows(actual, historical)


def test_gpu_entry_has_no_optimizer_or_gradient_surface():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = {
        alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
        for alias in node.names
    }
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert "optax" not in imported | names
    assert "value_and_grad" not in attributes | names
    assert "grad" not in names


def test_gpu_entry_help_does_not_import_openpi():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"], text=True, capture_output=True,
        cwd=SCRIPT.parents[1], timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "zero optimizer updates" in result.stdout
    assert "--handoff-manifest" in result.stdout


def test_behavior_mode_does_not_apply_a_roster_bound_before_sequence_load():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "if not behavior_mode and len(roster) > args.max_rows:" in source
    assert "if sequence_report[\"frame_count\"] > args.max_rows:" in source
