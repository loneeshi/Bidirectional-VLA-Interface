import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bvi.s2_offline_decomposition import SCHEMA, canonical_sha256, combination_spec


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/finalize_native_s2_decomposition.py"
SPEC = importlib.util.spec_from_file_location("finalize_native_s2_decomposition", SCRIPT)
finalize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalize)


def _artifacts():
    roster = []
    for window, family in enumerate(("reach", "grasp", "move", "release")):
        roster.append({
            "row_id": f"row-{family}", "window": window, "index": 0, "family": family,
            "loss_rng_seed": 123 + window * 1000,
            "sample_rng_seed": 700123 + window * 1000,
            "noise_rng_seed": 900123 + window * 1000,
            "action_label_valid": True,
        })
    head = {0: "a" * 64, 20: "b" * 64}
    bank = {
        0: {family: str(index + 1) * 64 for index, family in enumerate(
            ("reach", "grasp", "move", "release")
        )},
        20: {family: chr(97 + index) * 64 for index, family in enumerate(
            ("reach", "grasp", "move", "release")
        )},
    }
    rows = []
    for combination in combination_spec():
        for source in roster:
            rows.append({
                **source, "combination": combination["name"],
                "head_step": combination["head_step"], "bank_step": combination["bank_step"],
                "head_sha256": head[combination["head_step"]],
                "bank_sha256": bank[combination["bank_step"]][source["family"]],
                "action_loss": 1.0, "progress_loss": 2.0, "joint_loss": 1.2,
                "deployment_parity": {
                    "passed": True, "atol": 0.0, "observation_sha256": "c" * 64,
                    "max_abs_diff": {"normalized_actions": 0.0, "external_actions": 0.0, "progress": 0.0},
                    "bit_exact": {"normalized_actions": True, "external_actions": True, "progress": True},
                },
                "first_action": {
                    "eligible": True, "predicted_raw": [0.0] * 13,
                    "raw_oob_fraction": 0.0, "physical_rmse": 0.25,
                    "sign_diagnostics": {
                        name: {"match": True} for name in ("gripper", "torso_lift", "base_yaw")
                    },
                },
            })
    return roster, rows, head, bank


def test_known_metadata_failure_is_only_recoverable_status():
    result = {
        "schema": SCHEMA, "status": "failed", "optimizer_updates": 0,
        "simulator_steps": 0, "native_success_evaluated": False,
        "capability_admission": False, "wall_seconds": 792.105,
        "error": "FileNotFoundError(2, 'No such file or directory')",
    }
    assert finalize.validate_failed_result(result, expected_rows=276)["wall_seconds"] == 792.105
    with pytest.raises(ValueError, match="known post-evaluation"):
        finalize.validate_failed_result(dict(result, error="RuntimeError('model failed')"), expected_rows=276)


def test_roster_and_rows_require_complete_2x2_matrix():
    roster, rows, head, bank = _artifacts()
    document = {"schema": SCHEMA, "roster_sha256": canonical_sha256(roster), "rows": roster}
    returned, identities = finalize.validate_roster_and_rows(document, rows, expected_rows=16)
    assert returned == roster
    assert identities == {"head_sha256": head, "bank_sha256": bank}
    with pytest.raises(ValueError, match="row count"):
        finalize.validate_roster_and_rows(document, rows[:-1], expected_rows=16)


def test_prediction_archive_is_bound_to_row_first_actions(tmp_path):
    roster, rows, _, _ = _artifacts()
    values = {}
    for combination in combination_spec():
        name = combination["name"]
        values[f"{name}_normalized_actions"] = np.zeros((4, 10, 32), np.float32)
        values[f"{name}_external_actions"] = np.zeros((4, 10, 13), np.float32)
        values[f"{name}_progress"] = np.zeros((4, 10), np.float32)
    path = tmp_path / "predictions.npz"
    np.savez_compressed(path, **values)
    assert finalize.validate_predictions(path, rows, roster)["passed"] is True
    values["H20_B20_external_actions"][3, 0, 12] = 0.5
    np.savez_compressed(path, **values)
    with pytest.raises(ValueError, match="first action"):
        finalize.validate_predictions(path, rows, roster)


def test_queue_events_recompute_combination_identity():
    roster, rows, head, bank = _artifacts()
    frozen = "f" * 64
    events = []
    for combination in combination_spec():
        name, hs, bs = combination["name"], combination["head_step"], combination["bank_step"]
        combination_hash = canonical_sha256({
            "frozen": frozen, "head": head[hs], "banks": bank[bs],
            "head_step": hs, "bank_step": bs,
        })
        previous = None
        for index, source in enumerate(roster):
            selected = bank[bs][source["family"]]
            events.append({
                "call_id": f"{name}-{source['row_id']}", "combination": name,
                "previous_family": previous, "selected_family": source["family"],
                "selected_bank_sha256": selected, "response_adapter_sha256": selected,
                "response_progress_head_sha256": head[hs],
                "response_combination_sha256": combination_hash,
                "discarded_actions": 0 if index == 0 else 10,
                "queue_empty_before_inference": True, "queued_actions_after_inference": 10,
                "transport_evaluated": False,
            })
            previous = source["family"]
    document = {"passed": True, "deployment_adapter_evaluated": True,
                "deployment_transport_evaluated": False, "events": events}
    identities = {"head_sha256": head, "bank_sha256": bank}
    assert finalize.validate_queue(document, rows, roster, identities, frozen_sha256=frozen)["events"] == 16
    events[-1]["discarded_actions"] = 9
    with pytest.raises(ValueError, match="Queue/routing"):
        finalize.validate_queue(document, rows, roster, identities, frozen_sha256=frozen)


def test_finalizer_help_is_runnable_and_exposes_source_paths():
    result = subprocess.run([sys.executable, str(SCRIPT), "--help"], cwd=SCRIPT.parents[1],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert "--evaluation-audit-source" in result.stdout
    assert "--expected-failed-result-sha256" in result.stdout


def test_finalizer_has_no_model_gpu_or_subprocess_surface():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    text = SCRIPT.read_text(encoding="utf-8")
    assert not ({"jax", "flax", "openpi", "optax", "subprocess", "pickle"} & imports)
    assert "nvidia-smi" not in text
    assert "value_and_grad" not in text


def test_existing_result_hash_is_never_silently_accepted():
    with pytest.raises(ValueError, match="Invalid"):
        finalize.require_sha256("not-a-hash", "failed result")


def test_end_to_end_finalization_preserves_failure_and_uses_no_model(tmp_path, monkeypatch):
    run = tmp_path / "run02"
    run.mkdir()
    roster, rows, head, bank = _artifacts()
    frozen = "f" * 64
    failed = {
        "schema": SCHEMA, "status": "failed", "optimizer_updates": 0,
        "simulator_steps": 0, "native_success_evaluated": False,
        "capability_admission": False, "wall_seconds": 792.105,
        "error": "FileNotFoundError(2, 'No such file or directory')",
    }
    (run / "result.json").write_text(json.dumps(failed), encoding="utf-8")
    (run / "roster.json").write_text(json.dumps({
        "schema": SCHEMA, "roster_sha256": canonical_sha256(roster), "rows": roster,
    }), encoding="utf-8")
    (run / "rows.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    predictions = {}
    for combination in combination_spec():
        name = combination["name"]
        predictions[f"{name}_normalized_actions"] = np.zeros((4, 10, 32), np.float32)
        predictions[f"{name}_external_actions"] = np.zeros((4, 10, 13), np.float32)
        predictions[f"{name}_progress"] = np.zeros((4, 10), np.float32)
    np.savez_compressed(run / "predictions.npz", **predictions)
    events = []
    for combination in combination_spec():
        name, hs, bs = combination["name"], combination["head_step"], combination["bank_step"]
        combination_hash = canonical_sha256({
            "frozen": frozen, "head": head[hs], "banks": bank[bs],
            "head_step": hs, "bank_step": bs,
        })
        previous = None
        for index, source in enumerate(roster):
            selected = bank[bs][source["family"]]
            events.append({
                "call_id": f"{name}-{source['row_id']}", "combination": name,
                "previous_family": previous, "selected_family": source["family"],
                "selected_bank_sha256": selected, "response_adapter_sha256": selected,
                "response_progress_head_sha256": head[hs],
                "response_combination_sha256": combination_hash,
                "discarded_actions": 0 if index == 0 else 10,
                "queue_empty_before_inference": True, "queued_actions_after_inference": 10,
                "transport_evaluated": False,
            })
            previous = source["family"]
    (run / "queue-audit.json").write_text(json.dumps({
        "passed": True, "deployment_adapter_evaluated": True,
        "deployment_transport_evaluated": False, "events": events,
    }), encoding="utf-8")

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    paths = {}
    for name, payload in {
        "manifest": b"manifest", "step0": b"step0", "step20": b"step20",
        "normalizer": b"normalizer", "contract": b"contract",
        "audit_source": b"audit", "core_source": b"core", "deployment_source": b"deploy",
    }.items():
        paths[name] = evidence / name
        paths[name].write_bytes(payload)
    identity = {
        "manifest_sha256": finalize.sha256(paths["manifest"]), "npz_sha256": {},
        "normalizer_sha256": finalize.sha256(paths["normalizer"]),
        "state_contract_sha256": finalize.sha256(paths["contract"]),
    }
    config = evidence / "config.json"
    config.write_text(json.dumps({
        "identity": identity, "frozen_sha256": frozen, "initial_bank_sha256": bank[0],
    }), encoding="utf-8")
    checkpoint = evidence / "checkpoint.json"
    checkpoint.write_text(json.dumps({
        "step": 20, "frozen_sha256": frozen, "bank_sha256": bank[20],
        "head_sha256": head[20],
    }), encoding="utf-8")
    best = evidence / "best.json"
    best.write_text(json.dumps({
        "step": 20, "sha256": finalize.sha256(paths["step20"]),
    }), encoding="utf-8")
    validation0 = evidence / "validation0.json"
    validation20 = evidence / "validation20.json"
    fields = ("window", "index", "family", "action_loss", "progress_loss", "joint_loss")
    validation0.write_text(json.dumps({"rows": [
        {key: row[key] for key in fields} for row in rows if row["combination"] == "H0_B0"
    ]}), encoding="utf-8")
    validation20.write_text(json.dumps({"rows": [
        {key: row[key] for key in fields} for row in rows if row["combination"] == "H20_B20"
    ]}), encoding="utf-8")

    original_result_hash = finalize.sha256(run / "result.json")
    argv = [str(SCRIPT), "--run", str(run), "--expected-rows", "16"]
    for key, filename in finalize.EXISTING.items():
        argv += [f"--expected-{key.replace('_', '-')}-sha256", finalize.sha256(run / filename)]
    argv += [
        "--data-manifest", str(paths["manifest"]), "--training-config", str(config),
        "--checkpoint-record", str(checkpoint), "--best-record", str(best),
        "--validation-step0", str(validation0), "--validation-step20", str(validation20),
        "--step0", str(paths["step0"]), "--step20", str(paths["step20"]),
        "--normalizer-stats", str(paths["normalizer"]), "--state-contract", str(paths["contract"]),
        "--evaluation-audit-source", str(paths["audit_source"]),
        "--core-source", str(paths["core_source"]),
        "--deployment-source", str(paths["deployment_source"]),
        "--expected-evaluation-audit-source-sha256", finalize.sha256(paths["audit_source"]),
        "--expected-core-source-sha256", finalize.sha256(paths["core_source"]),
        "--expected-deployment-source-sha256", finalize.sha256(paths["deployment_source"]),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert finalize.main() == 0
    assert finalize.sha256(run / "result.json") == original_result_hash
    final = json.loads((run / "finalization-result.json").read_text())
    assert final["recovered_completed_rows"] == 16
    assert final["model_inference_calls"] == 0
    assert final["gpu_queried"] is False
