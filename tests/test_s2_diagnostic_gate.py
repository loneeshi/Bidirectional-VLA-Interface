import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from bvi.first_action_probe import GATE_SPEC
from bvi.same_key_confirmation import CONFIRMATION_SPEC, adjudicate_confirmed_tail
from bvi.s2_diagnostic_gate import (
    MAX_UPDATES,
    SCHEMA,
    SCOPE,
    STATUS,
    TRAINING_STAGE,
    sha256,
    validate_diagnostic_entry,
)
from bvi.transform_parity import REQUIRED_MODEL_LEAVES


REPO_ID = "bvi/s1-official-pick-medium-train"
PARAMS_SHA = "a" * 64
TRAINER = Path(__file__).resolve().parents[1] / "scripts" / "train_native_s2.py"


def write(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def fixture(tmp_path: Path):
    checkpoint = tmp_path / "best" / "855"
    assets = checkpoint / "assets" / Path(REPO_ID)
    normalizer = tmp_path / "normalizer"
    norm = {"state": {"q01": [0] * 24}, "actions": {"q01": [0] * 13}}
    norm_path = write(assets / "norm_stats.json", norm)
    write(normalizer / "norm_stats.json", norm)
    contract_path = write(
        assets / "bvi-state-contract.json",
        {
            "training_stage": TRAINING_STAGE,
            "training_repo": REPO_ID,
            "state_dim": 24,
            "action_dim": 13,
            "state_source": "env_native_agent",
            "base_camera": "fetch_head",
            "wrist_camera": "fetch_hand",
        },
    )
    best = write(
        tmp_path / "best.json",
        {
            "step": 855,
            "checkpoint": str(checkpoint.resolve()),
            "selection": "heldout_action_loss_only",
        },
    )
    identity = {
        "checkpoint": str(checkpoint.resolve()),
        "training_stage": TRAINING_STAGE,
        "pretrained_parameters_sha256": PARAMS_SHA,
        "normalizer_sha256": sha256(norm_path),
        "state_contract_sha256": sha256(contract_path),
        "rng_contract": "reset PRNGKey(episode_seed); key,subkey=split(key) per prediction",
    }
    seeds = [2024, 2025, 2026, 2027, 2028]
    per_seed = write(
        tmp_path / "probe" / "per-seed.json",
        [
            {
                "seed": seed,
                "decision": "stochastic_tail_not_supported",
                "sample_count": 16,
                "identity": {
                    "valid": True,
                    "model_identity_matched": True,
                    "first_sample_rng_matches": True,
                    "first_sample_reproduced": True,
                    "first_sample_reproduction_max_abs_raw": 0.0,
                    "exact_reproduction_atol": 1e-6,
                },
                "expected_first_rng": [seed, seed + 1],
                "observed_first_rng": [seed, seed + 1],
                "request_sha256": f"{index + 1:064x}",
                "model_identity": {
                    "status": "matched",
                    "matched": True,
                    "reasons": [],
                    "checks": {
                        field: {
                            "equal": True,
                            "current": identity[field],
                            "recorded": identity[field],
                        }
                        for field in (
                            "pretrained_parameters_sha256",
                            "normalizer_sha256",
                            "state_contract_sha256",
                        )
                    },
                },
            }
            for index, seed in enumerate(seeds)
        ],
    )
    summary = write(
        tmp_path / "probe" / "summary.json",
        {
            "status": "completed",
            "stage": "exact_start_first_action_stochastic_attribution",
            "training_updates": 0,
            "simulator_steps": 0,
            "rollout_episodes": 0,
            "api_calls": 0,
            "planned_inference_calls": 80,
            "completed_inference_calls": 80,
            "samples_per_seed": 16,
            "seeds": seeds,
            "current_model_identity": identity,
            "transform_parity": {
                "status": "bit_exact",
                "bit_exact": True,
                "artifact": "transform-parity.json",
            },
            "cohort_gate": {
                "decision": "stochastic_tail_not_supported",
                "contract_reasons": [],
            },
        },
    )
    samples = tmp_path / "probe" / "samples.npz"
    np.savez_compressed(
        samples,
        seeds=np.asarray(seeds),
        raw_first_actions=np.zeros((5, 16, 13), np.float32),
    )
    parity = write(
        tmp_path / "probe" / "transform-parity.json",
        {
            "status": "bit_exact",
            "bit_exact": True,
            "repo_id": REPO_ID,
            "cases": [
                {
                    "seed": seed,
                    "status": "bit_exact",
                    "bit_exact": True,
                    "required_leaves": list(REQUIRED_MODEL_LEAVES),
                    "missing_training": [],
                    "missing_server": [],
                    "training_only": [],
                    "server_only": [],
                    "leaves": [
                        {
                            "leaf": leaf,
                            "shape_equal": True,
                            "dtype_equal": True,
                            "array_equal": True,
                            "training_sha256": "b" * 64,
                            "server_sha256": "b" * 64,
                        }
                        for leaf in sorted(REQUIRED_MODEL_LEAVES)
                    ],
                }
                for seed in seeds
            ],
        },
    )
    manifest = write(
        tmp_path / "adjudication.json",
        {
            "schema": SCHEMA,
            "status": STATUS,
            "scope": SCOPE,
            "repo_id": REPO_ID,
            "authorization": {
                "max_optimizer_updates": MAX_UPDATES,
                "capability_admission": False,
                "bounded_sft_expansion": False,
                "online_evaluation": False,
            },
            "model_identity": identity,
            "s1_best_record": {"path": str(best), "sha256": sha256(best)},
            "first_action_probe": {
                "summary": {"path": str(summary), "sha256": sha256(summary)},
                "per_seed": {"path": str(per_seed), "sha256": sha256(per_seed)},
                "samples": {"path": str(samples), "sha256": sha256(samples)},
            },
            "preprocessing_parity": {"path": str(parity), "sha256": sha256(parity)},
        },
    )
    return {
        "manifest": manifest,
        "checkpoint": checkpoint,
        "normalizer": normalizer,
        "best": best,
        "summary": summary,
        "per_seed": per_seed,
        "samples": samples,
        "parity": parity,
    }


def alternate_fixture(tmp_path: Path):
    paths = fixture(tmp_path)
    summary = json.loads(paths["summary"].read_text())
    summary["cohort_gate"] = {
        "decision": "invalid_cohort",
        "contract_reasons": ["one_or_more_exact_reproduction_identity_failures"],
    }
    paths["summary"].write_text(json.dumps(summary))
    exploratory_rows = json.loads(paths["per_seed"].read_text())
    for row in exploratory_rows:
        row["decision"] = "identity_failed"
        row["identity"].update(
            valid=False,
            first_sample_reproduced=False,
            first_sample_reproduction_max_abs_raw=0.01,
        )
    paths["per_seed"].write_text(json.dumps(exploratory_rows))

    confirmation = tmp_path / "confirmation"
    confirmation.mkdir()
    identity = summary["current_model_identity"]
    rng_contract = identity["rng_contract"]
    values = {
        "server_source_sha256": "c" * 64,
        "author_commit": "d" * 40,
        "pretrained_parameters_sha256": identity["pretrained_parameters_sha256"],
        "normalizer_sha256": identity["normalizer_sha256"],
        "state_contract_sha256": identity["state_contract_sha256"],
        "rng_contract": rng_contract,
        "checkpoint": str(paths["checkpoint"].resolve()),
        "precision": "bfloat16",
        "denoising_steps": 10,
        "action_horizon": 10,
        "external_action_dim": 13,
    }
    reports = []
    anchors = []
    for row in exploratory_rows:
        seed = row["seed"]
        anchor = {
            "passed": True,
            "mutual_unique_nearest": True,
            "historical_to_exploratory_sample0_rmse": 0.001,
            "historical_to_nearest_different_key_rmse": 0.1,
            "sample0_to_nearest_different_key_rmse": 0.1,
            "historical_same_key_margin_ratio": 0.01,
            "sample0_same_key_margin_ratio": 0.01,
            "material_sign_flip_channels": [],
        }
        fresh = {
            "passed": True,
            "closer_to_both_same_key_anchors_than_any_different_key": True,
            "fresh_to_historical_rmse": 0.001,
            "fresh_to_exploratory_sample0_rmse": 0.001,
            "fresh_to_nearest_different_key_rmse": 0.1,
            "worst_same_key_margin_ratio": 0.01,
            "material_sign_flips_vs_historical": [],
            "material_sign_flips_vs_exploratory_sample0": [],
        }
        anchors.append({"seed": seed, **anchor})
        reports.append(
            {
                "seed": seed,
                "request_sha256": row["request_sha256"],
                "model_identity": {
                    "matched": True,
                    "reasons": [],
                    "checks": [
                        {
                            "field": field,
                            "equal": True,
                            "current": value,
                            "recorded": value,
                        }
                        for field, value in values.items()
                    ],
                },
                "expected_rng": row["expected_first_rng"],
                "fresh_rng": row["expected_first_rng"],
                "rng_matches": True,
                "anchor_preflight": anchor,
                "fresh_confirmation": fresh,
                "sac_reference_native_success": seed != 2024,
                "deployed_sac_distance_percentile": 0.9375 if seed == 2026 else 0.5,
                "sample_median_over_deployed": 0.5 if seed == 2026 else 1.0,
            }
        )
    confirmation_per_seed = write(confirmation / "per-seed.json", reports)
    confirmation_summary = write(
        confirmation / "summary.json",
        {
            "status": "completed",
            "stage": "fresh_server_same_key_cluster_confirmation",
            "scope": "five frozen calls only",
            "seeds": [2024, 2025, 2026, 2027, 2028],
            "planned_inference_calls": 5,
            "completed_inference_calls": 5,
            "training_updates": 0,
            "simulator_steps": 0,
            "rollout_episodes": 0,
            "api_calls": 0,
            "confirmation_spec_frozen_before_inference": dict(CONFIRMATION_SPEC),
            "tail_spec_unchanged": dict(GATE_SPEC),
            "exploratory_run": str(paths["summary"].parent.resolve()),
            "checkpoint": str(paths["checkpoint"].resolve()),
            "normalizer": str(paths["normalizer"].resolve()),
            "exploratory_artifact_sha256": {
                "summary.json": sha256(paths["summary"]),
                "transform-parity.json": sha256(paths["parity"]),
                "per-seed.json": sha256(paths["per_seed"]),
                "samples.npz": sha256(paths["samples"]),
            },
            "exploratory_transform_parity": "bit_exact",
            "anchor_preflight": anchors,
            "all_five_same_key_clusters_confirmed": True,
            "adjudication": adjudicate_confirmed_tail(reports),
            "outputs": ["per-seed.json", "fresh-actions.npz"],
        },
    )
    fresh_actions = confirmation / "fresh-actions.npz"
    np.savez_compressed(
        fresh_actions,
        seeds=np.asarray([2024, 2025, 2026, 2027, 2028]),
        raw_action_chunks=np.zeros((5, 10, 13), np.float32),
    )
    server_result = write(
        confirmation / "server" / "result.json",
        {
            "status": "stopped",
            "training_updates": 0,
            "progress_head": False,
            "tapt": False,
            "inference_calls": 5,
            "max_inference_calls": 5,
            "stop_reason": "explicit_shutdown",
        },
    )
    server_metadata = write(
        confirmation / "server" / "metadata.json",
        {
            **values,
            "max_inference_calls": 5,
            "training_updates": 0,
            "progress_head": False,
            "tapt": False,
            "state_contract": {"training_stage": TRAINING_STAGE},
        },
    )
    artifact_manifest = write(
        confirmation / "artifact-manifest.json",
        {
            "files": [
                {"path": str(path.relative_to(confirmation)).replace("\\", "/"), "sha256": sha256(path)}
                for path in (
                    fresh_actions,
                    confirmation_per_seed,
                    server_metadata,
                    server_result,
                    confirmation_summary,
                )
            ]
        },
    )
    manifest = json.loads(paths["manifest"].read_text())
    manifest["first_action_probe"]["summary"]["sha256"] = sha256(paths["summary"])
    manifest["first_action_probe"]["per_seed"]["sha256"] = sha256(paths["per_seed"])
    manifest["same_key_confirmation"] = {
        "summary": {"path": str(confirmation_summary), "sha256": sha256(confirmation_summary)},
        "per_seed": {"path": str(confirmation_per_seed), "sha256": sha256(confirmation_per_seed)},
        "fresh_actions": {"path": str(fresh_actions), "sha256": sha256(fresh_actions)},
        "artifact_manifest": {"path": str(artifact_manifest), "sha256": sha256(artifact_manifest)},
    }
    paths["manifest"].write_text(json.dumps(manifest))
    paths.update(
        confirmation_summary=confirmation_summary,
        confirmation_per_seed=confirmation_per_seed,
        fresh_actions=fresh_actions,
        confirmation_artifact_manifest=artifact_manifest,
        server_result=server_result,
        server_metadata=server_metadata,
    )
    return paths


def refresh_probe_reference(paths, name):
    manifest = json.loads(paths["manifest"].read_text())
    manifest["first_action_probe"][name]["sha256"] = sha256(paths[name])
    paths["manifest"].write_text(json.dumps(manifest))


def refresh_confirmation_reference(paths, name, relative):
    artifact_manifest = json.loads(paths["confirmation_artifact_manifest"].read_text())
    for row in artifact_manifest["files"]:
        if row["path"] == relative:
            row["sha256"] = sha256(paths[name])
            break
    else:
        raise AssertionError(relative)
    paths["confirmation_artifact_manifest"].write_text(json.dumps(artifact_manifest))
    manifest = json.loads(paths["manifest"].read_text())
    key = {
        "confirmation_summary": "summary",
        "confirmation_per_seed": "per_seed",
        "fresh_actions": "fresh_actions",
    }.get(name)
    if key is not None:
        manifest["same_key_confirmation"][key]["sha256"] = sha256(paths[name])
    manifest["same_key_confirmation"]["artifact_manifest"]["sha256"] = sha256(
        paths["confirmation_artifact_manifest"]
    )
    paths["manifest"].write_text(json.dumps(manifest))


def validate(paths, steps=20, manifest_sha=None):
    return validate_diagnostic_entry(
        paths["manifest"],
        manifest_sha or sha256(paths["manifest"]),
        paths["checkpoint"],
        paths["normalizer"],
        paths["best"],
        steps,
    )


def dataset(path: Path) -> Path:
    records = []
    for split in ("train", "validation"):
        for family in ("reach", "grasp", "move", "release"):
            source = path / f"{split}-{family}.npz"
            source.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                source,
                head_rgb=np.zeros((2, 128, 128, 3), np.uint8),
                wrist_rgb=np.zeros((2, 128, 128, 3), np.uint8),
                state=np.zeros((2, 24), np.float32),
                actions=np.zeros((2, 13), np.float32),
                progress=np.array([0, 1], np.float32),
                action_valid=np.array([1, 0], bool),
                progress_valid=np.ones(2, bool),
            )
            records.append(
                {
                    "path": source.name,
                    "sha256": sha256(source),
                    "split": split,
                    "family": family,
                    "instruction": f"{family} object",
                    "completion_evidence": "test",
                    "completion_verified": True,
                    "parent_episode": {"split": split, "family": family},
                }
            )
    write(path / "manifest.json", {"invocations": records})
    return path


def test_accepts_only_diagnostic_scope_and_returns_common_checkpoint_identity(tmp_path):
    paths = fixture(tmp_path)
    result = validate(paths)
    assert result["scope"] == SCOPE
    assert result["native"]["checkpoint_sha256"] == PARAMS_SHA
    assert result["capability_admission"] is False
    assert result["bounded_sft_expansion"] is False
    assert result["online_evaluation"] is False


def test_trainer_cpu_dry_run_reports_diagnostic_not_capability_admission(tmp_path):
    paths = fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(TRAINER),
            "--data",
            str(dataset(tmp_path / "data")),
            "--checkpoint",
            str(paths["checkpoint"]),
            "--normalizer",
            str(paths["normalizer"]),
            "--s1-best-record",
            str(paths["best"]),
            "--diagnostic-adjudication-manifest",
            str(paths["manifest"]),
            "--diagnostic-adjudication-sha256",
            sha256(paths["manifest"]),
            "--steps",
            "20",
            "--dry-run",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["status"] == "cpu_dataset_validated_no_training"
    assert report["entry_scope"] == SCOPE
    assert report["capability_admission"] is False


@pytest.mark.parametrize("steps", [0, 21])
def test_rejects_outside_twenty_update_total_cap(tmp_path, steps):
    with pytest.raises(ValueError, match="1..20"):
        validate(fixture(tmp_path), steps=steps)


def test_rejects_changed_manifest_or_referenced_evidence(tmp_path):
    paths = fixture(tmp_path)
    with pytest.raises(ValueError, match="manifest is missing or changed"):
        validate(paths, manifest_sha="f" * 64)
    paths = fixture(tmp_path / "second")
    summary = json.loads(paths["summary"].read_text())
    summary["completed_inference_calls"] = 63
    paths["summary"].write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="Changed first_action_probe.summary evidence"):
        validate(paths)


@pytest.mark.parametrize(
    ("target", "edit", "message"),
    [
        ("manifest", lambda x: x.update(status="draft"), "diagnostic adjudication contract"),
        (
            "manifest",
            lambda x: x["authorization"].update(online_evaluation=True),
            "authorization boundary",
        ),
        (
            "summary",
            lambda x: x["cohort_gate"].update(decision="stochastic_tail_supported"),
            "does not support an adaptation diagnostic",
        ),
        (
            "per_seed",
            lambda x: x[0]["identity"].update(first_sample_rng_matches=False),
            "exact request/RNG reproduction is incomplete",
        ),
        (
            "parity",
            lambda x: x.update(bit_exact=False),
            "preprocessing transform parity",
        ),
    ],
)
def test_fail_closed_contracts(tmp_path, target, edit, message):
    paths = fixture(tmp_path)
    path = paths[target]
    value = json.loads(path.read_text())
    edit(value)
    path.write_text(json.dumps(value))
    if target != "manifest":
        manifest = json.loads(paths["manifest"].read_text())
        record = (
            manifest["first_action_probe"]["summary"]
            if target == "summary"
            else manifest["first_action_probe"]["per_seed"]
            if target == "per_seed"
            else manifest["preprocessing_parity"]
        )
        record["sha256"] = sha256(path)
        paths["manifest"].write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=message):
        validate(paths)


def test_accepts_only_narrow_confirmed_invalid_cohort_exception(tmp_path):
    paths = alternate_fixture(tmp_path)
    result = validate(paths)
    assert result["scope"] == SCOPE
    assert result["first_action_probe"]["admission_mode"] == (
        "fresh_same_key_confirmation_required"
    )
    assert result["same_key_confirmation"]["decision"] == "stochastic_tail_not_supported"
    assert result["same_key_confirmation"]["inference_calls"] == 5
    assert result["capability_admission"] is False
    assert result["bounded_sft_expansion"] is False
    assert result["online_evaluation"] is False


@pytest.mark.parametrize(
    ("reason", "message"),
    [
        (["case_count_mismatch"], "does not support an adaptation diagnostic"),
        (
            [
                "one_or_more_exact_reproduction_identity_failures",
                "case_count_mismatch",
            ],
            "does not support an adaptation diagnostic",
        ),
    ],
)
def test_rejects_any_other_invalid_cohort_reason(tmp_path, reason, message):
    paths = alternate_fixture(tmp_path)
    summary = json.loads(paths["summary"].read_text())
    summary["cohort_gate"]["contract_reasons"] = reason
    paths["summary"].write_text(json.dumps(summary))
    refresh_probe_reference(paths, "summary")
    with pytest.raises(ValueError, match=message):
        validate(paths)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_identity_matched", False),
        ("first_sample_rng_matches", False),
        ("first_sample_reproduced", True),
    ],
)
def test_invalid_cohort_exception_requires_only_reproduction_to_fail(tmp_path, field, value):
    paths = alternate_fixture(tmp_path)
    rows = json.loads(paths["per_seed"].read_text())
    rows[0]["identity"][field] = value
    paths["per_seed"].write_text(json.dumps(rows))
    refresh_probe_reference(paths, "per_seed")
    with pytest.raises(ValueError, match="solely a first-sample reproduction failure"):
        validate(paths)


def test_rejects_confirmation_that_does_not_bind_exploratory_samples(tmp_path):
    paths = alternate_fixture(tmp_path)
    summary = json.loads(paths["confirmation_summary"].read_text())
    summary["exploratory_artifact_sha256"]["samples.npz"] = "f" * 64
    paths["confirmation_summary"].write_text(json.dumps(summary))
    refresh_confirmation_reference(paths, "confirmation_summary", "summary.json")
    with pytest.raises(ValueError, match="exploratory artifact binding"):
        validate(paths)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda rows: rows[0].update(request_sha256="f" * 64),
            "request identity differs",
        ),
        (
            lambda rows: rows[0]["model_identity"]["checks"][0].update(equal=False),
            "model identity mismatch",
        ),
        (
            lambda rows: rows[0]["fresh_confirmation"].update(passed=False),
            "fresh confirmation cluster",
        ),
    ],
)
def test_rejects_failed_or_identity_changed_confirmation(tmp_path, mutation, message):
    paths = alternate_fixture(tmp_path)
    rows = json.loads(paths["confirmation_per_seed"].read_text())
    mutation(rows)
    paths["confirmation_per_seed"].write_text(json.dumps(rows))
    refresh_confirmation_reference(paths, "confirmation_per_seed", "per-seed.json")
    with pytest.raises(ValueError, match=message):
        validate(paths)


def test_rejects_confirmation_with_more_or_fewer_than_five_calls(tmp_path):
    paths = alternate_fixture(tmp_path)
    result = json.loads(paths["server_result"].read_text())
    result["inference_calls"] = 4
    paths["server_result"].write_text(json.dumps(result))
    refresh_confirmation_reference(paths, "server_result", "server/result.json")
    with pytest.raises(ValueError, match="server call budget"):
        validate(paths)


def test_rejects_fresh_rng_that_does_not_match_exploratory_identity(tmp_path):
    paths = alternate_fixture(tmp_path)
    rows = json.loads(paths["confirmation_per_seed"].read_text())
    rows[0]["expected_rng"] = [99, 100]
    rows[0]["fresh_rng"] = [99, 100]
    paths["confirmation_per_seed"].write_text(json.dumps(rows))
    refresh_confirmation_reference(paths, "confirmation_per_seed", "per-seed.json")
    with pytest.raises(ValueError, match="RNG identity differs"):
        validate(paths)


def test_rejects_supported_tail_as_diagnostic_entry(tmp_path):
    paths = alternate_fixture(tmp_path)
    summary = json.loads(paths["confirmation_summary"].read_text())
    summary["adjudication"]["decision"] = "stochastic_tail_supported"
    paths["confirmation_summary"].write_text(json.dumps(summary))
    refresh_confirmation_reference(paths, "confirmation_summary", "summary.json")
    with pytest.raises(ValueError, match="final adjudication was not reproduced"):
        validate(paths)
