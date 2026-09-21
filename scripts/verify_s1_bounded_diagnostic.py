"""Fresh-process CPU evidence verification; never starts a candidate or rollout.

Default mode validates completed-run evidence, saved checkpoint metadata and the
preregistered reconstruction rule. --verify-checkpoint additionally restores the
full parameter tree on CPU and checks its keys/shapes/dtypes against the pinned
PI05 architecture. It does not run a forward pass or initialize an optimizer.
Incomplete/timeout/invalid evidence is not scored as a policy failure.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# Set before any optional JAX/OpenPI import, even if the caller exported CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

ACTIVE = (0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12)
REPO_ID = "bvi/s1-official-pick-medium-train"
AUTHOR_COMMIT = "f4eb160ba52b22c1e85fe432de59c24bbbac6187"
GO_RULE = ("Both train first-action and valid-chunk active11 pooled RMSE improve by at least20%; "
           "neither reach yaw nor torso first-action RMSE worsens by more than10%; "
           "this is only eligibility for one candidate, never native success")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError("Nonfinite JSON value: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def row_keys(rows):
    keys = [[row[k] for k in ("parent_id", "call_index", "observation_index")] for row in rows]
    require(all(type(value) is int and value >= 0 for key in keys for value in key),
            "Row identifiers must be nonnegative integers")
    require(len({tuple(key) for key in keys}) == len(keys), "Duplicate row identifiers")
    return np.asarray(keys, dtype=np.int64)


def read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"predicted", "predicted_unclipped", "target", "valid", "families", "row_keys"}
        require(required <= set(archive.files), "Missing reconstruction arrays: " + str(path))
        data = {key: archive[key].copy() for key in required}
    shape = data["predicted"].shape
    require(len(shape) == 3 and 64 <= shape[0] <= 128 and shape[1:] == (10, 13),
            "Expected 64-128 rows of ten Fetch13 actions")
    for key in ("predicted", "predicted_unclipped", "target"):
        value = data[key]
        require(value.shape == shape and np.issubdtype(value.dtype, np.floating)
                and np.isfinite(value).all(), "Invalid numeric action array: " + key)
    valid = data["valid"]
    require(valid.dtype == np.bool_ and valid.shape == shape[:2] and valid[:, 0].all(),
            "Every row must have a valid first action and a Boolean time mask")
    require(np.array_equal(valid, np.arange(10)[None, :] < valid.sum(axis=1)[:, None]),
            "Time validity must be a contiguous prefix")
    require(data["families"].shape == (shape[0],)
            and set(data["families"].tolist()) == {"reach", "grasp", "move"},
            "Exactly reach/grasp/move must be represented")
    keys = data["row_keys"]
    require(keys.shape == (shape[0], 3) and np.issubdtype(keys.dtype, np.integer)
            and np.all(keys >= 0), "Invalid row key array")
    require(len({tuple(key) for key in keys.tolist()}) == shape[0], "Duplicate array row keys")
    applied = np.clip(data["predicted_unclipped"], -1, 1)
    applied[:, :, 8:10] = 0
    require(np.array_equal(applied, data["predicted"]), "Prediction differs from clip/head-mask contract")
    require(np.all(np.abs(data["target"]) <= 1) and np.all(data["target"][:, :, 8:10] == 0),
            "Target differs from applied Fetch13/head-mask contract")
    return data


def paired_metrics(before, after):
    for key in ("target", "valid", "families", "row_keys"):
        require(before[key].dtype == after[key].dtype and np.array_equal(before[key], after[key]),
                "Before/after identity mismatch: " + key)
    result = {}
    for label, data in (("before", before), ("after", after)):
        error = data["predicted"].astype(np.float64) - data["target"].astype(np.float64)
        reach = data["families"] == "reach"
        result[label] = {
            "first_action_active11_rmse": float(np.sqrt(np.mean(error[:, 0, ACTIVE] ** 2))),
            "valid_chunk_active11_rmse": float(np.sqrt(np.mean(error[data["valid"]][:, ACTIVE] ** 2))),
            "reach_yaw_first_rmse": float(np.sqrt(np.mean(error[reach, 0, 12] ** 2))),
            "reach_torso_first_rmse": float(np.sqrt(np.mean(error[reach, 0, 10] ** 2))),
        }
    return result


def evaluate_rule(metrics):
    checks = {}
    for name, ceiling in (("first_action_active11_rmse", .8), ("valid_chunk_active11_rmse", .8),
                          ("reach_yaw_first_rmse", 1.1), ("reach_torso_first_rmse", 1.1)):
        before, after = metrics["before"][name], metrics["after"][name]
        require(np.isfinite([before, after]).all() and min(before, after) >= 0, "Invalid RMSE")
        improvement = name.endswith("active11_rmse")
        # An already-zero baseline cannot establish a relative 20% improvement.
        passed = after <= ceiling * before and (before > 0 or not improvement)
        checks[name] = dict(before=before, after=after, maximum_ratio=ceiling,
                            ratio=None if before == 0 else after / before, passed=bool(passed),
                            zero_baseline=before == 0)
    return dict(passed=all(check["passed"] for check in checks.values()), checks=checks,
                rule=GO_RULE, native_success_established=False,
                scope="training_mapping_only_not_generalization_or_native_success")


def parameter_digest(flat):
    digest = hashlib.sha256()
    for key, value in sorted(flat.items()):
        array = np.asarray(value)
        digest.update("/".join(map(str, key)).encode())
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def verify_parameter_tree(checkpoint, status):
    """Explicit opt-in: CPU restore and abstract shape creation, no forward."""
    import dataclasses
    import jax
    import flax.nnx as nnx
    from flax import traverse_util
    import openpi
    from openpi.models import model as models, pi0_config

    require(all(device.platform == "cpu" for device in jax.devices()), "CPU-only restore required")
    author = Path(openpi.__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "-C", str(author), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(author), "status", "--porcelain",
                                     "--untracked-files=no"], text=True)
    require(commit == AUTHOR_COMMIT and not dirty.strip(), "Pinned clean OpenPI checkout required")
    cfg = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=True,
                              paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    cfg = dataclasses.replace(cfg, enable_progress_head=False)
    reference = traverse_util.flatten_dict(nnx.state(nnx.eval_shape(cfg.create, jax.random.key(7))).to_pure_dict())
    loaded = models.restore_params(checkpoint / "params", restore_type=np.ndarray)
    flat = traverse_util.flatten_dict(loaded)
    require(set(flat) == set(reference), "Saved parameter keys differ from the no-progress shared-LoRA model")
    require(all(value.shape == reference[key].shape for key, value in flat.items()),
            "Saved parameter shapes differ from the pinned model")
    # Frozen weights may legitimately be bf16 while native shape specs are f32;
    # preserve restored dtypes and check the permitted floating storage types.
    require(all(str(value.dtype) in {"float32", "bfloat16"} and np.isfinite(value).all()
                for value in flat.values()), "Invalid/nonfinite saved parameter dtype or value")
    require(not any("progress_chunk" in "/".join(map(str, key)) for key in flat),
            "Unexpected progress-head parameters")
    digest = parameter_digest(flat)
    expected = status.get("checkpoint_parameters_sha256")
    if expected is not None:
        require(digest == expected, "Restored parameter hash differs from the completed worker")
    return dict(status="passed_cpu_restore", author_commit=commit, tensors=len(flat),
                parameters_sha256=digest, worker_hash_present=expected is not None,
                worker_hash_matches=None if expected is None else True,
                dtype_counts={dtype: sum(str(x.dtype) == dtype for x in flat.values())
                              for dtype in sorted({str(x.dtype) for x in flat.values()})},
                schema_sha256=hashlib.sha256(json.dumps([
                    ["/".join(map(str, key)), list(value.shape), str(value.dtype)]
                    for key, value in sorted(flat.items())]).encode()).hexdigest(),
                forward_calls=0, gpu_used=False)


def verify_run(args):
    run = args.run_dir.resolve()
    report = dict(schema="bvi.s1-bounded-diagnostic-decision/1", run_dir=str(run),
                  verified_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
                  verifier_sha256=sha256(__file__), status="verifying", decision="not_assessed",
                  candidate_eligible=False, candidate_started=False, policy_failure=False,
                  gpu_used=False, forward_calls=0, training_updates=0, api_calls=0,
                  checkpoint_verification={"status": "not_requested"}, evidence_sha256={})
    try:
        def document(name):
            path = run / name
            report["evidence_sha256"][name] = sha256(path)
            return read_json(path)
        supervisor = document("supervisor.json")
        status = document("status.json")
        report["run_status"] = status.get("status")
        report["supervisor_status"] = supervisor.get("status")
        if (supervisor.get("status") != "exited" or supervisor.get("worker_exit_code") != 0
                or status.get("status") != "completed_diagnostic_requires_review"):
            report.update(status="run_not_evaluable", decision="no_go_incomplete_or_infrastructure",
                          reason="Run did not complete normally; no policy failure or RMSE gate inferred")
            return report
        identity = document("identity.json")
        rows = document("selected-rows.json")
        panel_path = args.panel or run / "frozen-panel.json"
        panel = read_json(panel_path)
        report["panel_sha256"] = sha256(panel_path)
        require(report["panel_sha256"] == identity["panel_sha256"], "Original frozen-panel bytes differ")
        contract = read_json(args.contract)
        report["execution_contract_sha256"] = sha256(args.contract)
        require(contract.get("status") == "frozen_before_diagnostic"
                and contract["diagnostic"]["go_rule"] == GO_RULE, "Unexpected frozen diagnostic rule")
        bounds = contract["diagnostic"]
        updates = status.get("additional_updates")
        require(type(updates) is int and 1 <= updates <= identity["max_updates"] <= bounds["max_updates"] <= 100,
                "Update budget or completed update count invalid")
        require(0 < supervisor["elapsed_seconds"] <= supervisor["budget_seconds"]
                == identity["total_seconds"] <= bounds["max_seconds_including_load_compile_save"] <= 3600,
                "Whole-process time budget exceeded or inconsistent")
        require(status.get("checkpoint_complete") is True and status.get("progress") is False
                and status.get("family_bank") is False and status.get("api_calls") == 0,
                "Checkpoint/progress/family/API contract differs")
        require(identity.get("source_checkpoint_step") == panel.get("checkpoint_step") == 855,
                "Diagnostic source is not frozen best/855")
        for field in ("source_sha256", "source_manifest_sha256", "normalizer_sha256"):
            require(identity[field] == panel[field], "Panel/identity provenance mismatch: " + field)
        require(panel.get("checkpoint_parameters_sha256") == status.get("source_parameters_sha256")
                and isinstance(panel.get("checkpoint_parameters_sha256"), str)
                and len(panel["checkpoint_parameters_sha256"]) == 64, "Source parameter hash not frozen/matched")
        expected_keys = row_keys(rows)
        require(np.array_equal(expected_keys, row_keys(panel["rows"])), "Selected rows differ from frozen panel")
        require(64 <= len(rows) <= 128 and 1 <= len(set(expected_keys[:, 0])) <= 2
                and all(row["split"] == "train" for row in rows), "Unexpected diagnostic training roster")
        before, after = read_arrays(run / "before.npz"), read_arrays(run / "after.npz")
        report["evidence_sha256"].update({name: sha256(run / name) for name in ("before.npz", "after.npz")})
        require(np.array_equal(before["row_keys"], expected_keys), "Prediction rows differ from frozen panel")
        require(np.array_equal(before["families"], np.asarray([row["tool_family"] for row in rows]))
                and np.array_equal(before["valid"], np.asarray([row["action_valid"] for row in rows], dtype=bool)),
                "Prediction family/time mask differs from selected source rows")
        metrics = paired_metrics(before, after)
        report.update(metrics=metrics, metric_gate=evaluate_rule(metrics), rows=len(rows),
                      valid_actions=int(before["valid"].sum()), paired_arrays_exact=True)
        checkpoint = (args.checkpoint or run / "diagnostic" / str(updates)).resolve()
        require(checkpoint.name == str(updates) and (checkpoint / "params").is_dir(),
                "Saved checkpoint directory/step missing")
        assets = checkpoint / "assets" / REPO_ID
        metadata = read_json(assets / "bvi-state-contract.json")
        require(metadata == identity["policy_metadata"], "Saved metadata differs from resolved runtime metadata")
        expected_metadata = dict(robot="fetch", state_dim=24, action_dim=13,
                                 state_components=["native_qpos12", "native_qvel12"],
                                 state_source="env_native_agent", state_conditioning=True,
                                 base_position_reference="world", base_camera="fetch_head",
                                 wrist_camera="fetch_hand", training_repo=REPO_ID,
                                 action_convention="Fetch13_normalized_pd_joint_delta_pos_body_base_forward_velocity",
                                 diagnostic_only=True, training_stage="S1_IA_single_bank_no_progress",
                                 source_checkpoint_step=855, invocation_aligned=True,
                                 optimizer_reset=True, lr_clock="reset_constant_1e-4")
        require(all(metadata.get(key) == value for key, value in expected_metadata.items()),
                "Saved metadata is not the expected native24 S1-IA diagnostic")
        norm_hash = sha256(assets / "norm_stats.json")
        require(norm_hash == identity["normalizer_sha256"] == metadata["normalizer_sha256"],
                "Saved normalizer bytes differ from frozen source")
        report["checkpoint"] = dict(path=str(checkpoint), step=updates, normalizer_sha256=norm_hash,
                                     metadata_sha256=sha256(assets / "bvi-state-contract.json"))
        if args.verify_checkpoint:
            report["checkpoint_verification"] = verify_parameter_tree(checkpoint, status)
        if not report["metric_gate"]["passed"]:
            report.update(status="verified_no_go", decision="no_go_training_mapping_thresholds_not_met")
        elif not args.verify_checkpoint:
            report.update(status="metric_gate_passed_checkpoint_load_pending",
                          decision="no_go_until_fresh_cpu_checkpoint_restore")
        else:
            report.update(status="verified_metric_go", decision="eligible_for_one_bounded_candidate_review",
                          candidate_eligible=True)
        return report
    except Exception as exc:
        report.update(status="evidence_invalid", decision="no_go_insufficient_or_infrastructure_evidence",
                      candidate_eligible=False, error=repr(exc))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True,
                        help="Pre-run execution-contract.json containing the frozen diagnostic go rule")
    parser.add_argument("--panel", type=Path, help="Original frozen-panel bytes if not preserved in run directory")
    parser.add_argument("--checkpoint", type=Path, help="Relocated diagnostic/<update> directory, if applicable")
    parser.add_argument("--verify-checkpoint", action="store_true", help="CPU tree restore only; requires pinned OpenPI")
    parser.add_argument("--output", type=Path, help="Default: <run-dir>/decision.json; existing files are preserved")
    args = parser.parse_args(argv)
    output = args.output or args.run_dir / "decision.json"
    if output.exists():
        parser.error("Output already exists; choose a new --output to preserve prior verification")
    report = verify_run(args)
    write_json(output, report)
    print(json.dumps({key: report[key] for key in ("status", "decision", "candidate_eligible", "policy_failure")}))
    return 0 if report["status"] in {"verified_no_go", "verified_metric_go", "metric_gate_passed_checkpoint_load_pending"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
