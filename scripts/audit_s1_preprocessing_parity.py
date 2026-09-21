"""CPU-only exact parity audit for saved S1 online preprocessing requests.

The script does not restore model parameters, sample actions, create a simulator,
or use a GPU.  For each saved ``step000.npz`` request it compares the training
pipeline and the frozen inference-server pipeline after repacking, Libero input
conversion, quantile normalization, and model input transforms.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
from datetime import datetime, timezone
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
import traceback

# This audit is intentionally incapable of selecting an accelerator even when
# invoked from a shell that previously exported a GPU-oriented JAX setting.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import numpy as np

from bvi.s1_preprocessing_parity import (
    LIBERO_REQUIRED_PATHS,
    MODEL_REQUIRED_PATHS,
    compare_shared_leaves,
    file_sha256,
    load_saved_request,
    quantile_state_ood,
    tree_summary,
)


DEFAULT_REPO_ID = "bvi/s1-official-pick-medium-train"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, action="append", required=True,
                        help="Saved step000.npz; repeat for each validation seed")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalizer", type=Path, required=True)
    parser.add_argument("--config-module", type=Path,
                        default=Path(__file__).with_name("fetch_native_s1_config.py"))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_config_module(path: Path):
    path = path.resolve()
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("s1_parity_runtime_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "config", None)):
        raise ValueError("Config module must export config()")
    return module


def apply_all(functions, value):
    result = copy.deepcopy(value)
    for function in functions:
        result = function(result)
    return result


def source_sha256(value) -> str | None:
    try:
        source = inspect.getsource(value).encode("utf-8")
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source).hexdigest()


def stage(name, training, server, *, required_roots=(), required_paths=()):
    return {
        "name": name,
        "comparison": compare_shared_leaves(
            training, server, required_roots=tuple(required_roots),
            required_paths=tuple(required_paths)),
        "training_summary": tree_summary(training),
        "server_summary": tree_summary(server),
    }


def raw_summary(request):
    raw = {
        "head_rgb": request["head_rgb"],
        "wrist_rgb": request["wrist_rgb"],
        "state": request["state"],
        "prompt": np.asarray(request["prompt"]),
    }
    return tree_summary(raw)


def norm_stats_summary(stats):
    """Summarize the state/action normalization contract without raw arrays."""
    result = {}
    for key, expected_dimension in (("state", 24), ("actions", 13)):
        entry = stats[key]
        fields = {
            name: np.asarray(getattr(entry, name))
            for name in ("mean", "std", "q01", "q99")
        }
        if any(value.shape != (expected_dimension,) for value in fields.values()):
            raise ValueError(f"Checkpoint {key} normalization width differs")
        if any(not np.isfinite(value).all() for value in fields.values()):
            raise ValueError(f"Checkpoint {key} normalization contains nonfinite values")
        result[key] = tree_summary(fields)
    return result


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    report = {
        "schema": "bvi.s1-preprocessing-parity/1",
        "status": "initializing",
        "question": "Are exact saved S1 online requests transformed identically by training and inference preprocessing?",
        "scope": "CPU preprocessing only; no model parameters, inference, simulator, rollout, training, or API",
        "started_utc": started,
        "training_updates": 0,
        "model_inference_calls": 0,
        "simulator_steps": 0,
        "api_calls": 0,
        "gpu_used": False,
        "requests": [],
    }
    try:
        request_paths = [path.resolve() for path in args.request]
        if len(set(request_paths)) != len(request_paths):
            raise ValueError("Duplicate saved request paths")
        for path in request_paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        checkpoint = args.checkpoint.resolve()
        normalizer = args.normalizer.resolve()
        config_path = args.config_module.resolve()
        checkpoint_stats = checkpoint / "assets" / args.repo_id / "norm_stats.json"
        state_contract = checkpoint / "assets" / args.repo_id / "bvi-state-contract.json"
        supplied_stats = normalizer / "norm_stats.json"
        supplied_provenance = normalizer / "provenance.json"
        for path in (
            checkpoint_stats, state_contract, supplied_stats,
            supplied_provenance, config_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
        contract = json.loads(state_contract.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise ValueError("State contract must be a JSON object")
        hashes = {
            "checkpoint_norm_stats_sha256": file_sha256(checkpoint_stats),
            "supplied_norm_stats_sha256": file_sha256(supplied_stats),
            "state_contract_sha256": file_sha256(state_contract),
            "supplied_provenance_sha256": file_sha256(supplied_provenance),
            "config_module_sha256": file_sha256(config_path),
        }
        hashes["normalizer_files_exact"] = (
            hashes["checkpoint_norm_stats_sha256"] == hashes["supplied_norm_stats_sha256"])
        if not hashes["normalizer_files_exact"]:
            raise ValueError("Checkpoint and supplied normalizer files differ")
        if contract.get("training_repo") != args.repo_id:
            raise ValueError("State contract training_repo differs from --repo-id")
        if contract.get("normalizer_sha256") != hashes["checkpoint_norm_stats_sha256"]:
            raise ValueError("State contract normalizer hash differs from checkpoint")

        # Imports below belong to the pinned OpenPI CPU environment.  Nothing
        # creates model parameters or invokes a sample function.
        from openpi import transforms
        from openpi.policies.libero_policy import LiberoInputs
        from openpi.training import checkpoints, config as training_config
        import jax

        devices = jax.devices()
        if not devices or any(device.platform != "cpu" for device in devices):
            raise RuntimeError("P0-A audit requires JAX CPU devices only")

        config_module = load_config_module(config_path)
        cfg = config_module.config(
            args.repo_id, str(output.parent / "unused-s1-parity-work"),
            str(checkpoint), normalizer, steps=1, batch=1)
        training_stage = contract.get("training_stage")
        if training_stage not in {
            "S1_ordinary_target_domain_SFT_not_TAPT",
            "S1_IA_single_bank_no_progress",
        }:
            raise ValueError("State contract has an unsupported training_stage")
        cfg = dataclasses.replace(
            cfg,
            policy_metadata=dict(cfg.policy_metadata, training_stage=training_stage),
        )
        model_config = cfg.model
        if hasattr(model_config, "enable_progress_head"):
            model_config = dataclasses.replace(model_config, enable_progress_head=False)
        data_config = cfg.data.create(cfg.assets_dirs, model_config)
        contract_keys = (
            "robot", "state_dim", "state_components", "action_dim",
            "base_position_reference", "base_camera", "wrist_camera",
            "state_conditioning", "training_repo", "action_convention",
            "state_source", "normalizer_sha256", "training_stage",
        )
        for key in contract_keys:
            if contract.get(key) != cfg.policy_metadata.get(key):
                raise ValueError(f"State contract/config mismatch: {key}")
        checkpoint_norm = checkpoints.load_norm_stats(checkpoint / "assets", args.repo_id)
        training_normalize = transforms.Normalize(
            data_config.norm_stats, use_quantiles=data_config.use_quantile_norm)
        server_libero = LiberoInputs(model_config.model_type)
        server_normalize = transforms.Normalize(checkpoint_norm, use_quantiles=True)
        server_model_functions = list(training_config.ModelTransformFactory()(model_config).inputs)
        training_model_functions = list(data_config.model_transforms.inputs)
        normalizer_summary = norm_stats_summary(checkpoint_norm)
        q01 = np.asarray(checkpoint_norm["state"].q01, np.float64)
        q99 = np.asarray(checkpoint_norm["state"].q99, np.float64)
        if q01.shape != (24,) or q99.shape != (24,):
            raise ValueError("Checkpoint state quantiles are not native24")

        report.update(
            status="running",
            checkpoint=str(checkpoint),
            normalizer=str(normalizer),
            repo_id=args.repo_id,
            config_module=str(config_path),
            provenance=hashes,
            state_contract=contract,
            normalizer_summary=normalizer_summary,
            jax_devices=[str(device) for device in devices],
            transform_sources={
                "LiberoInputs": source_sha256(LiberoInputs),
                "Normalize": source_sha256(transforms.Normalize),
                "ModelTransformFactory": source_sha256(training_config.ModelTransformFactory),
            },
            transform_classes={
                "training_repack": [type(fn).__name__ for fn in data_config.repack_transforms.inputs],
                "training_data": [type(fn).__name__ for fn in data_config.data_transforms.inputs],
                "training_model": [type(fn).__name__ for fn in training_model_functions],
                "server_data": [type(server_libero).__name__],
                "server_model": [type(fn).__name__ for fn in server_model_functions],
            },
        )

        for request_path in request_paths:
            request = load_saved_request(request_path)
            training_raw = {
                "image": request["head_rgb"].copy(),
                "wrist_image": request["wrist_rgb"].copy(),
                "state": request["state"].copy(),
                "prompt": request["prompt"],
                # Presence of actions follows the training loader.  They are
                # excluded from every parity comparison and cannot affect the
                # shared observation/token leaves.
                "actions": np.zeros((model_config.action_horizon, 13), np.float32),
            }
            server_raw = {
                "observation/image": request["head_rgb"].copy(),
                "observation/wrist_image": request["wrist_rgb"].copy(),
                "observation/state": request["state"].copy(),
                "prompt": request["prompt"],
            }
            training_repacked = apply_all(data_config.repack_transforms.inputs, training_raw)
            training_libero = apply_all(data_config.data_transforms.inputs, training_repacked)
            inference_libero = server_libero(copy.deepcopy(server_raw))
            training_normalized = training_normalize(copy.deepcopy(training_libero))
            inference_normalized = server_normalize(copy.deepcopy(inference_libero))
            training_final = apply_all(training_model_functions, training_normalized)
            inference_final = apply_all(server_model_functions, inference_normalized)
            stages = [
                stage("training_repack_vs_server_raw", training_repacked, server_raw,
                      required_paths=("observation/image", "observation/wrist_image",
                                      "observation/state", "prompt")),
                stage("libero_inputs", training_libero, inference_libero,
                      required_paths=LIBERO_REQUIRED_PATHS),
                stage("quantile_normalized", training_normalized, inference_normalized,
                      required_paths=LIBERO_REQUIRED_PATHS),
                stage("model_inputs", training_final, inference_final,
                      required_paths=MODEL_REQUIRED_PATHS),
            ]
            report["requests"].append({
                "path": str(request_path),
                "file_sha256": file_sha256(request_path),
                "raw": raw_summary(request),
                "prompt": request["prompt"],
                "prompt_utf8_sha256": hashlib.sha256(request["prompt"].encode("utf-8")).hexdigest(),
                "state_quantile_ood": quantile_state_ood(request["state"], q01, q99),
                "stages": stages,
                "stage_exact": {item["name"]: item["comparison"]["exact"] for item in stages},
                "exact": all(item["comparison"]["exact"] for item in stages),
            })

        outside_counts = np.zeros(24, dtype=int)
        max_abs_normalized = np.zeros(24, dtype=float)
        for item in report["requests"]:
            for channel in item["state_quantile_ood"]["channels"]:
                index = channel["index"]
                outside_counts[index] += int(channel["outside_q01_q99"])
                max_abs_normalized[index] = max(
                    max_abs_normalized[index], abs(channel["normalized"]))
        report["state_ood_aggregate"] = {
            "requests": len(report["requests"]),
            "outside_request_counts_by_channel": outside_counts.tolist(),
            "max_abs_normalized_by_channel": max_abs_normalized.tolist(),
            "channels_ever_outside": np.flatnonzero(outside_counts).astype(int).tolist(),
        }
        report["exact"] = all(item["exact"] for item in report["requests"])
        report["status"] = "passed_exact" if report["exact"] else "failed_parity"
    except Exception as exc:
        report.update(status="infrastructure_failure", exact=False, error=repr(exc),
                      traceback=traceback.format_exc())
    output.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": report["status"],
        "exact": report.get("exact", False),
        "requests": len(report["requests"]),
        "output": str(output),
    }))
    return 0 if report["status"] == "passed_exact" else 1


if __name__ == "__main__":
    raise SystemExit(main())
