"""Probe whether deployed S1 first actions were repeated stochastic tail draws.

The fixed experiment is five exact starts (seeds 2024--2028), sixteen frozen
policy samples per saved ``step000`` request, and one same-start official SAC
reference action per seed.  The SAC cohort is 4/5 native-success; seed 2024 is
retained and explicitly labelled as a failure.  No simulator is created and no
parameter is updated.  SAC is a reference policy, not a supervised action label.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

import numpy as np

from bvi.closed_loop_divergence import run_identity, strict_pairing
from bvi.first_action_probe import (
    GATE_SPEC,
    analyze_first_action_distribution,
    analyze_probe_cohort,
    applied_actions,
    first_event_actions,
    model_identity_checks,
)
from bvi.transform_parity import compare_shared_model_inputs


GPU1_UUID = "GPU-b7ebba23-7824-7601-df32-be55628936c3"
SEEDS = (2024, 2025, 2026, 2027, 2028)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _first_event(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("step") != 1:
                    raise ValueError(f"First event is not step 1: {path}")
                return row
    raise ValueError(f"Empty event stream: {path}")


def _load_request(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"head_rgb", "wrist_rgb", "state", "prompt"}:
            raise ValueError(f"Unexpected saved request keys in {path}: {sorted(data.files)}")
        head = np.asarray(data["head_rgb"])
        wrist = np.asarray(data["wrist_rgb"])
        state = np.asarray(data["state"], np.float32)
        prompt_array = np.asarray(data["prompt"])
        if prompt_array.shape != ():
            raise ValueError(f"Prompt must be a scalar string in {path}")
        prompt = prompt_array.item()
    if head.shape != (128, 128, 3) or wrist.shape != (128, 128, 3):
        raise ValueError(f"Saved RGB contract mismatch in {path}")
    if head.dtype != np.uint8 or wrist.dtype != np.uint8:
        raise ValueError(f"Saved RGB dtype mismatch in {path}")
    if state.shape != (24,) or not np.isfinite(state).all():
        raise ValueError(f"Saved state24 contract mismatch in {path}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError(f"Saved prompt contract mismatch in {path}")
    return {"head_rgb": head, "wrist_rgb": wrist, "state": state, "prompt": prompt}


def _case(seed: int, policy_root: Path, sac_root: Path) -> dict:
    policy = policy_root / f"reach-seed{seed}"
    sac = sac_root / f"d1-a1-sac-seed{seed}"
    paths = {
        "request": policy / "requests" / "step000.npz",
        "policy_events": policy / "events.jsonl",
        "policy_result": policy / "result.json",
        "sac_events": sac / "events.jsonl",
        "sac_result": sac / "result.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Seed {seed} inputs missing: {missing}")
    policy_result = _json(paths["policy_result"])
    sac_result = _json(paths["sac_result"])
    if policy_result.get("seed") != seed or sac_result.get("seed") != seed:
        raise ValueError(f"Seed identity mismatch for case {seed}")
    if not isinstance(sac_result.get("success"), bool):
        raise ValueError(f"Seed {seed} SAC reference lacks a native-success boolean")
    pairing = strict_pairing(run_identity(policy_result), run_identity(sac_result))
    if not pairing["comparable"]:
        raise ValueError(f"Seed {seed} is not exact-start paired: {pairing['reasons']}")
    policy_event, sac_event = _first_event(paths["policy_events"]), _first_event(paths["sac_events"])
    deployed, sac_action = first_event_actions(policy_event, sac_event)
    expected_rng = policy_event.get("model_rng")
    if (
        not isinstance(expected_rng, list)
        or len(expected_rng) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in expected_rng)
    ):
        raise ValueError(f"Seed {seed} deployed first event lacks uint32 model_rng")
    recorded_metadata = policy_result.get("model_metadata")
    if not isinstance(recorded_metadata, dict):
        raise ValueError(f"Seed {seed} policy result lacks model metadata")
    return {
        "seed": seed,
        "paths": paths,
        "request": _load_request(paths["request"]),
        "deployed_raw": deployed,
        "sac_action": sac_action,
        "expected_rng": expected_rng,
        "recorded_metadata": recorded_metadata,
        "pairing": pairing,
        "sac_reference_native_success": sac_result["success"],
    }


def _audit_transform_parity(cases: list[dict], args) -> dict:
    """Compare the exact training and frozen-server input transforms on P0."""
    from openpi import transforms
    from openpi.policies.libero_policy import LiberoInputs
    from openpi.training import checkpoints, config as training_config
    from fetch_native_s1_config import config as fetch_config

    repo_id = cases[0]["recorded_metadata"]["state_contract"]["training_repo"]
    if any(
        case["recorded_metadata"].get("state_contract", {}).get("training_repo") != repo_id
        for case in cases
    ):
        raise ValueError("Training repo identity differs across saved policy runs")
    cfg = fetch_config(repo_id, str(args.output.resolve() / "unused"), str(args.checkpoint), args.normalizer)
    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    training_transform = transforms.compose(
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    checkpoint_stats = checkpoints.load_norm_stats(args.checkpoint / "assets", repo_id)
    server_transform = transforms.compose(
        [
            LiberoInputs(cfg.model.model_type),
            transforms.Normalize(checkpoint_stats, use_quantiles=True),
            *training_config.ModelTransformFactory()(cfg.model).inputs,
        ]
    )
    reports = []
    for case in cases:
        request = case["request"]
        # LeRobot decodes image features as float32 CHW in [0,1]. Reconstruct
        # that raw dataset representation from the exact saved online uint8 HWC
        # request so LiberoInputs' training-only parse branch is exercised.
        training_head = np.transpose(request["head_rgb"], (2, 0, 1)).astype(np.float32) / 255.0
        training_wrist = np.transpose(request["wrist_rgb"], (2, 0, 1)).astype(np.float32) / 255.0
        training_input = {
            "image": training_head,
            "wrist_image": training_wrist,
            "state": request["state"].copy(),
            "actions": np.zeros((10, 13), np.float32),
            "prompt": request["prompt"],
        }
        server_input = {
            "observation/image": request["head_rgb"].copy(),
            "observation/wrist_image": request["wrist_rgb"].copy(),
            "observation/state": request["state"].copy(),
            "prompt": request["prompt"],
        }
        comparison = compare_shared_model_inputs(
            training_transform(training_input), server_transform(server_input)
        )
        comparison["seed"] = case["seed"]
        comparison["request_sha256"] = _sha256(case["paths"]["request"])
        reports.append(comparison)
    return {
        "status": "bit_exact" if all(report["bit_exact"] for report in reports) else "mismatch",
        "bit_exact": all(report["bit_exact"] for report in reports),
        "scope": (
            "cfg.data.create training repack+data+quantile-normalize+model transforms versus "
            "server LiberoInputs+checkpoint quantile-normalize+model transforms; dummy action removed "
            "before comparing all shared model-input leaves"
        ),
        "repo_id": repo_id,
        "cases": reports,
    }


def _manifest(output: Path) -> None:
    files = []
    excluded = {"socket-auth.local", "model.sock", "artifact-manifest.json"}
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        files.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    (output / "artifact-manifest.json").write_text(
        json.dumps({"files": files}, indent=2) + "\n", encoding="utf-8"
    )


def _write_channel_csv(path: Path, seed_reports: list[dict]) -> None:
    rows = []
    for report in seed_reports:
        for channel in report["channel_metrics"]:
            rows.append({"seed": report["seed"], **channel})
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy-root", "sac-root", "checkpoint", "normalizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=GPU1_UUID)
    parser.add_argument("--max-seconds", type=int, default=900)
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    parser.add_argument(
        "--training-stage",
        choices=("S1_ordinary_target_domain_SFT_not_TAPT", "S1_IA_single_bank_no_progress"),
        default="S1_IA_single_bank_no_progress",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not 120 <= args.max_seconds <= 3600:
        raise ValueError("--max-seconds must be in [120,3600]")
    if not 10 <= args.request_timeout_seconds <= 300:
        raise ValueError("--request-timeout-seconds must be in [10,300]")
    normalizer_files = (args.normalizer / "norm_stats.json", args.normalizer / "provenance.json")
    if not args.checkpoint.is_dir() or not args.normalizer.is_dir() or not all(
        path.is_file() for path in normalizer_files
    ):
        raise FileNotFoundError(
            "Checkpoint and normalizer directories must exist; normalizer requires norm_stats.json and provenance.json"
        )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    server = connection = None
    auth = output / "socket-auth.local"
    server_dir = output / "server"
    error = None
    summary = {
        "status": "preflight",
        "stage": "exact_start_first_action_stochastic_attribution",
        "question": "Were the five deployed first actions repeated SAC-reference-distant stochastic tail draws?",
        "scope": (
            "P0 bit-exact training/server transform audit, then 5 exact starts x 16 sequential frozen-policy "
            "samples; official SAC first actions are references (4/5 native-success), not supervised labels; "
            "no simulator, rollout, training, checkpoint selection, or API"
        ),
        "seeds": list(SEEDS),
        "samples_per_seed": GATE_SPEC["samples_per_case"],
        "planned_inference_calls": len(SEEDS) * GATE_SPEC["samples_per_case"],
        "training_updates": 0,
        "simulator_steps": 0,
        "rollout_episodes": 0,
        "api_calls": 0,
        "gate_spec_frozen_before_inference": dict(GATE_SPEC),
        "checkpoint": str(args.checkpoint.resolve()),
        "normalizer": str(args.normalizer.resolve()),
    }
    try:
        cases = [_case(seed, args.policy_root.resolve(), args.sac_root.resolve()) for seed in SEEDS]
        recorded_hash_roster = {
            key: sorted({case["recorded_metadata"].get(key) for case in cases})
            for key in (
                "pretrained_parameters_sha256",
                "normalizer_sha256",
                "state_contract_sha256",
            )
        }
        if any(len(values) != 1 or not isinstance(values[0], str) for values in recorded_hash_roster.values()):
            raise ValueError(f"Recorded policy model identity differs across seeds: {recorded_hash_roster}")
        summary["input_provenance"] = [
            {
                "seed": case["seed"],
                **{
                    name: {"path": str(path.resolve()), "sha256": _sha256(path)}
                    for name, path in case["paths"].items()
                },
                "reference_state_sha256": case["pairing"]["policy"]["reference_state_sha256"],
                "sac_reference_native_success": case["sac_reference_native_success"],
            }
            for case in cases
        ]
        summary["recorded_model_identity_roster"] = recorded_hash_roster
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

        transform_parity = _audit_transform_parity(cases, args)
        (output / "transform-parity.json").write_text(
            json.dumps(transform_parity, indent=2) + "\n", encoding="utf-8"
        )
        summary["transform_parity"] = {
            "status": transform_parity["status"],
            "bit_exact": transform_parity["bit_exact"],
            "artifact": "transform-parity.json",
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if not transform_parity["bit_exact"]:
            raise RuntimeError("P0 training/server transform parity mismatch; inference probe blocked")

        auth.write_bytes(secrets.token_bytes(32))
        auth.chmod(0o600)
        server_dir.mkdir()
        socket_path = server_dir / "model.sock"
        command = [
            os.sys.executable,
            str(Path(__file__).with_name("serve_native_s1.py")),
            "--checkpoint",
            str(args.checkpoint),
            "--normalizer",
            str(args.normalizer),
            "--output",
            str(server_dir),
            "--socket",
            str(socket_path),
            "--auth-file",
            str(auth),
            "--gpu-uuid",
            args.gpu_uuid,
            "--max-inference-calls",
            str(summary["planned_inference_calls"]),
            "--training-stage",
            args.training_stage,
        ]
        with (server_dir / "stdout.log").open("x", encoding="utf-8") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while not (server_dir / "ready.json").exists():
            if server.poll() is not None:
                raise RuntimeError("Frozen model server failed before readiness")
            if time.monotonic() - started > min(300, args.max_seconds):
                raise TimeoutError("Frozen model loading timeout")
            time.sleep(1)

        from multiprocessing.connection import Client

        connection = Client(str(socket_path), family="AF_UNIX", authkey=auth.read_bytes())
        all_chunks, all_rngs, all_latencies, seed_reports = [], [], [], []
        current_metadata = None
        for case_index, case in enumerate(cases):
            if time.monotonic() - started >= args.max_seconds:
                raise TimeoutError("Exact-start first-action probe wall limit reached")
            connection.send({"op": "reset", "seed": case["seed"]})
            if not connection.poll(min(10, args.request_timeout_seconds)):
                raise TimeoutError(f"Seed {case['seed']} model reset timeout")
            reset = connection.recv()
            if reset.get("status") != "reset":
                raise RuntimeError(f"Seed {case['seed']} model reset failed: {reset}")
            metadata = reset.get("metadata", {})
            identity = model_identity_checks(metadata, case["recorded_metadata"])
            if not identity["matched"]:
                raise RuntimeError(f"Seed {case['seed']} frozen model identity mismatch: {identity['reasons']}")
            if current_metadata is None:
                current_metadata = metadata

            chunks, rngs, latencies = [], [], []
            for _ in range(GATE_SPEC["samples_per_case"]):
                remaining = args.max_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Exact-start first-action probe wall limit reached")
                connection.send({"op": "predict", "seed": case["seed"], **case["request"]})
                if not connection.poll(min(args.request_timeout_seconds, remaining)):
                    raise TimeoutError(f"Seed {case['seed']} model inference timeout")
                response = connection.recv()
                if response.get("status") != "ok":
                    raise RuntimeError(f"Seed {case['seed']} model inference failed: {response}")
                chunk = np.asarray(response["actions"], np.float32)
                rng = response.get("rng")
                if chunk.shape != (10, 13) or not np.isfinite(chunk).all():
                    raise ValueError(f"Seed {case['seed']} invalid action chunk")
                if not isinstance(rng, list) or len(rng) != 2:
                    raise ValueError(f"Seed {case['seed']} invalid model RNG evidence")
                chunks.append(chunk)
                rngs.append(rng)
                latencies.append(float(response["inference_seconds"]))
            chunks = np.asarray(chunks, np.float32)
            first_rng_matches = rngs[0] == case["expected_rng"]
            report = analyze_first_action_distribution(
                chunks[:, 0, :],
                case["deployed_raw"],
                case["sac_action"],
                first_sample_rng_matches=first_rng_matches,
                model_identity_matched=identity["matched"],
            )
            report.update(
                seed=case["seed"],
                expected_first_rng=case["expected_rng"],
                observed_first_rng=rngs[0],
                exact_start_pairing=case["pairing"],
                model_identity=identity,
                request_sha256=_sha256(case["paths"]["request"]),
                sac_reference_native_success=case["sac_reference_native_success"],
            )
            all_chunks.append(chunks)
            all_rngs.append(rngs)
            all_latencies.append(latencies)
            seed_reports.append(report)
            summary.update(
                status="running",
                completed_seeds=case_index + 1,
                completed_inference_calls=(case_index + 1) * GATE_SPEC["samples_per_case"],
                wall_seconds=time.monotonic() - started,
            )
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )

        cohort = analyze_probe_cohort(seed_reports)
        all_chunks_array = np.asarray(all_chunks, np.float32)
        np.savez_compressed(
            output / "samples.npz",
            seeds=np.asarray(SEEDS, np.int64),
            raw_action_chunks=all_chunks_array,
            raw_first_actions=all_chunks_array[:, :, 0, :],
            applied_first_actions=applied_actions(all_chunks_array[:, :, 0, :]).astype(np.float32),
            deployed_raw_first_actions=np.asarray([case["deployed_raw"] for case in cases], np.float32),
            sac_applied_first_actions=np.asarray([case["sac_action"] for case in cases], np.float32),
            rng=np.asarray(all_rngs, np.uint32),
            inference_seconds=np.asarray(all_latencies, np.float32),
        )
        _write_channel_csv(output / "per-channel.csv", seed_reports)
        (output / "per-seed.json").write_text(
            json.dumps(seed_reports, indent=2) + "\n", encoding="utf-8"
        )
        connection.send({"op": "shutdown"})
        if not connection.poll(10):
            raise TimeoutError("Frozen model server shutdown acknowledgement timeout")
        shutdown = connection.recv()
        if shutdown.get("status") != "shutdown":
            raise RuntimeError(f"Frozen model server shutdown failed: {shutdown}")
        connection.close()
        connection = None
        server.wait(timeout=30)
        if server.returncode != 0:
            raise RuntimeError(f"Frozen model server exited {server.returncode}")

        summary.update(
            status="completed",
            completed_seeds=len(SEEDS),
            completed_inference_calls=summary["planned_inference_calls"],
            wall_seconds=time.monotonic() - started,
            mean_inference_seconds=float(np.mean(all_latencies)),
            p95_inference_seconds=float(np.quantile(all_latencies, 0.95)),
            current_model_identity={
                key: current_metadata[key]
                for key in (
                    "checkpoint",
                    "pretrained_parameters_sha256",
                    "normalizer_sha256",
                    "state_contract_sha256",
                    "rng_contract",
                )
            },
            cohort_gate=cohort,
            sac_reference_native_successes=sum(
                case["sac_reference_native_success"] for case in cases
            ),
            outputs=["transform-parity.json", "per-seed.json", "per-channel.csv", "samples.npz"],
        )
    except Exception as exc:
        error = exc
        summary.update(status="failed", error=repr(exc), wall_seconds=time.monotonic() - started)
    finally:
        if connection is not None:
            connection.close()
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        auth.unlink(missing_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        _manifest(output)
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
