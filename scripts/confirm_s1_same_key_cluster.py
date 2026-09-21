"""Fresh-server confirmation of the S1 same-key BF16 action cluster.

This is a new five-call gate.  It preserves the invalid result of the original
bit-exact gate and does not change its tolerance.  Instead, one fresh process
produces one same-input/same-RNG action per seed and is tested against the
historical deployment, exploratory same-key sample0, and fifteen exploratory
different-key samples under a criterion written before inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import time

import numpy as np

from bvi.first_action_probe import GATE_SPEC
from bvi.same_key_confirmation import (
    CONFIRMATION_SPEC,
    adjudicate_confirmed_tail,
    anchor_cluster_preflight,
    confirm_fresh_cluster,
)


GPU1_UUID = "GPU-b7ebba23-7824-7601-df32-be55628936c3"
SEEDS = (2024, 2025, 2026, 2027, 2028)
MODEL_IDENTITY_FIELDS = (
    "server_source_sha256",
    "author_commit",
    "pretrained_parameters_sha256",
    "normalizer_sha256",
    "state_contract_sha256",
    "rng_contract",
    "checkpoint",
    "precision",
    "denoising_steps",
    "action_horizon",
    "external_action_dim",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def first_event(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        row = json.loads(next(line for line in handle if line.strip()))
    if row.get("step") != 1:
        raise ValueError(f"First event is not step 1: {path}")
    return row


def load_request(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != {"head_rgb", "wrist_rgb", "state", "prompt"}:
            raise ValueError(f"Unexpected request keys: {path}")
        result = {
            "head_rgb": np.asarray(data["head_rgb"]),
            "wrist_rgb": np.asarray(data["wrist_rgb"]),
            "state": np.asarray(data["state"], np.float32),
            "prompt": np.asarray(data["prompt"]).item(),
        }
    if (
        result["head_rgb"].shape != (128, 128, 3)
        or result["wrist_rgb"].shape != (128, 128, 3)
        or result["head_rgb"].dtype != np.uint8
        or result["wrist_rgb"].dtype != np.uint8
        or result["state"].shape != (24,)
        or not np.isfinite(result["state"]).all()
        or not isinstance(result["prompt"], str)
        or not result["prompt"].strip()
    ):
        raise ValueError(f"Saved request contract mismatch: {path}")
    return result


def verify_manifest(run: Path, relative_paths: tuple[str, ...]) -> dict:
    manifest = read_json(run / "artifact-manifest.json")
    entries = {row["path"]: row for row in manifest.get("files", [])}
    evidence = {}
    for relative in relative_paths:
        path = run / relative
        if relative not in entries or not path.is_file():
            raise ValueError(f"Exploratory manifest lacks {relative}")
        actual = sha256(path)
        if entries[relative].get("sha256") != actual:
            raise ValueError(f"Exploratory artifact hash mismatch: {relative}")
        evidence[relative] = actual
    return evidence


def model_identity(current: dict, recorded: dict) -> dict:
    rows, reasons = [], []
    for field in MODEL_IDENTITY_FIELDS:
        equal = field in current and field in recorded and current[field] == recorded[field]
        rows.append(
            {"field": field, "equal": equal, "current": current.get(field), "recorded": recorded.get(field)}
        )
        if not equal:
            reasons.append(f"{field}_missing_or_mismatch")
    return {"matched": not reasons, "reasons": reasons, "checks": rows}


def artifact_manifest(output: Path) -> None:
    rows = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.name in {
            "socket-auth.local",
            "model.sock",
            "artifact-manifest.json",
        }:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    (output / "artifact-manifest.json").write_text(
        json.dumps({"files": rows}, indent=2) + "\n", encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("exploratory-run", "policy-root", "checkpoint", "normalizer", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--gpu-uuid", default=GPU1_UUID)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--request-timeout-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 120 <= args.max_seconds <= 1800:
        raise ValueError("--max-seconds must be in [120,1800]")
    normalizer_files = (args.normalizer / "norm_stats.json", args.normalizer / "provenance.json")
    if (
        not args.checkpoint.is_dir()
        or not args.normalizer.is_dir()
        or not all(path.is_file() for path in normalizer_files)
    ):
        raise FileNotFoundError("Checkpoint/normalizer contract is missing")
    exploratory = args.exploratory_run.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    auth = output / "socket-auth.local"
    server_dir = output / "server"
    server = connection = None
    error = None
    summary = {
        "status": "preflight",
        "stage": "fresh_server_same_key_cluster_confirmation",
        "scope": (
            "5 fresh frozen-model calls; no simulator, rollout, training, checkpoint selection, API, "
            "or modification/reinterpretation of the invalid bit-exact run"
        ),
        "seeds": list(SEEDS),
        "planned_inference_calls": 5,
        "training_updates": 0,
        "simulator_steps": 0,
        "rollout_episodes": 0,
        "api_calls": 0,
        "confirmation_spec_frozen_before_inference": dict(CONFIRMATION_SPEC),
        "tail_spec_unchanged": dict(GATE_SPEC),
        "exploratory_run": str(exploratory),
        "checkpoint": str(args.checkpoint.resolve()),
        "normalizer": str(args.normalizer.resolve()),
    }
    try:
        verified = verify_manifest(
            exploratory,
            ("summary.json", "transform-parity.json", "per-seed.json", "samples.npz"),
        )
        transform_parity = read_json(exploratory / "transform-parity.json")
        if (
            transform_parity.get("bit_exact") is not True
            or [row.get("seed") for row in transform_parity.get("cases", [])] != list(SEEDS)
            or any(row.get("bit_exact") is not True for row in transform_parity["cases"])
        ):
            raise ValueError("Exploratory P0 transform evidence is not five-case bit exact")
        old_reports = {row["seed"]: row for row in read_json(exploratory / "per-seed.json")}
        if sorted(old_reports) != list(SEEDS):
            raise ValueError("Exploratory per-seed roster mismatch")
        with np.load(exploratory / "samples.npz", allow_pickle=False) as arrays:
            if not np.array_equal(arrays["seeds"], np.asarray(SEEDS)):
                raise ValueError("Exploratory sample seed roster mismatch")
            exploratory_first = np.asarray(arrays["raw_first_actions"], np.float32)
        if exploratory_first.shape != (5, 16, 13) or not np.isfinite(exploratory_first).all():
            raise ValueError("Exploratory first-action array contract mismatch")

        cases = []
        for index, seed in enumerate(SEEDS):
            policy = args.policy_root.resolve() / f"reach-seed{seed}"
            request_path = policy / "requests" / "step000.npz"
            result_path, events_path = policy / "result.json", policy / "events.jsonl"
            if not all(path.is_file() for path in (request_path, result_path, events_path)):
                raise FileNotFoundError(f"Seed {seed} historical evidence is missing")
            result, event = read_json(result_path), first_event(events_path)
            if result.get("seed") != seed:
                raise ValueError(f"Seed {seed} historical result identity mismatch")
            historical_chunk = np.asarray(event.get("raw_actions"), np.float32)
            if historical_chunk.shape != (10, 13) or not np.isfinite(historical_chunk).all():
                raise ValueError(f"Seed {seed} historical raw chunk is invalid")
            expected_rng = event.get("model_rng")
            if not isinstance(expected_rng, list) or len(expected_rng) != 2:
                raise ValueError(f"Seed {seed} historical RNG is invalid")
            request_hash = sha256(request_path)
            if old_reports[seed].get("request_sha256") != request_hash:
                raise ValueError(f"Seed {seed} request differs from exploratory run")
            anchor = anchor_cluster_preflight(historical_chunk[0], exploratory_first[index])
            cases.append(
                {
                    "seed": seed,
                    "request": load_request(request_path),
                    "request_path": request_path,
                    "request_sha256": request_hash,
                    "historical_raw": historical_chunk[0],
                    "exploratory": exploratory_first[index],
                    "expected_rng": expected_rng,
                    "recorded_metadata": result["model_metadata"],
                    "anchor_preflight": anchor,
                    "old_report": old_reports[seed],
                }
            )
        summary.update(
            exploratory_artifact_sha256=verified,
            exploratory_transform_parity="bit_exact",
            anchor_preflight=[
                {"seed": case["seed"], **case["anchor_preflight"]} for case in cases
            ],
        )
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        if any(not case["anchor_preflight"]["passed"] for case in cases):
            raise RuntimeError("Frozen mutual-nearest anchor preflight failed; fresh inference blocked")

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
            "5",
            "--training-stage",
            "S1_IA_single_bank_no_progress",
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
        fresh_chunks, reports = [], []
        for index, case in enumerate(cases):
            connection.send({"op": "reset", "seed": case["seed"]})
            if not connection.poll(min(10, args.request_timeout_seconds)):
                raise TimeoutError(f"Seed {case['seed']} reset timeout")
            reset = connection.recv()
            if reset.get("status") != "reset":
                raise RuntimeError(f"Seed {case['seed']} reset failed: {reset}")
            identity = model_identity(reset.get("metadata", {}), case["recorded_metadata"])
            if not identity["matched"]:
                raise RuntimeError(f"Seed {case['seed']} model identity mismatch: {identity['reasons']}")
            connection.send({"op": "predict", "seed": case["seed"], **case["request"]})
            remaining = args.max_seconds - (time.monotonic() - started)
            if remaining <= 0 or not connection.poll(min(args.request_timeout_seconds, remaining)):
                raise TimeoutError(f"Seed {case['seed']} inference timeout")
            response = connection.recv()
            if response.get("status") != "ok":
                raise RuntimeError(f"Seed {case['seed']} inference failed: {response}")
            chunk = np.asarray(response.get("actions"), np.float32)
            if chunk.shape != (10, 13) or not np.isfinite(chunk).all():
                raise ValueError(f"Seed {case['seed']} fresh action chunk invalid")
            rng_match = response.get("rng") == case["expected_rng"]
            confirmation = confirm_fresh_cluster(
                case["historical_raw"], case["exploratory"], chunk[0]
            )
            if not rng_match:
                confirmation["passed"] = False
            old = case["old_report"]
            reports.append(
                {
                    "seed": case["seed"],
                    "request_sha256": case["request_sha256"],
                    "model_identity": identity,
                    "expected_rng": case["expected_rng"],
                    "fresh_rng": response.get("rng"),
                    "rng_matches": rng_match,
                    "anchor_preflight": case["anchor_preflight"],
                    "fresh_confirmation": confirmation,
                    "sac_reference_native_success": old["sac_reference_native_success"],
                    "deployed_sac_distance_percentile": old[
                        "deployed_sac_distance_percentile"
                    ],
                    "sample_median_over_deployed": old["active_rmse_to_sac"][
                        "sample_median_over_deployed"
                    ],
                }
            )
            fresh_chunks.append(chunk)
            summary.update(
                status="running",
                completed_inference_calls=index + 1,
                wall_seconds=time.monotonic() - started,
            )
            (output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n", encoding="utf-8"
            )

        adjudication = adjudicate_confirmed_tail(reports)
        np.savez_compressed(
            output / "fresh-actions.npz",
            seeds=np.asarray(SEEDS, np.int64),
            raw_action_chunks=np.asarray(fresh_chunks, np.float32),
        )
        (output / "per-seed.json").write_text(
            json.dumps(reports, indent=2) + "\n", encoding="utf-8"
        )
        connection.send({"op": "shutdown"})
        if not connection.poll(10) or connection.recv().get("status") != "shutdown":
            raise RuntimeError("Frozen model server shutdown acknowledgement failed")
        connection.close()
        connection = None
        server.wait(timeout=30)
        if server.returncode != 0:
            raise RuntimeError(f"Frozen model server exited {server.returncode}")
        summary.update(
            status="completed",
            completed_inference_calls=5,
            wall_seconds=time.monotonic() - started,
            all_five_same_key_clusters_confirmed=all(
                row["fresh_confirmation"]["passed"] for row in reports
            ),
            adjudication=adjudication,
            outputs=["per-seed.json", "fresh-actions.npz"],
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
        artifact_manifest(output)
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
