"""Run three fresh-env same-action controls and freeze Level-3 qpos thresholds.

Default mode is preflight-only.  ``--execute`` is required to create an output
directory or start simulator children.  No IA event path is accepted.
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
import time


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "src"))

from bvi.null_drift_calibration import (  # noqa: E402
    CALIBRATION_ACTIONS,
    MIN_REPEATS,
    NULL_MULTIPLIER,
    RULE_VERSION,
    UNIT_FLOOR,
    generate_null_drift_thresholds,
    validate_recorded_actions,
)


GPU = "GPU-b7ebba23-7824-7601-df32-be55628936c3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def canonical_sha256(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def lowercase_sha(parser: argparse.ArgumentParser, value: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        parser.error("Expected hashes must be lowercase SHA256 values")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--expected-reference-sha256", required=True)
    parser.add_argument("--actions-jsonl", type=Path, required=True)
    parser.add_argument("--expected-actions-sha256", required=True)
    parser.add_argument("--expected-task-plan-sha256", required=True)
    parser.add_argument("--expected-spawn-sha256", required=True)
    parser.add_argument("--sim-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=CALIBRATION_ACTIONS)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--per-repeat-seconds", type=int, default=180)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.repeats != MIN_REPEATS:
        parser.error("Rule v1 is frozen to exactly three fresh-env repeats")
    if args.max_actions != CALIBRATION_ACTIONS:
        parser.error(f"Rule v1 requires exactly {CALIBRATION_ACTIONS} actions")
    if not 180 <= args.max_seconds <= 600:
        parser.error("--max-seconds must be within [180, 600]")
    if not 30 <= args.per_repeat_seconds <= 300:
        parser.error("--per-repeat-seconds must be within [30, 300]")
    if args.repeats * args.per_repeat_seconds + 30 > args.max_seconds:
        parser.error("Outer wall bound must cover all repeat bounds plus 30 seconds")
    hashes = {
        "reference_state": lowercase_sha(parser, args.expected_reference_sha256),
        "actions_jsonl": lowercase_sha(parser, args.expected_actions_sha256),
        "task_plan": lowercase_sha(parser, args.expected_task_plan_sha256),
        "spawn_data": lowercase_sha(parser, args.expected_spawn_sha256),
    }
    required = (args.reference_state, args.actions_jsonl, args.sim_python)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error(f"Required exact-start assets are missing: {missing}")
    if sha256(args.reference_state) != hashes["reference_state"]:
        parser.error("Reference-state SHA256 differs from preregistration")
    if sha256(args.actions_jsonl) != hashes["actions_jsonl"]:
        parser.error("Recorded-action SHA256 differs from preregistration")
    source_rows = load_events(args.actions_jsonl)
    source_actions = validate_recorded_actions(source_rows, args.max_actions)
    child = Path(__file__).with_name("eval_s1_null_drift_repeat.py")
    if not child.is_file():
        parser.error(f"Missing child runner: {child}")
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite output: {output}")
    started_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "stage": "S1-Level3-null-drift",
        "status": "preregistered_not_started",
        "question": "What is the same-action numerical/physics drift floor before IA-vs-SAC comparison?",
        "seed": args.seed,
        "task": "set_table/pick/013_apple",
        "gpu_uuid": GPU,
        "sim_backend": "gpu",
        "shader": "minimal",
        "fresh_process_and_environment_per_repeat": True,
        "repeats": args.repeats,
        "calibration_actions_per_repeat": args.max_actions,
        "expected_total_actions": args.repeats * args.max_actions,
        "max_seconds": args.max_seconds,
        "per_repeat_seconds": args.per_repeat_seconds,
        "training_updates": 0,
        "api_calls": 0,
        "policy_inference_calls": 0,
        "consumes_ia_events": False,
        "source_assets": {
            "reference_state": {"path": str(args.reference_state.resolve()), "sha256": hashes["reference_state"]},
            "actions_jsonl": {"path": str(args.actions_jsonl.resolve()), "sha256": hashes["actions_jsonl"]},
            "task_plan_sha256": hashes["task_plan"],
            "spawn_data_sha256": hashes["spawn_data"],
        },
        "threshold_generation_rule": {
            "version": RULE_VERSION,
            "expression": "threshold[channel] = max(unit_floor, null_multiplier * max_pairwise_abs_over_repeat_pairs_and_steps)",
            "strict_comparison_operator": ">",
            "null_multiplier": NULL_MULTIPLIER,
            "translation_floor_m": UNIT_FLOOR,
            "rotation_floor_rad": UNIT_FLOOR,
            "horizon": CALIBRATION_ACTIONS,
            "policy_under_test_excluded": True,
        },
        "runner_sha256": sha256(Path(__file__)),
        "child_runner_sha256": sha256(child),
        "started_utc": started_utc,
        "execute": args.execute,
    }
    manifest["threshold_generation_rule_sha256"] = canonical_sha256(manifest["threshold_generation_rule"])
    if not args.execute:
        print(json.dumps(manifest, indent=2))
        return

    output.mkdir(parents=True, exist_ok=False)
    manifest["status"] = "running"
    (output / "launch.json").write_text(json.dumps(manifest, indent=2))
    started = time.monotonic()
    summary = {
        "status": "running",
        "launch_sha256": sha256(output / "launch.json"),
        "training_updates": 0,
        "api_calls": 0,
        "policy_inference_calls": 0,
        "source_assets": manifest["source_assets"],
        "threshold_generation_rule": manifest["threshold_generation_rule"],
        "threshold_generation_rule_sha256": manifest["threshold_generation_rule_sha256"],
        "runner_sha256": manifest["runner_sha256"],
        "child_runner_sha256": manifest["child_runner_sha256"],
        "rows": [],
    }

    def save() -> None:
        summary["wall_seconds"] = time.monotonic() - started
        (output / "summary.json").write_text(json.dumps(summary, indent=2))

    def remaining() -> float:
        value = args.max_seconds - (time.monotonic() - started)
        if value <= 0:
            raise TimeoutError("Null-drift calibration exceeded its outer wall bound")
        return value

    save()
    try:
        repeats = []
        runtime_identity = None
        initial_states = []
        for index in range(args.repeats):
            used = int(
                subprocess.check_output(
                    ["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                ).strip()
            )
            if used >= 1024:
                raise RuntimeError(f"GPU1 occupied before repeat {index}; launch refused")
            directory = output / f"repeat-{index:02d}"
            log_path = output / f"repeat-{index:02d}.log"
            command = [
                str(args.sim_python),
                str(child),
                "--seed",
                str(args.seed),
                "--repeat-index",
                str(index),
                "--reference-state",
                str(args.reference_state.resolve()),
                "--expected-reference-sha256",
                hashes["reference_state"],
                "--actions-jsonl",
                str(args.actions_jsonl.resolve()),
                "--expected-actions-sha256",
                hashes["actions_jsonl"],
                "--expected-task-plan-sha256",
                hashes["task_plan"],
                "--expected-spawn-sha256",
                hashes["spawn_data"],
                "--output",
                str(directory),
                "--max-actions",
                str(args.max_actions),
                "--max-seconds",
                str(args.per_repeat_seconds),
            ]
            child_timeout = min(args.per_repeat_seconds + 20, max(1, int(remaining() - 15)))
            environment = dict(os.environ, PYTHONHASHSEED=str(args.seed))
            with log_path.open("x") as log:
                completed = subprocess.run(
                    command,
                    env=environment,
                    cwd=REPOSITORY,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=child_timeout,
                    check=False,
                )
            result_path = directory / "result.json"
            events_path = directory / "events.jsonl"
            result = json.loads(result_path.read_text()) if result_path.is_file() else {"status": "missing"}
            row = {
                "repeat_index": index,
                "exit_code": completed.returncode,
                "result_status": result.get("status"),
                "steps": result.get("steps", 0),
                "log": log_path.name,
            }
            summary["rows"].append(row)
            save()
            if completed.returncode != 0 or result.get("status") != "completed_null_repeat":
                raise RuntimeError(f"Null repeat {index} failed closed")
            if result.get("reference_state_sha256") != hashes["reference_state"]:
                raise RuntimeError(f"Null repeat {index} changed reference identity")
            if result.get("actions_jsonl_sha256") != hashes["actions_jsonl"]:
                raise RuntimeError(f"Null repeat {index} changed action identity")
            if result.get("steps") != CALIBRATION_ACTIONS or result.get("restore_max_abs_error", 1) > 1e-5:
                raise RuntimeError(f"Null repeat {index} lacks complete bounded restore evidence")
            if result.get("gpu_uuid") != GPU or result.get("sim_backend") != "gpu":
                raise RuntimeError(f"Null repeat {index} changed the GPU/backend contract")
            if result.get("runner_sha256") != manifest["child_runner_sha256"]:
                raise RuntimeError(f"Null repeat {index} changed the child runner identity")
            if result.get("task_plan_sha256") != hashes["task_plan"] or result.get("spawn_data_sha256") != hashes["spawn_data"]:
                raise RuntimeError(f"Null repeat {index} changed task/spawn identity")
            current_runtime = result.get("runtime_identity")
            if not isinstance(current_runtime, dict):
                raise RuntimeError(f"Null repeat {index} lacks runtime identity")
            if runtime_identity is None:
                runtime_identity = current_runtime
            elif current_runtime != runtime_identity:
                raise RuntimeError(f"Null repeat {index} changed runtime/config identity")
            initial = result.get("initial")
            if not isinstance(initial, dict):
                raise RuntimeError(f"Null repeat {index} lacks step0 state")
            initial_states.append(initial)
            events = load_events(events_path)
            repeats.append(events)
            row.update(
                result_sha256=sha256(result_path),
                events_sha256=sha256(events_path),
                restore_max_abs_error=result["restore_max_abs_error"],
            )
            save()
        initial_qpos = [row.get("qpos") for row in initial_states]
        if any(not isinstance(row, list) or len(row) != 15 for row in initial_qpos):
            raise RuntimeError("Null repeats lack complete step0 qpos")
        initial_max = max(
            abs(float(initial_qpos[left][channel]) - float(initial_qpos[right][channel]))
            for left in range(len(initial_qpos))
            for right in range(left + 1, len(initial_qpos))
            for channel in range(15)
        )
        initial_distance_max = max(
            abs(float(initial_states[left]["tcp_target_distance_m"]) - float(initial_states[right]["tcp_target_distance_m"]))
            for left in range(len(initial_states))
            for right in range(left + 1, len(initial_states))
        )
        initial_force_max = max(
            abs(float(initial_states[left]["robot_cumulative_force"]) - float(initial_states[right]["robot_cumulative_force"]))
            for left in range(len(initial_states))
            for right in range(left + 1, len(initial_states))
        )
        if max(initial_max, initial_distance_max, initial_force_max) > 1e-5:
            raise RuntimeError("Fresh-env repeats do not share step0 within the preregistered 1e-5 tolerance")
        calibration = generate_null_drift_thresholds(
            repeats,
            source_actions,
            null_multiplier=NULL_MULTIPLIER,
            translation_floor_m=UNIT_FLOOR,
            rotation_floor_rad=UNIT_FLOOR,
        )
        calibration["source_identity"] = manifest["source_assets"]
        calibration["raw_repeats"] = summary["rows"]
        (output / "thresholds.json").write_text(json.dumps(calibration, indent=2) + "\n")
        summary.update(
            status="completed_null_only_calibration",
            completed_repeats=len(repeats),
            total_actions=len(repeats) * CALIBRATION_ACTIONS,
            thresholds_sha256=sha256(output / "thresholds.json"),
            runtime_identity=runtime_identity,
            step0_pairwise_max={
                "qpos": initial_max,
                "tcp_target_distance_m": initial_distance_max,
                "robot_cumulative_force": initial_force_max,
            },
        )
    except Exception as error:
        summary.update(status="infrastructure_failure", error=repr(error))
        raise
    finally:
        used_after = None
        gpu_error = None
        for _ in range(10):
            try:
                used_after = int(
                    subprocess.check_output(
                        ["nvidia-smi", "-i", GPU, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                        text=True,
                    ).strip()
                )
                if used_after < 1024:
                    break
            except Exception as error:
                gpu_error = repr(error)
                break
            time.sleep(1)
        summary["gpu_memory_after_mib"] = used_after
        summary["gpu_release_verified"] = used_after is not None and used_after < 1024
        if gpu_error is not None:
            summary["gpu_release_check_error"] = gpu_error
        if not summary["gpu_release_verified"] and sys.exc_info()[0] is None:
            summary.update(status="infrastructure_failure", error="GPU did not return below 1024 MiB")
            save()
            raise RuntimeError("GPU did not return below 1024 MiB")
        save()


if __name__ == "__main__":
    main()
