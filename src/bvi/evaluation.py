"""Frozen TidyHouse evaluation profiles, resumable panels, and summaries."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

from .bridge import atomic_json


PANEL_SCHEMA = "bvi-eval-panel/1"
SUMMARY_SCHEMA = "bvi-tidyhouse-comparison/1"
PLAN_COUNT = 16
OBJECTS_PER_PLAN = 5
EXPECTED_TELEPORT_OBJECTS = 21
OFFICIAL_RUNTIME_COMMIT = "e9ff3d23496d38e4431c8d913e147ffa007f7f72"
TELEPORT_SOURCE_COMMIT = "4729821db3fc94a2470cfd625e6f8ab439f01478"


@dataclass(frozen=True)
class EvaluationProfile:
    setting: str
    navigation_policy: str
    manipulation_policy: str = "official_rl_per_object_sac"
    dispatcher: str = "official_fixed_order"
    organizer: bool = False
    goal_tools: bool = False
    model: str | None = None
    max_calls: int = 175
    max_env_steps: int = 7000
    max_wall_seconds: int = 900
    skill_wall_seconds: int = 180
    slice_steps: int = 40
    max_output_tokens: int | None = None
    max_input_bytes: int | None = None
    max_api_cost_usd: float | None = None
    request_cost_ceiling_usd: float | None = None
    training_updates: int = 0


PROFILES = {
    "fixed": EvaluationProfile(setting="fixed", navigation_policy="official_ppo"),
    "gpt": EvaluationProfile(
        setting="gpt",
        navigation_policy="official_ppo",
        dispatcher="vla_as_tools",
        organizer=True,
        goal_tools=True,
        model="gpt-5.6-luna",
        max_calls=40,
        max_output_tokens=2048,
        max_input_bytes=512_000,
        max_api_cost_usd=0.05,
        request_cost_ceiling_usd=0.00125,
    ),
    "teleport": EvaluationProfile(setting="teleport", navigation_policy="standardized_teleport"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_plans(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Select the frozen first occurrence of each of the 16 historical plans."""
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for row in manifest.get("episodes", [])[:20]:
        uid = row.get("plan_uid")
        subtasks = row.get("binding", {}).get("subtasks", [])
        if uid in seen:
            continue
        if not isinstance(uid, str) or not uid or len(subtasks) != 20:
            raise ValueError("Missing real five-object task binding")
        seen.add(uid)
        rows.append({"seed": int(row["seed"]), "plan_uid": uid})
    if len(rows) != PLAN_COUNT:
        raise ValueError("Expected exactly 16 unique historical plans")
    return rows


def _panel_path(path: Path) -> Path:
    return path / "panel-status.json" if path.is_dir() else path


def _result(row: dict[str, Any]) -> dict[str, Any]:
    result = row.get("result")
    if isinstance(result, dict):
        return result
    attempts = row.get("attempts", [])
    return attempts[-1].get("result", {}) if attempts else {}


def _initial_hash(row: dict[str, Any]) -> str:
    direct = row.get("initial_state_sha256")
    if isinstance(direct, str) and direct:
        return direct
    attempts = row.get("attempts", [])
    if not attempts:
        raise ValueError("Reference fixed row has no preserved attempt")
    initial = Path(attempts[-1]["directory"]) / "initial-state.json"
    data = read_json(initial)
    if data.get("plan_uid") != row.get("plan_uid"):
        raise ValueError("Reference initial state plan mismatch")
    value = data.get("state_sha256")
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError("Reference initial state hash is missing")
    return value


def _recorded_initial_hash(row: dict[str, Any]) -> str | None:
    value = row.get("initial_state_sha256")
    if isinstance(value, str) and len(value) == 64:
        return value
    for attempt in reversed(row.get("attempts", [])):
        argv = attempt.get("argv", [])
        if "--expected-initial-state-sha256" not in argv:
            continue
        index = argv.index("--expected-initial-state-sha256") + 1
        if index < len(argv) and isinstance(argv[index], str) and len(argv[index]) == 64:
            return argv[index]
    return None


def fixed_reference_rows(path: Path) -> dict[tuple[int, str], dict[str, Any]]:
    data = read_json(_panel_path(path))
    if data.get("schema") == "ppo-sac-paired16/1":
        rows = [row for row in data.get("episodes", []) if row.get("arm") == "fixed"]
    elif data.get("schema") == PANEL_SCHEMA and data.get("setting") == "fixed":
        rows = data.get("episodes", [])
    else:
        raise ValueError("Reference panel is not a completed fixed panel")
    completed = [row for row in rows if row.get("status") == "completed"]
    if len(completed) != PLAN_COUNT:
        raise ValueError("Reference panel must contain 16 completed fixed rows")
    return {(int(row["seed"]), row["plan_uid"]): row for row in completed}


def build_bindings(setting: str, manifest: dict[str, Any], reference_panel: Path | None) -> list[dict[str, Any]]:
    selected = select_plans(manifest)
    if setting == "fixed":
        return selected
    if reference_panel is None:
        raise ValueError(f"{setting} requires --reference-panel from the fixed setting")
    fixed = fixed_reference_rows(reference_panel)
    bindings = []
    for row in selected:
        key = (row["seed"], row["plan_uid"])
        if key not in fixed:
            raise ValueError("Every GPT/teleport row requires the same completed fixed binding")
        bindings.append({**row, "initial_state_sha256": _initial_hash(fixed[key])})
    return bindings


def runner_command(
    profile: EvaluationProfile,
    row: dict[str, Any],
    output: Path,
    checkpoint_root: Path,
    bridge_dir: Path | None,
    authorization_id: str | None,
    no_video: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "bvi.mshab_runner",
        "--setting",
        profile.setting,
        "--seed",
        str(row["seed"]),
        "--expected-plan-uid",
        row["plan_uid"],
        "--checkpoint-root",
        str(checkpoint_root),
        "--output",
        str(output),
    ]
    if row.get("initial_state_sha256"):
        command += ["--expected-initial-state-sha256", row["initial_state_sha256"]]
    if no_video:
        command.append("--no-video")
    if profile.setting == "gpt":
        if bridge_dir is None or not authorization_id:
            raise ValueError("GPT execution requires --bridge-dir and --authorization-id")
        command += ["--bridge-dir", str(bridge_dir), "--authorization-id", authorization_id]
    return command


def runtime_environment(mshab_root: Path, asset_dir: Path | None = None) -> dict[str, str]:
    root = mshab_root.resolve()
    expected = root / "mshab/envs/sequential_task.py"
    if not expected.is_file():
        raise ValueError("Missing pinned official MS-HAB runtime")
    package_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(package_root))))
    if asset_dir is not None:
        env["MS_ASSET_DIR"] = str(asset_dir.resolve())
    env.setdefault("PYTHONHASHSEED", "0")
    return env


def classify(returncode: int, summary: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if summary and summary.get("evaluation_eligible") and summary.get("benchmark_episode"):
        reason = str(summary.get("reason") or "")
        model_failure = reason in {"error:ModelResponseError", "error:APIBudgetExhausted"}
        infrastructure = reason.startswith(("adapter_error:", "error:")) and not model_failure
        if not infrastructure and (returncode == 0 or model_failure):
            return "completed", {
                "returncode": returncode,
                "reason": reason,
                "budget_terminated": reason in {
                    "error:APIBudgetExhausted",
                    "experiment_call_limit",
                    "experiment_step_limit",
                    "experiment_wall_clock_limit",
                    "request_exceeds_remaining_experiment_steps",
                    "request_exceeds_remaining_experiment_wall_budget",
                },
                "task_success": bool(summary.get("task_success")),
                "completed_objects": int(summary.get("completed_objects", 0)),
                "planned_objects": int(summary.get("planned_objects", OBJECTS_PER_PLAN)),
                "steps": int(summary.get("steps", 0)),
                "api_requests": int(summary.get("api_requests", 0)),
                "wall_seconds": float(summary.get("wall_seconds", 0)),
                "api_cost_usd": None if int(summary.get("api_requests", 0)) else 0,
                "api_cost_status": summary.get("api_cost_status", "pending_reconciliation"),
            }
    return "infrastructure_failure", {
        "returncode": returncode,
        "reason": summary.get("reason") if summary else "missing_summary",
        "api_requests": int(summary.get("api_requests", 0)) if summary else 0,
        "steps": summary.get("steps") if summary else None,
        "wall_seconds": summary.get("wall_seconds") if summary else None,
        "api_cost_usd": None,
        "api_cost_status": "pending_provider_reconciliation",
    }


def panel_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    completed = [row for row in rows if row.get("status") == "completed"]
    return {
        "planned": PLAN_COUNT,
        "completed": len(completed),
        "infrastructure_failed": sum(row.get("status") == "infrastructure_failure" for row in rows),
        "not_run": sum(row.get("status") == "not_run" for row in rows),
        "running": sum(row.get("status") == "running" for row in rows),
        "successes": sum(bool(_result(row).get("task_success")) for row in completed),
        "completed_objects": sum(int(_result(row).get("completed_objects", 0)) for row in completed),
        "api_requests": sum(
            int(attempt.get("result", {}).get("api_requests", 0))
            for row in rows
            for attempt in row.get("attempts", [])
        ),
    }


def validate_resume_state(state: dict[str, Any], setting: str, bindings: list[dict[str, Any]]) -> None:
    if state.get("schema") != PANEL_SCHEMA or state.get("setting") != setting:
        raise ValueError("Output directory belongs to a different evaluation setting")
    keys = ("seed", "plan_uid", "initial_state_sha256")
    expected = [tuple(row.get(key) for key in keys) for row in bindings]
    actual = [tuple(row.get(key) for key in keys) for row in state.get("episodes", [])]
    if actual != expected:
        raise ValueError("Resume panel does not match the frozen plan/state bindings")


def attempt_destination(output: Path, row: dict[str, Any]) -> Path:
    return output / f"seed-{row['seed']:03d}" / f"attempt-{len(row.get('attempts', [])) + 1:03d}"


def acquire_panel_lock(output: Path) -> Path:
    lock = output / "runner.lock"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    return lock


def run_panel(args: Any) -> int:
    profile = PROFILES[args.setting]
    manifest = read_json(args.source_manifest)
    bindings = build_bindings(args.setting, manifest, args.reference_panel)
    preview = {
        "schema": PANEL_SCHEMA,
        "setting": args.setting,
        "profile": asdict(profile),
        "plans": bindings,
        "execution": False,
    }
    if not args.execute:
        print(json.dumps(preview, indent=2))
        return 0
    if os.name != "posix":
        raise ValueError("Execution requires Linux process-group timeout handling")
    if args.max_new < 1:
        raise ValueError("--max-new must be positive")
    if args.setting == "gpt" and (args.bridge_dir is None or not args.authorization_id):
        raise ValueError("GPT execution requires --bridge-dir and --authorization-id")

    env = runtime_environment(args.mshab_root, args.asset_dir)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = acquire_panel_lock(args.output)
    status_path = args.output / "panel-status.json"
    try:
        state = read_json(status_path) if status_path.exists() else {
            "schema": PANEL_SCHEMA,
            "setting": args.setting,
            "profile": asdict(profile),
            "status": "running",
            "episodes": [{**row, "status": "not_run", "attempts": []} for row in bindings],
        }
        validate_resume_state(state, args.setting, bindings)
    except BaseException:
        lock.unlink()
        raise

    started = 0

    def save() -> None:
        state["summary"] = panel_summary(state["episodes"])
        atomic_json(status_path, state)

    try:
        state["status"] = "running"
        save()
        for row in state["episodes"]:
            if row["status"] == "completed":
                continue
            if row["status"] == "infrastructure_failure" and not args.retry_infrastructure:
                state["status"] = "stopped_infrastructure"
                break
            if started >= args.max_new:
                state["status"] = "chunk_complete"
                break
            for attempt in row["attempts"]:
                if attempt.get("status") == "running":
                    attempt["status"] = "interrupted"
            destination = attempt_destination(args.output, row)
            destination.parent.mkdir(parents=True, exist_ok=True)
            command = runner_command(
                profile,
                row,
                destination,
                args.checkpoint_root,
                args.bridge_dir,
                args.authorization_id,
                args.no_video,
            )
            attempt = {
                "attempt": len(row["attempts"]) + 1,
                "directory": str(destination),
                "status": "running",
                "started_unix": time.time(),
            }
            row["attempts"].append(attempt)
            row["status"] = "running"
            started += 1
            save()
            log_path = destination.with_suffix(".log")
            with log_path.open("w", encoding="utf-8") as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                attempt["pid"] = child.pid
                save()
                try:
                    code = child.wait(timeout=990)
                except BaseException:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise
            raw_path = destination / "summary.json"
            raw = read_json(raw_path) if raw_path.exists() else None
            status, result = classify(code, raw)
            row.update(status=status, result=result)
            attempt.update(status=status, result=result, finished_unix=time.time())
            save()
            if status == "infrastructure_failure":
                state["status"] = "stopped_infrastructure"
                break
        else:
            state["status"] = "finished"
    except BaseException as exc:
        state.update(status="interrupted", error_type=type(exc).__name__)
        raise
    finally:
        try:
            save()
        finally:
            lock.unlink(missing_ok=True)
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _setting_rows(data: dict[str, Any], setting: str) -> list[dict[str, Any]]:
    if data.get("schema") == "ppo-sac-paired16/1":
        return [row for row in data.get("episodes", []) if row.get("arm") == setting]
    if data.get("schema") in {PANEL_SCHEMA, "standardized-teleport16/1"}:
        actual = data.get("setting", "teleport" if data.get("schema") == "standardized-teleport16/1" else None)
        if actual != setting:
            raise ValueError(f"Expected {setting} panel, found {actual}")
        return data.get("episodes", [])
    raise ValueError("Unsupported panel schema")


def _public_setting(setting: str, rows: list[dict[str, Any]], source_path: Path) -> dict[str, Any]:
    if len(rows) != PLAN_COUNT or len({int(row["seed"]) for row in rows}) != PLAN_COUNT:
        raise ValueError(f"{setting} must contain exactly 16 unique episodes")
    if any(row.get("status") != "completed" for row in rows):
        raise ValueError(f"{setting} contains incomplete or infrastructure-invalid rows")
    episodes = []
    for row in sorted(rows, key=lambda item: int(item["seed"])):
        result = _result(row)
        if int(result.get("planned_objects", OBJECTS_PER_PLAN)) != OBJECTS_PER_PLAN:
            raise ValueError(f"{setting} changed the five-object denominator")
        episodes.append({
            "seed": int(row["seed"]),
            "plan_uid": row["plan_uid"],
            "task_success": bool(result.get("task_success")),
            "completed_objects": int(result.get("completed_objects", 0)),
            "reason": result.get("reason"),
            "steps": result.get("steps"),
            "api_requests": int(result.get("api_requests", 0)),
            "initial_state_sha256": _recorded_initial_hash(row),
        })
    completed_objects = sum(row["completed_objects"] for row in episodes)
    successes = sum(row["task_success"] for row in episodes)
    api_requests = sum(row["api_requests"] for row in episodes)
    if setting in {"fixed", "teleport"} and api_requests:
        raise ValueError(f"{setting} must make zero external model API requests")
    return {
        "setting": setting,
        "planned_episodes": PLAN_COUNT,
        "completed_episodes": PLAN_COUNT,
        "infrastructure_failures": 0,
        "full_task_successes": successes,
        "completed_objects": completed_objects,
        "planned_objects": PLAN_COUNT * OBJECTS_PER_PLAN,
        "completed_object_rate": completed_objects / (PLAN_COUNT * OBJECTS_PER_PLAN),
        "mean_completed_objects": completed_objects / PLAN_COUNT,
        "api_requests_final_attempts": api_requests,
        "episodes": episodes,
        "source_sha256": _sha256(source_path),
    }


def summarize_panels(paired_panel: Path, teleport_panel: Path) -> dict[str, Any]:
    paired_path = _panel_path(paired_panel)
    teleport_path = _panel_path(teleport_panel)
    paired = read_json(paired_path)
    teleport = read_json(teleport_path)
    settings = [
        _public_setting("fixed", _setting_rows(paired, "fixed"), paired_path),
        _public_setting("gpt", _setting_rows(paired, "gpt"), paired_path),
        _public_setting("teleport", _setting_rows(teleport, "teleport"), teleport_path),
    ]
    bindings = [
        {(row["seed"], row["plan_uid"]) for row in item["episodes"]}
        for item in settings
    ]
    if not (bindings[0] == bindings[1] == bindings[2]):
        raise ValueError("All settings must use the same 16-plan roster")
    fixed_by_binding = {
        (row["seed"], row["plan_uid"]): row
        for row in settings[0]["episodes"]
    }
    gpt_hashes = {
        (row["seed"], row["plan_uid"]): row["initial_state_sha256"]
        for row in settings[1]["episodes"]
    }
    teleport_hashes = {
        (row["seed"], row["plan_uid"]): row["initial_state_sha256"]
        for row in settings[2]["episodes"]
    }
    if any(value is None for value in gpt_hashes.values()) or gpt_hashes != teleport_hashes:
        raise ValueError("GPT and teleport evidence must bind the same fixed initial-state hashes")
    for binding, value in gpt_hashes.items():
        fixed_by_binding[binding]["initial_state_sha256"] = value
    teleport_objects = next(item["completed_objects"] for item in settings if item["setting"] == "teleport")
    if teleport_objects != EXPECTED_TELEPORT_OBJECTS:
        raise ValueError(f"Teleport evidence sums to {teleport_objects}, expected {EXPECTED_TELEPORT_OBJECTS}")
    return {
        "schema": SUMMARY_SCHEMA,
        "task": "TidyHouse",
        "scope": "matched_16_plan_diagnostic_not_1000_rollout_benchmark",
        "training_updates": 0,
        "runtime_provenance": {
            "official_mshab_commit": OFFICIAL_RUNTIME_COMMIT,
            "standardized_teleport_source_commit": TELEPORT_SOURCE_COMMIT,
        },
        "settings": settings,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    labels = {
        "fixed": "Fixed PPO + SAC（官方固定任务顺序）",
        "gpt": "GPT + PPO + SAC（VLA-as-Tools 通讯协议）",
        "teleport": "Teleport + SAC（官方固定任务顺序）",
    }
    lines = [
        "| 设置 | 完成 episode | 完整任务 SR | 完成物体 | 每集平均 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["settings"]:
        completed = item["completed_objects"]
        planned = item["planned_objects"]
        mean = Decimal(completed) / Decimal(PLAN_COUNT)
        mean_text = str(mean.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP))
        lines.append(
            f"| {labels[item['setting']]} | 16/16 | {item['full_task_successes']}/16 | "
            f"{completed}/{planned}（{100 * item['completed_object_rate']:.2f}%） | "
            f"{mean_text} |"
        )
    return "\n".join(lines) + "\n"
