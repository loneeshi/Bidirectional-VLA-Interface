"""Run the paper-release MS-HAB teleport + per-object SAC baseline on 16 plans."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


PAPER_COMMIT = "4729821db3fc94a2470cfd625e6f8ab439f01478"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def select_plans(manifest: dict) -> list[dict]:
    seen: set[str] = set()
    selected = []
    for row in manifest["episodes"][:20]:
        uid = row.get("plan_uid")
        if not uid or uid in seen:
            continue
        if len(row["binding"]["subtasks"]) != 20:
            raise ValueError("Teleport panel requires five-object TidyHouse plans")
        seen.add(uid)
        selected.append({"seed": int(row["seed"]), "plan_uid": uid})
    if len(selected) != 16:
        raise ValueError("Expected exactly 16 unique plans")
    return selected


def bind_plan(task_plans: dict, plan_uid: str) -> dict:
    matches = [p for p in task_plans["plans"]
               if p["subtasks"] and p["subtasks"][0]["uid"] == plan_uid]
    if len(matches) != 1:
        raise ValueError(f"Expected one plan for {plan_uid}; found {len(matches)}")
    if len(matches[0]["subtasks"]) != 20:
        raise ValueError("Bound plan is not a five-object TidyHouse plan")
    return {"dataset": task_plans["dataset"], "plans": matches}


def command(row: dict, attempt: Path, paper_source: Path, plan_path: Path) -> list[str]:
    workspace = attempt / "official-logs"
    return [sys.executable, "-m", "mshab.evaluate", str(paper_source / "configs/evaluate.yml"),
        f"seed={row['seed']}", "task=tidy_house", "save_trajectory=False",
        "max_trajectories=1", "policy_type=rl_per_obj",
        "eval_env.env_id=SequentialTask-v0", f"eval_env.task_plan_fp={plan_path}",
        "eval_env.make_env=True", "eval_env.num_envs=1", "eval_env.frame_stack=3",
        "eval_env.stack=null", "eval_env.max_episode_steps=1000",
        "eval_env.continuous_task=True", "eval_env.record_video=True",
        "eval_env.info_on_video=True", "eval_env.save_video_freq=1",
        "logger.wandb=False", "logger.tensorboard=False",
        f"logger.workspace={workspace}", "logger.clear_out=False",
        f"logger.exp_name=seed-{row['seed']:03d}"]


def parse_result(attempt: Path) -> dict:
    outputs = list((attempt / "official-logs").rglob("output.txt"))
    if len(outputs) != 1:
        raise ValueError(f"Expected one official output.txt; found {len(outputs)}")
    text = outputs[0].read_text(encoding="utf-8")
    match = re.search(r"'success_once': tensor\(([-+0-9.eE]+)(?:,[^)]*)?\)", text)
    if not match:
        raise ValueError("Official output lacks success_once")
    success = float(match.group(1)) > 0.5
    fail_files = list((attempt / "official-logs").rglob("subtask_fail_counts.json"))
    if len(fail_files) != 1:
        raise ValueError(f"Expected one subtask_fail_counts.json; found {len(fail_files)}")
    failures = json.loads(fail_files[0].read_text(encoding="utf-8"))
    if success:
        completed_objects, failure_subtask = 5, None
    else:
        if sum(int(v) for v in failures.values()) != 1:
            raise ValueError("One failed trajectory must have exactly one failure index")
        failure_subtask = int(next(k for k, v in failures.items() if int(v)))
        completed_objects = min(5, failure_subtask // 4)
    videos = sorted(str(p) for p in (attempt / "official-logs").rglob("*.mp4"))
    return {"task_success": success, "completed_objects": completed_objects,
            "planned_objects": 5, "failure_subtask_index": failure_subtask,
            "output": str(outputs[0]), "videos": videos}


def summarize(state: dict) -> dict:
    completed = [r for r in state["episodes"] if r["status"] == "completed"]
    return {"planned": 16, "completed": len(completed),
            "infrastructure_failed": sum(r["status"] == "infrastructure_failure"
                                         for r in state["episodes"]),
            "not_run": sum(r["status"] == "not_run" for r in state["episodes"]),
            "running": sum(r["status"] == "running" for r in state["episodes"]),
            "successes": sum(bool(r.get("result", {}).get("task_success")) for r in completed),
            "completed_objects": sum(int(r.get("result", {}).get("completed_objects", 0))
                                     for r in completed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--task-plans", type=Path, required=True)
    parser.add_argument("--paper-source", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new", type=int, default=2)
    parser.add_argument("--episode-timeout", type=float, default=1200)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_new < 1 or args.episode_timeout <= 0:
        parser.error("positive --max-new and --episode-timeout required")
    selected = select_plans(json.loads(args.source_manifest.read_text(encoding="utf-8")))
    if not args.execute:
        print(json.dumps({"paper_commit": PAPER_COMMIT, "episodes": selected}, indent=2))
        return
    revision = subprocess.check_output(
        ["git", "-C", str(args.paper_source), "rev-parse", "HEAD"], text=True).strip()
    if revision != PAPER_COMMIT:
        raise ValueError(f"Paper source must be exact commit {PAPER_COMMIT}; found {revision}")
    source_hash = hashlib.sha256(
        (args.paper_source / "mshab/evaluate.py").read_bytes()).hexdigest()
    task_plans = json.loads(args.task_plans.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_link = args.output / "mshab_checkpoints"
    if not checkpoint_link.exists():
        checkpoint_link.symlink_to(args.checkpoint_root, target_is_directory=True)
    lock = args.output / "runner.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    status_path = args.output / "panel-status.json"
    state = (json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists()
             else {"schema": "official-mshab-teleport16/1", "status": "running",
                   "paper_commit": PAPER_COMMIT, "evaluate_sha256": source_hash,
                   "source_manifest": str(args.source_manifest),
                   "episodes": [{**row, "status": "not_run", "attempts": []}
                                for row in selected]})
    started = 0

    def save() -> None:
        state["summary"] = summarize(state)
        atomic_json(status_path, state)

    try:
        for row in state["episodes"]:
            if row["status"] == "completed" or started >= args.max_new:
                continue
            # A parser-only failure must not cause a valid physical rollout to be repeated.
            if row["attempts"] and row["attempts"][-1].get("returncode") == 0:
                prior = row["attempts"][-1]
                try:
                    recovered = parse_result(Path(prior["directory"]))
                except Exception:
                    pass
                else:
                    prior.setdefault("validity_adjudications", []).append({
                        "classification": "recovered_result_parser_only",
                        "time_unix": time.time(),
                    })
                    prior.update(status="completed", result=recovered)
                    row.update(status="completed", result=recovered)
                    save()
                    continue
            for old in row["attempts"]:
                if old["status"] == "running":
                    old["status"] = "interrupted"
            attempt = args.output / f"seed-{row['seed']:03d}" / f"attempt-{len(row['attempts']) + 1:03d}"
            attempt.mkdir(parents=True, exist_ok=False)
            plan_path = attempt / "tidy_house" / "task-plan.json"
            plan_path.parent.mkdir(parents=True, exist_ok=False)
            atomic_json(plan_path, bind_plan(task_plans, row["plan_uid"]))
            argv = command(row, attempt, args.paper_source, plan_path)
            record = {"directory": str(attempt), "status": "running", "argv": argv,
                      "started_unix": time.time()}
            row["attempts"].append(record)
            row["status"] = "running"
            state["status"] = "running"
            save()
            started += 1
            env = os.environ.copy()
            env["PYTHONPATH"] = str(args.paper_source) + os.pathsep + env.get("PYTHONPATH", "")
            env["PYTHONHASHSEED"] = "0"
            with (attempt / "runner.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen(argv, cwd=args.output, env=env, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                record["pid"] = child.pid
                save()
                try:
                    code = child.wait(timeout=args.episode_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait(timeout=30)
                    code = 124
            record["returncode"] = code
            record["finished_unix"] = time.time()
            try:
                result = parse_result(attempt) if code == 0 else {"reason": f"returncode:{code}"}
                status = "completed" if code == 0 else "infrastructure_failure"
            except Exception as exc:
                result = {"reason": f"result_parse:{type(exc).__name__}", "detail": str(exc)}
                status = "infrastructure_failure"
            record.update(status=status, result=result)
            row.update(status=status, result=result)
            save()
            if status == "infrastructure_failure":
                state["status"] = "stopped_infrastructure"
                break
        else:
            state["status"] = ("finished" if all(r["status"] == "completed" for r in state["episodes"])
                               else "chunk_complete")
    finally:
        save()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
